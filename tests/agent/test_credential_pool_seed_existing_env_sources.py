"""_seed_from_env must seed pool entries whose env:VAR source is on disk but
whose VAR is not in the provider registry's api_key_env_vars tuple.

Without this, putting `source: env:OPENCODE_GO_API_KEY2` in auth.json with the
matching secret in Bitwarden/environment results in a permanently empty pool
entry, filtered out of rotation by _available_entries because its
runtime_api_key is empty. Round-robin degrades to single-key behavior.

The fix walks existing pool entries in _seed_from_env and appends any
env:VAR names not already in the registry's tuple to the env-var list
before the existing for-loop populates them.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


def _write_env_file(home: Path, **env_vars):
    """Write a .env file under HERMES_HOME."""
    home.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}={v}" for k, v in env_vars.items()]
    (home / ".env").write_text("\n".join(lines) + "\n")


def _make_pconfig(provider_id: str, env_vars: list[str]):
    """Create a minimal ProviderConfig for testing."""
    from hermes_cli.auth import ProviderConfig
    return ProviderConfig(
        id=provider_id,
        name=provider_id.title(),
        auth_type="api_key",
        api_key_env_vars=tuple(env_vars),
    )


@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
        "ZAI_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN", "OPENCODE_GO_API_KEY",
        "OPENCODE_GO_API_KEY2", "OPENAI_BASE_URL",
    ]:
        monkeypatch.delenv(key, raising=False)
    return home


class TestSeedFromEnvRespectsExistingPoolEntries:
    """Regression: env-source entries already in auth.json must be seeded."""

    def test_second_env_source_entry_seeded_from_env(self, isolated_hermes_home):
        """An env:OPENCODE_GO_API_KEY2 entry already in the pool gets
        populated when the env var is set, even though the registry only
        declares the primary OPENCODE_GO_API_KEY."""
        from agent.credential_pool import (
            PooledCredential,
            _seed_from_env,
        )

        # Both env vars set in .env
        _write_env_file(
            isolated_hermes_home,
            OPENCODE_GO_API_KEY="primary-key-12345",
            OPENCODE_GO_API_KEY2="backup-key-67890",
        )

        # Mock pconfig to only declare the primary env var
        pconfig = _make_pconfig("opencode-go", ["OPENCODE_GO_API_KEY"])

        # Pool already contains the second env-source entry on disk
        secondary = PooledCredential(
            provider="opencode-go",
            id="sec1234",
            label="secondary",
            auth_type="api_key",
            priority=1,
            source="env:OPENCODE_GO_API_KEY2",
            access_token="",  # empty before seeding
        )
        entries = [secondary]

        with patch(
            "agent.credential_pool.PROVIDER_REGISTRY",
            {"opencode-go": pconfig},
        ):
            changed, active_sources = _seed_from_env("opencode-go", entries)

        assert changed is True
        # Both sources reported active
        assert "env:OPENCODE_GO_API_KEY" in active_sources
        assert "env:OPENCODE_GO_API_KEY2" in active_sources

        # The existing pool entry was populated. _upsert_entry replaces
        # the entry in the list with a new dataclass instance, so check
        # the list, not the original reference.
        populated = next(
            (e for e in entries if e.source == "env:OPENCODE_GO_API_KEY2"),
            None,
        )
        assert populated is not None, "secondary entry was dropped from pool"
        assert populated.access_token == "backup-key-67890", (
            f"Expected secondary entry to be populated, "
            f"got: {populated.access_token!r}"
        )

    def test_no_pool_entries_unchanged_behavior(self, isolated_hermes_home):
        """If auth.json has no env-source entries, behavior matches the
        pre-fix code (only the registry's env_vars are scanned)."""
        from agent.credential_pool import _seed_from_env

        _write_env_file(isolated_hermes_home, OPENCODE_GO_API_KEY="primary-key")

        pconfig = _make_pconfig("opencode-go", ["OPENCODE_GO_API_KEY"])

        with patch(
            "agent.credential_pool.PROVIDER_REGISTRY",
            {"opencode-go": pconfig},
        ):
            changed, active_sources = _seed_from_env("opencode-go", [])

        assert changed is True
        assert "env:OPENCODE_GO_API_KEY" in active_sources
        # No second source because no entry on disk
        assert "env:OPENCODE_GO_API_KEY2" not in active_sources

    def test_registry_entry_takes_priority_over_pool(self, isolated_hermes_home):
        """If the registry declares a VAR and the pool also has it as an
        env-source entry, no duplicate seeding occurs."""
        from agent.credential_pool import PooledCredential, _seed_from_env

        _write_env_file(isolated_hermes_home, OPENCODE_GO_API_KEY="primary-key")

        pconfig = _make_pconfig("opencode-go", ["OPENCODE_GO_API_KEY"])

        existing = PooledCredential(
            provider="opencode-go",
            id="prim1234",
            label="primary",
            auth_type="api_key",
            priority=0,
            source="env:OPENCODE_GO_API_KEY",
            access_token="",
        )
        entries = [existing]

        with patch(
            "agent.credential_pool.PROVIDER_REGISTRY",
            {"opencode-go": pconfig},
        ):
            changed, active_sources = _seed_from_env("opencode-go", entries)

        assert changed is True
        assert "env:OPENCODE_GO_API_KEY" in active_sources
        # The list now contains a refreshed entry with the populated token.
        populated = next(
            (e for e in entries if e.source == "env:OPENCODE_GO_API_KEY"),
            None,
        )
        assert populated is not None
        assert populated.access_token == "primary-key"

    def test_manual_source_entry_not_touched_by_seed(self, isolated_hermes_home):
        """Manual entries in auth.json must not be re-seeded from env vars
        even if the env var happens to match."""
        from agent.credential_pool import PooledCredential, _seed_from_env

        _write_env_file(isolated_hermes_home, OPENCODE_GO_API_KEY="env-value")

        pconfig = _make_pconfig("opencode-go", ["OPENCODE_GO_API_KEY"])

        manual = PooledCredential(
            provider="opencode-go",
            id="man1234",
            label="manual",
            auth_type="api_key",
            priority=0,
            source="manual",
            access_token="manually-set-key",
        )
        entries = [manual]

        with patch(
            "agent.credential_pool.PROVIDER_REGISTRY",
            {"opencode-go": pconfig},
        ):
            _seed_from_env("opencode-go", entries)

        # Manual entry's access_token is preserved. Manual sources don't
        # start with "env:" so the new loop skips them.
        populated = next(
            (e for e in entries if e.source == "manual"),
            None,
        )
        assert populated is not None
        assert populated.access_token == "manually-set-key"

    def test_env_source_entry_with_no_env_value_unchanged(
        self, isolated_hermes_home
    ):
        """An env-source entry whose env var is unset stays empty (we don't
        invent values)."""
        from agent.credential_pool import PooledCredential, _seed_from_env

        # No env vars set
        pconfig = _make_pconfig("opencode-go", ["OPENCODE_GO_API_KEY"])

        secondary = PooledCredential(
            provider="opencode-go",
            id="sec1234",
            label="secondary",
            auth_type="api_key",
            priority=1,
            source="env:OPENCODE_GO_API_KEY2",
            access_token="",
        )
        entries = [secondary]

        with patch(
            "agent.credential_pool.PROVIDER_REGISTRY",
            {"opencode-go": pconfig},
        ):
            changed, _ = _seed_from_env("opencode-go", entries)

        assert changed is False
        populated = next(
            (e for e in entries if e.source == "env:OPENCODE_GO_API_KEY2"),
            None,
        )
        assert populated is not None
        assert populated.access_token == ""
