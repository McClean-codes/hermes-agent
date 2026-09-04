"""Tests for Discord persona reactions and dynamic per-tool emoji swapping.

Covers the persona-reaction state machine implemented via
``gateway.platforms.reaction_mixin.DynamicReactionMixin`` and
``plugins/platforms/discord/adapter.DiscordAdapter``:

- configured persona placement at processing start
- at least two distinct tool transitions each with add(new) before remove(previous)
- successful completion with add(persona) before remove(last_tool) and exactly one final persona
- no-tool completion without duplicate/untracked cleanup
- callback/reaction updates remain active when visible tool-progress messages are disabled
- repeated identical/rapid updates are coalesced according to bounded cooldown/hysteresis
- established disabled/failure/cancel behavior remains intact
- rate-limit/API failures leave tracked state recoverable and do not raise

All tests exercise the real current callback path with a fake Discord
message/API ledger and assert ordering and effective reaction set from calls
made by production code (no manual mock invocation after handler return).
"""

import asyncio
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.Interaction = object
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: lambda fn: fn,
        choices=lambda **kwargs: lambda fn: fn,
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


class FakeTree:
    def __init__(self):
        self.commands = {}

    def command(self, *, name, description):
        def decorator(fn):
            self.commands[name] = fn
            return fn

        return decorator


class LedgerMessage:
    """Fake Discord message that records add/remove order and effective set."""

    def __init__(self, msg_id=12345):
        self.id = msg_id
        self._ledger = []  # list of ("add", emoji) or ("remove", emoji)
        self._effective = set()
        self.add_reaction = AsyncMock(side_effect=self._add)
        self.remove_reaction = AsyncMock(side_effect=self._remove)

    async def _add(self, emoji):
        self._ledger.append(("add", emoji))
        self._effective.add(emoji)
        return None

    async def _remove(self, emoji, user=None):
        # discord.py signature is remove_reaction(emoji, user)
        self._ledger.append(("remove", emoji))
        self._effective.discard(emoji)
        return None

    def ledger(self):
        return list(self._ledger)

    def effective(self):
        return set(self._effective)


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="***")
    # ensure extra is a dict we can mutate
    if not isinstance(getattr(config, "extra", None), dict):
        config.extra = {}
    ad = DiscordAdapter(config)
    ad._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    return ad


def _make_adapter(extra=None, enabled=True):
    """Production-path helper: construct PlatformConfig.extra before DiscordAdapter init."""
    cfg = PlatformConfig(enabled=enabled, token="***", extra=dict(extra or {}))
    ad = DiscordAdapter(cfg)
    ad._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    return ad


def _make_event(message_id: str, raw_message, source=None) -> MessageEvent:
    src = source or SessionSource(
        platform=Platform.DISCORD,
        chat_id="123",
        chat_type="dm",
        user_id="42",
        user_name="Jezza",
    )
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=src,
        raw_message=raw_message,
        message_id=message_id,
    )


def _make_source(chat_id="123", thread_id=None):
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=chat_id,
        chat_type="dm",
        user_id="42",
        user_name="Jezza",
        thread_id=thread_id,
    )


# ---------------------------------------------------------------------------
# 1. Persona placement at processing start
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persona_emoji_added_on_processing_start():
    """Persona emoji (configured) is added when processing begins."""
    ad = _make_adapter({"persona_emoji": "🤖"})
    raw = LedgerMessage()
    event = _make_event("1", raw)
    await ad.on_processing_start(event)
    # production code called add_reaction with persona, not eyes or check
    assert ("add", "🤖") in raw.ledger()
    # must be first add, not after a remove
    assert raw.ledger()[0] == ("add", "🤖")
    assert "🤖" in raw.effective()


@pytest.mark.asyncio
async def test_persona_fallback_to_eyes_when_not_configured(tmp_path):
    """When no persona is configured, fallback is the established 👀."""
    # Use real temp HERMES_HOME with no persona configured to prove production fallback.
    home = tmp_path / "fallback_home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "platforms:\n  discord:\n    enabled: true\n", encoding="utf-8"
    )
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from gateway.config import load_gateway_config

    token = set_hermes_home_override(home)
    try:
        cfg = load_gateway_config()
        disc = cfg.platforms.get(Platform.DISCORD)
        ad = DiscordAdapter(disc)
        ad._client = SimpleNamespace(
            tree=FakeTree(),
            get_channel=lambda _id: None,
            fetch_channel=AsyncMock(),
            user=SimpleNamespace(id=99999, name="HermesBot"),
        )
        raw = LedgerMessage()
        event = _make_event("1", raw)
        await ad.on_processing_start(event)
        assert ("add", "👀") in raw.ledger()
        assert raw.ledger()[0] == ("add", "👀")
        assert ad._rxn_persona_emoji == "👀"
    finally:
        reset_hermes_home_override(token)


# ---------------------------------------------------------------------------
# 2. Two distinct tool transitions, each add(new) before remove(previous)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_tool_transitions_add_before_remove():
    ad = _make_adapter({
        "persona_emoji": "🤖",
        "dynamic_reactions": True,
        "reaction_cooldown": 0,
    })

    raw = LedgerMessage()
    event = _make_event("10", raw)
    source = event.source

    await ad.on_processing_start(event)
    # ledger so far: add persona
    assert raw.ledger() == [("add", "🤖")]

    # Mock get_tool_emoji to return distinct emojis per tool
    emoji_map = {"read_file": "📄", "web_search": "🔍", "terminal": "💻"}

    def fake_get_tool_emoji(name, default="⚙️"):
        return emoji_map.get(name, default)

    with patch("agent.display.get_tool_emoji", side_effect=fake_get_tool_emoji):
        await ad.on_tool_call_start(source, "read_file")
        # First tool swap: add 📄 before remove 🤖
        ledger = raw.ledger()
        assert ledger[1] == ("add", "📄")
        assert ledger[2] == ("remove", "🤖")
        assert raw.effective() == {"📄"}

        await ad.on_tool_call_start(source, "web_search")
        ledger = raw.ledger()
        # Second distinct transition: add 🔍 before remove 📄
        assert ledger[3] == ("add", "🔍")
        assert ledger[4] == ("remove", "📄")
        assert raw.effective() == {"🔍"}

    # ordering proof: each transition's add precedes its matching remove
    # Two transitions verified above


# ---------------------------------------------------------------------------
# 3. Successful completion: add(persona) before remove(last_tool), exactly one final persona
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_completion_adds_persona_before_removing_last_tool():
    ad = _make_adapter({
        "persona_emoji": "🦊",
        "dynamic_reactions": True,
        "reaction_cooldown": 0,
    })

    raw = LedgerMessage()
    event = _make_event("20", raw)
    source = event.source
    await ad.on_processing_start(event)

    emoji_map = {"read_file": "📄", "terminal": "💻"}

    def fake_get_tool_emoji(name, default="⚙️"):
        return emoji_map.get(name, default)

    with patch("agent.display.get_tool_emoji", side_effect=fake_get_tool_emoji):
        await ad.on_tool_call_start(source, "read_file")
        await ad.on_tool_call_start(source, "terminal")
        # Now active is 💻, ledger has: add 🦊, add 📄/rm 🦊, add 💻/rm 📄
        assert raw.effective() == {"💻"}
        pre_len = len(raw.ledger())
        await ad.on_processing_complete(event, ProcessingOutcome.SUCCESS)
        ledger = raw.ledger()
        # Successful completion must add persona before removing last tool
        # Find the indices of the final add and remove
        # Expect add 🦊 then remove 💻 (order matters)
        assert ledger[pre_len] == ("add", "🦊")
        assert ledger[pre_len + 1] == ("remove", "💻")
        # Resulting effective set must contain exactly persona, not check mark nor tool icon
        assert raw.effective() == {"🦊"}
        assert "✅" not in raw.effective()
        assert "💻" not in raw.effective()
        assert "📄" not in raw.effective()


# ---------------------------------------------------------------------------
# 4. No-tool completion without duplicate/untracked cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_tool_completion_no_duplicate_cleanup():
    ad = _make_adapter({
        "persona_emoji": "🤖",
        "dynamic_reactions": True,
        "reaction_cooldown": 0,
    })

    raw = LedgerMessage()
    event = _make_event("30", raw)
    await ad.on_processing_start(event)
    assert raw.ledger() == [("add", "🤖")]
    assert raw.effective() == {"🤖"}

    # No tool calls between start and complete
    await ad.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    # Should not duplicate add or attempt untracked remove
    # Persona already active, so no new add and no remove
    assert raw.ledger() == [("add", "🤖")]
    assert raw.effective() == {"🤖"}


# ---------------------------------------------------------------------------
# 5. Callback/reaction updates remain active when tool-progress messages disabled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reaction_via_progress_callback_even_when_progress_queue_disabled():
    """TurnRunner.progress_callback must fire on_tool_call_start even with progress_queue=None."""
    # This tests the real current callback path: TurnRunner.progress_callback
    # fires the adapter hook before the progress_queue guard.
    from gateway.turn_context import TurnContext
    from gateway.run_turn_runner import TurnRunner

    # Create a fake adapter that records hook invocations
    config = PlatformConfig(enabled=True, token="***")
    config.extra = {
        "persona_emoji": "🤖",
        "dynamic_reactions": True,
        "reaction_cooldown": 0,
    }
    hook_calls = []

    class HookAdapter(DiscordAdapter):
        async def on_tool_call_start(self, event, tool_name):
            hook_calls.append(tool_name)
            return await super().on_tool_call_start(event, tool_name)

    ad = HookAdapter(config)
    ad._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    # need a raw message to back the session key
    raw = LedgerMessage()
    src = _make_source(chat_id="999")
    # Prime the session cache via on_processing_start so tool swap can resolve message
    evt = _make_event("100", raw, source=src)
    await ad.on_processing_start(evt)
    assert ("add", "🤖") in raw.ledger()
    hook_calls.clear()
    raw._ledger.clear()
    raw._effective = {"🤖"}
    # Re-wire the adapter's active state to keep persona (since we cleared ledger but not internal)
    # Reset internal tracking to reflect persona still active for this source
    # The adapter's _rxn_active still holds persona for that session key; keep it.

    # Build a TurnContext with progress_queue=None (tool_progress off) but with status adapter set
    ctx = TurnContext(
        source=src,
        progress_queue=None,  # disabled
        tool_progress_enabled=False,
        log_queue=None,
        _status_adapter=ad,
        _run_still_current=lambda: True,
        _live_status_adapter=None,
        _thinking_enabled=False,
    )
    # TurnRunner needs runner and ctx; create minimal runner mock
    mock_runner = SimpleNamespace()
    tr = TurnRunner(mock_runner, ctx)
    # Ensure _loop_for_step is set on ctx (required for _schedule)
    ctx._loop_for_step = asyncio.get_running_loop()
    # Need to mock safe_schedule_threadsafe to execute immediately for test determinism
    # TurnRunner._schedule uses gateway.run.safe_schedule_threadsafe which schedules on loop.
    # We'll patch that to run the coro directly in the loop.
    with patch.object(
        TurnRunner,
        "_schedule",
        side_effect=lambda coro, msg, loop=None: asyncio.create_task(coro),
    ):
        tr.progress_callback(
            "tool.started", tool_name="read_file", preview="x", args={}
        )
        await asyncio.sleep(0.05)

    # Even though progress_queue was None, the hook should have been scheduled and executed
    # Verify that raw received a tool emoji swap (add before remove)
    # Since we mocked schedule to await directly, the ledger should show add 📄 before remove 🤖
    # However we need deterministic emoji: patch get_tool_emoji
    # Re-run with deterministic emoji and fresh state to check ledger
    raw2 = LedgerMessage()
    evt2 = _make_event("101", raw2, source=src)
    ad2 = HookAdapter(config)
    ad2._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    # ad2 uses production cooldown 0 via extra
    await ad2.on_processing_start(evt2)
    assert ("add", "🤖") in raw2.ledger()
    raw2._ledger.clear()
    # new ctx with ad2
    ctx2 = TurnContext(
        source=src,
        progress_queue=None,
        tool_progress_enabled=False,
        log_queue=None,
        _status_adapter=ad2,
        _run_still_current=lambda: True,
        _live_status_adapter=None,
        _thinking_enabled=False,
    )
    ctx2._loop_for_step = asyncio.get_running_loop()
    tr2 = TurnRunner(mock_runner, ctx2)
    emoji_map = {"read_file": "📄"}

    def fake_emoji(name, default="⚙️"):
        return emoji_map.get(name, default)

    with patch("agent.display.get_tool_emoji", side_effect=fake_emoji):
        with patch.object(
            TurnRunner,
            "_schedule",
            side_effect=lambda coro, msg, loop=None: asyncio.create_task(coro),
        ):
            tr2.progress_callback(
                "tool.started", tool_name="read_file", preview="x", args={}
            )
            await asyncio.sleep(0.05)
    # After the call, raw2 should have swapped to 📄 even though progress was off
    assert ("add", "📄") in raw2.ledger()
    assert raw2.ledger()[0] == ("add", "📄")
    assert raw2.ledger()[1] == ("remove", "🤖")
    assert raw2.effective() == {"📄"}


# ---------------------------------------------------------------------------
# 6. Repeated identical/rapid updates are coalesced (cooldown/hysteresis)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeated_identical_tool_coalesced():
    ad = _make_adapter({
        "persona_emoji": "🤖",
        "dynamic_reactions": True,
        "reaction_cooldown": 10,
    })

    raw = LedgerMessage()
    event = _make_event("40", raw)
    source = event.source
    await ad.on_processing_start(event)

    emoji_map = {"read_file": "📄"}

    def fake_emoji(name, default="⚙️"):
        return emoji_map.get(name, default)

    with patch("agent.display.get_tool_emoji", side_effect=fake_emoji):
        await ad.on_tool_call_start(source, "read_file")
        assert ("add", "📄") in raw.ledger()
        ledger_len_after_first = len(raw.ledger())
        # Repeated identical tool — should be coalesced (no new add/remove)
        await ad.on_tool_call_start(source, "read_file")
        assert len(raw.ledger()) == ledger_len_after_first  # no change


@pytest.mark.asyncio
async def test_rapid_different_tool_coalesced_by_cooldown():
    ad = _make_adapter({
        "persona_emoji": "🤖",
        "dynamic_reactions": True,
        "reaction_cooldown": 10,
    })

    raw = LedgerMessage()
    event = _make_event("41", raw)
    source = event.source
    await ad.on_processing_start(event)

    emoji_map = {"read_file": "📄", "web_search": "🔍"}

    def fake_emoji(name, default="⚙️"):
        return emoji_map.get(name, default)

    with patch("agent.display.get_tool_emoji", side_effect=fake_emoji):
        await ad.on_tool_call_start(source, "read_file")
        ledger_len = len(raw.ledger())
        # Rapid different tool within cooldown — should be coalesced (no swap)
        await ad.on_tool_call_start(source, "web_search")
        assert len(raw.ledger()) == ledger_len  # coalesced, still 📄
        assert raw.effective() == {"📄"}

        # Advance time beyond cooldown and try again — should now allow
        with patch("time.monotonic", return_value=time.monotonic() + 20):
            await ad.on_tool_call_start(source, "web_search")
        assert ("add", "🔍") in raw.ledger()
        assert raw.effective() == {"🔍"}


# ---------------------------------------------------------------------------
# 7. Disabled / failure / cancel behavior remains intact
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reactions_disabled_via_env_no_ops(monkeypatch):
    monkeypatch.setenv("DISCORD_REACTIONS", "false")
    ad = _make_adapter({
        "persona_emoji": "🤖",
        "dynamic_reactions": True,
        "reaction_cooldown": 0,
    })
    raw = LedgerMessage()
    event = _make_event("50", raw)
    await ad.on_processing_start(event)
    # No reactions when disabled via env
    assert raw.ledger() == []
    await ad.on_tool_call_start(event.source, "read_file")
    assert raw.ledger() == []
    await ad.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    assert raw.ledger() == []
    # Response delivery still works would be tested via _process_message_background but here we ensure no crash


@pytest.mark.asyncio
async def test_reactions_disabled_via_config_no_ops():
    ad = _make_adapter({"reactions": False})
    raw = LedgerMessage()
    event = _make_event("51", raw)
    await ad.on_processing_start(event)
    assert raw.ledger() == []


@pytest.mark.asyncio
async def test_failure_outcome_adds_cross_before_removing():
    ad = _make_adapter({
        "persona_emoji": "🤖",
        "dynamic_reactions": True,
        "reaction_cooldown": 0,
    })
    raw = LedgerMessage()
    event = _make_event("60", raw)
    source = event.source
    await ad.on_processing_start(event)
    emoji_map = {"read_file": "📄"}

    def fake_emoji(name, default="⚙️"):
        return emoji_map.get(name, default)

    with patch("agent.display.get_tool_emoji", side_effect=fake_emoji):
        await ad.on_tool_call_start(source, "read_file")
        assert raw.effective() == {"📄"}
        pre_len = len(raw.ledger())
        await ad.on_processing_complete(event, ProcessingOutcome.FAILURE)
        ledger = raw.ledger()
        assert ledger[pre_len] == ("add", "❌")
        assert ledger[pre_len + 1] == ("remove", "📄")
        assert raw.effective() == {"❌"}


@pytest.mark.asyncio
async def test_cancelled_outcome_removes_without_adding():
    ad = _make_adapter({
        "persona_emoji": "🤖",
        "dynamic_reactions": True,
        "reaction_cooldown": 0,
    })
    raw = LedgerMessage()
    event = _make_event("61", raw)
    source = event.source
    await ad.on_processing_start(event)
    emoji_map = {"read_file": "📄"}

    def fake_emoji(name, default="⚙️"):
        return emoji_map.get(name, default)

    with patch("agent.display.get_tool_emoji", side_effect=fake_emoji):
        await ad.on_tool_call_start(source, "read_file")
        assert raw.effective() == {"📄"}
        pre_len = len(raw.ledger())
        await ad.on_processing_complete(event, ProcessingOutcome.CANCELLED)
        ledger = raw.ledger()
        # Cancelled should remove current without adding persona or cross
        assert ledger[pre_len] == ("remove", "📄")
        assert len(ledger) == pre_len + 1
        assert raw.effective() == set()


# ---------------------------------------------------------------------------
# 8. Rate-limit / API failures leave tracked state recoverable and do not raise
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limit_failure_recoverable_and_no_exception():
    ad = _make_adapter({
        "persona_emoji": "🤖",
        "dynamic_reactions": True,
        "reaction_cooldown": 0,
    })

    raw = LedgerMessage()
    # Make the next add fail (simulate 429/500)
    original_add = raw.add_reaction

    call_count = {"n": 0}

    async def failing_add(emoji):
        call_count["n"] += 1
        if call_count["n"] == 2:  # second add (first tool swap) fails
            raise RuntimeError("429 rate limited")
        return await original_add(emoji)

    raw.add_reaction = AsyncMock(side_effect=failing_add)
    # remove should still work
    event = _make_event("70", raw)
    source = event.source
    await ad.on_processing_start(event)
    # start succeeded (first call n=1)
    assert call_count["n"] == 1

    emoji_map = {"read_file": "📄", "web_search": "🔍"}

    def fake_emoji(name, default="⚙️"):
        return emoji_map.get(name, default)

    with patch("agent.display.get_tool_emoji", side_effect=fake_emoji):
        # This tool swap's add will raise — should not propagate exception
        await ad.on_tool_call_start(source, "read_file")
        # No exception, and state should be recoverable (still persona)
        # Because our mixin returns early without updating active on failure, ledger should not have second add
        # Effective still persona
        assert raw.effective() == {"🤖"}
        # Next tool swap should succeed
        await ad.on_tool_call_start(source, "web_search")
        assert ("add", "🔍") in [
            ("add", e) for e in [x[1] for x in raw.ledger() if x[0] == "add"]
        ]
        assert raw.effective() == {"🔍"}
        # Completion should still work
        await ad.on_processing_complete(event, ProcessingOutcome.SUCCESS)
        assert "🤖" in raw.effective()
        assert "🔍" not in raw.effective()


@pytest.mark.asyncio
async def test_process_message_background_still_delivers_when_reactions_disabled(
    adapter, monkeypatch
):
    """Preserve established message-delivery semantics when reactions disabled."""
    monkeypatch.setenv("DISCORD_REACTIONS", "false")
    raw = LedgerMessage()

    # raw needs to mimic discord message for background path? _process_message_background uses raw_message.add_reaction etc.
    # With reactions disabled, it should still deliver via send()
    async def handler(_event):
        await asyncio.sleep(0)
        return "ack"

    async def hold_typing(_chat_id, interval=2.0, metadata=None):
        await asyncio.Event().wait()

    adapter.set_message_handler(handler)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="999"))
    adapter._keep_typing = hold_typing

    event = _make_event("4", raw)
    await adapter._process_message_background(event, build_session_key(event.source))
    raw.add_reaction.assert_not_called() if hasattr(raw, "add_reaction") else None
    # Response should still be sent
    adapter.send.assert_awaited_once()


# ---------------------------------------------------------------------------
# Legacy compatibility: original test expectations updated for persona
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_message_background_adds_and_swaps_reactions_legacy():
    """Legacy: processing start adds persona (default 👀) and success restores persona (not ✅)."""
    raw = LedgerMessage()
    # Default 👀 via production path (no persona configured)
    ad = _make_adapter({"dynamic_reactions": True, "reaction_cooldown": 0})
    adapter = ad

    async def handler(_event):
        await asyncio.sleep(0)
        return "ack"

    async def hold_typing(_chat_id, interval=2.0, metadata=None):
        await asyncio.Event().wait()

    adapter.set_message_handler(handler)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="999"))
    adapter._keep_typing = hold_typing

    event = _make_event("1", raw)
    await adapter._process_message_background(event, build_session_key(event.source))

    # With persona feature, success restores persona not check mark
    assert raw.ledger()[0] == ("add", "👀")
    # Complete should not have removed then added ✅; instead persona remains (no duplicate)
    # So ledger should be just the initial add (no tool swaps)
    assert raw.effective() == {"👀"}
    assert "✅" not in raw.effective()


# ---------------------------------------------------------------------------
# Identity isolation: participant/profile-aware session keys
# ---------------------------------------------------------------------------


def _make_group_source(
    chat_id="group-123", user_id="user-A", profile=None, thread_id=None
):
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=chat_id,
        chat_type="group",
        user_id=user_id,
        user_name=f"user-{user_id}",
        thread_id=thread_id,
        profile=profile,
    )


@pytest.mark.asyncio
async def test_identity_isolation_two_participants_and_profiles_share_channel():
    """Distinct participant/profile contexts sharing platform/chat/thread produce independent ledgers."""
    # Build adapter with group per-user isolation (default True)
    cfg = PlatformConfig(enabled=True, token="***")
    cfg.extra = {
        "persona_emoji": "🤖",
        "dynamic_reactions": True,
        "reaction_cooldown": 0,
    }
    ad = DiscordAdapter(cfg)
    ad._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    # Ensure mixin resolved correctly via production path (no manual _rxn mutation for config)
    # But we need cooldown 0 for deterministic test; set via extra and re-init is not allowed per spec for config tests, but this is not a config test.
    # For behavioral test we can set via direct attribute because we are not proving config plumbing here.
    # Two participants in same group channel, different profiles
    src_alice_red = _make_group_source(chat_id="chan-1", user_id="alice", profile="red")
    src_bob_blue = _make_group_source(chat_id="chan-1", user_id="bob", profile="blue")
    # They must have distinct session keys (canonical)
    key_alice = ad._session_key_from_source(src_alice_red)
    key_bob = ad._session_key_from_source(src_bob_blue)
    assert key_alice != key_bob, f"keys collided: {key_alice} vs {key_bob}"
    # Also direct build_session_key should differ
    assert build_session_key(src_alice_red) != build_session_key(src_bob_blue)
    # Create separate ledger messages
    raw_alice = LedgerMessage(msg_id=111)
    raw_bob = LedgerMessage(msg_id=222)
    evt_alice = MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=src_alice_red,
        raw_message=raw_alice,
        message_id="m1",
    )
    evt_bob = MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=src_bob_blue,
        raw_message=raw_bob,
        message_id="m2",
    )
    # Start both: each should add persona to its own message
    await ad.on_processing_start(evt_alice)
    await ad.on_processing_start(evt_bob)
    assert ("add", "🤖") in raw_alice.ledger()
    assert ("add", "🤖") in raw_bob.ledger()
    assert raw_alice.ledger() == [("add", "🤖")]
    assert raw_bob.ledger() == [("add", "🤖")]
    # Check internal tracking is separate
    assert ad._rxn_active[key_alice] == "🤖"
    assert ad._rxn_active[key_bob] == "🤖"
    assert ad._session_raw_messages[key_alice] is raw_alice
    assert ad._session_raw_messages[key_bob] is raw_bob
    # Tool call for Alice should only affect Alice's ledger (production path via SessionSource)
    emoji_map = {"read_file": "📄"}

    def _fake_emoji(name, default="⚙️"):
        return emoji_map.get(name, default)

    with patch("agent.display.get_tool_emoji", side_effect=_fake_emoji):
        await ad.on_tool_call_start(src_alice_red, "read_file")
        # Alice should have swapped to 📄
        assert raw_alice.ledger() == [("add", "🤖"), ("add", "📄"), ("remove", "🤖")]
        assert raw_alice.effective() == {"📄"}
        # Bob must be untouched
        assert raw_bob.ledger() == [("add", "🤖")]
        assert raw_bob.effective() == {"🤖"}
        assert ad._rxn_active[key_alice] == "📄"
        assert ad._rxn_active[key_bob] == "🤖"
        # Swap Bob independently
        await ad.on_tool_call_start(src_bob_blue, "read_file")
        assert raw_bob.ledger() == [("add", "🤖"), ("add", "📄"), ("remove", "🤖")]
    # Completion for Alice should not affect Bob
    await ad.on_processing_complete(evt_alice, ProcessingOutcome.SUCCESS)
    assert raw_alice.effective() == {"🤖"}
    # Alice ledgers: final add persona before remove tool
    assert raw_alice.ledger()[-2] == ("add", "🤖")
    assert raw_alice.ledger()[-1] == ("remove", "📄")
    # Bob still has tool emoji until his completion
    assert raw_bob.effective() == {"📄"}
    assert ad._rxn_active.get(key_bob) == "📄"
    assert key_alice not in ad._rxn_active  # cleaned
    assert key_alice not in ad._rxn_msg_refs
    assert key_alice not in ad._rxn_locks
    # Bob's lock still present until completion
    assert key_bob in ad._rxn_locks or key_bob in ad._rxn_active
    await ad.on_processing_complete(evt_bob, ProcessingOutcome.SUCCESS)
    assert raw_bob.effective() == {"🤖"}
    assert key_bob not in ad._rxn_active
    assert key_bob not in ad._rxn_locks
    # Ensure no cross-contamination after both completions
    assert ad._rxn_active == {}
    assert ad._rxn_stale == {}


@pytest.mark.asyncio
async def test_identity_isolation_thread_vs_group_participant():
    """Group per-user vs thread per-user isolation produces correct keys."""
    cfg = PlatformConfig(enabled=True, token="***")
    cfg.extra = {"group_sessions_per_user": True, "thread_sessions_per_user": False}
    ad = DiscordAdapter(cfg)
    ad._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=1, name="Bot"),
    )
    # Group without thread: participants isolate
    src_a = _make_group_source(chat_id="g1", user_id="u1", thread_id=None)
    src_b = _make_group_source(chat_id="g1", user_id="u2", thread_id=None)
    assert ad._session_key_from_source(src_a) != ad._session_key_from_source(src_b)
    # Same thread: by default thread_sessions_per_user=False, so same thread shares key even with different users
    src_a_thread = _make_group_source(chat_id="g1", user_id="u1", thread_id="thr-1")
    src_b_thread = _make_group_source(chat_id="g1", user_id="u2", thread_id="thr-1")
    assert ad._session_key_from_source(src_a_thread) == ad._session_key_from_source(
        src_b_thread
    )
    # Enable thread per-user isolation: now they differ
    ad2 = DiscordAdapter(
        PlatformConfig(
            enabled=True,
            token="***",
            extra={"thread_sessions_per_user": True, "group_sessions_per_user": True},
        )
    )
    ad2._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=1, name="Bot"),
    )
    assert ad2._session_key_from_source(src_a_thread) != ad2._session_key_from_source(
        src_b_thread
    )


# ---------------------------------------------------------------------------
# Failure-state correctness: primitive False and exceptions
# ---------------------------------------------------------------------------


class ControllableLedgerMessage(LedgerMessage):
    """Ledger that can be configured to fail specific ops with False or exception."""

    def __init__(
        self,
        msg_id=999,
        fail_add=False,
        fail_remove=False,
        fail_add_exc=False,
        fail_remove_exc=False,
    ):
        super().__init__(msg_id=msg_id)
        self._fail_add = fail_add
        self._fail_remove = fail_remove
        self._fail_add_exc = fail_add_exc
        self._fail_remove_exc = fail_remove_exc

    async def _add(self, emoji):
        if self._fail_add_exc:
            self._fail_add_exc = False  # one-shot
            raise RuntimeError("simulated add exception")
        if self._fail_add:
            # simulate API returned False: don't record
            self._ledger.append(("add", emoji + ":failed"))
            return None
        return await super()._add(emoji)

    async def _remove(self, emoji, user=None):
        if self._fail_remove_exc:
            self._fail_remove_exc = False
            raise RuntimeError("simulated remove exception")
        if self._fail_remove:
            self._ledger.append(("remove", emoji + ":failed"))
            return None
        return await super()._remove(emoji, user)


@pytest.mark.asyncio
async def test_start_add_false_leaves_no_active_and_recovers_on_completion():
    """Start add returning False does not mark active; no-tool SUCCESS later adds persona."""
    cfg = PlatformConfig(enabled=True, token="***")
    cfg.extra = {
        "persona_emoji": "🤖",
        "dynamic_reactions": True,
        "reaction_cooldown": 0,
    }
    ad = DiscordAdapter(cfg)
    ad._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=1, name="Bot"),
    )
    # Make start add fail
    raw = LedgerMessage(msg_id=1)
    src = _make_source(chat_id="123")
    evt = _make_event("1", raw, source=src)
    key = ad._session_key_from_source(src)
    with patch.object(ad, "_reaction_add", return_value=False) as mock_add:
        await ad.on_processing_start(evt)
        # Should have called add but not tracked active
        mock_add.assert_awaited()
        assert key not in ad._rxn_active, "active should not be set on False"
        assert key in ad._rxn_msg_refs, "msg_ref should still be cached for recovery"
        assert raw.ledger() == [], "ledger should have no successful add"
        assert raw.effective() == set()
    # Later no-tool completion should recover and add persona (outside patched context)
    await ad.on_processing_complete(evt, ProcessingOutcome.SUCCESS)
    # Now ledger should have the persona added via final path
    assert ("add", "🤖") in raw.ledger()
    assert raw.effective() == {"🤖"}
    assert key not in ad._rxn_active
    assert key not in ad._rxn_msg_refs
    assert key not in ad._rxn_locks
    assert key not in ad._rxn_stale


@pytest.mark.asyncio
async def test_start_add_exception_no_active_and_recovery():
    cfg = PlatformConfig(enabled=True, token="***")
    cfg.extra = {
        "persona_emoji": "🦊",
        "dynamic_reactions": True,
        "reaction_cooldown": 0,
    }
    ad = DiscordAdapter(cfg)
    ad._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=1, name="Bot"),
    )
    raw = LedgerMessage(msg_id=2)
    src = _make_source(chat_id="124")
    evt = _make_event("2", raw, source=src)
    with patch.object(ad, "_reaction_add", side_effect=RuntimeError("boom")):
        await ad.on_processing_start(evt)
        key = ad._session_key_from_source(src)
        assert key not in ad._rxn_active
        assert key in ad._rxn_msg_refs
    # Patch back to success for completion
    await ad.on_processing_complete(evt, ProcessingOutcome.SUCCESS)
    assert ("add", "🦊") in raw.ledger()
    assert raw.effective() == {"🦊"}
    key = ad._session_key_from_source(src)
    assert key not in ad._rxn_locks


@pytest.mark.asyncio
async def test_tool_transition_remove_false_leaves_stale_and_final_cleans():
    """Tool swap remove=False leaves stacked remote; stale tracked; final cleans both."""
    cfg = PlatformConfig(enabled=True, token="***")
    cfg.extra = {
        "persona_emoji": "🤖",
        "dynamic_reactions": True,
        "reaction_cooldown": 0,
    }
    ad = DiscordAdapter(cfg)
    ad._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=1, name="Bot"),
    )
    raw = LedgerMessage(msg_id=3)
    src = _make_source(chat_id="125")
    evt = _make_event("3", raw, source=src)
    await ad.on_processing_start(evt)
    assert raw.effective() == {"🤖"}
    key = ad._session_key_from_source(src)
    # Now cause remove to fail on first tool swap
    emoji_map = {"read_file": "📄", "web_search": "🔍"}

    def fake_emoji(name, default="⚙️"):
        return emoji_map.get(name, default)

    with patch("agent.display.get_tool_emoji", side_effect=fake_emoji):
        with patch.object(ad, "_reaction_remove", return_value=False) as mock_rm:
            await ad.on_tool_call_start(src, "read_file")
            # add succeeded, remove failed
            mock_rm.assert_awaited()
            assert raw.ledger()[1] == ("add", "📄")
            # No successful remove ledger entry
            assert len([x for x in raw.ledger() if x[0] == "remove"]) == 0
            # Both emojis remain remotely (ledger effective simulated as both present, but our LedgerMessage tracks effective as add without remove)
            # Our LedgerMessage still thinks both present because remove never called
            assert raw.effective() == {"🤖", "📄"}
            # Stale should contain the old persona that failed to remove
            assert key in ad._rxn_stale
            assert "🤖" in ad._rxn_stale[key]
            assert ad._rxn_active[key] == "📄"
        # Second tool swap should still work (add new, try remove previous tool) - real remove
        await ad.on_tool_call_start(src, "web_search")
        # Should add 🔍 and remove 📄
        assert ("add", "🔍") in raw.ledger()
        assert ("remove", "📄") in raw.ledger()
        # Stale still contains the original 🤖 that was never removed
        assert "🤖" in ad._rxn_stale.get(key, set())
        assert ad._rxn_active[key] == "🔍"
    # Final completion should clean all: add persona back and remove both stale
    await ad.on_processing_complete(evt, ProcessingOutcome.SUCCESS)
    # Ledger should now have added 🤖 and removed 🔍 and also removed stale 🤖 (but note 🤖 is also the final persona, so we should not remove the newly added final)
    # Our implementation removes stale excluding translated, so stale 🤖 equals final persona, thus it won't be removed again.
    # But we still have stacked old 🤖 from start that is same as final, so effectively final is already present? Need to reason: initial 🤖 remains, then 📄 stacked, then 🔍, stale holds original 🤖. Final wants persona 🤖, which is already present as stale, but current is 🔍. So final will add 🤖 (but it's already present as stale), then remove 🔍 and stale excluding translated.
    # Since stale 🤖 == translated, it won't be removed, so effective should be {🤖} only.
    assert raw.effective() == {"🤖"}
    assert key not in ad._rxn_active
    assert key not in ad._rxn_stale
    assert key not in ad._rxn_locks
    # Ensure no untracked stacked remains: only one persona
    assert raw.ledger().count(("add", "🤖")) >= 2  # start and final


@pytest.mark.asyncio
async def test_tool_transition_add_false_preserves_previous_and_recovers():
    cfg = PlatformConfig(enabled=True, token="***")
    cfg.extra = {
        "persona_emoji": "🤖",
        "dynamic_reactions": True,
        "reaction_cooldown": 0,
    }
    ad = DiscordAdapter(cfg)
    ad._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=1, name="Bot"),
    )
    raw = LedgerMessage(msg_id=4)
    src = _make_source(chat_id="126")
    evt = _make_event("4", raw, source=src)
    await ad.on_processing_start(evt)
    emoji_map = {"read_file": "📄", "web_search": "🔍"}

    def fake_emoji(name, default="⚙️"):
        return emoji_map.get(name, default)

    with patch("agent.display.get_tool_emoji", side_effect=fake_emoji):
        with patch.object(ad, "_reaction_add", return_value=False):
            await ad.on_tool_call_start(src, "read_file")
            # Should preserve persona
            assert raw.effective() == {"🤖"}
            key = ad._session_key_from_source(src)
            assert ad._rxn_active[key] == "🤖"
        # Next tool should succeed
        await ad.on_tool_call_start(src, "web_search")
        assert raw.effective() == {"🔍"}
        assert ("add", "🔍") in raw.ledger()


@pytest.mark.asyncio
async def test_final_add_false_preserves_and_no_lock_leak_and_recovery():
    """Final persona add returning False must not leak lock and must remain recoverable."""
    cfg = PlatformConfig(enabled=True, token="***")
    cfg.extra = {
        "persona_emoji": "🦊",
        "dynamic_reactions": True,
        "reaction_cooldown": 0,
    }
    ad = DiscordAdapter(cfg)
    ad._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=1, name="Bot"),
    )
    raw = LedgerMessage(msg_id=5)
    src = _make_source(chat_id="127")
    evt = _make_event("5", raw, source=src)
    await ad.on_processing_start(evt)
    emoji_map = {"read_file": "📄"}

    def fake_emoji(name, default="⚙️"):
        return emoji_map.get(name, default)

    with patch("agent.display.get_tool_emoji", side_effect=fake_emoji):
        await ad.on_tool_call_start(src, "read_file")
        assert raw.effective() == {"📄"}
        key = ad._session_key_from_source(src)
        # Patch final add to fail
        orig_add = ad._reaction_add

        async def failing_add(msg_ref, emoji):
            if emoji == "🦊":
                return False
            return await orig_add(msg_ref, emoji)

        with patch.object(ad, "_reaction_add", side_effect=failing_add):
            await ad.on_processing_complete(evt, ProcessingOutcome.SUCCESS)
            # Should NOT have cleaned tracking; current remains 📄, stale maybe, lock released
            assert (
                key in ad._rxn_active or key in ad._rxn_stale or key in ad._rxn_msg_refs
            ), "state must be preserved for recovery"
            assert key not in ad._rxn_locks, "lock must be released even on False"
            assert "📄" in raw.effective()
            assert "🦊" not in raw.effective()
            # Now recover with a subsequent completion retry (simulate later event)
        # Next event: same key but new message? For recovery we reuse same event with success
        await ad.on_processing_complete(evt, ProcessingOutcome.SUCCESS)
        assert raw.effective() == {"🦊"}
        assert key not in ad._rxn_active
        assert key not in ad._rxn_locks
        assert key not in ad._rxn_stale


@pytest.mark.asyncio
async def test_final_remove_false_preserves_stale_and_recovers():
    cfg = PlatformConfig(enabled=True, token="***")
    cfg.extra = {
        "persona_emoji": "🤖",
        "dynamic_reactions": True,
        "reaction_cooldown": 0,
    }
    ad = DiscordAdapter(cfg)
    ad._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=1, name="Bot"),
    )
    raw = LedgerMessage(msg_id=6)
    src = _make_source(chat_id="128")
    evt = _make_event("6", raw, source=src)
    await ad.on_processing_start(evt)
    emoji_map = {"read_file": "📄"}

    def fake_emoji(name, default="⚙️"):
        return emoji_map.get(name, default)

    with patch("agent.display.get_tool_emoji", side_effect=fake_emoji):
        await ad.on_tool_call_start(src, "read_file")
    assert raw.effective() == {"📄"}
    key = ad._session_key_from_source(src)
    # Make final remove fail (add succeeds, remove fails)
    orig_rm = ad._reaction_remove

    async def failing_rm(msg_ref, emoji):
        if emoji == "📄":
            return False
        return await orig_rm(msg_ref, emoji)

    with patch.object(ad, "_reaction_remove", side_effect=failing_rm):
        await ad.on_processing_complete(evt, ProcessingOutcome.SUCCESS)
        # Add succeeded, so persona added, but old tool remains
        assert "🤖" in raw.effective()
        assert "📄" in raw.effective(), "stale remains due to remove False"
        assert key in ad._rxn_stale
        assert "📄" in ad._rxn_stale[key]
        assert ad._rxn_active[key] == "🤖"
        assert key not in ad._rxn_locks
        # Next retry should clean
    await ad.on_processing_complete(evt, ProcessingOutcome.SUCCESS)
    assert raw.effective() == {"🤖"}
    assert "📄" not in raw.effective()
    assert key not in ad._rxn_stale


@pytest.mark.asyncio
async def test_cancel_with_remove_false_preserves_and_no_lock_leak():
    cfg = PlatformConfig(enabled=True, token="***")
    cfg.extra = {
        "persona_emoji": "🤖",
        "dynamic_reactions": True,
        "reaction_cooldown": 0,
    }
    ad = DiscordAdapter(cfg)
    ad._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=1, name="Bot"),
    )
    raw = LedgerMessage(msg_id=7)
    src = _make_source(chat_id="129")
    evt = _make_event("7", raw, source=src)
    await ad.on_processing_start(evt)
    emoji_map = {"read_file": "📄"}

    def fake_emoji(name, default="⚙️"):
        return emoji_map.get(name, default)

    with patch("agent.display.get_tool_emoji", side_effect=fake_emoji):
        await ad.on_tool_call_start(src, "read_file")
    key = ad._session_key_from_source(src)
    with patch.object(ad, "_reaction_remove", return_value=False):
        await ad.on_processing_complete(evt, ProcessingOutcome.CANCELLED)
        # Cancel should attempt remove but fail, so stale preserved and lock released
        assert key in ad._rxn_stale or key in ad._rxn_active
        assert key not in ad._rxn_locks
        # Remote still has tool emoji
        assert "📄" in raw.effective()
    # Retry cancel should eventually clean when remove succeeds
    await ad.on_processing_complete(evt, ProcessingOutcome.CANCELLED)
    assert raw.effective() == set()
    assert key not in ad._rxn_active
    assert key not in ad._rxn_stale


@pytest.mark.asyncio
async def test_lock_cleanup_after_all_false_paths():
    """Ensure per-key lock never leaks after any False path."""
    for outcome in [
        ProcessingOutcome.SUCCESS,
        ProcessingOutcome.FAILURE,
        ProcessingOutcome.CANCELLED,
    ]:
        cfg = PlatformConfig(enabled=True, token="***")
        cfg.extra = {
            "persona_emoji": "🤖",
            "dynamic_reactions": True,
            "reaction_cooldown": 0,
        }
        ad = DiscordAdapter(cfg)
        ad._client = SimpleNamespace(
            tree=FakeTree(),
            get_channel=lambda _id: None,
            fetch_channel=AsyncMock(),
            user=SimpleNamespace(id=1, name="Bot"),
        )
        raw = LedgerMessage(msg_id=100)
        src = _make_source(chat_id=f"lock-{outcome.name}")
        evt = _make_event("99", raw, source=src)
        # Force start to succeed
        await ad.on_processing_start(evt)
        key = ad._session_key_from_source(src)
        # Force final to fail via add False
        with patch.object(ad, "_reaction_add", return_value=False):
            await ad.on_processing_complete(evt, outcome)
            assert key not in ad._rxn_locks, f"lock leaked for {outcome}"


# ---------------------------------------------------------------------------
# Cooldown and boolean normalization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cooldown_fail_closed_nan_negative_and_string_false():
    """Cooldown NaN/negative/inf and quoted-false booleans must fail-closed, not disable hysteresis."""
    # Test cooldown normalization directly via adapter construction
    for raw_val, expected_default in [
        ("nan", 1.0),
        (float("nan"), 1.0),
        (-1, 1.0),
        (-0.5, 1.0),
        (float("inf"), 1.0),
        ("-2", 1.0),
        ("not-a-number", 1.0),
    ]:
        cfg = PlatformConfig(
            enabled=True, token="***", extra={"reaction_cooldown": raw_val}
        )
        ad = DiscordAdapter(cfg)
        ad._client = SimpleNamespace(
            tree=FakeTree(),
            get_channel=lambda _id: None,
            fetch_channel=AsyncMock(),
            user=SimpleNamespace(id=1, name="Bot"),
        )
        # _rxn_cooldown should be default 1.0, not the bad value
        assert ad._rxn_cooldown == 1.0, (
            f"cooldown {raw_val!r} should fallback to 1.0, got {ad._rxn_cooldown}"
        )
        # Verify hysteresis still applies: rapid second tool swap should be coalesced
        raw = LedgerMessage(msg_id=200)
        src = _make_source(chat_id=f"cool-{raw_val}")
        evt = _make_event("200", raw, source=src)
        # Need persona: cooldown already 1.0 via fail-closed normalization
        await ad.on_processing_start(evt)
        emoji_map = {"read_file": "📄", "web_search": "🔍"}

        def fake_emoji(name, default="⚙️"):
            return emoji_map.get(name, default)

        with patch("agent.display.get_tool_emoji", side_effect=fake_emoji):
            await ad.on_tool_call_start(src, "read_file")
            assert raw.effective() == {"📄"}
            # Immediate second swap within cooldown should be blocked
            await ad.on_tool_call_start(src, "web_search")
            assert raw.effective() == {"📄"}, (
                "cooldown should block rapid swap even with bad original config"
            )
    # Test quoted-false booleans for dynamic_reactions
    for false_val in ["false", "False", "0", "no", "off", "FALSE"]:
        cfg = PlatformConfig(
            enabled=True,
            token="***",
            extra={
                "dynamic_reactions": false_val,
                "persona_emoji": "🤖",
                "reaction_cooldown": 0,
            },
        )
        ad = DiscordAdapter(cfg)
        ad._client = SimpleNamespace(
            tree=FakeTree(),
            get_channel=lambda _id: None,
            fetch_channel=AsyncMock(),
            user=SimpleNamespace(id=1, name="Bot"),
        )
        assert ad._rxn_dynamic is False, (
            f"dynamic_reactions={false_val!r} should be False, got {ad._rxn_dynamic}"
        )
        # Verify no swap occurs
        raw = LedgerMessage(msg_id=201)
        src = _make_source(chat_id=f"bool-{false_val}")
        evt = _make_event("201", raw, source=src)
        await ad.on_processing_start(evt)
        with patch("agent.display.get_tool_emoji", return_value="📄"):
            await ad.on_tool_call_start(src, "read_file")
            # Should still be persona, not tool
            assert raw.effective() == {"🤖"}
    # 0 cooldown should be allowed (disables hysteresis for tests)
    cfg = PlatformConfig(enabled=True, token="***", extra={"reaction_cooldown": 0})
    ad = DiscordAdapter(cfg)
    ad._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=1, name="Bot"),
    )
    assert ad._rxn_cooldown == 0.0


# ---------------------------------------------------------------------------
# Per-platform config plumbing via real temp config (no _rxn mutation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_platform_config_via_real_temp_config_loader(tmp_path, monkeypatch):
    """Wire platforms.discord.persona_emoji/dynamic_reactions through normal loader/model/adapter path."""
    # Create a real HERMES_HOME with config.yaml containing per-platform overrides
    home = tmp_path / "hermes_home_cf"
    home.mkdir()
    config_yaml = """
platforms:
  discord:
    persona_emoji: "🦊"
    dynamic_reactions: false
    reaction_cooldown: 0.75
    reactions: true
"""
    (home / "config.yaml").write_text(config_yaml, encoding="utf-8")
    # Also test global defaults: create hermes_cli config layer?
    # For gateway loader, we need to set HERMES_HOME override
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from gateway.config import load_gateway_config
    from gateway.config_loader import load_yaml_layer

    token = set_hermes_home_override(home)
    try:
        # Use production loader
        cfg = load_gateway_config()
        disc = cfg.platforms.get(Platform.DISCORD)
        assert disc is not None, "discord platform should be present"
        # extra should contain bridged keys
        assert disc.extra.get("persona_emoji") == "🦊"
        assert (
            disc.extra.get("dynamic_reactions") is False
            or disc.extra.get("dynamic_reactions") == False
        )
        # reaction_cooldown may be string or float; loader preserves raw, mixin normalizes
        assert str(disc.extra.get("reaction_cooldown")) == "0.75"
        # Now construct production adapter without mutating _rxn_*
        ad = DiscordAdapter(disc)
        ad._client = SimpleNamespace(
            tree=FakeTree(),
            get_channel=lambda _id: None,
            fetch_channel=AsyncMock(),
            user=SimpleNamespace(id=1, name="Bot"),
        )
        # Adapter's mixin should have resolved via production path
        assert ad._rxn_persona_emoji == "🦊", f"expected 🦊 got {ad._rxn_persona_emoji}"
        assert ad._rxn_dynamic is False, f"expected False got {ad._rxn_dynamic}"
        assert math.isclose(ad._rxn_cooldown, 0.75, rel_tol=1e-6)
        # Verify add-before-remove still works with this config (dynamic False => no swaps)
        raw = LedgerMessage(msg_id=300)
        src = _make_source(chat_id="cfg1")
        evt = _make_event("300", raw, source=src)
        await ad.on_processing_start(evt)
        assert raw.effective() == {"🦊"}
        with patch("agent.display.get_tool_emoji", return_value="📄"):
            await ad.on_tool_call_start(src, "read_file")
            # Dynamic disabled => no swap
            assert raw.effective() == {"🦊"}
        await ad.on_processing_complete(evt, ProcessingOutcome.SUCCESS)
        assert raw.effective() == {"🦊"}
        assert ad._rxn_active == {}
        assert ad._rxn_locks == {}
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_per_platform_config_precedence_over_global_and_malformed_fallback(
    tmp_path,
):
    """Global vs per-platform precedence and malformed fail-closed via real temp loader."""
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from gateway.config import load_gateway_config

    # Phase 1: per-platform wins over global; quoted-false and NaN fail-closed
    home1 = tmp_path / "home_precedence"
    home1.mkdir()
    (home1 / "config.yaml").write_text(
        'persona_emoji: "🌟"\n'
        "dynamic_reactions: true\n"
        "platforms:\n"
        "  discord:\n"
        "    enabled: true\n"
        '    persona_emoji: "🐱"\n'
        '    dynamic_reactions: "false"\n'
        '    reaction_cooldown: "NaN"\n'
        "    reactions: true\n",
        encoding="utf-8",
    )
    token = set_hermes_home_override(home1)
    try:
        cfg = load_gateway_config()
        disc = cfg.platforms.get(Platform.DISCORD)
        assert disc is not None
        # extra bridged via production loader
        assert disc.extra.get("persona_emoji") == "🐱"
        assert disc.extra.get("dynamic_reactions") == "false"
        assert str(disc.extra.get("reaction_cooldown")) == "NaN"
        ad = DiscordAdapter(disc)
        ad._client = SimpleNamespace(
            tree=FakeTree(),
            get_channel=lambda _id: None,
            fetch_channel=AsyncMock(),
            user=SimpleNamespace(id=1, name="Bot"),
        )
        assert ad._rxn_persona_emoji == "🐱", (
            f"per-platform should win over global 🌟, got {ad._rxn_persona_emoji}"
        )
        assert ad._rxn_dynamic is False, "quoted 'false' must be False, not truthy"
        assert ad._rxn_cooldown == 1.0, (
            f"NaN cooldown must fail-closed to 1.0, got {ad._rxn_cooldown}"
        )
        # Verify behavior: dynamic false keeps persona, no tool swap
        raw = LedgerMessage(msg_id=310)
        src = _make_source(chat_id="prec1")
        evt = _make_event("310", raw, source=src)
        await ad.on_processing_start(evt)
        assert raw.effective() == {"🐱"}
        with patch("agent.display.get_tool_emoji", return_value="📄"):
            await ad.on_tool_call_start(src, "read_file")
            assert raw.effective() == {"🐱"}, "dynamic false should not swap"
    finally:
        reset_hermes_home_override(token)

    # Phase 2: global fallback when per-platform absent
    home2 = tmp_path / "home_global_fallback"
    home2.mkdir()
    (home2 / "config.yaml").write_text(
        'persona_emoji: "🌟"\n'
        "dynamic_reactions: true\n"
        "platforms:\n"
        "  discord:\n"
        "    enabled: true\n"
        "    reactions: true\n",
        encoding="utf-8",
    )
    token = set_hermes_home_override(home2)
    try:
        cfg = load_gateway_config()
        disc = cfg.platforms.get(Platform.DISCORD)
        assert disc is not None
        assert disc.extra.get("persona_emoji") in (None, ""), (
            "per-platform persona should be absent"
        )
        ad2 = DiscordAdapter(disc)
        ad2._client = SimpleNamespace(
            tree=FakeTree(),
            get_channel=lambda _id: None,
            fetch_channel=AsyncMock(),
            user=SimpleNamespace(id=1, name="Bot"),
        )
        assert ad2._rxn_persona_emoji == "🌟", (
            f"global persona should be used when per-platform absent, got {ad2._rxn_persona_emoji}"
        )
        assert ad2._rxn_dynamic is True, (
            "global dynamic true should apply when per-platform absent"
        )
    finally:
        reset_hermes_home_override(token)

    # Phase 3: negative and string-false cooldown fail-closed to 1.0 via real loader
    home3 = tmp_path / "home_malformed_cooldown"
    home3.mkdir()
    (home3 / "config.yaml").write_text(
        "platforms:\n"
        "  discord:\n"
        "    enabled: true\n"
        '    persona_emoji: "🤖"\n'
        "    dynamic_reactions: true\n"
        "    reaction_cooldown: -5\n",
        encoding="utf-8",
    )
    token = set_hermes_home_override(home3)
    try:
        cfg = load_gateway_config()
        disc = cfg.platforms.get(Platform.DISCORD)
        ad3 = DiscordAdapter(disc)
        ad3._client = SimpleNamespace(
            tree=FakeTree(),
            get_channel=lambda _id: None,
            fetch_channel=AsyncMock(),
            user=SimpleNamespace(id=1, name="Bot"),
        )
        assert ad3._rxn_cooldown == 1.0, (
            f"negative cooldown must fail-closed to 1.0, got {ad3._rxn_cooldown}"
        )
        # Also test quoted negative string
        home3b = tmp_path / "home_malformed_cooldown2"
        home3b.mkdir()
        (home3b / "config.yaml").write_text(
            'platforms:\n  discord:\n    enabled: true\n    reaction_cooldown: "-2"\n',
            encoding="utf-8",
        )
        reset_hermes_home_override(token)
        token = set_hermes_home_override(home3b)
        cfg = load_gateway_config()
        disc = cfg.platforms.get(Platform.DISCORD)
        ad3b = DiscordAdapter(disc)
        ad3b._client = SimpleNamespace(
            tree=FakeTree(),
            get_channel=lambda _id: None,
            fetch_channel=AsyncMock(),
            user=SimpleNamespace(id=1, name="Bot"),
        )
        assert ad3b._rxn_cooldown == 1.0
    finally:
        reset_hermes_home_override(token)

    # Phase 4: quoted-false variants via real loader still fail-closed
    for false_val in ["false", "False", "0", "no", "off"]:
        home = tmp_path / f"home_false_{false_val}"
        home.mkdir()
        (home / "config.yaml").write_text(
            f'platforms:\n  discord:\n    enabled: true\n    persona_emoji: "🤖"\n    dynamic_reactions: "{false_val}"\n    reaction_cooldown: 0\n',
            encoding="utf-8",
        )
        token = set_hermes_home_override(home)
        try:
            cfg = load_gateway_config()
            disc = cfg.platforms.get(Platform.DISCORD)
            ad = DiscordAdapter(disc)
            ad._client = SimpleNamespace(
                tree=FakeTree(),
                get_channel=lambda _id: None,
                fetch_channel=AsyncMock(),
                user=SimpleNamespace(id=1, name="Bot"),
            )
            assert ad._rxn_dynamic is False, (
                f"dynamic_reactions={false_val!r} should be False via loader, got {ad._rxn_dynamic}"
            )
        finally:
            reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_per_platform_config_absent_fallback_to_default(tmp_path):
    """When per-platform and global absent, fallback to defaults (👀 and 1.0, dynamic from global default)."""
    home = tmp_path / "home_absent"
    home.mkdir()
    (home / "config.yaml").write_text(
        "platforms:\n  discord:\n    enabled: true\n    reactions: true\n",
        encoding="utf-8",
    )
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from gateway.config import load_gateway_config

    token = set_hermes_home_override(home)
    try:
        cfg = load_gateway_config()
        disc = cfg.platforms.get(Platform.DISCORD)
        assert disc is not None
        assert disc.extra.get("persona_emoji") in (None, "")
        ad = DiscordAdapter(disc)
        ad._client = SimpleNamespace(
            tree=FakeTree(),
            get_channel=lambda _id: None,
            fetch_channel=AsyncMock(),
            user=SimpleNamespace(id=1, name="Bot"),
        )
        # Concrete expectations: persona falls back to 👀, cooldown to 1.0, dynamic to global default True
        assert ad._rxn_persona_emoji == "👀", (
            f"expected fallback 👀, got {ad._rxn_persona_emoji}"
        )
        assert ad._rxn_dynamic is True, (
            f"absent per-platform/global should fallback to global default True, got {ad._rxn_dynamic}"
        )
        assert ad._rxn_cooldown == 1.0, (
            f"expected default cooldown 1.0, got {ad._rxn_cooldown}"
        )
        # Verify no crash on lifecycle with defaults
        raw = LedgerMessage(msg_id=320)
        src = _make_source(chat_id="absent1")
        evt = _make_event("320", raw, source=src)
        await ad.on_processing_start(evt)
        assert raw.ledger() == [("add", "👀")]
        assert raw.effective() == {"👀"}
    finally:
        reset_hermes_home_override(token)


# ---------------------------------------------------------------------------
# Telegram regression (replace-mode) and add-before-remove ordering
# ---------------------------------------------------------------------------


class FakeTelegramAdapter(DiscordAdapter):
    """Minimal telegram-like adapter using replace_mode to ensure no regression."""

    _reaction_replace_mode = True

    def __init__(self, config):
        # Don't call DiscordAdapter.__init__ which expects discord-specific, just init mixin
        self.config = config
        self.platform = Platform.TELEGRAM
        self._client = SimpleNamespace(user=SimpleNamespace(id=1))
        self._session_raw_messages = {}
        self._init_reaction_mixin()

    async def _reaction_set(self, msg_ref, emoji):
        # Simulate Telegram set that replaces; ledger is msg_ref
        try:
            await msg_ref.set_reaction(emoji)
            return True
        except Exception:
            return False

    def _reaction_resolve_message(self, event):
        return getattr(event, "raw_message", None) or self._session_raw_messages.get(
            self._reaction_msg_key(event)
        )

    def _reaction_msg_key(self, event):
        source = getattr(event, "source", event)
        if hasattr(source, "source") and source.source is not None:
            source = source.source
        return f"{source.platform}:{source.chat_id}:{getattr(source, 'user_id', '')}"


class TelegramLedger:
    def __init__(self):
        self._current = None
        self._ledger = []

    async def set_reaction(self, emoji):
        self._ledger.append(("set", emoji))
        self._current = emoji

    def ledger(self):
        return list(self._ledger)

    def current(self):
        return self._current


@pytest.mark.asyncio
async def test_telegram_replace_mode_lifecycle():
    """Telegram replace-mode must still work: persona start, swaps via set, final persona."""
    cfg = PlatformConfig(
        enabled=True,
        token="***",
        extra={
            "persona_emoji": "🤖",
            "dynamic_reactions": True,
            "reaction_cooldown": 0,
        },
    )
    ad = FakeTelegramAdapter(cfg)
    raw = TelegramLedger()
    src = SessionSource(
        platform=Platform.TELEGRAM, chat_id="tg-1", chat_type="dm", user_id="u1"
    )
    evt = MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=src,
        raw_message=raw,
        message_id="tgm1",
    )
    await ad._rxn_on_processing_start(evt)
    assert raw.ledger() == [("set", "🤖")]
    assert raw.current() == "🤖"
    emoji_map = {"read_file": "📄", "web_search": "🔍"}

    def fake_emoji(name, default="⚙️"):
        return emoji_map.get(name, default)

    with patch("agent.display.get_tool_emoji", side_effect=fake_emoji):
        await ad._rxn_on_tool_call_start(src, "read_file")
        assert raw.ledger()[-1] == ("set", "📄")
        assert raw.current() == "📄"
        await ad._rxn_on_tool_call_start(src, "web_search")
        assert raw.ledger()[-1] == ("set", "🔍")
        assert raw.current() == "🔍"
    await ad._rxn_on_processing_complete(evt, ProcessingOutcome.SUCCESS)
    assert raw.ledger()[-1] == ("set", "🤖")
    assert raw.current() == "🤖"
    # Ensure no lock leak
    assert ad._rxn_locks == {}


@pytest.mark.asyncio
async def test_add_before_remove_ordering_proven_via_ledger():
    """Every tool transition must be add(new) before remove(old) in ledger order."""
    cfg = PlatformConfig(
        enabled=True,
        token="***",
        extra={
            "persona_emoji": "🤖",
            "dynamic_reactions": True,
            "reaction_cooldown": 0,
        },
    )
    ad = DiscordAdapter(cfg)
    ad._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=1, name="Bot"),
    )
    raw = LedgerMessage(msg_id=400)
    src = _make_source(chat_id="order-1")
    evt = _make_event("400", raw, source=src)
    await ad.on_processing_start(evt)
    emoji_map = {"a": "🅰️", "b": "🅱️", "c": "🇨"}

    def fake_emoji(name, default="⚙️"):
        return emoji_map.get(name, default)

    with patch("agent.display.get_tool_emoji", side_effect=fake_emoji):
        await ad.on_tool_call_start(src, "a")
        assert raw.ledger()[1] == ("add", "🅰️")
        assert raw.ledger()[2] == ("remove", "🤖")
        await ad.on_tool_call_start(src, "b")
        assert raw.ledger()[3] == ("add", "🅱️")
        assert raw.ledger()[4] == ("remove", "🅰️")
        await ad.on_tool_call_start(src, "c")
        assert raw.ledger()[5] == ("add", "🇨")
        assert raw.ledger()[6] == ("remove", "🅱️")
        # Final must also be add before remove
        pre = len(raw.ledger())
        await ad.on_processing_complete(evt, ProcessingOutcome.SUCCESS)
        assert raw.ledger()[pre] == ("add", "🤖")
        assert raw.ledger()[pre + 1] == ("remove", "🇨")


# ---------------------------------------------------------------------------
# Regression: stale cleanup remains reachable across same-key turns (RXN-STALE-NEXT-TURN)
# ---------------------------------------------------------------------------


class FailableLedgerMessage(LedgerMessage):
    """Ledger that fails the first remove of a specific emoji to simulate transient API failure."""

    def __init__(self, msg_id=12345, fail_once_emoji=None, fail_twice=False):
        super().__init__(msg_id=msg_id)
        self.fail_once_emoji = fail_once_emoji
        self.fail_twice = fail_twice
        # override to track attempts
        self.remove_attempts: dict[str, int] = {}
        # replace the AsyncMock with our custom side_effect that can fail
        self.remove_reaction = AsyncMock(side_effect=self._failable_remove)

    async def _failable_remove(self, emoji, user=None):
        attempts = self.remove_attempts.get(emoji, 0)
        self.remove_attempts[emoji] = attempts + 1
        self._ledger.append(("remove", emoji))
        should_fail = False
        if self.fail_once_emoji == emoji:
            if self.fail_twice:
                should_fail = attempts < 2
            else:
                should_fail = attempts == 0
        if should_fail:
            raise Exception(f"simulated transient remove failure for {emoji}")
        self._effective.discard(emoji)
        return None


@pytest.mark.asyncio
async def test_rxn_stale_next_turn_retains_cleanup_reachability():
    """Same-key second start must retain failed stale from first message and drain it without cross-turn targeting."""
    # Use production-constructor path: PlatformConfig.extra before adapter init
    cfg = PlatformConfig(
        enabled=True,
        token="***",
        extra={
            "persona_emoji": "🤖",
            "dynamic_reactions": True,
            "reaction_cooldown": 0,
        },
    )
    ad = DiscordAdapter(cfg)
    ad._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    source = _make_source(chat_id="123")
    # First turn: start + tool swap + complete where final remove fails
    msg1 = FailableLedgerMessage(msg_id=1, fail_once_emoji="📄")
    evt1 = _make_event("1", msg1, source)
    await ad.on_processing_start(evt1)
    assert ("add", "🤖") in msg1.ledger()
    with patch("agent.display.get_tool_emoji", return_value="📄"):
        await ad.on_tool_call_start(source, "read_file")
    assert msg1.effective() == {"📄"}
    # Complete will add 🤖 and try to remove 📄 which will fail (first attempt)
    await ad.on_processing_complete(evt1, ProcessingOutcome.SUCCESS)
    # First message should retain both due to failed remove, and stale tracked
    assert msg1.effective() == {"🤖", "📄"}
    assert ad._rxn_stale.get(ad._reaction_msg_key(evt1)) == {"📄"}
    assert ad._rxn_active.get(ad._reaction_msg_key(evt1)) == "🤖"
    # Second same-key turn with new message
    msg2 = LedgerMessage(msg_id=2)
    evt2 = _make_event("2", msg2, source)
    await ad.on_processing_start(evt2)
    # Old stale should have been drained from msg1 (second attempt succeeds)
    assert msg1.effective() == {"🤖"}, (
        f"old msg1 stale should be drained, got {msg1.effective()} ledger {msg1.ledger()}"
    )
    assert msg2.effective() == {"🤖"}
    # Pending should be cleared after successful drain
    key = ad._reaction_msg_key(evt2)
    assert not ad._rxn_pending.get(key), (
        f"pending should be empty after successful drain, got {ad._rxn_pending.get(key)}"
    )
    # No cross-turn targeting: msg2 ledger must not contain remove of old emoji
    assert ("remove", "📄") not in msg2.ledger()
    # msg1 ledger should show the drain remove as second remove of 📄
    assert msg1.ledger().count(("remove", "📄")) == 2
    # Generation should have advanced
    assert ad._rxn_generation.get(key) == 2
    # Cleanup: complete second turn normally
    await ad.on_processing_complete(evt2, ProcessingOutcome.SUCCESS)
    assert ad._rxn_active == {}
    assert ad._rxn_locks == {}


@pytest.mark.asyncio
async def test_rxn_stale_next_turn_pending_retains_and_eventually_drained():
    """When drain fails again, pending retains per-message stale and is eventually drained via complete."""
    cfg = PlatformConfig(
        enabled=True,
        token="***",
        extra={
            "persona_emoji": "🤖",
            "dynamic_reactions": True,
            "reaction_cooldown": 0,
        },
    )
    ad = DiscordAdapter(cfg)
    ad._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    source = _make_source(chat_id="123")
    msg1 = FailableLedgerMessage(msg_id=10, fail_once_emoji="📄", fail_twice=True)
    evt1 = _make_event("10", msg1, source)
    await ad.on_processing_start(evt1)
    with patch("agent.display.get_tool_emoji", return_value="📄"):
        await ad.on_tool_call_start(source, "read_file")
    await ad.on_processing_complete(evt1, ProcessingOutcome.SUCCESS)
    assert msg1.effective() == {"🤖", "📄"}
    # Second start: drain will fail again (second attempt)
    msg2 = LedgerMessage(msg_id=11)
    evt2 = _make_event("11", msg2, source)
    await ad.on_processing_start(evt2)
    # Old msg1 still has stale because drain failed again
    assert msg1.effective() == {"🤖", "📄"}
    key = ad._reaction_msg_key(evt2)
    pending = ad._rxn_pending.get(key)
    assert pending is not None and len(pending) == 1
    assert pending[0][0] is msg1
    assert "📄" in pending[0][1]
    assert msg2.effective() == {"🤖"}
    assert ("remove", "📄") not in msg2.ledger()
    # Complete second turn should drain pending (third attempt succeeds)
    await ad.on_processing_complete(evt2, ProcessingOutcome.SUCCESS)
    assert msg1.effective() == {"🤖"}
    assert not ad._rxn_pending.get(key)


# ---------------------------------------------------------------------------
# Regression: durable acknowledgement must not record false success (Raven)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rxn_durable_ack_not_false_when_disabled_or_primitive_fails(monkeypatch):
    """Disabled, missing cap, False and exception starts must not record emoji_ack=True."""
    source = _make_source(chat_id="123")

    # Helper to capture ack via patching _record_discord_processing_start
    def _capture_ack_for(adapter, event):
        captured = {}
        orig = adapter._record_discord_processing_start

        def wrapper(evt, *, emoji_ack):
            captured["ack"] = emoji_ack
            return orig(evt, emoji_ack=emoji_ack)

        # Use monkeypatch to avoid recursion? We'll patch via patch.object
        return captured, wrapper

    # 1. Disabled via extra reactions=False
    cfg = PlatformConfig(
        enabled=True,
        token="***",
        extra={"reactions": False, "persona_emoji": "🤖", "reaction_cooldown": 0},
    )
    ad = DiscordAdapter(cfg)
    ad._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    # Ensure backfill enabled to allow recording path to run, but we will capture via patch
    # _missed_message_backfill_enabled reads config; patch it to True for test
    with patch.object(ad, "_missed_message_backfill_enabled", return_value=True):
        with patch.object(ad, "_record_discord_processing_start") as mock_record:
            with patch.object(ad, "_record_discord_message_seen"):
                raw = LedgerMessage(msg_id=100)
                evt = _make_event("100", raw, source)
                await ad.on_processing_start(evt)
                # Should have been called with emoji_ack=False because reactions disabled
                assert mock_record.called, "record should be called even when disabled"
                ack = (
                    mock_record.call_args.kwargs.get("emoji_ack")
                    if mock_record.call_args.kwargs
                    else mock_record.call_args[1].get("emoji_ack")
                    if len(mock_record.call_args) > 1
                    else None
                )
                # Extract from args: second arg is emoji_ack via kw
                if ack is None and mock_record.call_args.args:
                    # positional? but our code uses kw
                    pass
                # More robust: check call
                called_ack = (
                    mock_record.call_args[1]["emoji_ack"]
                    if len(mock_record.call_args) > 1
                    else mock_record.call_args.kwargs["emoji_ack"]
                )
                assert called_ack is False, (
                    f"disabled should record ack False, got {called_ack}"
                )
                assert raw.ledger() == [], "disabled should not add reaction"

    # 2. Primitive returns False
    cfg2 = PlatformConfig(
        enabled=True, token="***", extra={"persona_emoji": "🤖", "reaction_cooldown": 0}
    )
    ad2 = DiscordAdapter(cfg2)
    ad2._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    # Make _add_reaction return False
    with patch.object(ad2, "_add_reaction", new=AsyncMock(return_value=False)):
        with patch.object(ad2, "_missed_message_backfill_enabled", return_value=True):
            with patch.object(ad2, "_record_discord_processing_start") as mock_record2:
                with patch.object(ad2, "_record_discord_message_seen"):
                    raw2 = LedgerMessage(msg_id=101)
                    evt2 = _make_event("101", raw2, source)
                    await ad2.on_processing_start(evt2)
                    called_ack2 = (
                        mock_record2.call_args[1]["emoji_ack"]
                        if len(mock_record2.call_args) > 1
                        else mock_record2.call_args.kwargs["emoji_ack"]
                    )
                    assert called_ack2 is False, (
                        f"primitive False should record ack False, got {called_ack2}"
                    )
                    # Ledger should be empty because add failed at adapter level (we mocked _add_reaction, so Ledger not used)
                    # But ensure no active tracking
                    assert ad2._rxn_active.get(ad2._reaction_msg_key(evt2)) is None

    # 3. Exception during add
    cfg3 = PlatformConfig(
        enabled=True, token="***", extra={"persona_emoji": "🤖", "reaction_cooldown": 0}
    )
    ad3 = DiscordAdapter(cfg3)
    ad3._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    with patch.object(
        ad3, "_add_reaction", new=AsyncMock(side_effect=Exception("boom"))
    ):
        with patch.object(ad3, "_missed_message_backfill_enabled", return_value=True):
            with patch.object(ad3, "_record_discord_processing_start") as mock_record3:
                with patch.object(ad3, "_record_discord_message_seen"):
                    raw3 = LedgerMessage(msg_id=102)
                    evt3 = _make_event("102", raw3, source)
                    await ad3.on_processing_start(evt3)
                    called_ack3 = (
                        mock_record3.call_args[1]["emoji_ack"]
                        if len(mock_record3.call_args) > 1
                        else mock_record3.call_args.kwargs["emoji_ack"]
                    )
                    assert called_ack3 is False, (
                        f"exception should record ack False, got {called_ack3}"
                    )

    # 4. Missing capability (no add_reaction)
    cfg4 = PlatformConfig(
        enabled=True, token="***", extra={"persona_emoji": "🤖", "reaction_cooldown": 0}
    )
    ad4 = DiscordAdapter(cfg4)
    ad4._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    raw4 = SimpleNamespace(id=103)  # no add_reaction attribute
    evt4 = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        raw_message=raw4,
        message_id="103",
    )
    with patch.object(ad4, "_missed_message_backfill_enabled", return_value=True):
        with patch.object(ad4, "_record_discord_processing_start") as mock_record4:
            with patch.object(ad4, "_record_discord_message_seen"):
                await ad4.on_processing_start(evt4)
                called_ack4 = (
                    mock_record4.call_args[1]["emoji_ack"]
                    if len(mock_record4.call_args) > 1
                    else mock_record4.call_args.kwargs["emoji_ack"]
                )
                assert called_ack4 is False, (
                    f"missing capability should record ack False, got {called_ack4}"
                )


# ---------------------------------------------------------------------------
# Regression: config fail-closed via real HERMES_HOME YAML and production adapter construction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rxn_config_malformed_fail_closed_via_real_loader(tmp_path, monkeypatch):
    """Only documented scalar bool forms are accepted; malformed strings/lists/dicts and overflow fail closed."""
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from gateway.config import load_gateway_config

    # 1. Direct PlatformConfig extra malformed should fail closed (production adapter construction, no _rxn_* mutation)
    for val in [123, ["true"], {"enabled": True}]:
        cfg = PlatformConfig(
            enabled=True,
            token="***",
            extra={
                "dynamic_reactions": val,
                "persona_emoji": "🤖",
                "reaction_cooldown": 0,
            },
        )
        ad = DiscordAdapter(cfg)
        ad._client = SimpleNamespace(
            tree=FakeTree(),
            get_channel=lambda _id: None,
            fetch_channel=AsyncMock(),
            user=SimpleNamespace(id=99999, name="HermesBot"),
        )
        assert ad._rxn_dynamic is False, (
            f"dynamic_reactions {val!r} should be False, got {ad._rxn_dynamic}"
        )

    # reactions garbage via direct extra (string)
    cfg = PlatformConfig(enabled=True, token="***", extra={"reactions": "garbage"})
    ad = DiscordAdapter(cfg)
    ad._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    assert ad._reactions_enabled() is False, "reactions='garbage' should be disabled"

    # reaction_cooldown overflow via direct extra
    cfg = PlatformConfig(
        enabled=True, token="***", extra={"reaction_cooldown": 10**1000}
    )
    ad = DiscordAdapter(cfg)
    ad._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    assert ad._rxn_cooldown == 1.0, (
        f"overflow cooldown should be 1.0, got {ad._rxn_cooldown}"
    )

    # 2. Real HERMES_HOME YAML: dynamic_reactions list and reactions garbage and cooldown overflow
    # YAML with platforms.discord.dynamic_reactions as list
    home = tmp_path / "home_malformed_dynamic"
    home.mkdir()
    (home / "config.yaml").write_text(
        "platforms:\n  discord:\n    enabled: true\n    dynamic_reactions: [true]\n",
        encoding="utf-8",
    )
    token = set_hermes_home_override(home)
    try:
        cfg = load_gateway_config()
        disc = cfg.platforms.get(Platform.DISCORD)
        ad = DiscordAdapter(disc)
        ad._client = SimpleNamespace(
            tree=FakeTree(),
            get_channel=lambda _id: None,
            fetch_channel=AsyncMock(),
            user=SimpleNamespace(id=99999, name="HermesBot"),
        )
        assert ad._rxn_dynamic is False, (
            f"YAML dynamic_reactions [true] should be False, got {ad._rxn_dynamic}"
        )
    finally:
        reset_hermes_home_override(token)
        monkeypatch.delenv("DISCORD_REACTIONS", raising=False)

    # YAML with discord.reactions garbage (via top-level discord block -> env)
    home2 = tmp_path / "home_garbage_reactions"
    home2.mkdir()
    (home2 / "config.yaml").write_text(
        "discord:\n  enabled: true\n  reactions: garbage\n", encoding="utf-8"
    )
    monkeypatch.delenv("DISCORD_REACTIONS", raising=False)
    token = set_hermes_home_override(home2)
    try:
        cfg = load_gateway_config()
        disc = cfg.platforms.get(Platform.DISCORD)
        ad = DiscordAdapter(disc)
        ad._client = SimpleNamespace(
            tree=FakeTree(),
            get_channel=lambda _id: None,
            fetch_channel=AsyncMock(),
            user=SimpleNamespace(id=99999, name="HermesBot"),
        )
        # Env should be set to garbage via _apply_yaml_config, so _reactions_enabled should be False
        assert ad._reactions_enabled() is False, (
            "YAML reactions garbage should be disabled via env"
        )
        assert os.getenv("DISCORD_REACTIONS") == "garbage"
    finally:
        reset_hermes_home_override(token)
        monkeypatch.delenv("DISCORD_REACTIONS", raising=False)

    # YAML with reaction_cooldown overflow
    home3 = tmp_path / "home_overflow"
    home3.mkdir()
    (home3 / "config.yaml").write_text(
        f"platforms:\n  discord:\n    enabled: true\n    reaction_cooldown: {10**1000}\n",
        encoding="utf-8",
    )
    token = set_hermes_home_override(home3)
    try:
        cfg = load_gateway_config()
        disc = cfg.platforms.get(Platform.DISCORD)
        ad = DiscordAdapter(disc)
        ad._client = SimpleNamespace(
            tree=FakeTree(),
            get_channel=lambda _id: None,
            fetch_channel=AsyncMock(),
            user=SimpleNamespace(id=99999, name="HermesBot"),
        )
        assert ad._rxn_cooldown == 1.0, (
            f"YAML overflow cooldown should be 1.0, got {ad._rxn_cooldown}"
        )
    finally:
        reset_hermes_home_override(token)
        monkeypatch.delenv("DISCORD_REACTIONS", raising=False)

    # Also test malformed reactions string via direct extra already done, and via YAML extra.reactions?
    # Ensure no _rxn_* mutation after construction is needed; we already proved via production adapter.
