"""Regression tests for explicit Kanban board routing (task t_6ff59562).

Covers the fix where an explicitly supplied ``board`` argument was silently
overridden by the ``HERMES_KANBAN_DB`` env pin.  The fix introduces scoped
resolution:

- **Unscoped orchestrator** (no ``HERMES_KANBAN_TASK``): explicit ``board``
  is honoured, env pin is bypassed.
- **Worker-scoped** (``HERMES_KANBAN_TASK`` set): explicit ``board`` must
  match the pinned board; ``ValueError`` on mismatch.
- **No explicit board**: env pin / current-board / default fallback is
  unchanged.

Also covers helper functions ``is_worker_scoped``,
``_board_slug_from_db_path``, and ``_resolve_board_db_path``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Helper: is_worker_scoped
# ---------------------------------------------------------------------------


class TestIsWorkerScoped:
    """``is_worker_scoped`` returns True iff HERMES_KANBAN_TASK is set."""

    def test_returns_true_when_task_set(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_123")
        assert kb.is_worker_scoped() is True

    def test_returns_false_when_task_unset(self, monkeypatch):
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        assert kb.is_worker_scoped() is False

    def test_returns_false_when_task_empty(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "")
        assert kb.is_worker_scoped() is False


# ---------------------------------------------------------------------------
# Helper: _board_slug_from_db_path
# ---------------------------------------------------------------------------


class TestBoardSlugFromDbPath:
    """Infer board slug from a HERMES_KANBAN_DB path."""

    def test_default_board_legacy_path(self):
        assert kb._board_slug_from_db_path(Path("/root/kanban.db")) == "default"

    def test_non_default_board(self):
        path = Path("/root/kanban/boards/caribbean-monitor/kanban.db")
        assert kb._board_slug_from_db_path(path) == "caribbean-monitor"

    def test_nested_boards_path(self):
        path = Path("/a/b/c/kanban/boards/my-board/kanban.db")
        assert kb._board_slug_from_db_path(path) == "my-board"


# ---------------------------------------------------------------------------
# Helper: _resolve_board_db_path
# ---------------------------------------------------------------------------


class TestResolveBoardDbPath:
    """Resolve a board slug to its DB path."""

    def test_default_board(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
        result = kb._resolve_board_db_path("default")
        assert result == tmp_path / "kanban.db"

    def test_custom_board(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
        result = kb._resolve_board_db_path("caribbean-monitor")
        assert result == tmp_path / "kanban" / "boards" / "caribbean-monitor" / "kanban.db"


# ---------------------------------------------------------------------------
# Core: kanban_db_path with explicit board
# ---------------------------------------------------------------------------


class TestKanbanDbPathExplicitBoard:
    """Board resolution when an explicit ``board`` arg is passed."""

    def _setup_boards(self, tmp_path):
        """Create two board directories with empty DB files."""
        kanban_home = tmp_path / "kanban_home"
        kanban_home.mkdir()
        # Default board at legacy path.
        (kanban_home / "kanban.db").touch()
        # Non-default board.
        other_board = kanban_home / "kanban" / "boards" / "other-board"
        other_board.mkdir(parents=True)
        (other_board / "kanban.db").touch()
        # Target board (for explicit routing).
        target_board = kanban_home / "kanban" / "boards" / "target-board"
        target_board.mkdir(parents=True)
        (target_board / "kanban.db").touch()
        return kanban_home

    # -- Unscoped orchestrator (no HERMES_KANBAN_TASK) --

    def test_unscoped_explicit_board_bypasses_pin(
        self, tmp_path, monkeypatch
    ):
        """Unscoped caller with explicit board resolves to that board,
        not the env pin."""
        kanban_home = self._setup_boards(tmp_path)
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(kanban_home))
        monkeypatch.setenv(
            "HERMES_KANBAN_DB", str(kanban_home / "kanban.db")
        )
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

        result = kb.kanban_db_path(board="target-board")
        expected = (
            kanban_home / "kanban" / "boards" / "target-board" / "kanban.db"
        )
        assert result == expected

    def test_unscoped_explicit_board_not_env_pin(
        self, tmp_path, monkeypatch
    ):
        """Verify we get the TARGET board, not the pinned default."""
        kanban_home = self._setup_boards(tmp_path)
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(kanban_home))
        monkeypatch.setenv(
            "HERMES_KANBAN_DB", str(kanban_home / "kanban.db")
        )
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

        result = kb.kanban_db_path(board="target-board")
        # Must NOT be the default board path.
        assert result != kanban_home / "kanban.db"

    # -- Worker-scoped (HERMES_KANBAN_TASK set) --

    def test_worker_same_board_uses_pin(
        self, tmp_path, monkeypatch
    ):
        """Worker passing the same board as the pin uses the pinned path."""
        kanban_home = self._setup_boards(tmp_path)
        pinned_db = kanban_home / "kanban.db"
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(kanban_home))
        monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned_db))
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")

        # 'default' matches the legacy pinned path.
        result = kb.kanban_db_path(board="default")
        assert result == pinned_db

    def test_worker_different_board_raises(
        self, tmp_path, monkeypatch
    ):
        """Worker passing a different board than the pin raises ValueError."""
        kanban_home = self._setup_boards(tmp_path)
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(kanban_home))
        monkeypatch.setenv(
            "HERMES_KANBAN_DB", str(kanban_home / "kanban.db")
        )
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")

        with pytest.raises(ValueError, match="board-pinned"):
            kb.kanban_db_path(board="target-board")

    def test_worker_different_board_error_mentions_both_boards(
        self, tmp_path, monkeypatch
    ):
        """The error message mentions both the pinned and requested boards."""
        kanban_home = self._setup_boards(tmp_path)
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(kanban_home))
        monkeypatch.setenv(
            "HERMES_KANBAN_DB", str(kanban_home / "kanban.db")
        )
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")

        with pytest.raises(ValueError, match="target-board"):
            kb.kanban_db_path(board="target-board")

    # -- No-board behaviour (preserve existing fallback) --

    def test_no_board_with_pin_uses_pin(
        self, tmp_path, monkeypatch
    ):
        """When no board arg is passed, the env pin is used (existing)."""
        kanban_home = self._setup_boards(tmp_path)
        pinned_db = kanban_home / "kanban.db"
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(kanban_home))
        monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned_db))
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

        result = kb.kanban_db_path()
        assert result == pinned_db

    def test_no_board_without_pin_uses_current(
        self, tmp_path, monkeypatch
    ):
        """Without env pin and no board arg, falls through to current."""
        kanban_home = self._setup_boards(tmp_path)
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(kanban_home))
        monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

        result = kb.kanban_db_path()
        # Should resolve via get_current_board → default.
        assert result == kanban_home / "kanban.db"

    # -- Explicit same-board selection (unscoped) --

    def test_unscoped_explicit_same_board_as_pin(
        self, tmp_path, monkeypatch
    ):
        """Unscoped caller explicitly requesting the pinned board resolves
        to the board's canonical path (not the pin's literal path)."""
        kanban_home = self._setup_boards(tmp_path)
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(kanban_home))
        monkeypatch.setenv(
            "HERMES_KANBAN_DB", str(kanban_home / "kanban.db")
        )
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

        result = kb.kanban_db_path(board="default")
        assert result == kanban_home / "kanban.db"

    # -- Malformed board slug --

    def test_malformed_board_slug_raises(self, tmp_path, monkeypatch):
        """Invalid board slug raises ValueError."""
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

        with pytest.raises(ValueError, match="invalid board slug"):
            kb.kanban_db_path(board="../escape")

    def test_empty_board_slug_falls_through(
        self, tmp_path, monkeypatch
    ):
        """Empty string board arg is treated as None (no board)."""
        kanban_home = self._setup_boards(tmp_path)
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(kanban_home))
        monkeypatch.setenv(
            "HERMES_KANBAN_DB", str(kanban_home / "kanban.db")
        )
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

        # Empty string normalizes to None → uses env pin.
        result = kb.kanban_db_path(board="")
        assert result == kanban_home / "kanban.db"

    # -- Preservation of task creation behaviour --

    def test_connect_with_explicit_board_unscoped(
        self, tmp_path, monkeypatch
    ):
        """``connect(board=...)`` for unscoped caller opens the right DB."""
        kanban_home = self._setup_boards(tmp_path)
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(kanban_home))
        monkeypatch.setenv(
            "HERMES_KANBAN_DB", str(kanban_home / "kanban.db")
        )
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

        kb._INITIALIZED_PATHS.clear()
        conn = kb.connect(board="target-board")
        try:
            # Verify we can write to the target board's DB.
            tid = kb.create_task(conn, title="routed-task")
            task = kb.get_task(conn, tid)
            assert task is not None
            assert task.title == "routed-task"
        finally:
            conn.close()

    def test_connect_with_explicit_board_worker_isolation(
        self, tmp_path, monkeypatch
    ):
        """``connect(board=...)`` for worker with mismatched board raises."""
        kanban_home = self._setup_boards(tmp_path)
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(kanban_home))
        monkeypatch.setenv(
            "HERMES_KANBAN_DB", str(kanban_home / "kanban.db")
        )
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")

        kb._INITIALIZED_PATHS.clear()
        with pytest.raises(ValueError, match="board-pinned"):
            kb.connect(board="target-board")
