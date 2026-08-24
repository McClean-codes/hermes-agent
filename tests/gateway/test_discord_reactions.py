"""Tests for Discord message reactions tied to processing lifecycle hooks."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome, SendResult
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
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
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

from plugins.platforms.discord.adapter import DiscordAdapter


class FakeTree:
    def __init__(self):
        self.commands = {}
    def command(self, *, name, description):
        def decorator(fn):
            self.commands[name] = fn
            return fn
        return decorator

@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="***")
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    return adapter

def _make_event(message_id, raw_message):
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="123",
            chat_type="dm",
            user_id="42",
            user_name="Jezza",
        ),
        raw_message=raw_message,
        message_id=message_id,
    )


@pytest.mark.asyncio
async def test_persona_emoji_added_on_processing_start(adapter):
    """Persona emoji (default '👀') is added when processing begins."""
    raw_message = SimpleNamespace(
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )

    async def handler(_event):
        await asyncio.sleep(0)
        return "ack"

    async def hold_typing(_chat_id, interval=2.0, metadata=None):
        await asyncio.Event().wait()

    adapter.set_message_handler(handler)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="999"))
    adapter._keep_typing = hold_typing

    event = _make_event("1", raw_message)
    await adapter._process_message_background(event, build_session_key(event.source))

    # Persona emoji added on start
    assert raw_message.add_reaction.await_args_list[0].args == ("👀",)


@pytest.mark.asyncio
async def test_reactions_disabled_via_env(adapter, monkeypatch):
    """When DISCORD_REACTIONS=false, no reactions should be added."""
    monkeypatch.setenv("DISCORD_REACTIONS", "false")

    raw_message = SimpleNamespace(
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )

    async def handler(_event):
        await asyncio.sleep(0)
        return "ack"

    async def hold_typing(_chat_id, interval=2.0, metadata=None):
        await asyncio.Event().wait()

    adapter.set_message_handler(handler)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="999"))
    adapter._keep_typing = hold_typing

    event = _make_event("2", raw_message)
    await adapter._process_message_background(event, build_session_key(event.source))

    raw_message.add_reaction.assert_not_called()
    raw_message.remove_reaction.assert_not_called()


# ---------------------------------------------------------------------------
# Regression tests: tool transitions, final cleanup, and edge cases
# ---------------------------------------------------------------------------


def _make_tool_source():
    """SessionSource matching the adapter fixture's chat_id."""
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="123",
        chat_type="dm",
        user_id="42",
        user_name="Jezza",
    )


@pytest.mark.asyncio
async def test_tool_transition_add_before_remove(adapter):
    """Tool calls store state under the same message-ID key as start/complete.
    Before the fix, tool calls stored under a session key, causing the final
    cleanup to miss the tool state entirely."""
    adapter._rxn_dynamic = True
    adapter._rxn_cooldown = 0

    raw = SimpleNamespace(
        id="msg1",
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )
    event = _make_event("msg1", raw)

    # processing start — must go through adapter.on_processing_start
    # (populates _session_raw_messages for on_tool_call_start to find)
    await adapter.on_processing_start(event)
    persona_emoji = adapter._rxn_persona_emoji
    assert raw.add_reaction.await_args_list[0].args == (persona_emoji,)

    # tool call via adapter.on_tool_call_start (SessionSource, not event)
    source = _make_tool_source()
    await adapter.on_tool_call_start(source, "tool_a")

    # tool emoji added before persona removed
    tool_add = raw.add_reaction.await_args_list[-1]
    persona_removes = [
        c for c in raw.remove_reaction.await_args_list if c.args[0] == persona_emoji
    ]
    assert len(persona_removes) == 1
    assert tool_add.called_before(persona_removes[0])

    # State is tracked under the message-ID key (not session key)
    assert "msg1" in adapter._rxn_active, (
        "tool state must be under message-ID key, not session key"
    )

    # Now complete — persona add should come before tool remove
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    persona_adds = [
        c for c in raw.add_reaction.await_args_list if c.args[0] == persona_emoji
    ]
    tool_removes = [
        c for c in raw.remove_reaction.await_args_list
        if c.args[0] == tool_add.args[0]
    ]
    assert len(tool_removes) == 1, "tool emoji should be removed on completion"
    assert persona_adds[-1].called_before(tool_removes[0])
    assert adapter._rxn_active == {}


@pytest.mark.asyncio
async def test_completion_after_tool_removes_tool_emoji(adapter):
    """Successful completion after a tool call produces add(persona) before
    remove(last_tool) and leaves only the persona emoji effective."""
    adapter._rxn_dynamic = True
    adapter._rxn_cooldown = 0

    raw = SimpleNamespace(
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )
    event = _make_event("msg2", raw)

    # processing start
    await adapter._rxn_on_processing_start(event)

    # tool call
    source = _make_tool_source()
    await adapter.on_tool_call_start(source, "read_file")

    # get the tool emoji that was added
    tool_emoji = raw.add_reaction.await_args_list[-1].args[0]

    # successful completion
    await adapter._rxn_on_processing_complete(event, ProcessingOutcome.SUCCESS)

    # find persona add and tool remove
    persona_adds = [
        c for c in raw.add_reaction.await_args_list if c.args[0] == "👀"
    ]
    tool_removes = [
        c for c in raw.remove_reaction.await_args_list if c.args[0] == tool_emoji
    ]
    assert len(persona_adds) >= 2  # start + completion
    assert len(tool_removes) == 1
    # last persona add must come before tool remove
    assert persona_adds[-1].called_before(tool_removes[0])

    # effective final state: only persona emoji
    assert adapter._rxn_active == {}


@pytest.mark.asyncio
async def test_completion_no_tool_call_no_duplicate(adapter):
    """Completion without any tool call succeeds cleanly — no leftover state.
    The persona emoji is already active so no redundant add/remove happens."""
    raw = SimpleNamespace(
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )
    event = _make_event("msg3", raw)

    await adapter._rxn_on_processing_start(event)
    await adapter._rxn_on_processing_complete(event, ProcessingOutcome.SUCCESS)

    # Only persona was added once at start — completion sees it already active
    # and correctly skips a redundant add/remove (the "!= current" guard).
    persona_adds = [
        c for c in raw.add_reaction.await_args_list if c.args[0] == "👀"
    ]
    assert len(persona_adds) == 1
    # No removals (no tool emoji existed)
    raw.remove_reaction.assert_not_called()
    # Active state is cleaned up
    assert adapter._rxn_active == {}


@pytest.mark.asyncio
async def test_tool_progress_disabled_still_fires_hook(adapter):
    """on_tool_call_start fires even when progress messages are off,
    because the reaction hook runs before the progress_queue guard."""
    adapter._rxn_dynamic = True
    adapter._rxn_cooldown = 0

    raw = SimpleNamespace(
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )
    event = _make_event("msg4", raw)

    await adapter._rxn_on_processing_start(event)
    # The hook should succeed regardless of progress queue state
    source = _make_tool_source()
    await adapter.on_tool_call_start(source, "terminal")

    tool_adds = [
        c for c in raw.add_reaction.await_args_list if c.args[0] != "👀"
    ]
    assert len(tool_adds) == 1, "tool emoji was added despite no progress queue"


@pytest.mark.asyncio
async def test_disabled_reactions_no_changes(adapter, monkeypatch):
    """Reactions disabled via env: no add or remove at any lifecycle stage."""
    monkeypatch.setenv("DISCORD_REACTIONS", "false")

    raw = SimpleNamespace(
        add_reaction=AsyncMock(),
        remove_reaction=AsyncMock(),
    )
    event = _make_event("msg5", raw)

    await adapter._rxn_on_processing_start(event)
    source = _make_tool_source()
    await adapter._rxn_on_tool_call_start(source, "terminal")
    await adapter._rxn_on_processing_complete(event, ProcessingOutcome.SUCCESS)

    raw.add_reaction.assert_not_called()
    raw.remove_reaction.assert_not_called()
