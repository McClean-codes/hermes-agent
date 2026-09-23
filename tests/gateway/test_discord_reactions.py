"""Discord persona and dynamic reaction lifecycle tests."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
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

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


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
    config = PlatformConfig(
        enabled=True,
        extra={"persona_emoji": "🧪", "dynamic_reactions": True, "reaction_cooldown": 0},
    )
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(
        tree=FakeTree(),
        get_channel=lambda _id: None,
        fetch_channel=AsyncMock(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    return adapter


def test_global_persona_config_is_used_when_platform_extra_is_unset(monkeypatch):
    from hermes_cli import config as config_module

    monkeypatch.setattr(
        config_module, "load_config",
        lambda: {"persona_emoji": "🌟", "dynamic_reactions": False},
    )
    candidate = DiscordAdapter(PlatformConfig(enabled=True, extra={}))

    assert candidate._rxn_persona_emoji == "🌟"
    assert candidate._rxn_dynamic is False


def _make_event(message_id: str, raw_message) -> MessageEvent:
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
async def test_processing_lifecycle_uses_persona_and_tool_reactions(adapter):
    raw_message = SimpleNamespace(add_reaction=AsyncMock(), remove_reaction=AsyncMock())
    event = _make_event("1", raw_message)

    await adapter.on_processing_start(event)
    await adapter.on_tool_call_start(event.source, "terminal")
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    added = [call.args[0] for call in raw_message.add_reaction.await_args_list]
    removed = [call.args[0] for call in raw_message.remove_reaction.await_args_list]
    assert added[0] == "🧪"
    assert len(added) == 3
    assert added[1] != "🧪"
    assert added[2] == "🧪"
    assert removed == ["🧪", added[1]]
    assert not adapter._rxn_active
    assert not adapter._rxn_msg_refs
    assert not adapter._session_raw_messages


@pytest.mark.asyncio
async def test_dynamic_reactions_are_independent_of_visible_progress(adapter):
    """The runner hook remains active when no progress queue is allocated."""
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    raw_message = SimpleNamespace(add_reaction=AsyncMock(), remove_reaction=AsyncMock())
    event = _make_event("2", raw_message)
    await adapter.on_processing_start(event)

    ctx = TurnContext(
        source=event.source,
        _run_still_current=lambda: True,
        progress_mode="off",
        tool_progress_enabled=False,
        progress_queue=None,
        _status_adapter=adapter,
        _loop_for_step=asyncio.get_running_loop(),
    )
    TurnRunner(SimpleNamespace(), ctx).progress_callback("tool.started", "terminal", "ls", {})
    await asyncio.sleep(0.05)

    added = [call.args[0] for call in raw_message.add_reaction.await_args_list]
    assert len(added) == 2
    assert added[0] == "🧪"
    assert added[1] != "🧪"


@pytest.mark.asyncio
async def test_failure_reaction_is_not_success_persona(adapter):
    raw_message = SimpleNamespace(add_reaction=AsyncMock(), remove_reaction=AsyncMock())
    event = _make_event("3", raw_message)

    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)

    added = [call.args[0] for call in raw_message.add_reaction.await_args_list]
    removed = [call.args[0] for call in raw_message.remove_reaction.await_args_list]
    assert added == ["🧪", "❌"]
    assert removed == ["🧪"]


@pytest.mark.asyncio
async def test_cancelled_lifecycle_cleans_up_without_terminal_reaction(adapter):
    raw_message = SimpleNamespace(add_reaction=AsyncMock(), remove_reaction=AsyncMock())
    event = _make_event("4", raw_message)

    await adapter.on_processing_start(event)
    await adapter.on_tool_call_start(event.source, "terminal")
    await adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED)

    added = [call.args[0] for call in raw_message.add_reaction.await_args_list]
    removed = [call.args[0] for call in raw_message.remove_reaction.await_args_list]
    assert len(added) == 2
    assert removed == [added[0], added[1]]
    assert not adapter._rxn_active
    assert not adapter._rxn_msg_refs
    assert not adapter._session_raw_messages


@pytest.mark.asyncio
async def test_reactions_disabled_still_delivers_response(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REACTIONS", "false")
    raw_message = SimpleNamespace(add_reaction=AsyncMock(), remove_reaction=AsyncMock())

    async def handler(_event):
        await asyncio.sleep(0)
        return "ack"

    async def hold_typing(_chat_id, interval=2.0, metadata=None):
        await asyncio.Event().wait()

    adapter.set_message_handler(handler)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="999"))
    adapter._keep_typing = hold_typing
    event = _make_event("5", raw_message)

    await adapter._process_message_background(event, build_session_key(event.source))

    raw_message.add_reaction.assert_not_awaited()
    raw_message.remove_reaction.assert_not_awaited()
    adapter.send.assert_awaited_once()
