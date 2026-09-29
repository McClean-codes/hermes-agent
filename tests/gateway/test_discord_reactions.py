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
            chat_type="group",
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
    # Runner seam shape: SessionSource + the turn's message identity (turn_identity).
    await adapter.on_tool_call_start(event.source, "terminal", turn_identity="1")
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
        # The runner binds tool hooks to this turn's message so reaction state keys
        # per message (identity flows through the real scheduling seam below).
        turn_identity="2",
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
    await adapter.on_tool_call_start(event.source, "terminal", turn_identity="4")
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


@pytest.mark.asyncio
async def test_concurrent_same_channel_users_keep_isolated_reaction_state(adapter):
    """SEC-1 regression: two users processed concurrently in ONE group channel.

    Every add/remove must target only its own message, and one turn's completion
    must not overwrite or clear the other turn's reaction/cache/lock state.
    """
    raw_a = SimpleNamespace(add_reaction=AsyncMock(), remove_reaction=AsyncMock())
    raw_b = SimpleNamespace(add_reaction=AsyncMock(), remove_reaction=AsyncMock())
    event_a = _make_event("100", raw_a)
    event_b = _make_event("200", raw_b)
    # Same chat, same thread — only the author and message id differ.
    assert (event_a.source.chat_id, event_a.source.thread_id) == (event_b.source.chat_id, event_b.source.thread_id)
    key_a, key_b = adapter._reaction_msg_key(event_a), adapter._reaction_msg_key(event_b)
    assert key_a != key_b

    # Both turns are in flight concurrently in the shared channel.
    await asyncio.gather(
        adapter.on_processing_start(event_a),
        adapter.on_processing_start(event_b),
    )
    assert [c.args[0] for c in raw_a.add_reaction.await_args_list] == ["🧪"]
    assert [c.args[0] for c in raw_b.add_reaction.await_args_list] == ["🧪"]
    assert key_a in adapter._rxn_active and key_b in adapter._rxn_active

    # Tool hooks arrive through the runner seam: SessionSource + turn identity.
    await asyncio.gather(
        adapter.on_tool_call_start(event_a.source, "terminal", turn_identity="100"),
        adapter.on_tool_call_start(event_b.source, "read_file", turn_identity="200"),
    )
    # Each tool swap removed only ITS OWN message's persona ack.
    assert [c.args[0] for c in raw_a.remove_reaction.await_args_list] == ["🧪"]
    assert [c.args[0] for c in raw_b.remove_reaction.await_args_list] == ["🧪"]
    assert raw_a.add_reaction.await_count == 2  # persona + A's tool emoji
    assert raw_b.add_reaction.await_count == 2  # persona + B's tool emoji

    # A completes while B is still running.
    await adapter.on_processing_complete(event_a, ProcessingOutcome.SUCCESS)
    adds_a = [c.args[0] for c in raw_a.add_reaction.await_args_list]
    removes_a = [c.args[0] for c in raw_a.remove_reaction.await_args_list]
    assert adds_a[-1] == "🧪" and len(adds_a) == 3       # final persona on A's message
    assert key_a not in adapter._rxn_active and key_a not in adapter._rxn_msg_refs
    # B's in-flight state and message survived A's completion untouched.
    assert key_b in adapter._rxn_active and key_b in adapter._rxn_msg_refs
    assert raw_b.add_reaction.await_count == 2 and raw_b.remove_reaction.await_count == 1
    assert removes_a == ["🧪", adds_a[1]]                # A's own tool emoji removed from A only

    # B finishes normally afterwards.
    await adapter.on_processing_complete(event_b, ProcessingOutcome.SUCCESS)
    adds_b = [c.args[0] for c in raw_b.add_reaction.await_args_list]
    assert adds_b[-1] == "🧪" and len(adds_b) == 3
    assert not adapter._rxn_active and not adapter._rxn_msg_refs
    assert not adapter._rxn_locks and not adapter._rxn_residuals
    assert not adapter._session_raw_messages


@pytest.mark.asyncio
async def test_failed_initial_add_is_not_marked_active_and_completion_retries(adapter):
    """Raven P2 regression: a failed persona ack is never marked active, and a
    successful completion retries the final reaction against the same message."""
    raw_message = SimpleNamespace(add_reaction=AsyncMock(), remove_reaction=AsyncMock())
    event = _make_event("6", raw_message)
    key = adapter._reaction_msg_key(event)
    # First add (the ack) fails transiently; the completion retry would succeed.
    raw_message.add_reaction.side_effect = [RuntimeError("429"), None]

    await adapter.on_processing_start(event)
    assert raw_message.add_reaction.await_count == 1
    assert key not in adapter._rxn_active          # failed add NOT marked active
    assert key in adapter._rxn_msg_refs            # message association retained for retry

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    adds = [c.args[0] for c in raw_message.add_reaction.await_args_list]
    assert adds == ["🧪", "🧪"]                    # completion retried and landed
    assert not adapter._rxn_active and not adapter._rxn_msg_refs
    assert not adapter._rxn_locks


@pytest.mark.asyncio
async def test_failed_remove_is_tracked_as_residual_and_reconciled_at_completion(adapter):
    """SEC-4 regression: a false remove does not advance state as if the old emoji
    were gone; completion reconciles the residual and releases all per-turn state."""
    raw_message = SimpleNamespace(add_reaction=AsyncMock(), remove_reaction=AsyncMock())
    event = _make_event("7", raw_message)
    key = adapter._reaction_msg_key(event)

    await adapter.on_processing_start(event)                       # persona 🧪 lands
    # Tool swap: the new emoji lands, but removing the persona fails transiently.
    raw_message.remove_reaction.side_effect = [RuntimeError("429"), None, None]
    await adapter.on_tool_call_start(event.source, "terminal", turn_identity="7")

    tool_emoji = [c.args[0] for c in raw_message.add_reaction.await_args_list][-1]
    assert tool_emoji != "🧪"
    assert adapter._rxn_active[key] == tool_emoji                  # new emoji is active
    assert "🧪" in adapter._rxn_residuals[key]                     # old one NOT forgotten

    # FAILURE: final ❌ differs from both — completion must remove the active emoji AND
    # reconcile the residual persona.
    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)
    removes = [c.args[0] for c in raw_message.remove_reaction.await_args_list]
    assert removes[0] == "🧪"                                      # failed swap attempt
    assert set(removes[1:]) == {"🧪", tool_emoji}                  # completion reconciled both
    assert not adapter._rxn_active and not adapter._rxn_msg_refs
    assert not adapter._rxn_locks and not adapter._rxn_residuals
    assert adapter._rxn_residual_leaks == {}                       # successful reconciliation


@pytest.mark.asyncio
async def test_failed_final_add_still_releases_locks_and_state(adapter):
    """SEC-4 regression: a failed final add must not leak the per-turn lock entry
    (and must not strip the still-visible prior reaction)."""
    raw_message = SimpleNamespace(add_reaction=AsyncMock(), remove_reaction=AsyncMock())
    event = _make_event("8", raw_message)
    key = adapter._reaction_msg_key(event)

    await adapter.on_processing_start(event)                       # ack lands
    raw_message.add_reaction.side_effect = RuntimeError("500")     # completion add fails

    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)

    assert not adapter._rxn_locks and key not in adapter._rxn_locks
    assert not adapter._rxn_active and not adapter._rxn_msg_refs
    assert not adapter._rxn_last_swap and not adapter._rxn_residuals
    # The final add failed, so nothing was removed: visible state stays intact.
    assert raw_message.remove_reaction.await_count == 0
