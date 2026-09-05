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
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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
async def test_persona_emoji_added_on_processing_start(adapter):
    """Persona emoji (configured) is added when processing begins."""
    adapter.config.extra["persona_emoji"] = "🤖"
    # re-init mixin to pick up new persona (or set directly)
    adapter._rxn_persona_emoji = "🤖"
    raw = LedgerMessage()
    event = _make_event("1", raw)
    await adapter.on_processing_start(event)
    # production code called add_reaction with persona, not eyes or check
    assert ("add", "🤖") in raw.ledger()
    # must be first add, not after a remove
    assert raw.ledger()[0] == ("add", "🤖")
    assert "🤖" in raw.effective()


@pytest.mark.asyncio
async def test_persona_fallback_to_eyes_when_not_configured(adapter):
    """When no persona is configured, fallback is the established 👀."""
    # Ensure no persona configured
    adapter.config.extra.pop("persona_emoji", None)
    adapter._rxn_persona_emoji = "👀"
    # also ensure load_config fallback not interfering — set explicitly
    raw = LedgerMessage()
    event = _make_event("1", raw)
    await adapter.on_processing_start(event)
    assert ("add", "👀") in raw.ledger()
    assert raw.ledger()[0] == ("add", "👀")


# ---------------------------------------------------------------------------
# 2. Two distinct tool transitions, each add(new) before remove(previous)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_two_tool_transitions_add_before_remove(adapter):
    adapter.config.extra["persona_emoji"] = "🤖"
    adapter.config.extra["dynamic_reactions"] = True
    adapter.config.extra["reaction_cooldown"] = 0  # disable cooldown for this test
    adapter._rxn_persona_emoji = "🤖"
    adapter._rxn_dynamic = True
    adapter._rxn_cooldown = 0.0

    raw = LedgerMessage()
    event = _make_event("10", raw)
    source = event.source

    await adapter.on_processing_start(event)
    # ledger so far: add persona
    assert raw.ledger() == [("add", "🤖")]

    # Mock get_tool_emoji to return distinct emojis per tool
    emoji_map = {"read_file": "📄", "web_search": "🔍", "terminal": "💻"}

    def fake_get_tool_emoji(name, default="⚙️"):
        return emoji_map.get(name, default)

    with patch("agent.display.get_tool_emoji", side_effect=fake_get_tool_emoji):
        await adapter.on_tool_call_start(source, "read_file")
        # First tool swap: add 📄 before remove 🤖
        ledger = raw.ledger()
        assert ledger[1] == ("add", "📄")
        assert ledger[2] == ("remove", "🤖")
        assert raw.effective() == {"📄"}

        await adapter.on_tool_call_start(source, "web_search")
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
async def test_successful_completion_adds_persona_before_removing_last_tool(adapter):
    adapter.config.extra["persona_emoji"] = "🦊"
    adapter.config.extra["dynamic_reactions"] = True
    adapter.config.extra["reaction_cooldown"] = 0
    adapter._rxn_persona_emoji = "🦊"
    adapter._rxn_dynamic = True
    adapter._rxn_cooldown = 0.0

    raw = LedgerMessage()
    event = _make_event("20", raw)
    source = event.source
    await adapter.on_processing_start(event)

    emoji_map = {"read_file": "📄", "terminal": "💻"}

    def fake_get_tool_emoji(name, default="⚙️"):
        return emoji_map.get(name, default)

    with patch("agent.display.get_tool_emoji", side_effect=fake_get_tool_emoji):
        await adapter.on_tool_call_start(source, "read_file")
        await adapter.on_tool_call_start(source, "terminal")
        # Now active is 💻, ledger has: add 🦊, add 📄/rm 🦊, add 💻/rm 📄
        assert raw.effective() == {"💻"}
        pre_len = len(raw.ledger())
        await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
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
async def test_no_tool_completion_no_duplicate_cleanup(adapter):
    adapter.config.extra["persona_emoji"] = "🤖"
    adapter.config.extra["dynamic_reactions"] = True
    adapter._rxn_persona_emoji = "🤖"
    adapter._rxn_dynamic = True
    adapter._rxn_cooldown = 0.0

    raw = LedgerMessage()
    event = _make_event("30", raw)
    await adapter.on_processing_start(event)
    assert raw.ledger() == [("add", "🤖")]
    assert raw.effective() == {"🤖"}

    # No tool calls between start and complete
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
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
    config.extra = {"persona_emoji": "🤖", "dynamic_reactions": True, "reaction_cooldown": 0}
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
    ad._rxn_cooldown = 0.0
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
    with patch.object(TurnRunner, "_schedule", side_effect=lambda coro, msg, loop=None: asyncio.create_task(coro)):
        tr.progress_callback("tool.started", tool_name="read_file", preview="x", args={})
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
    ad2.config.extra["reaction_cooldown"] = 0
    ad2._rxn_cooldown = 0.0
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
        with patch.object(TurnRunner, "_schedule", side_effect=lambda coro, msg, loop=None: asyncio.create_task(coro)):
            tr2.progress_callback("tool.started", tool_name="read_file", preview="x", args={})
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
async def test_repeated_identical_tool_coalesced(adapter):
    adapter.config.extra["persona_emoji"] = "🤖"
    adapter.config.extra["dynamic_reactions"] = True
    adapter.config.extra["reaction_cooldown"] = 10  # large cooldown
    adapter._rxn_persona_emoji = "🤖"
    adapter._rxn_dynamic = True
    adapter._rxn_cooldown = 10.0

    raw = LedgerMessage()
    event = _make_event("40", raw)
    source = event.source
    await adapter.on_processing_start(event)

    emoji_map = {"read_file": "📄"}

    def fake_emoji(name, default="⚙️"):
        return emoji_map.get(name, default)

    with patch("agent.display.get_tool_emoji", side_effect=fake_emoji):
        await adapter.on_tool_call_start(source, "read_file")
        assert ("add", "📄") in raw.ledger()
        ledger_len_after_first = len(raw.ledger())
        # Repeated identical tool — should be coalesced (no new add/remove)
        await adapter.on_tool_call_start(source, "read_file")
        assert len(raw.ledger()) == ledger_len_after_first  # no change


@pytest.mark.asyncio
async def test_rapid_different_tool_coalesced_by_cooldown(adapter):
    adapter.config.extra["persona_emoji"] = "🤖"
    adapter.config.extra["dynamic_reactions"] = True
    adapter.config.extra["reaction_cooldown"] = 10  # large
    adapter._rxn_persona_emoji = "🤖"
    adapter._rxn_dynamic = True
    adapter._rxn_cooldown = 10.0

    raw = LedgerMessage()
    event = _make_event("41", raw)
    source = event.source
    await adapter.on_processing_start(event)

    emoji_map = {"read_file": "📄", "web_search": "🔍"}

    def fake_emoji(name, default="⚙️"):
        return emoji_map.get(name, default)

    with patch("agent.display.get_tool_emoji", side_effect=fake_emoji):
        await adapter.on_tool_call_start(source, "read_file")
        ledger_len = len(raw.ledger())
        # Rapid different tool within cooldown — should be coalesced (no swap)
        await adapter.on_tool_call_start(source, "web_search")
        assert len(raw.ledger()) == ledger_len  # coalesced, still 📄
        assert raw.effective() == {"📄"}

        # Advance time beyond cooldown and try again — should now allow
        with patch("time.monotonic", return_value=time.monotonic() + 20):
            await adapter.on_tool_call_start(source, "web_search")
        assert ("add", "🔍") in raw.ledger()
        assert raw.effective() == {"🔍"}


# ---------------------------------------------------------------------------
# 7. Disabled / failure / cancel behavior remains intact
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reactions_disabled_via_env_no_ops(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_REACTIONS", "false")
    # Force re-evaluate enabled flag (mixin caches dynamic flag but start checks enabled each time)
    adapter._rxn_dynamic = False  # not needed but ensure
    raw = LedgerMessage()
    event = _make_event("50", raw)
    await adapter.on_processing_start(event)
    # No reactions when disabled
    assert raw.ledger() == []
    await adapter.on_tool_call_start(event.source, "read_file")
    assert raw.ledger() == []
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    assert raw.ledger() == []
    # Response delivery still works would be tested via _process_message_background but here we ensure no crash


@pytest.mark.asyncio
async def test_reactions_disabled_via_config_no_ops(adapter):
    adapter.config.extra["reactions"] = False
    # Need to re-resolve: easiest set directly
    # _reactions_enabled will read config.extra
    raw = LedgerMessage()
    event = _make_event("51", raw)
    await adapter.on_processing_start(event)
    assert raw.ledger() == []


@pytest.mark.asyncio
async def test_failure_outcome_adds_cross_before_removing(adapter):
    adapter.config.extra["persona_emoji"] = "🤖"
    adapter.config.extra["dynamic_reactions"] = True
    adapter.config.extra["reaction_cooldown"] = 0
    adapter._rxn_persona_emoji = "🤖"
    adapter._rxn_dynamic = True
    adapter._rxn_cooldown = 0.0
    raw = LedgerMessage()
    event = _make_event("60", raw)
    source = event.source
    await adapter.on_processing_start(event)
    emoji_map = {"read_file": "📄"}

    def fake_emoji(name, default="⚙️"):
        return emoji_map.get(name, default)

    with patch("agent.display.get_tool_emoji", side_effect=fake_emoji):
        await adapter.on_tool_call_start(source, "read_file")
        assert raw.effective() == {"📄"}
        pre_len = len(raw.ledger())
        await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)
        ledger = raw.ledger()
        assert ledger[pre_len] == ("add", "❌")
        assert ledger[pre_len + 1] == ("remove", "📄")
        assert raw.effective() == {"❌"}


@pytest.mark.asyncio
async def test_cancelled_outcome_removes_without_adding(adapter):
    adapter.config.extra["persona_emoji"] = "🤖"
    adapter.config.extra["dynamic_reactions"] = True
    adapter.config.extra["reaction_cooldown"] = 0
    adapter._rxn_persona_emoji = "🤖"
    adapter._rxn_dynamic = True
    adapter._rxn_cooldown = 0.0
    raw = LedgerMessage()
    event = _make_event("61", raw)
    source = event.source
    await adapter.on_processing_start(event)
    emoji_map = {"read_file": "📄"}

    def fake_emoji(name, default="⚙️"):
        return emoji_map.get(name, default)

    with patch("agent.display.get_tool_emoji", side_effect=fake_emoji):
        await adapter.on_tool_call_start(source, "read_file")
        assert raw.effective() == {"📄"}
        pre_len = len(raw.ledger())
        await adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED)
        ledger = raw.ledger()
        # Cancelled should remove current without adding persona or cross
        assert ledger[pre_len] == ("remove", "📄")
        assert len(ledger) == pre_len + 1
        assert raw.effective() == set()


# ---------------------------------------------------------------------------
# 8. Rate-limit / API failures leave tracked state recoverable and do not raise
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rate_limit_failure_recoverable_and_no_exception(adapter):
    adapter.config.extra["persona_emoji"] = "🤖"
    adapter.config.extra["dynamic_reactions"] = True
    adapter.config.extra["reaction_cooldown"] = 0
    adapter._rxn_persona_emoji = "🤖"
    adapter._rxn_dynamic = True
    adapter._rxn_cooldown = 0.0

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
    await adapter.on_processing_start(event)
    # start succeeded (first call n=1)
    assert call_count["n"] == 1

    emoji_map = {"read_file": "📄", "web_search": "🔍"}

    def fake_emoji(name, default="⚙️"):
        return emoji_map.get(name, default)

    with patch("agent.display.get_tool_emoji", side_effect=fake_emoji):
        # This tool swap's add will raise — should not propagate exception
        await adapter.on_tool_call_start(source, "read_file")
        # No exception, and state should be recoverable (still persona)
        # Because our mixin returns early without updating active on failure, ledger should not have second add
        # Effective still persona
        assert raw.effective() == {"🤖"}
        # Next tool swap should succeed
        await adapter.on_tool_call_start(source, "web_search")
        assert ("add", "🔍") in [("add", e) for e in [x[1] for x in raw.ledger() if x[0]=="add"]]
        assert raw.effective() == {"🔍"}
        # Completion should still work
        await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
        assert "🤖" in raw.effective()
        assert "🔍" not in raw.effective()


@pytest.mark.asyncio
async def test_process_message_background_still_delivers_when_reactions_disabled(adapter, monkeypatch):
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
async def test_process_message_background_adds_and_swaps_reactions_legacy(adapter):
    """Legacy: processing start adds persona (default 👀) and success restores persona (not ✅)."""
    raw = LedgerMessage()
    # Ensure no persona override, so default 👀
    adapter.config.extra.pop("persona_emoji", None)
    adapter._rxn_persona_emoji = "👀"
    adapter._rxn_cooldown = 0.0
    adapter._rxn_dynamic = True

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
# Confirmed-start parity regression — provider outcome must gate state and ACK
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_on_processing_start_provider_false_leaves_no_state_and_ack_false(adapter):
    """Provider False must not commit active/msg_refs and must record emoji_ack=False."""
    adapter.config.extra["persona_emoji"] = "🤖"
    adapter._rxn_persona_emoji = "🤖"
    raw = LedgerMessage(msg_id=9001)
    event = _make_event("9001", raw)
    # deterministic fake at provider boundary only
    adapter._reaction_add = AsyncMock(return_value=False)
    ack_record = MagicMock()
    adapter._record_discord_processing_start = ack_record

    await adapter.on_processing_start(event)

    key = adapter._reaction_msg_key(event)
    assert raw.ledger() == []
    assert key not in adapter._rxn_active
    assert key not in adapter._rxn_msg_refs
    assert ack_record.call_count == 1
    assert ack_record.call_args.kwargs["emoji_ack"] is False


@pytest.mark.asyncio
async def test_on_processing_start_provider_exception_leaves_no_state_and_ack_false(adapter):
    """Provider exception must not commit state and must record emoji_ack=False."""
    adapter.config.extra["persona_emoji"] = "🤖"
    adapter._rxn_persona_emoji = "🤖"
    raw = LedgerMessage(msg_id=9002)
    event = _make_event("9002", raw)
    adapter._reaction_add = AsyncMock(side_effect=RuntimeError("boom"))
    ack_record = MagicMock()
    adapter._record_discord_processing_start = ack_record

    await adapter.on_processing_start(event)

    key = adapter._reaction_msg_key(event)
    assert raw.ledger() == []
    assert key not in adapter._rxn_active
    assert key not in adapter._rxn_msg_refs
    assert ack_record.call_count == 1
    assert ack_record.call_args.kwargs["emoji_ack"] is False


@pytest.mark.asyncio
async def test_on_processing_start_missing_capability_leaves_no_state_and_ack_false(adapter, monkeypatch):
    """Missing capability (no add_reaction) / disabled must not commit state and ack False."""
    adapter.config.extra["persona_emoji"] = "🤖"
    adapter._rxn_persona_emoji = "🤖"
    # use a raw message without add_reaction to simulate missing capability
    raw_no_cap = SimpleNamespace(id=9003)
    event = _make_event("9003", raw_no_cap)
    ack_record = MagicMock()
    adapter._record_discord_processing_start = ack_record

    await adapter.on_processing_start(event)

    key = adapter._reaction_msg_key(event)
    # no provider add should have been attempted via ledger, active/msg_refs empty
    assert key not in adapter._rxn_active
    assert key not in adapter._rxn_msg_refs
    assert ack_record.call_count == 1
    assert ack_record.call_args.kwargs["emoji_ack"] is False

    # also verify disabled path produces same false ack via env
    monkeypatch.setenv("DISCORD_REACTIONS", "false")
    raw2 = LedgerMessage(msg_id=9004)
    event2 = _make_event("9004", raw2)
    ack2 = MagicMock()
    adapter._record_discord_processing_start = ack2
    await adapter.on_processing_start(event2)
    key2 = adapter._reaction_msg_key(event2)
    assert raw2.ledger() == []
    assert key2 not in adapter._rxn_active
    assert key2 not in adapter._rxn_msg_refs
    assert ack2.call_count == 1
    assert ack2.call_args.kwargs["emoji_ack"] is False


@pytest.mark.asyncio
async def test_on_processing_start_confirmed_add_records_ack_true(adapter):
    """Confirmed provider success commits state, populates ledger, and records emoji_ack=True."""
    adapter.config.extra["persona_emoji"] = "🤖"
    adapter._rxn_persona_emoji = "🤖"
    raw = LedgerMessage(msg_id=9005)
    event = _make_event("9005", raw)
    ack_record = MagicMock()
    adapter._record_discord_processing_start = ack_record

    await adapter.on_processing_start(event)

    key = adapter._reaction_msg_key(event)
    assert raw.ledger() == [("add", "🤖")]
    assert raw.effective() == {"🤖"}
    assert adapter._rxn_active.get(key) == "🤖"
    assert adapter._rxn_msg_refs.get(key) is raw
    assert ack_record.call_count == 1
    assert ack_record.call_args.kwargs["emoji_ack"] is True
