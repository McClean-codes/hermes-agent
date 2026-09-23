"""Production-seam tests for per-tool gateway progress filtering."""

from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.display_config import _norm_tool_progress_filter, resolve_tool_progress_filter
from gateway.turn_context import TurnContext
from gateway.run_turn_runner import _resolve_effective_mode, TurnRunner


def _make_ctx(
    *,
    progress_mode: str = "all",
    tool_progress_filter: dict | None = None,
    tool_progress_enabled: bool | None = None,
    with_queue: bool = True,
) -> TurnContext:
    if tool_progress_enabled is None:
        tool_progress_enabled = progress_mode not in {"off", "log"}
    return TurnContext(
        source=SimpleNamespace(chat_id="test-chat"),
        _run_still_current=lambda: True,
        progress_mode=progress_mode,
        tool_progress_enabled=tool_progress_enabled,
        tool_progress_filter=tool_progress_filter,
        progress_queue=queue.Queue() if with_queue else None,
    )


def _make_runner(ctx: TurnContext) -> TurnRunner:
    class StubRunner:
        def _adapter_for_source(self, _source):
            adapter = MagicMock()
            adapter.supports_code_blocks = False
            adapter.format_tool_preview = lambda value, **_kwargs: getattr(value, "text", str(value))
            return adapter

        async def _deliver_platform_notice(self, _source, _content):
            return None

    return TurnRunner(StubRunner(), ctx)  # type: ignore[arg-type]


def _drain(progress_queue: queue.Queue) -> list:
    values = []
    while not progress_queue.empty():
        values.append(progress_queue.get_nowait())
    return values


class TestFilterResolution:
    def test_global_and_platform_entries_merge_with_platform_precedence(self):
        config = {
            "display": {
                "tool_progress_filter": {"terminal": "off", "read_file": "off"},
                "platforms": {"telegram": {"tool_progress_filter": {"terminal": "all"}}},
            }
        }

        assert resolve_tool_progress_filter(config, "telegram") == {
            "terminal": "all", "read_file": "off",
        }
        assert resolve_tool_progress_filter(config, "discord") == {
            "terminal": "off", "read_file": "off",
        }

    def test_list_and_boolean_forms_normalize_without_invalid_entries(self):
        assert _norm_tool_progress_filter(["terminal", " skill_view ", ""]) == {
            "terminal": "all", "skill_view": "all",
        }
        assert _norm_tool_progress_filter({"terminal": True, "read_file": False}) == {
            "terminal": "all", "read_file": "off",
        }
        assert _norm_tool_progress_filter(
            {"terminal": "all", "": "off", "read_file": "unknown", "other": None, 4: "off"}
        ) == {"terminal": "all"}

    def test_malformed_and_empty_platform_values_fail_safe_to_global(self):
        config = {
            "display": {
                "tool_progress_filter": {"terminal": "off"},
                "platforms": {"telegram": {"tool_progress_filter": "bad"}},
            }
        }
        assert resolve_tool_progress_filter(config, "telegram") == {"terminal": "off"}
        assert resolve_tool_progress_filter({"display": "bad"}, "telegram") == {}
        assert resolve_tool_progress_filter({}, "telegram") == {}


class TestEffectiveMode:
    def test_exact_tool_precedes_category_and_global(self):
        filt = {"skills": "off", "skill_view": "verbose"}
        assert _resolve_effective_mode("skill_view", "all", filt) == "verbose"
        assert _resolve_effective_mode("skill_manage", "all", filt) == "off"
        assert _resolve_effective_mode("terminal", "all", filt) == "all"

    def test_category_aliases_and_runtime_mcp_metadata_are_resolved(self, monkeypatch):
        import tools.mcp_tool as mcp_tool

        monkeypatch.setitem(mcp_tool._mcp_tool_server_names, "_test_progress_mcp", "server")
        assert _resolve_effective_mode("skill_view", "off", {"skill": "all"}) == "all"
        assert _resolve_effective_mode("_test_progress_mcp", "all", {"mcp_tools": "off"}) == "off"
        assert _resolve_effective_mode("ordinary_tool", "off", {"mcp": "all"}) == "off"

    def test_filter_never_changes_global_mode_when_no_entry_matches(self):
        assert _resolve_effective_mode("read_file", "off", {"unknown_category": "all"}) == "off"
        assert _resolve_effective_mode("read_file", "verbose", None) == "verbose"


class TestTurnRunnerProgressSeam:
    def test_exact_tool_off_suppresses_only_that_tool(self):
        ctx = _make_ctx(tool_progress_filter={"terminal": "off"})
        runner = _make_runner(ctx)

        runner.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
        runner.progress_callback("tool.started", "read_file", "README", {"path": "README.md"})

        messages = _drain(ctx.progress_queue)
        assert len(messages) == 1
        assert "read_file" in str(messages[0]) or "README" in str(messages[0])

    def test_global_off_with_whitelisted_tool_stays_on_at_callback_seam(self):
        ctx = _make_ctx(
            progress_mode="off",
            tool_progress_enabled=True,
            tool_progress_filter={"skill_view": "all"},
        )
        runner = _make_runner(ctx)

        runner.progress_callback("tool.started", "skill_view", "view", {})
        runner.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})

        messages = _drain(ctx.progress_queue)
        assert len(messages) == 1
        assert "skill_view" in str(messages[0]) or "view" in str(messages[0])

    def test_effective_verbose_and_new_modes_are_applied(self):
        verbose_ctx = _make_ctx(tool_progress_filter={"terminal": "verbose"})
        _make_runner(verbose_ctx).progress_callback(
            "tool.started", "terminal", "long", {"command": "echo " + "x" * 100},
        )
        verbose_messages = _drain(verbose_ctx.progress_queue)
        assert len(verbose_messages) == 1
        assert "echo " + "x" * 100 in str(verbose_messages[0])

        new_ctx = _make_ctx(tool_progress_filter={"terminal": "new"})
        new_runner = _make_runner(new_ctx)
        new_runner.progress_callback("tool.started", "terminal", "first", {})
        _drain(new_ctx.progress_queue)
        new_runner.progress_callback("tool.started", "terminal", "second", {})
        assert _drain(new_ctx.progress_queue) == []
        new_runner.progress_callback("tool.started", "read_file", "third", {})
        assert len(_drain(new_ctx.progress_queue)) == 1

    def test_thinking_and_log_rails_remain_independent_of_filter(self):
        ctx = _make_ctx(
            progress_mode="off",
            tool_progress_enabled=False,
            tool_progress_filter={"terminal": "off"},
        )
        ctx._thinking_enabled = True
        runner = _make_runner(ctx)

        runner.progress_callback("_thinking", "_thinking", "still working", {})
        messages = _drain(ctx.progress_queue)
        assert messages == ["💬 still working"]

    @pytest.mark.asyncio
    async def test_native_stream_consumer_receives_allowed_progress(self):
        consumer = SimpleNamespace(accepts_tool_progress=True, on_tool_progress=MagicMock())
        ctx = _make_ctx(tool_progress_filter={"terminal": "all"})
        ctx.stream_consumer_holder[0] = consumer

        _make_runner(ctx).progress_callback("tool.started", "terminal", "ls", {"command": "ls"})

        consumer.on_tool_progress.assert_called_once()
        assert "Running ls" in consumer.on_tool_progress.call_args.args[0]

    def test_filtered_progress_does_not_mutate_execution_or_result_state(self):
        ctx = _make_ctx(tool_progress_filter={"terminal": "off"})
        ctx.result_holder[0] = {"final_response": "Error: connection refused"}
        ctx.tools_holder[0] = [{"role": "tool", "content": "result"}]
        runner = _make_runner(ctx)

        runner.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})

        assert ctx.result_holder[0]["final_response"] == "Error: connection refused"
        assert ctx.tools_holder[0][0]["content"] == "result"
        assert ctx.progress_mode == "all"
        assert ctx.tool_progress_enabled is True


class TestDisplayResolutionProductionWiring:
    def test_runner_display_resolution_allocates_queue_for_positive_filter(self, monkeypatch):
        """Exercise GatewayTurnMixin's real display-resolution seam."""
        import gateway.run as run_module
        from gateway.run import GatewayRunner
        from gateway.session import SessionSource

        config = {
            "display": {
                "tool_progress": "off",
                "tool_progress_filter": {"skill_view": "all"},
            }
        }
        monkeypatch.setattr(run_module, "_load_gateway_config", lambda: config)
        runner = object.__new__(GatewayRunner)
        runner._resolve_turn_toolsets = lambda *_args: ([], [])
        runner._adapter_for_source = lambda _source: SimpleNamespace(supports_status_text=False)
        source = SessionSource(platform=Platform.DISCORD, chat_id="chat", user_id="user", chat_type="dm")

        display = runner._run_agent_display_settings(source)

        assert display.tool_progress_enabled is True
        assert display.tool_progress_filter == {"skill_view": "all"}
        assert display.needs_progress_queue is True

    def test_webhook_filter_does_not_enable_chat_progress(self, monkeypatch):
        import gateway.run as run_module
        from gateway.run import GatewayRunner
        from gateway.session import SessionSource

        config = {
            "display": {
                "tool_progress": "off",
                "tool_progress_filter": {"skill_view": "all"},
            }
        }
        monkeypatch.setattr(run_module, "_load_gateway_config", lambda: config)
        runner = object.__new__(GatewayRunner)
        runner._resolve_turn_toolsets = lambda *_args: ([], [])
        runner._adapter_for_source = lambda _source: SimpleNamespace(supports_status_text=False)
        source = SessionSource(platform=Platform.WEBHOOK, chat_id="chat", user_id="user", chat_type="dm")

        display = runner._run_agent_display_settings(source)

        assert display.tool_progress_enabled is False
        assert display.needs_progress_queue is False
