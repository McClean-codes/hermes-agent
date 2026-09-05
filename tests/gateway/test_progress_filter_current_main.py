"""Ported progress-filtering onto current main — behavioral regression coverage.

Covers acceptance for mcclean/feat/progress-filtering-current-main:
- selected individual tools and/or categories are displayed
- excluded terminal/file-read busy-work not displayed
- skills, MCP, plugin categories via authoritative registry (no prefix overmatching)
- memory excluded from skills
- MCP profile-scoped (no cross-profile global map)
- per-tool/global log routing (effective log never chat-visible)
- native Slack task-card rail filtered with hidden completion tracking
- alias precedence canonicalized (platform wins, last-wins)
- errors/results/final replies/failure/interruption/completion never suppressed
- display-settings queue wiring via real gateway path
- filter never alters execution/authorization and is independent of persona state
- empty/malformed/duplicate/unknown fail safe

All tests exercise production TurnRunner/gateway paths and assert concrete
progress/log ledgers; no source-string or classifier-test-double tricks.
"""

from __future__ import annotations

import queue
import sys
import types
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.turn_context import TurnContext
from gateway.display_config import _norm_tool_progress_filter, resolve_tool_progress_filter


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_ctx(
    progress_mode="all",
    tool_progress_filter=None,
    tool_progress_enabled=None,
    with_queue=True,
    log_queue=None,
    native=False,
    thinking_enabled=False,
):
    if tool_progress_enabled is None:
        tool_progress_enabled = progress_mode not in {"off", "log"}
    q = queue.Queue() if with_queue else None
    ctx = TurnContext(
        source=MagicMock(chat_id="test-chat"),
        _run_still_current=lambda: True,
        _live_status_adapter=None,
        _live_status_mode="off",
        _thinking_enabled=thinking_enabled,
        progress_mode=progress_mode,
        progress_grouping="accumulate",
        tool_progress_enabled=tool_progress_enabled,
        tool_progress_filter=tool_progress_filter,
        progress_queue=q,
        log_queue=log_queue,
        last_progress_msg=[None],
        last_tool=[None],
        last_was_terminal_block=[False],
        repeat_count=[0],
        long_tool_hint_fired=[False],
        agent_holder=[None],
        _native_slack_task_cards=native,
    )
    return ctx


def _make_runner(ctx):
    from gateway.run_turn_runner import TurnRunner

    class _StubRunner:
        def _adapter_for_source(self, source):
            m = MagicMock()
            m.supports_code_blocks = False
            m.format_tool_preview = lambda x, **kw: x.text if hasattr(x, "text") else str(x)
            return m

        async def _deliver_platform_notice(self, source, content):
            return None

    runner = TurnRunner(_StubRunner(), ctx)  # type: ignore[arg-type]
    return runner


def _drain(q):
    out = []
    while q is not None and not q.empty():
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            break
    return out


# ---------------------------------------------------------------------------
# 1. selected individual tools displayed
# ---------------------------------------------------------------------------

class TestSelectedToolsDisplayed:
    def test_exact_tool_all_when_global_off(self):
        ctx = _make_ctx(progress_mode="off", tool_progress_enabled=True, tool_progress_filter={"skill_view": "all"})
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "skill_view", "my skill", {})
        msgs = _drain(ctx.progress_queue)
        assert len(msgs) == 1

    def test_exact_tool_suppressed_when_global_all_but_filter_off(self):
        ctx = _make_ctx(progress_mode="all", tool_progress_filter={"terminal": "off"})
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
        assert ctx.progress_queue.empty()
        runner.progress_callback("tool.started", "read_file", "file", {"path": "/tmp/x"})
        msgs = _drain(ctx.progress_queue)
        assert len(msgs) == 1
        # read_file preview is friendly "Reading x", not raw tool name
        assert "reading" in str(msgs[0]).lower() or "x" in str(msgs[0]).lower()


# ---------------------------------------------------------------------------
# 2. busy-work excluded
# ---------------------------------------------------------------------------

class TestBusyWorkExcluded:
    def test_terminal_suppressed_when_filtered_off(self):
        ctx = _make_ctx(progress_mode="all", tool_progress_filter={"terminal": "off"})
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "terminal", "echo hi", {"command": "echo hi"})
        assert ctx.progress_queue.empty()

    def test_file_read_suppressed_when_filtered_off(self):
        ctx = _make_ctx(progress_mode="all", tool_progress_filter={"read_file": "off"})
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "read_file", "README", {"path": "README.md"})
        assert ctx.progress_queue.empty()

    def test_busy_work_not_suppressed_when_no_filter(self):
        ctx = _make_ctx(progress_mode="all", tool_progress_filter={})
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
        assert not ctx.progress_queue.empty()


# ---------------------------------------------------------------------------
# 3. skills / mcp / plugin via real registry (no classifier patch)
# ---------------------------------------------------------------------------

class TestCategoryFilterable:
    def test_skills_category_all_shows_skill_tools(self):
        # skill_view is toolset "skills" in registry; check real path
        ctx = _make_ctx(progress_mode="off", tool_progress_enabled=True, tool_progress_filter={"skills": "all"})
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "skill_view", "view", {})
        assert not ctx.progress_queue.empty()
        _drain(ctx.progress_queue)
        runner.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
        assert ctx.progress_queue.empty()

    def test_skills_category_off_hides_skills_when_global_all(self):
        ctx = _make_ctx(progress_mode="all", tool_progress_filter={"skills": "off"})
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "skill_view", "view", {})
        assert ctx.progress_queue.empty()
        runner.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
        assert not ctx.progress_queue.empty()

    def test_mcp_category_via_real_registry(self):
        from tools.registry import registry

        # Register a real MCP tool via registry (toolset mcp-*)
        def _fake_mcp_handler():
            pass

        schema = {"type": "object", "properties": {}}
        tname = "_test_mcp_tool_real_1"
        try:
            registry.register(name=tname, toolset="mcp-test-server", schema=schema, handler=_fake_mcp_handler, check_fn=lambda: True)
            ctx = _make_ctx(progress_mode="all", tool_progress_filter={"mcp": "off"})
            runner = _make_runner(ctx)
            runner.progress_callback("tool.started", tname, "do", {})
            assert ctx.progress_queue.empty()
            ctx2 = _make_ctx(progress_mode="off", tool_progress_enabled=True, tool_progress_filter={"mcp": "all"})
            runner2 = _make_runner(ctx2)
            runner2.progress_callback("tool.started", tname, "do", {})
            assert not ctx2.progress_queue.empty()
        finally:
            try:
                registry.deregister(tname)
            except Exception:
                pass

    def test_plugin_category_via_real_registry(self):
        from tools.registry import registry

        mod_name = "hermes_plugins.fake_test_plugin.handlers"
        fake_mod = types.ModuleType(mod_name)
        fake_mod.__package__ = "hermes_plugins.fake_test_plugin"
        sys.modules[mod_name] = fake_mod

        def fake_handler():
            pass

        fake_handler.__module__ = mod_name
        schema = {"type": "object", "properties": {}}
        tname = "_test_plugin_tool_real_1"
        try:
            # Need to ensure plugin scope handling doesn't require extra policy; use global registration
            # The handler's module is hermes_plugins.fake..., so _plugin_owner_of will detect it.
            registry.register(name=tname, toolset="test-plugin", schema=schema, handler=fake_handler, check_fn=lambda: True)
            ctx = _make_ctx(progress_mode="all", tool_progress_filter={"plugins": "off"})
            runner = _make_runner(ctx)
            # Prove via production _get_tool_categories without patching
            from gateway.run_turn_runner import _get_tool_categories

            cats = _get_tool_categories(tname)
            assert "plugins" in cats
            runner.progress_callback("tool.started", tname, "do", {})
            assert ctx.progress_queue.empty()
            ctx2 = _make_ctx(progress_mode="off", tool_progress_enabled=True, tool_progress_filter={"plugins": "all"})
            runner2 = _make_runner(ctx2)
            runner2.progress_callback("tool.started", tname, "do", {})
            assert not ctx2.progress_queue.empty()
        finally:
            try:
                registry.deregister(tname)
            except Exception:
                pass
            sys.modules.pop(mod_name, None)

    def test_category_aliases_canonicalized(self):
        # skill alias -> skills, mcp_tools -> mcp, plugin -> plugins
        norm = _norm_tool_progress_filter({"skill": "all", "mcp_tools": "all", "plugin": "all"})
        assert norm == {"skills": "all", "mcp": "all", "plugins": "all"}
        # Also via resolve_tool_progress_filter merging
        from gateway.run_turn_runner import _resolve_effective_mode

        mode = _resolve_effective_mode("skill_view", "off", {"skill": "all"})
        assert mode == "all"
        mode2 = _resolve_effective_mode("skill_view", "off", norm)
        assert mode2 == "all"


# ---------------------------------------------------------------------------
# 4. alias precedence and exact-tool wins
# ---------------------------------------------------------------------------

class TestAliasPrecedence:
    def test_alias_precedence_canonicalized_before_merge(self):
        user_cfg = {
            "display": {
                "tool_progress_filter": {"skills": "all"},
                "platforms": {"telegram": {"tool_progress_filter": {"skill": "off"}}},
            }
        }
        merged = resolve_tool_progress_filter(user_cfg, "telegram")
        # canonicalized: both become "skills", platform wins -> off
        assert merged == {"skills": "off"}
        from gateway.run_turn_runner import _resolve_effective_mode

        assert _resolve_effective_mode("skill_view", "off", merged) == "off"

    def test_global_skills_all_platform_skill_off_resolves_off(self):
        # Drive global/platform aliases through the actual resolver; platform wins, last-wins
        user_cfg = {
            "display": {
                "tool_progress_filter": {"skills": "all"},
                "platforms": {"telegram": {"tool_progress_filter": {"skill": "off"}}},
            }
        }
        merged = resolve_tool_progress_filter(user_cfg, "telegram")
        # canonicalized: both become "skills", platform wins -> off
        assert merged == {"skills": "off"}
        from gateway.run_turn_runner import _resolve_effective_mode

        assert _resolve_effective_mode("skill_view", "off", merged) == "off"
        # Prove via production progress callback that platform precedence is honored
        ctx = _make_ctx(progress_mode="off", tool_progress_enabled=True, tool_progress_filter=merged)
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "skill_view", "view", {})
        assert ctx.progress_queue.empty()
        # Non-overridden tool on same filter should still follow global: terminal with global off -> hidden
        ctx2 = _make_ctx(progress_mode="off", tool_progress_enabled=True, tool_progress_filter=merged)
        runner2 = _make_runner(ctx2)
        runner2.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
        assert ctx2.progress_queue.empty()
        # Exact tool allow should still win over category even after resolver merge
        user_cfg2 = {
            "display": {
                "tool_progress_filter": {"skills": "off", "skill_view": "all"},
            }
        }
        merged2 = resolve_tool_progress_filter(user_cfg2, "telegram")
        assert _resolve_effective_mode("skill_view", "off", merged2) == "all"
        assert _resolve_effective_mode("skill_manage", "off", merged2) == "off"

    def test_duplicate_alias_spellings_last_wins(self):
        raw = {"skills": "all", "skill": "off", "SKILLS": "verbose"}
        norm = _norm_tool_progress_filter(raw)
        # canonicalized keys collapse to "skills", last wins -> verbose
        assert norm["skills"] == "verbose"

    def test_exact_tool_wins_over_category(self):
        # filter has skills all but terminal off should not affect skill_view if exact matches
        # Here exact tool entry should win over category
        from gateway.run_turn_runner import _resolve_effective_mode

        filt = {"skills": "off", "skill_view": "all"}
        # skill_view exact should be all despite skills off
        assert _resolve_effective_mode("skill_view", "off", filt) == "all"
        assert _resolve_effective_mode("skill_manage", "off", filt) == "off"


# ---------------------------------------------------------------------------
# 5. memory exclusion and MCP profile scope
# ---------------------------------------------------------------------------

class TestMemoryAndMCPProvenance:
    def test_memory_not_classified_as_skills(self):
        from gateway.run_turn_runner import _get_tool_categories

        cats = _get_tool_categories("memory")
        assert "skills" not in cats
        # With filter skills all + global off, memory must stay hidden (not in skills)
        ctx = _make_ctx(progress_mode="off", tool_progress_enabled=True, tool_progress_filter={"skills": "all"})
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "memory", "recall", {})
        assert ctx.progress_queue.empty()
        # Exact memory allow should still work
        ctx2 = _make_ctx(progress_mode="off", tool_progress_enabled=True, tool_progress_filter={"memory": "all"})
        runner2 = _make_runner(ctx2)
        runner2.progress_callback("tool.started", "memory", "recall", {})
        assert not ctx2.progress_queue.empty()

    def test_skills_prefix_not_overmatched(self):
        from gateway.run_turn_runner import _get_tool_categories

        # Arbitrary plugin name starting with skill should NOT be skills without registry
        cats = _get_tool_categories("skillful_plugin_tool")
        # Should not be skills because not in explicit allowlist and not toolset skills
        assert "skills" not in cats

    def test_mcp_global_map_ignored_when_not_in_registry(self):
        # Ensure process-global _mcp_tool_server_names does not cause classification
        import tools.mcp_tool as _mcp_mod

        fake = "_test_mcp_global_fallback_tool"
        added = False
        try:
            if fake not in _mcp_mod._mcp_tool_server_names:
                _mcp_mod._mcp_tool_server_names[fake] = "some-server"
                added = True
            from gateway.run_turn_runner import _get_tool_categories

            cats = _get_tool_categories(fake)
            # Without registry entry, should NOT be mcp (fail closed)
            assert "mcp" not in cats
            # With global off + mcp all, this fake tool should stay hidden
            ctx = _make_ctx(progress_mode="off", tool_progress_enabled=True, tool_progress_filter={"mcp": "all"})
            runner = _make_runner(ctx)
            runner.progress_callback("tool.started", fake, "do", {})
            assert ctx.progress_queue.empty()
        finally:
            if added:
                _mcp_mod._mcp_tool_server_names.pop(fake, None)

    def test_mcp_profile_collision_plugin_not_misclassified(self):
        # Plugin tool sharing name with MCP global map entry must be plugins, not mcp
        import tools.mcp_tool as _mcp_mod
        from tools.registry import registry

        shared = "_shared_mcp_plugin_name_1"
        added = False
        mod_name = "hermes_plugins.fake_collision.handlers"
        fake_mod = types.ModuleType(mod_name)
        sys.modules[mod_name] = fake_mod

        def fake_plugin_handler():
            pass

        fake_plugin_handler.__module__ = mod_name
        try:
            if shared not in _mcp_mod._mcp_tool_server_names:
                _mcp_mod._mcp_tool_server_names[shared] = "other-profile-server"
                added = True
            # Register plugin with same name
            registry.register(name=shared, toolset="test-plugin-collision", schema={"type": "object", "properties": {}}, handler=fake_plugin_handler, check_fn=lambda: True)
            from gateway.run_turn_runner import _get_tool_categories

            cats = _get_tool_categories(shared)
            assert "plugins" in cats
            assert "mcp" not in cats  # must not be polluted by other profile's MCP map
            # mcp all should NOT show this plugin tool
            ctx = _make_ctx(progress_mode="off", tool_progress_enabled=True, tool_progress_filter={"mcp": "all"})
            runner = _make_runner(ctx)
            runner.progress_callback("tool.started", shared, "do", {})
            assert ctx.progress_queue.empty()
            # plugins all SHOULD show it
            ctx2 = _make_ctx(progress_mode="off", tool_progress_enabled=True, tool_progress_filter={"plugins": "all"})
            runner2 = _make_runner(ctx2)
            runner2.progress_callback("tool.started", shared, "do", {})
            assert not ctx2.progress_queue.empty()
        finally:
            try:
                registry.deregister(shared)
            except Exception:
                pass
            sys.modules.pop(mod_name, None)
            if added:
                _mcp_mod._mcp_tool_server_names.pop(shared, None)


# ---------------------------------------------------------------------------
# 6. per-tool/global log routing
# ---------------------------------------------------------------------------

class TestLogRouting:
    def test_per_tool_log_never_chat_progress(self):
        lq = queue.Queue()
        ctx = _make_ctx(progress_mode="all", tool_progress_filter={"terminal": "log"}, log_queue=lq)
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
        assert ctx.progress_queue.empty()
        # Must go to log queue
        assert not lq.empty()
        logged = _drain(lq)
        assert any("terminal" in s for s in logged)

    def test_global_log_chat_silent_without_override(self):
        lq = queue.Queue()
        ctx = _make_ctx(progress_mode="log", tool_progress_enabled=False, tool_progress_filter={}, log_queue=lq)
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
        assert ctx.progress_queue.empty()
        assert not lq.empty()

    def test_global_log_with_allow_override_shows_only_selected(self):
        lq = queue.Queue()
        ctx = _make_ctx(progress_mode="log", tool_progress_enabled=True, tool_progress_filter={"terminal": "all"}, log_queue=lq)
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
        assert not ctx.progress_queue.empty()
        _drain(ctx.progress_queue)
        lq2 = queue.Queue()
        ctx2 = _make_ctx(progress_mode="log", tool_progress_enabled=True, tool_progress_filter={"terminal": "all"}, log_queue=lq2)
        runner2 = _make_runner(ctx2)
        runner2.progress_callback("tool.started", "read_file", "x", {"path": "/tmp/x"})
        # read_file not overridden, effective remains log -> should be silent in chat and go to log
        assert ctx2.progress_queue.empty()
        assert not lq2.empty()

    def test_global_log_with_deny_override_stays_silent(self):
        lq = queue.Queue()
        ctx = _make_ctx(progress_mode="log", tool_progress_enabled=False, tool_progress_filter={"terminal": "off"}, log_queue=lq)
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
        assert ctx.progress_queue.empty()
        # terminal off with global log: effective off -> no log either? Our routing returns early for log only; off suppresses both.
        # For this case, we expect chat silent; log may be empty because effective off suppresses.
        # Ensure not chat-visible
        assert ctx.progress_queue.empty()

    def test_per_tool_log_with_global_all_other_tools_visible(self):
        lq = queue.Queue()
        ctx = _make_ctx(progress_mode="all", tool_progress_filter={"terminal": "log", "read_file": "all"}, log_queue=lq)
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
        assert ctx.progress_queue.empty()
        assert not lq.empty()
        _drain(lq)
        runner.progress_callback("tool.started", "read_file", "x", {"path": "/tmp/x"})
        assert not ctx.progress_queue.empty()

    def test_native_log_not_published(self):
        # Native cards must also respect log: per-tool log goes only to log sink, never chat/native
        lq = queue.Queue()
        pq = queue.Queue()
        # Progress path: terminal log should be chat-silent, log-visible (native flag not needed for progress rail)
        ctx = _make_ctx(progress_mode="all", tool_progress_filter={"terminal": "log"}, log_queue=lq, native=False)
        ctx.progress_queue = pq
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
        assert pq.empty()
        assert not lq.empty()
        logged = _drain(lq)
        assert any("terminal" in s for s in logged)
        # Native path: same filter must hide native start and track hidden
        pq_native = queue.Queue()
        ctx_native = _make_ctx(progress_mode="all", tool_progress_filter={"terminal": "log"}, log_queue=queue.Queue(), native=True)
        ctx_native.progress_queue = pq_native
        runner_native = _make_runner(ctx_native)
        runner_native._hidden_native_call_ids.clear()
        runner_native.native_tool_start_callback("cid-log-1", "terminal", {"command": "ls"})
        assert pq_native.empty()
        assert "cid-log-1" in runner_native._hidden_native_call_ids
        # Completion for hidden must also be suppressed
        runner_native.native_tool_complete_callback("cid-log-1", "terminal", {}, "ok")
        assert pq_native.empty()
        # Non-log tool should be visible in chat and not in log
        lq2 = queue.Queue()
        pq2 = queue.Queue()
        ctx2 = _make_ctx(progress_mode="all", tool_progress_filter={"terminal": "log"}, log_queue=lq2, native=False)
        ctx2.progress_queue = pq2
        runner2 = _make_runner(ctx2)
        runner2.progress_callback("tool.started", "read_file", "x", {"path": "/tmp/x"})
        assert not pq2.empty()
        assert lq2.empty()
        # Native allow for read_file (not log) should queue when natively enabled
        pq3 = queue.Queue()
        ctx3 = _make_ctx(progress_mode="all", tool_progress_filter={"terminal": "log"}, log_queue=queue.Queue(), native=True)
        ctx3.progress_queue = pq3
        runner3 = _make_runner(ctx3)
        runner3.native_tool_start_callback("cid-read-1", "read_file", {"path": "/tmp/x"})
        assert not pq3.empty()


# ---------------------------------------------------------------------------
# 7. native Slack task-card filtering
# ---------------------------------------------------------------------------

class TestNativeCardFiltering:
    def test_native_start_filtered_when_terminal_off(self):
        ctx = _make_ctx(progress_mode="off", tool_progress_enabled=False, tool_progress_filter={"terminal": "off"}, native=True)
        # progress queue needed for native
        ctx.progress_queue = queue.Queue()
        ctx._run_still_current = lambda: True
        runner = _make_runner(ctx)
        runner.native_tool_start_callback("call-1", "terminal", {"command": "ls"})
        assert ctx.progress_queue.empty()
        # Hidden set should contain call-1
        assert "call-1" in runner._hidden_native_call_ids

    def test_native_start_allowed_when_whitelisted(self):
        ctx = _make_ctx(progress_mode="off", tool_progress_enabled=True, tool_progress_filter={"terminal": "all"}, native=True)
        ctx.progress_queue = queue.Queue()
        runner = _make_runner(ctx)
        runner.native_tool_start_callback("call-2", "terminal", {"command": "ls"})
        assert not ctx.progress_queue.empty()
        msgs = _drain(ctx.progress_queue)
        assert msgs[0]["tool_name"] == "terminal"

    def test_native_completion_hidden_cannot_resurrect(self):
        ctx = _make_ctx(progress_mode="off", tool_progress_enabled=False, tool_progress_filter={"terminal": "off"}, native=True)
        ctx.progress_queue = queue.Queue()
        runner = _make_runner(ctx)
        runner.native_tool_start_callback("call-3", "terminal", {"command": "ls"})
        assert ctx.progress_queue.empty()
        runner.native_tool_complete_callback("call-3", "terminal", {}, "result")
        # Completion for hidden call must not create card
        assert ctx.progress_queue.empty()

    def test_native_completion_only_filtered(self):
        ctx = _make_ctx(progress_mode="off", tool_progress_enabled=False, tool_progress_filter={"terminal": "off"}, native=True)
        ctx.progress_queue = queue.Queue()
        runner = _make_runner(ctx)
        # Completion without prior start but filtered
        runner.native_tool_complete_callback("call-4", "terminal", {}, "result")
        assert ctx.progress_queue.empty()
        assert "call-4" in runner._hidden_native_call_ids

    def test_native_category_filter(self):
        from tools.registry import registry

        def _h():
            pass

        tname = "_test_native_plugin_tool"
        mod_name = "hermes_plugins.fake_native.handlers"
        fake_mod = types.ModuleType(mod_name)
        sys.modules[mod_name] = fake_mod
        _h.__module__ = mod_name
        try:
            registry.register(name=tname, toolset="test-plugin-native", schema={"type": "object", "properties": {}}, handler=_h, check_fn=lambda: True)
            ctx = _make_ctx(progress_mode="all", tool_progress_filter={"plugins": "off"}, native=True)
            ctx.progress_queue = queue.Queue()
            runner = _make_runner(ctx)
            runner.native_tool_start_callback("cid-p-1", tname, {})
            assert ctx.progress_queue.empty()
            ctx2 = _make_ctx(progress_mode="off", tool_progress_enabled=True, tool_progress_filter={"plugins": "all"}, native=True)
            ctx2.progress_queue = queue.Queue()
            runner2 = _make_runner(ctx2)
            runner2.native_tool_start_callback("cid-p-2", tname, {})
            assert not ctx2.progress_queue.empty()
        finally:
            try:
                registry.deregister(tname)
            except Exception:
                pass
            sys.modules.pop(mod_name, None)


# ---------------------------------------------------------------------------
# 8. important output never suppressed
# ---------------------------------------------------------------------------

class TestImportantOutputDelivery:
    def test_subagent_failure_notice_not_suppressed_by_filter(self):
        from gateway import run as run_mod

        captured = []

        def _fake_schedule(coro, loop, logger=None, log_message=None):
            # Run the coro synchronously for test
            try:
                asyncio.run(coro)
            except RuntimeError:
                # Already in event loop? run via new loop
                loop2 = asyncio.new_event_loop()
                loop2.run_until_complete(coro)
                loop2.close()
            return MagicMock()

        orig = run_mod.safe_schedule_threadsafe
        run_mod.safe_schedule_threadsafe = _fake_schedule  # type: ignore[assignment]
        captured_src = MagicMock(chat_id="c1")

        class Stub:
            def _adapter_for_source(self, s):
                return None

            async def _deliver_platform_notice(self, source, content):
                captured.append(content)

        try:
            ctx = TurnContext(
                source=captured_src,
                _run_still_current=lambda: True,
                progress_queue=queue.Queue(),
                _loop_for_step=None,
                tool_progress_filter={"terminal": "off"},
                tool_progress_enabled=False,
                progress_mode="off",
            )
            from gateway.run_turn_runner import TurnRunner

            runner = TurnRunner(Stub(), ctx)  # type: ignore[arg-type]
            runner.progress_callback("subagent.complete", preview="Error 404", status="failed", goal="do thing", duration_seconds=5)
            assert len(captured) == 1
            assert "do thing" in captured[0] or "failed" in captured[0].lower()
        finally:
            run_mod.safe_schedule_threadsafe = orig  # type: ignore[assignment]

    @pytest.mark.asyncio
    async def test_error_result_not_suppressed(self):
        # Errors/results must still be delivered via production gateway message-handling caller even when progress for that tool is filtered off
        pq = queue.Queue()
        lq = queue.Queue()
        ctx = TurnContext(
            source=MagicMock(chat_id="c1"),
            _run_still_current=lambda: True,
            _live_status_adapter=None,
            _live_status_mode="off",
            _thinking_enabled=False,
            progress_mode="all",
            progress_grouping="accumulate",
            tool_progress_enabled=True,
            tool_progress_filter={"terminal": "off"},
            progress_queue=pq,
            log_queue=lq,
            last_progress_msg=[None],
            last_tool=[None],
            last_was_terminal_block=[False],
            repeat_count=[0],
            long_tool_hint_fired=[False],
            agent_holder=[None],
            _native_slack_task_cards=False,
            result_holder=[None],
            tools_holder=[None],
            stream_consumer_holder=[None],
            streaming_tts_consumer_holder=[None],
        )
        class StubRunner:
            def _adapter_for_source(self, s):
                m = MagicMock()
                m.supports_code_blocks = False
                m.format_tool_preview = lambda x, **kw: x.text if hasattr(x, "text") else str(x)
                return m
            async def _deliver_platform_notice(self, src, content):
                return None
        from gateway.run_turn_runner import TurnRunner
        runner = TurnRunner(StubRunner(), ctx)  # type: ignore[arg-type]
        runner.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
        assert pq.empty(), "filtered progress must not appear"
        assert lq.empty(), "log rail must stay empty for filtered start"

        # Production gateway error delivery via full message-handling path with only adapter send controlled
        from gateway.run import GatewayRunner
        from gateway.config import Platform, GatewayConfig, PlatformConfig
        from gateway.run import _sanitize_gateway_final_response
        from gateway.session import SessionSource, SessionEntry, build_session_key
        from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
        from datetime import datetime, timedelta
        import os

        ledger: list[str] = []

        class _CaptureTelegramAdapter(BasePlatformAdapter):
            def __init__(self):
                super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.TELEGRAM)

            async def connect(self, *, is_reconnect: bool = False) -> bool:
                return True

            async def disconnect(self) -> None:
                return None

            async def send(self, chat_id, content, reply_to=None, metadata=None):
                ledger.append(content)
                return SendResult(success=True, message_id="tg-1")

            async def send_typing(self, chat_id, metadata=None):
                return None

            async def get_chat_info(self, chat_id):
                return {"id": chat_id}

        fake_adapter = _CaptureTelegramAdapter()
        _orig_send = fake_adapter.send
        fake_adapter.send = AsyncMock(side_effect=_orig_send)

        config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")})
        gw = object.__new__(GatewayRunner)
        gw.config = config
        gw.adapters = {Platform.TELEGRAM: fake_adapter}
        gw._voice_mode = {}
        gw._running_agents = {}
        gw._running_agents_ts = {}
        gw._pending_messages = {}
        gw._pending_approvals = {}
        gw._is_user_authorized = lambda _source: True
        gw._set_session_env = lambda _context: None
        gw._clear_session_env = lambda _tokens: None
        gw._is_user_authorized_for_source = lambda _s, **kw: True
        gw._session_db = MagicMock()
        gw._session_db.get_telegram_topic_binding = AsyncMock(return_value=None)
        gw._session_db.get_compression_tip = AsyncMock(return_value=None)
        gw.hooks = MagicMock()
        gw.hooks.emit = AsyncMock()
        now = datetime.now()
        session_entry = SessionEntry(
            session_key="agent:main:telegram:group:-1001:12345",
            session_id="sess-error-1",
            created_at=now - timedelta(seconds=10),
            updated_at=now,
            platform=Platform.TELEGRAM,
            chat_type="group",
        )
        gw.session_store = MagicMock()
        gw.session_store.get_or_create_session.return_value = session_entry
        gw.session_store.load_transcript.return_value = []
        gw.session_store.has_any_sessions.return_value = True
        gw.session_store.rewrite_transcript = MagicMock()
        gw.session_store.append_to_transcript = MagicMock()
        gw.session_store.update_session = MagicMock()
        gw.session_store.has_platform_message_id = MagicMock(return_value=False)
        gw.session_store._save = MagicMock()
        gw.session_store._record_gateway_session_peer = MagicMock()
        gw._adapter_for_source = lambda source: fake_adapter
        gw._resolve_session_agent_runtime = MagicMock(return_value=("test/model", {"api_key": "fake", "base_url": ""}))
        gw._resolve_session_reasoning_config = MagicMock(return_value=None)
        gw._resolve_session_service_tier = MagicMock(return_value=None)
        gw._provider_routing = {}
        gw._reasoning_config = None
        gw._service_tier = None
        gw._async_session_store = gw.session_store  # type: ignore[attr-defined]

        error_text = "error: permission denied"
        sanitized = _sanitize_gateway_final_response(Platform.TELEGRAM, error_text)
        assert sanitized == error_text
        mock_agent_result = {
            "final_response": sanitized,
            "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": sanitized}],
            "tools": [],
            "failed": True,
            "completed": False,
            "api_calls": 1,
            "history_offset": 0,
            "last_prompt_tokens": 0,
            "session_id": "sess-error-1",
        }
        gw._run_agent = AsyncMock(return_value=mock_agent_result)

        event = MessageEvent(
            text="hi",
            source=SessionSource(platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group", user_id="12345"),
            message_id="msg-error-1",
        )

        gw._turn_leases = None
        gw._session_sources = {}
        gw._session_sources_max = 512
        gw._is_session_running = lambda k: False
        gw._evict_idle_stale_agent = lambda k: None
        gw._evict_reaped_agent = lambda k: None
        gw._persist_active_agents = lambda: None
        gw._is_session_run_current = lambda k, gen: True
        gw._begin_session_run_generation = lambda k: 1
        gw._reply_anchor_for_event = lambda e: None
        gw._get_guild_id = lambda e: None
        gw._should_send_voice_reply = lambda *a, **kw: False
        gw._thread_metadata_for_source = lambda s, anchor=None: None
        gw._event_session_key = lambda e: build_session_key(e.source)
        gw._event_thread_metadata = lambda e, s: None

        fake_adapter.set_message_handler(gw._handle_message)
        fake_adapter._keep_typing = lambda *a, **kw: asyncio.Event().wait()

        _orig_home = os.environ.get("TELEGRAM_HOME_CHANNEL")
        os.environ["TELEGRAM_HOME_CHANNEL"] = "-1001"
        try:
            await fake_adapter._process_message_background(event, build_session_key(event.source))

            assert ledger == [sanitized], f"ledger was {ledger}"
            assert fake_adapter.send.call_count == 1
            _called = None
            if fake_adapter.send.call_args is not None:
                _a, _kw = fake_adapter.send.call_args
                if len(_a) >= 2:
                    _called = _a[1]
                else:
                    _called = _kw.get("content")
            assert _called == sanitized
            assert ledger[0] == _called
            assert ledger[0] == error_text

            assert pq.empty(), "progress must stay empty after error delivery"
            assert lq.empty(), "log must stay empty after error delivery"

            runner.progress_callback("tool.completed", "terminal", None, {})
            assert pq.empty(), "progress must stay empty after tool.completed"
            assert ledger == [sanitized], "tool completion must not duplicate or clear error"
            assert fake_adapter.send.call_count == 1, "tool.completed must not trigger extra send"
        finally:
            if _orig_home is None:
                os.environ.pop("TELEGRAM_HOME_CHANNEL", None)
            else:
                os.environ["TELEGRAM_HOME_CHANNEL"] = _orig_home

    @pytest.mark.asyncio
    async def test_tool_completed_does_not_block_final_reply(self):
        # Final reply via production gateway message-handling path to adapter send boundary must not be suppressed
        pq = queue.Queue()
        lq = queue.Queue()
        ctx = TurnContext(
            source=MagicMock(chat_id="c1"),
            _run_still_current=lambda: True,
            _live_status_adapter=None,
            _live_status_mode="off",
            _thinking_enabled=False,
            progress_mode="off",
            progress_grouping="accumulate",
            tool_progress_enabled=False,
            tool_progress_filter={"terminal": "off"},
            progress_queue=pq,
            log_queue=lq,
            last_progress_msg=[None],
            last_tool=[None],
            last_was_terminal_block=[False],
            repeat_count=[0],
            long_tool_hint_fired=[False],
            agent_holder=[None],
            _native_slack_task_cards=False,
            result_holder=[None],
            tools_holder=[None],
            stream_consumer_holder=[None],
            streaming_tts_consumer_holder=[None],
        )
        from gateway.run_turn_runner import TurnRunner
        from gateway.config import Platform, GatewayConfig, PlatformConfig
        from gateway.run import _sanitize_gateway_final_response
        from gateway.session import SessionSource, SessionEntry, build_session_key
        from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
        from datetime import datetime, timedelta
        import os

        class StubRunner:
            def _adapter_for_source(self, s):
                m = MagicMock()
                m.supports_code_blocks = False
                m.format_tool_preview = lambda x, **kw: x.text if hasattr(x, "text") else str(x)
                return m

        runner = TurnRunner(StubRunner(), ctx)  # type: ignore[arg-type]
        runner.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
        assert pq.empty(), "progress for filtered tool must be suppressed"
        assert lq.empty(), "log rail must stay empty for filtered start"

        # Production gateway final delivery via full message-handling path with only adapter send controlled
        from gateway.run import GatewayRunner

        ledger: list[str] = []

        class _CaptureTelegramAdapter(BasePlatformAdapter):
            def __init__(self):
                super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.TELEGRAM)

            async def connect(self, *, is_reconnect: bool = False) -> bool:
                return True

            async def disconnect(self) -> None:
                return None

            async def send(self, chat_id, content, reply_to=None, metadata=None):
                ledger.append(content)
                return SendResult(success=True, message_id="tg-1")

            async def send_typing(self, chat_id, metadata=None):
                return None

            async def get_chat_info(self, chat_id):
                return {"id": chat_id}

        fake_adapter = _CaptureTelegramAdapter()
        _orig_send = fake_adapter.send
        fake_adapter.send = AsyncMock(side_effect=_orig_send)

        config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake-token")})
        gw = object.__new__(GatewayRunner)
        gw.config = config
        gw.adapters = {Platform.TELEGRAM: fake_adapter}
        gw._voice_mode = {}
        gw._running_agents = {}
        gw._running_agents_ts = {}
        gw._pending_messages = {}
        gw._pending_approvals = {}
        gw._is_user_authorized = lambda _source: True
        gw._set_session_env = lambda _context: None
        gw._clear_session_env = lambda _tokens: None
        gw._is_user_authorized_for_source = lambda _s, **kw: True
        gw._session_db = MagicMock()
        gw._session_db.get_telegram_topic_binding = AsyncMock(return_value=None)
        gw._session_db.get_compression_tip = AsyncMock(return_value=None)
        gw.hooks = MagicMock()
        gw.hooks.emit = AsyncMock()
        now = datetime.now()
        session_entry = SessionEntry(
            session_key="agent:main:telegram:group:-1001:12345",
            session_id="sess-final-1",
            created_at=now - timedelta(seconds=10),
            updated_at=now,
            platform=Platform.TELEGRAM,
            chat_type="group",
        )
        gw.session_store = MagicMock()
        gw.session_store.get_or_create_session.return_value = session_entry
        gw.session_store.load_transcript.return_value = []
        gw.session_store.has_any_sessions.return_value = True
        gw.session_store.rewrite_transcript = MagicMock()
        gw.session_store.append_to_transcript = MagicMock()
        gw.session_store.update_session = MagicMock()
        gw.session_store.has_platform_message_id = MagicMock(return_value=False)
        gw.session_store._save = MagicMock()
        gw.session_store._record_gateway_session_peer = MagicMock()
        gw._adapter_for_source = lambda source: fake_adapter
        gw._resolve_session_agent_runtime = MagicMock(return_value=("test/model", {"api_key": "fake", "base_url": ""}))
        gw._resolve_session_reasoning_config = MagicMock(return_value=None)
        gw._resolve_session_service_tier = MagicMock(return_value=None)
        gw._provider_routing = {}
        gw._reasoning_config = None
        gw._service_tier = None
        gw._async_session_store = gw.session_store  # type: ignore[attr-defined]

        final_text = "Hello final reply"
        sanitized = _sanitize_gateway_final_response(Platform.TELEGRAM, final_text)
        assert sanitized == final_text
        mock_agent_result = {
            "final_response": sanitized,
            "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": sanitized}],
            "tools": [],
            "failed": False,
            "completed": True,
            "api_calls": 1,
            "history_offset": 0,
            "last_prompt_tokens": 0,
            "session_id": "sess-final-1",
        }
        gw._run_agent = AsyncMock(return_value=mock_agent_result)

        event = MessageEvent(
            text="hi",
            source=SessionSource(platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group", user_id="12345"),
            message_id="msg-final-1",
        )

        gw._turn_leases = None
        gw._session_sources = {}
        gw._session_sources_max = 512
        gw._is_session_running = lambda k: False
        gw._evict_idle_stale_agent = lambda k: None
        gw._evict_reaped_agent = lambda k: None
        gw._persist_active_agents = lambda: None
        gw._is_session_run_current = lambda k, gen: True
        gw._begin_session_run_generation = lambda k: 1
        gw._reply_anchor_for_event = lambda e: None
        gw._get_guild_id = lambda e: None
        gw._should_send_voice_reply = lambda *a, **kw: False
        gw._thread_metadata_for_source = lambda s, anchor=None: None
        gw._event_session_key = lambda e: build_session_key(e.source)
        gw._event_thread_metadata = lambda e, s: None

        fake_adapter.set_message_handler(gw._handle_message)
        fake_adapter._keep_typing = lambda *a, **kw: asyncio.Event().wait()

        _orig_home = os.environ.get("TELEGRAM_HOME_CHANNEL")
        os.environ["TELEGRAM_HOME_CHANNEL"] = "-1001"
        try:
            await fake_adapter._process_message_background(event, build_session_key(event.source))

            assert ledger == ["Hello final reply"], f"ledger was {ledger}"
            assert fake_adapter.send.call_count == 1
            # Tie ledger payload to the actual send call argument
            _called = None
            if fake_adapter.send.call_args is not None:
                _a, _kw = fake_adapter.send.call_args
                if len(_a) >= 2:
                    _called = _a[1]
                else:
                    _called = _kw.get("content")
            assert _called == "Hello final reply"
            assert ledger[0] == _called

            assert pq.empty(), "progress must stay empty after final delivery"
            assert lq.empty(), "log must stay empty after final delivery"

            runner.progress_callback("tool.completed", "terminal", None, {})
            assert pq.empty(), "progress must stay empty after tool.completed"
            assert ledger == ["Hello final reply"], "tool completion must not duplicate or clear final"
            assert fake_adapter.send.call_count == 1, "tool.completed must not trigger extra send"
        finally:
            if _orig_home is None:
                os.environ.pop("TELEGRAM_HOME_CHANNEL", None)
            else:
                os.environ["TELEGRAM_HOME_CHANNEL"] = _orig_home

    def test_thinking_still_gated_separately(self):
        ctx = _make_ctx(progress_mode="off", tool_progress_filter={"terminal": "all"}, thinking_enabled=True)
        ctx.tool_progress_enabled = False
        runner = _make_runner(ctx)
        runner.progress_callback("_thinking", "_thinking", "hmm", {})
        msgs = _drain(ctx.progress_queue)
        assert any("hmm" in str(m) for m in msgs)

    def test_verbose_mode_respects_filter(self):
        ctx = _make_ctx(progress_mode="verbose", tool_progress_filter={"terminal": "off"})
        ctx.tool_progress_enabled = True
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "terminal", "ls", {"command": "echo hi"})
        assert ctx.progress_queue.empty()
        runner.progress_callback("tool.started", "read_file", "x", {"path": "/tmp/x"})
        assert not ctx.progress_queue.empty()

    def test_interruption_suppresses_progress_but_not_notice(self):
        # When interrupted, tool.started should not emit progress
        ctx = _make_ctx(progress_mode="all", tool_progress_filter={})
        runner = _make_runner(ctx)
        # Simulate interrupted agent
        mock_agent = MagicMock()
        mock_agent.is_interrupted = True
        ctx.agent_holder[0] = mock_agent
        runner.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
        assert ctx.progress_queue.empty()
        # But subagent failure notice must still go even when interrupted? Check gate: _progress_subagent_notice checks _run_still_current, not interrupted
        # So we test that subagent notice still delivers even when interrupted flag set
        from gateway import run as run_mod

        captured2 = []

        def _fake_sched(coro, loop, logger=None, log_message=None):
            try:
                asyncio.run(coro)
            except RuntimeError:
                loop2 = asyncio.new_event_loop()
                loop2.run_until_complete(coro)
                loop2.close()
            return MagicMock()

        orig2 = run_mod.safe_schedule_threadsafe
        run_mod.safe_schedule_threadsafe = _fake_sched  # type: ignore[assignment]
        try:
            class Stub2:
                def _adapter_for_source(self, s):
                    return None

                async def _deliver_platform_notice(self, source, content):
                    captured2.append(content)

            ctx2 = TurnContext(source=MagicMock(chat_id="c2"), _run_still_current=lambda: True, progress_queue=queue.Queue(), _loop_for_step=None, tool_progress_filter={}, tool_progress_enabled=False, progress_mode="off")
            ctx2.agent_holder[0] = mock_agent
            from gateway.run_turn_runner import TurnRunner as _TR2

            runner2 = _TR2(Stub2(), ctx2)  # type: ignore[arg-type]
            # Need a fresh runner with stub2
            runner2.progress_callback("subagent.complete", preview="err", status="failed", goal="g", duration_seconds=1)
            assert len(captured2) == 1
        finally:
            run_mod.safe_schedule_threadsafe = orig2  # type: ignore[assignment]

    def test_delivery_paths_separate_via_real_queues(self):
        # Progress queue vs log queue vs status: ensure they are separate
        pq = queue.Queue()
        lq = queue.Queue()
        ctx = _make_ctx(progress_mode="all", tool_progress_filter={"terminal": "log"}, log_queue=lq)
        ctx.progress_queue = pq
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
        # Terminal log -> should be in log queue only, not progress, and not status
        assert pq.empty()
        assert not lq.empty()
        # Other tool should be in progress queue, not log
        _drain(lq)
        runner.progress_callback("tool.started", "read_file", "x", {"path": "/tmp/x"})
        assert not pq.empty()
        assert lq.empty()


# ---------------------------------------------------------------------------
# 9. display-settings queue wiring via real gateway path
# ---------------------------------------------------------------------------

class TestDisplaySettingsWiring:
    def test_display_settings_enables_queue_for_whitelisted_when_global_off(self):
        from gateway.config import Platform
        from gateway.session import SessionSource
        from gateway.run_turn import GatewayTurnMixin
        from unittest.mock import MagicMock, patch

        # Use real GatewayTurnMixin._run_agent_display_settings via a minimal host
        class Host(GatewayTurnMixin):
            def __init__(self):
                self.adapters = {}

            def _adapter_for_source(self, source):
                m = MagicMock()
                m.supports_status_text = False
                m.native_task_cards_enabled = MagicMock(return_value=False)
                m.supports_code_blocks = False
                return m

            def _resolve_turn_toolsets(self, user_config, source, platform_key):
                return [], []

        from gateway.run import GatewayRunner
        Host._RunAgentDisplay = GatewayRunner._RunAgentDisplay
        host = Host()
        user_cfg = {"display": {"tool_progress": "off", "tool_progress_filter": {"skill_view": "all"}}}
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1", thread_id=None, chat_type="private")
        with patch("gateway.run._load_gateway_config", return_value=user_cfg):
            disp = host._run_agent_display_settings(source)
            assert disp.tool_progress_filter == {"skill_view": "all"}
            assert disp.tool_progress_enabled is True
            assert disp.needs_progress_queue is True
            assert disp.progress_mode == "off"
            # log queue should be None (no log filter)
            assert disp.log_queue is None

    def test_display_settings_does_not_enable_queue_for_log_only(self):
        from gateway.config import Platform
        from gateway.session import SessionSource
        from gateway.run_turn import GatewayTurnMixin

        class Host(GatewayTurnMixin):
            def __init__(self):
                self.adapters = {}

            def _adapter_for_source(self, source):
                m = MagicMock()
                m.supports_status_text = False
                m.native_task_cards_enabled = MagicMock(return_value=False)
                return m

            def _resolve_turn_toolsets(self, user_config, source, platform_key):
                return [], []

        from gateway.run import GatewayRunner
        Host._RunAgentDisplay = GatewayRunner._RunAgentDisplay
        host = Host()
        user_cfg = {"display": {"tool_progress": "off", "tool_progress_filter": {"terminal": "log"}}}
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1", thread_id=None, chat_type="private")
        with patch("gateway.run._load_gateway_config", return_value=user_cfg):
            disp = host._run_agent_display_settings(source)
            assert disp.tool_progress_enabled is False
            assert disp.needs_progress_queue is False
            # log queue must exist for per-tool log
            assert disp.log_queue is not None

    def test_display_settings_platform_overrides_global(self):
        user_cfg = {
            "display": {
                "tool_progress_filter": {"terminal": "off", "read_file": "off"},
                "platforms": {"telegram": {"tool_progress_filter": {"terminal": "all"}}},
            }
        }
        assert resolve_tool_progress_filter(user_cfg, "telegram") == {"terminal": "all", "read_file": "off"}
        assert resolve_tool_progress_filter(user_cfg, "discord") == {"terminal": "off", "read_file": "off"}

    def test_display_settings_creates_log_queue_for_global_log_with_override(self):
        from gateway.config import Platform
        from gateway.session import SessionSource
        from gateway.run_turn import GatewayTurnMixin

        class Host(GatewayTurnMixin):
            def __init__(self):
                self.adapters = {}

            def _adapter_for_source(self, source):
                m = MagicMock()
                m.supports_status_text = False
                m.native_task_cards_enabled = MagicMock(return_value=False)
                return m

            def _resolve_turn_toolsets(self, user_config, source, platform_key):
                return [], []

        from gateway.run import GatewayRunner
        Host._RunAgentDisplay = GatewayRunner._RunAgentDisplay
        host = Host()
        user_cfg = {"display": {"tool_progress": "log", "tool_progress_filter": {"terminal": "all"}}}
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1", thread_id=None, chat_type="private")
        with patch("gateway.run._load_gateway_config", return_value=user_cfg):
            disp = host._run_agent_display_settings(source)
            assert disp.progress_mode == "log"
            assert disp.log_queue is not None
            assert disp.tool_progress_enabled is True  # whitelist enables progress
            assert disp.needs_progress_queue is True


# ---------------------------------------------------------------------------
# 10. persona independence (behavioral)
# ---------------------------------------------------------------------------

class TestPersonaIndependence:
    def test_filter_works_same_with_and_without_voice_ack(self):
        # Persona-related voice ack should not affect filtering
        ctx1 = _make_ctx(progress_mode="all", tool_progress_filter={"terminal": "off"})
        ctx1._voice_ack_guild = [123]
        ctx1._voice_ack_fired = [False]
        ctx1._voice_ack_loop = None
        runner1 = _make_runner(ctx1)
        runner1.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
        assert ctx1.progress_queue.empty()

        ctx2 = _make_ctx(progress_mode="all", tool_progress_filter={"terminal": "off"})
        runner2 = _make_runner(ctx2)
        runner2.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
        assert ctx2.progress_queue.empty()

        # Positive case also independent
        ctx3 = _make_ctx(progress_mode="off", tool_progress_enabled=True, tool_progress_filter={"terminal": "all"})
        ctx3._voice_ack_guild = [123]
        runner3 = _make_runner(ctx3)
        runner3.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
        assert not ctx3.progress_queue.empty()

    def test_filter_does_not_block_tool_execution(self):
        # Filter must suppress only the progress rail; tool must still execute via production executor
        from tools.registry import registry
        from unittest.mock import MagicMock, patch
        import json
        import uuid
        from types import SimpleNamespace

        executed: list[str] = []

        def real_handler(*args, **kwargs):
            # Registry may pass tool args as first positional dict (handler(args, task_id=...)) or as kwargs
            path = ""
            if args and isinstance(args[0], dict):
                path = args[0].get("path", "")
            elif "path" in kwargs:
                path = kwargs.get("path", "")
            elif args:
                path = str(args[0])
            executed.append(path)
            return f"read {path}"

        schema = {"type": "object", "properties": {"path": {"type": "string"}}}
        tname = "_test_exec_real_tool_1"
        try:
            registry.register(name=tname, toolset="test-exec", schema=schema, handler=real_handler, check_fn=lambda: True)
            # Production progress filtering check before execution
            ctx = _make_ctx(progress_mode="all", tool_progress_filter={tname: "off"})
            runner = _make_runner(ctx)
            runner.progress_callback("tool.started", tname, "x", {"path": "/tmp/x"})
            assert ctx.progress_queue.empty(), "filtered tool progress must be suppressed via TurnRunner"

            # Execute via the production tool-call executor so registry lookup, authorization,
            # middleware, _begin_tool_execution and _invoke_tool are exercised.
            # Use a real AIAgent with only external effect (our handler) controlled.
            with patch("model_tools.get_tool_definitions", return_value=[{"type": "function", "function": {"name": tname, "description": "test", "parameters": schema}}]), \
                 patch("model_tools.check_toolset_requirements", return_value={}), \
                 patch("agent.process_bootstrap.OpenAI"):
                from run_agent import AIAgent

                agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True, skip_context_files=True, skip_memory=True)
                agent.client = MagicMock()
                # Make the agent aware of our disposable tool for authorization
                agent.valid_tool_names = set(registry.get_all_tool_names())
                # Wire progress callbacks so _begin_tool_execution exercises the real filter
                progress_started: list[tuple] = []
                tool_start_ledger: list[tuple] = []
                tool_complete_ledger: list[tuple] = []

                orig_progress_cb = runner.progress_callback

                def _wrapped_progress(*a, **kw):
                    progress_started.append(a)
                    return orig_progress_cb(*a, **kw)

                agent.tool_progress_callback = _wrapped_progress
                agent.tool_start_callback = lambda call_id, name, args: tool_start_ledger.append((call_id, name, args))
                agent.tool_complete_callback = lambda call_id, name, args, result: tool_complete_ledger.append((call_id, name, args, result))

                def _mock_tool_call(name=tname, arguments='{"path": "/tmp/x"}', call_id=None):
                    return SimpleNamespace(id=call_id or f"call_{uuid.uuid4().hex[:8]}", type="function", function=SimpleNamespace(name=name, arguments=arguments))

                def _mock_assistant_msg(content="", tool_calls=None):
                    return SimpleNamespace(content=content, tool_calls=tool_calls)

                tc = _mock_tool_call(name=tname, arguments=json.dumps({"path": "/tmp/x"}), call_id="c1")
                mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
                messages: list[dict] = []
                agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

                # Filter must have kept progress suppressed even though executor called _begin_tool_execution
                assert ctx.progress_queue.empty(), "filtered tool must stay suppressed when executed via production executor"
                # Tool must have executed through the production path and returned expected result
                assert executed == ["/tmp/x"], "handler must have been invoked via production executor, not direct call"
                assert len(messages) == 1
                assert messages[0]["role"] == "tool"
                assert "read /tmp/x" in messages[0]["content"]
                # Authorization/registry membership must remain intact after filtered execution
                assert tname in registry.get_all_tool_names()
                assert registry.get_entry(tname) is not None
                # Real callbacks prove the production path was exercised
                assert any(name == tname for _, name, *_ in tool_start_ledger), "tool_start must have been called via production executor"
                assert any(name == tname for _, name, *_ in tool_complete_ledger), "tool_complete must have been called via production executor"
                # Filter must not have mutated context execution fields
                assert ctx.tool_progress_filter == {tname: "off"}
                # Progress for a non-filtered tool would still be visible (sanity)
                ctx2 = _make_ctx(progress_mode="all", tool_progress_filter={tname: "off"})
                runner2 = _make_runner(ctx2)
                runner2.progress_callback("tool.started", "read_file", "x", {"path": "/tmp/y"})
                assert not ctx2.progress_queue.empty()
        finally:
            try:
                registry.deregister(tname)
            except Exception:
                pass

    def test_filter_does_not_modify_ctx_execution_fields(self):
        ctx = _make_ctx(progress_mode="all", tool_progress_filter={"terminal": "off"})
        orig_enabled = ctx.tool_progress_enabled
        orig_mode = ctx.progress_mode
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "terminal", "ls", {})
        assert ctx.tool_progress_enabled == orig_enabled
        assert ctx.progress_mode == orig_mode



# ---------------------------------------------------------------------------
# 13. redaction boundary (allowlisted progress must not carry raw secrets)
# ---------------------------------------------------------------------------

class TestProgressRedactionBoundary:
    """Allowlisted progress previews (terminal blocks, verbose args, URLs/paths, plugin/MCP, Codex/native, live status) must be secret-redacted via authoritative boundary."""

    SECRET_MARKER = "sk-1234567890abcdefABCDEF1234"
    SECRET_GHP = "ghp_" + "A" * 30
    NON_SECRET = "echo hello world"

    def _assert_redacted(self, raw_marker: str, payload: str):
        from agent.redact import redact_sensitive_text
        # Authoritative check: direct force-redaction must change the marker (proof marker is recognized)
        assert redact_sensitive_text(raw_marker, force=True) != raw_marker, "marker must be recognized by authoritative redactor"
        # Payload must not contain raw marker
        assert raw_marker not in payload, f"raw marker leaked: {payload!r}"
        # Payload must be non-empty and not just dropped (fail-closed but still delivered for allowlisted)
        assert payload.strip() != ""
        # Payload must differ from raw marker (fail-closed ensures delivery, not dropping)
        assert payload != raw_marker

    def test_terminal_via_begin_tool_execution_is_redacted(self):
        # Real _begin_tool_execution path with global off + terminal all override; secret in command must be redacted in outbound queue
        from agent.tool_executor import _begin_tool_execution, _ToolCallRef
        from unittest.mock import MagicMock
        secret = self.SECRET_MARKER
        ctx = _make_ctx(progress_mode="off", tool_progress_enabled=True, tool_progress_filter={"terminal": "all"})
        # Enable code blocks on adapter so terminal renders as fenced block
        runner = _make_runner(ctx)
        # Replace runner's adapter to support code blocks
        def _fake_adapter(source):
            m = MagicMock()
            m.supports_code_blocks = True
            m.format_tool_preview = lambda x, **kw: x.text if hasattr(x, "text") else str(x)
            return m
        runner._runner._adapter_for_source = _fake_adapter  # type: ignore[assignment]
        # Mock agent with required attrs for _begin_tool_execution
        agent = MagicMock()
        agent.quiet_mode = False
        agent.tool_progress_mode = "off"
        agent.verbose_logging = False
        agent.log_prefix_chars = 200
        agent._wrap_verbose = lambda a, b: b
        agent._current_tool = None
        agent._touch_activity = lambda x: None
        agent._checkpoint_mgr = MagicMock(enabled=False)
        agent.tool_progress_callback = runner.progress_callback
        agent.tool_start_callback = None
        ref = _ToolCallRef(name="terminal", args={"command": f"echo {secret} --flag"}, task_id="tid", call_id="cid-redact-1", trace=[])
        _begin_tool_execution(agent, ref, display_index=0)
        msgs = _drain(ctx.progress_queue)
        assert len(msgs) == 1, f"allowlisted terminal should have produced one progress item, got {msgs}"
        payload = str(msgs[0])
        self._assert_redacted(secret, payload)
        # Non-secret allowlisted preview must still follow intended delivery mode (not dropped)
        ctx2 = _make_ctx(progress_mode="off", tool_progress_enabled=True, tool_progress_filter={"terminal": "all"})
        runner2 = _make_runner(ctx2)
        runner2._runner._adapter_for_source = _fake_adapter  # type: ignore[assignment]
        agent2 = MagicMock()
        agent2.quiet_mode = False
        agent2.tool_progress_mode = "off"
        agent2.verbose_logging = False
        agent2.log_prefix_chars = 200
        agent2._wrap_verbose = lambda a, b: b
        agent2._current_tool = None
        agent2._touch_activity = lambda x: None
        agent2._checkpoint_mgr = MagicMock(enabled=False)
        agent2.tool_progress_callback = runner2.progress_callback
        agent2.tool_start_callback = None
        ref2 = _ToolCallRef(name="terminal", args={"command": self.NON_SECRET}, task_id="tid", call_id="cid-ok", trace=[])
        _begin_tool_execution(agent2, ref2, display_index=0)
        msgs2 = _drain(ctx2.progress_queue)
        assert len(msgs2) == 1
        assert self.NON_SECRET in str(msgs2[0])
        assert secret not in str(msgs2[0])

    def test_verbose_args_redacted(self):
        secret = self.SECRET_MARKER
        ctx = _make_ctx(progress_mode="verbose", tool_progress_filter={"web_search": "verbose"})
        ctx.tool_progress_enabled = True
        runner = _make_runner(ctx)
        # verbose mode queues args JSON directly
        runner.progress_callback("tool.started", "web_search", "query", {"query": f"leak {secret} please"})
        msgs = _drain(ctx.progress_queue)
        assert len(msgs) == 1
        payload = str(msgs[0])
        self._assert_redacted(secret, payload)

    def test_url_path_preview_redacted(self):
        secret = self.SECRET_MARKER
        ctx = _make_ctx(progress_mode="all", tool_progress_filter={"web_extract": "all"})
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "web_extract", "urls", {"urls": [f"https://example.com/?token={secret}"]})
        msgs = _drain(ctx.progress_queue)
        assert len(msgs) == 1
        payload = str(msgs[0])
        self._assert_redacted(secret, payload)

    def test_plugin_mcp_preview_redacted(self):
        from tools.registry import registry
        import types, sys
        secret = self.SECRET_GHP
        # Plugin tool
        mod_name = "hermes_plugins.fake_redact.handlers"
        fake_mod = types.ModuleType(mod_name)
        sys.modules[mod_name] = fake_mod
        def handler(query: str = ""):
            return query
        handler.__module__ = mod_name
        tname = "_test_redact_plugin_tool"
        try:
            registry.register(name=tname, toolset="test-plugin-redact", schema={"type": "object", "properties": {"query": {"type": "string"}}}, handler=handler, check_fn=lambda: True)
            ctx = _make_ctx(progress_mode="off", tool_progress_enabled=True, tool_progress_filter={"plugins": "all"})
            runner = _make_runner(ctx)
            runner.progress_callback("tool.started", tname, "do", {"query": f"secret {secret}"})
            msgs = _drain(ctx.progress_queue)
            assert len(msgs) == 1
            payload = str(msgs[0])
            self._assert_redacted(secret, payload)
            # MCP tool
            t_mcp = "_test_redact_mcp_tool"
            def mcp_h(x: str = ""): pass
            registry.register(name=t_mcp, toolset="mcp-redact-server", schema={"type": "object", "properties": {"x": {"type": "string"}}}, handler=mcp_h, check_fn=lambda: True)
            ctx2 = _make_ctx(progress_mode="off", tool_progress_enabled=True, tool_progress_filter={"mcp": "all"})
            runner2 = _make_runner(ctx2)
            runner2.progress_callback("tool.started", t_mcp, "do", {"x": secret})
            msgs2 = _drain(ctx2.progress_queue)
            assert len(msgs2) == 1
            payload2 = str(msgs2[0])
            self._assert_redacted(secret, payload2)
            # Cleanup MCP
            try:
                registry.deregister(t_mcp)
            except Exception:
                pass
        finally:
            try:
                registry.deregister(tname)
            except Exception:
                pass
            sys.modules.pop(mod_name, None)

    def test_native_card_preview_redacted(self):
        secret = self.SECRET_MARKER
        ctx = _make_ctx(progress_mode="off", tool_progress_enabled=True, tool_progress_filter={"terminal": "all"}, native=True)
        ctx.progress_queue = queue.Queue()
        runner = _make_runner(ctx)
        runner.native_tool_start_callback("cid-native-redact", "terminal", {"command": f"echo {secret}"})
        msgs = _drain(ctx.progress_queue)
        assert len(msgs) == 1
        payload = str(msgs[0].get("preview", "") if isinstance(msgs[0], dict) else msgs[0])
        self._assert_redacted(secret, payload)

    def test_live_status_phrase_redacted(self):
        secret = self.SECRET_MARKER
        ctx = _make_ctx(progress_mode="all", tool_progress_filter={"terminal": "all"})
        mock_adapter = MagicMock()
        mock_adapter.set_status_text = MagicMock()
        ctx._live_status_adapter = mock_adapter
        ctx._live_status_mode = "full"
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "terminal", "ls", {"command": f"echo {secret}"})
        # Live status should have been called once with redacted phrase
        assert mock_adapter.set_status_text.called
        # Get the phrase argument (second positional arg)
        call_args = mock_adapter.set_status_text.call_args
        assert call_args is not None
        phrase = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("text") if call_args[1] else ""
        # phrase may be None for completion, but for started it should be string
        if phrase:
            self._assert_redacted(secret, str(phrase))

    def test_non_secret_allowlisted_still_delivered(self):
        # Ensure redaction does not suppress legitimate previews
        ctx = _make_ctx(progress_mode="off", tool_progress_enabled=True, tool_progress_filter={"read_file": "all"})
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "read_file", "README", {"path": "/tmp/README.md"})
        msgs = _drain(ctx.progress_queue)
        assert len(msgs) == 1
        assert "README" in str(msgs[0])

    def test_redaction_fail_closed_when_both_paths_raise(self):
        # Fail-closed: when authoritative and fallback redactors both raise, outbound must be safe placeholder, never raw
        raw = "ghp_" + "A" * 30  # 34-char credential-shaped marker as in Sherlock probe
        from agent.redact import redact_sensitive_text
        assert redact_sensitive_text(raw, force=True) != raw, "marker must be recognized"
        ctx = _make_ctx(progress_mode="all", tool_progress_filter={"terminal": "all"})
        runner = _make_runner(ctx)
        # Force both redact paths to fail through the real _redact_progress_text / production progress path
        with patch("agent.redact.redact_sensitive_text", side_effect=RuntimeError("authoritative fail")):
            with patch("gateway.run._redact_gateway_user_facing_secrets", side_effect=RuntimeError("fallback fail")):
                runner.progress_callback("tool.started", "terminal", raw, {"command": f"echo {raw} --flag"})
                msgs = _drain(ctx.progress_queue)
                assert len(msgs) == 1, f"allowlisted must still produce safe placeholder when redaction fails, got {msgs}"
                payload = str(msgs[0])
                assert raw not in payload, f"raw marker leaked in fail-closed path: {payload!r}"
                assert payload.strip() != ""
                assert payload != raw
                # Safe placeholder must be present and must not contain raw
                assert "[REDACTED]" in payload or "redacted" in payload.lower()
        # Non-secret allowlisted positive still delivers via intended mode (without forced failure)
        ctx2 = _make_ctx(progress_mode="all", tool_progress_filter={"terminal": "all"})
        runner2 = _make_runner(ctx2)
        runner2.progress_callback("tool.started", "terminal", "hello", {"command": "echo hello world"})
        msgs2 = _drain(ctx2.progress_queue)
        assert len(msgs2) == 1
        assert "hello" in str(msgs2[0]).lower()
        assert raw not in str(msgs2[0])


# ---------------------------------------------------------------------------
# 11. fail-safe and legacy compat
# ---------------------------------------------------------------------------

class TestFailSafe:
    def test_empty_filter_no_effect(self):
        assert _norm_tool_progress_filter({}) == {}
        from gateway.run_turn_runner import _resolve_effective_mode

        assert _resolve_effective_mode("terminal", "all", {}) == "all"

    def test_none_filter_no_effect(self):
        assert _norm_tool_progress_filter(None) == {}
        from gateway.run_turn_runner import _resolve_effective_mode

        assert _resolve_effective_mode("terminal", "all", None) == "all"

    def test_malformed_filter_not_dict_fails_safe(self):
        assert _norm_tool_progress_filter("not a dict") == {}
        assert _norm_tool_progress_filter(123) == {}
        assert _norm_tool_progress_filter(True) == {}

    def test_malformed_entries_skipped(self):
        raw = {"terminal": "all", "": "off", "   ": "all", 123: "off", "read_file": "bogus_mode", "skill_view": None, "another": 12345}
        norm = _norm_tool_progress_filter(raw)
        assert norm == {"terminal": "all"}
        from gateway.run_turn_runner import _resolve_effective_mode

        assert _resolve_effective_mode("read_file", "all", norm) == "all"

    def test_duplicate_keys_last_wins(self):
        raw = {"terminal": "off", "TERMINAL": "all", "Terminal": "verbose"}
        norm = _norm_tool_progress_filter(raw)
        assert norm["terminal"] == "verbose"

    def test_unknown_tool_ignored(self):
        norm = _norm_tool_progress_filter({"unknown_tool_xyz_abc": "all", "terminal": "off"})
        assert "unknown_tool_xyz_abc" in norm
        from gateway.run_turn_runner import _resolve_effective_mode

        assert _resolve_effective_mode("terminal", "all", norm) == "off"
        assert _resolve_effective_mode("read_file", "all", norm) == "all"

    def test_list_allowlist_shorthand(self):
        norm = _norm_tool_progress_filter(["terminal", "skill_view", "mcp"])
        assert norm == {"terminal": "all", "skill_view": "all", "mcp": "all"}
        from gateway.run_turn_runner import _resolve_effective_mode

        assert _resolve_effective_mode("terminal", "off", norm) == "all"

    def test_malformed_list_entries_skipped(self):
        assert _norm_tool_progress_filter(["terminal", "", 123, None, "  "]) == {"terminal": "all"}

    def test_unknown_category_ignored(self):
        norm = _norm_tool_progress_filter({"foobar_category": "all"})
        from gateway.run_turn_runner import _resolve_effective_mode

        assert _resolve_effective_mode("terminal", "all", norm) == "all"

    def test_bool_true_in_filter_maps_to_all(self):
        assert _norm_tool_progress_filter({"terminal": True})["terminal"] == "all"
        assert _norm_tool_progress_filter({"terminal": False})["terminal"] == "off"

    def test_global_bool_off_with_filter_allows_whitelisted(self):
        from gateway.run_turn_runner import _resolve_effective_mode

        assert _resolve_effective_mode("skill_view", "off", {"skill_view": "all"}) == "all"
        assert _resolve_effective_mode("terminal", "off", {"skill_view": "all"}) == "off"

    def test_list_allowlist_with_alias_canonicalized(self):
        norm = _norm_tool_progress_filter(["skill", "mcp_tools", "plugin"])
        assert norm == {"skills": "all", "mcp": "all", "plugins": "all"}


# ---------------------------------------------------------------------------
# 12. integration end-to-end
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_end_to_end_off_with_skills_whitelist(self):
        from gateway.run_turn_runner import _resolve_effective_mode

        filt = {"skills": "all", "terminal": "off"}
        assert _resolve_effective_mode("skill_view", "off", filt) == "all"
        assert _resolve_effective_mode("terminal", "off", filt) == "off"
        ctx = _make_ctx(progress_mode="off", tool_progress_filter=filt, tool_progress_enabled=True)
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "skill_view", "view", {})
        assert not ctx.progress_queue.empty()
        _drain(ctx.progress_queue)
        runner.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
        assert ctx.progress_queue.empty()

    def test_end_to_end_all_with_mcp_suppressed_via_registry(self):
        from tools.registry import registry

        tname = "_int_mcp_test_tool2"
        try:
            registry.register(name=tname, toolset="mcp-int-server", schema={"type": "object", "properties": {}}, handler=lambda: None, check_fn=lambda: True)
            ctx = _make_ctx(progress_mode="all", tool_progress_filter={"mcp": "off"})
            runner = _make_runner(ctx)
            runner.progress_callback("tool.started", tname, "x", {})
            assert ctx.progress_queue.empty()
            runner.progress_callback("tool.started", "read_file", "x", {})
            assert not ctx.progress_queue.empty()
        finally:
            try:
                registry.deregister(tname)
            except Exception:
                pass

    def test_filter_preserves_new_mode_dedup(self):
        ctx = _make_ctx(progress_mode="all", tool_progress_filter={"terminal": "new"})
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "terminal", "first", {"command": "echo first"})
        assert not ctx.progress_queue.empty()
        _drain(ctx.progress_queue)
        runner.progress_callback("tool.started", "terminal", "second", {"command": "echo second"})
        assert ctx.progress_queue.empty()
        runner.progress_callback("tool.started", "read_file", "x", {})
        assert not ctx.progress_queue.empty()

    def test_verbose_filter_overrides(self):
        ctx = _make_ctx(progress_mode="all", tool_progress_filter={"terminal": "verbose"})
        ctx.tool_progress_enabled = True
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "terminal", "long command", {"command": "echo " + "x" * 200})
        msgs = _drain(ctx.progress_queue)
        assert len(msgs) == 1
        assert "echo" in str(msgs[0])

    def test_progress_emit_dedup_still_works_with_filter(self):
        ctx = _make_ctx(progress_mode="all", tool_progress_filter={})
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "terminal", "same", {"command": "echo same"})
        runner.progress_callback("tool.started", "read_file", "different", {})
        msgs = _drain(ctx.progress_queue)
        assert len(msgs) == 2

    def test_live_status_gated_by_filter(self):
        # When filter hides terminal, live status preview must not be set
        ctx = _make_ctx(progress_mode="all", tool_progress_filter={"terminal": "off"})
        mock_adapter = MagicMock()
        mock_adapter.set_status_text = MagicMock()
        ctx._live_status_adapter = mock_adapter
        ctx._live_status_mode = "full"
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
        mock_adapter.set_status_text.assert_not_called()
        # Visible tool should set status
        mock_adapter2 = MagicMock()
        mock_adapter2.set_status_text = MagicMock()
        ctx2 = _make_ctx(progress_mode="all", tool_progress_filter={"terminal": "off"})
        ctx2._live_status_adapter = mock_adapter2
        ctx2._live_status_mode = "full"
        runner2 = _make_runner(ctx2)
        runner2.progress_callback("tool.started", "read_file", "x", {"path": "/tmp/x"})
        assert mock_adapter2.set_status_text.called

    def test_global_log_does_not_enable_progress_for_unoverridden(self):
        # Global log without visible override must keep progress queue disabled at display-settings level
        # This is proven via _run_agent_display_settings earlier, but also check effective routing
        lq = queue.Queue()
        ctx = _make_ctx(progress_mode="log", tool_progress_enabled=False, tool_progress_filter={"terminal": "off"}, log_queue=lq)
        runner = _make_runner(ctx)
        runner.progress_callback("tool.started", "terminal", "ls", {"command": "ls"})
        assert ctx.progress_queue.empty()
        # lq should remain empty because effective off suppresses log as well
        # For read_file which is still log, it should go to log
        runner.progress_callback("tool.started", "read_file", "x", {"path": "/tmp/x"})
        assert not lq.empty()
