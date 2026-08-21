"""Workflow engine plugin — registers the workflow_analyst auxiliary task and loads plugin config.

Plugin config lives at ``~/.hermes/profiles/<profile>/workflow/config.yaml``.
See that file for available settings (auto_approve_extensions, max_nodes_per_workflow, etc.).

The engine invokes the analyst via ``get_text_auxiliary_client("workflow_analyst")``
for three analysis modes: escalation, status summary, and failure diagnosis.

See ``plugins/workflow/analyst.py`` for the auxiliary module.
"""

from __future__ import annotations

import asyncio
import glob as _glob
import json
import logging
import os
import tempfile
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("plugins.workflow")
# Completions directory: alongside workflow files
_wf_dir = os.environ.get("HERMES_WORKFLOW_FILES", "")
if not _wf_dir:
    _wf_dir = str(Path(__file__).resolve().parent.parent.parent / "docs" / "fleet-pipelines")
_COMPLETIONS_DIR = Path(_wf_dir) / "completions"

# ---------------------------------------------------------------------------
# Plugin config loader
# ---------------------------------------------------------------------------

_CONFIG: Dict[str, Any] | None = None

_DEFAULTS: Dict[str, Any] = {
    "auto_discovery": True,
    "auto_approve_extensions": False,
    "auto_approve_template_saves": False,
    "auto_approve_optimizations": False,
    "max_nodes_per_workflow": 256,
    "max_dispatch_per_call": 16,
    "max_extensions_per_workflow": 10,
    "max_nodes_per_extension": 3,
    "default_scope": "project",
    "default_assignee": "",
    "persist_dir": "~/.hermes/workflow-logs",
}


def load_config() -> Dict[str, Any]:
    """Load workflow plugin config from ``~/.hermes/workflows/config.yaml``.

    Returns a dict with defaults merged under any user-set values.
    Caches the result for the lifetime of the process.
    """
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    hermes_home = Path(os.environ.get("HERMES_HOME", "")).expanduser()
    if not hermes_home or not hermes_home.is_dir():
        hermes_home = Path.home() / ".hermes"

    # Try profile-scoped config first, then fall back to shared
    config_paths = [
        hermes_home / "workflows" / "config.yaml",
        Path.home() / ".hermes" / "workflows" / "config.yaml",
    ]

    user_config: Dict[str, Any] = {}
    for path in config_paths:
        if path.is_file():
            try:
                import yaml
                user_config = yaml.safe_load(path.read_text()) or {}
            except Exception as _exc:
                logger.debug("Failed to read user config %s: %s", path, _exc)
            break

    _CONFIG = {**_DEFAULTS, **user_config}
    return _CONFIG


def get_config() -> Dict[str, Any]:
    """Return the cached workflow plugin config.  Loads on first call."""
    return load_config()


def register(ctx):
    """Register workflow tools, the workflow_analyst auxiliary, kanban hooks, and the skill."""
    ctx.register_auxiliary_task(
        key="workflow_analyst",
        display_name="Workflow analyst",
        description="pipeline escalation, status, and failure analysis",
        defaults={
            "timeout": 180,
            "extra_body": {},
        },
    )

    # --- Register the workflow-engine skill ------------------------------------
    skill_path = Path(__file__).parent / "skills" / "workflow-engine" / "SKILL.md"
    if skill_path.exists():
        ctx.register_skill(
            "workflow-engine",
            skill_path,
            "Run DAG-based pipelines via workflow_start",
        )

    # --- Agent-facing workflow tools -------------------------------------------
    from plugins.workflow.tools import (
        check_workflow_requirements,
        handle_workflow_start,
        handle_workflow_view,
        handle_workflow_validate,
        handle_workflow_status,
        handle_workflow_list,
        handle_workflow_show,
        WORKFLOW_START_SCHEMA,
        WORKFLOW_VIEW_SCHEMA,
        WORKFLOW_VALIDATE_SCHEMA,
        WORKFLOW_STATUS_SCHEMA,
        WORKFLOW_LIST_SCHEMA,
        WORKFLOW_SHOW_SCHEMA,
    )

    _TOOLS = [
        (WORKFLOW_START_SCHEMA,    handle_workflow_start),
        (WORKFLOW_VIEW_SCHEMA,     handle_workflow_view),
        (WORKFLOW_VALIDATE_SCHEMA, handle_workflow_validate),
        (WORKFLOW_STATUS_SCHEMA,   handle_workflow_status),
        (WORKFLOW_LIST_SCHEMA,     handle_workflow_list),
        (WORKFLOW_SHOW_SCHEMA,     handle_workflow_show),
    ]

    for schema, handler in _TOOLS:
        ctx.register_tool(
            name=schema["name"],
            toolset="workflow",
            schema=schema,
            handler=handler,
            check_fn=check_workflow_requirements,
        )

    # Register kanban lifecycle hooks to update the job log DB
    ctx.register_hook("kanban_task_completed", _on_kanban_task_completed)
    ctx.register_hook("kanban_task_blocked", _on_kanban_task_blocked)

    # Register pre_gateway_dispatch hook to capture gateway reference
    # for the workflow completion watcher thread
    ctx.register_hook("pre_gateway_dispatch", _capture_gateway)


def _on_kanban_task_completed(*, task_id: str, **kwargs):
    """Update the job log DB when a workflow node card completes."""
    _update_node_card_db(task_id, "done")
    _handle_workflow_node_event(task_id, "done")
    # Check if this completed card is a final-layer card and notify
    _notify_workflow_complete(task_id)


def _on_kanban_task_blocked(*, task_id: str, reason: str = None, **kwargs):
    """Update the job log DB when a workflow node card is blocked."""
    _update_node_card_db(task_id, "blocked")
    _handle_workflow_node_event(task_id, "blocked", reason=reason)


def _update_node_card_db(card_id: str, status: str):
    """Update a node card's status and check if the run is complete."""
    try:
        import sqlite3
        import os
        from pathlib import Path
        from hermes_cli.kanban_db import kanban_home
        db_path = kanban_home() / "workflows" / "executions.db"
        if not db_path.exists():
            return
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "UPDATE workflow_node_cards SET status = ? WHERE card_id = ?",
                (status, card_id)
            )
            # Check if all cards for this run are terminal
            row = conn.execute(
                "SELECT run_id FROM workflow_node_cards WHERE card_id = ?",
                (card_id,)
            ).fetchone()
            if not row:
                return
            run_id = row[0]
            total = conn.execute(
                "SELECT COUNT(*) FROM workflow_node_cards WHERE run_id = ?",
                (run_id,)
            ).fetchone()[0]
            done = conn.execute(
                "SELECT COUNT(*) FROM workflow_node_cards WHERE run_id = ? AND status IN ('done','failed')",
                (run_id,)
            ).fetchone()[0]
            if done >= total:
                has_failed = conn.execute(
                    "SELECT COUNT(*) FROM workflow_node_cards WHERE run_id = ? AND status = 'failed'",
                    (run_id,)
                ).fetchone()[0]
                final = "failed" if has_failed > 0 else "completed"
                from datetime import datetime, timezone
                conn.execute(
                    "UPDATE workflow_executions SET status = ?, finished_at = ? WHERE run_id = ?",
                    (final, datetime.now(timezone.utc).isoformat(), run_id)
                )
    except Exception:
        pass  # Non-fatal — state files still work


# ---------------------------------------------------------------------------
# Workflow node event handler — loop logic and layer advancement
# ---------------------------------------------------------------------------

def _find_state_for_card(task_id: str):
    """Find the workflow state file that contains this task_id.

    Returns (state_dict, state_file_path) or None.
    """
    # State files live in .engine-state/ under the workflow files directory
    wf_dir = os.environ.get("HERMES_WORKFLOW_FILES", "")
    if not wf_dir:
        wf_dir = str(Path(__file__).resolve().parent.parent.parent / "docs" / "fleet-pipelines")
    state_dir = Path(wf_dir) / ".engine-state"
    if not state_dir.exists():
        return None
    for state_file in sorted(state_dir.glob("*_state.json"), reverse=True):
        try:
            import json
            state = json.loads(state_file.read_text())
            states = state.get("states", {})
            for nid, node_state in states.items():
                if node_state.get("kanban_card_id") == task_id:
                    return (state, str(state_file))
        except Exception:
            continue
    return None


def _find_verify_nodes(workflow_name: str):
    """Find verify→revision mappings for a workflow.

    Returns {verify_node_id: revision_node_id} — nodes where a
    revision node depends on the verify node (loop pattern).
    """
    wf_dir = os.environ.get("HERMES_WORKFLOW_FILES", "")
    if not wf_dir:
        wf_dir = str(Path(__file__).resolve().parent.parent.parent / "docs" / "fleet-pipelines")
    wf_files = Path(wf_dir)
    import yaml
    wf_path = wf_files / f"{workflow_name}.yaml"
    if not wf_path.exists():
        return {}
    wf = yaml.safe_load(wf_path.read_text())
    nodes = wf.get("nodes", {})
    verify_map = {}
    for name, node in nodes.items():
        name_lower = name.lower()
        if "revise" in name_lower:
            for dep in node.get("depends_on", []):
                if dep in nodes:
                    verify_map[dep] = name
    return verify_map


def _handle_workflow_node_event(task_id: str, status: str, reason: str = None):
    """Handle a workflow node card completion or block event.

    Core loop mechanism:
    - BLOCK of a verify node: enrich the implementer's card with the
      failure report and re-dispatch (loop).
    - COMPLETION: check if the layer is done and advance.
    """
    try:
        result = _find_state_for_card(task_id)
        if result is None:
            return  # Not a workflow card
        state, state_path = result
        workflow_name = state.get("workflow_name", "")
        layers = state.get("layers", [])
        states = state.get("states", {})
        loop_counts = state.get("loop_counts", {})
        max_loops = state.get("max_revision_loops", 3)

        # Find which node this card belongs to
        node_id = None
        for nid, ns in states.items():
            if ns.get("kanban_card_id") == task_id:
                node_id = nid
                break
        if not node_id:
            return

        verify_map = _find_verify_nodes(workflow_name)

        if status == "blocked" and node_id in verify_map:
            # This is a verify node that blocked — LOOP
            revision_node = verify_map[node_id]
            loop_key = f"{node_id}:{revision_node}"
            current_loop = loop_counts.get(loop_key, 0)

            if current_loop >= max_loops:
                print(f"   🚫 Workflow loop exceeded max ({max_loops}) — escalating")
                states[node_id]["status"] = "escalated"
                states[node_id]["error"] = f"Exceeded {max_loops} revision loops"
                _save_state_file(state_path, state)
                return

            # Find the implementer's card (the node this verify depends on)
            wf_dir = os.environ.get("HERMES_WORKFLOW_FILES", "")
            if not wf_dir:
                wf_dir = str(Path(__file__).resolve().parent.parent.parent / "docs" / "fleet-pipelines")
            wf_files = Path(wf_dir)
            import yaml
            wf_path = wf_files / f"{workflow_name}.yaml"
            if not wf_path.exists():
                return
            wf = yaml.safe_load(wf_path.read_text())
            verify_node_def = wf.get("nodes", {}).get(node_id, {})
            node_deps = verify_node_def.get("depends_on", [])

            # The implementer is the first dependency that isn't a revision node
            implementer_nid = None
            for dep in node_deps:
                if dep != revision_node:
                    implementer_nid = dep
                    break
            if not implementer_nid and node_deps:
                implementer_nid = node_deps[0]

            if not implementer_nid:
                return

            impl_state = states.get(implementer_nid, {})
            impl_card_id = impl_state.get("kanban_card_id")
            if not impl_card_id:
                return

            # Get the failure report from the blocked card
            kb, conn = _get_board_conn(state)
            if conn is None:
                logger.error("kanban_board missing from state file — cannot process hook event")
                return
            try:
                blocked_card = kb.get_task(conn, task_id)
                failure_report = blocked_card.body if blocked_card else (reason or "Unknown failure")

                # Enrich the implementer's card with the failure report
                card = kb.get_task(conn, impl_card_id)
                if card:
                    original_body = card.body or ""
                    enriched_body = (
                        f"{original_body}\n\n"
                        f"## LOOP #{current_loop + 1} — Revision Required\n\n"
                        f"The review failed. Here is the failure report:\n\n"
                        f"{failure_report}\n\n"
                        f"Fix the issues above and try again."
                    )
                    conn.execute(
                        "UPDATE tasks SET body = ?, status = 'ready' WHERE id = ?",
                        (enriched_body, impl_card_id),
                    )
                    conn.commit()
            except Exception:
                failure_report = reason or "Unknown failure"
            finally:
                conn.close()

            # Increment loop count
            loop_counts[loop_key] = current_loop + 1
            state["loop_counts"] = loop_counts
            states[node_id]["status"] = "looping"
            states[node_id]["loop_count"] = current_loop + 1
            _save_state_file(state_path, state)
            print(f"   ↩  LOOP #{current_loop + 1}: {implementer_nid} re-dispatched with failure report")

        elif status == "done":
            # Update this node's status in the state file
            states[node_id]["status"] = "done"
            from datetime import datetime, timezone
            states[node_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            _save_state_file(state_path, state)

            # Check if all nodes in the current layer are done
            current_layer = state.get("current_layer", 0)
            if current_layer >= len(layers):
                return
            layer_nodes = layers[current_layer]
            # Check actual card status from kanban DB, not just state file
            kb, conn = _get_board_conn(state)
            if conn is None:
                logger.error("kanban_board missing from state file — cannot process hook event")
                return
            try:
                all_done = True
                for nid in layer_nodes:
                    ns = states.get(nid, {})
                    card_id = ns.get("kanban_card_id")
                    if not card_id:
                        all_done = False
                        continue
                    card = kb.get_task(conn, card_id)
                    if not card or card.status != "done":
                        all_done = False
                        break
                if all_done and node_id in layer_nodes:
                    # Update all node statuses in state file from kanban DB
                    for nid in layer_nodes:
                        ns = states.get(nid, {})
                        card_id = ns.get("kanban_card_id")
                        if card_id:
                            card = kb.get_task(conn, card_id)
                            if card:
                                ns["status"] = card.status
                    state["current_layer"] = current_layer + 1
                    _save_state_file(state_path, state)
                    print(f"   ✓ Layer {current_layer} complete — advancing to {current_layer + 1}")

                    # Spawn supervisor to create next layer's cards
                    _spawn_supervisor_for_next_layer(state, state_path)
            finally:
                conn.close()

    except Exception as e:
        print(f"   ⚠  Workflow event handler error: {e}")


def _save_state_file(path, state):
    """Persist the workflow state file."""
    import json
    from datetime import datetime, timezone
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    # Ensure session_info is always present — inject from tool handler's
    # temp file if missing. The engine's ContextVar-based capture doesn't
    # survive USR1 restarts, but the tool handler writes to /tmp.
    if not state.get("session_info"):
        try:
            with open("/tmp/wfe-session.json") as _f:
                _si = json.load(_f)
            if _si and _si.get("platform"):
                state["session_info"] = _si
        except Exception:
            pass
    Path(path).write_text(json.dumps(state, indent=2, default=str))


def _spawn_supervisor_for_next_layer(state, state_path):
    """Spawn the supervisor subprocess to create cards for the next layer.

    Called by the kanban hook when a layer completes. The supervisor
    creates cards for the next layer and returns — no monitoring loop.
    """
    try:
        import subprocess
        import sys
        workflow_name = state.get("workflow_name", "")
        run_id = state.get("run_id", "")
        # Fallback: extract from filename if not in state dict
        if not run_id and state_path:
            sf_name = Path(state_path).stem  # remove .json
            parts = sf_name.split("_", 1)
            if len(parts) > 1:
                run_id = parts[1]  # {run_id}_state → {run_id}
        board = state.get("kanban_board")
        if not board:
            return
        current_layer = state.get("current_layer", 0)
        layers = state.get("layers", [])

        if current_layer >= len(layers):
            print(f"   ✓ All layers complete — workflow done")
            return

        # Check if a supervisor is already running for this workflow
        if run_id and run_id != "NONE":
            existing = subprocess.run(
                ["pgrep", "-f", f"workflow_engine.*{workflow_name}.*{run_id}"],
                capture_output=True, text=True
            )
            if existing.stdout.strip():
                print(f"   ℹ  Supervisor already running for {workflow_name} (run {run_id}) — skipping spawn")
                return

        # Spawn supervisor to create next layer's cards
        cmd = [
            sys.executable, "-m", "tools.workflow_engine",
            "start", workflow_name,
            "--resume",
            "--board", board,
            "--run-id", run_id,
        ]
        env = os.environ.copy()
        env["HERMES_WORKFLOW_FILES"] = os.environ.get(
            "HERMES_WORKFLOW_FILES",
            str(Path(__file__).resolve().parent.parent.parent / "docs" / "fleet-pipelines"),
        )
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        print(f"   👤 supervisor spawned for layer {current_layer}")
    except Exception as e:
        print(f"   ⚠  Failed to spawn supervisor: {e}")


# ---------------------------------------------------------------------------
# Direct session injection — completion notification via /tmp markers
# ---------------------------------------------------------------------------


def _notify_workflow_complete(task_id: str, state=None):
    """Write a completion marker to /tmp for the watcher thread to inject.

    When a final-layer card completes, reads the state file for session info
    and writes a JSON marker that the polling daemon picks up and injects
    into the correct session.  The message body is produced by the workflow
    analyst auxiliary so the report is actionable — not a hardcoded string.
    """
    try:
        if state is None:
            result = _find_state_for_card(task_id)
            if result is None:
                return
            state, _ = result
        session_info = state.get("session_info", {})
        if not session_info.get("platform") or not session_info.get("chat_id"):
            return

        layers = state.get("layers", [])
        if not layers:
            return

        final_layer_nids = layers[-1]
        # Check if all final-layer nodes are done
        all_final_done = all(
            state.get("states", {}).get(nid, {}).get("status") == "done"
            for nid in final_layer_nids
        )
        if not all_final_done:
            return

        # All done — produce analyst report, then write marker
        workflow_name = state.get("workflow_name", "unknown")
        session_key = session_info.get("session_key", "")

        all_nodes = []
        for layer_nids in layers:
            for nid in layer_nids:
                ns = state.get("states", {}).get(nid, {})
                all_nodes.append({
                    "node": nid,
                    "agent": ns.get("agent", ""),
                    "status": ns.get("status", "unknown"),
                    "summary": ns.get("summary", ""),
                })

        # Enrich with task_runs summaries from kanban DB
        try:
            from hermes_cli.kanban_db import connect_closing
            board = state.get("kanban_board", "")
            if board:
                with connect_closing(board=board) as db_conn:
                    rows = db_conn.execute(
                        "SELECT task_id, summary FROM task_runs WHERE outcome='completed' "
                        "ORDER BY ended_at DESC"
                    ).fetchall()
                    summary_map = {}
                    for tid, s in rows:
                        if s and tid not in summary_map:
                            summary_map[tid] = s
                    # Match summaries to nodes by card_id
                    for node in all_nodes:
                        card_id = state.get("states", {}).get(node["node"], {}).get("kanban_card_id", "")
                        if card_id and card_id in summary_map:
                            node["summary"] = summary_map[card_id]
        except Exception as _summary_exc:
            logger.debug("summary enrichment failed: %s", _summary_exc)

        # Count stats
        done_count = sum(1 for n in all_nodes if n["status"] == "done")
        failed_count = sum(1 for n in all_nodes if n["status"] in ("failed", "timed_out"))
        total = len(all_nodes)

        # Try to get analyst-generated report
        message = ""
        try:
            from plugins.workflow.analyst import analyze_status
            state_json = json.dumps(state, indent=2, default=str)
            outcome = analyze_status(
                pipeline_name=workflow_name,
                state_json=state_json,
                timeout=30,
            )
            if outcome.success and outcome.result:
                # The analyst returns structured JSON — format into readable
                r = outcome.result
                parts = []
                # Layer summary
                for layer_info in r.get("layer_summary", []):
                    for node_info in layer_info.get("nodes", []):
                        status_label = {
                            "done": "✅", "running": "⏳", "pending": "⬜",
                            "failed": "❌", "blocked": "🚫", "timed_out": "⏰",
                        }.get(node_info.get("status", ""), "❓")
                        agent = node_info.get("agent", "")
                        node_name = node_info.get("node", "")
                        parts.append(f"  {status_label} {node_name} ({agent})")
                # Attention needed
                attention = r.get("attention_needed", [])
                if attention:
                    parts.append("")
                    parts.append("⚠️ Attention:")
                    for a in attention:
                        parts.append(f"  • {a}")
                message = "\n".join(parts)
        except Exception as _analyst_exc:
            logger.debug("workflow analyst unavailable for completion report: %s", _analyst_exc)

        # Build message with summaries
        if not message:
            lines = []
            for n in all_nodes:
                lines.append(f"  {n['node']} ({n['agent']}): {n['status']}")
                if n.get("summary"):
                    for sline in n["summary"].split("\n")[:5]:
                        lines.append(f"    {sline}")
            message = "\n".join(lines)

        # Build the full notification
        heading = f"Workflow '{workflow_name}' completed on board '{board}' — {done_count}/{total} nodes succeeded"
        if failed_count:
            heading += f" ({failed_count} failed)"
        full_message = f"{heading}\n\nNodes:\n{message}"

        marker = {
            "session_key": session_key,
            "platform": session_info.get("platform", ""),
            "chat_id": session_info.get("chat_id", ""),
            "thread_id": session_info.get("thread_id"),
            "user_id": session_info.get("user_id"),
            "profile": session_info.get("profile"),
            "workflow_name": workflow_name,
            "board": board,
            "status": "completed",
            "message": full_message,
            "nodes": all_nodes,
            "run_id": state.get("run_id", ""),
        }
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")
        run_id = state.get("run_id", "unknown")
        wf_marker_dir = _COMPLETIONS_DIR / workflow_name
        wf_marker_dir.mkdir(parents=True, exist_ok=True)
        marker_path = wf_marker_dir / f"{ts}_{run_id}.json"
        # Atomic write: temp file → rename (prevents TOCTOU reads of partial JSON)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(tmp_fd, "w") as tmp_f:
                json.dump(marker, tmp_f, indent=2, default=str)
            os.rename(tmp_path, str(marker_path))
        except Exception:
            os.unlink(tmp_path)
            raise
        print(f"   Workflow completion marker written: {marker_path.relative_to(_COMPLETIONS_DIR)}")
    except Exception as e:
        print(f"   ⚠  Failed to write completion marker: {e}")


# Module-level gateway reference, captured via pre_gateway_dispatch hook
_gateway_ref = None
_watcher_started = False


def _capture_gateway(**kwargs):
    global _gateway_ref, _watcher_started
    if _watcher_started:
        return None




    gw = kwargs.get("gateway")
    if gw is not None:
        _gateway_ref = gw
        if not _watcher_started:
            _watcher_started = True
            _start_completion_watcher()
    return None  # Don't interfere with the dispatch flow


def _start_completion_watcher():
    """Start a daemon thread that polls /tmp for workflow completion markers.

    When a marker is found, builds a synthetic MessageEvent with the correct
    session key (including chat_type="thread" for Discord threads) and injects
    it into the gateway.
    """
    import threading

    def _watcher_loop():

        while True:
            time.sleep(2)
            try:
                markers = sorted(_glob.glob(str(_COMPLETIONS_DIR / "*" / "*.json")))
                # Skip stale markers older than 10 minutes
                now = time.time()
                markers = [m for m in markers if now - os.path.getmtime(m) < 600]
                for marker_path_str in markers:
                    try:
                        marker_path = Path(marker_path_str)
                        data = json.loads(marker_path.read_text())

                        # Skip already-processed markers (kept as job logs)
                        if data.get("processed_at"):
                            continue

                        # Only process markers for this gateway's profile
                        _my_profile = (
                            os.environ.get("HERMES_PROFILE")
                            or (Path(os.environ.get("HERMES_HOME", "")).name
                                if "profiles/" in os.environ.get("HERMES_HOME", "")
                                else "default")
                        )
                        _marker_profile = data.get("profile") or "default"
                        if _marker_profile != _my_profile:
                            continue

                        platform_str = data.get("platform", "")
                        chat_id = data.get("chat_id", "")
                        thread_id = data.get("thread_id")
                        user_id = data.get("user_id")
                        profile = data.get("profile")
                        message = data.get("message", "Workflow completed")

                        if not platform_str or not chat_id:
                            marker_path.unlink(missing_ok=True)
                            continue

                        # Derive correct chat_type from thread_id
                        chat_type = "thread" if thread_id else "group"

                        # Build session key if not already captured
                        session_key = data.get("session_key", "")

                        gw = _gateway_ref
                        if gw is None:
                            continue

                        # Import gateway types
                        from gateway.session import SessionSource, build_session_key
                        from gateway.platforms.base import MessageEvent, MessageType
                        from gateway.config import Platform

                        # Resolve platform enum
                        try:
                            platform = Platform(platform_str)
                        except ValueError:
                            logger.warning(
                                "wf-completion watcher: unknown platform %s, skipping marker",
                                platform_str,
                            )
                            marker_path.unlink(missing_ok=True)
                            continue

                        # Build the source with correct chat_type
                        source = SessionSource(
                            platform=platform,
                            chat_id=chat_id,
                            chat_type=chat_type,
                            thread_id=thread_id,
                            user_id=user_id,
                            profile=profile,
                        )

                        # If no session_key was captured, build one
                        if not session_key:
                            session_key = build_session_key(source, profile=profile)

                        # Create synthetic message event
                        synth_event = MessageEvent(
                            text=message,
                            message_type=MessageType.TEXT,
                            source=source,
                            internal=True,
                        )

                        # Find the adapter and inject (use .value comparison
                        # like the gateway's own _inject_watch_notification)
                        adapter = None
                        for _p, _a in gw.adapters.items():
                            if _p.value == platform.value:
                                adapter = _a
                                break
                        if adapter is None:
                            logger.warning(
                                "wf-completion watcher: no adapter for platform %s",
                                platform_str,
                            )
                            marker_path.unlink(missing_ok=True)
                            continue

                        from agent.async_utils import safe_schedule_threadsafe
                        loop = getattr(gw, "_gateway_loop", None)
                        fut = safe_schedule_threadsafe(
                            adapter.handle_message(synth_event), loop,
                            logger=logger,
                            log_message="wf-completion watcher: failed to schedule injection",
                        )
                        if fut is not None:
                            try:
                                fut.result(timeout=10)
                            except Exception as _fut_exc:
                                logger.warning(
                                    "wf-completion watcher: injection timed out or failed: %s",
                                    _fut_exc,
                                )
                                marker_path.unlink(missing_ok=True)
                                continue

                        logger.info(
                            "wf-completion watcher: injected notification into %s/%s profile=%s",
                            platform_str, chat_id, profile or "default",
                        )
                        # Mark as processed — keep the file as a job log
                        data["processed_at"] = datetime.now(timezone.utc).isoformat()
                        marker_path.write_text(json.dumps(data, indent=2, default=str))

                    except Exception as _proc_exc:
                        logger.warning(
                            "wf-completion watcher: failed to process %s: %s",
                            marker_path_str, _proc_exc,
                        )
                        # Clean up marker on error
                        try:
                            Path(marker_path_str).unlink(missing_ok=True)
                        except Exception:
                            pass
            except Exception:
                pass  # Non-fatal: next poll cycle will retry

    t = threading.Thread(target=_watcher_loop, daemon=True, name="wf-completion-watcher")
    t.start()
    print("   👁  Workflow completion watcher started")
