"""Dynamic workflow bridge — fleet integration layer.

Wraps the in-memory dynamic engine (``dynamic.py``) with:
  - Kanban card lifecycle (create on project-scope, complete on record)
  - Optional JSON persistence for durable scopes
  - Cost guards (max nodes, max dispatch, single-flight)

This module does NOT re-implement DAG mechanics.  It delegates all
graph logic to ``handle_workflow_dynamic`` in ``dynamic.py``.

Public entry point:
    ``run_dynamic_workflow(...)`` — single call that creates a workflow,
    creates kanban cards, and returns the result.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from plugins.workflow import get_config

logger = logging.getLogger(__name__)

# ── Constants (from plugin config) ──────────────────────────────────

def _max_nodes() -> int:
    return get_config().get("max_nodes_per_workflow", 256)

def _max_dispatch() -> int:
    return get_config().get("max_dispatch_per_call", 16)

def _default_scope() -> str:
    return get_config().get("default_scope", "project")

def _auto_approve(key: str) -> bool:
    return bool(get_config().get(key, False))

def _auto_discover() -> bool:
    return bool(get_config().get("auto_discovery", True))

VALID_SCOPES = frozenset({"project", "global", "durable"})

KANBAN_BOARD = "dynamic-workflows"

# ── Hermes binary resolution ──────────────────────────────────────
# Self-contained — mirrors the logic in engine.py's hermes_binary()
# but does NOT import engine.py (bridge is independent).


from plugins.workflow.utils import hermes_binary


# ── Kanban helpers ────────────────────────────────────────────────


def _kanban_create_card(node: dict, workflow_id: str, context: str = "", board: str = "") -> str | None:
    """Create a kanban card for a worker node.  Returns the card ID
    on success, None on failure.

    Follows the pattern from engine.py's ``create_kanban_card`` method.

    ``board`` overrides the kanban board.  When empty, uses the
    module-level ``KANBAN_BOARD`` constant.
    """
    node_id = node.get("node_id", "")
    goal = node.get("goal", "")
    title = f"[{node_id}] dynamic: {goal[:60]}"

    body = goal
    if context:
        body += f"\n\nContext: {context}"

    kanban_board = board if board else KANBAN_BOARD
    _default_assignee = get_config().get("default_assignee", "")
    cmd = [
        hermes_binary(), "kanban", "create",
        title,
        "--tenant", kanban_board,
        "--body", body,
        "--goal",
        "--priority", "2",
    ]
    if _default_assignee:
        cmd.extend(["--assignee", _default_assignee])
    run_env = dict(os.environ)
    run_env["HERMES_KANBAN_BOARD"] = kanban_board
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, env=run_env,
        )
        if result.returncode != 0:
            logger.warning(
                "kanban create failed for node %s: %s", node_id, result.stderr,
            )
            return None
        # Parse card ID from output — try JSON first, then regex.
        out = result.stdout.strip()
        try:
            card_obj = json.loads(out)
            return card_obj.get("id") or card_obj.get("card_id")
        except (json.JSONDecodeError, TypeError):
            pass
        # Fallback: look for "Created card <id>" line.
        for line in out.splitlines():
            if "created card" in line.lower():
                parts = line.split()
                if len(parts) >= 3:
                    return parts[-1]
        return None
    except Exception as exc:
        logger.warning("kanban create subprocess error for node %s: %s", node_id, exc)
        return None


def _kanban_complete_card(card_id: str, summary: str = "") -> bool:
    """Mark a kanban card as complete.  Returns True on success."""
    cmd = [hermes_binary(), "kanban", "complete", card_id]
    if summary:
        cmd.extend(["--result", summary[:500]])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.warning("kanban complete failed for %s: %s", card_id, result.stderr)
            return False
        return True
    except Exception as exc:
        logger.warning("kanban complete subprocess error for %s: %s", card_id, exc)
        return False


# ── Persistence helpers ───────────────────────────────────────────


def _state_dir(workflow_id: str) -> Path:
    """Return the state directory for a durable workflow."""
    base = Path.home() / ".hermes" / "workflow-logs" / workflow_id
    base.mkdir(parents=True, exist_ok=True)
    return base


def _state_path(workflow_id: str) -> Path:
    """Path to the state JSON file for a durable workflow."""
    return _state_dir(workflow_id) / "state.json"


def _save_state(workflow_id: str, wf_view: dict) -> None:
    """Persist workflow node state to disk (durable scope only)."""
    nodes = wf_view.get("nodes", [])
    cards_map = wf_view.get("cards", {})  # {node_id: card_id}
    state = {
        "workflow_id": workflow_id,
        "objective": wf_view.get("objective", ""),
        "context": wf_view.get("context", ""),
        "scope": wf_view.get("scope", "project"),
        "cards": cards_map,
        "nodes": {},
        "created_at": wf_view.get("created_at", datetime.now(timezone.utc).isoformat()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    for n in nodes:
        state["nodes"][n["node_id"]] = {
            "goal": n.get("goal", ""),
            "depends_on": n.get("depends_on", []),
            "status": n.get("status", "pending"),
            "summary": n.get("summary"),
            "error": n.get("error"),
            "delegation_id": n.get("delegation_id"),
        }
    path = _state_path(workflow_id)
    try:
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as exc:
        logger.warning("failed to save durable state for %s: %s", workflow_id, exc)


def _recover_workflow(workflow_id: str) -> "DynamicWorkflow | None":
    """Recover a workflow from its saved state and kanban card statuses.

    Returns a ``DynamicWorkflow`` instance (not a dict), or None if no
    state found.  Callers that store the result in ``_workflows`` expect
    a live object with methods like ``.status``, not a plain dict.
    """
    state = _load_state(workflow_id)
    if not state:
        return None

    from plugins.workflow.dynamic import DynamicWorkflow, DynamicNode

    wf = DynamicWorkflow(
        workflow_id=workflow_id,
        objective=state.get("objective", ""),
        context=state.get("context", ""),
        scope_key=state.get("scope", "project"),
    )

    cards_map = state.get("cards", {})
    for node_id, card_id in cards_map.items():
        # Look up the card's current status from the kanban DB
        card_status = _get_kanban_card_status(card_id)
        node_state = state.get("nodes", {}).get(node_id, {})
        node = DynamicNode(
            node_id=node_id,
            goal=node_state.get("goal", ""),
            depends_on=node_state.get("depends_on", []),
            status=_map_card_status(card_status),
            summary=node_state.get("summary"),
            error=node_state.get("error"),
            delegation_id=node_state.get("delegation_id"),
        )
        wf.nodes[node_id] = node
        wf.node_order.append(node_id)

    return wf


def _get_kanban_card_status(card_id: str) -> str:
    """Look up a kanban card's current status. Returns 'unknown' if card not found."""
    try:
        result = subprocess.run(
            [hermes_binary(), "kanban", "show", card_id, "--json"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return data.get("status", "unknown")
    except Exception:
        pass
    return "unknown"


# "blocked" → "failed" because blocked cards wait on upstream failures
# or manual intervention. Treating them as pending causes infinite redispatch.
_STATUS_MAP = {
    "ready": "pending",
    "in_progress": "running",
    "done": "completed",
    "failed": "failed",
    "blocked": "failed",  # blocked cards are not redispatched
    "skipped": "skipped",
    "unknown": "pending",
}

def _map_card_status(card_status: str) -> str:
    """Map kanban card status to workflow node status."""
    return _STATUS_MAP.get(card_status, "pending")


def _load_state(workflow_id: str) -> dict | None:
    """Load persisted state if it exists.  Returns None if missing."""
    path = _state_path(workflow_id)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ── Cost guards ───────────────────────────────────────────────────


def _check_single_flight(workflow_id: str) -> str | None:
    """Check if a workflow with the given ID is already in progress.

    Returns an error message if a conflict exists, None if clear.
    Uses the engine's module-level ``_workflows`` dict to check scope.
    """
    try:
        from plugins.workflow.dynamic import _workflows, _workflows_lock, WF_RUNNING, WF_READY, WF_WAITING
        with _workflows_lock:
            for (scope_key, wf_id), wf in _workflows.items():
                if wf_id == workflow_id:
                    status = wf.status
                    if status in (WF_RUNNING, WF_READY, WF_WAITING):
                        return (
                            f"single_flight: workflow '{workflow_id}' "
                            f"is already in progress (status: {status})"
                        )
    except ImportError:
        pass
    return None


def _validate_nodes_count(nodes: list) -> str | None:
    """Reject workflows exceeding the node cap.  Returns error or None."""
    if len(nodes) > _max_nodes():
        return (
            f"too many nodes: {len(nodes)} exceeds cap of "
            f"{_max_nodes()}"
        )
    return None


def _validate_dispatch_count(count: int) -> int:
    """Clamp dispatch count to the per-call cap."""
    return max(1, min(count, _max_dispatch()))


# ── Active run tracking ───────────────────────────────────────────

# Simple in-memory set of workflow_ids with kanban cards created.
# Used to avoid duplicate card creation and to track which cards to
# complete on record.
_kanban_card_map: dict[str, dict[str, str]] = {}
# workflow_id -> {node_id: card_id}



# ── Public entry point ────────────────────────────────────────────


def run_dynamic_workflow(
    workflow_id: str,
    objective: str,
    nodes: list[dict],
    context: str = "",
    scope: str = "project",
    single_flight: bool = False,
    dispatch_ready: bool = True,
    board: str = "",
) -> dict:
    """Create and optionally dispatch a dynamic workflow with fleet integration.

    Args:
        workflow_id: Unique identifier for the workflow.
        objective: High-level goal description.
        nodes: List of node dicts with at least ``node_id`` and ``goal``.
        context: Optional context string for all nodes.
        scope: One of "project", "global", or "durable".
        single_flight: If True, refuse creation when a same-id run is active.
        dispatch_ready: If True, dispatch ready nodes immediately.
        board: Kanban board override. When empty, uses the bridge default
               ("dynamic-workflows").

    Returns:
        The workflow result dict from the engine (parsed from JSON).
    """
    import json as _json

    from plugins.workflow.dynamic import handle_workflow_dynamic

    # Validate scope
    if scope not in VALID_SCOPES:
        return {"ok": False, "error": f"invalid scope: {scope!r}; must be one of {sorted(VALID_SCOPES)}"}

    # Validate node count
    node_err = _validate_nodes_count(nodes)
    if node_err:
        return {"ok": False, "error": node_err}

    # Single-flight check
    if single_flight:
        sf_err = _check_single_flight(workflow_id)
        if sf_err:
            return {"ok": False, "error": sf_err, "hint": "wait for the current run to finish"}

    # Clamp dispatch count
    max_dispatch = _validate_dispatch_count(_max_dispatch())

    # Build the engine args
    engine_args = {
        "action": "create",
        "workflow_id": workflow_id,
        "objective": objective,
        "nodes": nodes,
        "context": context,
        "dispatch_ready": dispatch_ready,
        "max_dispatch": max_dispatch,
    }

    # Delegate to the engine
    result_json = handle_workflow_dynamic(engine_args)
    try:
        result = _json.loads(result_json) if isinstance(result_json, str) else result_json
    except (_json.JSONDecodeError, TypeError):
        return {"ok": False, "error": f"failed to parse engine result: {result_json}"}

    if not result.get("ok"):
        return result

    # Extract workflow view from engine result
    wf_view = result.get("workflow", {})
    actual_id = wf_view.get("workflow_id", workflow_id)

    # Kanban integration for project scope
    if scope == "project":
        wf_nodes = wf_view.get("nodes", [])
        card_map: dict[str, str] = {}
        for node in wf_nodes:
            card_id = _kanban_create_card(node, actual_id, context, board)
            if card_id:
                card_map[node["node_id"]] = card_id
                node["kanban_card_id"] = card_id
        _kanban_card_map[actual_id] = card_map
        logger.info(
            "created %d kanban cards for workflow %s",
            len(card_map), actual_id,
        )

    # Durable persistence — write initial state
    if scope == "durable":
        _save_state(actual_id, wf_view)

    result["workflow"] = wf_view
    return result


def record_node(
    workflow_id: str,
    node_id: str,
    status: str,
    summary: str = "",
    error: str = "",
    result: Any = None,
    context: Any = None,
    scope: str = "project",
) -> dict:
    """Record a node's result and handle side effects.

    Wraps the engine's ``record`` action with kanban completion
    and durable persistence.

    Returns the engine result dict.
    """
    import json as _json

    from plugins.workflow.dynamic import handle_workflow_dynamic

    engine_args = {
        "action": "record",
        "workflow_id": workflow_id,
        "node_id": node_id,
        "status": status,
        "summary": summary,
    }
    if error:
        engine_args["error"] = error
    if result is not None:
        engine_args["result"] = result

    result_json = handle_workflow_dynamic(engine_args)
    try:
        res = _json.loads(result_json) if isinstance(result_json, str) else result_json
    except (_json.JSONDecodeError, TypeError):
        return {"ok": False, "error": f"failed to parse engine result: {result_json}"}

    if not res.get("ok"):
        return res

    wf_view = res.get("workflow", {})

    # Kanban completion for project scope
    if scope == "project":
        card_map = _kanban_card_map.get(workflow_id, {})
        card_id = card_map.get(node_id)
        if card_id:
            card_summary = summary or (f"completed: {node_id}" if status == "completed" else f"{status}: {node_id}")
            _kanban_complete_card(card_id, card_summary)

    # Durable persistence
    if scope == "durable":
        _save_state(workflow_id, wf_view)

    return res


def dispatch_nodes(
    workflow_id: str,
    max_dispatch: int = _max_dispatch(),
    scope: str = "project",
) -> dict:
    """Manually trigger dispatch of ready nodes.

    Wraps the engine's ``dispatch`` action with dispatch count capping,
    kanban creation, and durable persistence.
    """
    import json as _json

    from plugins.workflow.dynamic import handle_workflow_dynamic

    capped = _validate_dispatch_count(max_dispatch)

    engine_args = {
        "action": "dispatch",
        "workflow_id": workflow_id,
        "max_dispatch": capped,
    }

    result_json = handle_workflow_dynamic(engine_args)
    try:
        res = _json.loads(result_json) if isinstance(result_json, str) else result_json
    except (_json.JSONDecodeError, TypeError):
        return {"ok": False, "error": f"failed to parse engine result: {result_json}"}

    if not res.get("ok"):
        return res

    wf_view = res.get("workflow", {})

    # Kanban creation for dispatched nodes in project scope
    if scope == "project":
        dispatched_list = res.get("dispatched", [])
        card_map = _kanban_card_map.setdefault(workflow_id, {})
        for d in dispatched_list:
            nid = d.get("node_id", "")
            if nid and nid not in card_map:
                # Find the node data from the workflow view
                for n in wf_view.get("nodes", []):
                    if n.get("node_id") == nid:
                        card_id = _kanban_create_card(n, workflow_id)
                        if card_id:
                            card_map[nid] = card_id
                            n["kanban_card_id"] = card_id
                        break

    # Durable persistence
    if scope == "durable":
        _save_state(workflow_id, wf_view)

    return res


# ── Internal helpers ──────────────────────────────────────────────


def _append_extension_artifact(workflow_id: str, node_id: str, node_summary: str, suggestions: list, auto_approved: bool) -> None:
    """Append extension suggestions to the workflow's audit artifact.

    Called every time analyze_extension returns suggestions, regardless of
    whether auto_approve_extensions is true. Provides permanent audit trail.
    """
    artifact_dir = Path.home() / ".hermes" / "workflow-logs" / workflow_id
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_file = artifact_dir / "extensions.jsonl"
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "workflow_id": workflow_id,
            "node_id": node_id,
            "node_summary": node_summary,
            "suggestions": suggestions,
            "auto_approved": auto_approved,
        }
        with open(artifact_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as exc:
        logger.warning("failed to write extension artifact for %s: %s", workflow_id, exc)
