"""
Dynamic Workflow Engine — model-authored DAG execution.

Unlike the pre-defined engine (``engine.py``) which reads YAML pipeline
definitions, the dynamic engine lets the model author the workflow shape
at runtime: create nodes with goals and dependencies, dispatch ready nodes
as background delegations, record results, and extend the graph mid-run.

Usage (from a tool handler)::

    from plugins.workflow.dynamic import handle_workflow_dynamic
    result_json = handle_workflow_dynamic(args, parent_agent)

Architecture:
    Model (tool call) → handle_workflow_dynamic (this) → delegate_task
    → async_delegation → completion_queue → reconcile → dispatch next layer

The engine is in-memory only.  Persistence, kanban integration,
and cost guards belong in ``dynamic_bridge.py``.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────

# Node lifecycle:  pending → dispatched → completed | failed | cancelled
PENDING = "pending"
DISPATCHED = "dispatched"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"

# Workflow-level status values (derived from node states)
WF_RUNNING = "running"
WF_READY = "ready"
WF_WAITING = "waiting"
WF_COMPLETED = "completed"
WF_FAILED = "failed"
WF_EMPTY = "empty"
WF_CANCELLED = "cancelled"

# Terminal and non-terminal node sets
_TERMINAL_STATUSES: set[str] = {COMPLETED, FAILED, CANCELLED}
_RECORD_STATUSES: set[str] = {COMPLETED, FAILED, CANCELLED}

# Async delegation status → workflow node status mapping
_ASYNC_COMPLETED: set[str] = {"completed", "success", "succeeded"}
_ASYNC_FAILED: set[str] = {"error", "failed", "failure", "timeout"}
_ASYNC_CANCELLED: set[str] = {"cancelled", "canceled", "interrupted"}

# ID validation: alphanumeric, underscores, dots, hyphens, up to 96 chars
_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,96}$")


# ── Data classes ──────────────────────────────────────────────────


@dataclass
class DynamicNode:
    """A single worker node in the dynamic DAG.

    Nodes are created by the model via ``create`` or ``extend`` actions.
    Each node has a goal (what the worker should do) and zero or more
    dependencies (other node_ids that must complete first).
    """

    node_id: str
    goal: str
    depends_on: list[str] = field(default_factory=list)
    toolsets: list[str] | None = None
    role: str = "leaf"
    status: str = PENDING
    delegation_id: str | None = None
    summary: str | None = None
    result: Any | None = None
    error: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
    dispatched_at: float | None = None
    completed_at: float | None = None
    duration_seconds: float | None = None

    def public_view(self) -> dict[str, Any]:
        """Snapshot for the result envelope (no internal bookkeeping)."""
        return {
            "node_id": self.node_id,
            "goal": self.goal,
            "depends_on": list(self.depends_on),
            "toolsets": list(self.toolsets) if self.toolsets else None,
            "role": self.role,
            "status": self.status,
            "delegation_id": self.delegation_id,
            "summary": self.summary,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "dispatched_at": self.dispatched_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class DynamicWorkflow:
    """A model-authored directed acyclic workflow.

    Maintains an ordered set of nodes with dependency edges.  The engine
    computes which nodes are ready, dispatches them as background
    delegations, and advances the graph as results arrive.
    """

    workflow_id: str
    objective: str
    context: str
    nodes: dict[str, DynamicNode] = field(default_factory=dict)
    node_order: list[str] = field(default_factory=list)
    status: str = WF_EMPTY
    created_at: float = 0.0
    updated_at: float = 0.0
    cancelled_at: float | None = None
    scope_key: str = "global"
    extension_count: int = 0

    def public_view(self) -> dict[str, Any]:
        """Snapshot for the result envelope."""
        return {
            "workflow_id": self.workflow_id,
            "objective": self.objective,
            "context": self.context,
            "status": _derive_status(self),
            "ready_node_ids": _compute_ready(self),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "cancelled_at": self.cancelled_at,
            "nodes": [
                self.nodes[nid].public_view()
                for nid in self.node_order
                if nid in self.nodes
            ],
        }


# ── Result envelope helpers ───────────────────────────────────────


def _ok(payload: Any) -> str:
    """Wrap a successful result in the standard tool-output envelope."""
    return json.dumps({"ok": True, **payload}, indent=2, default=str)


def _err(message: str, **extra: Any) -> str:
    """Wrap an error result; ``message`` is a short agent-readable string."""
    return json.dumps(
        {"ok": False, "error": message, **extra}, indent=2, default=str
    )


# ── Module-level storage ──────────────────────────────────────────

# Keyed by (scope_key, workflow_id)
_workflows: dict[tuple[str, str], DynamicWorkflow] = {}
# Completed workflows awaiting eviction: key → (workflow, completed_timestamp)
_completed_workflows: dict[tuple[str, str], tuple[DynamicWorkflow, float]] = {}
_workflows_lock = threading.RLock()

# How long to retain completed workflows before full eviction (seconds)
_COMPLETED_RETENTION_SECONDS = 3600


# ── Internal helpers ──────────────────────────────────────────────

# Workflow-level statuses that are terminal (no further state changes expected)
_WORKFLOW_TERMINAL_STATUSES: set[str] = {WF_COMPLETED, WF_FAILED, WF_CANCELLED}


def _now() -> float:
    """Current wall-clock timestamp."""
    return time.time()


def _is_workflow_terminal(workflow: DynamicWorkflow) -> bool:
    """Return True if the workflow has reached a terminal status."""
    return _derive_status(workflow) in _WORKFLOW_TERMINAL_STATUSES


def _evict_completed() -> None:
    """Promote terminal workflows to the completed cache and purge stale entries.

    This is called on every ``_action_status`` and ``_action_dispatch``
    invocation so that finished workflows don't leak memory indefinitely.
    """
    now = _now()
    # 1) Promote terminal workflows out of the active dict
    terminal_keys: list[tuple[str, str]] = []
    for key, wf in _workflows.items():
        if _is_workflow_terminal(wf):
            terminal_keys.append(key)
    for key in terminal_keys:
        _completed_workflows[key] = (_workflows.pop(key), now)

    # 2) Evict completed entries older than the retention window
    expired = [
        key
        for key, (_, ts) in _completed_workflows.items()
        if now - ts > _COMPLETED_RETENTION_SECONDS
    ]
    for key in expired:
        del _completed_workflows[key]


def _new_workflow_id() -> str:
    """Generate a unique workflow identifier."""
    return f"wf_{uuid.uuid4().hex[:10]}"


def _resolve_scope(parent_agent: Any) -> str:
    """Determine the in-process visibility scope for workflows.

    Scope ensures agents in different sessions don't see each other's
    dynamic workflows.  Resolution order:
      1. ``session_id`` on the parent agent
      2. ``_gateway_session_key`` on the parent agent
      3. ``"global"`` fallback
    """
    for attr in ("session_id", "_gateway_session_key"):
        value = getattr(parent_agent, attr, None)
        if value:
            return f"{attr}:{value}"
    return "global"


def _lookup_workflow(
    workflow_id: str, scope_key: str
) -> DynamicWorkflow | None:
    """Fetch a workflow by id within a scope.  Returns None if missing."""
    return _workflows.get((scope_key, workflow_id))


def _validate_id_format(value: str, label: str) -> str | None:
    """Check that an id matches the allowed pattern.  Returns an error
    string on failure, ``None`` on success."""
    if not value:
        return f"{label} is required"
    if not _ID_RE.match(value):
        return f"{label} must match {_ID_RE.pattern} (got: {value!r})"
    return None


# ── Graph validation ──────────────────────────────────────────────


def _check_cycles(nodes: dict[str, DynamicNode]) -> list[str]:
    """Detect cycles via DFS with visiting/visited sets.

    Returns a list of human-readable issue strings.  Empty list means
    no cycles found.
    """
    issues: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def _visit(node_id: str, stack: list[str]) -> None:
        if node_id in visited:
            return
        if node_id in visiting:
            issues.append(
                "cycle detected: " + " -> ".join(stack + [node_id])
            )
            return
        visiting.add(node_id)
        for dep in nodes[node_id].depends_on:
            if dep in nodes:
                _visit(dep, stack + [node_id])
        visiting.discard(node_id)
        visited.add(node_id)

    for nid in nodes:
        _visit(nid, [])

    return issues


def _verify_dag(
    nodes: dict[str, DynamicNode], existing_ids: set[str] | None = None
) -> list[str]:
    """Full graph validation for a set of nodes.

    Checks:
      - ID format for every node
      - Non-empty goal
      - No self-dependencies
      - No duplicate node_ids
      - All dependency references resolve (within the set or in existing_ids)
      - No cycles

    ``existing_ids`` is the set of node_ids already in the workflow
    (relevant for ``extend`` actions).
    """
    issues: list[str] = []
    known: set[str] = set(existing_ids) if existing_ids else set()

    for nid, node in nodes.items():
        # ID format
        fmt_err = _validate_id_format(nid, f"node {nid!r} id")
        if fmt_err:
            issues.append(fmt_err)

        # Non-empty goal
        if not node.goal or not node.goal.strip():
            issues.append(f"node {nid!r} has an empty goal")

        # Self-dependency
        if nid in node.depends_on:
            issues.append(f"node {nid!r} cannot depend on itself")

        # Dependency references
        for dep in node.depends_on:
            if dep not in nodes and dep not in known:
                issues.append(
                    f"node {nid!r} depends on unknown node {dep!r}"
                )

    # Duplicate check (structural — should already be caught by dict keys,
    # but validates during construction from raw input)
    seen_ids: set[str] = set()
    for nid in nodes:
        if nid in seen_ids:
            issues.append(f"duplicate node_id: {nid!r}")
        seen_ids.add(nid)

    # Cycle detection (only if no reference issues, to avoid noise)
    if not any("depends on unknown" in i for i in issues):
        issues.extend(_check_cycles(nodes))

    return issues


# ── Ready-set computation ─────────────────────────────────────────


def _compute_ready(workflow: DynamicWorkflow) -> list[str]:
    """Return node_ids of all pending nodes whose deps are completed.

    Order follows ``node_order`` (insertion order).
    """
    ready: list[str] = []
    for nid in workflow.node_order:
        node = workflow.nodes.get(nid)
        if node is None or node.status != PENDING:
            continue
        deps_met = all(
            workflow.nodes[dep].status == COMPLETED
            for dep in node.depends_on
            if dep in workflow.nodes
        )
        if deps_met:
            ready.append(nid)
    return ready


# ── Workflow status derivation ────────────────────────────────────


def _derive_status(workflow: DynamicWorkflow) -> str:
    """Compute the aggregate workflow status from node states."""
    if workflow.cancelled_at is not None:
        return WF_CANCELLED
    if not workflow.nodes:
        return WF_EMPTY
    statuses = {n.status for n in workflow.nodes.values()}
    if FAILED in statuses:
        return WF_FAILED
    if DISPATCHED in statuses:
        return WF_RUNNING
    if statuses <= {COMPLETED}:
        return WF_COMPLETED
    if _compute_ready(workflow):
        return WF_READY
    return WF_WAITING


# ── Async reconciliation ──────────────────────────────────────────


def _node_status_from_delegation(raw_status: Any) -> str | None:
    """Map an async delegation status string to a workflow node status."""
    text = str(raw_status or "").strip().lower()
    if text in _ASYNC_COMPLETED:
        return COMPLETED
    if text in _ASYNC_FAILED:
        return FAILED
    if text in _ASYNC_CANCELLED:
        return CANCELLED
    return None


def _sync_delegation_state(workflow: DynamicWorkflow) -> list[str]:
    """Poll async delegations and update dispatched nodes.

    For each node in ``dispatched`` state, look up its delegation record.
    If the delegation terminal status is reached, transition the node
    accordingly.  Returns list of updated node_ids.
    """
    dispatched_map: dict[str, DynamicNode] = {}
    for nid in workflow.node_order:
        node = workflow.nodes.get(nid)
        if (
            node is not None
            and node.status == DISPATCHED
            and node.delegation_id
        ):
            dispatched_map[node.delegation_id] = node

    if not dispatched_map:
        return []

    try:
        from tools.async_delegation import list_async_delegations

        records = list_async_delegations()
    except Exception as exc:
        logger.debug(
            "dynamic_workflow async reconciliation failed: %s", exc
        )
        return []

    updated: list[str] = []
    for record in records:
        delegation_id = str(record.get("delegation_id") or "")
        if not delegation_id:
            continue
        node = dispatched_map.get(delegation_id)
        if node is None:
            continue

        next_status = _node_status_from_delegation(record.get("status"))
        if next_status is None:
            continue

        node.status = next_status
        node.summary = record.get("summary") or node.summary
        node.error = record.get("error") if record.get("error") else None
        node.completed_at = record.get("completed_at") or _now()
        node.duration_seconds = record.get("duration_seconds")
        node.updated_at = _now()
        updated.append(node.node_id)

    if updated:
        workflow.updated_at = _now()

    return updated


# ── Dispatch mechanics ────────────────────────────────────────────


def _build_worker_context(
    workflow: DynamicWorkflow, node: DynamicNode
) -> str:
    """Build the context string passed to a delegated worker node."""
    lines: list[str] = [
        "[Dynamic workflow worker]",
        f"workflow_id: {workflow.workflow_id}",
        f"node_id: {node.node_id}",
        f"objective: {workflow.objective}",
    ]
    if workflow.context:
        lines.extend(["", "Workflow context:", workflow.context])
    if node.depends_on:
        # Include summaries from completed dependencies
        dep_summaries: list[str] = []
        for dep_id in node.depends_on:
            dep_node = workflow.nodes.get(dep_id)
            if dep_node and dep_node.status == COMPLETED and dep_node.summary:
                dep_summaries.append(f"  {dep_id}: {dep_node.summary[:500]}")
        if dep_summaries:
            lines.extend(["", "Completed dependency summaries:"])
            lines.extend(dep_summaries)
    lines.extend([
        "",
        "When you finish, include workflow_id and node_id in your summary. "
        "The workflow engine reconciles async completion automatically; "
        "the parent may still record or refine results with "
        "dynamic_workflow(action='record').",
    ])
    return "\n".join(lines)


def _fire_delegation(
    workflow: DynamicWorkflow,
    node: DynamicNode,
    parent_agent: Any,
) -> None:
    """Dispatch a single node as a background delegation.

    Builds the worker context, calls ``delegate_task(background=True)``,
    and updates the node status based on the response.  On failure, marks
    the node as ``failed``.
    """
    from tools.delegate_tool import delegate_task

    ctx = _build_worker_context(workflow, node)

    try:
        raw = delegate_task(
            goal=node.goal,
            context=ctx,
            toolsets=node.toolsets,
            role=node.role,
            background=True,
            parent_agent=parent_agent,
        )
    except Exception as exc:
        logger.exception(
            "dynamic_workflow delegation failed for node %s", node.node_id
        )
        node.status = FAILED
        node.error = f"delegation error: {exc}"
        node.updated_at = _now()
        return

    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        parsed = {"error": str(raw)}

    if parsed.get("status") == "dispatched" and parsed.get("delegation_id"):
        node.status = DISPATCHED
        node.delegation_id = parsed["delegation_id"]
        node.dispatched_at = _now()
        node.updated_at = _now()
    else:
        node.status = FAILED
        node.error = parsed.get("error") or str(parsed)
        node.updated_at = _now()


def _dispatch_ready_nodes(
    workflow: DynamicWorkflow,
    parent_agent: Any,
    max_dispatch: int = 16,
) -> dict[str, Any]:
    """Find ready nodes and dispatch them as background delegations.

    Returns a dict with ``dispatched`` (list of dispatched node info)
    and ``dispatch_errors`` (list of error dicts).
    """
    _sync_delegation_state(workflow)

    if parent_agent is None:
        return {
            "dispatched": [],
            "dispatch_errors": [
                {"error": "dispatch requires a parent agent context"}
            ],
        }
    if workflow.cancelled_at is not None:
        return {"dispatched": [], "dispatch_errors": [{"error": "workflow is cancelled"}]}

    try:
        cap = max(1, min(int(max_dispatch), 16))
    except (TypeError, ValueError):
        cap = 16

    dispatched: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for node_id in _compute_ready(workflow)[:cap]:
        node = workflow.nodes[node_id]
        _fire_delegation(workflow, node, parent_agent)

        if node.status == DISPATCHED and node.delegation_id:
            dispatched.append({
                "node_id": node_id,
                "delegation_id": node.delegation_id,
            })
        else:
            errors.append({
                "node_id": node_id,
                "error": node.error or "unknown dispatch failure",
            })

    workflow.updated_at = _now()
    return {"dispatched": dispatched, "dispatch_errors": errors}


# ── Cancellation ──────────────────────────────────────────────────


def _stop_single_delegation(delegation_id: str) -> bool:
    """Best-effort interrupt of a single async delegation.

    Looks up the delegation record's ``interrupt_fn`` closure and
    invokes it.  Returns True if the interrupt was fired.

    Note: ``list_async_delegations()`` strips ``interrupt_fn`` for
    safety, so we access the module-internal record store.  This is
    intentionally a best-effort fallback; the bridge layer should
    provide a cleaner interrupt path when available.
    """
    try:
        from tools import async_delegation as _ad_mod

        with _ad_mod._records_lock:
            record = _ad_mod._records.get(delegation_id)
            if record and record.get("status") == "running":
                fn = record.get("interrupt_fn")
                if callable(fn):
                    fn()
                    return True
    except Exception as exc:
        logger.debug(
            "interrupt delegation %s failed: %s", delegation_id, exc
        )
    return False


# ── Action handlers ───────────────────────────────────────────────


def _action_create(
    args: dict[str, Any], parent_agent: Any
) -> str:
    """Create a new dynamic workflow from a list of nodes.

    Validates the objective, nodes, and full graph structure.  Stores the
    workflow and optionally dispatches ready nodes.
    """
    objective = str(args.get("objective") or "").strip()
    if not objective:
        return _err("objective is required for action='create'")

    raw_nodes = args.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        return _err("nodes must be a non-empty list for action='create'")

    # Generate or validate workflow_id
    wf_id_raw = str(args.get("workflow_id") or _new_workflow_id()).strip()
    fmt_err = _validate_id_format(wf_id_raw, "workflow_id")
    if fmt_err:
        return _err(fmt_err)

    # Build node dataclasses from raw input
    nodes: dict[str, DynamicNode] = {}
    node_order: list[str] = []
    issues: list[str] = []

    for idx, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict):
            issues.append(f"nodes[{idx}] must be an object")
            continue

        nid = str(raw.get("node_id") or "").strip()
        fmt_err = _validate_id_format(nid, f"nodes[{idx}].node_id")
        if fmt_err:
            issues.append(fmt_err)
            continue
        if nid in nodes:
            issues.append(f"duplicate node_id: {nid!r}")
            continue

        goal = str(raw.get("goal") or "").strip()
        if not goal:
            issues.append(f"nodes[{idx}].goal is required")
            continue

        depends_raw = raw.get("depends_on") or []
        if not isinstance(depends_raw, list):
            issues.append(f"nodes[{idx}].depends_on must be a list")
            continue
        depends_on = [str(d).strip() for d in depends_raw if str(d).strip()]

        if nid in depends_on:
            issues.append(f"node {nid!r} cannot depend on itself")
            continue

        toolsets_raw = raw.get("toolsets")
        toolsets: list[str] | None = None
        if toolsets_raw is not None:
            if not isinstance(toolsets_raw, list):
                issues.append(f"nodes[{idx}].toolsets must be a list")
                continue
            toolsets = [str(t).strip() for t in toolsets_raw if str(t).strip()]
            if not toolsets:
                toolsets = None

        role = str(raw.get("role") or "leaf").strip().lower()
        if role not in ("leaf", "orchestrator"):
            role = "leaf"

        now = _now()
        nodes[nid] = DynamicNode(
            node_id=nid,
            goal=goal,
            depends_on=depends_on,
            toolsets=toolsets,
            role=role,
            created_at=now,
            updated_at=now,
        )
        node_order.append(nid)

    if issues:
        return _err("; ".join(issues))

    # Full graph validation
    graph_issues = _verify_dag(nodes)
    if graph_issues:
        return _err("; ".join(graph_issues))

    # Build the workflow
    now = _now()
    scope_key = _resolve_scope(parent_agent)
    workflow = DynamicWorkflow(
        workflow_id=wf_id_raw,
        objective=objective,
        context=str(args.get("context") or "").strip(),
        nodes=nodes,
        node_order=node_order,
        status=WF_EMPTY,
        created_at=now,
        updated_at=now,
        scope_key=scope_key,
    )

    # Store under lock
    with _workflows_lock:
        key = (scope_key, wf_id_raw)
        if key in _workflows:
            return _err(f"workflow_id already exists: {wf_id_raw}")
        _workflows[key] = workflow

    # Optionally dispatch ready nodes
    dispatch_result: dict[str, Any] = {"dispatched": [], "dispatch_errors": []}
    if args.get("dispatch_ready"):
        with _workflows_lock:
            dispatch_result = _dispatch_ready_nodes(
                workflow,
                parent_agent,
                args.get("max_dispatch") or 16,
            )

    return _ok({"workflow": workflow.public_view(), **dispatch_result})


def _action_extend(
    args: dict[str, Any], parent_agent: Any
) -> str:
    """Add new nodes to an existing workflow.

    Validates the new nodes against the existing graph, merges them in,
    and optionally dispatches ready nodes.
    """
    wf_id = str(args.get("workflow_id") or "").strip()
    fmt_err = _validate_id_format(wf_id, "workflow_id")
    if fmt_err:
        return _err(fmt_err)

    raw_nodes = args.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        return _err("nodes must be a non-empty list for action='extend'")

    with _workflows_lock:
        scope_key = _resolve_scope(parent_agent)
        workflow = _lookup_workflow(wf_id, scope_key)
        if workflow is None:
            return _err(f"unknown workflow_id: {wf_id!r}")
        if workflow.cancelled_at is not None:
            return _err("cannot add nodes to a cancelled workflow")

        # Build new node dataclasses
        new_nodes: dict[str, DynamicNode] = {}
        node_order_additions: list[str] = []
        issues: list[str] = []

        for idx, raw in enumerate(raw_nodes):
            if not isinstance(raw, dict):
                issues.append(f"nodes[{idx}] must be an object")
                continue

            nid = str(raw.get("node_id") or "").strip()
            fmt_err = _validate_id_format(nid, f"nodes[{idx}].node_id")
            if fmt_err:
                issues.append(fmt_err)
                continue
            if nid in workflow.nodes or nid in new_nodes:
                issues.append(f"duplicate node_id: {nid!r}")
                continue

            goal = str(raw.get("goal") or "").strip()
            if not goal:
                issues.append(f"nodes[{idx}].goal is required")
                continue

            depends_raw = raw.get("depends_on") or []
            if not isinstance(depends_raw, list):
                issues.append(f"nodes[{idx}].depends_on must be a list")
                continue
            depends_on = [
                str(d).strip() for d in depends_raw if str(d).strip()
            ]

            if nid in depends_on:
                issues.append(f"node {nid!r} cannot depend on itself")
                continue

            toolsets_raw = raw.get("toolsets")
            toolsets: list[str] | None = None
            if toolsets_raw is not None:
                if not isinstance(toolsets_raw, list):
                    issues.append(f"nodes[{idx}].toolsets must be a list")
                    continue
                toolsets = [
                    str(t).strip()
                    for t in toolsets_raw
                    if str(t).strip()
                ]
                if not toolsets:
                    toolsets = None

            role = str(raw.get("role") or "leaf").strip().lower()
            if role not in ("leaf", "orchestrator"):
                role = "leaf"

            now = _now()
            new_nodes[nid] = DynamicNode(
                node_id=nid,
                goal=goal,
                depends_on=depends_on,
                toolsets=toolsets,
                role=role,
                created_at=now,
                updated_at=now,
            )
            node_order_additions.append(nid)

        if issues:
            return _err("; ".join(issues))

        # Validate merged graph
        merged = dict(workflow.nodes)
        for nid, node in new_nodes.items():
            merged[nid] = node
        existing_ids = set(workflow.nodes.keys())
        graph_issues = _verify_dag(merged, existing_ids=existing_ids)
        if graph_issues:
            return _err("; ".join(graph_issues))

        # Merge into workflow
        for nid in node_order_additions:
            workflow.nodes[nid] = new_nodes[nid]
            workflow.node_order.append(nid)
        workflow.updated_at = _now()

        # Optionally dispatch
        dispatch_result: dict[str, Any] = {
            "dispatched": [],
            "dispatch_errors": [],
        }
        if args.get("dispatch_ready"):
            dispatch_result = _dispatch_ready_nodes(
                workflow,
                parent_agent,
                args.get("max_dispatch") or 16,
            )

        return _ok({"workflow": workflow.public_view(), **dispatch_result})


def _action_record(
    args: dict[str, Any], parent_agent: Any
) -> str:
    """Record the result of a completed, failed, or cancelled node.

    Updates the node's status, summary, result, and error fields.
    Then runs reconciliation + dispatch to advance the graph.
    """
    wf_id = str(args.get("workflow_id") or "").strip()
    fmt_err = _validate_id_format(wf_id, "workflow_id")
    if fmt_err:
        return _err(fmt_err)

    nid = str(args.get("node_id") or "").strip()
    fmt_err = _validate_id_format(nid, "node_id")
    if fmt_err:
        return _err(fmt_err)

    status = str(args.get("status") or "").strip().lower()
    if status not in _RECORD_STATUSES:
        return _err(
            f"status must be one of: {', '.join(sorted(_RECORD_STATUSES))}"
        )

    with _workflows_lock:
        scope_key = _resolve_scope(parent_agent)
        workflow = _lookup_workflow(wf_id, scope_key)
        if workflow is None:
            return _err(f"unknown workflow_id: {wf_id!r}")

        node = workflow.nodes.get(nid)
        if node is None:
            return _err(f"unknown node_id: {nid!r}")

        # Enforce dependency ordering for completion
        if status == COMPLETED:
            for dep_id in node.depends_on:
                dep_node = workflow.nodes.get(dep_id)
                if dep_node and dep_node.status != COMPLETED:
                    return _err(
                        f"node {nid!r} cannot complete before dependency "
                        f"{dep_id!r} (status: {dep_node.status})"
                    )

        now = _now()
        node.status = status
        node.summary = str(args.get("summary") or "").strip() or None
        node.error = (
            str(args.get("error") or "").strip() if args.get("error") else None
        )
        if "result" in args:
            node.result = deepcopy(args.get("result"))
        node.completed_at = now
        node.updated_at = now
        if node.dispatched_at and node.completed_at:
            node.duration_seconds = round(
                node.completed_at - node.dispatched_at, 2
            )
        workflow.updated_at = now

        # Reconcile + dispatch to advance the graph
        _sync_delegation_state(workflow)
        _dispatch_ready_nodes(workflow, parent_agent)

        # Move workflow to _completed_workflows if it reached terminal state
        if _is_workflow_terminal(workflow):
            key = (scope_key, wf_id)
            _workflows.pop(key, None)
            _completed_workflows[key] = (workflow, now)

        # Auto-extension: suggest follow-up nodes on completion
        ext_payload: dict[str, Any] = {}
        if status == COMPLETED and node.summary:
            from plugins.workflow import get_config
            from plugins.workflow.analyst import analyze_extension

            cfg = get_config()
            max_nodes = cfg.get("max_nodes_per_workflow", 256)
            max_ext = cfg.get("max_extensions_per_workflow", 10)
            max_per_ext = cfg.get("max_nodes_per_extension", 3)

            # Guard: skip if workflow is already at the node cap
            if len(workflow.nodes) >= max_nodes:
                ext_payload["extension_note"] = (
                    f"skipped: workflow at node limit ({max_nodes})"
                )
            # Guard: skip if extension count reached the cap
            elif workflow.extension_count >= max_ext:
                ext_payload["extension_note"] = (
                    f"skipped: extension limit reached ({max_ext})"
                )
            else:
                existing_ids = set(workflow.nodes.keys())
                suggestions = analyze_extension(
                    summary=node.summary,
                    objective=workflow.objective,
                    existing_nodes=list(existing_ids),
                )
                if suggestions:
                    # Dedup: filter out suggestions whose node_id already exists
                    suggestions = [
                        s for s in suggestions
                        if s.get("node_id") not in existing_ids
                    ]
                    # Truncate to max_nodes_per_extension
                    if len(suggestions) > max_per_ext:
                        suggestions = suggestions[:max_per_ext]

                if suggestions:
                    auto_approve = cfg.get(
                        "auto_approve_extensions", False
                    )
                    if auto_approve:
                        ext_args = {
                            "workflow_id": wf_id,
                            "nodes": suggestions,
                            "dispatch_ready": True,
                        }
                        ext_result = _action_extend(ext_args, parent_agent)
                        workflow.extension_count += 1
                        ext_payload["auto_extended"] = True
                        ext_payload["extension_result"] = (
                            json.loads(ext_result)
                            if isinstance(ext_result, str)
                            else ext_result
                        )
                    else:
                        existing = getattr(workflow, "_pending_extensions", None) or []
                        if not isinstance(existing, list):
                            existing = [existing]
                        combined = existing + suggestions
                        workflow._pending_extensions = combined
                        ext_payload["pending_extensions"] = combined
                        ext_payload["extension_note"] = (
                            "suggestions stored for user review"
                        )
                    # Always write extension artifact for audit trail
                    from plugins.workflow.dynamic_bridge import (
                        _append_extension_artifact,
                    )
                    _append_extension_artifact(
                        workflow_id=wf_id,
                        node_id=nid,
                        node_summary=node.summary or "",
                        suggestions=suggestions,
                        auto_approved=auto_approve,
                    )
                else:
                    ext_payload["extension_note"] = "no follow-up nodes suggested"

        return _ok({"workflow": workflow.public_view(), **ext_payload})


def _action_dispatch(
    args: dict[str, Any], parent_agent: Any
) -> str:
    """Manually trigger dispatch of ready nodes.

    Reconciles async state first, then dispatches up to ``max_dispatch``
    ready nodes as background delegations.
    """
    wf_id = str(args.get("workflow_id") or "").strip()
    fmt_err = _validate_id_format(wf_id, "workflow_id")
    if fmt_err:
        return _err(fmt_err)

    with _workflows_lock:
        _evict_completed()

        scope_key = _resolve_scope(parent_agent)
        workflow = _lookup_workflow(wf_id, scope_key)
        if workflow is None:
            return _err(f"unknown workflow_id: {wf_id!r}")

        result = _dispatch_ready_nodes(
            workflow,
            parent_agent,
            args.get("max_dispatch") or 16,
        )
        return _ok({"workflow": workflow.public_view(), **result})


def _action_status(
    args: dict[str, Any], parent_agent: Any
) -> str:
    """Return the current state of a workflow (or all workflows in scope).

    Reconciles async delegation state before reporting.
    """
    with _workflows_lock:
        _evict_completed()

        wf_id = str(args.get("workflow_id") or "").strip()
        scope_key = _resolve_scope(parent_agent)

        if wf_id:
            fmt_err = _validate_id_format(wf_id, "workflow_id")
            if fmt_err:
                return _err(fmt_err)

            workflow = _lookup_workflow(wf_id, scope_key)
            if workflow is None:
                return _err(f"unknown workflow_id: {wf_id!r}")

            _sync_delegation_state(workflow)
            return _ok({"workflow": workflow.public_view()})

        # Return all workflows in this scope
        workflows_in_scope = [
            wf.public_view()
            for (wf_scope, _), wf in _workflows.items()
            if wf_scope == scope_key
        ]
        # Reconcile each one
        for (wf_scope, _), wf in _workflows.items():
            if wf_scope == scope_key:
                _sync_delegation_state(wf)

        return _ok({"workflows": workflows_in_scope})


def _action_cancel(
    args: dict[str, Any], parent_agent: Any
) -> str:
    """Cancel a workflow: mark pending nodes as cancelled, optionally
    interrupt dispatched delegations.

    Parameters:
        workflow_id: required
        interrupt: whether to interrupt dispatched delegations (default True)
    """
    wf_id = str(args.get("workflow_id") or "").strip()
    fmt_err = _validate_id_format(wf_id, "workflow_id")
    if fmt_err:
        return _err(fmt_err)

    interrupt = bool(args.get("interrupt", True))
    interrupted_ids: list[str] = []

    with _workflows_lock:
        scope_key = _resolve_scope(parent_agent)
        workflow = _lookup_workflow(wf_id, scope_key)
        if workflow is None:
            return _err(f"unknown workflow_id: {wf_id!r}")

        now = _now()
        workflow.cancelled_at = now
        workflow.updated_at = now

        for node in workflow.nodes.values():
            if node.status == PENDING:
                node.status = CANCELLED
                node.updated_at = now
            elif node.status == DISPATCHED and node.delegation_id:
                if interrupt:
                    if _stop_single_delegation(node.delegation_id):
                        interrupted_ids.append(node.delegation_id)
                node.updated_at = now

        # Move cancelled workflow to _completed_workflows
        key = (scope_key, wf_id)
        _workflows.pop(key, None)
        _completed_workflows[key] = (workflow, now)

        return _ok({
            "workflow": workflow.public_view(),
            "interrupted_delegation_ids": interrupted_ids,
        })


# ── Public entry point ────────────────────────────────────────────

# ── Durable recovery ─────────────────────────────────────────────


def _restore_durable_workflows() -> None:
    """Scan persist_dir for state files and restore workflows to memory."""
    from pathlib import Path as _Path
    from plugins.workflow import get_config as _get_config
    persist_dir = _Path(_get_config().get("persist_dir", "~/.hermes/workflow-logs")).expanduser()
    if not persist_dir.is_dir():
        return

    for state_file in persist_dir.glob("*/state.json"):
        try:
            state = json.loads(state_file.read_text())
        except Exception:
            continue
        scope_key = state.get("scope", "project")
        workflow_id = state_file.parent.name
        registry_key = (scope_key, workflow_id)
        if registry_key in _workflows:  # already loaded
            continue
        try:
            from plugins.workflow.dynamic_bridge import _recover_workflow
            recovered = _recover_workflow(workflow_id)
            if recovered:
                with _workflows_lock:
                    _workflows[registry_key] = recovered
        except Exception as exc:
            logger.warning("failed to recover workflow %s: %s", workflow_id, exc)


_ACTIONS = {
    "create": _action_create,
    "extend": _action_extend,
    "record": _action_record,
    "dispatch": _action_dispatch,
    "status": _action_status,
    "cancel": _action_cancel,
}

# ── Durable scope recovery (module load) ──────────────────────────
# Restore previously persisted workflows at import time so the durable
# scope is available before any tool call or workflow mutation.
try:
    _restore_durable_workflows()
except Exception as exc:
    logger.warning("durable recovery at module load failed: %s", exc)


def handle_workflow_dynamic(args: dict[str, Any], parent_agent: Any = None) -> str:
    """Tool entry point for the dynamic workflow engine.

    Dispatches on ``args["action"]`` to the appropriate handler.  Returns
    a JSON envelope string (``{"ok": True, ...}`` or ``{"ok": False, ...}``).

    Actions:
        create   — build a new workflow from a list of nodes
        extend   — add new nodes to an existing workflow
        record   — record a node's completion/failure/cancellation
        dispatch — manually trigger dispatch of ready nodes
        status   — query workflow state (single or all in scope)
        cancel   — cancel a workflow and optionally interrupt delegations
    """
    if not isinstance(args, dict):
        return _err("dynamic_workflow arguments must be an object")

    action = str(args.get("action") or "").strip().lower()
    handler = _ACTIONS.get(action)
    if handler is None:
        return _err(
            f"action must be one of: {', '.join(sorted(_ACTIONS))}"
        )

    try:
        return handler(args, parent_agent)
    except Exception as exc:
        logger.exception("dynamic_workflow action=%s failed", action)
        return _err(f"action {action!r} failed: {exc}")


# ── Test helper ───────────────────────────────────────────────────


def _reset_for_tests() -> None:
    """Clear all in-memory workflow state.  Test-only."""
    with _workflows_lock:
        _workflows.clear()
        _completed_workflows.clear()
