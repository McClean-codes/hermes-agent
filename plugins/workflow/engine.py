"""
Workflow Execution Engine — mechanical DAG runner for multi-agent pipelines.

Reads a workflow YAML, topologically sorts agent nodes, creates kanban cards
for ready nodes, monitors completion, and advances the graph. Supports revision
review pipelines via the reviews attribute on nodes.

Usage:
    python -m plugins.workflow.engine start ideation --context pr=123
    python -m plugins.workflow.engine validate ideation
    python -m plugins.workflow.engine list

Architecture:
    Trigger (Discord/webhook) → Classify (Sherlock) → Engine (this) → Kanban → Agents

Revision Loops:
    When an agent rejects work, they block the card with reason
    "LOOP:<verify-node> | <human-readable details>". The engine:
    1. Finds the revision node that depends on the verify node
    2. Creates + monitors the revision card
    3. Re-creates the verify card (loop back)
    4. Repeats up to 3 times, then escalates to Sherlock

    Example:
      Nikola blocks nikola-verify-spec with:
        "LOOP:nikola-verify-spec | Missing billing edge case, auth rate limiting"
      Engine: runs edison-revise-spec → re-runs nikola-verify-spec
"""

import yaml
import json
import time
import subprocess
import sys
import os
import re
import logging
from pathlib import Path
import hermes_cli.kanban_db as kanban_db
from collections import defaultdict, deque
from dataclasses import dataclass, field

logger = logging.getLogger("plugins.workflow.engine")
from typing import Optional

from plugins.workflow.utils import hermes_binary
from datetime import datetime, timezone

# ── Data structures ──────────────────────────────────────────────

@dataclass
class WorkflowNode:
    """A single agent task in the DAG.

    A node is either a real agent task (synthetic=False, agent is the
    dispatch target) or a synthetic gate (synthetic=True, no agent).
    Synthetic gates are auto-completed once their dependencies are done —
    no kanban card is created, no agent is dispatched. They exist to
    enforce ordering in the DAG (e.g. a privacy barrier between two
    phases) without adding a no-op task to the board.
    """
    id: str
    agent: Optional[str]          # Agent name (matches kanban worker).
                                  # None for synthetic gate nodes.
    task: str                     # Task description for the kanban card
    description: str = ""         # Human-readable description of what this node does
    depends_on: list[str] = field(default_factory=list)
    timeout_minutes: int = 30
    model: Optional[str] = None   # Optional model override
    channel: str = "debug"        # Where to send notifications
    synthetic: bool = False       # True for gate nodes (no agent, auto-complete)
    phase: Optional[str] = None   # Explicit phase label for {phaseN.X} template
                                  # substitution. When None, the engine auto-
                                  # derives it from the topological layer
                                  # index ("phase0", "phase1", ...) at
                                  # lookup time.
    # ── New fields ──────────────────────────────────────────────
    outputs: list[dict] = field(default_factory=list)
    """"Expected artifact outputs. Each entry:
        path: str — resolved against agent workspace + {run_id}
        required: bool — if True and file missing, validation fails
        schema: str — optional: "json", "markdown", "text"
    """
    fallback_on_timeout: str = "skip"
    """Behavior when this node times out:
        skip  — mark failed, cascade skip downstream (default)
        degraded — mark degraded, downstream proceeds with warning
        retry — re-create the card (up to 3 attempts)
    """
    privacy_gate: bool = False
    """When True, this node's output is excluded from template lookup
       for downstream nodes. Used for premortem isolation — position
       agents should not see the premortem's failure imagination.
    """
    goal_max_turns: Optional[int] = None
    """Per-node goal-mode turn limit. When None (default), the
       kanban CLI's own default (20) is used. Set in YAML to
       constrain deep-research tasks that would otherwise exhaust
       the session before calling kanban_complete, or to tighten
       limits for trivial tasks.
    """
    triage: bool = False
    """When True, the card is created in 'triage' status instead of
       'ready'. The auto-decomposer (or manual decompose) will break
       it into sub-tasks with sibling dependencies. The root card
       keeps its depends_on relationships and promotes to 'ready'
       when all children complete.
    """
    when: str = ""
    """Conditional expression controlling whether this node dispatches.

    Empty string (default) means always run — preserving the existing
    behavior for all workflows.  Non-empty strings are evaluated against
    the workflow state + context at dispatch time; a truthy result means
    the node runs, falsy means it is skipped.

    Supported references inside the expression:
      {node-id}.status, {node-id}.result, {node-id}.error,
      {node-id}.attempts, {node-id}.duration_seconds, {node-id}.error_count
      {context.key}

    Supported operators:
      ==, !=, >, <, >=, <=, contains, starts_with, in, and, or, not
    """
    max_retries: int | None = None
    """Max times each reviewer can review this node. None = use workflow default."""
    attachment: Optional[str] = None
    """Attachment name. References a declared attachment by name.
    E.g. "grill_artifact" picks the attachment with that key.
    If None, all attachments are attached (first layer) or none."""
    reviews: list[str] = field(default_factory=list)
    """Sequential review pipeline. Each entry is a node ID that reviews
    this node's output. When the card moves to 'review' status, the
    supervisor creates the first reviewer card. When a reviewer passes
    (moves to 'done'), the next reviewer is created. When a reviewer
    fails (moves to 'blocked'), the original card is enriched with
    feedback and set back to 'ready' for rework.
    """

@dataclass
class Workflow:
    """Complete workflow definition."""
    name: str
    description: str = ""
    trigger_events: list[str] = field(default_factory=list)
    nodes: dict[str, WorkflowNode] = field(default_factory=dict)
    run_id: str = ""              # Generated at execute() time
    kanban_board: str = ""         # Per-pipeline board override (empty = engine default)
    inputs: list = field(default_factory=list)
                                    # Declared inputs for this workflow.
                                    # Each entry: {"name": str, "required": bool, "description": str}
    attachments: list = field(default_factory=list)
                                    # Declared attachments for this workflow.
                                    # Each entry: {"name": str, "required": bool, "description": str}
    scope: str = "project"         # "project" (default) — creates kanban cards per node.
                                    # "global" — in-process only, no cards created.
                                    # Used for maintenance / notification / heartbeat
                                    # workflows that should not pollute project boards.
    single_flight: bool = False     # When True, refuse to start a new run if any
                                    # run for this workflow is already in progress.
                                    # Used to prevent duplicate parallel runs from
                                    # webhook storms or repeated dispatch signals.
                                    # Default False preserves the existing "multiple
                                    # parallel runs allowed" behavior.


@dataclass
class NodeState:
    """Runtime state for a workflow node."""
    node_id: str
    status: str = "pending"       # pending | running | done | failed | blocked | timed_out | degraded
    kanban_card_id: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    attempts: int = 0
    error: Optional[str] = None

    result: Optional[str] = None  # Captured card body (output text) populated
                                  # when the node transitions to "done" or
                                  # "blocked". Available to
                                  # downstream nodes via {phaseN.X} or
                                  # {node-id} template substitution.
    duration_seconds: Optional[float] = None
    """Wall-clock duration from started_at → completed_at.

    Populated when the node enters a terminal status (done / failed /
    timed_out / blocked / blocked). Computed from started_at
    and completed_at; agents don't need to set it manually. Exposed
    via workflow_status for cost / bottleneck analysis.
    """
    error_count: int = 0
    """Cumulative count of failed / timed_out transitions for this node.

    Incremented each time the node enters a failure state. Multiple
    retries on a flaky node will increment this counter — useful for
    spotting nodes that keep failing across runs.
    """
    review_index: int = 0
    """Current index in the review pipeline (which reviewer is active)."""
    review_body: Optional[str] = None
    """Body of the card when it moved to review status."""
    review_counts: dict = field(default_factory=dict)
    """Tracks how many times each reviewer has reviewed this node.
    Key: reviewer node ID, Value: count."""
    validation_warnings: list[str] = field(default_factory=list)

# ── Engine core ──────────────────────────────────────────────────

class CycleDetectedError(Exception):
    """Raised when the workflow graph contains a cycle."""
    pass

class WorkflowEngine:
    """
    Mechanical DAG runner. No HTTP, no webhooks — consumes a workflow file
    and drives kanban cards. Triggered externally (Sherlock, cron, watcher).

    Supports review pipelines via the reviews attribute on nodes.
    """


    POLL_INTERVAL = 15  # seconds between kanban status checks
    STATE_DIR = None     # Set after init for state persistence


    def __init__(self, workflows_dir: str = None):
        if workflows_dir is None:
            workflows_dir = os.environ.get(
                "HERMES_WORKFLOW_FILES",
                ""
            )
            if not workflows_dir:
                # Profile-scoped: $HERMES_HOME/workflows/
                hermes_home = os.environ.get("HERMES_HOME", "")
                if hermes_home:
                    candidate = Path(hermes_home) / "workflows"
                    if candidate.is_dir():
                        workflows_dir = str(candidate)
            if not workflows_dir:
                # Ship defaults: next to the engine module
                workflows_dir = str(
                    Path(__file__).resolve().parent.parent / "docs" / "fleet-pipelines"
                )
        self.workflows_dir = Path(workflows_dir)
        self.kanban_board = ""  # Set by three-tier resolution in execute()
        WorkflowEngine.STATE_DIR = self.workflows_dir / ".engine-state"
        self.STATE_DIR.mkdir(parents=True, exist_ok=True)
        # Job log DB at ~/.hermes/workflows/executions.db
        from hermes_cli.kanban_db import kanban_home
        self._exec_db_path = kanban_home() / "workflows" / "executions.db"
        self._exec_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_exec_db()

    # ── Loading ───────────────────────────────────────────────

    def load_workflow(self, name: str) -> Workflow:
        """Load a workflow definition from YAML."""
        path = self.workflows_dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Workflow not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        # Read role→profile bindings from the top-level `roles:` block.
        # Nodes can use `agent: "{role_name}"` placeholders that resolve to
        # the bound profile at load time. To swap a profile, edit one line
        # in `roles:` and every node that references the role picks it up.
        roles = raw.get("roles", {}) or {}

        workflow = Workflow(
            name=raw.get("name", name),
            description=raw.get("description", ""),
            trigger_events=raw.get("trigger_events", []),
            kanban_board=raw.get("kanban_board", ""),
            inputs=raw.get("inputs", []),
            attachments=raw.get("attachments", []),
            scope=raw.get("scope", "project"),
            single_flight=bool(raw.get("single_flight", False)),

        )

        for node_id, node_data in raw.get("nodes", {}).items():
            # Synthetic gate nodes: no agent dispatch, auto-completed when
            # their depends_on are satisfied. Used for privacy barriers
            # and pure ordering (e.g. council-ready gate between the
            # premortem phase and the position phase).
            synthetic = bool(node_data.get("synthetic", False))
            if synthetic:
                # Warn if the author left a redundant agent field —
                # easy to forget to remove when converting an existing
                # node into a synthetic gate.
                if "agent" in node_data:
                    print(f"   ⚠  Node '{node_id}' has synthetic: true "
                          f"with an explicit agent — ignoring agent field")
                agent_value: Optional[str] = None
                # Task is optional for synthetic gates; default to the
                # node id so logs/UI still have something to display.
                task_value = node_data.get("task", f"[synthetic gate] {node_id}")
                # Default to a trivial timeout — synthetic gates
                # auto-complete in the dispatch loop, but we still want
                # a sane value if anyone reads the node later.
                timeout_value = node_data.get("timeout_minutes", 1)
            else:
                agent_value = node_data["agent"]
                # Resolve {role} placeholders against the `roles:` block.
                # Templates that don't match any role key pass through
                # unchanged (the engine surfaces the missing assignee on
                # the resulting kanban card, so the failure is visible).
                if isinstance(agent_value, str) and "{" in agent_value and roles:
                    try:
                        agent_value = agent_value.format(**{
                            k: v for k, v in roles.items()
                            if isinstance(v, (str, int, float))
                        })
                    except (KeyError, ValueError, TypeError):
                        pass
                task_value = node_data["task"]
                timeout_value = node_data.get("timeout_minutes", 30)

            # Parse new-style fields
            outputs_raw = node_data.get("outputs", [])
            if isinstance(outputs_raw, list):
                for o in outputs_raw:
                    if isinstance(o, str):
                        # Shorthand: just a path string
                        outputs_raw = [{"path": o, "required": True, "schema": "text"}]
                        break

            fallback_raw = node_data.get("fallback_on_timeout", "skip")
            if fallback_raw not in ("skip", "degraded", "retry"):
                print(f"   ⚠  Node '{node_id}' has invalid "
                      f"fallback_on_timeout='{fallback_raw}' — defaulting to 'skip'")
                fallback_raw = "skip"

            workflow.nodes[node_id] = WorkflowNode(
                id=node_id,
                description=node_data.get("description", ""),
                agent=agent_value,
                task=task_value,
                depends_on=node_data.get("depends_on", []),
                timeout_minutes=timeout_value,
                model=node_data.get("model"),
                channel=node_data.get("channel", "debug"),
                synthetic=synthetic,
                # `phase:` is optional in YAML. When omitted, the engine
                # auto-derives a phase label from the topological layer
                # index at template-substitution time (e.g. "phase0",
                # "phase1", "phase2"). Setting an explicit phase is
                # useful when the author wants sub-phases ("phase2a",
                # "phase2b") that don't map 1:1 to a single layer.
                phase=node_data.get("phase"),
                outputs=outputs_raw,
                fallback_on_timeout=fallback_raw,
                privacy_gate=bool(node_data.get("privacy_gate", False)),
                goal_max_turns=node_data.get("goal_max_turns"),
                triage=bool(node_data.get("triage", False)),
                when=node_data.get("when", ""),
                reviews=node_data.get("reviews", []),
                max_retries=node_data.get("max_retries"),
                attachment=node_data.get("attachment"),
            )

        return workflow

    # ── Topological sort ──────────────────────────────────────

    def topological_sort(self, workflow: Workflow) -> list[list[str]]:
        """
        Returns layers of nodes that can run in parallel.
        Layer 0 has no dependencies. Layer N depends only on layers < N.
        Also detects cycles.
        """
        in_degree = {nid: len(node.depends_on) for nid, node in workflow.nodes.items()}
        dependents = defaultdict(list)

        for nid, node in workflow.nodes.items():
            for dep in node.depends_on:
                if dep not in workflow.nodes:
                    raise ValueError(f"Node '{nid}' depends on unknown node '{dep}'")
                dependents[dep].append(nid)

        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        layers = []
        processed = 0

        while queue:
            layer = list(queue)
            layers.append(layer)
            queue.clear()
            processed += len(layer)

            for nid in layer:
                for dependent in dependents[nid]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        if processed != len(workflow.nodes):
            remaining = [nid for nid, deg in in_degree.items() if deg > 0]
            raise CycleDetectedError(
                f"Cycle detected involving nodes: {', '.join(remaining)}"
            )

        return layers

    # ── Dependency lookup ─────────────────────────────────────

    def _find_revision_node(self, workflow: Workflow, verify_node_id: str) -> Optional[str]:
        """
        Find the revision node that depends on a verify node.

        Assumes a single-revision pattern: each verify node has exactly
        one revision node that depends on it. If future pipelines have
        multiple revision nodes for one verify node, this returns the
        first one encountered — upgrade to return a list when needed.
        """
        for nid, node in workflow.nodes.items():
            if verify_node_id in node.depends_on:
                return nid
        return None

    def _fire_completion_notification(self, workflow_name, workflow, states, layers, layer_idx, context=None, session_info=None):
        """Fire completion notification after all layers complete."""
        try:
            from plugins.workflow import _notify_workflow_complete
            _notif_state = {
                "workflow_name": workflow_name,
                "kanban_board": self.kanban_board,
                "run_id": workflow.run_id,
                "session_info": session_info or (context or {}).get("_session_info", {}),
                "states": {nid: {"status": s.status, "kanban_card_id": s.kanban_card_id} for nid, s in states.items()},
                "layers": layers,
                "current_layer": layer_idx,
            }
            for nid in reversed(list(states.keys())):
                if states[nid].status == "done" and states[nid].kanban_card_id:
                    _notify_workflow_complete(states[nid].kanban_card_id, state=_notif_state)
                    break
        except Exception as _notify_exc:
            print(f"   ⚠  Completion notification failed: {_notify_exc}")

    def _find_loop_zones(self, workflow: Workflow, layers: list[list[str]]) -> list[int]:
        """Return layer indices that contain nodes with reviews (need a supervisor)."""
        review_layers: list[int] = []
        for i, layer in enumerate(layers):
            for nid in layer:
                node = workflow.nodes.get(nid)
                if node and node.reviews:
                    review_layers.append(i)
                    break
        return review_layers

    def _find_layer_for_node(self, layers: list[list[str]], node_id: str) -> int:
        """Find which layer a node belongs to."""
        for i, layer in enumerate(layers):
            if node_id in layer:
                return i
        return -1

    # ── Kanban dispatch ────────────────────────────────────────

    def create_kanban_card(self, node: WorkflowNode, context: dict = None,
                            *, workflow: Optional["Workflow"] = None,
                            states: Optional[dict] = None,
                            layers: Optional[list] = None,
                            initial_status: str = "ready") -> str:
        """Create a kanban card for a workflow node. Returns card ID.

        Refuses to create a card for a synthetic gate node — those are
        auto-completed by the executor and never reach this function in
        the normal flow. The explicit guard exists so a future caller
        that forgets the check gets a clear error rather than a
        confusing subprocess failure on the None agent.

        When ``workflow``, ``states``, and ``layers`` are provided, the
        engine runs the B2 template-substitution pass on ``node.task``
        before posting. The resolved text becomes the kanban card body.
        The legacy ``\\n\\nContext: {json}`` footer is appended
        after substitution so it does not get treated as a template.

        The ``workflow``/``states``/``layers`` keyword args are
        optional for backward compatibility with direct callers (e.g.
        the synthetic-node guard test) that invoke this method without
        a running workflow. Callers that omit them get the
        pre-substitution behavior — context footer only, no
        ``{namespace.field}`` resolution.
        """
        if node.synthetic:
            raise ValueError(
                f"Refusing to create a kanban card for synthetic gate "
                f"node '{node.id}' — synthetic nodes are auto-completed "
                f"by the executor and do not dispatch."
            )

        if workflow is not None and states is not None and layers is not None:
            # Full B2 path: resolve {namespace.field} and {bare}
            # references, then append the legacy Context footer. The
            # footer goes after substitution because the JSON literal
            # contains its own braces that the resolver would otherwise
            # chew on.
            task_with_context = self._build_task_body(
                node, workflow, states, layers, context
            )
        else:
            # Backward-compat path: direct callers that pass only
            # (node, context). Preserves the pre-B2 footer-only
            # behavior exactly. Used by tests like
            # test_create_kanban_card_refuses_synthetic that never
            # reach the substitution step.
            task_with_context = node.task
            if context:
                task_with_context += f"\n\nContext: {json.dumps(context)}"

        title = f"[{node.id}] {node.agent}: {task_with_context[:60]}"
        # Resolve {context_var} placeholders in the agent field.
        # This lets workflows accept agent profile names at runtime
        # via context (e.g., agent: "{target_agent}").
        assignee = node.agent
        if assignee and "{" in assignee and context:
            try:
                assignee = assignee.format(**{
                    k: v for k, v in context.items()
                    if isinstance(v, (str, int, float))
                })
            except (KeyError, ValueError, TypeError):
                pass

        # Use the kanban DB Python API directly — no subprocess.
        conn = kanban_db.connect(board=self.kanban_board)
        try:
            import datetime as _dt
            new_tid = kanban_db.create_task(
                conn,
                title=title,
                body=task_with_context,
                assignee=str(assignee),
                parents=(),
                tenant=self.kanban_board,
                priority=2,
                workspace_kind="scratch",
                workspace_path=None,
                project_id=None,
                triage=node.triage if hasattr(node, 'triage') else False,
                max_runtime_seconds=(
                    int(node.timeout_minutes * 60)
                    if node.timeout_minutes is not None else None
                ),
            )
            # Attach files based on node's attachment_index.
            # If attachment_index is set, attach only that specific file.
            # If None, attach all current attachments (first-layer default).
            all_attachments = getattr(self, "_current_attachments", [])
            if node.attachment is not None:
                # Attach only the specified index
                import re as _re
                _m = _re.match(r"attachments\[(\d+)\]", node.attachment)
                idx = int(_m.group(1)) if _m else -1
                if 0 <= idx < len(all_attachments):
                    attachments_to_attach = [all_attachments[idx]]
                else:
                    attachments_to_attach = []
            else:
                # Attach all (first-layer behavior)
                attachments_to_attach = all_attachments
            for fpath in attachments_to_attach:
                try:
                    from pathlib import Path as _P
                    p = _P(fpath)
                    if p.is_file():
                        kanban_db.store_attachment_bytes(
                            conn, new_tid, p.name, p.read_bytes(),
                            content_type=None,
                            uploaded_by="workflow-engine",
                        )
                except Exception:
                    pass  # don't fail card creation over an attachment
            return new_tid
        finally:
            conn.close()

    def dispatch_node(self, state: NodeState, node: WorkflowNode, context: dict,
                       workflow: "Workflow", states: dict, layers: list,
                       initial_status: str = "ready") -> Optional[str]:
        """Dispatch a node to kanban, or mark it done in-process.

        For ``scope: global`` workflows (maintenance, notifications, heartbeat)
        no kanban card is created — the node is marked ``done`` with a
        sentinel ``result`` and ``None`` is returned. Callers should treat
        ``None`` as "in-process, no card to monitor" and skip heartbeat /
        monitoring for this node.

        For ``scope: project`` (default), this delegates straight to
        :meth:`create_kanban_card` and returns the card ID.

        ``initial_status`` controls the card's initial kanban status.
        ``"ready"`` (default) makes the card immediately available to the
        dispatcher. ``"pending"`` hides it from the dispatcher until the
        supervisor flips it.
        """
        if workflow is not None and getattr(workflow, "scope", "project") == "global":
            state.status = "done"
            state.completed_at = datetime.now(timezone.utc).isoformat()
            state.result = "[in-process, scope: global]"
            return None
        return self.create_kanban_card(
            node, context,
            workflow=workflow, states=states, layers=layers,
            initial_status=initial_status,
        )

    def _get_session_info(self) -> dict:
        """Capture gateway session info for subscription routing.

        ContextVars from gateway.session_context are NOT available inside
        engine.execute() — they're lost across the tool-handler → engine call
        boundary.  First try ContextVars (works in some contexts), then
        check the module-level bridge written by the tool handler.
        """
        # 1. Try ContextVars (works when called from gateway session directly)
        try:
            from gateway.session_context import get_session_env
            platform = get_session_env("HERMES_SESSION_PLATFORM", "")
            chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "")
            if platform and chat_id:
                return {
                    "platform": platform,
                    "chat_id": chat_id,
                    "thread_id": get_session_env("HERMES_SESSION_THREAD_ID", "") or None,
                    "user_id": get_session_env("HERMES_SESSION_USER_ID", "") or None,
                    "profile": (
                        get_session_env("HERMES_SESSION_PROFILE", "")
                        or os.environ.get("HERMES_PROFILE")
                    ),
                }
        except Exception as _exc:
            logger.debug("ContextVars session lookup failed: %s", _exc)
        # 2. Check temp file written by tool handler
        try:
            import json as _json
            from pathlib import Path as _P
            _profile = os.environ.get("HERMES_PROFILE") or "default"
            for _p in [f"/tmp/wfe-session-{_profile}.json",
                       _P(__file__).resolve().parent.parent.parent / "docs" / "fleet-pipelines" / ".engine-state" / "_session.json"]:
                try:
                    with open(_p) as _f:
                        info = _json.load(_f)
                    if info and info.get("platform"):
                        return info
                except Exception:
                    continue
        except Exception:
            pass
        # 3. Fallback: os.environ
        try:
            platform = os.environ.get("HERMES_SESSION_PLATFORM", "")
            chat_id = os.environ.get("HERMES_SESSION_CHAT_ID", "")
            if platform and chat_id:
                return {
                    "platform": platform,
                    "chat_id": chat_id,
                    "thread_id": os.environ.get("HERMES_SESSION_THREAD_ID") or None,
                    "user_id": os.environ.get("HERMES_SESSION_USER_ID") or None,
                    "profile": os.environ.get("HERMES_SESSION_PROFILE") or os.environ.get("HERMES_PROFILE"),
                }
        except Exception as _exc:
            logger.debug("os.environ session lookup failed: %s", _exc)
        return {}

    def _spawn_supervisor(self, workflow_name: str, run_id: str) -> None:
        """Spawn a detached subprocess to supervise loop zones.

        The subprocess runs ``python -m tools.workflow_engine start <name> --resume``
        which loads the saved state and enters the monitoring loop for the
        loop zone layers.  It exits when the loop zone resolves.
        """
        try:
            import subprocess as _sp
            cmd = [
                sys.executable, "-m", "tools.workflow_engine", "start", workflow_name,
                "--resume",
                "--board", self.kanban_board,
            ]
            if run_id:
                cmd.extend(["--run-id", run_id])
            log_dir = str(self.STATE_DIR)
            log_path = os.path.join(log_dir, f"supervisor-{run_id}.log")
            log_fd = open(log_path, "w")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            _sp.Popen(
                cmd,
                stdout=log_fd,
                stderr=log_fd,
                stdin=_sp.DEVNULL,
                start_new_session=True,
                env=env,
            )
            print(f"   👤 supervisor spawned for {workflow_name} (run: {run_id})")
            print(f"   📝 supervisor log: {log_path}")
        except Exception as e:
            print(f"   ⚠ failed to spawn supervisor: {e}")



    # Terminal statuses for which we record telemetry. Anything else
    # (running, pending, blocked, blocked, degraded) is mid-flight.
    _TELEMETRY_TERMINAL_STATUSES = frozenset({
        "done", "failed", "timed_out",
    })
    """Statuses that mark a node as finished for telemetry purposes.

    "blocked" and "blocked" are mid-flight (the engine may
    rerun the node), so we don't count them as terminal for duration
    tracking. "degraded" is also mid-flight (downstream proceeds).
    """

    # Tracks which (state-id, status) pairs have already had telemetry
    # recorded, so repeated _save_state calls don't double-count
    # error_count or recompute duration. Reset per engine instance.
    _telemetry_recorded: "set[tuple[str, str]]" = None  # lazy-init in __init__

    def _record_node_completion(self, state: NodeState) -> None:
        """Capture telemetry when a node enters a terminal status.

        Computes ``duration_seconds`` from ``started_at`` and
        ``completed_at``, and increments ``error_count`` for failure
        outcomes. Idempotent — safe to call multiple times via
        ``_save_state`` without double-counting, using a per-node-id
        dedup set.
        """
        if state.status not in self._TELEMETRY_TERMINAL_STATUSES:
            return
        if self._telemetry_recorded is None:
            self._telemetry_recorded = set()
        dedup_key = (state.node_id, state.status)
        if dedup_key in self._telemetry_recorded:
            return
        self._telemetry_recorded.add(dedup_key)
        if state.duration_seconds is None and state.started_at and state.completed_at:
            try:
                start = datetime.fromisoformat(state.started_at)
                end = datetime.fromisoformat(state.completed_at)
                state.duration_seconds = (end - start).total_seconds()
            except (ValueError, TypeError):
                pass
        if state.status in ("failed", "timed_out"):
            state.error_count += 1

    # State files older than this are considered stale — the engine
    # crashed or was killed mid-run and the state is no longer accurate.
    # Single-flight checks ignore stale state files so a single bad
    # crash doesn't permanently block a workflow from running again.
    ACTIVE_RUN_STALE_SECONDS = 3600

    # How many historical state files to retain per workflow. Older
    # state files are pruned at the end of each save so disk usage
    # doesn't grow unbounded with long-running fleet usage. Set to
    # a low default to keep telemetry disk-cheap; raise via
    # ``STATE_RETENTION_PER_WORKFLOW`` env var if more history needed.
    STATE_RETENTION_PER_WORKFLOW = 20

    def _prune_old_runs(self, keep: int = None) -> int:
        """Delete oldest state files beyond ``keep`` per workflow.

        Walks the state directory, groups files by workflow name,
        sorts each group by mtime, and unlinks everything past the
        ``keep`` threshold. Returns the number of files pruned.

        Called automatically at the end of ``_save_state`` so retention
        is enforced without callers needing to remember. Safe to call
        when STATE_DIR doesn't exist yet (returns 0).
        """
        if self.STATE_DIR is None or not self.STATE_DIR.exists():
            return 0
        if keep is None:
            keep = self.STATE_RETENTION_PER_WORKFLOW
        # Group state files by workflow name (strip "_<run_id>_state.json"
        # or "_state.json" suffix).
        groups: dict[str, list[Path]] = defaultdict(list)
        for path in self.STATE_DIR.glob("*_state.json"):
            stem = path.stem  # e.g. "council_20260101T120000_state"
            # Strip "_state" suffix and split off the trailing timestamp.
            if stem.endswith("_state"):
                stem = stem[:-len("_state")]
            # If the stem still has an underscore-separated timestamp suffix
            # (looks like YYYYMMDDTHHMMSS), strip it for grouping.
            parts = stem.rsplit("_", 1)
            workflow_name = parts[0] if len(parts) > 1 else stem
            groups[workflow_name].append(path)
        pruned = 0
        for wf_name, paths in groups.items():
            # Sort by mtime ascending; prune the oldest.
            paths.sort(key=lambda p: p.stat().st_mtime)
            for old in paths[:-keep] if keep > 0 else paths:
                try:
                    old.unlink()
                    pruned += 1
                except OSError:
                    pass
        return pruned

    def _has_active_run(self, workflow_name: str) -> bool:
        """Return True if any in-progress run exists for ``workflow_name``.

        Used to enforce single-flight semantics: workflows with
        ``single_flight: true`` refuse to start a new run when another
        run is already in progress. A run is considered active when its
        state file was updated within ``ACTIVE_RUN_STALE_SECONDS`` and
        contains at least one node in ``running``, ``pending``, or
        ``blocked`` status.

        Returns False if no state file exists, all state files are stale,
        or all nodes in the active state file are in terminal status.
        """
        for path in sorted(self.STATE_DIR.glob(f"{workflow_name}_*_state.json")):
            try:
                with open(path) as f:
                    state = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            updated_at = state.get("updated_at")
            if updated_at:
                try:
                    age = (
                        datetime.now(timezone.utc)
                        - datetime.fromisoformat(updated_at)
                    ).total_seconds()
                    if age > self.ACTIVE_RUN_STALE_SECONDS:
                        continue
                except (ValueError, TypeError):
                    pass
            states = state.get("states", {})
            if any(
                s.get("status") in ("running", "pending", "blocked")
                for s in states.values()
            ):
                return True
        return False

    def get_card_status(self, card_id: str) -> dict:
        """Query a kanban card's current state.

        Uses the kanban DB Python API directly — no subprocess.
        Returns a dict with status, body, latest_summary, etc.
        """
        conn = kanban_db.connect(board=self.kanban_board)
        try:
            task = kanban_db.get_task(conn, card_id)
            if task is None:
                return {"status": "unknown", "error": f"Card {card_id} not found"}
            result = {
                "status": task.status,
                "body": task.body or "",
                "assignee": task.assignee or "",
                "title": task.title or "",
            }
            # Get the latest summary from task_events if available
            try:
                events = conn.execute(
                    "SELECT message FROM task_events WHERE task_id = ? ORDER BY rowid DESC LIMIT 1",
                    (card_id,)
                ).fetchone()
                if events and events[0]:
                    result["latest_summary"] = events[0]
            except Exception as _exc:
                logger.debug("Failed to load card summary events: %s", _exc)
            return result
        finally:
            conn.close()

    def get_card_body(self, card_id: str) -> str:
        """Get the agent's output from a completed kanban card.

        Resolution order:
          1. ``latest_summary`` — the agent's completion summary
             (set via ``kanban_complete(summary=...)``).  This is the
             agent's actual output, not the input prompt.
          2. ``result`` — legacy field (set via
             ``kanban_complete(result=...)``).
          3. ``body`` — the task description / input prompt.  This is
             a fallback; in normal operation it should NOT be the
             agent's output.
        """
        card = self.get_card_status(card_id)
        return card.get("latest_summary",
                        card.get("result",
                                 card.get("body",
                                          card.get("reason",
                                                   card.get("description", "")))))

    def _check_pending_review(self, card_id: str) -> bool:
        """Check if a card is blocked with reason "pending review".

        The block reason is stored in the most recent ``blocked`` event
        payload, NOT in the task body.  This method queries the event
        log directly.
        """
        try:
            with kanban_db.connect_closing(board=self.kanban_board) as _conn:
                row = _conn.execute(
                    "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'blocked' ORDER BY rowid DESC LIMIT 1",
                    (card_id,),
                ).fetchone()
                if row and row[0]:
                    import json as _json
                    payload = _json.loads(row[0])
                    reason = payload.get("reason", "")
                    return reason.strip().lower() == "pending review"
        except Exception:
            pass
        return False

    # ── Template substitution ───────────────────────────────────
    #
    # When a workflow node is dispatched, its `task` text may contain
    # template references like:
    #
    #     {context.question}            <- value from the -c flag set
    #     {phase1.position-edison}      <- specific upstream node's output
    #     {phase1.all}                  <- concatenation of all phase-1 outputs
    #     {question}                    <- legacy bare form: tries context
    #                                    first, then top-level node ids
    #
    # The engine resolves these before posting the body to kanban. The
    # resolution is a *single substitution pass* applied to the task
    # text — no other transformation is layered on top, so YAML authors
    # can predict the output.
    #
    # Unresolved references are left as literal text in the body. The
    # engine prints a one-line warning to stdout for each so operators
    # notice missing upstream data instead of silently shipping empty
    # fields to agents.

    # Pattern matches {namespace.field} OR {field} (legacy bare form).
    # Namespace allows letters/digits/underscore; field allows the same
    # plus hyphens (so node ids like "position-edison" match as a field
    # under the "phase1" namespace). Bare form is a single token.
    _TEMPLATE_RE = re.compile(
        r"\{(?P<ns>[A-Za-z_][A-Za-z0-9_]*)"
        r"\.(?P<field>[A-Za-z0-9_\-]+)"
        r"(?:\.(?P<subfield>[A-Za-z0-9_\-]+))?"
        r"\}"
        r"|\{(?P<bare>[A-Za-z_][A-Za-z0-9_\-]*)\}"
    )

    def _build_template_lookup(self, workflow: "Workflow",
                                states: dict[str, "NodeState"],
                                layers: list[list[str]],
                                context: Optional[dict] = None) -> dict:
        """Build the substitution lookup dict for downstream nodes.

        Returns a dict with three top-level flavors:

        1. ``"context"`` — the start-time context dict (from ``-c`` flags).
           This is what ``{context.X}`` and legacy ``{X}`` look up first.

        2. ``"phaseN"`` (or the explicit ``phase:`` label) — a sub-dict
           containing:
             - one key per completed node in that phase, mapped to the
               captured card body
             - an ``"all"`` key whose value is the concatenation of every
               completed node's result in that phase, in a stable order

        3. Each completed node id is also exposed at the top level. This
           is purely for legacy support — the original council.yaml uses
           un-prefixed names like ``{position-edison-output}``. New
           pipelines should prefer ``{phase1.position-edison}``.

        The lookup is keyed by *node name + phase label*, not by Kanban
        card id. The card body is captured into ``state.result`` when a
        node completes (see ``_monitor_layer``).
        """
        lookup: dict = {"context": dict(context or {})}
        # Promote {inputs.<key>} to a top-level namespace so the
        # template resolver can resolve e.g. {inputs.grill_artifact}
        # directly without requiring {context.inputs.grill_artifact}.
        # Also promote bare input keys so {issue_number} resolves without
        # requiring {inputs.issue_number}.
        ctx_dict = lookup["context"]
        if "inputs" in ctx_dict and isinstance(ctx_dict["inputs"], dict):
            lookup["inputs"] = ctx_dict["inputs"]
            for k, v in ctx_dict["inputs"].items():
                if k not in lookup:
                    lookup[k] = v
        run_id = workflow.run_id or "no-run-id"

        # Add {run_id} and {date} so YAML authors can reference them in
        # output paths and task prompts (e.g. "council/{date}/{run_id}/premortem.json").
        if workflow.run_id:
            lookup["run_id"] = workflow.run_id
            lookup["context"]["run_id"] = workflow.run_id
            # date = YYYY-MM-DD derived from the run_id timestamp
            if "-" in workflow.run_id:
                ts_part = workflow.run_id.split("-", 1)[1]  # "20260728-185117-123456"
                date_part = ts_part.split("-")[0]            # "20260728"
                lookup["date"] = date_part
                lookup["context"]["date"] = date_part
                # Short run ID: time portion past the date for
                # disambiguation in card names and task prompts.
                # e.g. run_id = "implementation-20260728-185117-123456"
                #      run_short_id = "185117-123456"
                time_parts = ts_part.split("-")[1:]
                if time_parts:
                    join = "-".join(time_parts)
                    lookup["run_short_id"] = join
                    lookup["context"]["run_short_id"] = join

        # Pre-compute the phase label for each node. Authors can set
        # `phase:` explicitly in YAML; otherwise we default to the
        # layer index ("phase0", "phase1", ...). We do the
        # default-derivation here, at lookup time, so the loader stays
        # simple and there's no chance of the derived phase drifting
        # from the actual topological layout.
        node_phase: dict[str, str] = {}
        for layer_idx, layer in enumerate(layers):
            for nid in layer:
                node = workflow.nodes[nid]
                node_phase[nid] = node.phase or f"phase{layer_idx}"

        # Collect completed nodes' outputs, grouped by phase.
        # "Completed" means the node has a captured result — i.e. it
        # transitioned to done / blocked and we successfully
        # pulled its body off the kanban card. Failed/timed-out nodes
        # are intentionally excluded so the agent prompt doesn't
        # silently embed a half-finished output; the author should
        # gate the pipeline on success via depends_on.
        #
        # Privacy gates: nodes with privacy_gate=True are excluded from
        # the template lookup entirely. Their output is never visible to
        # downstream agents — not even as "{phase0.premortem-nikola}".
        # This prevents e.g. the premortem's failure imagination from
        # biasing position agents. The privacy is enforced here, at the
        # substitution layer, not by the prompt.
        by_phase: dict[str, dict[str, str]] = defaultdict(dict)
        states_with_result = 0
        states_without_result = 0
        privacy_dropped = 0
        orphan_dropped = 0
        for nid, st in states.items():
            if st.result is None:
                states_without_result += 1
                continue
            states_with_result += 1
            node = workflow.nodes[nid]
            # Privacy gate: skip nodes whose output should not leak
            # to downstream agents (e.g. premortem → position barrier).
            if node.privacy_gate:
                privacy_dropped += 1
                continue
            phase_label = node_phase.get(nid)
            if phase_label is None:
                orphan_dropped += 1
                # Node isn't in the topological layout (shouldn't happen
                # in practice). Skip rather than crash — the engine
                # tolerates orphan state.
                continue
            by_phase[phase_label][nid] = st.result
            # Also expose at top level for legacy {node-id} lookups.
            # Namespace collisions (e.g. a node literally named
            # "context" or "phase1") would clobber the namespace keys
            # here. We don't try to defend against that — the YAML
            # author is responsible for choosing unambiguous ids, and
            # the docstring above calls out the convention.
            lookup[nid] = st.result

        # ── Node metadata namespace ──
        # Expose {nodes.<node-id>.name} and {nodes.<node-id>.card_id}
        # so YAML authors can reference a node's disambiguated name
        # and its kanban task ID directly without manual template tricks.
        # e.g. {nodes.coder-implement.name} → "coder-implement (185117)"
        #      {nodes.coder-implement.card_id} → "t_abc1234"
        run_short_id = lookup.get("run_short_id", "")
        nodes_ns: dict[str, dict] = {}
        for nid in workflow.nodes:
            entry: dict[str, str] = {}
            if run_short_id:
                entry["name"] = f"{nid} ({run_short_id})"
            else:
                entry["name"] = nid
            # Card ID — only available for dispatched nodes
            st = states.get(nid)
            if st and st.kanban_card_id:
                entry["card_id"] = st.kanban_card_id
            else:
                entry["card_id"] = ""
            nodes_ns[nid] = entry
        lookup["nodes"] = nodes_ns

        for phase_label, members in by_phase.items():
            # Stable concatenation order: follow the topological layer
            # order, not the dict iteration order. Within a single
            # layer, fall back to the workflow.nodes dict order which
            # matches YAML declaration order.
            ordered = []
            for layer in layers:
                for nid in layer:
                    if nid in members:
                        ordered.append((nid, members[nid]))
            members_ordered = dict(ordered)
            members_ordered["all"] = "\n\n---\n\n".join(
                f"[{nid}]\n{body}" for nid, body in ordered
            )
            lookup[phase_label] = members_ordered

        # ── Diagnostics snapshot 📋 ──
        diag = {
            "run_id": run_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "lookup_built",
            "total_states": len(states),
            "states_with_result": states_with_result,
            "states_without_result": states_without_result,
            "privacy_dropped": privacy_dropped,
            "orphan_dropped": orphan_dropped,
            "namespace_labels": list(by_phase.keys()),
            "namespace_node_counts": {k: [pk for pk in v.keys() if pk != "all"] for k, v in by_phase.items()},
            "top_level_lookup_keys": [k for k in lookup.keys() if k not in ("context",)],
        }
        # Use a well-known prefix so log grepping is reliable
        print(f"\n📋 WFE:DIAG {json.dumps(diag, default=str)}", file=sys.stderr)

        return lookup

    def _resolve_template(self, text: str, lookup: dict) -> str:
        """Apply a single substitution pass over ``text``.

        Resolution rules, in order:
          1. ``{namespace.field}`` — ``lookup[namespace][field]`` if both
             keys exist. The namespace must be a string of letters /
             digits / underscores; the field is the same plus hyphens
             (so node ids like ``position-edison`` are reachable as a
             field under the ``phase1`` namespace).
          2. ``{field}`` (bare, no dot) — try ``lookup["context"][field]``
             first, then ``lookup[field]``. This is the legacy form
             (e.g. ``{question}`` in the original council.yaml) and
             makes the existing pipelines work without touching them.

        Unresolved references are left in the output unchanged. A one-
        line warning is printed to stdout per unresolved reference so
        operators notice missing upstream data instead of silently
        shipping a card with literal ``{...}`` braces.
        """

        def _replace(match: re.Match) -> str:
            if match.group("ns") is not None:
                ns, field = match.group("ns"), match.group("field")
                subfield = match.group("subfield")
                ns_val = lookup.get(ns)
                if isinstance(ns_val, dict) and field in ns_val:
                    val = ns_val[field]
                    if subfield is not None:
                        if isinstance(val, dict) and subfield in val:
                            return str(val[subfield])
                    else:
                        return str(val)
                # ── Resolution failure diagnostics ──
                ns_type = type(ns_val).__name__ if ns_val is not None else "MISSING"
                ns_keys = sorted(ns_val.keys())[:20] if isinstance(ns_val, dict) else "N/A"
                ctx_keys = sorted(lookup.get("context", {}).keys())[:10]
                top_keys = sorted(k for k in lookup.keys() if k not in ("context",))[:10]
                diag = {
                    "run_id": lookup.get("run_id", lookup.get("context", {}).get("run_id", "unknown")),
                    "event": "unresolved_namespace",
                    "template_ref": f"{{{ns}.{field}}}",
                    "ns_key": ns,
                    "field_key": field,
                    "lookup_has_ns": ns in lookup,
                    "ns_type": ns_type,
                    "ns_keys": ns_keys,
                    "ns_is_dict_and_has_field": isinstance(ns_val, dict) and field in ns_val,
                    "context_keys": ctx_keys,
                    "top_level_keys": top_keys,
                    "known_namespaces": sorted(k for k in lookup.keys() if isinstance(lookup.get(k), dict) and k != "context"),
                }
                print(f"   📋 WFE:DIAG {json.dumps(diag, default=str)}", file=sys.stderr)
                print(
                    f"   ⚠  Unresolved template {{{ns}.{field}}} "
                    f"— leaving literal"
                )
                return match.group(0)
            # Bare form — legacy compat
            bare = match.group("bare")
            ctx = lookup.get("context")
            if isinstance(ctx, dict) and bare in ctx:
                return str(ctx[bare])
            if bare in lookup:
                return str(lookup[bare])
            print(
                f"   ⚠  Unresolved template {{{bare}}} "
                f"— leaving literal"
            )
            return match.group(0)

        return self._TEMPLATE_RE.sub(_replace, text)

    def _build_task_body(self, node: WorkflowNode, workflow: "Workflow",
                          states: dict[str, "NodeState"],
                          layers: list[list[str]],
                          context: Optional[dict] = None) -> str:
        """Compose the final card body for a workflow node.

        Steps:
          1. Resolve ``{namespace.field}`` and ``{bare}`` template
             references in ``node.task`` against upstream nodes' results
             and the start-time context.
          2. Append the legacy ``\n\nContext: {json}`` footer. This is
             preserved verbatim from the pre-substitution behavior so
             the kanban card still carries the raw context dict for
             agents that prefer to read it explicitly.

        The Context footer is intentionally appended *after* template
        resolution so it doesn't get accidentally treated as a template
        (the value is JSON, which contains its own braces).
        """
        lookup = self._build_template_lookup(
            workflow, states, layers, context
        )
        resolved_task = self._resolve_template(node.task, lookup)

        # ── Post-resolution diagnostics ──
        # Count how many template variables survived unresolved
        unresolved = self._TEMPLATE_RE.findall(resolved_task)
        unresolved_formatted = []
        for parts in unresolved:
            ns, field, subfield, bare = parts[0], parts[1], parts[2], parts[3]
            if ns:
                ref = f"{{{ns}.{field}}}"
                if subfield:
                    ref = f"{{{ns}.{field}.{subfield}}}"
                unresolved_formatted.append(ref)
            else:
                unresolved_formatted.append(f"{{{bare}}}")
        if unresolved_formatted:
            diag = {
                "run_id": lookup.get("run_id", "unknown"),
                "event": "post_resolution",
                "node_id": node.id,
                "agent": node.agent,
                "unresolved_count": len(unresolved_formatted),
                "unresolved_refs": unresolved_formatted,
            }
            print(f"   📋 WFE:DIAG {json.dumps(diag, default=str)}", file=sys.stderr)

        return resolved_task

    # ── State persistence ──────────────────────────────────────

    def _init_exec_db(self) -> None:
        """Create the workflow executions and node_cards tables if they don't exist."""
        try:
            import sqlite3
            with sqlite3.connect(str(self._exec_db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS workflow_executions (
                        run_id TEXT PRIMARY KEY,
                        workflow_name TEXT NOT NULL,
                        board TEXT,
                        status TEXT NOT NULL DEFAULT 'running'
                            CHECK(status IN ('running','completed','failed','blocked')),
                        started_at TEXT,
                        finished_at TEXT,
                        error TEXT,
                        current_layer INTEGER DEFAULT 0,
                        total_layers INTEGER DEFAULT 0
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_we_status ON workflow_executions(status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_we_name ON workflow_executions(workflow_name)")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS workflow_node_cards (
                        card_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        node_id TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        FOREIGN KEY (run_id) REFERENCES workflow_executions(run_id)
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_wnc_run ON workflow_node_cards(run_id)")
        except Exception:
            pass  # Non-fatal — state files still work

    def _record_execution(self, workflow_name: str, run_id: str, board: str,
                          total_layers: int) -> None:
        """Insert a new execution record at workflow start."""
        try:
            import sqlite3
            from datetime import datetime, timezone
            with sqlite3.connect(str(self._exec_db_path)) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO workflow_executions "
                    "(run_id, workflow_name, board, status, started_at, total_layers) "
                    "VALUES (?, ?, ?, 'running', ?, ?)",
                    (run_id, workflow_name, board,
                     datetime.now(timezone.utc).isoformat(), total_layers)
                )
        except Exception as _exc:
            logger.warning("Failed to record execution to job log DB: %s", _exc)

    def _update_execution(self, run_id: str, status: str = None,
                          current_layer: int = None, error: str = None) -> None:
        """Update an execution record (layer advance, completion, failure)."""
        try:
            import sqlite3
            from datetime import datetime, timezone
            updates = []
            params = []
            if status is not None:
                updates.append("status = ?")
                params.append(status)
                if status in ("completed", "failed"):
                    updates.append("finished_at = ?")
                    params.append(datetime.now(timezone.utc).isoformat())
            if current_layer is not None:
                updates.append("current_layer = ?")
                params.append(current_layer)
            if error is not None:
                updates.append("error = ?")
                params.append(error)
            if not updates:
                return
            params.append(run_id)
            with sqlite3.connect(str(self._exec_db_path)) as conn:
                conn.execute(
                    f"UPDATE workflow_executions SET {', '.join(updates)} WHERE run_id = ?",
                    params
                )
        except Exception as _exc:
            logger.warning("Failed to record node card to job log DB: %s", _exc)

    def _record_node_card(self, card_id: str, run_id: str, node_id: str) -> None:
        """Record a card→run→node mapping in the job log DB."""
        try:
            import sqlite3
            with sqlite3.connect(str(self._exec_db_path)) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO workflow_node_cards (card_id, run_id, node_id, status) VALUES (?, ?, ?, 'pending')",
                    (card_id, run_id, node_id)
                )
        except Exception as _exc:
            logger.warning("Failed to record node card to job log DB: %s", _exc)


    def _state_path(self, workflow_name: str, run_id: str = None) -> Path:
        if run_id:
            return self.STATE_DIR / f"{workflow_name}_{run_id}_state.json"
        return self.STATE_DIR / f"{workflow_name}_state.json"

    def _save_state(self, workflow_name: str, states: dict, results: dict,
                    current_layer: int, layers: list[list[str]],
                    run_id: str = None, context: dict = None,
                    attachments: list = None, session_info: dict = None):
        """Persist engine state for crash recovery."""
        # Telemetry: capture duration_seconds + error_count for any node
        # that has reached a terminal status but hasn't been recorded yet.
        # Idempotent — running _record_node_completion on already-recorded
        # states is a no-op (duration_seconds check guards).
        for node_state in states.values():
            self._record_node_completion(node_state)
        state = {
            "workflow_name": workflow_name,
            "kanban_board": self.kanban_board,
            "current_layer": current_layer,
            "layers": layers,
            "context": context or {},
            "attachments": attachments or [],
            "states": {nid: {
                "node_id": s.node_id,
                "status": s.status,
                "kanban_card_id": s.kanban_card_id,
                "started_at": s.started_at,
                "completed_at": s.completed_at,
                "attempts": s.attempts,
                "error": s.error,

                # Round-trip the captured output so a resume can still
                # substitute {phaseN.X} / {node-id} references for
                # downstream nodes that haven't been dispatched yet.
                "result": s.result,
                # Telemetry: populated by _record_node_completion before
                # this save runs. Surface to workflow_status for cost /
                # bottleneck analysis.
                "duration_seconds": s.duration_seconds,
                "error_count": s.error_count,
            } for nid, s in states.items()},
            "results": results,
            "run_id": run_id or "",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        # Persist session info so hooks can create subscriptions for
        # final-layer cards even when the supervisor runs as a subprocess.
        if session_info:
            state["session_info"] = session_info
        else:
            # Preserve session_info from any existing state file that has it.
            # Different supervisors may use different run_ids, so scan all.
            try:
                for sf in sorted(self.STATE_DIR.glob(f"{workflow_name}_*_state.json"),
                                 key=lambda p: p.stat().st_mtime, reverse=True):
                    existing = json.loads(sf.read_text())
                    if existing.get("session_info"):
                        state["session_info"] = existing["session_info"]
                        break
            except Exception:
                pass
        with open(self._state_path(workflow_name, run_id), "w") as f:
            json.dump(state, f, indent=2)
        # Retention: prune state files beyond STATE_RETENTION_PER_WORKFLOW
        # so disk usage stays bounded. No-op if nothing to prune.
        self._prune_old_runs()

    def _load_state(self, workflow_name: str, run_id: str = None) -> Optional[dict]:
        """Load persisted state if it exists."""
        path = self._state_path(workflow_name, run_id)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def _clear_state(self, workflow_name: str, run_id: str = None):
        """Remove state file after successful completion."""
        path = self._state_path(workflow_name, run_id)
        if path.exists():
            path.unlink()

    def _find_latest_state(self, workflow_name: str) -> Optional[dict]:
        """Find the most recent state file for a workflow (supports parallel runs)."""
        # Prefer run_id-tagged files
        tagged = sorted(self.STATE_DIR.glob(f"{workflow_name}_*_state.json"))
        if tagged:
            with open(tagged[-1]) as f:
                return json.load(f)
        # Fall back to legacy untagged file
        legacy = self.STATE_DIR / f"{workflow_name}_state.json"
        if legacy.exists():
            with open(legacy) as f:
                return json.load(f)
        return None

    # ── Output validation ──────────────────────────────────────

    def _validate_outputs(self, node: WorkflowNode,
                          state: NodeState) -> list[str]:
        """Check expected artifact outputs for a completed node.

        Returns a list of warning strings. Empty list = all checks pass.
        Checks are best-effort — a missing file is a warning, not a
        failure. The engine logs the warning and continues.

        For each output entry:
          1. Resolve {run_id} in the path if the node belongs to a
             workflow with a run_id.
          2. Stat the resolved path in the agent's workspace (same
             dir: logic as create_kanban_card).
          3. If required=True and file doesn't exist, add a warning.
          4. If schema is set, do a basic content check.
        """
        warnings = []
        if not node.outputs:
            return warnings

        # Resolve agent workspace path (same logic as create_kanban_card)
        if node.agent:
            hermes_home = os.environ.get("HERMES_HOME")
            if hermes_home:
                profiles_root = Path(hermes_home).parent
            else:
                profiles_root = Path.home() / ".hermes" / "profiles"
            agent_ws = profiles_root / node.agent / "workspace"
        else:
            return warnings  # No agent = no workspace to check

        for entry in node.outputs:
            if isinstance(entry, str):
                raw_path = entry
                required = True
                schema = None
            else:
                raw_path = entry.get("path", "")
                required = entry.get("required", False)
                schema = entry.get("schema")

            if not raw_path:
                continue

            # Resolve {run_id} and {date} in the path
            resolved = raw_path
            run_id = state.node_id.split("_")[0] if "_" in state.node_id else "run"
            from datetime import date as _date
            resolved = resolved.replace("{run_id}", run_id)
            resolved = resolved.replace("{date}", _date.today().isoformat())

            full_path = agent_ws / resolved
            if not full_path.exists():
                if required:
                    msg = f"Expected output not found: {resolved}"
                    print(f"   ⚠  [{node.id}] {msg}")
                    warnings.append(msg)
                continue

            # Basic schema check for JSON files
            if schema == "json":
                try:
                    data = json.loads(full_path.read_text())
                    if not data:
                        warnings.append(f"Output {resolved} is empty JSON")
                        print(f"   ⚠  [{node.id}] Output {resolved} is empty JSON")
                except (json.JSONDecodeError, ValueError) as e:
                    msg = f"Output {resolved} failed JSON validation: {e}"
                    warnings.append(msg)
                    print(f"   ⚠  [{node.id}] {msg}")

        return warnings

    # ── Auxiliary analyst (LLM-backed, best-effort) ──────────────

    def _try_escalation_analysis(self, workflow: Workflow,
                                  verify_nid: str, verify_state: NodeState,
                                  context: dict = None):
        """Try LLM analysis of a deadlocked revision loop. Best-effort —
        failure is silent; the engine continues with mechanical escalation."""
        try:
            from plugins.workflow.analyst import analyze_escalation
        except Exception:
            return  # Auxiliary module not available

        # Build loop history from the full list of LOOP rejections
        loop_history = verify_state.error or "No review history available"
        project = (context or {}).get("project", "unknown")

        outcome = analyze_escalation(
            project=project,
            gate=verify_nid,
            verify_node=verify_nid,
            loop_history=loop_history,
        )

        if outcome.success and outcome.result:
            summary = outcome.result.get("summary", "")
            sticking = outcome.result.get("sticking_point", "")
            actions = outcome.result.get("suggested_actions", [])
            escalation = outcome.result.get("recommended_escalation", "sherlock_can_resolve")

            print(f"   🧠 Escalation analysis: {summary}")
            if sticking:
                print(f"      Sticking point: {sticking}")
            for i, action in enumerate(actions[:3], 1):
                print(f"      Option {i}: {action}")
            if escalation == "needs_randy":
                print(f"   ⚠  Analyst recommends Randy involvement")
        else:
            print(f"   ⚠  Escalation analysis unavailable — "
                  f"Sherlock must review manually")

    def _try_loop_decision(self, verify_node: "WorkflowNode",
                           revision_node: "WorkflowNode",
                           rejection: str) -> str:
        """Ask the analyst whether a LOOP rejection is genuine.

        Returns ``"loop"`` (re-dispatch revision) or ``"proceed"``
        (mark verify as done and advance).  Falls back to ``"loop"``
        when the analyst is unavailable or returns an unparseable
        response — conservative default that preserves the existing
        mechanical behaviour.
        """
        try:
            from plugins.workflow.analyst import analyze_loop_decision
        except Exception:
            return "loop"

        outcome = analyze_loop_decision(
            verify_task=verify_node.task,
            rejection=rejection,
            revision_task=revision_node.task,
        )

        if outcome.success and isinstance(outcome.result, dict):
            decision = outcome.result.get("decision", "loop")
            reason = outcome.result.get("reason", "")
            confidence = outcome.result.get("confidence", "low")
            if decision == "proceed":
                print(f"   🧠 Analyst: proceed — {reason} (confidence: {confidence})")
                return "proceed"
            else:
                print(f"   🧠 Analyst: loop — {reason} (confidence: {confidence})")
                return "loop"

        # Fall back to mechanical loop
        return "loop"

    def _try_block_notify(self, workflow: "Workflow", nid: str,
                          state: "NodeState", rejection: str,
                          context: dict = None):
        """When a non-LOOP block is detected, call the analyst and push
        the assessment to the calling agent's session.

        This is the mechanism that tells the calling agent something
        unexpected went wrong — without the agent having to poll or
        subscribe to every card.
        """
        try:
            from plugins.workflow.analyst import analyze_block_notification
        except Exception:
            return

        node = workflow.nodes[nid]
        ctx = context or {}
        project = ctx.get("project", "")
        repo = ctx.get("repo", "")
        workflow_context = f"Project: {project}, Repository: {repo}" if project else ""

        outcome = analyze_block_notification(
            node_id=nid,
            workflow_name=workflow.name,
            node_task=node.task,
            block_reason=rejection,
            workflow_context=workflow_context,
        )

        if not outcome.success or not isinstance(outcome.result, dict):
            print(f"   ⚠  Block analysis unavailable for {nid}")
            return

        severity = outcome.result.get("severity", "warning")
        summary = outcome.result.get("summary", "")
        detail = outcome.result.get("detail", "")
        action = outcome.result.get("suggested_action", "")

        # Build the notification message
        emoji = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(severity, "⚠️")
        msg = (
            f"{emoji} Workflow anomaly: **{nid}** blocked in **{workflow.name}**\n"
            f"**Summary:** {summary}\n"
            f"**Detail:** {detail}\n"
            f"**Action:** {action}"
        )

        print(f"   {emoji} Block notified to calling agent: {summary}")

        # Push to the calling session via the adapter
        platform = ctx.get("platform", "")
        chat_id = ctx.get("chat_id", "")
        thread_id = ctx.get("thread_id", "")
        profile = ctx.get("profile", "")

        if not platform or not chat_id:
            return

        try:
            from gateway.platforms import get_adapter as get_platform_adapter
            adapter = get_platform_adapter(platform)
            if adapter:
                import asyncio
                meta = {}
                if thread_id:
                    meta["thread_id"] = thread_id
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(adapter.send(chat_id, msg, metadata=meta))
                except RuntimeError:
                    asyncio.run(adapter.send(chat_id, msg, metadata=meta))
        except Exception as exc:
            print(f"   ⚠  Failed to notify calling agent: {exc}")

    def _try_failure_analysis(self, node: WorkflowNode, state: NodeState,
                               elapsed_sec: float):
        """Try LLM diagnosis of a node failure. Best-effort — silent on failure."""
        # Synthetic gates don't fail (they auto-complete), so the analyst
        # has nothing useful to say about them. Returning early also
        # avoids a type error passing None to analyze_failure's `agent`
        # argument.
        if node.synthetic:
            return
        try:
            from plugins.workflow.analyst import analyze_failure
        except Exception:
            return

        outcome = analyze_failure(
            node_id=node.id,
            agent=node.agent,
            task=node.task[:500],
            timeout_minutes=node.timeout_minutes,
            elapsed=f"{elapsed_sec:.0f}s",
            error=state.error or "No error details",
        )

        if outcome.success and outcome.result:
            cause = outcome.result.get("likely_cause", "unknown")
            category = outcome.result.get("cause_category", "unknown")
            fix = outcome.result.get("suggested_fix", "")
            retry = outcome.result.get("should_retry", False)

            print(f"   🧠 Failure diagnosis [{category}]: {cause}")
            if fix:
                print(f"      Fix: {fix}")
            if retry:
                print(f"      Analyst suggests retry")
        # Silent on failure — mechanical handling continues

    def _try_status_summary(self, workflow_name: str,
                             saved_state: dict) -> Optional[str]:
        """Try LLM summary of pipeline state. Returns summary text or None."""
        try:
            from plugins.workflow.analyst import analyze_status
        except Exception:
            return None

        outcome = analyze_status(
            pipeline_name=workflow_name,
            state_json=json.dumps(saved_state, indent=2)[:8000],
        )

        if outcome.success and outcome.result:
            status = outcome.result.get("overall_status", "unknown")
            alerts = outcome.result.get("attention_needed", [])
            eta = outcome.result.get("estimated_completion", "")

            lines = [f"Pipeline: {workflow_name} | Status: {status}"]
            if eta:
                lines.append(f"Estimated: {eta}")
            for alert in alerts:
                lines.append(f"⚠ {alert}")
            return "\n".join(lines)
        return None

    def _classify_block_reason(self, nid: str, body: str,
                                workflow: "Workflow",
                                context: dict = None) -> bool:
        """Use the auxiliary to classify why a reviewer blocked.

        Returns True if the block is a quality review result (enrich upstream,
        unblock reviewer, continue loop). Returns False if it's a technical
        issue (notify calling agent).

        Falls back to heuristic: if the body contains review-like keywords,
        treat as quality block. Otherwise treat as technical.
        """
        try:
            from plugins.workflow.analyst import analyze_block_notification
        except Exception:
            return self._heuristic_quality_check(body)

        ctx = context or {}
        project = ctx.get("project", "")
        workflow_context = f"Project: {project}" if project else ""

        outcome = analyze_block_notification(
            node_id=nid,
            workflow_name=workflow.name,
            node_task=workflow.nodes[nid].task if nid in workflow.nodes else "",
            block_reason=body[:500],
            workflow_context=workflow_context,
        )

        if outcome.success and isinstance(outcome.result, dict):
            block_type = outcome.result.get("block_type", "quality")
            return block_type == "quality"

        return self._heuristic_quality_check(body)

    @staticmethod
    def _heuristic_quality_check(body: str) -> bool:
        """Fallback heuristic: check if body looks like review feedback."""
        body_lower = body.lower()
        quality_keywords = [
            "review", "feedback", "failed", "issue", "bug", "error",
            "incorrect", "wrong", "fix", "improve", "quality",
            "does not", "should", "expected", "actual",
        ]
        return any(kw in body_lower for kw in quality_keywords)

    # ── Validation ─────────────────────────────────────────────

    def validate(self, workflow_name: str) -> dict:
        """
        Validate a workflow without executing. Checks:
        - YAML loads cleanly (syntax + structure)
        - All dependency references resolve
        - No cycles in DAG
        - All agents referenced exist (best-effort)
        - Required fields present on all nodes
        - Reviews references resolve to valid node IDs
        - max_retries values are positive integers
        """
        result = {"valid": True, "issues": [], "layers": 0, "nodes": 0}

        # Step 1: YAML syntax check (catches malformed YAML before load)
        yaml_path = self.workflows_dir / f"{workflow_name}.yaml"
        if not yaml_path.exists():
            result["valid"] = False
            result["issues"].append(f"Pipeline file not found: {yaml_path}")
            return result

        try:
            raw_text = yaml_path.read_text()
        except Exception as e:
            result["valid"] = False
            result["issues"].append(f"Cannot read YAML file: {e}")
            return result

        try:
            raw = yaml.safe_load(raw_text)
        except yaml.YAMLError as e:
            result["valid"] = False
            result["issues"].append(f"YAML syntax error: {e}")
            return result

        if not isinstance(raw, dict):
            result["valid"] = False
            result["issues"].append("YAML root must be a mapping (key: value)")
            return result

        try:
            workflow = self.load_workflow(workflow_name)
        except Exception as e:
            result["valid"] = False
            result["issues"].append(f"YAML load failed: {e}")
            return result

        result["nodes"] = len(workflow.nodes)

        # Check dependency references
        for nid, node in workflow.nodes.items():
            for dep in node.depends_on:
                if dep not in workflow.nodes:
                    result["valid"] = False
                    result["issues"].append(
                        f"Node '{nid}' depends on unknown node '{dep}'"
                    )

        # Check for cycles
        try:
            layers = self.topological_sort(workflow)
            result["layers"] = len(layers)
        except CycleDetectedError as e:
            result["valid"] = False
            result["issues"].append(str(e))
            return result
        except ValueError as e:
            result["valid"] = False
            result["issues"].append(str(e))
            return result

        # Check agents exist (best-effort — checks profiles dir)
        profiles_dir = Path.home() / ".hermes" / "profiles"
        for nid, node in workflow.nodes.items():
            # Synthetic gate nodes have no agent — skip the profile
            # existence check entirely. This is what makes the loader
            # able to accept synthetic nodes without a "real" agent
            # name to validate against.
            if node.synthetic:
                continue
            agent_profile = profiles_dir / node.agent
            if not agent_profile.exists():
                result["issues"].append(
                    f"Node '{nid}': agent '{node.agent}' profile not found at {agent_profile}"
                )

        # Check reviews references
        for nid, node in workflow.nodes.items():
            for review_entry in (node.reviews or []):
                if isinstance(review_entry, dict):
                    rev_id = review_entry.get("review", "")
                    if rev_id and rev_id not in workflow.nodes:
                        result["issues"].append(
                            f"Node '{nid}' reviews unknown node '{rev_id}'"
                        )
                    max_retries = review_entry.get("max_retries")
                    if max_retries is not None and (not isinstance(max_retries, int) or max_retries < 1):
                        result["issues"].append(
                            f"Node '{nid}' review '{rev_id}' has invalid max_retries={max_retries}"
                        )
                elif isinstance(review_entry, str):
                    if review_entry not in workflow.nodes:
                        result["issues"].append(
                            f"Node '{nid}' reviews unknown node '{review_entry}'"
                        )
                else:
                    result["issues"].append(
                        f"Node '{nid}' has invalid review entry: {review_entry}"
                    )

        # Check max_retries values
        for nid, node in workflow.nodes.items():
            if node.max_retries is not None:
                if not isinstance(node.max_retries, int) or node.max_retries < 1:
                    result["issues"].append(
                        f"Node '{nid}' has invalid max_retries={node.max_retries}"
                    )

        # Check attachment references against declared attachments
        declared_attachments = {a.get("name", "") for a in raw.get("attachments", [])}
        for nid, node in workflow.nodes.items():
            if node.attachment is not None:
                if declared_attachments and node.attachment not in declared_attachments:
                    result["issues"].append(
                        f"Node '{nid}' references attachment '{node.attachment}' "
                        f"which is not declared in attachments section"
                    )

        # Check all references against declarations
        declared_inputs = {i.get("name", "") for i in raw.get("inputs", [])}
        declared_nodes = set(workflow.nodes.keys())
        builtin_ns = {"context", "run_id", "date", "inputs"}

        for nid, node in workflow.nodes.items():
            # Extract template references from task text (both {ns.field} and {bare})
            ns_refs = re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\.[A-Za-z0-9_\-]+\}", node.task)
            bare_refs = re.findall(r"(?<!\.)\{([A-Za-z_][A-Za-z0-9_\-]*)\}(?!\.)", node.task)
            # Filter out bare refs that are actually ns.field (already checked)
            bare_refs = [r for r in bare_refs if r not in [f.split(".")[0] for f in ns_refs]]

            all_refs = set(ns_refs) | set(bare_refs)
            for ref in all_refs:
                if ref in builtin_ns:
                    continue
                if ref in declared_inputs or ref in declared_nodes:
                    continue
                result["issues"].append(
                    f"Node '{nid}' references '{{{ref}}}' which is not "
                    f"a declared input or node"
                )

        # Check that all declared inputs are actually used in node tasks
        all_task_text = " ".join(
            node.task + " " + str(node.description)
            for node in workflow.nodes.values()
        )
        for inp in raw.get("inputs", []):
            inp_name = inp.get("name", "")
            if inp_name and f"{{{inp_name}}}" not in all_task_text:
                result["issues"].append(
                    f"Declared input '{inp_name}' is never referenced in any node task"
                )

        # Check for undeclared inputs used in node tasks
        declared_input_names = {i.get("name", "") for i in raw.get("inputs", [])}
        bare_refs_in_tasks = set(re.findall(r"(?<!\.)\{([A-Za-z_][A-Za-z0-9_\-]*)\}(?!\.)", all_task_text))
        for ref in bare_refs_in_tasks:
            if ref in ("context", "run_id", "date", "inputs"):
                continue
            if ref not in declared_input_names and ref not in workflow.nodes:
                result["issues"].append(
                    f"Node tasks reference '{{{ref}}}' which is not a declared input"
                )

        # Check that all declared attachments are actually used
        for att in raw.get("attachments", []):
            att_name = att.get("name", "")
            if att_name:
                used_in_attachment_field = any(
                    getattr(n, "attachment", None) == att_name
                    for n in workflow.nodes.values()
                )
                if not used_in_attachment_field:
                    result["issues"].append(
                        f"Declared attachment '{att_name}' is never referenced by any node"
                    )

        # Check reviews/depends_on conflicts
        for nid, node in workflow.nodes.items():
            for review_entry in (node.reviews or []):
                rev_id = review_entry if isinstance(review_entry, str) else review_entry.get("review", "")
                if not rev_id:
                    continue
                # Check if reviewer depends on this node (circular)
                reviewer = workflow.nodes.get(rev_id)
                if reviewer and nid in reviewer.depends_on:
                    # This is acceptable — reviewer nodes are skipped in the
                    # initial card creation and dispatched by the supervisor
                    # on demand. The depends_on ensures the reviewer runs
                    # after the creator in the DAG ordering.
                    result["issues"].append(
                        f"Node '{nid}' reviews '{rev_id}' and '{rev_id}' depends_on '{nid}' — "
                        f"reviewer will be dispatched by supervisor (not DAG)"
                    )
                # Check if this node depends on reviewer (conflict)
                if rev_id in node.depends_on:
                    result["valid"] = False
                    result["issues"].append(
                        f"Node '{nid}' depends_on '{rev_id}' and also reviews '{rev_id}' — "
                        f"reviewer would be dispatched twice"
                    )

        # incomplete_branch rule (adapted from itechmeat/hermes-workflows).
        # Catches non-terminal nodes that rely on the implicit default
        # ``fallback_on_timeout="skip"`` — which silently cascades skip to
        # all downstream nodes. Authors should be intentional about how a
        # node handles timeout / failure when other nodes depend on it.
        # Non-fatal: surfaces as an issue for the caller to act on, but
        # doesn't flip ``valid`` (existing fleet workflows commonly omit
        # the explicit declaration — warn, don't break).
        try:
            yaml_path = self.workflows_dir / f"{workflow_name}.yaml"
            raw_yaml = yaml.safe_load(yaml_path.read_text()) if yaml_path.exists() else None
            nodes_raw = (raw_yaml or {}).get("nodes", {}) if raw_yaml else {}
        except Exception:
            nodes_raw = {}
        for nid, node in workflow.nodes.items():
            if node.synthetic:
                continue
            # Terminal = no downstream consumers; skip the check.
            has_downstream = any(
                nid in other.depends_on
                for other_id, other in workflow.nodes.items()
                if other_id != nid
            )
            if not has_downstream:
                continue
            raw_node = nodes_raw.get(nid, {})
            if "fallback_on_timeout" not in raw_node:
                result["issues"].append(
                    f"Node '{nid}' has downstream consumers but no explicit "
                    f"fallback_on_timeout in YAML. Add one of: skip | degraded "
                    f"| retry to make failure routing intentional, not implicit."
                )

        # ── when: dependency validation (non-fatal) ──
        # Warn if a when: expression references a node that is not in
        # the current node's depends_on list.  This catches missing
        # dependency declarations — the engine would still skip nodes
        # with failed deps, but the when: condition might silently
        # evaluate to a stale value instead of being properly gated.
        for nid, node in workflow.nodes.items():
            if not node.when:
                continue
            # Extract all {node-id.field} references from the when expr
            refs = re.findall(
                r"\{([A-Za-z_][A-Za-z0-9_]*)\.[A-Za-z0-9_\-]+\}",
                node.when,
            )
            for ref_nid in set(refs):
                if ref_nid == "context":
                    continue  # context is always available
                if ref_nid not in workflow.nodes:
                    continue  # unknown node — separate issue
                if ref_nid not in node.depends_on and ref_nid != nid:
                    result["issues"].append(
                        f"Node '{nid}' has when: referencing '{ref_nid}' "
                        f"but does not declare it in depends_on. Add "
                        f"'{ref_nid}' to depends_on or use a context "
                        f"variable instead."
                    )

        return result

    # ── When-condition evaluation ──────────────────────────────

    # Operator tokens recognised in when: expressions
    _WHEN_OPS = frozenset({"==", "!=", ">", "<", ">=", "<=", "contains", "starts_with"})
    _WHEN_KEYWORDS = frozenset({"and", "or", "not", "in", "True", "False", "None"})

    def _resolve_when_references(self, when_expr: str, states: dict,
                                  context: Optional[dict] = None) -> str:
        """Resolve ``{node-id.field}`` and ``{context.key}`` in a
        ``when:`` expression, replacing them with their literal values
        (quoted strings, raw numbers).

        Unresolved references are left as literal text — the evaluator
        will treat unknown identifiers as string values.
        """
        # Build a lookup of node state fields
        when_lookup: dict = {}
        for nid, st in states.items():
            when_lookup[nid] = {
                "status": st.status,
                "result": st.result or "",
                "error": st.error or "",
                "attempts": st.attempts,
                "duration_seconds": st.duration_seconds,
                "error_count": st.error_count,
            }
        when_lookup["context"] = dict(context or {})

        def _replace(match: re.Match) -> str:
            if match.group("ns") is not None:
                ns, field = match.group("ns"), match.group("field")
                ns_val = when_lookup.get(ns)
                if isinstance(ns_val, dict) and field in ns_val:
                    val = ns_val[field]
                    if isinstance(val, str):
                        return f'"{val}"'
                    if val is None:
                        return "None"
                    return str(val)
                return match.group(0)  # Leave unresolved
            bare = match.group("bare")
            ctx = when_lookup.get("context")
            if isinstance(ctx, dict) and bare in ctx:
                val = ctx[bare]
                if isinstance(val, str):
                    return f'"{val}"'
                if val is None:
                    return "None"
                return str(val)
            if bare in when_lookup:
                val = when_lookup[bare]
                if isinstance(val, str):
                    return f'"{val}"'
                if val is None:
                    return "None"
                return str(val)
            return match.group(0)

        return self._TEMPLATE_RE.sub(_replace, when_expr)

    def _tokenize_when(self, expr: str) -> list:
        """Tokenize a resolved when: expression into a flat list of
        (type, value) tuples.  Unquoted identifiers that are not
        reserved keywords are treated as string literals.
        """
        tokens = []
        i = 0
        n = len(expr)
        while i < n:
            # Whitespace — skip
            if expr[i].isspace():
                i += 1
                continue
            # Quoted string
            if expr[i] == '"':
                j = i + 1
                while j < n and expr[j] != '"':
                    if expr[j] == '\\':
                        j += 1
                    j += 1
                tokens.append(("STRING", expr[i + 1:j]))
                i = j + 1
                continue
            # Number (possibly negative)
            if expr[i].isdigit() or (
                expr[i] == '-' and i + 1 < n and expr[i + 1].isdigit()
                and (not tokens or tokens[-1][0] in ("OP", "KEYWORD", "BRACKET", "COMMA"))
            ):
                j = i
                if expr[j] == '-':
                    j += 1
                while j < n and (expr[j].isdigit() or expr[j] == '.'):
                    j += 1
                num_str = expr[i:j]
                tokens.append(
                    ("NUMBER", float(num_str) if '.' in num_str else int(num_str))
                )
                i = j
                continue
            # Brackets
            if expr[i] in '[]()':
                tokens.append(("BRACKET", expr[i]))
                i += 1
                continue
            # Comma
            if expr[i] == ',':
                tokens.append(("COMMA", ","))
                i += 1
                continue
            # Multi-char operators (>=, <=, !=, ==) — check before single-char
            if i + 1 < n and expr[i:i+2] in ("==", "!=", ">=", "<="):
                tokens.append(("OP", expr[i:i+2]))
                i += 2
                continue
            # Single-char operators
            if expr[i] in "><":
                tokens.append(("OP", expr[i]))
                i += 1
                continue
            # Word (identifier / keyword / operator-name)
            j = i
            while j < n and not expr[j].isspace() and expr[j] not in '[],();':
                j += 1
            word = expr[i:j]
            if word in self._WHEN_OPS:
                tokens.append(("OP", word))
            elif word in self._WHEN_KEYWORDS:
                tokens.append(("KEYWORD", word))
            else:
                # Unknown identifier → string literal
                tokens.append(("STRING", word))
            i = j
        return tokens

    def _eval_when_tokens(self, tokens: list) -> bool:
        """Evaluate a tokenized when: expression via recursive descent.

        Grammar (precedence low → high):
            or_expr  → and_expr ('or' and_expr)*
            and_expr → not_expr ('and' not_expr)*
            not_expr → 'not' not_expr | in_expr
            in_expr  → comparison ('in' '[' list ']')?
            comparison → atom (op atom)?
            atom      → STRING | NUMBER | 'True' | 'False' | 'None'
                        | '(' or_expr ')'
        """
        pos = [0]

        def _peek():
            return tokens[pos[0]] if pos[0] < len(tokens) else None

        def _consume():
            t = tokens[pos[0]]
            pos[0] += 1
            return t

        def _to_num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        def _parse_or():
            left = _parse_and()
            while _peek() == ("KEYWORD", "or"):
                _consume()
                right = _parse_and()
                left = left or right
            return left

        def _parse_and():
            left = _parse_not()
            while _peek() == ("KEYWORD", "and"):
                _consume()
                right = _parse_not()
                left = left and right
            return left

        def _parse_not():
            if _peek() == ("KEYWORD", "not"):
                _consume()
                return not _parse_not()
            return _parse_in()

        def _parse_in():
            left = _parse_comparison()
            if _peek() == ("KEYWORD", "in"):
                _consume()
                # Expect '['
                if _peek() != ("BRACKET", "["):
                    raise ValueError("Expected '[' after 'in'")
                _consume()
                items = []
                while _peek() and _peek() != ("BRACKET", "]"):
                    items.append(_parse_atom())
                    if _peek() == ("COMMA", ","):
                        _consume()
                if _peek() != ("BRACKET", "]"):
                    raise ValueError("Expected ']' to close list")
                _consume()
                return left in items
            return left

        def _parse_comparison():
            left = _parse_atom()
            pk = _peek()
            if pk and pk[0] == "OP" and pk[1] in self._WHEN_OPS:
                op = _consume()[1]
                right = _parse_atom()
                if op == "==":
                    return left == right
                elif op == "!=":
                    return left != right
                elif op == ">":
                    return _to_num(left) > _to_num(right)
                elif op == "<":
                    return _to_num(left) < _to_num(right)
                elif op == ">=":
                    return _to_num(left) >= _to_num(right)
                elif op == "<=":
                    return _to_num(left) <= _to_num(right)
                elif op == "contains":
                    return str(right) in str(left)
                elif op == "starts_with":
                    return str(left).startswith(str(right))
            return left

        def _parse_atom():
            pk = _peek()
            if pk is None:
                raise ValueError("Unexpected end of when: expression")
            if pk[0] == "NUMBER":
                return _consume()[1]
            if pk[0] == "STRING":
                return _consume()[1]
            if pk[0] == "KEYWORD" and pk[1] in ("True", "False", "None"):
                val = _consume()[1]
                return {"True": True, "False": False, "None": None}[val]
            if pk[0] == "BRACKET" and pk[1] == "(":
                _consume()
                val = _parse_or()
                if _peek() != ("BRACKET", ")"):
                    raise ValueError("Expected ')' in when: expression")
                _consume()
                return val
            raise ValueError(f"Unexpected token in when: expression: {pk}")

        return bool(_parse_or())

    def evaluate_when(self, when_expr: str, node: "WorkflowNode",
                      states: dict, context: Optional[dict] = None,
                      layers: list = None,
                      workflow: "Workflow" = None) -> bool:
        """Evaluate a node's ``when:`` condition against the current
        workflow state.

        Returns True when the node should dispatch (truthy expression
        or empty ``when``), False when it should be skipped.
        """
        if not when_expr or not when_expr.strip():
            return True  # Empty = always run

        try:
            resolved = self._resolve_when_references(
                when_expr, states, context
            )
            tokens = self._tokenize_when(resolved)
            if not tokens:
                return True  # Empty after resolution = always run
            return self._eval_when_tokens(tokens)
        except Exception as exc:
            # Fail open — evaluation error defaults to skip to avoid
            # dispatching a node whose condition couldn't be checked.
            print(
                f"   ⚠  when: evaluation error for "
                f"'{when_expr}': {exc} — skipping node",
                file=sys.stderr,
            )
            return False

    def _review_waiting_loop(self, workflow: "Workflow", states: dict,
                              implement_nid: str) -> bool:
        """Poll implement card until it re-blocks 'pending review' or
        all reviewers reach terminal states. Called from the post-monitor
        review check when a review iteration just completed.
        Returns True if the reviewer was unblocked (advance to next layer),
        False if reviewers are terminal (advance to next layer anyway)."""
        implement_state = states[implement_nid]
        max_wait_polls = int((workflow.nodes[implement_nid].timeout_minutes * 120) / self.POLL_INTERVAL)
        wait_polls = 0
        while wait_polls < max_wait_polls:
            time.sleep(self.POLL_INTERVAL)
            wait_polls += 1
            # Check if reviewer reached terminal state
            all_reviewers_done = True
            for rev_entry in workflow.nodes[implement_nid].reviews:
                rev_id = rev_entry if isinstance(rev_entry, str) else rev_entry.get("review", "")
                if rev_id and rev_id in states:
                    if states[rev_id].status not in ("done", "skipped", "failed", "timed_out"):
                        all_reviewers_done = False
                        break
            if all_reviewers_done:
                print(f"   ✓ All reviewers terminal — advancing to next layer")
                return False
            # Check if reviewer was claimed by dispatcher (transitioned
            # from "ready" to "running")
            for rev_entry in workflow.nodes[implement_nid].reviews:
                rev_id = rev_entry if isinstance(rev_entry, str) else rev_entry.get("review", "")
                if rev_id and rev_id in states:
                    rev_state = states[rev_id]
                    if rev_state.status == "running":
                        print(f"   🔄 {rev_id} claimed by dispatcher — monitoring layer")
                        return True
                    if rev_state.kanban_card_id:
                        try:
                            card = self.get_card_status(rev_state.kanban_card_id)
                            card_status = card.get("status", "").lower()
                            if card_status == "running":
                                rev_state.status = "running"
                                print(f"   🔄 {rev_id} claimed by dispatcher — monitoring layer")
                                return True
                        except Exception:
                            pass
            # Check if implement re-blocked "pending review"
            if implement_state.kanban_card_id:
                try:
                    card = self.get_card_status(implement_state.kanban_card_id)
                    card_status = card.get("status", card.get("column", "unknown")).lower()
                    if card_status == "blocked":
                        if self._check_pending_review(implement_state.kanban_card_id):
                            print(f"   📋 {implement_nid} re-blocked pending review — unblocking reviewer")
                            # Unblock the reviewer
                            for rev_entry in workflow.nodes[implement_nid].reviews:
                                rev_id = rev_entry if isinstance(rev_entry, str) else rev_entry.get("review", "")
                                if rev_id and rev_id in states:
                                    rev_state = states[rev_id]
                                    if rev_state.kanban_card_id and rev_state.status == "blocked":
                                        with kanban_db.connect_closing(board=self.kanban_board) as _conn:
                                            kanban_db.unblock_task(_conn, rev_state.kanban_card_id)
                                        rev_state.status = "running"
                                        rev_state.completed_at = None
                                        rev_state.result = None
                                        print(f"   🔓 {rev_id} unblocked — reviewer re-engaged")
                            # Wait for the dispatcher to claim the reviewer
                            # before returning, so the next layer's
                            # _monitor_layer finds it as "running".
                            for rev_entry in workflow.nodes[implement_nid].reviews:
                                rev_id = rev_entry if isinstance(rev_entry, str) else rev_entry.get("review", "")
                                if rev_id and rev_id in states:
                                    rev_state = states[rev_id]
                                    if rev_state.kanban_card_id:
                                        for _ in range(10):  # Wait up to ~30s
                                            time.sleep(3)
                                            try:
                                                card = self.get_card_status(rev_state.kanban_card_id)
                                                card_status = card.get("status", "").lower()
                                                if card_status == "running":
                                                    rev_state.status = "running"
                                                    print(f"   🔄 {rev_id} claimed by dispatcher — monitoring layer")
                                                    return True
                                            except Exception:
                                                pass
                            return True
                except Exception:
                    pass
        # While loop exited without break — reviewers done or timeout
        return False

    def _has_active_review(self, workflow: "Workflow", states: dict,
                            layer: list[str] = None,
                            require_ready_implement: bool = False) -> bool:
        """Check if any node has an active (running/blocked/ready) reviewer.

        When ``layer`` is provided, only checks nodes in that layer.
        When ``require_ready_implement`` is True, only returns True when
        the implement node is in "ready" status (post-enrichment state).
        """
        for nid, state in states.items():
            if layer and nid not in layer:
                continue
            node = workflow.nodes.get(nid)
            if not node or not node.reviews:
                continue
            if require_ready_implement and state.status != "ready":
                continue
            for rev_entry in node.reviews:
                rev_id = rev_entry if isinstance(rev_entry, str) else rev_entry.get("review", "")
                if not rev_id or rev_id not in states:
                    continue
                rev_state = states[rev_id]
                if rev_state.status in ("running", "blocked", "ready"):
                    return True
                if rev_state.kanban_card_id:
                    try:
                        card = self.get_card_status(rev_state.kanban_card_id)
                        card_status = card.get("status", "").lower()
                        if card_status in ("running", "blocked", "ready"):
                            return True
                    except Exception:
                        pass
        return False

    # ── Execution ──────────────────────────────────────────────

    def execute(self, workflow_name: str, context: dict = None,
                start_node: str = None, dry_run: bool = False,
                resume: bool = False, board: str = None,
                inputs: dict = None,
                fire_and_forget: bool = False,
                run_id: Optional[str] = None,
                attachments: list = None,
                session_info: dict = None) -> dict:
        """
        Run a workflow to completion. Supports revision loops via
        the LOOP:<target> convention in block reasons.

        When ``fire_and_forget=True``, creates kanban cards for all
        nodes across all layers and returns immediately without entering
        the monitoring loop.  The kanban dispatcher picks up ready cards
        and spawns workers; the gateway notifier pushes terminal events
        back to the originating session via the kanban subscription on
        the final-layer node(s).  Use ``workflow_status`` to check
        progress.

        Board resolution priority:
          1. Workflow YAML ``kanban_board`` field
          2. ``board`` parameter passed at invocation
          3. Auto-create ``wf_<workflow_name>``

        ``inputs`` are merged into context and available as
        ``{inputs.<key>}`` template substitutions across all nodes.

        Returns execution summary: {node_id: final_status, ...}
        """
        workflow = self.load_workflow(workflow_name)

        # Three-tier board resolution — caller-passed board wins
        if board:
            # Tier 1: Caller passes a board at invocation
            from hermes_cli.kanban_db import _normalize_board_slug
            self.kanban_board = _normalize_board_slug(board)
        elif workflow.kanban_board:
            # Tier 2: YAML declares a board
            self.kanban_board = workflow.kanban_board
        else:
            # Tier 3: Auto-create wf_<workflow_name>
            auto_slug = f"wf_{workflow_name}"
            from hermes_cli.kanban_db import _normalize_board_slug
            self.kanban_board = _normalize_board_slug(auto_slug)

        # Merge inputs into context — inputs are available as
        # {inputs.<key>} in template substitution. Input keys are also
        # promoted to top-level context for backward compatibility with
        # YAML templates that use bare {key} references (e.g. {question}).
        # If context already has a key, the explicit context value wins.
        if inputs:
            if context is None:
                context = {}
            context["inputs"] = inputs
            for k, v in inputs.items():
                if k not in context:
                    context[k] = v

        # Generate a run ID for this invocation. Available as {run_id}
        # in template substitution so YAML authors can create unique
        # artifact filenames per run. Microsecond precision ensures
        # concurrent dispatches get unique IDs.
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")[:21]  # YYYYMMDD-HHMMSS-ffffff → 21 chars
        workflow.run_id = f"{workflow_name}-{ts}"

        layers = self.topological_sort(workflow)

        # Try resume from saved state
        states = None
        results = {}
        layer_idx = 0

        if resume:
            saved = self._find_latest_state(workflow_name)
            if saved:
                print(f"Resuming {workflow_name} from layer {saved['current_layer']}")
                layer_idx = saved["current_layer"]
                results = saved["results"]
                # Restore context from saved state so the supervisor
                # subprocess has the original context for template
                # substitution when creating downstream cards.
                if context is None and "context" in saved:
                    context = saved["context"]
                # Restore _session_info from saved session_info so
                # completion notifications work in the supervisor subprocess.
                if "session_info" in saved:
                    if context is None:
                        context = {}
                    context.setdefault("_session_info", saved["session_info"])
                # Restore attachments from saved state so the supervisor
                # subprocess can attach files to first-layer cards.
                if not attachments and "attachments" in saved:
                    attachments = saved["attachments"]
                    self._current_attachments = attachments
                states = {
                    nid: NodeState(
                        node_id=s["node_id"],
                        status=s["status"],
                        kanban_card_id=s.get("kanban_card_id"),
                        started_at=s.get("started_at"),
                        completed_at=s.get("completed_at"),
                        attempts=s.get("attempts", 0),
                        error=s.get("error"),

                        # Restore the captured output so resume still has
                        # the upstream nodes' bodies available for
                        # {phaseN.X} template substitution.
                        result=s.get("result"),
                    )
                    for nid, s in saved["states"].items()
                }

        # Initialize fresh state
        if states is None:
            states = {nid: NodeState(node_id=nid) for nid in workflow.nodes}
            results = {}

        # On resume, check for blocked reviewer cards that need enrichment
        if resume:
            for nid, state in list(states.items()):
                if state.status == "blocked" and state.kanban_card_id:
                    # Check if this node is a reviewer for an upstream node
                    for upstream_nid, upstream_state in states.items():
                        upstream_node = workflow.nodes.get(upstream_nid)
                        if upstream_node and nid in upstream_node.reviews:
                            # Reviewer blocked — enrich upstream with failure
                            body = self.get_card_body(state.kanban_card_id)
                            print(f"   ↩ {nid} BLOCKED (reviewer) — enriching {upstream_nid} on resume")
                            if upstream_state.kanban_card_id:
                                try:
                                    with kanban_db.connect_closing(board=self.kanban_board) as conn:
                                        kanban_db.add_comment(conn, upstream_state.kanban_card_id, "workflow-engine", f"Review Failed ({nid}):\n{body}")
                                        conn.execute(
                                            "UPDATE tasks SET status = 'ready', completed_at = NULL WHERE id = ?",
                                            (upstream_state.kanban_card_id,)
                                        )
                                        conn.commit()
                                    upstream_state.status = "ready"
                                    upstream_state.completed_at = None
                                    upstream_state.result = None
                                    print(f"   ✓ {upstream_nid} enriched with failure feedback, reset to ready")
                                except Exception as e:
                                    print(f"   ⚠  Failed to enrich upstream card on resume: {e}")
                            break

        # Handle partial start
        if start_node and start_node in workflow.nodes:
            layer_idx = self._find_layer_for_node(layers, start_node)
            if layer_idx < 0:
                raise ValueError(f"Node '{start_node}' not found in any layer")
            # Mark all nodes before this layer as done
            for i in range(layer_idx):
                for nid in layers[i]:
                    states[nid].status = "done"
                    results[nid] = "done"

        print(f"Starting workflow: {workflow.name}")
        print(f"  Layers: {len(layers)} | Nodes: {len(workflow.nodes)}")
        if dry_run:
            print("  DRY RUN — no cards will be created")
        if resume:
            print("  RESUME — skipping already-completed nodes")
        if fire_and_forget:
            print("  FIRE-AND-FORGET — creating all cards, no monitoring loop")
        print()

        # Record execution in job log DB
        if not dry_run:
            self._record_execution(
                workflow_name, workflow.run_id,
                board or self.kanban_board, len(layers)
            )

        # ── Fire-and-forget: create cards, detect loop zones, spawn supervisor ──
        # Store attachments on the engine instance so create_kanban_card
        # can access them without threading through every method signature.
        self._current_attachments = attachments or []
        # Capture session info once — used for subscription routing and state persistence.
        # Session info: parameter > context["_session_info"] > ContextVars/file fallback
        _session_info = (
            session_info
            or (context or {}).get("_session_info")
            or self._get_session_info()
        )

        # Resolve max_retries: env var > workflow default > engine default
        env_retries = os.environ.get("HERMES_WORKFLOW_MAX_RETRIES")
        if env_retries and env_retries.isdigit():
            workflow.max_retries = int(env_retries)

        if fire_and_forget:
            loop_layers = self._find_loop_zones(workflow, layers)
            has_loops = len(loop_layers) > 0

            if not has_loops:
                # Simple path: no loop zones — create all cards at once
                last_layer_card_ids: list[str] = []
                for layer_idx, layer in enumerate(layers):
                    # Only attach files to the first layer — downstream
                    # nodes consume upstream outputs, not the original input.
                    self._current_attachments = attachments or [] if layer_idx == 0 else []
                    for nid in layer:
                        node = workflow.nodes[nid]
                        state = states[nid]
                        if state.status in ("done", "skipped"):
                            continue
                        if node.synthetic:
                            state.status = "done"
                            state.completed_at = datetime.now(timezone.utc).isoformat()
                            results[nid] = "done"
                            print(f"   🔓 {nid} — SYNTHETIC (auto-complete)")
                            continue
                        # Skip nodes that are reviewers for other nodes — the
                        # supervisor will dispatch them on demand when the
                        # creator blocks with "pending review".
                        is_reviewer = any(
                            nid in other_node.reviews
                            for other_nid, other_node in workflow.nodes.items()
                            if other_nid != nid
                        )
                        if is_reviewer:
                            state.status = "pending"
                            print(f"   ⏳ {nid} — REVIEWER (will be dispatched by supervisor)")
                            continue
                        state.status = "running"
                        state.started_at = datetime.now(timezone.utc).isoformat()
                        try:
                            card_id = self.dispatch_node(
                                state, node, context,
                                workflow=workflow, states=states, layers=layers,
                            )
                            if card_id is None:
                                results[nid] = "done"
                                print(f"   ⊙ {nid} → in-process (scope: global)")
                                continue
                            if not card_id:
                                state.status = "failed"
                                state.error = "Card creation returned empty ID"
                                results[nid] = "failed"
                                print(f"   ✗ {nid} → failed: empty card ID")
                                continue
                            state.kanban_card_id = card_id
                            if card_id and workflow.run_id:
                                self._record_node_card(card_id, workflow.run_id, nid)
                            if layer == layers[-1]:
                                last_layer_card_ids.append(card_id)
                            print(f"   ✓ {nid} → card {card_id}")
                        except Exception as e:
                            state.status = "failed"
                            state.error = str(e)
                            results[nid] = "failed"
                            print(f"   ✗ {nid} → failed: {e}")

                self._save_state(workflow_name, states, results, len(layers) - 1, layers,
                                run_id=workflow.run_id, context=context,
                                session_info=_session_info)
                self._update_execution(workflow.run_id, status="completed",
                                      current_layer=len(layers) - 1)

                # Fire completion notification for simple (no-loop) workflows
                self._fire_completion_notification(workflow_name, workflow, states, layers, len(layers) - 1, context, session_info=_session_info)

                return results

            # ── Has loop zones: save state, spawn supervisor, return ──
            # The supervisor subprocess runs the existing layer-by-layer
            # monitoring loop, creating cards as it goes and handling LOOPs.
            # The calling agent gets an immediate response.
            self._save_state(workflow_name, states, results, 0, layers,
                            run_id=workflow.run_id, context=context,
                            attachments=attachments,
                            session_info=_session_info)
            self._spawn_supervisor(workflow_name, workflow.run_id)
            return results

        # ── Main execution loop (layer-based with loop support) ──
        while layer_idx < len(layers):
            layer = layers[layer_idx]
            print(f"── Layer {layer_idx + 1}/{len(layers)} ──")
            print(f"   Nodes: {', '.join(layer)}")

            if dry_run:
                for nid in layer:
                    node = workflow.nodes[nid]
                    if states[nid].status in ("done", "skipped"):
                        print(f"   [SKIP] {nid} — already {states[nid].status}")
                        continue
                    deps_failed = []
                    for d in node.depends_on:
                        ds = states[d].status
                        if ds in ("failed", "timed_out", "blocked", "skipped"):
                            cause = states[d].error or ds
                            deps_failed.append(f"{d}={cause}")
                    if deps_failed:
                        print(f"   [SKIP] {nid} — {'; '.join(deps_failed)}")
                        continue
                    # dry-run when: check
                    if node.when:
                        if not self.evaluate_when(
                            node.when, node, states, context,
                            layers=layers, workflow=workflow,
                        ):
                            print(f"   [SKIP] {nid} — when: {node.when}")
                            continue
                    if node.synthetic:
                        # Synthetic gates auto-complete — there is no
                        # card to create, so dry-run should still
                        # reflect that rather than printing a fake one.
                        print(f"   [DRY RUN] {nid} — synthetic gate (auto-complete)")
                    else:
                        print(f"   [DRY RUN] Would create card for {node.agent}: {node.task[:60]}")
                layer_idx += 1
                continue

            # Create cards for this layer
            for nid in layer:
                state = states[nid]
                node = workflow.nodes[nid]

                # Skip already-completed nodes (resume)
                if state.status in ("done", "skipped"):
                    print(f"   ⏭ {nid} — {state.status}")
                    continue

                # Skip nodes with failed dependencies
                # NOTE: "blocked" is excluded — it's transient (e.g.,
                # heartbeat sweep, human gate). Downstream nodes wait
                # instead of skipping.
                deps_failed = []
                deps_blocked = []
                for d in node.depends_on:
                    ds = states[d].status
                    if ds in ("failed", "timed_out", "skipped"):
                        # Capture WHY the dependency was skipped/degraded —
                        # the error carries the upstream cause chain so
                        # downstream nodes see the root failure, not just
                        # the immediate skip.
                        cause = states[d].error or ds
                        deps_failed.append(f"{d}={cause}")
                    elif ds == "blocked":
                        deps_blocked.append(d)
                if deps_failed:
                    state.status = "skipped"
                    causes = "; ".join(deps_failed)
                    state.error = f"Skipped: dependency {' '.join(d.split('=')[0] for d in deps_failed)} failed — {causes}"
                    results[nid] = "skipped"
                    print(f"   ⏭ {nid} — SKIPPED ({causes})")
                    continue
                if deps_blocked:
                    state.status = "blocked"
                    state.error = f"Waiting: dependency {', '.join(deps_blocked)} is blocked"
                    results[nid] = "blocked"
                    print(f"   🚧 {nid} — WAITING ({', '.join(deps_blocked)} blocked)")
                    continue

                # ── when: conditional dispatch ──
                # Evaluate the node's when: expression against the
                # current workflow state.  Empty when: = always run
                # (preserves existing behavior for all workflows).
                if node.when:
                    if not self.evaluate_when(
                        node.when, node, states, context,
                        layers=layers, workflow=workflow,
                    ):
                        state.status = "skipped"
                        state.error = f"Skipped: when condition not met ({node.when})"
                        results[nid] = "skipped"
                        print(f"   ⏭ {nid} — SKIPPED (when: {node.when})")
                        continue

                # Synthetic gate nodes: auto-complete here. By the time
                # we reach this layer, all depends_on are done (the
                # topological sort guarantees this), so the gate is
                # satisfied. No kanban card is created — the gate is
                # purely an ordering primitive.
                if node.synthetic:
                    state.status = "done"
                    state.completed_at = datetime.now(timezone.utc).isoformat()
                    results[nid] = "done"
                    print(f"   🔓 {nid} — SYNTHETIC (auto-complete)")
                    continue

                # Create the card — skip if already has one
                if state.kanban_card_id:
                    # Card already exists. If it's already "ready" or
                    # "running", the dispatcher is handling it — skip.
                    # Only reset if it's in a terminal/blocked state.
                    if state.status in ("ready", "running"):
                        state.status = "running"
                        state.started_at = datetime.now(timezone.utc).isoformat()
                        print(f"   ⏳ {nid} — card {state.kanban_card_id} already in-flight ({state.status})")
                        continue
                    # Reuse: update kanban status to ready so the
                    # dispatcher picks it up again.
                    state.status = "running"
                    state.started_at = datetime.now(timezone.utc).isoformat()
                    print(f"   🔄 {nid} → reusing card {state.kanban_card_id}")
                    try:
                        with kanban_db.connect_closing(board=self.kanban_board) as _conn:
                            _conn.execute(
                                "UPDATE tasks SET status = 'ready', completed_at = NULL WHERE id = ?",
                                (state.kanban_card_id,)
                            )
                            _conn.commit()
                            kanban_db.heartbeat_worker(_conn, state.kanban_card_id)
                    except Exception:
                        pass
                    continue

                state.status = "running"
                state.started_at = datetime.now(timezone.utc).isoformat()
                state.attempts += 1

                try:
                    card_id = self.dispatch_node(
                        state, node, context,
                        workflow=workflow, states=states, layers=layers,
                    )
                    if card_id is None:
                        # scope: global — in-process, no card to monitor
                        results[nid] = "done"
                        print(f"   ⊙ {nid} → in-process (scope: global)")
                        continue
                    state.kanban_card_id = card_id
                    if card_id and workflow.run_id:
                        self._record_node_card(card_id, workflow.run_id, nid)
                    # Initialize heartbeat so the sweep doesn't auto-block
                    # the card before the worker picks it up.
                    try:
                        with kanban_db.connect_closing(board=self.kanban_board) as _conn:
                            kanban_db.heartbeat_worker(_conn, card_id)
                    except Exception:
                        pass  # Non-fatal: heartbeat sweep has created_at fallback
                    print(f"   ✓ {nid} → card {card_id}")
                except Exception as e:
                    state.status = "failed"
                    state.error = str(e)
                    results[nid] = "failed"
                    print(f"   ✗ {nid} → failed: {e}")

            # Save state after dispatching layer
            self._save_state(workflow_name, states, results, layer_idx, layers,
                            run_id=workflow.run_id, context=context)

            # Monitor completion for this layer. Synthetic nodes were
            # auto-completed in the dispatch loop above (state.status
            # == "done"), so they have no work to do here. Filtering
            # them out avoids an unnecessary 15s sleep when a layer
            # contains only synthetic gates.
            running_nodes = [
                nid for nid in layer
                if states[nid].status == "running"
            ]
            if running_nodes:
                revision_result = self._monitor_layer(
                    workflow, running_nodes, states, results, context,
                    layers=layers,
                )

                # ── Post-monitor review check ──
                # Only runs when _monitor_layer was actually called
                # (i.e., the quality block handler could have fired).
                # After _monitor_layer returns, check if any node in the
                # workflow has an active review. If so, enter the review
                # waiting loop directly.
                if self._has_active_review(workflow, states, require_ready_implement=True):
                    implement_nid = None
                    for nid, state in states.items():
                        node = workflow.nodes.get(nid)
                        if node and node.reviews and state.status == "ready":
                            implement_nid = nid
                            break
                    if implement_nid:
                        if self._review_waiting_loop(workflow, states, implement_nid):
                            layer_idx += 1
                        else:
                            layer_idx += 1
                    continue
            else:
                revision_result = None

            # ── Review loop detection ──
            # After _monitor_layer returns, check if this layer has
            # active reviews (reviewer is running or blocked). If so,
            # enter the waiting loop instead of advancing to the next
            # layer. The waiting loop polls the implement card for its
            # next "pending review" block.
            if self._has_active_review(workflow, states, layer=layer):
                # Check if reviewer is actually blocked (waiting for
                # implement to re-block) vs just unblocked (dispatcher
                # will claim it). Only enter the waiting loop when the
                # reviewer is blocked.
                reviewer_blocked = False
                for nid in layer:
                    node = workflow.nodes.get(nid)
                    if node and node.reviews:
                        for rev_entry in node.reviews:
                            rev_id = rev_entry if isinstance(rev_entry, str) else rev_entry.get("review", "")
                            if rev_id and rev_id in states:
                                if states[rev_id].status == "blocked":
                                    reviewer_blocked = True
                                    break
                        if reviewer_blocked:
                            break
                if reviewer_blocked:
                    print(f"   🔍 has_active_review=True — layer={layer}")
                    print(f"   🔄 Review loop active — waiting for {layer[0]} to re-block pending review")
                    if self._review_waiting_loop(workflow, states, layer[0]):
                        layer_idx += 1
                    else:
                        layer_idx += 1
                    # Continue to top of outer loop so the next layer
                    # gets processed naturally — don't fall through to
                    # the blocked-nodes re-check which could re-dispatch
                    # an already-completed card.
                    continue

            # Re-check blocked nodes — if their dependencies unblocked,
            # dispatch them now instead of waiting for the next layer.
            blocked_nodes = [
                nid for nid in layer
                if states[nid].status == "blocked"
            ]
            for nid in blocked_nodes:
                node = workflow.nodes[nid]
                deps_still_blocked = any(
                    states[d].status == "blocked"
                    for d in node.depends_on
                    if d in states
                )
                if not deps_still_blocked:
                    # Dependencies cleared — dispatch this node
                    state = states[nid]
                    state.status = "running"
                    state.started_at = datetime.now(timezone.utc).isoformat()
                    state.attempts += 1
                    try:
                        card_id = self.dispatch_node(
                            state, node, context,
                            workflow=workflow, states=states, layers=layers,
                        )
                        if card_id is None:
                            results[nid] = "done"
                            print(f"   ⊙ {nid} → in-process (scope: global)")
                            continue
                        state.kanban_card_id = card_id
                        if card_id and workflow.run_id:
                            self._record_node_card(card_id, workflow.run_id, nid)
                        print(f"   ✓ {nid} → card {card_id} (unblocked)")
                        running_nodes.append(nid)
                    except Exception as e:
                        state.status = "failed"
                        state.error = str(e)
                        results[nid] = "failed"
                        print(f"   ✗ {nid} → failed: {e}")

            # Re-monitor if we dispatched new nodes
            if len(running_nodes) > 0 and not revision_result:
                revision_result = self._monitor_layer(
                    workflow, running_nodes, states, results, context,
                    layers=layers,
                )

            # Check for revision loops
            if revision_result:
                # A node in this layer triggered a LOOP
                verify_nid = revision_result["verify_node"]
                revision_nid = revision_result["revision_node"]
                verify_state = states[verify_nid]

                if verify_state.loop_count >= self.MAX_REVISION_LOOPS:
                    # Escalate
                    verify_state.status = "blocked"
                    verify_state.error = (
                        f"Exceeded {self.MAX_REVISION_LOOPS} revision loops — "
                        f"escalating to Sherlock"
                    )
                    results[verify_nid] = "blocked"
                    print(f"   🚫 {verify_nid} exceeded {self.MAX_REVISION_LOOPS} "
                          f"revision loops — escalating to Sherlock")
                    # Try LLM analysis of the deadlock
                    self._try_escalation_analysis(
                        workflow, verify_nid, verify_state, context
                    )
                    layer_idx += 1  # Advance past this layer
                else:
                    # Ask the analyst whether this LOOP is genuine
                    verify_node = workflow.nodes[verify_nid]
                    revision_node = workflow.nodes[revision_nid]
                    rejection = verify_state.error or ""
                    analyst_decision = self._try_loop_decision(
                        verify_node, revision_node, rejection
                    )

                    if analyst_decision == "proceed":
                        print(f"   🧠 Analyst: rejection does not match criteria — proceeding")
                        verify_state.status = "done"
                        verify_state.completed_at = datetime.now(timezone.utc).isoformat()
                        results[verify_nid] = "done"
                        layer_idx += 1
                        continue

                    # Run the revision node
                    print(f"\n   ↩  LOOP #{verify_state.loop_count}: "
                          f"{verify_nid} → {revision_nid} → {verify_nid}")
                    rev_state = states[revision_nid]
                    rev_node = workflow.nodes[revision_nid]

                    # Run revision node
                    rev_state.status = "running"
                    rev_state.started_at = datetime.now(timezone.utc).isoformat()
                    rev_state.attempts += 1
                    try:
                        card_id = self.dispatch_node(
                            rev_state, rev_node, context,
                            workflow=workflow, states=states, layers=layers,
                        )
                        if card_id is None:
                            results[revision_nid] = "done"
                            print(f"   ⊙ {revision_nid} → in-process (scope: global)")
                            continue
                        rev_state.kanban_card_id = card_id
                        if card_id and workflow.run_id:
                            self._record_node_card(card_id, workflow.run_id, revision_nid)
                        print(f"   ✓ {revision_nid} → card {card_id}")
                    except Exception as e:
                        rev_state.status = "failed"
                        rev_state.error = str(e)
                        results[revision_nid] = "failed"
                        print(f"   ✗ {revision_nid} → failed: {e}")
                        layer_idx += 1
                        continue

                    # Monitor revision node
                    rev_layer = [revision_nid]
                    rev_states = {revision_nid: rev_state}
                    rev_results = {}
                    self._monitor_layer(
                        workflow, rev_layer, rev_states, rev_results, context,
                        layers=layers,
                        skip_loop_detection=True  # Don't recurse
                    )

                    if rev_results.get(revision_nid) == "done":
                        print(f"   ✓ {revision_nid} complete — re-triggering {verify_nid}")
                        # Reset verify node for re-run. B2: clear the
                        # stale result so a re-run that fails won't
                        # leave the previous output in the lookup
                        # and mislead downstream nodes. The re-run
                        # will populate state.result fresh on its
                        # next "done" transition.
                        verify_state.status = "pending"
                        verify_state.kanban_card_id = None
                        verify_state.started_at = None
                        verify_state.completed_at = None
                        verify_state.result = None
                        # Do NOT advance layer_idx — re-run same layer
                    else:
                        print(f"   ✗ {revision_nid} failed — cannot continue loop")
                        layer_idx += 1
            else:
                # No loops — advance to next layer
                # BUT: if any node in this layer has reviews and the
                # reviewer is still in-flight (running or blocked),
                # stay in this layer — the review loop is still active.
                if self._has_active_review(workflow, states, layer=layer):
                    print(f"   🔍 has_active_review=True — layer={layer}")
                    print(f"   🔄 Review loop active — waiting for {layer[0]} to re-block pending review")
                    if self._review_waiting_loop(workflow, states, layer[0]):
                        layer_idx += 1
                    else:
                        layer_idx += 1
                else:
                    layer_idx += 1

            print()

        # ── Summary ──
        completed = sum(1 for s in states.values() if s.status == "done")
        failed = sum(1 for s in states.values()
                    if s.status in ("failed", "timed_out"))
        skipped = sum(1 for s in states.values() if s.status == "skipped")
        blocked = sum(1 for s in states.values() if s.status == "blocked")
        print(f"Workflow complete: {completed} done, {failed} failed, "
              f"{skipped} skipped, {blocked} blocked")

        # Update job log
        final_status = "failed" if failed > 0 else "completed"
        self._update_execution(workflow.run_id, status=final_status,
                              current_layer=layer_idx)

        # Fire completion notification BEFORE clearing state
        if final_status == "completed":
            self._fire_completion_notification(workflow_name, workflow, states, layers, layer_idx, context, session_info=(context or {}).get("_session_info"))

        self._clear_state(workflow_name, run_id=workflow.run_id)

        return results

    def _monitor_layer(self, workflow: Workflow, layer: list[str],
                       states: dict[str, NodeState], results: dict,
                       context: dict = None,
                       layers: list[list[str]] = None,
                       skip_loop_detection: bool = False
                       ) -> Optional[dict]:
        """
        Poll kanban until all nodes in a layer complete or time out.

        Returns dict with 'verify_node' and 'revision_node' if a LOOP is
        detected, or None if the layer completed normally.
        """
        pending = set(layer)

        # Calculate dynamic max_polls from the longest node timeout in this layer
        max_node_timeout = max(
            (workflow.nodes[nid].timeout_minutes for nid in layer
             if states[nid].status == "running"),
            default=30
        )
        max_polls = int((max_node_timeout * 60) / self.POLL_INTERVAL)
        polls = 0

        while pending and polls < max_polls:
            time.sleep(self.POLL_INTERVAL)
            polls += 1


            # Sweep stale heartbeats once per poll tick.  Idempotent
            # and cheap (single SELECT + conditional UPDATEs).
            try:
                with kanban_db.connect_closing(board=self.kanban_board) as _conn:
                    kanban_db.sweep_stale_heartbeats(_conn)
            except Exception:
                pass  # Non-fatal: heartbeat sweep is a safety net

            for nid in list(pending):
                state = states[nid]
                # Discard only terminal states — "ready" and "blocked"
                # are still in-flight and need monitoring.
                if state.status in ("done", "skipped", "failed", "timed_out"):
                    pending.discard(nid)
                    continue

                node = workflow.nodes[nid]

                # ── Ready state (waiting for dispatcher) ──
                if state.status == "ready":
                    # Card is waiting for the kanban dispatcher to claim it.
                    # Check actual card status — it may have transitioned to
                    # "running" or "blocked" since our last poll.
                    if state.kanban_card_id:
                        try:
                            card = self.get_card_status(state.kanban_card_id)
                            card_status = card.get("status", "").lower()
                            if card_status == "running":
                                state.status = "running"
                                state.started_at = datetime.now(timezone.utc).isoformat()
                                continue
                            elif card_status == "blocked":
                                state.status = "blocked"
                                continue
                        except Exception:
                            pass
                    # Nothing to do — the dispatcher will pick it up.
                    # We keep it in pending so we detect when it transitions.
                    continue

                # ── Blocked state ──
                if state.status == "blocked":
                    # Check if this is a reviewer waiting for implement
                    # to re-block "pending review", or a genuine blocker.
                    # We keep it in pending so we detect when it transitions.
                    continue

                # ── Running state — check timeout and card status ──
                elapsed = (datetime.now(timezone.utc) -
                          datetime.fromisoformat(state.started_at)).total_seconds()

                # Timeout check — uses node's own timeout
                if elapsed > node.timeout_minutes * 60:
                    # Check whether this node has a degraded fallback
                    if node.fallback_on_timeout == "degraded":
                        state.status = "degraded"
                        state.error = (f"Node timed out but has "
                                       f"fallback_on_timeout=degraded — "
                                       f"proceeding with partial data")
                        results[nid] = "degraded"
                        print(f"   ⚡ {nid} degraded (timeout {elapsed:.0f}s) "
                              f"— proceeding with partial data")
                        pending.discard(nid)
                        continue
                    elif node.fallback_on_timeout == "retry" and state.attempts < 3:
                        state.status = "running"
                        state.attempts += 1
                        state.started_at = datetime.now(timezone.utc).isoformat()
                        print(f"   🔄 {nid} timeout — retrying "
                              f"(attempt {state.attempts}/3)")
                        # Re-create the card
                        try:
                            card_id = self.dispatch_node(
                                state, node, context,
                                workflow=workflow, states=states, layers=[layer],
                            )
                            if card_id is None:
                                results[nid] = "done"
                                print(f"   ⊙ {nid} → in-process (scope: global)")
                                continue
                            state.kanban_card_id = card_id
                        except Exception as e:
                            state.status = "failed"
                            state.error = f"Retry card creation failed: {e}"
                            results[nid] = "failed"
                            print(f"   ✗ {nid} retry failed: {e}")
                            pending.discard(nid)
                        continue
                    # Default: skip
                    state.status = "timed_out"
                    state.error = f"Exceeded {node.timeout_minutes}min timeout"
                    results[nid] = "timed_out"
                    print(f"   ⏰ {nid} timed out after {elapsed:.0f}s")
                    self._try_failure_analysis(node, state, elapsed)
                    pending.discard(nid)
                    continue

                # Check kanban card status
                if not state.kanban_card_id:
                    continue

                try:
                    card = self.get_card_status(state.kanban_card_id)
                    card_status = card.get("status", card.get("column", "unknown"))
                    card_status_lower = card_status.lower()

                    # ── Done states ──
                    if card_status_lower in ("done", "completed", "complete"):
                        state.status = "done"
                        state.completed_at = datetime.now(timezone.utc).isoformat()
                        results[nid] = "done"
                        # B2: capture the card body so downstream nodes
                        # can reference this output via
                        # {phaseN.node-id} or {node-id} in their
                        # task templates. We pull the body AFTER
                        # marking done so the kanban tool sees the
                        # latest state. A failure here is non-fatal —
                        # the node is still considered done, just
                        # without a captured result (so downstream
                        # templates would see "Unresolved" and leave
                        # the literal in place).
                        try:
                            state.result = self.get_card_body(
                                state.kanban_card_id
                            )
                        except Exception as e:
                            print(f"   ⚠  {nid} done but result "
                                  f"capture failed: {e}")
                        pending.discard(nid)
                        # Validate expected output artifacts
                        try:
                            validation = self._validate_outputs(node, state)
                            state.validation_warnings = validation
                            print(f"   ✓ {nid} completed ({elapsed:.0f}s)"
                                  + (f" [{len(validation)} validation warnings]"
                                     if validation else ""))
                        except Exception as e:
                            print(f"   ✓ {nid} completed ({elapsed:.0f}s)"
                                  f" [validation error: {e}]")

                        # Check if this node is a reviewer for an upstream node
                        reviewer_for = None
                        for upstream_nid, upstream_state in states.items():
                            upstream_node = workflow.nodes.get(upstream_nid)
                            if upstream_node and nid in upstream_node.reviews:
                                reviewer_for = upstream_nid
                                break

                        if reviewer_for:
                            # Reviewer passed — enrich upstream with test branch info
                            upstream_state = states[reviewer_for]
                            reviewer_body = state.result or ""
                            print(f"   ✅ {nid} PASSED — enriching {reviewer_for} with review results")
                            if upstream_state.kanban_card_id:
                                try:
                                    with kanban_db.connect_closing(board=self.kanban_board) as conn:
                                        kanban_db.add_comment(conn, upstream_state.kanban_card_id, "workflow-engine", f"Review Passed ({nid}):\n{reviewer_body}")
                                        conn.execute(
                                            "UPDATE tasks SET status = 'ready', completed_at = NULL, block_recurrences = 0 WHERE id = ?",
                                            (upstream_state.kanban_card_id,)
                                        )
                                        conn.commit()
                                    upstream_state.status = "ready"
                                    upstream_state.completed_at = None
                                    upstream_state.result = None
                                    print(f"   ↩  {reviewer_for} enriched with pass results, reset to ready")
                                except Exception as e:
                                    print(f"   ⚠  Failed to enrich upstream card: {e}")

                    # ── Blocked states ──
                    elif card_status_lower in ("blocked",):
                        body = self.get_card_body(state.kanban_card_id)

                        # Check for "pending review" block convention.
                        # The block reason is stored in the most recent
                        # ``blocked`` event payload, NOT in the task body.
                        pending_review = self._check_pending_review(state.kanban_card_id)

                        if pending_review:
                            # Node blocked itself pending review — find reviewer
                            node = workflow.nodes.get(nid)
                            if node and node.reviews:
                                for review_entry in node.reviews:
                                    if isinstance(review_entry, dict):
                                        rev_id = review_entry.get("review", "")
                                    else:
                                        rev_id = review_entry
                                    if not rev_id:
                                        continue

                                    rev_state = states.get(rev_id)
                                    reviewer_node = workflow.nodes.get(rev_id)
                                    if not reviewer_node:
                                        continue

                                    if rev_state and rev_state.kanban_card_id:
                                        # Reviewer card already exists.
                                        # Only unblock if it's actually blocked.
                                        if rev_state.status == "blocked":
                                            try:
                                                with kanban_db.connect_closing(board=self.kanban_board) as _conn:
                                                    kanban_db.unblock_task(_conn, rev_state.kanban_card_id)
                                                rev_state.status = "ready"
                                                rev_state.completed_at = None
                                                rev_state.result = None
                                                print(f"   🔄 {nid} pending review → unblocked {rev_id} (card {rev_state.kanban_card_id})")
                                            except Exception as e:
                                                print(f"   ⚠  Failed to unblock reviewer card: {e}")
                                        else:
                                            # Already running or ready — keep polling
                                            print(f"   ⏳ {nid} — {rev_id} already in-flight (status: {rev_state.status})")
                                    else:
                                        # First time — create the reviewer card
                                        try:
                                            reviewer_card_id = self.dispatch_node(
                                                states[rev_id], reviewer_node, context,
                                                workflow=workflow, states=states, layers=layers,
                                            )
                                            if reviewer_card_id:
                                                states[rev_id].kanban_card_id = reviewer_card_id
                                                states[rev_id].status = "running"
                                                states[rev_id].started_at = datetime.now(timezone.utc).isoformat()
                                                state.review_counts[rev_id] = state.review_counts.get(rev_id, 0) + 1
                                                pending.add(rev_id)
                                                # Save state so the has_active_review check
                                                # can find the reviewer card ID.
                                                self._save_state(
                                                    workflow.name, states, results,
                                                    0, layers,
                                                    run_id=workflow.run_id, context=context,
                                                )
                                                print(f"   📋 {nid} pending review → created {rev_id} (card {reviewer_card_id})")
                                        except Exception as e:
                                            print(f"   ⚠  Failed to create reviewer card: {e}")
                                    break  # One reviewer at a time
                            else:
                                # No reviews attribute — but check if this node is
                                # a reviewer for an upstream node.
                                reviewer_for = None
                                for upstream_nid, upstream_state in states.items():
                                    upstream_node = workflow.nodes.get(upstream_nid)
                                    if upstream_node and nid in upstream_node.reviews:
                                        reviewer_for = upstream_nid
                                        break
                                if reviewer_for:
                                    # Reviewer blocked with "pending review" — treat
                                    # as a quality review block.
                                    upstream_state = states[reviewer_for]
                                    is_quality_block = self._classify_block_reason(
                                        nid, body, workflow, context
                                    )
                                    if is_quality_block:
                                        print(f"   📋 {nid} BLOCKED (quality review) — enriching {reviewer_for}")
                                        if upstream_state.kanban_card_id:
                                            try:
                                                with kanban_db.connect_closing(board=self.kanban_board) as conn:
                                                    kanban_db.add_comment(conn, upstream_state.kanban_card_id, "workflow-engine", f"Review Feedback ({nid}):\n{body}")
                                                    conn.execute(
                                                        "UPDATE tasks SET status = 'ready', completed_at = NULL, block_recurrences = 0 WHERE id = ?",
                                                        (upstream_state.kanban_card_id,)
                                                    )
                                                    conn.commit()
                                                upstream_state.status = "ready"
                                                upstream_state.completed_at = None
                                                upstream_state.result = None
                                                print(f"   ↩  {reviewer_for} enriched with review feedback, reset to ready")
                                            except Exception as e:
                                                print(f"   ⚠  Failed to enrich upstream card: {e}")
                                        state.status = "blocked"
                                        pending.discard(nid)
                                        if upstream_state.kanban_card_id:
                                            pending.discard(reviewer_for)
                                        print(f"   ⏸  {nid} stays blocked — waiting for {reviewer_for} to re-block pending review")
                                        # Go back to layer 0 so the review
                                        # waiting loop can re-engage.
                                        layer_idx = 0
                                    else:
                                        state.status = "blocked"
                                        state.error = f"Reviewer blocked (technical): {body[:100]}"
                                        results[nid] = "blocked"
                                        print(f"   🚫 {nid} BLOCKED (technical) — notifying calling agent")
                                        pending.discard(nid)
                                        self._try_block_notify(workflow, nid, state, body, context)
                                else:
                                    # Genuine blocker — not pending review, not a reviewer
                                    state.status = "blocked"
                                    state.error = f"Blocked: {body[:100]}"
                                    results[nid] = "blocked"
                                    print(f"   🚫 {nid} BLOCKED (no reviews) — notifying calling agent")
                                    pending.discard(nid)
                                    self._try_block_notify(workflow, nid, state, body, context)
                        else:
                            # Not "pending review" — check if this node is a reviewer
                            # for an upstream node (i.e., it blocked with review results)
                            reviewer_for = None
                            for upstream_nid, upstream_state in states.items():
                                upstream_node = workflow.nodes.get(upstream_nid)
                                if upstream_node and nid in upstream_node.reviews:
                                    reviewer_for = upstream_nid
                                    break

                            if reviewer_for:
                                # Reviewer blocked — use auxiliary to classify the reason
                                upstream_state = states[reviewer_for]
                                is_quality_block = self._classify_block_reason(
                                    nid, body, workflow, context
                                )

                                if is_quality_block:
                                    # Quality review results — enrich upstream, reset to ready.
                                    # The reviewer stays blocked until implement re-blocks
                                    # "pending review", at which point the pending-review
                                    # handler unblocks it.
                                    print(f"   📋 {nid} BLOCKED (quality review) — enriching {reviewer_for}")
                                    if upstream_state.kanban_card_id:
                                        try:
                                            with kanban_db.connect_closing(board=self.kanban_board) as conn:
                                                kanban_db.add_comment(conn, upstream_state.kanban_card_id, "workflow-engine", f"Review Feedback ({nid}):\n{body}")
                                                conn.execute(
                                                    "UPDATE tasks SET status = 'ready', completed_at = NULL, block_recurrences = 0 WHERE id = ?",
                                                    (upstream_state.kanban_card_id,)
                                                )
                                                conn.commit()
                                            upstream_state.status = "ready"
                                            upstream_state.completed_at = None
                                            upstream_state.result = None
                                            print(f"   ↩  {reviewer_for} enriched with review feedback, reset to ready")
                                        except Exception as e:
                                            print(f"   ⚠  Failed to enrich upstream card: {e}")

                                    # Update in-memory state to reflect actual kanban status
                                    state.status = "blocked"
                                    # Remove both from pending so _monitor_layer exits
                                    # and the caller enters the review waiting loop.
                                    pending.discard(nid)
                                    if upstream_state.kanban_card_id:
                                        pending.discard(reviewer_for)
                                    print(f"   ⏸  {nid} stays blocked — waiting for {reviewer_for} to re-block pending review")
                                else:
                                    # Technical block — notify calling agent
                                    state.status = "blocked"
                                    state.error = f"Reviewer blocked (technical): {body[:100]}"
                                    results[nid] = "blocked"
                                    print(f"   🚫 {nid} BLOCKED (technical) — notifying calling agent")
                                    pending.discard(nid)
                                    self._try_block_notify(workflow, nid, state, body, context)
                            else:
                                # Genuine blocker — not pending review, not a reviewer
                                state.status = "blocked"
                                state.error = f"Blocked: {body[:100]}"
                                results[nid] = "blocked"
                                print(f"   🚫 {nid} BLOCKED — notifying calling agent")
                                pending.discard(nid)
                                self._try_block_notify(workflow, nid, state, body, context)

                except Exception as e:
                    # Card query failed — log and keep polling
                    logger.debug("monitor: card status check failed for %s: %s", nid, e)

        # Anything still pending after max_polls
        for nid in list(pending):
            state = states[nid]
            node = workflow.nodes[nid]
            state.status = "timed_out"
            state.error = f"Still running after {max_polls * self.POLL_INTERVAL}s (node timeout: {node.timeout_minutes}min)"
            results[nid] = "timed_out"
            pending.discard(nid)
            print(f"   ⏰ {nid} timed out (layer poll exhausted)")
            self._try_failure_analysis(node, state, max_polls * self.POLL_INTERVAL)

        return None  # No loop detected

    # ── Status query ────────────────────────────────────────────

    def status(self, workflow_name: str = None) -> dict:
        """Query current state of running or saved workflows."""
        if workflow_name:
            saved = self._find_latest_state(workflow_name)
            if saved:
                result = {
                    "workflow": workflow_name,
                    "current_layer": saved["current_layer"],
                    "total_layers": len(saved["layers"]),
                    "states": saved["states"],
                    "results": saved["results"],
                    "updated_at": saved.get("updated_at"),
                }
                # Try LLM summary
                summary = self._try_status_summary(workflow_name, saved)
                if summary:
                    result["summary"] = summary
                return result
            return {"workflow": workflow_name, "status": "no saved state"}

        # List all saved states
        runs = []
        for state_file in sorted(self.STATE_DIR.glob("*_state.json")):
            runs.append(state_file.stem.replace("_state", ""))
        return {"active_runs": runs}


# ── CLI ─────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Workflow Execution Engine")
    sub = parser.add_subparsers(dest="command")

    # start
    start = sub.add_parser("start", help="Start a workflow")
    start.add_argument("workflow", help="Workflow name (YAML file in docs/fleet-pipelines/)")
    start.add_argument("--context", "-c", action="append", help="Key=value context pairs (repeatable)")
    start.add_argument("--board", "-b", help="Board slug to use (overrides YAML and auto-create)")
    start.add_argument("--inputs", "-i", action="append", help="Input key=value pairs (repeatable, available as {inputs.<key>})")
    start.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    start.add_argument("--node", help="Start from a specific node (partial execution)")
    start.add_argument("--resume", action="store_true", help="Resume from saved state")

    # validate
    validate = sub.add_parser("validate", help="Validate a workflow without executing")
    validate.add_argument("workflow", help="Workflow name to validate")

    # status
    status = sub.add_parser("status", help="Query workflow state")
    status.add_argument("workflow", nargs="?", help="Workflow name (omit for all)")

    # list
    sub.add_parser("list", help="List available workflow definitions")

    # show
    show = sub.add_parser("show", help="Show pipeline structure (layers + nodes)")
    show.add_argument("workflow", help="Workflow name to display")

    args = parser.parse_args()
    engine = WorkflowEngine()

    if args.command == "start":
        context = {}
        if args.context:
            for pair in args.context:
                k, v = pair.split("=", 1)
                context[k] = v
        inputs = {}
        if args.inputs:
            for pair in args.inputs:
                k, v = pair.split("=", 1)
                inputs[k] = v
        engine.execute(args.workflow, context=context, start_node=args.node,
                      dry_run=args.dry_run, resume=args.resume,
                      board=args.board, inputs=inputs or None)

    elif args.command == "validate":
        result = engine.validate(args.workflow)
        if result["valid"]:
            print(f"✓ {args.workflow} — {result['nodes']} nodes, "
                  f"{result['layers']} layers, valid DAG")
        else:
            print(f"✗ {args.workflow} — INVALID")
        if result["issues"]:
            for issue in result["issues"]:
                print(f"  • {issue}")
        sys.exit(0 if result["valid"] else 1)

    elif args.command == "status":
        state = engine.status(args.workflow)
        print(json.dumps(state, indent=2))

    elif args.command == "list":
        for f in sorted(engine.workflows_dir.glob("*.yaml")):
            print(f"  {f.stem}")

    elif args.command == "show":
        workflow = engine.load_workflow(args.workflow)
        layers = engine.topological_sort(workflow)
        print(f"Pipeline: {workflow.name}")
        print(f"Description: {workflow.description[:80]}...")
        print(f"Layers: {len(layers)} | Nodes: {len(workflow.nodes)}")
        print()
        for i, layer in enumerate(layers):
            print(f"Layer {i}:")
            for nid in layer:
                node = workflow.nodes[nid]
                deps = f" ← {', '.join(node.depends_on)}" if node.depends_on else ""
                # Synthetic gates have no agent — show [synthetic]
                # so operators can distinguish gate nodes from real
                # dispatch targets at a glance.
                agent_label = "synthetic" if node.synthetic else node.agent
                print(f"  [{agent_label}] {nid}{deps}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
