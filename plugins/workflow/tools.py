"""Agent-facing tools for the workflow plugin.

These wrap the in-process ``WorkflowEngine`` class so an agent (e.g. Sherlock)
can drive pipeline execution through its normal tool calls without having
to know the CLI exists. Every handler returns a JSON-serializable dict the
agent can read directly.

Tools
-----
- ``workflow_start``    — kick off a pipeline (predefined or dynamic mode);
                         creates kanban cards and monitors them layer-by-layer
- ``workflow_view``     — load a workflow template for inspection
- ``workflow_validate`` — structural check: DAG, cycles, missing nodes
- ``workflow_status``   — current state of a running (or last-run) pipeline
- ``workflow_list``     — available pipeline definitions
- ``workflow_show``     — pipeline structure: layers + nodes + dependencies

All tools are read-only except ``workflow_start``, which creates kanban
cards via the same code path the CLI uses. The engine handles revision
loops (``LOOP:<target>`` blocked-card convention) internally; the agent
sees the final summary in the returned dict.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime gate
# ---------------------------------------------------------------------------

def check_workflow_requirements() -> bool:
    """Return True when the workflow engine can be invoked."""
    try:
        from plugins.workflow.engine import WorkflowEngine  # noqa: F401
    except Exception as exc:
        logger.warning("check_workflow: engine import FAILED: %s %s", type(exc).__name__, exc)
        import traceback; traceback.print_exc()
        return False

    try:
        from pathlib import Path
        import os
        workflows_dir = None
        env_path = os.environ.get("HERMES_WORKFLOW_FILES")
        if env_path:
            workflows_dir = Path(env_path)
        else:
            hermes_home = os.environ.get("HERMES_HOME", "")
            if hermes_home:
                candidate = Path(hermes_home) / "workflows"
                if candidate.is_dir():
                    workflows_dir = candidate
            if not workflows_dir:
                engine_mod = __import__("plugins.workflow.engine", fromlist=["WorkflowEngine"])
                workflows_dir = Path(engine_mod.__file__).resolve().parent.parent / "docs" / "fleet-pipelines"
        if not workflows_dir.is_dir():
            logger.warning("check_workflow: workflows_dir missing: %s", workflows_dir)
            return False
    except Exception as exc:
        logger.warning("check_workflow: dir check FAILED: %s %s", type(exc).__name__, exc)
        import traceback; traceback.print_exc()
        return False
    return True


# ---------------------------------------------------------------------------
# Session info bridge — tool handler captures ContextVars, engine reads them
# ---------------------------------------------------------------------------
# ContextVars from gateway.session_context are available in the tool handler
# scope but lost by the time engine.execute() runs.  The tool handler writes
# session info here; the engine reads it.



def _capture_session_for_engine() -> dict:
    """Capture current gateway session info for the engine.

    Called from the tool handler where ContextVars are live.
    Returns session info dict directly — no temp file needed.
    """
    try:
        from gateway.session_context import get_session_env
        from pathlib import Path as _P
        platform = get_session_env("HERMES_SESSION_PLATFORM", "")
        chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "")
        _cv_profile = get_session_env("HERMES_SESSION_PROFILE", "")
        _env_profile = os.environ.get("HERMES_PROFILE")
        _hermes_home = os.environ.get("HERMES_HOME", "")
        _derived = (_P(_hermes_home).name
                    if "profiles/" in _hermes_home else "default")
        _profile = _cv_profile or _env_profile or _derived
        _session_key = get_session_env("HERMES_SESSION_KEY", "")
        # Rebuild session_key with profile if it's missing from ContextVar
        if _profile and _profile != "default" and "agent:main:" in _session_key:
            _session_key = _session_key.replace("agent:main:", f"agent:{_profile}:")
        if platform and chat_id:
            return {
                "platform": platform,
                "chat_id": chat_id,
                "thread_id": get_session_env("HERMES_SESSION_THREAD_ID", "") or None,
                "user_id": get_session_env("HERMES_SESSION_USER_ID", "") or None,
                "profile": _profile,
                "session_key": _session_key,
            }
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def _engine():
    """Lazy import + instantiate the engine. Kept inside a function so the
    plugin still loads in test contexts where ``tools.workflow_engine`` may
    not be importable (the check_fn already gates on this, but defense in
    depth is cheap)."""
    from plugins.workflow.engine import WorkflowEngine
    return WorkflowEngine()


def _ok(payload: Any) -> str:
    """Wrap a successful result in the standard tool-output envelope."""
    return json.dumps({"ok": True, "result": payload}, indent=2, default=str)


def _err(message: str, **extra: Any) -> str:
    """Wrap an error result; ``message`` is a short agent-readable string."""
    return json.dumps({"ok": False, "error": message, **extra}, indent=2, default=str)

def handle_workflow_start(
    args: Dict[str, Any],
    **kwargs: Any,
) -> str:
    """Start a pipeline in the given mode.

    Mode ``"predefined"`` (default):
        Reads a YAML pipeline definition from ``docs/fleet-pipelines/``,
        validates it, and dispatches via the kanban engine.  Creates
        kanban cards for layer-0 nodes and returns immediately — the
        engine monitors layer completion on a background thread.
        Always fire-and-forget; use ``workflow_status`` to check progress.

    Mode ``"dynamic"``:
        Delegates to ``dynamic_bridge.run_dynamic_workflow()`` which
        creates an ad-hoc DAG at runtime from the objective and node
        list passed in *context*.  Also fire-and-forget — each node is
        dispatched via ``delegate_task(background=True)``.

    ``board`` overrides the kanban board for this run.  When empty,
    the engine uses the YAML ``kanban_board`` field or auto-creates
    ``wf_<workflow_name>``.

    See ``workflow_list`` for available pipelines, ``workflow_show`` to
    inspect structure, and ``workflow_status`` to check a running run.
    """
    workflow = args.get("workflow", "")
    context = args.get("context")
    mode = args.get("mode", "predefined")
    node = args.get("node")
    dry_run = args.get("dry_run", False)
    resume = args.get("resume", False)
    scope = args.get("scope", "project")
    single_flight = args.get("single_flight", False)
    inputs = args.get("inputs")
    board = args.get("board", "")
    attachments = args.get("attachments")

    if not workflow or not isinstance(workflow, str):
        return _err("workflow must be a non-empty string")

    if mode == "dynamic":
        return _handle_workflow_start_dynamic(
            workflow=workflow,
            context=context,
            scope=scope,
            single_flight=single_flight,
            dry_run=dry_run,
            board=board,
            **kwargs,
        )

    # Default: predefined mode
    return _handle_workflow_start_predefined(
        workflow=workflow,
        context=context,
        node=node,
        dry_run=dry_run,
        resume=resume,
        single_flight=single_flight,
        inputs=inputs,
        board=board,
        attachments=attachments,
    )
def _handle_workflow_start_predefined(
    workflow: str,
    context: Optional[Dict[str, Any]] = None,
    node: Optional[str] = None,
    dry_run: bool = False,
    resume: bool = False,
    single_flight: bool = False,
    inputs: Optional[Dict[str, Any]] = None,
    board: str = "",
    attachments: Optional[list] = None,
) -> str:
    """Predefined mode: look up YAML in docs/fleet-pipelines/, validate,
    dispatch via engine.

    Always fire-and-forget.  Creates kanban cards for all nodes and
    subscribes the final-layer card(s) for notification — the gateway
    notifier pushes terminal events (completed, blocked, etc.) back
    to the originating session.  No monitoring loop, no delegate_task,
    no daemon thread.

    Use ``workflow_status`` to check progress and ``workflow_list`` to
    discover available pipelines.
    """
    try:
        engine = _engine()
    except Exception as exc:
        return _err(f"engine import failed: {exc}")

    # Validate before executing (skip for dry-run and resume)
    if not dry_run and not resume:
        try:
            result = engine.validate(workflow)
            if not result["valid"]:
                issues = "\n".join(f"  - {i}" for i in result["issues"])
                return _err(f"Workflow validation failed:\n{issues}")
            # Validate provided inputs and attachments against declarations
            try:
                wf_def = engine.load_workflow(workflow)
                for inp in getattr(wf_def, "inputs", []):
                    if inp.get("required", False):
                        if not inputs or inp["name"] not in inputs:
                            return _err(
                                f"Missing required input: '{inp['name']}'",
                                hint=inp.get("description", ""),
                            )
                for att in getattr(wf_def, "attachments", []):
                    if att.get("required", False):
                        if not attachments or att["name"] not in attachments:
                            return _err(
                                f"Missing required attachment: '{att['name']}'",
                                hint=att.get("description", ""),
                            )
            except FileNotFoundError:
                pass
        except FileNotFoundError:
            pass  # Will be caught by execute()
        except Exception as exc:
            logger.debug("validation warning for %s: %s", workflow, exc)

    # Single-flight opt-in check: if the workflow declares
    # ``single_flight: true`` in YAML, refuse to start when another
    # run is already in progress. Prevents duplicate parallel runs
    # from webhook storms or repeated dispatch signals.
    # Skipped for dry-run and resume — those are explicitly about
    # inspecting / continuing an existing run, not starting fresh.
    if not dry_run and not resume and single_flight:
        try:
            wf_def = engine.load_workflow(workflow)
        except Exception:
            wf_def = None
        if wf_def is not None and getattr(wf_def, "single_flight", False):
            if engine._has_active_run(workflow):
                return _err(
                    f"single_flight: another run of '{workflow}' is in progress",
                    hint="wait for the current run to finish, or call workflow_status to inspect",
                )

    # Dry-run is always synchronous — no cards created, no monitoring needed.
    if dry_run:
        try:
            result = engine.execute(
                workflow_name=workflow,
                context=_ctx,
                start_node=node,
                dry_run=True,
                resume=resume,
                inputs=inputs,
                board=board,
                attachments=attachments,
                session_info=_sess or None,
            )
        except FileNotFoundError as exc:
            return _err(f"workflow not found: {workflow}", hint=str(exc))
        except Exception as exc:
            logger.exception("workflow_start dry-run failed for %s", workflow)
            return _err(f"dry-run failed: {exc}")
        return _ok(result)

    # Capture session info and inject into context (which persists in state file)
    _sess = _capture_session_for_engine()
    _ctx = context or None
    if _sess:
        if _ctx is None:
            _ctx = {}
        _ctx["_session_info"] = _sess

    # Fire-and-forget: create all kanban cards and subscribe the final
    # layer for notifications.  The kanban dispatcher picks up ready
    # cards and spawns workers; the gateway notifier pushes terminal
    # events back to the originating session.  No monitoring loop,
    # no delegate_task, no daemon thread.
    try:
        result = engine.execute(
            workflow_name=workflow,
            context=_ctx,
            start_node=node,
            dry_run=False,
            resume=resume,
            inputs=inputs,
            board=board,
            fire_and_forget=True,
            attachments=attachments,
            session_info=_sess or None,
        )
    except FileNotFoundError as exc:
        return _err(f"workflow not found: {workflow}", hint=str(exc))
    except Exception as exc:
        logger.exception("workflow_start failed for %s", workflow)
        return _err(f"execution failed: {exc}")

    # Inject session_info into the state file — the engine's _save_state
    # may not persist it because the supervisor subprocess overwrites the
    # file. The tool handler runs in the gateway process where ContextVars
    # are live, so we patch the state file directly after execute() returns.
    if _sess:
        try:
            from pathlib import Path as _P
            import json as _j
            state_dir = _P(__file__).resolve().parent.parent.parent / "docs" / "fleet-pipelines" / ".engine-state"
            for sf in sorted(state_dir.glob(f"{workflow}_*_state.json"),
                             key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    data = _j.loads(sf.read_text())
                    if not data.get("session_info"):
                        data["session_info"] = _sess
                        sf.write_text(_j.dumps(data, indent=2))
                except Exception:
                    continue
        except Exception:
            pass

    return _ok({
        "status": "dispatched",
        "workflow": workflow,
        "message": f"Workflow '{workflow}' started — cards created, final node will notify on completion",
    })


def _handle_workflow_start_dynamic(
    workflow: str,
    context: Optional[Dict[str, Any]] = None,
    scope: str = "project",
    single_flight: bool = False,
    dry_run: bool = False,
    board: str = "",
    **kwargs: Any,
) -> str:
    """Dynamic mode: delegate to dynamic_bridge.run_dynamic_workflow().

    Unlike predefined mode which reads pre-defined YAML pipeline
    definitions, this creates an ad-hoc DAG at runtime.  The
    ``workflow`` parameter is the workflow_id to create (or reuse), and
    ``context`` carries the objective and nodes from the calling agent.

    ``board`` overrides the kanban board for project-scope cards.
    When empty, the bridge uses its default ``"dynamic-workflows"``.

    Scope controls fleet integration:
      - ``project`` (default): creates kanban cards for worker nodes
      - ``global``: no kanban, in-memory only
      - ``durable``: persists state to disk
    """
    from plugins.workflow.dynamic_bridge import (
        run_dynamic_workflow,
    )

    # Extract objective and nodes from context or kwargs
    ctx = context or {}
    objective = ctx.get("objective", "")
    nodes = ctx.get("nodes", [])
    wf_context = ctx.get("context", "")

    # Allow overriding via kwargs (for future flexibility)
    if not objective and "objective" in kwargs:
        objective = kwargs["objective"]
    if not nodes and "nodes" in kwargs:
        nodes = kwargs["nodes"]
    if not wf_context and "wf_context" in kwargs:
        wf_context = kwargs["wf_context"]

    if not objective:
        return _err("objective is required (pass in context.objective)")
    if not isinstance(nodes, list) or not nodes:
        return _err("nodes must be a non-empty list (pass in context.nodes)")
    if scope not in ("project", "global", "durable"):
        return _err(f"invalid scope: {scope!r}; must be project, global, or durable")

    if dry_run:
        return _ok({
            "dry_run": True,
            "workflow_id": workflow,
            "objective": objective,
            "node_count": len(nodes),
            "scope": scope,
            "single_flight": single_flight,
            "board": board,
        })

    try:
        result = run_dynamic_workflow(
            workflow_id=workflow,
            objective=objective,
            nodes=nodes,
            context=wf_context,
            scope=scope,
            single_flight=single_flight,
            dispatch_ready=True,
            board=board,
        )
    except Exception as exc:
        logger.exception("dynamic_workflow_start failed for %s", workflow)
        return _err(f"dynamic workflow failed: {exc}")

    return _ok(result)


def handle_workflow_view(args: Dict[str, Any], **kwargs: Any) -> str:
    """Load a workflow template (predefined YAML or dynamic starter) for inspection."""
    from plugins.workflow.registry import _fleet_pipelines_dirs, _user_workflows_dir

    workflow = args.get("workflow", "")

    if not workflow or not isinstance(workflow, str):
        return _err("workflow must be a non-empty string")

    # Check if it's a predefined pipeline
    for fp_dir in _fleet_pipelines_dirs():
        path = fp_dir / f"{workflow}.yaml"
        if path.is_file():
            return _ok({
                "name": workflow,
                "mode": "predefined",
                "path": str(path),
                "yaml": path.read_text(),
            })

    # Check if it's a dynamic template
    uw_dir = _user_workflows_dir()
    if uw_dir:
        path = uw_dir / f"{workflow}.yaml"
        if path.is_file():
            return _ok({
                "name": workflow,
                "mode": "dynamic",
                "path": str(path),
                "yaml": path.read_text(),
            })

    return _err(f"workflow template not found: {workflow}")


def handle_workflow_validate(args: Dict[str, Any], **kwargs: Any) -> str:
    """Structural validation only. Returns nodes/layers/cycle check without
    creating kanban cards. Safe to call before committing to a start."""
    try:
        engine = _engine()
    except Exception as exc:
        return _err(f"engine import failed: {exc}")

    workflow = args.get("workflow", "")

    if not workflow or not isinstance(workflow, str):
        return _err("workflow must be a non-empty string")

    try:
        result = engine.validate(workflow)
    except FileNotFoundError as exc:
        return _err(f"workflow not found: {workflow}", hint=str(exc))
    except Exception as exc:
        logger.exception("workflow_validate failed for %s", workflow)
        return _err(f"validation failed: {exc}")

    return _ok(result)


def handle_workflow_status(args: Dict[str, Any], **kwargs: Any) -> str:
    """Current state of a running pipeline (or all pipelines if workflow is
    omitted). Mirrors ``hermes kanban status`` for the kanban cards the
    engine owns."""
    try:
        engine = _engine()
    except Exception as exc:
        return _err(f"engine import failed: {exc}")

    workflow = args.get("workflow") or None

    try:
        result = engine.status(workflow)
    except Exception as exc:
        logger.exception("workflow_status failed for %s", workflow)
        return _err(f"status query failed: {exc}")

    return _ok(result)


def handle_workflow_list(
    args: Dict[str, Any],
    **kwargs: Any,
) -> str:
    """List available workflow definitions from both the fleet pipelines
    directory (pre-defined) and ``~/.hermes/workflows/`` (dynamic).

    When *trigger* is provided the list is filtered to workflows whose
    trigger keywords appear in the given string (case-insensitive).
    Without *trigger* every registered workflow is returned.
    """
    trigger = args.get("trigger") or None
    from plugins.workflow.registry import list_workflows, match_workflow_trigger

    try:
        workflows = list_workflows()
    except Exception as exc:
        return _err(f"registry scan failed: {exc}")

    # Optional trigger-based filter
    if trigger:
        matched = match_workflow_trigger(trigger)
        if matched is not None:
            workflows = [w for w in workflows if w["name"] == matched["name"]]
        else:
            workflows = []

    # Partition for a clearer response
    predefined = [w for w in workflows if w["mode"] == "predefined"]
    dynamic = [w for w in workflows if w["mode"] == "dynamic"]

    return _ok({
        "workflows": workflows,
        "predefined_count": len(predefined),
        "dynamic_count": len(dynamic),
        "total": len(workflows),
    })


def handle_workflow_show(args: Dict[str, Any], **kwargs: Any) -> str:
    """Show pipeline structure: layers, nodes, dependencies. Use before
    ``workflow_start`` to understand the DAG."""
    try:
        engine = _engine()
    except Exception as exc:
        return _err(f"engine import failed: {exc}")

    workflow = args.get("workflow", "")

    if not workflow or not isinstance(workflow, str):
        return _err("workflow must be a non-empty string")

    try:
        wf = engine.load_workflow(workflow)
        layers = engine.topological_sort(wf)
    except FileNotFoundError as exc:
        return _err(f"workflow not found: {workflow}", hint=str(exc))
    except Exception as exc:
        logger.exception("workflow_show failed for %s", workflow)
        return _err(f"show failed: {exc}")

    nodes = []
    for nid, node in wf.nodes.items():
        nodes.append({
            "id": nid,
            "description": getattr(node, "description", ""),
            "agent": node.agent,
            "task": node.task[:200] + "..." if len(node.task) > 200 else node.task,
            "deps": sorted(node.depends_on),
            "reviews": getattr(node, "reviews", []),
            "timeout_min": node.timeout_minutes,
            "layer": next((i for i, l in enumerate(layers) if nid in l), None),
        })
    return _ok({
        "name": wf.name,
        "description": wf.description,
        "layers": len(layers),
        "nodes": len(wf.nodes),
        "inputs": getattr(wf, "inputs", []),
        "attachments": getattr(wf, "attachments", []),
        "structure": nodes,
    })


# ---------------------------------------------------------------------------
# Deprecated alias — prefer handle_workflow_start(mode="dynamic") instead
# ---------------------------------------------------------------------------

def handle_workflow_dynamic_start(
    args: Dict[str, Any],
    **kwargs: Any,
) -> str:
    """Deprecated: use ``handle_workflow_start`` with ``mode="dynamic"`` instead.

    This thin wrapper maintains backward compatibility for callers that
    still reference the old entry point.
    """
    args["mode"] = "dynamic"
    return handle_workflow_start(args, **kwargs)


# ---------------------------------------------------------------------------
# Tool schemas — fed to PluginContext.register_tool() in __init__.py
# ---------------------------------------------------------------------------

WORKFLOW_START_SCHEMA: Dict[str, Any] = {
    "name": "workflow_start",
    "description": (
        "Start a pipeline by name in the given mode. "
        "Mode 'predefined' (default): reads a YAML pipeline from "
        "docs/fleet-pipelines/, creates kanban cards for all nodes "
        "in fire-and-forget fashion — no monitoring loop. "
        "Subscribes the final-layer card(s) for notification. "
        "Mode 'dynamic': creates an ad-hoc DAG at runtime from the objective "
        "and node list in context. "
        "Optional 'attachments' array attaches files to first-layer cards. "
        "Returns immediately. Use workflow_list to see available "
        "pipelines, workflow_show to inspect structure, and workflow_status "
        "to check a running pipeline."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "workflow": {
                "type": "string",
                "description": (
                    "Pipeline name (without .yaml). For predefined mode, "
                    "current canon: 'ideation' (spec→security→validate→decompose), "
                    "'feature-dev' (build→CI→review→merge→post-merge). "
                    "For dynamic mode, this is the workflow_id to create or reuse."
                ),
            },
            "context": {
                "type": "object",
                "description": (
                    "Optional key=value context pairs (e.g. {'project': 'foo'}). "
                    "Available as substitutions in the pipeline YAML. "
                    "For dynamic mode, must contain 'objective' (string) and "
                    "'nodes' (array of node dicts with node_id, goal, depends_on)."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["predefined", "dynamic"],
                "default": "predefined",
                "description": (
                    "Workflow mode. 'predefined' uses YAML pipeline from "
                    "docs/fleet-pipelines/. 'dynamic' creates a model-authored "
                    "DAG from a template."
                ),
            },
            "node": {
                "type": "string",
                "description": (
                    "Start from a specific node id (partial execution). "
                    "Only used in predefined mode."
                ),
            },
            "dry_run": {
                "type": "boolean",
                "description": "Print the execution plan without creating kanban cards.",
                "default": False,
            },
            "resume": {
                "type": "boolean",
                "description": "Resume from saved state if a previous run was interrupted.",
                "default": False,
            },
            "scope": {
                "type": "string",
                "enum": ["project", "global", "durable"],
                "description": (
                    "Fleet integration scope (dynamic mode only). "
                    "'project' creates kanban cards, 'durable' persists state, "
                    "'global' is in-memory only."
                ),
                "default": "project",
            },
            "single_flight": {
                "type": "boolean",
                "description": (
                    "If True, refuse to create a new workflow when a run "
                    "with the same workflow_id is already in progress. "
                    "In predefined mode, also checks the YAML single_flight flag."
                ),
                "default": False,
            },
            "inputs": {
                "type": "object",
                "description": (
                    "Optional key=value input pairs (e.g. {'question': 'Should we ship X?'}). "
                    "Available as {inputs.<key>} in YAML templates. Also promoted to top-level "
                    "context for backward compatibility with bare {key} references. "
                    "Use this instead of context for workflow-specific parameters."
                ),
            },
            "board": {
                "type": "string",
                "description": (
                    "Optional kanban board name override. When set, all kanban cards for this "
                    "workflow run are created on the specified board. When empty, the engine "
                    "uses the YAML 'kanban_board' field or auto-creates 'wf_<workflow_name>'."
                ),
                "default": "",
            },
            "attachments": {
                "type": "object",
                "description": (
                    "Optional dict of named file paths to attach to kanban cards. "
                    "Keys are attachment names (must match declarations in the workflow YAML). "
                    "Values are local file paths. E.g. "
                    '{"grill_artifact": "/path/to/file.md", "source_video": "/path/to/video.mp4"}'
                ),
            },
        },
        "required": ["workflow"],
    },
}

WORKFLOW_VIEW_SCHEMA: Dict[str, Any] = {
    "name": "workflow_view",
    "description": (
        "Load a workflow template for inspection. Checks predefined YAML "
        "pipelines from docs/fleet-pipelines/ first, then user-saved dynamic "
        "templates from ~/.hermes/workflows/. Returns the template name, "
        "mode (predefined/dynamic), filesystem path, and raw YAML content."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "workflow": {
                "type": "string",
                "description": "Workflow name to load (without .yaml).",
            },
        },
        "required": ["workflow"],
    },
}

WORKFLOW_VALIDATE_SCHEMA: Dict[str, Any] = {
    "name": "workflow_validate",
    "description": (
        "Validate a pipeline definition without executing. Checks for cycles, "
        "missing dependencies, and unknown agent references. Returns "
        "{valid, nodes, layers, issues}. Safe to call any time."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "workflow": {
                "type": "string",
                "description": "Pipeline name to validate (without .yaml).",
            },
        },
        "required": ["workflow"],
    },
}

WORKFLOW_STATUS_SCHEMA: Dict[str, Any] = {
    "name": "workflow_status",
    "description": (
        "Query the current state of a running or last-completed pipeline. "
        "Omit the workflow argument to get status for all known pipelines. "
        "Mirrors the engine's internal state file plus live kanban card state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "workflow": {
                "type": "string",
                "description": "Pipeline name to query (omit for all).",
            },
        },
    },
}

WORKFLOW_LIST_SCHEMA: Dict[str, Any] = {
    "name": "workflow_list",
    "description": (
        "List available workflow definitions from both fleet pipelines "
        "(pre-defined) and user-saved templates (~/.hermes/workflows/). "
        "Returns workflows with metadata: name, description, trigger "
        "keywords, mode (predefined/dynamic), category, and path. "
        "Pass trigger to filter by keyword match."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "trigger": {
                "type": "string",
                "description": (
                    "Optional trigger string to filter workflows by keyword "
                    "match (case-insensitive). When provided, only workflows "
                    "whose trigger keywords appear in this string are returned."
                ),
            },
        },
    },
}

WORKFLOW_SHOW_SCHEMA: Dict[str, Any] = {
    "name": "workflow_show",
    "description": (
        "Show the structure of a pipeline: its nodes, the agent each node "
        "targets, the dependencies between nodes, and the layer index of "
        "each node. Use before workflow_start to understand the DAG."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "workflow": {
                "type": "string",
                "description": "Pipeline name to inspect (without .yaml).",
            },
        },
        "required": ["workflow"],
    },
}

# ---------------------------------------------------------------------------
# Dynamic workflow tools — model-authored DAGs (deprecated entry point)
# ---------------------------------------------------------------------------

def check_dynamic_workflow_requirements() -> bool:
    """Return True when the dynamic workflow engine can be invoked.

    Gates on the ``dynamic`` and ``dynamic_bridge`` modules being
    importable.  No external API keys or subprocess runners needed.
    """
    try:
        from plugins.workflow.dynamic import handle_workflow_dynamic  # noqa: F401
        from plugins.workflow.dynamic_bridge import run_dynamic_workflow  # noqa: F401
    except ImportError as exc:
        logger.debug("dynamic workflow import failed: %s", exc)
        return False
    return True


DYNAMIC_WORKFLOW_SCHEMA: Dict[str, Any] = {
    "name": "workflow_dynamic_start",
    "description": (
        "Deprecated: use workflow_start with mode='dynamic' instead. "
        "Start a dynamic (model-authored) workflow — create an ad-hoc DAG at "
        "runtime instead of reading pre-defined YAML pipelines.  Pass the "
        "workflow_id, objective, and node list.  Nodes define worker goals "
        "and dependencies; the engine dispatches ready nodes as background "
        "delegations and advances the graph as results arrive.\n\n"
        "Scope controls fleet integration:\n"
        "  - project (default): creates kanban cards on the dynamic-workflows board\n"
        "  - global: in-memory only, no kanban\n"
        "  - durable: persists node state to ~/.hermes/workflow-logs/\n\n"
        "Use workflow_status to query progress."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "workflow": {
                "type": "string",
                "description": (
                    "Workflow ID (unique identifier).  If empty, the engine "
                    "generates one.  Must match ^[A-Za-z0-9_.\\-]{1,96}$."
                ),
            },
            "context": {
                "type": "object",
                "description": (
                    "Workflow configuration object with keys: "
                    "'objective' (required, string — the high-level goal), "
                    "'nodes' (required, array — each with 'node_id', 'goal', "
                    "and optional 'depends_on'), 'context' (optional string "
                    "shared context for all nodes)."
                ),
            },
            "scope": {
                "type": "string",
                "enum": ["project", "global", "durable"],
                "description": (
                    "Fleet integration scope. 'project' creates kanban cards, "
                    "'durable' persists state, 'global' is in-memory only."
                ),
                "default": "project",
            },
            "single_flight": {
                "type": "boolean",
                "description": (
                    "If True, refuse to create a new workflow when a run "
                    "with the same workflow_id is already in progress."
                ),
                "default": False,
            },
            "dry_run": {
                "type": "boolean",
                "description": (
                    "Print the execution plan without creating kanban cards "
                    "or dispatching nodes."
                ),
                "default": False,
            },
        },
        "required": ["workflow", "context"],
    },
}
