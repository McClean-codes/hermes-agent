"""Workflow analyst — LLM-backed auxiliary for pipeline inference tasks.

Three analysis modes, one auxiliary key. Used by the workflow engine
when a task benefits from reasoning rather than mechanical processing.

Invoked via ``get_text_auxiliary_client("workflow_analyst")`` with
a mode-specific system prompt and structured JSON output schema.

Design notes
------------
* Mirrors ``hermes_cli/kanban_decompose.py``: lazy aux client import,
  lenient JSON parse, never raises on expected failure modes.
* Configured under ``auxiliary.workflow_analyst`` in config.yaml.
  Falls back to the auto provider when not explicitly set.
* System prompt defines the output schema — each mode produces
  a different JSON shape documented inline.
"""

from __future__ import annotations

import json as _json
import logging
import re as _re
from dataclasses import dataclass
from typing import Optional

from plugins.workflow.llm_utils import extract_json_blob

logger = logging.getLogger(__name__)


# ── Mode: escalation analysis ─────────────────────────────────────

_ESCALATION_SYSTEM = """You are the workflow escalation analyst for the Hermes Agent fleet.

A workflow engine revision loop has exhausted its 3-cycle limit. You are given
the full history of a review gate (the original spec/section, each round of
rejection, each round of revision, and the final state). Your job is to
produce a structured brief so the orchestrator (Sherlock) can resolve the
deadlock without re-reading all the raw card bodies.

Output a single JSON object:

  {
    "deadlock_type": "spec_disagreement" | "security_blocker" | "incomplete_decomposition" | "other",
    "summary": "<two-sentence summary of what's happening>",
    "rounds": [
      {"round": 1, "rejection": "<what was rejected>", "revision": "<what changed>"},
      ...
    ],
    "sticking_point": "<the one issue that never got resolved across all rounds>",
    "suggested_actions": ["<option A>", "<option B>", "<option C>"],
    "recommended_escalation": "sherlock_can_resolve" | "needs_randy"
  }

Rules:
- Be specific. Name the exact section/file/requirement in dispute.
- If the same issue appeared in all 3 rounds, that's the sticking point.
- suggested_actions should be concrete next steps, not vague guidance.
- If the deadlock is a genuine design trade-off (not a misunderstanding),
  recommended_escalation should be "needs_randy".
- No preamble, no code fences. Output only the JSON object."""


_ESCALATION_USER = """Project: {project}
Gate: {gate}
Verify node: {verify_node}

Loop history:
{loop_history}
"""


# ── Mode: pipeline status summary ─────────────────────────────────

_STATUS_SYSTEM = """You are the workflow status summariser for the Hermes Agent fleet.

You are given the raw state of a running pipeline (JSON from the engine's
state file). Produce a concise, human-readable status summary for the
orchestrator (Sherlock).

Output a single JSON object:

  {
    "pipeline": "<name>",
    "current_layer": <N>,
    "total_layers": <M>,
    "overall_status": "running" | "blocked" | "completed" | "failed",
    "layer_summary": [
      {
        "layer": <N>,
        "nodes": [
          {"node": "<id>", "agent": "<name>", "status": "<status>", "error": "<if any>"},
          ...
        ]
      },
      ...
    ],
    "attention_needed": [
      "<human-readable alert for any blocked/failed/timed_out node>"
    ],
    "estimated_completion": "<best guess, e.g. '~2 hours if unblocked'>"
  }

Rules:
- List ALL layers, not just the current one. Skip layers with all statuses "pending".
- For each node, include agent name, status, and error if present.
- attention_needed is empty array if nothing is blocked/failed/timed_out.
- Be concrete about estimated completion based on remaining timeout windows.
- No preamble, no code fences. Output only the JSON object."""


_STATUS_USER = """Pipeline: {pipeline_name}

Raw engine state:
{state_json}
"""


# ── Mode: failure diagnosis ────────────────────────────────────────

_FAILURE_SYSTEM = """You are the workflow failure diagnostician for the Hermes Agent fleet.

A node in the pipeline has failed or timed out. You are given:
- The node's task description (what it was supposed to do)
- The node's agent (which profile was working on it)
- The error or timeout details
- How long it ran before failing

Produce a structured diagnosis:

  {
    "likely_cause": "<most probable explanation>",
    "cause_category": "timeout" | "agent_error" | "config_missing" | "dependency_failure" | "resource_exhaustion" | "unknown",
    "evidence": ["<fact 1 supporting the diagnosis>", "<fact 2>", ...],
    "suggested_fix": "<concrete next step for Sherlock>",
    "should_retry": true | false,
    "retry_instructions": "<if should_retry: what to change before retrying, else empty string>"
  }

Rules:
- Be specific. If the error mentions a missing file, name it.
- If the node timed out after running the full timeout window, cause_category is likely "timeout" or "resource_exhaustion".
- suggested_fix should be one concrete action Sherlock can take.
- If the error is clearly transient (network timeout, API rate limit), should_retry should be true.
- No preamble, no code fences. Output only the JSON object."""


_FAILURE_USER = """Node: {node_id}
Agent: {agent}
Task: {task}
Timeout: {timeout_minutes}min
Elapsed before failure: {elapsed}
Error: {error}
"""


# ── Mode: extension suggestion ─────────────────────────────────────

_EXTENSION_SYSTEM = """You are the workflow extension analyst for the Hermes Agent fleet.

A worker node in a dynamic workflow has completed. You analyze the completion
summary and the workflow objective to suggest follow-up nodes that would
continue progress toward the objective.

Output a JSON array of objects:

  [
    {
      "node_id": "<suggested_node_id>",
      "goal": "<concise actionable goal for the worker>",
      "depends_on": ["<node_id this depends on>"]
    },
    ...
  ]

Rules:
- Suggest 0-3 follow-up nodes. If the objective appears fully met, return [].
- Each node_id must be unique and not already in the existing_nodes list.
- Goals should be concise and actionable (one sentence).
- depends_on should list node_ids from the existing graph (typically the node that just completed).
- Do not create circular dependencies.
- No preamble, no code fences. Output only the JSON array."""


_EXTENSION_USER = """Workflow objective: {objective}

Worker completion summary: {summary}

Existing nodes: {existing_nodes}

Based on the summary, suggest 0-3 follow-up nodes to continue working toward the objective.
If the objective appears to be met, return an empty array.
Only suggest nodes that are NOT already in the existing_nodes list."""


# ── Mode: loop decision ─────────────────────────────────────

_LOOP_DECISION_SYSTEM = """You are the workflow loop decision analyst for the Hermes Agent fleet.

A verify node has rejected a worker's output. You must decide whether to loop back
for revision or proceed to the next phase. You are given:

1. The verify node's task body — contains the pass/fail criteria
2. The rejection reason — what the verifier reported
3. The revision node's task — what the fix would do

Evaluate the rejection against the criteria. Consider:
- Does the rejection genuinely violate the criteria?
- Could the verifier have made a mistake (false positive)?
- Is the rejection a matter of interpretation rather than a clear failure?

Output a single JSON object:

  {
    "decision": "loop" | "proceed",
    "reason": "<one-sentence explanation>",
    "confidence": "high" | "medium" | "low"
  }

Rules:
- "loop" = the rejection is valid, send back for revision
- "proceed" = the rejection does not match the criteria, or the output is acceptable
- Be conservative: if unsure, prefer "loop" so the verifier's judgment is respected
- No preamble, no code fences. Output only the JSON object."""

_LOOP_DECISION_USER = """Verify node task body (contains pass/fail criteria):
{verify_task}

Rejection reason:
{rejection}

Revision node task (what the fix would do):
{revision_task}

Based on the criteria in the verify node's task body, should we loop back for revision or proceed?"""


def analyze_loop_decision(
    *,
    verify_task: str = "",
    rejection: str = "",
    revision_task: str = "",
    timeout: Optional[int] = None,
) -> AnalystOutcome:
    """Evaluate a LOOP rejection against the verify node's criteria and decide whether to loop or proceed."""
    user_msg = _LOOP_DECISION_USER.format(
        verify_task=verify_task,
        rejection=rejection,
        revision_task=revision_task,
    )
    return _invoke(
        mode="loop_decision",
        system_prompt=_LOOP_DECISION_SYSTEM,
        user_message=user_msg,
        timeout=timeout,
    )


# ── Mode: unexpected block notification ──────────────────────────

_BLOCK_NOTIFY_SYSTEM = """You are the workflow anomaly analyst for the Hermes Agent fleet.

A node in a workflow has been unexpectedly blocked (not a LOOP revision).
You must assess the situation and produce a concise report for the calling agent.

You are given:
1. The node's task body — what it was supposed to do
2. The block reason — why it was blocked
3. The workflow context — what project/repo this is for

Produce a concise JSON report:

  {
    "severity": "critical" | "warning" | "info",
    "summary": "One-line summary of what happened",
    "detail": "2-3 sentence explanation of the failure and its impact",
    "suggested_action": "What the calling agent should do next"
  }

Rules:
- severity=critical: workflow cannot proceed without human intervention
- severity=warning: workflow can proceed but something is wrong
- severity=info: informational, no action needed
- Be concise — the calling agent needs a quick summary, not an essay
- No preamble, no code fences. Output only the JSON object."""

_BLOCK_NOTIFY_USER = """Node "{node_id}" in workflow "{workflow_name}" has been unexpectedly blocked.

Node task (what it was supposed to do):
{node_task}

Block reason:
{block_reason}

Workflow context:
{workflow_context}

Assess the situation and produce a report for the calling agent."""


def analyze_block_notification(
    *,
    node_id: str = "",
    workflow_name: str = "",
    node_task: str = "",
    block_reason: str = "",
    workflow_context: str = "",
    timeout: Optional[int] = None,
) -> AnalystOutcome:
    """Assess an unexpected block and produce a report for the calling agent."""
    user_msg = _BLOCK_NOTIFY_USER.format(
        node_id=node_id,
        workflow_name=workflow_name,
        node_task=node_task,
        block_reason=block_reason,
        workflow_context=workflow_context,
    )
    return _invoke(
        mode="block_notification",
        system_prompt=_BLOCK_NOTIFY_SYSTEM,
        user_message=user_msg,
        timeout=timeout,
    )


@dataclass
class AnalystOutcome:
    """Result of an analyst invocation."""
    mode: str
    success: bool
    result: Optional[dict] = None
    error: Optional[str] = None
    raw_response: Optional[str] = None


# ── Public API ─────────────────────────────────────────────────────

def analyze_escalation(
    *,
    project: str = "",
    gate: str = "",
    verify_node: str = "",
    loop_history: str = "",
    timeout: Optional[int] = None,
) -> AnalystOutcome:
    """Analyze a deadlocked revision loop and produce a structured brief."""
    user_msg = _ESCALATION_USER.format(
        project=project,
        gate=gate,
        verify_node=verify_node,
        loop_history=loop_history,
    )
    return _invoke(
        mode="escalation",
        system_prompt=_ESCALATION_SYSTEM,
        user_message=user_msg,
        timeout=timeout,
    )


def analyze_status(
    *,
    pipeline_name: str = "",
    state_json: str = "",
    timeout: Optional[int] = None,
) -> AnalystOutcome:
    """Summarize engine state into a human-readable status report."""
    user_msg = _STATUS_USER.format(
        pipeline_name=pipeline_name,
        state_json=state_json,
    )
    return _invoke(
        mode="status",
        system_prompt=_STATUS_SYSTEM,
        user_message=user_msg,
        timeout=timeout,
    )


def analyze_failure(
    *,
    node_id: str = "",
    agent: str = "",
    task: str = "",
    timeout_minutes: int = 30,
    elapsed: str = "",
    error: str = "",
    timeout: Optional[int] = None,
) -> AnalystOutcome:
    """Diagnose a node failure and suggest remediation."""
    user_msg = _FAILURE_USER.format(
        node_id=node_id,
        agent=agent,
        task=task,
        timeout_minutes=timeout_minutes,
        elapsed=elapsed,
        error=error,
    )
    return _invoke(
        mode="failure",
        system_prompt=_FAILURE_SYSTEM,
        user_message=user_msg,
        timeout=timeout,
    )


def analyze_extension(
    *,
    summary: str = "",
    objective: str = "",
    existing_nodes: list[str] | None = None,
    timeout: Optional[int] = None,
) -> list[dict]:
    """Suggest follow-up nodes after a worker completes.

    Calls the workflow_analyst auxiliary with the extension prompt.
    Returns a list of suggestion dicts (node_id, goal, depends_on)
    or an empty list if the analyst is unavailable / returns nothing useful.
    """
    if existing_nodes is None:
        existing_nodes = []
    user_msg = _EXTENSION_USER.format(
        summary=summary,
        objective=objective,
        existing_nodes=", ".join(existing_nodes) if existing_nodes else "(none)",
    )
    outcome = _invoke(
        mode="extension",
        system_prompt=_EXTENSION_SYSTEM,
        user_message=user_msg,
        timeout=timeout,
    )

    # Try parsing from result dict first (LLM might wrap array in object)
    if outcome.success and isinstance(outcome.result, dict):
        for key in ("nodes", "suggestions", "extensions"):
            val = outcome.result.get(key)
            if isinstance(val, list):
                return _validate_suggestions(val)

    # Fall back to parsing raw response as JSON array
    raw = outcome.raw_response or ""
    nodes = _extract_json_list(raw)
    if nodes is not None:
        return _validate_suggestions(nodes)

    return []


# ── Internal ───────────────────────────────────────────────────────

def _invoke(
    *,
    mode: str,
    system_prompt: str,
    user_message: str,
    timeout: Optional[int] = None,
) -> AnalystOutcome:
    """Call the auxiliary LLM and return a structured outcome."""
    try:
        from agent.auxiliary_client import (  # type: ignore
            get_auxiliary_extra_body,
            get_text_auxiliary_client,
        )
    except Exception as exc:
        logger.debug("workflow_analyst: aux client import failed: %s", exc)
        return AnalystOutcome(mode=mode, success=False, error="auxiliary client unavailable")

    try:
        client, model = get_text_auxiliary_client("workflow_analyst")
    except Exception as exc:
        logger.debug("workflow_analyst: get_text_auxiliary_client failed: %s", exc)
        return AnalystOutcome(mode=mode, success=False, error="auxiliary client unavailable")

    if client is None or not model:
        return AnalystOutcome(mode=mode, success=False, error="no auxiliary client configured")

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=2000,
            timeout=timeout or 180,
            extra_body=get_auxiliary_extra_body() or None,
        )
    except Exception as exc:
        logger.info("workflow_analyst: API call failed for mode=%s (%s)", mode, exc)
        return AnalystOutcome(mode=mode, success=False, error=f"LLM error: {type(exc).__name__}")

    try:
        raw = resp.choices[0].message.content or ""
    except Exception:
        raw = ""

    parsed = extract_json_blob(raw)
    if parsed is None:
        return AnalystOutcome(
            mode=mode, success=False,
            error="LLM returned malformed JSON",
            raw_response=raw,
        )

    return AnalystOutcome(mode=mode, success=True, result=parsed, raw_response=raw)


# ── Extension helpers ──────────────────────────────────────────────

_FENCE_RE = _re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", _re.IGNORECASE)


def _extract_json_list(raw: str) -> list | None:
    """Lenient extraction of a JSON array from an LLM response.

    Mirrors extract_json_blob but handles ``[...]`` arrays.
    Returns the parsed list, or ``None`` if extraction fails.
    """
    if not raw:
        return None
    stripped = _FENCE_RE.sub("", raw.strip())
    first = stripped.find("[")
    last = stripped.rfind("]")
    if first == -1 or last == -1 or last <= first:
        return None
    candidate = stripped[first:last + 1]
    try:
        val = _json.loads(candidate)
    except (ValueError, _json.JSONDecodeError):
        return None
    if not isinstance(val, list):
        return None
    return val


def _validate_suggestions(items: list) -> list[dict]:
    """Validate and normalise extension suggestions from the analyst."""
    validated: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        nid = str(item.get("node_id") or "").strip()
        goal = str(item.get("goal") or "").strip()
        if not nid or not goal:
            continue
        depends_on = item.get("depends_on")
        if not isinstance(depends_on, list):
            depends_on = []
        validated.append({
            "node_id": nid,
            "goal": goal,
            "depends_on": [str(d).strip() for d in depends_on if str(d).strip()],
        })
    return validated
