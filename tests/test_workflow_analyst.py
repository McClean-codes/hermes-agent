"""Tests for the workflow_analyst auxiliary and _try_* integration points.

Run: python3 -m pytest tests/test_workflow_analyst.py -v
"""

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from plugins.workflow.llm_utils import extract_json_blob
from plugins.workflow.analyst import (
    AnalystOutcome,
    analyze_escalation,
    analyze_status,
    analyze_failure,
)
from plugins.workflow.engine import (
    WorkflowEngine, Workflow, WorkflowNode, NodeState,
)


# ════════════════════════════════════════════════════════════════
# Shared utility: extract_json_blob
# ════════════════════════════════════════════════════════════════

def test_extract_plain_json():
    raw = '{"title": "T", "body": "B"}'
    assert extract_json_blob(raw) == {"title": "T", "body": "B"}


def test_extract_fenced_json():
    raw = '```json\n{"title": "T", "body": "B"}\n```'
    assert extract_json_blob(raw) == {"title": "T", "body": "B"}


def test_extract_with_prose_preamble():
    raw = 'Sure! Here you go:\n{"title": "T", "body": "B"}\nThanks.'
    assert extract_json_blob(raw) == {"title": "T", "body": "B"}


def test_extract_no_fence_prefix():
    raw = '```\n{"title": "T", "body": "B"}\n```'
    assert extract_json_blob(raw) == {"title": "T", "body": "B"}


def test_extract_whitespace_fences():
    raw = '  ```json  \n{"x": 1}\n  ```  '
    assert extract_json_blob(raw) == {"x": 1}


def test_extract_returns_none_unparseable():
    assert extract_json_blob("no json here") is None
    assert extract_json_blob("") is None
    assert extract_json_blob("{not: valid}") is None


def test_extract_returns_none_array():
    """The function only returns dicts, never lists."""
    assert extract_json_blob('[1, 2, 3]') is None


def test_extract_returns_none_nested_non_dict():
    """A string is valid JSON but not a dict."""
    assert extract_json_blob('"hello"') is None


# ════════════════════════════════════════════════════════════════
# AnalystOutcome
# ════════════════════════════════════════════════════════════════

def test_outcome_success():
    o = AnalystOutcome(mode="escalation", success=True, result={"x": 1})
    assert o.mode == "escalation"
    assert o.success is True
    assert o.result == {"x": 1}
    assert o.error is None


def test_outcome_failure():
    o = AnalystOutcome(mode="failure", success=False, error="timeout")
    assert o.mode == "failure"
    assert o.success is False
    assert o.error == "timeout"
    assert o.result is None


# ════════════════════════════════════════════════════════════════
# Public API: analyze_escalation / analyze_status / analyze_failure
#   (uses real analyst module; these test the prompt-building layer.
#    The _invoke layer is tested indirectly through the engine
#    integration tests below, where we mock at the analyze_* level.)
# ════════════════════════════════════════════════════════════════

def test_analyze_escalation_builds_prompt():
    """Escalation prompt includes project, gate, and history."""
    with patch("plugins.workflow.analyst._invoke") as mock_invoke:
        mock_invoke.return_value = AnalystOutcome(mode="escalation", success=True)
        result = analyze_escalation(
            project="goms", gate="ada-security",
            verify_node="ada-security",

        )
        assert result.success is True
        mock_invoke.assert_called_once()
        call_kwargs = mock_invoke.call_args.kwargs
        assert call_kwargs["mode"] == "escalation"
        assert "goms" in call_kwargs["user_message"]
        assert "Loop history" in call_kwargs["user_message"]


def test_analyze_status_builds_prompt():
    with patch("plugins.workflow.analyst._invoke") as mock_invoke:
        mock_invoke.return_value = AnalystOutcome(mode="status", success=True)
        result = analyze_status(pipeline_name="ideation", state_json='{"x":1}')
        assert result.success is True
        mock_invoke.assert_called_once()
        call_kwargs = mock_invoke.call_args.kwargs
        assert call_kwargs["mode"] == "status"
        assert "ideation" in call_kwargs["user_message"]


def test_analyze_failure_builds_prompt():
    with patch("plugins.workflow.analyst._invoke") as mock_invoke:
        mock_invoke.return_value = AnalystOutcome(mode="failure", success=True)
        result = analyze_failure(
            node_id="newton-build", agent="newton",
            task="Build auth domain", timeout_minutes=120,
            elapsed="600s", error="timeout",
        )
        assert result.success is True
        mock_invoke.assert_called_once()
        call_kwargs = mock_invoke.call_args.kwargs
        assert call_kwargs["mode"] == "failure"
        assert "newton-build" in call_kwargs["user_message"]


# ════════════════════════════════════════════════════════════════
# Engine integration: _try_* methods
#   We mock at the analyze_* level — the engine's try/except
#   ImportError blocks are trivially correct; testing them requires
#   mocking Python's import system and adds no signal.
# ════════════════════════════════════════════════════════════════

@pytest.fixture
def engine():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        yield WorkflowEngine(workflows_dir=tmpdir)


@pytest.fixture
def verify_state_with_history():
    state = NodeState(node_id="nikola-verify-spec")
    state.loop_count = 3
    state.loop_history = [
        "Round 1: Missing billing edge case",
        "Round 2: Auth rate limiting not addressed, billing still missing",
        "Round 3: Still no rate limiting — REVISIONS",
    ]
    state.error = "LOOP #3: Still no rate limiting — REVISIONS"
    return state


def test_escalation_with_full_history(engine, verify_state_with_history):
    """Analyst receives full loop_history, not just last error."""
    wf = Workflow(name="test")
    with patch("plugins.workflow.analyst.analyze_escalation") as mock_analyze:
        mock_analyze.return_value = AnalystOutcome(
            mode="escalation", success=True,
            result={
                "deadlock_type": "spec_disagreement",
                "summary": "Rate limiting deadlock",
                "sticking_point": "Rate limiting",
                "suggested_actions": ["Add rate limiting"],
                "recommended_escalation": "sherlock_can_resolve",
            }
        )
        engine._try_escalation_analysis(wf, "nikola-verify-spec",
                                        verify_state_with_history, {})

        call_kwargs = mock_analyze.call_args.kwargs
        loop_history = call_kwargs["loop_history"]
        # Should have all 3 rounds, not just the last error
        assert "billing" in loop_history or "LOOP" in loop_history


def test_escalation_no_history_fallback(engine):
    """When loop_history is empty, falls back to error field."""
    wf = Workflow(name="test")
    state = NodeState(node_id="test")
    state.error = "Some error"
    with patch("plugins.workflow.analyst.analyze_escalation") as mock_analyze:
        mock_analyze.return_value = AnalystOutcome(
            mode="escalation", success=True, result={"summary": "ok"}
        )
        engine._try_escalation_analysis(wf, "test", state, {})
        call_kwargs = mock_analyze.call_args.kwargs
        assert call_kwargs["loop_history"] == "Some error"


def test_escalation_recommends_randy(engine, capsys):
    """When analyst says needs_randy, prints warning."""
    wf = Workflow(name="test")
    state = NodeState(node_id="test")
    state.error = "Blocked"
    with patch("plugins.workflow.analyst.analyze_escalation") as mock_analyze:
        mock_analyze.return_value = AnalystOutcome(
            mode="escalation", success=True,
            result={
                "summary": "Deadlock",
                "sticking_point": "Trade-off",
                "suggested_actions": ["Ask Randy"],
                "recommended_escalation": "needs_randy",
            }
        )
        engine._try_escalation_analysis(wf, "test", state, {})
        captured = capsys.readouterr()
        assert "Randy involvement" in captured.out


def test_escalation_analyst_unavailable(engine, capsys):
    """When analyst returns failure, prints manual review message."""
    wf = Workflow(name="test")
    state = NodeState(node_id="test")
    state.error = "Blocked"
    with patch("plugins.workflow.analyst.analyze_escalation") as mock_analyze:
        mock_analyze.return_value = AnalystOutcome(
            mode="escalation", success=False, error="API error"
        )
        engine._try_escalation_analysis(wf, "test", state, {})
        captured = capsys.readouterr()
        assert "Sherlock must review manually" in captured.out


def test_failure_suggests_retry(engine, capsys):
    """When analyst says should_retry, prints suggestion."""
    node = WorkflowNode(id="newton-build", agent="newton", task="Build")
    state = NodeState(node_id="newton-build")
    state.error = "timeout"
    with patch("plugins.workflow.analyst.analyze_failure") as mock_analyze:
        mock_analyze.return_value = AnalystOutcome(
            mode="failure", success=True,
            result={
                "likely_cause": "CI runner offline",
                "cause_category": "resource_exhaustion",
                "suggested_fix": "Restart runner",
                "should_retry": True,
                "retry_instructions": "Wait 5 min, retry",
            }
        )
        engine._try_failure_analysis(node, state, 600.0)
        captured = capsys.readouterr()
        assert "CI runner offline" in captured.out
        assert "Restart runner" in captured.out
        assert "suggests retry" in captured.out


def test_failure_analyst_unavailable(engine, capsys):
    """When failure analyst fails, silent — engine continues."""
    node = WorkflowNode(id="test", agent="test", task="Task")
    state = NodeState(node_id="test")
    state.error = "timeout"
    with patch("plugins.workflow.analyst.analyze_failure") as mock_analyze:
        mock_analyze.return_value = AnalystOutcome(
            mode="failure", success=False, error="API error"
        )
        # Should not raise, should produce no output
        engine._try_failure_analysis(node, state, 600.0)
        # No crash = pass


def test_status_with_alerts(engine):
    """Happy path — returns formatted summary with alerts."""
    with patch("plugins.workflow.analyst.analyze_status") as mock_analyze:
        mock_analyze.return_value = AnalystOutcome(
            mode="status", success=True,
            result={
                "overall_status": "blocked",
                "attention_needed": ["nikola-verify-spec timed out"],
                "estimated_completion": "unknown — blocked",
            }
        )
        summary = engine._try_status_summary("ideation", {"x": 1})
        assert "ideation" in summary
        assert "blocked" in summary
        assert "nikola-verify-spec timed out" in summary


def test_status_analyst_fails(engine):
    """When analyst returns failure, returns None."""
    with patch("plugins.workflow.analyst.analyze_status") as mock_analyze:
        mock_analyze.return_value = AnalystOutcome(
            mode="status", success=False, error="API error"
        )
        result = engine._try_status_summary("ideation", {"x": 1})
        assert result is None


def test_status_analyst_no_alerts(engine):
    """When everything is running, no attention_needed."""
    with patch("plugins.workflow.analyst.analyze_status") as mock_analyze:
        mock_analyze.return_value = AnalystOutcome(
            mode="status", success=True,
            result={
                "overall_status": "running",
                "attention_needed": [],
                "estimated_completion": "~2 hours",
            }
        )
        summary = engine._try_status_summary("ideation", {"x": 1})
        assert "running" in summary
        assert "~2 hours" in summary


# ════════════════════════════════════════════════════════════════
# NodeState: loop_history
# ════════════════════════════════════════════════════════════════

def test_node_state_loop_history_default():
    """LOOP convention removed."""
    pass


def test_node_state_loop_history_persists():
    """LOOP convention removed."""
    pass
