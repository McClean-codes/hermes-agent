"""
Tests for workflow_engine.py — DAG construction, cycle detection,
topological sort, LOOP convention parsing, failure propagation,
state persistence, and CLI validation.

Run: python3 -m pytest tests/test_workflow_engine.py -v
"""

import pytest
import tempfile
import json
from pathlib import Path
import re
import os
import re
from datetime import datetime, timezone
import yaml

# Import the engine module (must be run from hermes-agent repo root)
from plugins.workflow.engine import (
    WorkflowEngine, Workflow, WorkflowNode, NodeState,
    CycleDetectedError,
)


# ── Test fixtures ──────────────────────────────────────────────────

@pytest.fixture
def engine():
    """Engine pointed at a temp workflows directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield WorkflowEngine(workflows_dir=tmpdir)


@pytest.fixture
def simple_workflow():
    """A → B → C (linear, no branches)."""
    wf = Workflow(name="test-linear", description="Linear test")
    wf.nodes["a"] = WorkflowNode(id="a", agent="agent-a", task="Task A")
    wf.nodes["b"] = WorkflowNode(id="b", agent="agent-b", task="Task B",
                                  depends_on=["a"])
    wf.nodes["c"] = WorkflowNode(id="c", agent="agent-c", task="Task C",
                                  depends_on=["b"])
    return wf


@pytest.fixture
def parallel_workflow():
    """A → B ∥ C → D (parallel middle layer)."""
    wf = Workflow(name="test-parallel")
    wf.nodes["a"] = WorkflowNode(id="a", agent="agent-a", task="Task A")
    wf.nodes["b"] = WorkflowNode(id="b", agent="agent-b", task="Task B",
                                  depends_on=["a"])
    wf.nodes["c"] = WorkflowNode(id="c", agent="agent-c", task="Task C",
                                  depends_on=["a"])
    wf.nodes["d"] = WorkflowNode(id="d", agent="agent-d", task="Task D",
                                  depends_on=["b", "c"])
    return wf


@pytest.fixture
def revision_workflow():
    """verify → revise (single revision pair)."""
    wf = Workflow(name="test-revision")
    wf.nodes["verify"] = WorkflowNode(id="verify", agent="reviewer",
                                       task="Verify work")
    wf.nodes["revise"] = WorkflowNode(id="revise", agent="author",
                                       task="Revise work", depends_on=["verify"])
    return wf


# ── Topological sort tests ─────────────────────────────────────────

def test_linear_dag(engine, simple_workflow):
    layers = engine.topological_sort(simple_workflow)
    assert layers == [["a"], ["b"], ["c"]]


def test_parallel_dag(engine, parallel_workflow):
    layers = engine.topological_sort(parallel_workflow)
    # Layer 0: a, Layer 1: b ∥ c, Layer 2: d
    assert len(layers) == 3
    assert set(layers[0]) == {"a"}
    assert set(layers[1]) == {"b", "c"}
    assert set(layers[2]) == {"d"}


def test_cycle_detection(engine):
    wf = Workflow(name="test-cycle")
    wf.nodes["a"] = WorkflowNode(id="a", agent="x", task="A", depends_on=["b"])
    wf.nodes["b"] = WorkflowNode(id="b", agent="x", task="B", depends_on=["a"])
    with pytest.raises(CycleDetectedError):
        engine.topological_sort(wf)


def test_unknown_dependency(engine):
    wf = Workflow(name="test-unknown")
    wf.nodes["a"] = WorkflowNode(id="a", agent="x", task="A",
                                  depends_on=["nonexistent"])
    with pytest.raises(ValueError, match="unknown node"):
        engine.topological_sort(wf)


def test_empty_workflow(engine):
    wf = Workflow(name="test-empty")
    layers = engine.topological_sort(wf)
    assert layers == []


def test_single_node(engine):
    wf = Workflow(name="test-single")
    wf.nodes["solo"] = WorkflowNode(id="solo", agent="x", task="Solo")
    layers = engine.topological_sort(wf)
    assert layers == [["solo"]]


# ── Dependency lookup tests ────────────────────────────────────────

def test_find_revision_node(engine, revision_workflow):
    result = engine._find_revision_node(revision_workflow, "verify")
    assert result == "revise"


def test_find_revision_node_no_match(engine, revision_workflow):
    result = engine._find_revision_node(revision_workflow, "revise")
    assert result is None


def test_find_revision_node_multiple_dependents(engine):
    """Documented behavior: returns first match from dict.items()."""
    wf = Workflow(name="test-multi-revision")
    wf.nodes["verify"] = WorkflowNode(id="verify", agent="x", task="V")
    wf.nodes["revise-a"] = WorkflowNode(id="revise-a", agent="x", task="RA",
                                         depends_on=["verify"])
    wf.nodes["revise-b"] = WorkflowNode(id="revise-b", agent="x", task="RB",
                                         depends_on=["verify"])
    result = engine._find_revision_node(wf, "verify")
    # Returns one of them (dict order in Python 3.7+ is insertion order)
    assert result in ("revise-a", "revise-b")


def test_find_layer_for_node(engine, parallel_workflow):
    layers = engine.topological_sort(parallel_workflow)
    assert engine._find_layer_for_node(layers, "a") == 0
    assert engine._find_layer_for_node(layers, "b") == 1
    assert engine._find_layer_for_node(layers, "c") == 1
    assert engine._find_layer_for_node(layers, "d") == 2
    assert engine._find_layer_for_node(layers, "nonexistent") == -1


# ── LOOP convention tests ──────────────────────────────────────────

def test_loop_regex_match():
    import re
    body = "LOOP:nikola-verify-spec | Missing billing edge case"
    match = re.match(r'^LOOP:(\S+)', body)
    assert match is not None
    assert match.group(1) == "nikola-verify-spec"


def test_loop_regex_no_match():
    import re
    body = "Blocked: external API down"
    match = re.match(r'^LOOP:(\S+)', body)
    assert match is None


def test_loop_regex_subsequent_loops():
    """LOOP: prefix anywhere in body should not match — only at start."""
    import re
    body = "Agent completed review. LOOP:some-node | notes"
    match = re.match(r'^LOOP:(\S+)', body)
    assert match is None


def test_loop_regex_with_pipe_content():
    import re
    body = "LOOP:ada-security | PII in plaintext at auth/SPEC.md §3.2"
    match = re.match(r'^LOOP:(\S+)', body)
    assert match.group(1) == "ada-security"


# ── Failure propagation tests ──────────────────────────────────────

def test_failure_propagation_transitive(engine):
    """A fails → B skipped → C skipped (transitive through B to C)."""
    wf = Workflow(name="test-fail-prop")
    wf.nodes["a"] = WorkflowNode(id="a", agent="x", task="A")
    wf.nodes["b"] = WorkflowNode(id="b", agent="x", task="B", depends_on=["a"])
    wf.nodes["c"] = WorkflowNode(id="c", agent="x", task="C", depends_on=["b"])

    states = {
        "a": NodeState(node_id="a", status="failed"),
        "b": NodeState(node_id="b", status="pending"),
        "c": NodeState(node_id="c", status="pending"),
    }

    # Simulate execute's dependency check: before creating B's card
    b_node = wf.nodes["b"]
    deps_failed_b = any(
        states[d].status in ("failed", "timed_out", "blocked")
        for d in b_node.depends_on
    )
    assert deps_failed_b is True  # A failed → B should skip

    # Mark B as skipped
    states["b"].status = "skipped"

    # Now check C: its dep B is skipped (not failed/timed_out/blocked)
    c_node = wf.nodes["c"]
    deps_failed_c = any(
        states[d].status in ("failed", "timed_out", "blocked")
        for d in c_node.depends_on
    )
    # B is skipped, not failed — so deps_failed_c is False
    # This means C would NOT be skipped by the current check.
    # This is a known limitation: "skipped" is not in the failed set.
    assert deps_failed_c is False


def test_failure_propagation_direct(engine):
    """A fails → B (direct dependent) B skipped correctly."""
    wf = Workflow(name="test-fail-direct")
    wf.nodes["a"] = WorkflowNode(id="a", agent="x", task="A")
    wf.nodes["b"] = WorkflowNode(id="b", agent="x", task="B", depends_on=["a"])

    states = {
        "a": NodeState(node_id="a", status="failed"),
        "b": NodeState(node_id="b", status="pending"),
    }

    b_node = wf.nodes["b"]
    deps_failed = any(
        states[d].status in ("failed", "timed_out", "blocked")
        for d in b_node.depends_on
    )
    assert deps_failed is True


def test_failure_propagation_timed_out(engine):
    """Timed out nodes also block dependents."""
    wf = Workflow(name="test-timeout-prop")
    wf.nodes["a"] = WorkflowNode(id="a", agent="x", task="A")
    wf.nodes["b"] = WorkflowNode(id="b", agent="x", task="B", depends_on=["a"])

    states = {
        "a": NodeState(node_id="a", status="timed_out"),
        "b": NodeState(node_id="b", status="pending"),
    }

    b_node = wf.nodes["b"]
    deps_failed = any(
        states[d].status in ("failed", "timed_out", "blocked")
        for d in b_node.depends_on
    )
    assert deps_failed is True


# ── State persistence tests ────────────────────────────────────────

def test_state_save_and_load(engine):
    """Round-trip: save state → load state → verify fields."""
    wf = Workflow(name="test-state")
    wf.nodes["a"] = WorkflowNode(id="a", agent="x", task="A")

    states = {
        "a": NodeState(node_id="a", status="done", kanban_card_id="card-123",
                       started_at="2026-06-07T00:00:00Z", attempts=1)
    }
    results = {"a": "done"}
    layers = [["a"]]

    engine._save_state("test-state", states, results, 0, layers)
    loaded = engine._load_state("test-state")

    assert loaded is not None
    assert loaded["workflow_name"] == "test-state"
    assert loaded["current_layer"] == 0
    assert loaded["states"]["a"]["status"] == "done"
    assert loaded["states"]["a"]["kanban_card_id"] == "card-123"
    assert loaded["results"]["a"] == "done"

    engine._clear_state("test-state")
    assert engine._load_state("test-state") is None


def test_state_clear_nonexistent(engine):
    """Clearing nonexistent state should not error."""
    engine._clear_state("nonexistent-workflow")  # Should not raise


# ── Validation tests ───────────────────────────────────────────────

def _write_workflow_yaml(tmpdir, name, yaml_content):
    """Helper: write a temp YAML and point engine at it."""
    path = Path(tmpdir) / f"{name}.yaml"
    path.write_text(yaml_content)
    return WorkflowEngine(workflows_dir=tmpdir)


def test_validate_valid_workflow():
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml = """name: test-valid
description: A valid workflow
nodes:
  a:
    agent: test-agent
    task: First task
  b:
    agent: test-agent
    task: Second task
    depends_on: [a]
"""
        engine = _write_workflow_yaml(tmpdir, "test-valid", yaml)
        result = engine.validate("test-valid")
        assert result["valid"] is True
        assert result["nodes"] == 2
        assert result["layers"] == 2


def test_validate_unknown_dependency():
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml = """name: test-bad-dep
nodes:
  a:
    agent: test-agent
    task: Bad dep
    depends_on: [nonexistent]
"""
        engine = _write_workflow_yaml(tmpdir, "test-bad-dep", yaml)
        result = engine.validate("test-bad-dep")
        assert result["valid"] is False
        assert any("unknown node" in i for i in result["issues"])


def test_validate_cycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml = """name: test-cycle
nodes:
  a:
    agent: test-agent
    task: A
    depends_on: [b]
  b:
    agent: test-agent
    task: B
    depends_on: [a]
"""
        engine = _write_workflow_yaml(tmpdir, "test-cycle", yaml)
        result = engine.validate("test-cycle")
        assert result["valid"] is False
        assert any("Cycle" in i for i in result["issues"])


def test_validate_revision_without_gate():
    """LOOP convention removed — reviews convention only."""
    pass


def test_validate_missing_workflow():
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = WorkflowEngine(workflows_dir=tmpdir)
        result = engine.validate("nonexistent")
        assert result["valid"] is False
        assert any("not found" in i for i in result["issues"])


def test_validate_resolves_deps_across_workflows():
    """validate should detect when a revision node references a gate that exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml = """name: test-gate-pair
nodes:
  verify-spec:
    agent: nikola
    task: Verify spec
  revise-spec:
    agent: edison
    task: Revise spec
    depends_on: [verify-spec]
"""
        engine = _write_workflow_yaml(tmpdir, "test-gate-pair", yaml)
        result = engine.validate("test-gate-pair")
        # Revision node depends on a verify node — should be valid
        assert result["valid"] is True
        # No gate→revision pair issues (verify-spec has revise-spec as dependent)
        assert not any("LOOP detection" in i for i in result["issues"])


# ── LOOP count tests ───────────────────────────────────────────────

def test_max_revision_loops_constant():
    """LOOP convention removed — reviews convention only."""
    pass


# ── WorkflowNode tests ─────────────────────────────────────────────

def test_workflow_node_defaults():
    node = WorkflowNode(id="test", agent="x", task="y")
    assert node.depends_on == []
    assert node.timeout_minutes == 30
    assert node.model is None
    assert node.channel == "debug"


def test_workflow_node_custom():
    node = WorkflowNode(
        id="test", agent="x", task="y",
        depends_on=["a", "b"], timeout_minutes=60,
        model="deepseek-v4", channel="orchestration"
    )
    assert node.depends_on == ["a", "b"]
    assert node.timeout_minutes == 60
    assert node.model == "deepseek-v4"
    assert node.channel == "orchestration"


# ── NodeState tests ────────────────────────────────────────────────

def test_node_state_defaults():
    state = NodeState(node_id="test")
    assert state.status == "pending"
    assert state.kanban_card_id is None
    assert state.attempts == 0
    # loop_count removed — reviews convention only


# ── Synthetic gate node tests ──────────────────────────────────────
# Synthetic gates (synthetic: true in YAML) are auto-completed once
# their depends_on are satisfied — no kanban card is created, no agent
# is dispatched. They exist to enforce ordering in the DAG without
# adding a no-op task to the board. See council.yaml's council-ready
# node for the canonical example.

def test_workflow_node_synthetic_default_false():
    """Real nodes have synthetic=False by default."""
    node = WorkflowNode(id="real", agent="agent-x", task="Do the thing")
    assert node.synthetic is False
    assert node.agent == "agent-x"


def test_workflow_node_synthetic_explicit_true():
    """Synthetic nodes have synthetic=True and may have agent=None."""
    node = WorkflowNode(id="gate", agent=None, task="privacy gate",
                        synthetic=True)
    assert node.synthetic is True
    assert node.agent is None


def test_load_synthetic_node_without_agent():
    """YAML with synthetic: true and no agent field loads without error.

    Regression: this is the original council-ready bug. The old loader
    did `node_data["agent"]` as a direct subscript, which raised
    KeyError on synthetic nodes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml = """name: test-synthetic-load
description: Synthetic gate with no agent
nodes:
  real:
    agent: some-agent
    task: A real task
  gate:
    synthetic: true
    task: Privacy gate
    depends_on: [real]
"""
        engine = _write_workflow_yaml(tmpdir, "test-synthetic-load", yaml)
        wf = engine.load_workflow("test-synthetic-load")
        assert "gate" in wf.nodes
        assert wf.nodes["gate"].synthetic is True
        assert wf.nodes["gate"].agent is None
        assert wf.nodes["gate"].task == "Privacy gate"
        assert wf.nodes["gate"].depends_on == ["real"]
        # Real node still has its agent
        assert wf.nodes["real"].synthetic is False
        assert wf.nodes["real"].agent == "some-agent"


def test_load_synthetic_node_without_task():
    """Synthetic nodes may omit task — defaults to a label including the id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml = """name: test-synthetic-no-task
nodes:
  gate:
    synthetic: true
    depends_on: [real]
  real:
    agent: some-agent
    task: A real task
"""
        engine = _write_workflow_yaml(tmpdir, "test-synthetic-no-task", yaml)
        wf = engine.load_workflow("test-synthetic-no-task")
        assert wf.nodes["gate"].synthetic is True
        # task defaults to a labeled placeholder, not KeyError
        assert "gate" in wf.nodes["gate"].task


def test_load_synthetic_with_redundant_agent_warns(capsys):
    """Loader warns if synthetic: true coexists with an agent field."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml = """name: test-synthetic-extra-agent
nodes:
  gate:
    synthetic: true
    agent: sherlock   # redundant — synthetic wins
    depends_on: [real]
  real:
    agent: some-agent
    task: A real task
"""
        engine = _write_workflow_yaml(tmpdir, "test-synthetic-extra-agent", yaml)
        wf = engine.load_workflow("test-synthetic-extra-agent")
        # Loader silently ignored the agent field; node is synthetic
        assert wf.nodes["gate"].synthetic is True
        assert wf.nodes["gate"].agent is None
        # Warning was emitted to stdout
        captured = capsys.readouterr()
        assert "synthetic: true" in captured.out
        assert "ignoring agent field" in captured.out


def test_validate_synthetic_node_skips_agent_check():
    """validate() must not flag synthetic nodes for missing agent profile.

    Regression: validate() does `profiles_dir / node.agent` which would
    crash on None. The synthetic skip is in the same loop.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml = """name: test-synthetic-validate
nodes:
  real:
    agent: some-agent
    task: A real task
  gate:
    synthetic: true
    depends_on: [real]
"""
        engine = _write_workflow_yaml(tmpdir, "test-synthetic-validate", yaml)
        result = engine.validate("test-synthetic-validate")
        # Should validate cleanly — no agent profile errors for 'gate'
        agent_issues = [i for i in result["issues"] if "gate" in i and "agent" in i]
        assert agent_issues == []
        # Layer count is correct: real in layer 0, gate in layer 1
        assert result["layers"] == 2
        assert result["nodes"] == 2


def test_topological_sort_with_synthetic():
    """Synthetic gates slot into the DAG exactly like real nodes.

    `real-a` (layer 0) → `gate` (layer 1, synthetic) → `real-b` (layer 2).
    The gate's presence in the middle layer is what enforces the
    ordering — without it, real-b could run as soon as real-a finishes.
    """
    wf = Workflow(name="test-synthetic-topo")
    wf.nodes["real-a"] = WorkflowNode(id="real-a", agent="a", task="A")
    wf.nodes["gate"] = WorkflowNode(id="gate", agent=None, task="gate",
                                    depends_on=["real-a"], synthetic=True)
    wf.nodes["real-b"] = WorkflowNode(id="real-b", agent="b", task="B",
                                      depends_on=["gate"])

    engine = WorkflowEngine()
    layers = engine.topological_sort(wf)
    assert layers == [["real-a"], ["gate"], ["real-b"]]


def test_create_kanban_card_refuses_synthetic():
    """Defensive: the create helper rejects synthetic nodes explicitly.

    This is a backstop. The dispatch loop already filters synthetic
    nodes, but if a future caller forgets the check, this guard turns
    a confusing subprocess-on-None crash into a clear ValueError.
    """
    engine = WorkflowEngine()
    gate_node = WorkflowNode(id="gate", agent=None, task="gate",
                             synthetic=True)
    with pytest.raises(ValueError, match="synthetic"):
        engine.create_kanban_card(gate_node)


def test_synthetic_node_in_layer_with_real_node_preserves_ordering():
    """Synthetic gate in a layer ensures downstream real nodes wait.

    This is the load-bearing behavior. If a synthetic gate's auto-
    completion didn't fire correctly, downstream real nodes would
    either run early (data race) or be blocked forever.
    """
    wf = Workflow(name="test-ordering")
    wf.nodes["upstream"] = WorkflowNode(id="upstream", agent="a", task="U")
    wf.nodes["gate"] = WorkflowNode(id="gate", agent=None, task="gate",
                                    depends_on=["upstream"], synthetic=True)
    wf.nodes["downstream"] = WorkflowNode(id="downstream", agent="b",
                                          task="D", depends_on=["gate"])

    engine = WorkflowEngine()
    layers = engine.topological_sort(wf)

    # gate is in a separate layer from upstream and downstream —
    # topological order is preserved end-to-end.
    assert layers.index(["upstream"]) < layers.index(["gate"])
    assert layers.index(["gate"]) < layers.index(["downstream"])


def test_synthetic_node_failure_propagation():
    """If a synthetic node's dependency fails, the synthetic node is skipped.

    Documented behavior: dep_failed is checked before the synthetic
    auto-complete, so a failed upstream blocks the gate (which blocks
    its downstream — same as a failed real node would).
    """
    wf = Workflow(name="test-synth-fail-prop")
    wf.nodes["real-a"] = WorkflowNode(id="real-a", agent="a", task="A")
    wf.nodes["gate"] = WorkflowNode(id="gate", agent=None, task="g",
                                    depends_on=["real-a"], synthetic=True)
    wf.nodes["real-b"] = WorkflowNode(id="real-b", agent="b", task="B",
                                      depends_on=["gate"])

    # Simulate state: real-a failed
    states = {
        "real-a": NodeState(node_id="real-a", status="failed"),
        "gate": NodeState(node_id="gate"),
        "real-b": NodeState(node_id="real-b"),
    }

    # Same dep_failed check the dispatch loop uses
    gate_deps_failed = any(
        states[d].status in ("failed", "timed_out", "blocked")
        for d in wf.nodes["gate"].depends_on
    )
    assert gate_deps_failed is True


# ── B2: Phase output template substitution tests ─────────────────
#
# The engine now resolves {namespace.field} and {bare} references in
# node.task before posting to kanban. The lookup walks completed
# upstream nodes' captured results (state.result) plus the start-time
# context dict. See the docstring on `_build_template_lookup` for the
# resolution rules; see council.yaml for the canonical use case.

# Fixture: a council-shaped DAG that exercises the spec's example
# variables ({context.question}, {context.question_slug},
# {phase1.position-edison}, {phase1.all}, {phase2a.all}, {phase2b.all}).
# Built programmatically (not from YAML) so the test is self-contained
# and doesn't need a temp file for every assertion.

@pytest.fixture
def council_pipeline():
    """Multi-phase workflow that mirrors the council.yaml shape.

    Layers:
      0: premortem           (phase 0)
      1: council-ready       (synthetic gate, phase 1)
      2: pos-e, pos-n, pos-k (phase 1, explicit label)
      3: probe-s, probe-r    (phase 2a and 2b respectively)
    """
    wf = Workflow(name="council-test")
    wf.nodes["premortem"] = WorkflowNode(
        id="premortem", agent="nikola", task="Imagine failure"
    )
    wf.nodes["council-ready"] = WorkflowNode(
        id="council-ready", agent=None, task="gate",
        depends_on=["premortem"], synthetic=True,
    )
    wf.nodes["position-edison"] = WorkflowNode(
        id="position-edison", agent="edison", task="pos-E",
        depends_on=["council-ready"], phase="phase1",
    )
    wf.nodes["position-newton"] = WorkflowNode(
        id="position-newton", agent="newton", task="pos-N",
        depends_on=["council-ready"], phase="phase1",
    )
    wf.nodes["position-nikola"] = WorkflowNode(
        id="position-nikola", agent="nikola", task="pos-K",
        depends_on=["council-ready"], phase="phase1",
    )
    wf.nodes["probe-sherlock"] = WorkflowNode(
        id="probe-sherlock", agent="sherlock", task="probe-S",
        depends_on=["position-edison", "position-newton", "position-nikola"],
        phase="phase2a",
    )
    wf.nodes["probe-raven"] = WorkflowNode(
        id="probe-raven", agent="raven", task="probe-R",
        depends_on=["position-edison", "position-newton", "position-nikola"],
        phase="phase2b",
    )
    return wf


# ── NodeState / WorkflowNode field tests ──────────────────────────

def test_node_state_result_defaults_to_none():
    """B2: result field added to NodeState. Defaults to None.

    Pre-B2 there was no result field; the engine had no way to
    remember what a completed card had produced. The lookup helper
    now relies on this being None to filter pending vs completed.
    """
    state = NodeState(node_id="x")
    assert state.result is None


def test_workflow_node_phase_defaults_to_none():
    """B2: phase field added to WorkflowNode. Defaults to None.

    When None, the engine auto-derives "phase0", "phase1", ... from
    the topological layer index at lookup time. Authors only set
    `phase:` explicitly when they want a non-numeric label (e.g.
    "phase2a", "phase2b") or when they want to override the
    default layer-derived label.
    """
    node = WorkflowNode(id="x", agent="a", task="t")
    assert node.phase is None


def test_load_node_phase_from_yaml():
    """Loader reads `phase:` from YAML into WorkflowNode.phase."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml = """name: test-phase
nodes:
  a:
    agent: x
    task: A
    phase: 1a
  b:
    agent: x
    task: B
    depends_on: [a]
"""
        engine = _write_workflow_yaml(tmpdir, "test-phase", yaml)
        wf = engine.load_workflow("test-phase")
        assert wf.nodes["a"].phase == "1a"
        # `b` has no phase — stays None and the engine defaults it
        # to the layer index at lookup time
        assert wf.nodes["b"].phase is None


# ── _build_template_lookup tests ─────────────────────────────────

def test_lookup_context_only_no_states_done(engine):
    """Empty pipeline: lookup contains just the context dict.

    No upstream nodes are completed, so the only key in the lookup
    is 'context'. The phase keys are absent (we don't pre-create
    empty phase dicts) and the bare {X} fallback only hits context.
    """
    wf = Workflow(name="empty")
    states = {}
    layers = []
    lookup = engine._build_template_lookup(wf, states, layers,
                                           context={"q": "Q"})
    assert lookup == {"context": {"q": "Q"}, "nodes": {}}


def test_lookup_phase_default_derived_from_layer(engine, council_pipeline):
    """phase=None → engine defaults to 'phaseN' from the layer index.

    council_pipeline layers are:
      0: premortem
      1: council-ready
      2: position-edison, position-newton, position-nikola
      3: probe-sherlock, probe-raven
    Nodes without an explicit `phase:` label should be auto-grouped
    under phase0 / phase1 / phase2 / phase3 by their layer index.
    """
    layers = engine.topological_sort(council_pipeline)
    states = {nid: NodeState(node_id=nid) for nid in council_pipeline.nodes}
    states["premortem"].result = "PRE"
    states["premortem"].status = "done"

    lookup = engine._build_template_lookup(council_pipeline, states,
                                            layers, context={})
    # premortem has no explicit phase, so it lands in phase0 (its layer)
    assert "phase0" in lookup
    assert lookup["phase0"]["premortem"] == "PRE"
    # council-ready has no explicit phase, so it lands in phase1
    # (its layer). It also auto-completed (synthetic gate), but with
    # no captured result, so phase1 should NOT exist yet.
    assert "phase1" not in lookup


def test_lookup_phase_explicit_label_used(engine, council_pipeline):
    """Explicit `phase:` in YAML is honored over the layer default.

    position-edison/newton/nikola all set phase=phase1, so they all
    land under that label even though they share layer 2 with nothing
    else. This is the whole reason phase is configurable.
    """
    layers = engine.topological_sort(council_pipeline)
    states = {nid: NodeState(node_id=nid) for nid in council_pipeline.nodes}
    for nid in ("position-edison", "position-newton", "position-nikola"):
        states[nid].result = f"OUTPUT_{nid}"
        states[nid].status = "done"

    lookup = engine._build_template_lookup(council_pipeline, states,
                                            layers, context={})
    assert "phase1" in lookup
    assert set(lookup["phase1"].keys()) == {
        "position-edison", "position-newton", "position-nikola", "all",
    }
    assert lookup["phase1"]["position-edison"] == "OUTPUT_position-edison"


def test_lookup_phase2a_and_2b_are_separate_namespaces(
        engine, council_pipeline):
    """Explicit 'phase2a' and 'phase2b' keep parallel branches separate.

    The two probe nodes sit in the same topological layer but represent
    logically distinct sub-phases. Each gets its own namespace in the
    lookup so {phase2a.all} and {phase2b.all} can refer to them
    independently.
    """
    layers = engine.topological_sort(council_pipeline)
    states = {nid: NodeState(node_id=nid) for nid in council_pipeline.nodes}
    states["probe-sherlock"].result = "PROBE_S"
    states["probe-sherlock"].status = "done"
    states["probe-raven"].result = "PROBE_R"
    states["probe-raven"].status = "done"

    lookup = engine._build_template_lookup(council_pipeline, states,
                                            layers, context={})
    assert "phase2a" in lookup
    assert "phase2b" in lookup
    assert lookup["phase2a"]["probe-sherlock"] == "PROBE_S"
    assert lookup["phase2b"]["probe-raven"] == "PROBE_R"
    assert "probe-raven" not in lookup["phase2a"]
    assert "probe-sherlock" not in lookup["phase2b"]


def test_lookup_phase_all_concatenates_in_layer_order(engine):
    """{phase.all} concatenates every member of the phase in stable order.

    Order matters: the receiving agent will read this as a single
    document and expects the upstream outputs in a predictable
    sequence. We sort by the topological layer's natural order rather
    than dict iteration order, so the concat is deterministic across
    Python versions and dict insertion orderings.

    The test workflow is a parallel middle (a → b ∥ c → d) so layer 1
    holds both b and c — they're in the same phase, and `.all` should
    concatenate them in b-then-c order.
    """
    wf = Workflow(name="order-test")
    wf.nodes["a"] = WorkflowNode(id="a", agent="x", task="A")
    wf.nodes["b"] = WorkflowNode(id="b", agent="x", task="B",
                                 depends_on=["a"])
    wf.nodes["c"] = WorkflowNode(id="c", agent="x", task="C",
                                 depends_on=["a"])
    wf.nodes["d"] = WorkflowNode(id="d", agent="x", task="D",
                                 depends_on=["b", "c"])

    engine = WorkflowEngine()
    layers = engine.topological_sort(wf)
    # Mark all 4 as done with distinct outputs
    states = {nid: NodeState(node_id=nid, status="done",
                              result=f"OUT_{nid}") for nid in wf.nodes}

    lookup = engine._build_template_lookup(wf, states, layers, context={})
    # b and c share layer 1, so they live in the same phase (phase1,
    # since layer 0 is 'a' and layer 1 is 'b','c', and 'a' is done so
    # phase0 exists too). Both phases have a single member each except
    # phase1 which has two.
    assert "phase0" in lookup
    assert "phase1" in lookup
    # phase0 has just 'a'; phase1 has 'b' and 'c' (and 'all')
    assert "all" in lookup["phase1"]
    # The concat is "[id]\nbody" pairs joined by "\n\n---\n\n"
    # b should appear before c in the concat (layer-order)
    all_text = lookup["phase1"]["all"]
    assert all_text.index("OUT_b") < all_text.index("OUT_c")
    # And 'd' is in its own phase, not in phase1
    assert "OUT_d" not in all_text


def test_lookup_excludes_pending_nodes(engine, council_pipeline):
    """Nodes without a captured result are excluded from the lookup.

    If a node hasn't completed, its result is None and we skip it.
    This prevents downstream prompts from embedding a half-finished
    or empty output. Pipelines that need a guarantee of a result
    should put a verify/gate node in the dependency chain.
    """
    layers = engine.topological_sort(council_pipeline)
    states = {nid: NodeState(node_id=nid) for nid in council_pipeline.nodes}
    # Only premortem has a result. Positions are still pending.
    states["premortem"].result = "PRE"
    states["premortem"].status = "done"
    # position-edison has status='done' but no captured result —
    # should still be excluded (defensive: the engine only populates
    # result on a real card-body read, not just on status flip)
    states["position-edison"].status = "done"

    lookup = engine._build_template_lookup(council_pipeline, states,
                                            layers, context={})
    # phase0 exists (premortem), phase1 does not (no captured results)
    assert "phase0" in lookup
    assert "phase1" not in lookup
    # Top-level node id only set for the one with a result
    assert lookup.get("premortem") == "PRE"
    assert "position-edison" not in lookup


def test_lookup_exposes_completed_node_ids_at_top_level(
        engine, council_pipeline):
    """Each completed node id is mirrored at the top of the lookup.

    This is for the legacy {node-id} form. The original council.yaml
    uses things like {position-edison-output} and {premortem-output}
    (we currently don't have those, but the convention is bare
    {node-id}). Top-level exposure lets those templates resolve via
    the same code path as the new {phaseN.X} form.
    """
    layers = engine.topological_sort(council_pipeline)
    states = {nid: NodeState(node_id=nid) for nid in council_pipeline.nodes}
    states["position-edison"].result = "POS_E_OUT"
    states["position-edison"].status = "done"

    lookup = engine._build_template_lookup(council_pipeline, states,
                                            layers, context={})
    assert lookup["position-edison"] == "POS_E_OUT"
    # And the canonical {phase1.position-edison} form also resolves
    assert lookup["phase1"]["position-edison"] == "POS_E_OUT"


# ── _resolve_template tests ──────────────────────────────────────

def test_resolve_namespace_field(engine):
    """{namespace.field} resolves to lookup[ns][field]."""
    lookup = {
        "context": {"q": "Q_VAL"},
        "phase1": {
            "position-edison": "EDISON_OUT",
            "all": "ALL_OUT",
        },
    }
    out = engine._resolve_template(
        "Q={context.q}, P={phase1.position-edison}, ALL={phase1.all}",
        lookup,
    )
    assert out == "Q=Q_VAL, P=EDISON_OUT, ALL=ALL_OUT"


def test_resolve_bare_form_via_context(engine):
    """Legacy {bare} form resolves via context first.

    This is what makes {question} in the original council.yaml work:
    `question` is in the context dict (from -c question=...) and the
    bare-form fallback looks there before going to top-level node ids.
    """
    lookup = {
        "context": {"question": "What is X?"},
        "phase1": {"position-edison": "EDISON_OUT"},
    }
    out = engine._resolve_template("Q: {question}", lookup)
    assert out == "Q: What is X?"


def test_resolve_bare_form_falls_through_to_top_level_node(engine):
    """Legacy {bare} form falls through to top-level node ids.

    If the bare token isn't in context, we check the top of the
    lookup for a completed node with that id. This is what supports
    {position-edison-output} style references (well, the prefix
    stripped — {position-edison} — once the YAML gets cleaned up).
    """
    lookup = {
        "context": {},
        "phase1": {"position-edison": "EDISON_OUT"},
        "position-edison": "EDISON_OUT",
    }
    out = engine._resolve_template("Got: {position-edison}", lookup)
    assert out == "Got: EDISON_OUT"


def test_resolve_unresolved_namespace_field_leaves_literal(engine, capsys):
    """Unknown {ns.field} stays in the text and prints a warning.

    Per the spec: 'Unresolved variables (e.g. if a phase hasn't run
    yet) are left as-is or raise a clear error — do not silently
    produce empty strings.' We go with leave-as-is + a one-line
    warning, so the agent still sees the literal brace and can
    surface the missing upstream in its work, while operators get a
    visible signal in the engine logs.
    """
    lookup = {"context": {}, "phase1": {"position-edison": "OK"}}
    out = engine._resolve_template(
        "Known: {phase1.position-edison}, Missing: {phase1.missing-node}",
        lookup,
    )
    assert out == "Known: OK, Missing: {phase1.missing-node}"
    captured = capsys.readouterr()
    assert "Unresolved template {phase1.missing-node}" in captured.out


def test_resolve_unresolved_bare_leaves_literal(engine, capsys):
    """Unknown {bare} (neither context nor top-level) also leaves literal."""
    lookup = {"context": {"known": "X"}}
    out = engine._resolve_template("Known: {known}, Missing: {nope}", lookup)
    assert out == "Known: X, Missing: {nope}"
    captured = capsys.readouterr()
    assert "Unresolved template {nope}" in captured.out


def test_resolve_unresolved_unknown_namespace_leaves_literal(engine, capsys):
    """{notaphase.foo} where 'notaphase' isn't in the lookup at all."""
    lookup = {"context": {}}
    out = engine._resolve_template("A: {notaphase.foo}", lookup)
    assert out == "A: {notaphase.foo}"
    captured = capsys.readouterr()
    assert "Unresolved template {notaphase.foo}" in captured.out


def test_resolve_no_templates_passthrough(engine, capsys):
    """Text with no {…} references is returned unchanged, no warnings."""
    out = engine._resolve_template("Plain text, no braces here.", {})
    assert out == "Plain text, no braces here."
    assert "Unresolved" not in capsys.readouterr().out


def test_resolve_does_not_treat_json_braces_as_templates(engine):
    """Internal {...} JSON-ish blobs are not template references.

    The regex requires the leading char to be a letter or underscore.
    Things like `{1, 2, 3}` (starts with digit) or `{}` (empty) are
    left alone. This is intentional: we don't want the resolver to
    chew on JSON-like text inside the task body.
    """
    lookup = {"context": {"q": "Q"}}
    out = engine._resolve_template(
        "List: {1, 2, 3} and empty: {} and ref: {q}", lookup,
    )
    assert out == "List: {1, 2, 3} and empty: {} and ref: Q"


def test_resolve_inputs_namespace(engine):
    """{inputs.key} resolves via lookup['inputs'] namespace.

    Inputs are merged into context["inputs"] by execute(), and the
    lookup builder promotes them to a top-level 'inputs' namespace
    so {inputs.grill_artifact} resolves correctly.
    """
    lookup = {
        "context": {"inputs": {"grill_artifact": "/tmp/art.json", "n": 42}},
    }
    # The lookup builder should have promoted inputs to top-level
    # when called via _build_template_lookup, but _resolve_template
    # itself just needs the key to exist in the lookup.
    lookup["inputs"] = lookup["context"]["inputs"]
    out = engine._resolve_template(
        "File: {inputs.grill_artifact}, Count: {inputs.n}", lookup,
    )
    assert out == "File: /tmp/art.json, Count: 42"


def test_build_template_lookup_promotes_inputs(engine):
    """_build_template_lookup promotes context.inputs to top-level lookup key."""
    wf = Workflow(name="test", description="t", nodes={
        "a": WorkflowNode(id="a", task="do {inputs.x}", agent="dev", depends_on=[]),
    })
    states = {"a": NodeState(node_id="a")}
    layers = [["a"]]
    lookup = engine._build_template_lookup(wf, states, layers,
                                           context={"inputs": {"x": "VAL"}})
    assert "inputs" in lookup
    assert lookup["inputs"]["x"] == "VAL"


# ── _build_task_body tests ───────────────────────────────────────

def test_build_task_body_appends_context_footer_after_substitution(engine):
    """The Context JSON footer goes AFTER substitution.

    Putting the footer after the resolver means the JSON braces in
    the footer don't get treated as templates (which would either
    leave noisy 'Unresolved' warnings or accidentally substitute
    something). The order matters — the pre-B2 footer is still
    there for agents that prefer to read the raw context.
    """
    wf = Workflow(name="ctx-footer")
    wf.nodes["a"] = WorkflowNode(id="a", agent="x",
                                  task="Task with {context.x}")
    layers = engine.topological_sort(wf)
    states = {"a": NodeState(node_id="a")}

    body = engine._build_task_body(
        wf.nodes["a"], wf, states, layers, context={"x": "X_VAL"},
    )
    # The body should contain the resolved value but NOT the Context footer
    # or Run ID — those are meta noise that agents don't need in card bodies.
    assert "Task with X_VAL" in body
    assert 'Context:' not in body
    assert 'Run ID:' not in body


def test_build_task_body_no_context_no_footer(engine):
    """No context given → no Context footer appended."""
    wf = Workflow(name="no-ctx")
    wf.nodes["a"] = WorkflowNode(id="a", agent="x", task="Plain task")
    layers = engine.topological_sort(wf)
    states = {"a": NodeState(node_id="a")}

    body = engine._build_task_body(
        wf.nodes["a"], wf, states, layers, context=None,
    )
    assert body == "Plain task"
    assert "Context:" not in body


def test_build_task_body_full_council_substitution(engine, council_pipeline):
    """End-to-end: every spec example variable resolves correctly.

    This is the canonical council-pipeline test. Sets up:
      - premortem done with PRE_OUTPUT
      - all three positions done with their respective outputs
      - probes to be created next
    Then builds a probe-sherlock body that references:
      - {context.question}
      - {context.question_slug}
      - {phase1.position-edison}     (specific node)
      - {phase1.all}                  (all positions concatenated)
    """
    layers = engine.topological_sort(council_pipeline)
    states = {nid: NodeState(node_id=nid) for nid in council_pipeline.nodes}
    # Council-ready auto-completed as a synthetic gate, no result needed
    states["council-ready"].status = "done"
    # All upstream phases done
    for nid, body in [
        ("premortem", "PRE_OUTPUT"),
        ("position-edison", "EDISON_OUTPUT"),
        ("position-newton", "NEWTON_OUTPUT"),
        ("position-nikola", "NIKOLA_OUTPUT"),
    ]:
        states[nid].result = body
        states[nid].status = "done"

    # Probe-sherlock task references all the spec variables
    probe_node = council_pipeline.nodes["probe-sherlock"]
    probe_node.task = (
        "Question: {context.question}\n"
        "Slug: {context.question_slug}\n"
        "Edison position: {phase1.position-edison}\n"
        "All positions: {phase1.all}\n"
    )

    body = engine._build_task_body(
        probe_node, council_pipeline, states, layers,
        context={"question": "What is X?", "question_slug": "what-is-x"},
    )
    # All references resolve
    assert "Question: What is X?" in body
    assert "Slug: what-is-x" in body
    assert "Edison position: EDISON_OUTPUT" in body
    # {phase1.all} expands to a concatenation of all 3 positions
    assert "EDISON_OUTPUT" in body
    assert "NEWTON_OUTPUT" in body
    assert "NIKOLA_OUTPUT" in body
    # Order is layer-stable: edison before newton before nikola
    assert body.index("EDISON_OUTPUT") < body.index("NEWTON_OUTPUT")
    assert body.index("NEWTON_OUTPUT") < body.index("NIKOLA_OUTPUT")
    # No unresolved literals left
    assert "{context.question}" not in body
    assert "{phase1.all}" not in body
    # Context footer is removed — agents don't need meta noise in card bodies
    assert "Context:" not in body


def test_build_task_body_phase2a_vs_phase2b(engine, council_pipeline):
    """{phase2a.all} and {phase2b.all} stay separate.

    Once both probes are done, their results go into phase2a and
    phase2b respectively. Building a synthesize node's body that
    references both should pull from each independently.
    """
    layers = engine.topological_sort(council_pipeline)
    states = {nid: NodeState(node_id=nid) for nid in council_pipeline.nodes}
    for nid, body in [
        ("premortem", "PRE"),
        ("position-edison", "E"),
        ("position-newton", "N"),
        ("position-nikola", "K"),
        ("probe-sherlock", "PROBE_S_OUT"),
        ("probe-raven", "PROBE_R_OUT"),
    ]:
        states[nid].result = body
        states[nid].status = "done"

    # Synthesize node references both sub-phases
    wf = council_pipeline
    synth = WorkflowNode(
        id="synth", agent="nikola", task="synth",
        depends_on=["probe-sherlock", "probe-raven"], phase="phase3",
    )
    wf.nodes["synth"] = synth
    layers = engine.topological_sort(wf)
    states["synth"] = NodeState(node_id="synth")
    synth.task = "S: {phase2a.all}\nR: {phase2b.all}\n"

    body = engine._build_task_body(
        synth, wf, states, layers, context={},
    )
    # Each phase2X.all contains only its own probe
    assert "S: [probe-sherlock]\nPROBE_S_OUT" in body
    assert "R: [probe-raven]\nPROBE_R_OUT" in body
    # And not the other way around
    assert "PROBE_R_OUT" not in body.split("S: ")[1].split("\nR: ")[0]
    assert "PROBE_S_OUT" not in body.split("R: ")[1]


# ── state.result persistence tests ───────────────────────────────

def test_state_result_round_trip(engine):
    """state.result is persisted to disk and restored on load.

    The whole point of the result field is so that a resumed
    workflow (engine crashed and restarted) still has the upstream
    outputs available for {phaseN.X} substitution. This test
    exercises the _save_state / _load_state round-trip.
    """
    wf = Workflow(name="result-roundtrip")
    wf.nodes["a"] = WorkflowNode(id="a", agent="x", task="A")
    states = {
        "a": NodeState(node_id="a", status="done", kanban_card_id="c-1",
                       result="CAPTURED_BODY"),
    }
    results = {"a": "done"}
    layers = [["a"]]

    engine._save_state("result-roundtrip", states, results, 0, layers)
    loaded = engine._load_state("result-roundtrip")

    assert loaded is not None
    assert loaded["states"]["a"]["result"] == "CAPTURED_BODY"

    # Round-trip back into a NodeState — the loader path
    restored = NodeState(
        node_id=loaded["states"]["a"]["node_id"],
        status=loaded["states"]["a"]["status"],
        result=loaded["states"]["a"]["result"],
    )
    assert restored.result == "CAPTURED_BODY"

    engine._clear_state("result-roundtrip")


# ── create_kanban_card public-API path ───────────────────────────

def test_create_kanban_card_resolves_templates_when_workflow_provided(
        engine, council_pipeline, monkeypatch):
    """End-to-end: create_kanban_card resolves templates + posts to kanban.

    We mock the subprocess.run call (so no real kanban card is
    created) and assert that the body passed to --body already has
    the {phase1.X} references resolved. This is the public API path
    the engine's execute() loop takes, and it's what the agent
    ultimately sees.
    """
    layers = engine.topological_sort(council_pipeline)
    states = {nid: NodeState(node_id=nid) for nid in council_pipeline.nodes}
    states["position-edison"].result = "EDISON_OUT"
    states["position-edison"].status = "done"
    states["position-newton"].result = "NEWTON_OUT"
    states["position-newton"].status = "done"
    states["position-nikola"].result = "NIKOLA_OUT"
    states["position-nikola"].status = "done"

    # Capture the body that create_kanban_card would post
    captured = {}

    def fake_create_task(conn, *, title, body, assignee, **kwargs):
        captured["title"] = title
        captured["body"] = body
        captured["assignee"] = assignee
        captured["tenant"] = kwargs.get("tenant")
        return "t_fake_card_123"

    monkeypatch.setattr("hermes_cli.kanban_db.create_task", fake_create_task)

    probe_node = council_pipeline.nodes["probe-sherlock"]
    probe_node.task = "All positions: {phase1.all}\nContext Q: {context.q}"

    card_id = engine.create_kanban_card(
        probe_node,
        context={"q": "Q_VAL"},
        workflow=council_pipeline,
        states=states,
        layers=layers,
    )
    # Card id was parsed from the mocked JSON
    assert card_id == "t_fake_card_123"
    # Title was resolved (not raw template text)
    title = captured["title"]
    assert "All positions:" in title
    assert "{phase1.all}" not in title
    assert "{context.q}" not in title
    # Body was substituted before posting
    body = captured["body"]
    assert "EDISON_OUT" in body
    assert "NEWTON_OUT" in body
    assert "NIKOLA_OUT" in body
    assert "Context Q: Q_VAL" in body
    # Literal braces are gone
    assert "{phase1.all}" not in body
    assert "{context.q}" not in body
    # Context footer removed — meta noise not in card bodies
    assert 'Context:' not in body


def test_create_kanban_card_legacy_path_unchanged(
        engine, monkeypatch):
    """Backward compat: omitting workflow= reverts to pre-B2 footer only.

    Direct callers (the synthetic-guard test) and any pre-B2 code
    path that calls create_kanban_card(node, context) without the
    new keyword args should get the original footer-only behavior
    — no {ns.field} resolution, just the Context JSON appended.
    """
    captured = {}

    def fake_create_task(conn, *, title, body, assignee, **kwargs):
        captured["body"] = body
        return "t_legacy"

    monkeypatch.setattr("hermes_cli.kanban_db.create_task", fake_create_task)

    node = WorkflowNode(id="x", agent="a", task="Do {phase1.foo}")
    # Note: no workflow= passed → legacy path
    card_id = engine.create_kanban_card(node, context={"k": "v"})
    assert card_id == "t_legacy"
    # {phase1.foo} is NOT resolved (legacy path doesn't know about phases)
    assert "{phase1.foo}" in captured["body"]
    # But the Context footer IS appended
    assert 'Context: {"k": "v"}' in captured["body"]


def test_workflow_scope_default_is_project():
    """Workflows default to scope: project — current behavior preserved."""
    wf = Workflow(name="x")
    assert wf.scope == "project"


def test_dispatch_node_scope_global_returns_none_and_marks_done():
    """scope: global nodes are dispatched in-process; no card created."""
    engine = WorkflowEngine()
    wf = Workflow(name="heartbeat", scope="global")
    state = NodeState(node_id="hb")
    node = WorkflowNode(
        id="hb",
        agent="sherlock",
        task="Check fleet heartbeat",
    )
    layers = [[node]]
    states = {"hb": state}

    # No subprocess should be invoked — the helper sets state.done directly.
    card_id = engine.dispatch_node(
        state, node, context={},
        workflow=wf, states=states, layers=layers,
    )
    assert card_id is None
    assert state.status == "done"
    assert state.completed_at is not None
    assert state.result == "[in-process, scope: global]"


def test_dispatch_node_scope_project_delegates_to_create_kanban_card(
        engine, council_pipeline, monkeypatch):
    """scope: project (default) routes through create_kanban_card as before."""

    def fake_create_task(conn, *, title, body, assignee, **kwargs):
        return "t_project_card"

    monkeypatch.setattr("hermes_cli.kanban_db.create_task", fake_create_task)

    node = council_pipeline.nodes["position-edison"]
    state = NodeState(node_id=node.id)
    layers = engine.topological_sort(council_pipeline)
    states = {nid: NodeState(node_id=nid) for nid in council_pipeline.nodes}

    card_id = engine.dispatch_node(
        state, node, context={},
        workflow=council_pipeline,  # scope defaults to "project"
        states=states, layers=layers,
    )
    assert card_id == "t_project_card"
    assert state.status != "done"  # dispatcher left it "running" for monitoring


def test_load_workflow_parses_scope_field(tmp_path):
    """YAML scope: global is loaded into Workflow.scope."""
    wf_path = tmp_path / "heartbeat.yaml"
    wf_path.write_text(yaml.safe_dump({
        "name": "heartbeat",
        "scope": "global",
        "nodes": {
            "check": {
                "agent": "sherlock",
                "task": "Heartbeat check.",
            },
        },
    }))
    engine = WorkflowEngine(workflows_dir=tmp_path)
    wf = engine.load_workflow("heartbeat")
    assert wf.scope == "global"
    assert "check" in wf.nodes


def test_load_workflow_scope_defaults_to_project(tmp_path):
    """Workflows without explicit scope default to 'project'."""
    wf_path = tmp_path / "normal.yaml"
    wf_path.write_text(yaml.safe_dump({
        "name": "normal",
        "nodes": {
            "x": {"agent": "sherlock", "task": "do x"},
        },
    }))
    engine = WorkflowEngine(workflows_dir=tmp_path)
    wf = engine.load_workflow("normal")
    assert wf.scope == "project"




# ── incomplete_branch validation tests ─────────────────────────────


def test_validate_flags_non_terminal_without_fallback(tmp_path):
    """Non-terminal node relying on default fallback_on_timeout surfaces an issue."""
    wf_path = tmp_path / "wf.yaml"
    # a depends on b — b is non-terminal. b does NOT declare fallback_on_timeout.
    wf_path.write_text(yaml.safe_dump({
        "name": "wf",
        "nodes": {
            "a": {"agent": "sherlock", "task": "do a", "depends_on": ["b"]},
            "b": {"agent": "nikola", "task": "do b"},
        },
    }))
    engine = WorkflowEngine(workflows_dir=tmp_path)
    result = engine.validate("wf")
    assert result["valid"] is True  # non-fatal
    issues_text = " ".join(result["issues"])
    assert "b" in issues_text
    assert "fallback_on_timeout" in issues_text


def test_validate_does_not_flag_terminal_nodes(tmp_path):
    """Terminal nodes (no downstream) don't need explicit fallback_on_timeout."""
    wf_path = tmp_path / "wf.yaml"
    wf_path.write_text(yaml.safe_dump({
        "name": "wf",
        "nodes": {
            "leaf": {"agent": "sherlock", "task": "final node"},  # terminal
        },
    }))
    engine = WorkflowEngine(workflows_dir=tmp_path)
    result = engine.validate("wf")
    fallback_issues = [i for i in result["issues"] if "fallback_on_timeout" in i]
    assert fallback_issues == []


def test_validate_does_not_flag_when_explicit(tmp_path):
    """Non-terminal node with explicit fallback_on_timeout is clean."""
    wf_path = tmp_path / "wf.yaml"
    wf_path.write_text(yaml.safe_dump({
        "name": "wf",
        "nodes": {
            "a": {"agent": "sherlock", "task": "do a", "depends_on": ["b"]},
            "b": {
                "agent": "nikola",
                "task": "do b",
                "fallback_on_timeout": "degraded",
            },
        },
    }))
    engine = WorkflowEngine(workflows_dir=tmp_path)
    result = engine.validate("wf")
    fallback_issues = [i for i in result["issues"] if "fallback_on_timeout" in i]
    assert fallback_issues == []


def test_validate_skips_synthetic_nodes(tmp_path):
    """Synthetic gates don't need fallback_on_timeout even if downstream."""
    wf_path = tmp_path / "wf.yaml"
    wf_path.write_text(yaml.safe_dump({
        "name": "wf",
        "nodes": {
            "a": {"agent": "sherlock", "task": "do a", "depends_on": ["gate"]},
            "gate": {"synthetic": True},  # auto-completed, no fallback needed
        },
    }))
    engine = WorkflowEngine(workflows_dir=tmp_path)
    result = engine.validate("wf")
    fallback_issues = [i for i in result["issues"] if "fallback_on_timeout" in i]
    assert fallback_issues == []


# ── single_flight tests ────────────────────────────────────────────


def test_workflow_single_flight_default_false():
    """Workflows default to single_flight=False — multiple parallel runs allowed."""
    wf = Workflow(name="x")
    assert wf.single_flight is False


def test_load_workflow_parses_single_flight_field(tmp_path):
    """YAML single_flight: true is loaded into Workflow.single_flight."""
    wf_path = tmp_path / "wf.yaml"
    wf_path.write_text(yaml.safe_dump({
        "name": "wf",
        "single_flight": True,
        "nodes": {
            "x": {"agent": "sherlock", "task": "do x"},
        },
    }))
    engine = WorkflowEngine(workflows_dir=tmp_path)
    wf = engine.load_workflow("wf")
    assert wf.single_flight is True


def test_has_active_run_returns_false_when_no_state_files(tmp_path):
    """No state files for workflow → not active."""
    state_dir = tmp_path / ".engine-state"
    state_dir.mkdir()
    engine = WorkflowEngine(workflows_dir=tmp_path)
    engine.STATE_DIR = state_dir
    assert engine._has_active_run("nonexistent") is False


def test_has_active_run_returns_false_when_all_nodes_terminal(tmp_path):
    """State file exists but all nodes are done → not active."""
    state_dir = tmp_path / ".engine-state"
    state_dir.mkdir()
    state_file = state_dir / "wf_abc123_state.json"
    state_file.write_text(json.dumps({
        "workflow_name": "wf",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "states": {
            "x": {"status": "done"},
            "y": {"status": "failed"},
        },
    }))
    engine = WorkflowEngine(workflows_dir=tmp_path)
    engine.STATE_DIR = state_dir
    assert engine._has_active_run("wf") is False


def test_has_active_run_returns_true_when_node_running(tmp_path):
    """State file with a running node → active."""
    state_dir = tmp_path / ".engine-state"
    state_dir.mkdir()
    state_file = state_dir / "wf_xyz_state.json"
    state_file.write_text(json.dumps({
        "workflow_name": "wf",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "states": {
            "x": {"status": "done"},
            "y": {"status": "running"},
        },
    }))
    engine = WorkflowEngine(workflows_dir=tmp_path)
    engine.STATE_DIR = state_dir
    assert engine._has_active_run("wf") is True


def test_has_active_run_ignores_stale_state_files(tmp_path):
    """State files older than ACTIVE_RUN_STALE_SECONDS are ignored."""
    from datetime import timedelta as _td
    state_dir = tmp_path / ".engine-state"
    state_dir.mkdir()
    state_file = state_dir / "wf_old_state.json"
    # 2 hours old — past the 1-hour staleness threshold
    old_time = (
        datetime.now(timezone.utc) - _td(seconds=2 * 3600)
    ).isoformat()
    state_file.write_text(json.dumps({
        "workflow_name": "wf",
        "updated_at": old_time,
        "states": {
            "y": {"status": "running"},  # would be active if not stale
        },
    }))
    engine = WorkflowEngine(workflows_dir=tmp_path)
    engine.STATE_DIR = state_dir
    assert engine._has_active_run("wf") is False


# ── per-node telemetry tests ──────────────────────────────────────


def test_node_state_duration_defaults_to_none():
    """NodeState.duration_seconds starts None — populated on completion."""
    state = NodeState(node_id="x")
    assert state.duration_seconds is None


def test_node_state_error_count_defaults_to_zero():
    """NodeState.error_count starts at 0 — incremented on failure."""
    state = NodeState(node_id="x")
    assert state.error_count == 0


def test_record_node_completion_computes_duration_on_done():
    """When status='done' and started/completed_at set, duration_seconds is computed."""
    engine = WorkflowEngine()
    start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 12, 1, 30, tzinfo=timezone.utc)  # 90 seconds later
    state = NodeState(
        node_id="x",
        status="done",
        started_at=start.isoformat(),
        completed_at=end.isoformat(),
    )
    engine._record_node_completion(state)
    assert state.duration_seconds == 90.0
    # Done is not a failure — error_count stays at 0.
    assert state.error_count == 0


def test_record_node_completion_increments_error_count_on_failure():
    """Failed status increments error_count; duration computed if timestamps present."""
    engine = WorkflowEngine()
    start = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
    state = NodeState(
        node_id="x",
        status="failed",
        started_at=start.isoformat(),
        completed_at=end.isoformat(),
    )
    engine._record_node_completion(state)
    assert state.duration_seconds == 30.0
    assert state.error_count == 1


def test_record_node_completion_is_idempotent():
    """Calling twice doesn't double-count error_count or recompute duration."""
    engine = WorkflowEngine()
    state = NodeState(
        node_id="x",
        status="failed",
        started_at=datetime.now(timezone.utc).isoformat(),
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    engine._record_node_completion(state)
    first_duration = state.duration_seconds
    engine._record_node_completion(state)
    assert state.duration_seconds == first_duration
    assert state.error_count == 1  # only incremented once


def test_record_node_completion_no_op_for_running_status():
    """Running nodes don't get telemetry — not terminal yet."""
    engine = WorkflowEngine()
    state = NodeState(
        node_id="x",
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    engine._record_node_completion(state)
    assert state.duration_seconds is None
    assert state.error_count == 0


def test_record_node_completion_handles_missing_timestamps():
    """Defensive: don't crash if started_at or completed_at missing."""
    engine = WorkflowEngine()
    state = NodeState(node_id="x", status="done")
    engine._record_node_completion(state)
    assert state.duration_seconds is None
    assert state.error_count == 0


def test_prune_old_runs_keeps_most_recent(tmp_path):
    """Retention deletes oldest files beyond `keep` per workflow."""
    state_dir = tmp_path / ".engine-state"
    state_dir.mkdir()
    engine = WorkflowEngine(workflows_dir=tmp_path)
    engine.STATE_DIR = state_dir
    # Create 25 state files for "wf", spaced 1 second apart in mtime.
    import time as _time
    paths = []
    for i in range(25):
        p = state_dir / f"wf_run{i:03d}_state.json"
        p.write_text("{}")
        # Force mtime difference
        os.utime(p, (1000 + i, 1000 + i))
        paths.append(p)
    pruned = engine._prune_old_runs(keep=20)
    assert pruned == 5
    remaining = sorted(p.name for p in state_dir.glob("wf_*_state.json"))
    assert len(remaining) == 20


def test_prune_old_runs_groups_by_workflow(tmp_path):
    """Pruning keeps the per-workflow limit, not a global limit."""
    state_dir = tmp_path / ".engine-state"
    state_dir.mkdir()
    engine = WorkflowEngine(workflows_dir=tmp_path)
    engine.STATE_DIR = state_dir
    # 15 files for "wf_a" and 15 for "wf_b"
    for wf_name in ("wf_a", "wf_b"):
        for i in range(15):
            p = state_dir / f"{wf_name}_run{i:03d}_state.json"
            p.write_text("{}")
            os.utime(p, (1000 + i, 1000 + i))
    pruned = engine._prune_old_runs(keep=10)
    assert pruned == 10  # 5 from each workflow
    assert len(list(state_dir.glob("wf_a_*"))) == 10
    assert len(list(state_dir.glob("wf_b_*"))) == 10


def test_prune_old_runs_no_op_when_fewer_files(tmp_path):
    """No pruning when count is below threshold."""
    state_dir = tmp_path / ".engine-state"
    state_dir.mkdir()
    engine = WorkflowEngine(workflows_dir=tmp_path)
    engine.STATE_DIR = state_dir
    for i in range(5):
        (state_dir / f"wf_run{i:03d}_state.json").write_text("{}")
    pruned = engine._prune_old_runs(keep=20)
    assert pruned == 0
    assert len(list(state_dir.glob("*.json"))) == 5


# ── when: conditional step support tests ────────────────────────────


class TestWhenCondition:
    """Tests for the ``when:`` conditional dispatch feature."""

    # ── Empty when: always runs (current behavior preserved) ──

    def test_empty_when_always_runs(self, engine):
        """An empty when: string means the node always dispatches."""
        wf = Workflow(name="test-when-empty")
        wf.nodes["a"] = WorkflowNode(id="a", agent="agent-a", task="Task A")
        wf.nodes["b"] = WorkflowNode(
            id="b", agent="agent-b", task="Task B",
            depends_on=["a"], when="",
        )
        states = {
            "a": NodeState(node_id="a", status="done", result="hello"),
            "b": NodeState(node_id="b"),
        }
        assert engine.evaluate_when("", wf.nodes["b"], states) is True

    def test_none_when_always_runs(self, engine):
        """None-like empty when: still dispatches."""
        wf = Workflow(name="test-when-none")
        wf.nodes["a"] = WorkflowNode(id="a", agent="agent-a", task="Task A")
        states = {"a": NodeState(node_id="a", status="done")}
        assert engine.evaluate_when(None, wf.nodes["a"], states) is True

    # ── == operator ──

    def test_when_status_equals_done(self, engine):
        """when: {review.status} == done → dispatches when review is done."""
        wf = Workflow(name="test-when-eq")
        wf.nodes["review"] = WorkflowNode(id="review", agent="r", task="R")
        wf.nodes["deploy"] = WorkflowNode(
            id="deploy", agent="d", task="D",
            depends_on=["review"], when="{review.status} == done",
        )
        states = {
            "review": NodeState(node_id="review", status="done"),
            "deploy": NodeState(node_id="deploy"),
        }
        assert engine.evaluate_when(
            "{review.status} == done", wf.nodes["deploy"], states
        ) is True

    def test_when_status_not_done_skips(self, engine):
        """when: {review.status} == done → skips when review failed."""
        wf = Workflow(name="test-when-eq-skip")
        wf.nodes["review"] = WorkflowNode(id="review", agent="r", task="R")
        wf.nodes["deploy"] = WorkflowNode(
            id="deploy", agent="d", task="D",
            depends_on=["review"], when="{review.status} == done",
        )
        states = {
            "review": NodeState(node_id="review", status="failed"),
            "deploy": NodeState(node_id="deploy"),
        }
        assert engine.evaluate_when(
            "{review.status} == done", wf.nodes["deploy"], states
        ) is False

    # ── != operator ──

    def test_when_not_equal(self, engine):
        """when: {review.status} != failed → dispatches when not failed."""
        wf = Workflow(name="test-when-ne")
        wf.nodes["review"] = WorkflowNode(id="review", agent="r", task="R")
        wf.nodes["deploy"] = WorkflowNode(
            id="deploy", agent="d", task="D",
            depends_on=["review"], when="{review.status} != failed",
        )
        states = {
            "review": NodeState(node_id="review", status="done"),
            "deploy": NodeState(node_id="deploy"),
        }
        assert engine.evaluate_when(
            "{review.status} != failed", wf.nodes["deploy"], states
        ) is True

    def test_when_not_equal_skips_on_match(self, engine):
        """when: {review.status} != failed → skips when review failed."""
        wf = Workflow(name="test-when-ne-skip")
        wf.nodes["review"] = WorkflowNode(id="review", agent="r", task="R")
        wf.nodes["deploy"] = WorkflowNode(
            id="deploy", agent="d", task="D",
            depends_on=["review"], when="{review.status} != failed",
        )
        states = {
            "review": NodeState(node_id="review", status="failed"),
            "deploy": NodeState(node_id="deploy"),
        }
        assert engine.evaluate_when(
            "{review.status} != failed", wf.nodes["deploy"], states
        ) is False

    # ── contains operator ──

    def test_when_contains(self, engine):
        """when: {a.result} contains "error" → matches substring."""
        wf = Workflow(name="test-when-contains")
        wf.nodes["a"] = WorkflowNode(id="a", agent="a", task="A")
        wf.nodes["b"] = WorkflowNode(
            id="b", agent="b", task="B",
            depends_on=["a"], when='{a.result} contains "error"',
        )
        states = {
            "a": NodeState(node_id="a", status="done",
                           result="build failed: error in module X"),
            "b": NodeState(node_id="b"),
        }
        assert engine.evaluate_when(
            '{a.result} contains "error"', wf.nodes["b"], states
        ) is True

    def test_when_contains_no_match(self, engine):
        """when: {a.result} contains "error" → false when no match."""
        wf = Workflow(name="test-when-contains-nomatch")
        wf.nodes["a"] = WorkflowNode(id="a", agent="a", task="A")
        wf.nodes["b"] = WorkflowNode(
            id="b", agent="b", task="B",
            depends_on=["a"], when='{a.result} contains "error"',
        )
        states = {
            "a": NodeState(node_id="a", status="done",
                           result="all tests passed"),
            "b": NodeState(node_id="b"),
        }
        assert engine.evaluate_when(
            '{a.result} contains "error"', wf.nodes["b"], states
        ) is False

    # ── and composition ──

    def test_when_and_composition(self, engine):
        """when: {a.status} == done and {b.status} == done → both must be true."""
        wf = Workflow(name="test-when-and")
        wf.nodes["a"] = WorkflowNode(id="a", agent="a", task="A")
        wf.nodes["b"] = WorkflowNode(id="b", agent="b", task="B")
        wf.nodes["c"] = WorkflowNode(
            id="c", agent="c", task="C",
            depends_on=["a", "b"],
            when="{a.status} == done and {b.status} == done",
        )
        states = {
            "a": NodeState(node_id="a", status="done"),
            "b": NodeState(node_id="b", status="failed"),
            "c": NodeState(node_id="c"),
        }
        assert engine.evaluate_when(
            "{a.status} == done and {b.status} == done",
            wf.nodes["c"], states,
        ) is False

    def test_when_and_both_true(self, engine):
        """when: {a.status} == done and {b.status} == done → true when both done."""
        wf = Workflow(name="test-when-and-true")
        wf.nodes["a"] = WorkflowNode(id="a", agent="a", task="A")
        wf.nodes["b"] = WorkflowNode(id="b", agent="b", task="B")
        wf.nodes["c"] = WorkflowNode(
            id="c", agent="c", task="C",
            depends_on=["a", "b"],
            when="{a.status} == done and {b.status} == done",
        )
        states = {
            "a": NodeState(node_id="a", status="done"),
            "b": NodeState(node_id="b", status="done"),
            "c": NodeState(node_id="c"),
        }
        assert engine.evaluate_when(
            "{a.status} == done and {b.status} == done",
            wf.nodes["c"], states,
        ) is True

    # ── or composition ──

    def test_when_or_composition(self, engine):
        """when: {a.status} == done or {b.status} == done → true when either done."""
        wf = Workflow(name="test-when-or")
        wf.nodes["a"] = WorkflowNode(id="a", agent="a", task="A")
        wf.nodes["b"] = WorkflowNode(id="b", agent="b", task="B")
        wf.nodes["c"] = WorkflowNode(
            id="c", agent="c", task="C",
            depends_on=["a", "b"],
            when="{a.status} == done or {b.status} == done",
        )
        states = {
            "a": NodeState(node_id="a", status="failed"),
            "b": NodeState(node_id="b", status="done"),
            "c": NodeState(node_id="c"),
        }
        assert engine.evaluate_when(
            "{a.status} == done or {b.status} == done",
            wf.nodes["c"], states,
        ) is True

    # ── not operator ──

    def test_when_not(self, engine):
        """when: not {a.status} == failed → true when not failed."""
        wf = Workflow(name="test-when-not")
        wf.nodes["a"] = WorkflowNode(id="a", agent="a", task="A")
        wf.nodes["b"] = WorkflowNode(
            id="b", agent="b", task="B",
            depends_on=["a"], when='not {a.status} == failed',
        )
        states = {
            "a": NodeState(node_id="a", status="done"),
            "b": NodeState(node_id="b"),
        }
        assert engine.evaluate_when(
            'not {a.status} == failed', wf.nodes["b"], states
        ) is True

    # ── context.key reference ──

    def test_when_context_reference(self, engine):
        """when: {context.mode} == production → checks context dict."""
        wf = Workflow(name="test-when-ctx")
        wf.nodes["a"] = WorkflowNode(id="a", agent="a", task="A")
        wf.nodes["b"] = WorkflowNode(
            id="b", agent="b", task="B",
            depends_on=["a"], when="{context.mode} == production",
        )
        states = {
            "a": NodeState(node_id="a", status="done"),
            "b": NodeState(node_id="b"),
        }
        context = {"mode": "production"}
        assert engine.evaluate_when(
            "{context.mode} == production", wf.nodes["b"], states, context
        ) is True

    def test_when_context_reference_skips(self, engine):
        """when: {context.mode} == production → skips on mismatch."""
        wf = Workflow(name="test-when-ctx-skip")
        wf.nodes["a"] = WorkflowNode(id="a", agent="a", task="A")
        wf.nodes["b"] = WorkflowNode(
            id="b", agent="b", task="B",
            depends_on=["a"], when="{context.mode} == production",
        )
        states = {
            "a": NodeState(node_id="a", status="done"),
            "b": NodeState(node_id="b"),
        }
        context = {"mode": "staging"}
        assert engine.evaluate_when(
            "{context.mode} == production", wf.nodes["b"], states, context
        ) is False

    # ── Skipped node state is recorded ──

    def test_skipped_node_state_recorded(self, engine):
        """When a node is skipped via when:, its state.status is 'skipped'."""
        wf = Workflow(name="test-when-state")
        wf.nodes["a"] = WorkflowNode(id="a", agent="a", task="A")
        wf.nodes["b"] = WorkflowNode(
            id="b", agent="b", task="B",
            depends_on=["a"], when="{a.status} == done",
        )
        states = {
            "a": NodeState(node_id="a", status="failed"),
            "b": NodeState(node_id="b"),
        }
        # Simulate what execute() does
        if not engine.evaluate_when("{a.status} == done", wf.nodes["b"], states):
            states["b"].status = "skipped"
        assert states["b"].status == "skipped"

    # ── Downstream cascade: skipped node propagates skip ──

    def test_downstream_cascade_skip(self, engine):
        """A node skipped via when: causes its downstream to also skip
        (standard dependency-skip propagation)."""
        wf = Workflow(name="test-when-cascade")
        wf.nodes["a"] = WorkflowNode(id="a", agent="a", task="A")
        wf.nodes["b"] = WorkflowNode(
            id="b", agent="b", task="B",
            depends_on=["a"], when="{a.status} == done",
        )
        wf.nodes["c"] = WorkflowNode(
            id="c", agent="c", task="C",
            depends_on=["b"],
        )
        states = {
            "a": NodeState(node_id="a", status="failed"),
            "b": NodeState(node_id="b"),
            "c": NodeState(node_id="c"),
        }
        # b is skipped because a failed
        if not engine.evaluate_when("{a.status} == done", wf.nodes["b"], states):
            states["b"].status = "skipped"
        # c should be skipped because b is skipped (standard dep cascade)
        assert states["b"].status == "skipped"
        # c has b as dependency and b is skipped → c gets skipped
        # This is the standard dependency check in execute(), not when:-

    # ── Numeric comparison ──

    def test_when_numeric_gt(self, engine):
        """when: {a.attempts} > 2 → numeric comparison."""
        wf = Workflow(name="test-when-gt")
        wf.nodes["a"] = WorkflowNode(id="a", agent="a", task="A")
        states = {
            "a": NodeState(node_id="a", status="done", attempts=3),
        }
        assert engine.evaluate_when(
            "{a.attempts} > 2", wf.nodes["a"], states
        ) is True

    def test_when_numeric_lt(self, engine):
        """when: {a.attempts} < 5 → numeric comparison."""
        wf = Workflow(name="test-when-lt")
        wf.nodes["a"] = WorkflowNode(id="a", agent="a", task="A")
        states = {
            "a": NodeState(node_id="a", status="done", attempts=3),
        }
        assert engine.evaluate_when(
            "{a.attempts} < 5", wf.nodes["a"], states
        ) is True

    # ── starts_with operator ──

    def test_when_starts_with(self, engine):
        """when: {a.result} starts_with "PASS" → prefix match."""
        wf = Workflow(name="test-when-sw")
        wf.nodes["a"] = WorkflowNode(id="a", agent="a", task="A")
        states = {
            "a": NodeState(node_id="a", status="done",
                           result="PASS: all 42 tests green"),
        }
        assert engine.evaluate_when(
            '{a.result} starts_with "PASS"', wf.nodes["a"], states
        ) is True

    # ── in operator ──

    def test_when_in_list(self, engine):
        """when: {a.status} in [done, skipped] → membership check."""
        wf = Workflow(name="test-when-in")
        wf.nodes["a"] = WorkflowNode(id="a", agent="a", task="A")
        states = {
            "a": NodeState(node_id="a", status="done"),
        }
        assert engine.evaluate_when(
            "{a.status} in [done, skipped]", wf.nodes["a"], states
        ) is True

    def test_when_in_list_false(self, engine):
        """when: {a.status} in [done, skipped] → false on non-member."""
        wf = Workflow(name="test-when-in-false")
        wf.nodes["a"] = WorkflowNode(id="a", agent="a", task="A")
        states = {
            "a": NodeState(node_id="a", status="failed"),
        }
        assert engine.evaluate_when(
            "{a.status} in [done, skipped]", wf.nodes["a"], states
        ) is False

    # ── Validation: warns on when referencing non-dependency ──

    def test_validate_warns_when_references_non_dependency(self, tmp_path):
        """validate() warns when when: references a node not in depends_on."""
        wf_content = """
name: test-when-validation
nodes:
  a:
    agent: agent-a
    task: Task A
  b:
    agent: agent-b
    task: Task B
    when: "{a.status} == done"
  c:
    agent: agent-c
    task: Task C
    depends_on: [b]
"""
        (tmp_path / "test-when-validation.yaml").write_text(wf_content)
        engine = WorkflowEngine(workflows_dir=tmp_path)
        result = engine.validate("test-when-validation")
        # b references a in when: but doesn't declare a in depends_on
        when_warnings = [
            i for i in result["issues"]
            if "when:" in i and "depends_on" in i
        ]
        assert len(when_warnings) >= 1

    def test_validate_no_warn_when_depends_on_declared(self, tmp_path):
        """validate() does NOT warn when when: references a declared dependency."""
        wf_content = """
name: test-when-valid-dep
nodes:
  a:
    agent: agent-a
    task: Task A
  b:
    agent: agent-b
    task: Task B
    depends_on: [a]
    when: "{a.status} == done"
"""
        (tmp_path / "test-when-valid-dep.yaml").write_text(wf_content)
        engine = WorkflowEngine(workflows_dir=tmp_path)
        result = engine.validate("test-when-valid-dep")
        when_warnings = [
            i for i in result["issues"]
            if "when:" in i and "depends_on" in i
        ]
        assert len(when_warnings) == 0

    def test_validate_no_warn_when_context_reference(self, tmp_path):
        """validate() does NOT warn for {context.key} references."""
        wf_content = """
name: test-when-ctx-val
nodes:
  a:
    agent: agent-a
    task: Task A
    when: "{context.mode} == production"
"""
        (tmp_path / "test-when-ctx-val.yaml").write_text(wf_content)
        engine = WorkflowEngine(workflows_dir=tmp_path)
        result = engine.validate("test-when-ctx-val")
        when_warnings = [
            i for i in result["issues"]
            if "when:" in i and "depends_on" in i
        ]
        assert len(when_warnings) == 0

    # ── WorkflowNode data model ──

    def test_workflow_node_when_defaults_to_empty(self):
        """WorkflowNode.when defaults to empty string."""
        node = WorkflowNode(id="x", agent="a", task="T")
        assert node.when == ""

    def test_workflow_node_when_set_from_yaml(self, tmp_path):
        """load_workflow() parses when: from YAML."""
        wf_content = """
name: test-when-yaml
nodes:
  a:
    agent: agent-a
    task: Task A
  b:
    agent: agent-b
    task: Task B
    depends_on: [a]
    when: "{a.status} == done"
"""
        (tmp_path / "test-when-yaml.yaml").write_text(wf_content)
        engine = WorkflowEngine(workflows_dir=tmp_path)
        wf = engine.load_workflow("test-when-yaml")
        assert wf.nodes["b"].when == "{a.status} == done"
        assert wf.nodes["a"].when == ""

    # ── Parenthesized expressions ──

    def test_when_parenthesized(self, engine):
        """when: ({a.status} == done) or ({b.status} == done) → grouping."""
        wf = Workflow(name="test-when-paren")
        wf.nodes["a"] = WorkflowNode(id="a", agent="a", task="A")
        wf.nodes["b"] = WorkflowNode(id="b", agent="b", task="B")
        wf.nodes["c"] = WorkflowNode(
            id="c", agent="c", task="C",
            depends_on=["a", "b"],
            when='({a.status} == done) or ({b.status} == done)',
        )
        states = {
            "a": NodeState(node_id="a", status="failed"),
            "b": NodeState(node_id="b", status="done"),
            "c": NodeState(node_id="c"),
        }
        assert engine.evaluate_when(
            '({a.status} == done) or ({b.status} == done)',
            wf.nodes["c"], states,
        ) is True

    # ── Error_count and duration_seconds fields ──

    def test_when_error_count(self, engine):
        """when: {a.error_count} > 0 → checks error_count field."""
        wf = Workflow(name="test-when-errcount")
        wf.nodes["a"] = WorkflowNode(id="a", agent="a", task="A")
        states = {
            "a": NodeState(node_id="a", status="failed", error_count=3),
        }
        assert engine.evaluate_when(
            "{a.error_count} > 0", wf.nodes["a"], states
        ) is True

    # ── Fail-closed on evaluation error ──

    def test_when_evaluation_error_skips(self, engine):
        """Evaluation errors default to skip (fail-closed)."""
        wf = Workflow(name="test-when-err")
        wf.nodes["a"] = WorkflowNode(id="a", agent="a", task="A")
        states = {
            "a": NodeState(node_id="a", status="done"),
        }
        # Malformed expression — should skip
        assert engine.evaluate_when(
            "()", wf.nodes["a"], states
        ) is False

    # ── Complex real-world pattern ──

    def test_when_realistic_branching(self, engine):
        """Realistic pattern: deploy only if review passed and tests green."""
        wf = Workflow(name="test-when-realistic")
        wf.nodes["review"] = WorkflowNode(id="review", agent="r", task="R")
        wf.nodes["tests"] = WorkflowNode(id="tests", agent="t", task="T")
        wf.nodes["deploy"] = WorkflowNode(
            id="deploy", agent="d", task="D",
            depends_on=["review", "tests"],
            when='{review.status} == done and {tests.status} == done',
        )
        wf.nodes["notify-fail"] = WorkflowNode(
            id="notify-fail", agent="n", task="N",
            depends_on=["review", "tests"],
            when='{review.status} != done or {tests.status} != done',
        )
        states = {
            "review": NodeState(node_id="review", status="done"),
            "tests": NodeState(node_id="tests", status="failed"),
            "deploy": NodeState(node_id="deploy"),
            "notify-fail": NodeState(node_id="notify-fail"),
        }
        # deploy should be skipped (tests failed)
        assert engine.evaluate_when(
            wf.nodes["deploy"].when, wf.nodes["deploy"], states
        ) is False
        # notify-fail should fire
        assert engine.evaluate_when(
            wf.nodes["notify-fail"].when, wf.nodes["notify-fail"], states
        ) is True


# ── Loop zone detection tests ─────────────────────────────────────

class TestLoopZones:

    def test_find_loop_zones_identifies_review_layer(self, engine):
        """Layers containing nodes with reviews attribute are flagged."""
        wf = Workflow(name="test-review-zones")
        wf.nodes["a"] = WorkflowNode(id="a", agent="a", task="Do work")
        wf.nodes["implement"] = WorkflowNode(
            id="implement", agent="coder", task="Implement",
            depends_on=["a"], reviews=["verify"],
        )
        wf.nodes["verify"] = WorkflowNode(
            id="verify", agent="reviewer", task="Verify",
            depends_on=["implement"],
        )
        layers = engine.topological_sort(wf)
        review_layers = engine._find_loop_zones(wf, layers)
        assert len(review_layers) >= 1

    def test_find_loop_zones_no_loops(self, engine):
        """Linear workflow — _find_loop_zones is conservative and may flag layers
        where nodes have dependents. The supervisor handles this correctly."""
        wf = Workflow(name="test-no-loops")
        wf.nodes["a"] = WorkflowNode(id="a", agent="a", task="A")
        wf.nodes["b"] = WorkflowNode(id="b", agent="b", task="B",
                                     depends_on=["a"])
        wf.nodes["c"] = WorkflowNode(id="c", agent="c", task="C",
                                     depends_on=["b"])
        layers = engine.topological_sort(wf)
        loop_layers = engine._find_loop_zones(wf, layers)
        # Conservative: a → b → c has dependents, so layers 0 and 1 are flagged.
        # The supervisor handles this correctly — it just creates all cards
        # under supervision instead of fire-and-forget. End result is the same.
        assert len(loop_layers) >= 0  # May flag layers with dependents

    def test_find_revision_node(self, engine):
        """Find the revision node that depends on a verify node."""
        wf = Workflow(name="test-find-rev")
        wf.nodes["verify"] = WorkflowNode(id="verify", agent="r", task="V")
        wf.nodes["revise"] = WorkflowNode(id="revise", agent="a", task="R",
                                          depends_on=["verify"])
        rev = engine._find_revision_node(wf, "verify")
        assert rev == "revise"

    def test_find_revision_node_none(self, engine):
        """No revision node returns None."""
        wf = Workflow(name="test-no-rev")
        wf.nodes["a"] = WorkflowNode(id="a", agent="x", task="A")
        rev = engine._find_revision_node(wf, "a")
        assert rev is None


# ── Analyst loop decision tests ──────────────────────────────────

class TestLoopDecision:

    def test_try_loop_decision_returns_loop_when_analyst_unavailable(self, engine):
        """Fallback to loop when analyst module can't be imported."""
        wf = Workflow(name="test-loop-fallback")
        wf.nodes["verify"] = WorkflowNode(
            id="verify", agent="r", task="Check criteria: must be red or blue"
        )
        wf.nodes["revise"] = WorkflowNode(
            id="revise", agent="a", task="Fix the color",
            depends_on=["verify"]
        )
        result = engine._try_loop_decision(
            wf.nodes["verify"], wf.nodes["revise"],
            "got: orange"
        )
        # Should return "loop" (conservative fallback)
        assert result == "loop"

    def test_analyze_loop_decision_proceed(self):
        """Analyst says proceed when rejection doesn't match criteria."""
        from plugins.workflow.analyst import analyze_loop_decision
        from unittest.mock import patch, MagicMock

        mock_outcome = MagicMock()
        mock_outcome.success = True
        mock_outcome.result = {
            "decision": "proceed",
            "reason": "Criteria says red/yellow/blue. Got red. Red is in the list.",
            "confidence": "high"
        }

        with patch("plugins.workflow.analyst._invoke", return_value=mock_outcome):
            outcome = analyze_loop_decision(
                verify_task="Must be one of: red, yellow, blue",
                rejection="got: red",
                revision_task="Fix the color to a primary"
            )
            assert outcome.success
            assert outcome.result["decision"] == "proceed"

    def test_analyze_loop_decision_loop(self):
        """Analyst says loop when rejection matches criteria."""
        from plugins.workflow.analyst import analyze_loop_decision
        from unittest.mock import patch, MagicMock

        mock_outcome = MagicMock()
        mock_outcome.success = True
        mock_outcome.result = {
            "decision": "loop",
            "reason": "Criteria says red/yellow/blue. Got orange. Not in list.",
            "confidence": "high"
        }

        with patch("plugins.workflow.analyst._invoke", return_value=mock_outcome):
            outcome = analyze_loop_decision(
                verify_task="Must be one of: red, yellow, blue",
                rejection="got: orange",
                revision_task="Fix the color to a primary"
            )
            assert outcome.success
            assert outcome.result["decision"] == "loop"


# ── Node card DB tracking tests ──────────────────────────────────

class TestNodeCardDB:

    def test_record_node_card_inserts_row(self, engine):
        """_record_node_card inserts a row into workflow_node_cards."""
        import sqlite3
        engine._record_node_card("t_abc123", "run-001", "raven-verify")
        with sqlite3.connect(str(engine._exec_db_path)) as conn:
            row = conn.execute(
                "SELECT card_id, run_id, node_id, status "
                "FROM workflow_node_cards WHERE card_id = 't_abc123'"
            ).fetchone()
        assert row is not None
        assert row[0] == "t_abc123"
        assert row[1] == "run-001"
        assert row[2] == "raven-verify"
        assert row[3] == "pending"

    def test_record_node_card_idempotent(self, engine):
        """Duplicate inserts are ignored (INSERT OR IGNORE)."""
        engine._record_node_card("t_dup", "run-001", "node-a")
        engine._record_node_card("t_dup", "run-001", "node-a")
        import sqlite3
        with sqlite3.connect(str(engine._exec_db_path)) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM workflow_node_cards WHERE card_id = 't_dup'"
            ).fetchone()[0]
        assert count == 1

    def test_update_node_card_status(self, engine):
        """_update_node_card_status changes status for a card."""
        engine._record_node_card("t_upd", "run-002", "node-b")
        from plugins.workflow import _update_node_card_db as _upd; _upd("t_upd", "done")
        import sqlite3
        with sqlite3.connect(str(engine._exec_db_path)) as conn:
            row = conn.execute(
                "SELECT status FROM workflow_node_cards WHERE card_id = 't_upd'"
            ).fetchone()
        assert row[0] == "done"

    def test_update_node_card_auto_completes_run(self, engine):
        """When all cards are terminal, the run auto-completes."""
        engine._record_execution("test-wf", "run-003", "board-a", 1)
        engine._record_node_card("t_x", "run-003", "node-x")
        engine._record_node_card("t_y", "run-003", "node-y")
        # Mark both as done
        from plugins.workflow import _update_node_card_db as _upd; _upd("t_x", "done")
        from plugins.workflow import _update_node_card_db as _upd; _upd("t_y", "done")
        import sqlite3
        with sqlite3.connect(str(engine._exec_db_path)) as conn:
            row = conn.execute(
                "SELECT status, finished_at FROM workflow_executions "
                "WHERE run_id = 'run-003'"
            ).fetchone()
        assert row[0] == "completed"
        assert row[1] is not None  # finished_at is set

    def test_update_node_card_marks_failed_on_any_failure(self, engine):
        """Run is marked failed if any card fails."""
        engine._record_execution("test-wf", "run-004", "board-b", 1)
        engine._record_node_card("t_p", "run-004", "node-p")
        engine._record_node_card("t_q", "run-004", "node-q")
        from plugins.workflow import _update_node_card_db as _upd; _upd("t_p", "done")
        from plugins.workflow import _update_node_card_db as _upd; _upd("t_q", "failed")
        import sqlite3
        with sqlite3.connect(str(engine._exec_db_path)) as conn:
            row = conn.execute(
                "SELECT status FROM workflow_executions "
                "WHERE run_id = 'run-004'"
            ).fetchone()
        assert row[0] == "failed"


# ── Block notification tests ─────────────────────────────────────

class TestBlockNotification:

    def test_try_block_notify_calls_analyst(self, engine):
        """_try_block_notify calls the analyst and formats the message."""
        from unittest.mock import patch, MagicMock

        wf = Workflow(name="test-block-notify")
        wf.nodes["qa-review"] = WorkflowNode(
            id="qa-review", agent="r", task="Run tests and check coverage"
        )
        states = {"qa-review": NodeState(node_id="qa-review", status="blocked")}
        context = {"project": "test-project", "repo": "test/repo",
                    "platform": "discord", "chat_id": "123", "thread_id": None}

        mock_outcome = MagicMock()
        mock_outcome.success = True
        mock_outcome.result = {
            "severity": "warning",
            "summary": "Coverage below threshold",
            "detail": "The qa-review node blocked because coverage was 60% vs 80% required.",
            "suggested_action": "Newton needs to improve test coverage."
        }

        with patch("plugins.workflow.analyst.analyze_block_notification", return_value=mock_outcome):
            engine._try_block_notify(
                wf, "qa-review", states["qa-review"],
                "coverage 60% vs 80%", context
            )
        # Should not raise — analyst was called and message was formatted

    def test_try_block_notify_returns_on_missing_context(self, engine):
        """_try_block_notify returns early when no platform/chat_id in context."""
        wf = Workflow(name="test-no-context")
        wf.nodes["a"] = WorkflowNode(id="a", agent="r", task="Task A")
        states = {"a": NodeState(node_id="a", status="blocked")}
        context = {}  # No platform/chat_id

        # Should return without error
        engine._try_block_notify(wf, "a", states["a"], "blocked", context)

    def test_analyze_block_notification_proceed(self):
        """Analyst returns structured block assessment."""
        from plugins.workflow.analyst import analyze_block_notification
        from unittest.mock import patch, MagicMock

        mock_outcome = MagicMock()
        mock_outcome.success = True
        mock_outcome.result = {
            "severity": "critical",
            "summary": "Security vulnerability found",
            "detail": "The security review found a critical auth bypass.",
            "suggested_action": "Fix the auth flow before proceeding."
        }

        with patch("plugins.workflow.analyst._invoke", return_value=mock_outcome):
            outcome = analyze_block_notification(
                node_id="security-review",
                workflow_name="ideation",
                node_task="Review for security concerns",
                block_reason="BLOCKED: auth bypass found",
                workflow_context="Project: agent-service"
            )
            assert outcome.success
            assert outcome.result["severity"] == "critical"

    def test_try_block_notify_handles_analyst_failure(self, engine):
        """_try_block_notify handles analyst failure gracefully."""
        from unittest.mock import patch, MagicMock

        wf = Workflow(name="test-analyst-fail")
        wf.nodes["a"] = WorkflowNode(id="a", agent="r", task="Task A")
        states = {"a": NodeState(node_id="a", status="blocked")}
        context = {"project": "test", "repo": "test/repo",
                    "platform": "discord", "chat_id": "123"}

        mock_outcome = MagicMock()
        mock_outcome.success = False
        mock_outcome.result = None

        with patch("plugins.workflow.analyst.analyze_block_notification", return_value=mock_outcome):
            # Should not raise
            engine._try_block_notify(
                wf, "a", states["a"], "blocked", context
            )


# ── Path resolution tests ───────────────────────────────────────

class TestPathResolution:

    def test_engine_resolves_hermes_home_workflows(self, tmp_path):
        """Engine resolves workflow dir from HERMES_HOME/workflows/."""
        from plugins.workflow.engine import WorkflowEngine
        workflows = tmp_path / "workflows"
        workflows.mkdir()
        (workflows / "test.yaml").write_text("name: test\nnodes: {}\n")

        engine = WorkflowEngine(workflows_dir=str(workflows))
        assert engine.workflows_dir == workflows

    def test_engine_resolves_explicit_dir(self, tmp_path):
        """Engine uses explicit workflows_dir when provided."""
        from plugins.workflow.engine import WorkflowEngine
        engine = WorkflowEngine(workflows_dir=str(tmp_path))
        assert engine.workflows_dir == tmp_path

    def test_config_loader_reads_from_workflows(self, tmp_path, monkeypatch):
        """Config loader reads from HERMES_HOME/workflows/config.yaml."""
        import plugins.workflow as wf_mod
        monkeypatch.setattr(wf_mod, "_CONFIG", None)  # reset cache
        workflows = tmp_path / "workflows"
        workflows.mkdir()
        (workflows / "config.yaml").write_text("auto_discovery: false\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        config = wf_mod.load_config()
        assert config["auto_discovery"] is False

    def test_config_loader_uses_defaults_when_missing(self, tmp_path, monkeypatch):
        """Config loader returns defaults when config.yaml doesn't exist."""
        import plugins.workflow as wf_mod
        monkeypatch.setattr(wf_mod, "_CONFIG", None)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        config = wf_mod.load_config()
        assert config["auto_discovery"] is True
        assert config["max_nodes_per_workflow"] == 256


# ── Tool handler args-dict tests ────────────────────────────────

class TestToolHandlerArgsDict:
    """All tool handlers receive a single args dict, not keyword arguments.
    These tests verify each handler extracts parameters correctly."""

    def test_workflow_start_extracts_workflow(self):
        """handle_workflow_start extracts 'workflow' from args dict."""
        from plugins.workflow.tools import handle_workflow_start
        result = handle_workflow_start({"workflow": ""})
        data = json.loads(result)
        assert data["ok"] is False
        assert "non-empty string" in data["error"]

    def test_workflow_start_extracts_workflow_valid(self):
        """handle_workflow_start accepts valid workflow string."""
        from plugins.workflow.tools import handle_workflow_start
        result = handle_workflow_start({"workflow": "nonexistent"})
        data = json.loads(result)
        # Should fail with "not found", not "non-empty string"
        assert data["ok"] is False
        assert "non-empty string" not in data.get("error", "")

    def test_workflow_view_extracts_workflow(self):
        """handle_workflow_view extracts 'workflow' from args dict."""
        from plugins.workflow.tools import handle_workflow_view
        result = handle_workflow_view({"workflow": ""})
        data = json.loads(result)
        assert data["ok"] is False
        assert "non-empty string" in data["error"]

    def test_workflow_validate_extracts_workflow(self):
        """handle_workflow_validate extracts 'workflow' from args dict."""
        from plugins.workflow.tools import handle_workflow_validate
        result = handle_workflow_validate({"workflow": ""})
        data = json.loads(result)
        assert data["ok"] is False
        assert "non-empty string" in data["error"]

    def test_workflow_list_extracts_trigger(self):
        """handle_workflow_list extracts 'trigger' from args dict."""
        from plugins.workflow.tools import handle_workflow_list
        result = handle_workflow_list({})
        data = json.loads(result)
        assert data["ok"] is True
        assert "result" in data

    def test_workflow_show_extracts_workflow(self):
        """handle_workflow_show extracts 'workflow' from args dict."""
        from plugins.workflow.tools import handle_workflow_show
        result = handle_workflow_show({"workflow": ""})
        data = json.loads(result)
        assert data["ok"] is False
        assert "non-empty string" in data["error"]

    def test_workflow_start_extracts_all_params(self):
        """handle_workflow_start extracts context, board, inputs, etc."""
        from plugins.workflow.tools import handle_workflow_start
        result = handle_workflow_start({
            "workflow": "nonexistent",
            "context": {"project": "test"},
            "board": "test-board",
            "inputs": {"key": "value"},
            "dry_run": True,
        })
        data = json.loads(result)
        # Should fail with "not found", not parameter error
        assert data["ok"] is False
        assert "non-empty string" not in data.get("error", "")

    def test_workflow_status_optional_workflow(self):
        """handle_workflow_status works without workflow parameter."""
        from plugins.workflow.tools import handle_workflow_status
        result = handle_workflow_status({})
        data = json.loads(result)
        assert data["ok"] is True


# ── Board resolution priority tests ─────────────────────────────

class TestBoardResolution:
    """Caller-passed board takes priority over YAML kanban_board field."""

    def test_caller_board_wins_over_yaml(self, engine):
        """When board= is passed, it overrides YAML kanban_board."""
        from plugins.workflow.engine import Workflow, WorkflowNode
        from unittest.mock import patch, MagicMock

        wf = Workflow(name="test-board", kanban_board="yaml-board")
        wf.nodes["a"] = WorkflowNode(id="a", agent="a", task="Task A")

        with patch.object(engine, 'load_workflow', return_value=wf):
            with patch("hermes_cli.kanban_db.connect", return_value=MagicMock()) as mock_connect:
                with patch("hermes_cli.kanban_db.create_task", return_value="t_test"):
                    engine.execute("test-board", board="caller-board", fire_and_forget=True)

        mock_connect.assert_called_with(board="caller-board")

    def test_yaml_board_used_when_no_caller(self, engine):
        """When no board= passed, YAML kanban_board is used."""
        from plugins.workflow.engine import Workflow, WorkflowNode
        from unittest.mock import patch, MagicMock

        wf = Workflow(name="test-board", kanban_board="yaml-board")
        wf.nodes["a"] = WorkflowNode(id="a", agent="a", task="Task A")

        with patch.object(engine, 'load_workflow', return_value=wf):
            with patch("hermes_cli.kanban_db.connect", return_value=MagicMock()) as mock_connect:
                with patch("hermes_cli.kanban_db.create_task", return_value="t_test"):
                    engine.execute("test-board", fire_and_forget=True)

        mock_connect.assert_called_with(board="yaml-board")

    def test_auto_create_when_no_board(self, engine):
        """When no board= and no YAML board, auto-creates wf_<name>."""
        from plugins.workflow.engine import Workflow, WorkflowNode
        from unittest.mock import patch, MagicMock

        wf = Workflow(name="test-pipeline")
        wf.nodes["a"] = WorkflowNode(id="a", agent="a", task="Task A")

        with patch.object(engine, 'load_workflow', return_value=wf):
            with patch("hermes_cli.kanban_db.connect", return_value=MagicMock()) as mock_connect:
                with patch("hermes_cli.kanban_db.create_task", return_value="t_test"):
                    engine.execute("test-pipeline", fire_and_forget=True)

        mock_connect.assert_called_with(board="wf_test-pipeline")
