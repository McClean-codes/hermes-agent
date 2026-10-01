"""Reviewer gate + profile/skill validation for Kanban task creation and routing.

What these tests pin down (all of it new behaviour; every test here is red on
the base commit ``655010d6``):

* ``kanban_create``'s optional ``reviewer`` persists ``tasks.required_reviewer``
  and records it on the ``created`` event; omitting it changes nothing.
* Validation happens BEFORE every side effect: a bad reviewer profile, a
  missing forced skill, or a mixed valid/missing skill list leaves no task row,
  no dependency edge, no event and no workspace behind.
* Nonexistent profiles are reported as ``profile '<name>' was not found`` with
  the roster, ``kanban_discover`` guidance and ``Nothing changed`` — never as
  "not installed".
* A reviewer must carry the skill the dispatcher force-loads for review-phase
  startup (``sdlc-review``); an existing profile without it is rejected.
* ``request_review`` routes to the SAVED reviewer and refuses any override;
  the gate survives the request_changes -> implementer cycle.
* ``complete_task`` refuses an implementation run (or an unclaimed/spoofed
  caller) on a gated card, allows the trusted review run, and records an audit
  event when an operator explicitly overrides.
* ``kanban_complete`` is absent from the schema of a gated implementation run,
  while the ungated / review-phase schemas keep it.
* Reassignment preserves forced skills and the reviewer gate.
* Migration + readback expose ``required_reviewer``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_validation as kv
from hermes_cli import profiles as profiles_mod


PROFILES = ("default", "qa", "reviewer", "worker")


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME + empty board, with the profile roster pinned.

    The roster is stubbed (as the other review-lifecycle tests do) so profile
    existence is a decision of this test, not of whatever happens to be on the
    host. The skill library is deliberately left to the real resolver: a bare
    hermetic home falls back to the checkout's bundled tree, exactly like a
    profile ``seed_profile_skills`` has populated.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda name: name in PROFILES)
    monkeypatch.setattr(profiles_mod, "list_profile_names", lambda: list(PROFILES))
    return home


def _seed_skill(home: Path, name: str) -> Path:
    """Give ``home`` a skill tree containing exactly ``name`` (and nothing else).

    A non-empty tree makes skill resolution strict for that home, so these
    tests exercise the real verdict rather than the bundled fallback.
    """
    skill_dir = home / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test double\n---\n\nbody\n",
        encoding="utf-8",
    )
    return home / "skills"


def _counts(conn) -> dict:
    return {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("tasks", "task_links", "task_events", "task_runs")
    }


def _events(conn, tid, kind=None):
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id", (tid,),
    ).fetchall()
    out = [(r["kind"], json.loads(r["payload"]) if r["payload"] else None) for r in rows]
    return [e for e in out if e[0] == kind] if kind else out


# ---------------------------------------------------------------------------
# Creation: gate persisted, validated before every side effect
# ---------------------------------------------------------------------------


def test_create_with_reviewer_persists_gate_and_event(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        task = kb.get_task(conn, tid)
        assert task.required_reviewer == "reviewer"
        created = _events(conn, tid, "created")[0][1]
        assert created["required_reviewer"] == "reviewer"
        # Readback surfaces the gate on every task view.
        from hermes_cli.kanban_output import _task_to_dict

        assert _task_to_dict(task)["required_reviewer"] == "reviewer"


def test_create_without_reviewer_is_ungated(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="plain work", assignee="worker")
        assert kb.get_task(conn, tid).required_reviewer is None


def test_unknown_reviewer_profile_is_zero_writes(board: Path) -> None:
    with kbc.connect() as conn:
        before = _counts(conn)
        with pytest.raises(kv.ProfileNotFoundError) as excinfo:
            kb.create_task(
                conn, title="gated work", assignee="worker", reviewer="ghost",
            )
        message = str(excinfo.value)
        assert "profile 'ghost' was not found" in message
        assert "not installed" not in message
        assert "Available profiles:" in message
        assert "kanban_discover" in message
        assert message.rstrip().endswith("Nothing changed.")
        assert _counts(conn) == before, "a refused create wrote to the board"


def test_reviewer_without_review_skill_is_zero_writes(board: Path) -> None:
    # A non-empty skill tree that simply lacks the dispatcher's review skill:
    # the profile exists, but cannot run the review phase.
    _seed_skill(board, "some-other-skill")
    with kbc.connect() as conn:
        before = _counts(conn)
        with pytest.raises(kv.MissingSkillsError) as excinfo:
            kb.create_task(
                conn, title="gated work", assignee="worker", reviewer="reviewer",
            )
        message = str(excinfo.value)
        assert "reviewer" in message and "sdlc-review" in message
        assert "not found for profile" in message
        assert _counts(conn) == before


def test_valid_reviewer_with_bundled_review_skill_is_accepted(board: Path) -> None:
    # Bare home -> bundled fallback; the checkout ships sdlc-review, which is
    # what seed_profile_skills installs into every real profile.
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        assert kb.get_task(conn, tid).required_reviewer == "reviewer"


def test_absent_forced_skill_is_zero_writes(board: Path) -> None:
    with kbc.connect() as conn:
        before = _counts(conn)
        with pytest.raises(kv.MissingSkillsError) as excinfo:
            kb.create_task(
                conn, title="specialist work", assignee="worker",
                skills=["definitely-not-a-skill"],
            )
        message = str(excinfo.value)
        assert "worker" in message and "definitely-not-a-skill" in message
        assert _counts(conn) == before
        assert not any((board / "workspaces").glob("*")), "a refused create made a workspace"


def test_mixed_valid_and_missing_skills_reject_atomically(board: Path) -> None:
    with kbc.connect() as conn:
        before = _counts(conn)
        with pytest.raises(kv.MissingSkillsError) as excinfo:
            kb.create_task(
                conn, title="mixed work", assignee="worker",
                skills=["sdlc-review", "definitely-not-a-skill"],
            )
        message = str(excinfo.value)
        # The whole list is judged at once: the profile and ONLY the missing
        # names are reported, and nothing is written.
        assert "worker" in message
        assert "definitely-not-a-skill" in message
        assert "sdlc-review" not in message.split("Nothing changed.")[0].split(": ", 1)[-1]
        assert _counts(conn) == before


def test_external_shared_skill_is_accepted(board: Path) -> None:
    """A skill living in a shared/external dir (``skills.external_dirs``) is
    part of the profile's effective library."""
    shared = board.parent / "shared-skills"
    skill_dir = shared / "team-playbook"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: team-playbook\ndescription: shared\n---\n\nbody\n",
        encoding="utf-8",
    )
    # The skill tree is read from the ASSIGNEE's own home, so give that profile
    # a home whose config points at the shared tree (an absolute entry, as
    # ``skills.external_dirs`` documents).
    profile_dir = board / "profiles" / "worker"
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "config.yaml").write_text(
        "skills:\n  external_dirs:\n    - {}\n".format(shared), encoding="utf-8",
    )
    _seed_skill(board, "sdlc-review")  # non-empty -> strict mode
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="shared work", assignee="worker",
            skills=["team-playbook"],
        )
        assert kb.get_task(conn, tid).skills == ["team-playbook"]


# ---------------------------------------------------------------------------
# request_review: saved routing, override rejection, capability
# ---------------------------------------------------------------------------


def test_request_review_routes_to_the_saved_reviewer(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        run = kb.claim_task(conn, tid)
        ok, why = kb.request_review(
            conn, tid, summary="done", expected_run_id=run.current_run_id,
            with_reason=True,
        )
        assert ok is True, why
        task = kb.get_task(conn, tid)
        assert task.status == "review"
        assert task.assignee == "reviewer"
        assert task.required_reviewer == "reviewer"
        payload = _events(conn, tid, "review_requested")[0][1]
        assert payload["reviewer"] == "reviewer"
        assert payload["implementer"] == "worker"


def test_request_review_forbids_a_reviewer_override(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        run = kb.claim_task(conn, tid)
        before = _events(conn, tid)
        ok, why = kb.request_review(
            conn, tid, summary="done", reviewer="qa",
            expected_run_id=run.current_run_id, with_reason=True,
        )
        assert ok is False
        assert "override is not allowed" in why
        assert "reviewer" in why and "qa" in why
        task = kb.get_task(conn, tid)
        assert (task.status, task.assignee) == ("running", "worker")
        assert _events(conn, tid) == before


def test_request_review_rejects_a_reviewer_without_the_review_skill(board: Path) -> None:
    _seed_skill(board, "some-other-skill")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ungated work", assignee="worker")
        run = kb.claim_task(conn, tid)
        before = _events(conn, tid)
        ok, why = kb.request_review(
            conn, tid, summary="done", reviewer="qa",
            expected_run_id=run.current_run_id, with_reason=True,
        )
        assert ok is False
        assert "sdlc-review" in why and "qa" in why
        assert kb.get_task(conn, tid).status == "running"
        assert _events(conn, tid) == before


def test_review_changes_cycle_retains_the_gate(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        impl_run = kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="first pass", expected_run_id=impl_run.current_run_id,
        )
        review_run = kb.claim_review_task(conn, tid)
        assert review_run is not None
        assert review_run.assignee == "reviewer"
        ok, implementer = kb.request_changes(conn, tid, reason="needs tests")
        assert ok is True and implementer == "worker"
        task = kb.get_task(conn, tid)
        # Back with the implementer, gate intact and reviewer unchanged.
        assert task.status in ("ready", "todo")
        assert task.required_reviewer == "reviewer"
        assert task.assignee == "worker"
        # Second handoff still routes to the saved reviewer, still no override.
        ok, why = kb.request_review(conn, tid, summary="second pass", with_reason=True)
        assert ok is True, why
        assert kb.get_task(conn, tid).assignee == "reviewer"


# ---------------------------------------------------------------------------
# complete_task: backend gate (authoritative lifecycle/run state)
# ---------------------------------------------------------------------------


def test_implementation_completion_is_refused(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        run = kb.claim_task(conn, tid)
        with pytest.raises(kb.ReviewerGateError) as excinfo:
            kb.complete_task(
                conn, tid, summary="all done",
                expected_run_id=run.current_run_id,
            )
        message = str(excinfo.value)
        assert "required reviewer" in message and "reviewer" in message
        assert "Nothing changed." in message
        task = kb.get_task(conn, tid)
        assert task.status == "running"
        assert task.completed_at is None


def test_unclaimed_and_forced_completion_are_still_refused(board: Path) -> None:
    """Neither an unclaimed caller nor ``force=True`` gets past the gate:
    ``force`` only governs a live claim, and the phase is read from the board."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        with pytest.raises(kb.ReviewerGateError):
            kb.complete_task(conn, tid, summary="done without claiming")
        with pytest.raises(kb.ReviewerGateError):
            kb.complete_task(conn, tid, summary="forced", force=True)
        assert kb.get_task(conn, tid).status in ("ready", "todo", "running")


def test_trusted_same_profile_review_run_can_complete(board: Path) -> None:
    """reviewer == implementer: the PHASE, not the profile string, decides."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="self-reviewed work", assignee="worker", reviewer="worker",
        )
        impl_run = kb.claim_task(conn, tid)
        with pytest.raises(kb.ReviewerGateError):
            kb.complete_task(conn, tid, summary="done", expected_run_id=impl_run.current_run_id)
        assert kb.request_review(
            conn, tid, summary="ready", expected_run_id=impl_run.current_run_id,
        )
        review_run = kb.claim_review_task(conn, tid)
        assert review_run is not None and review_run.assignee == "worker"
        assert kb.complete_task(
            conn, tid, summary="approved by reviewer",
            expected_run_id=review_run.current_run_id,
        )
        assert kb.get_task(conn, tid).status == "done"


def test_wrong_profile_claiming_review_is_refused(board: Path) -> None:
    """A live review run that is NOT the saved reviewer's cannot close the card."""
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        run = kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="ready", expected_run_id=run.current_run_id,
        )
        review_run = kb.claim_review_task(conn, tid)
        assert review_run is not None
        # Forge the identity the way a spoofing caller would: the run still
        # belongs to the card, but not to the saved reviewer.
        conn.execute(
            "UPDATE task_runs SET profile = 'qa' WHERE id = ?", (review_run.current_run_id,),
        )
        conn.commit()
        with pytest.raises(kb.ReviewerGateError) as excinfo:
            kb.complete_task(
                conn, tid, summary="approved",
                expected_run_id=review_run.current_run_id,
            )
        assert "not the saved reviewer's review run" in str(excinfo.value)


def test_explicit_override_completes_and_is_audited(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        run = kb.claim_task(conn, tid)
        assert kb.complete_task(
            conn, tid, summary="operator override",
            expected_run_id=run.current_run_id, review_gate_override=True,
        )
        assert kb.get_task(conn, tid).status == "done"
        assert len(_events(conn, tid, "reviewer_gate_overridden")) == 1


def test_ungated_completion_is_unchanged(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="plain work", assignee="worker")
        run = kb.claim_task(conn, tid)
        assert kb.complete_task(
            conn, tid, summary="done", expected_run_id=run.current_run_id,
        )
        task = kb.get_task(conn, tid)
        assert task.status == "done"
        assert _events(conn, tid, "reviewer_gate_overridden") == []


# ---------------------------------------------------------------------------
# Reassignment preserves skills and the gate
# ---------------------------------------------------------------------------


def test_reassign_refuses_a_profile_without_the_forced_skill(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="specialist work", assignee="worker",
            skills=["sdlc-review"],
        )
        kb.assign_task(conn, tid, "reviewer")
        _seed_skill(board, "some-other-skill")  # strict mode from here on
        with pytest.raises(kv.MissingSkillsError) as excinfo:
            kb.assign_task(conn, tid, "qa")
        assert "qa" in str(excinfo.value) and "sdlc-review" in str(excinfo.value)
        assert kb.get_task(conn, tid).assignee == "reviewer"


def test_reassign_cannot_move_a_review_phase_card_off_its_reviewer(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        run = kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="ready", expected_run_id=run.current_run_id,
        )
        with pytest.raises(ValueError) as excinfo:
            kb.assign_task(conn, tid, "qa")
        assert "required reviewer" in str(excinfo.value)
        assert kb.get_task(conn, tid).assignee == "reviewer"


# ---------------------------------------------------------------------------
# Tool schema / dispatcher startup context
# ---------------------------------------------------------------------------


def test_kanban_complete_is_hidden_from_a_gated_implementation_run(
    board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import kanban_tools as kt
    from tools.registry import invalidate_check_fn_cache, registry

    def offered() -> set:
        return {
            d["function"]["name"]
            for d in registry.get_definitions({"kanban_complete"}, quiet=True)
        }

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_deadbeef")
    monkeypatch.setenv("HERMES_KANBAN_REQUIRED_REVIEWER", "reviewer")

    monkeypatch.setenv("HERMES_KANBAN_RUN_PHASE", "implementation")
    invalidate_check_fn_cache()
    assert kt._check_kanban_complete_mode() is False
    assert offered() == set(), "gated implementation run still received kanban_complete"

    monkeypatch.setenv("HERMES_KANBAN_RUN_PHASE", "review")
    invalidate_check_fn_cache()
    assert kt._check_kanban_complete_mode() is True
    assert offered() == {"kanban_complete"}

    # Ungated: neither var set, the tool is offered exactly as before.
    monkeypatch.delenv("HERMES_KANBAN_REQUIRED_REVIEWER")
    monkeypatch.delenv("HERMES_KANBAN_RUN_PHASE")
    invalidate_check_fn_cache()
    assert kt._check_kanban_complete_mode() is True
    assert offered() == {"kanban_complete"}


def test_dispatcher_review_gate_env_contract(board: Path) -> None:
    gated = kb.Task(
        id="t_deadbeef", title="t", body=None, assignee="worker", status="running",
        priority=0, created_by=None, created_at=0, started_at=None, completed_at=None,
        workspace_kind="scratch", workspace_path=None, claim_lock=None,
        claim_expires=None, tenant=None, required_reviewer="reviewer",
    )
    assert kbd.review_gate_env(gated) == {
        "HERMES_KANBAN_REQUIRED_REVIEWER": "reviewer",
        "HERMES_KANBAN_RUN_PHASE": "implementation",
    }
    gated.run_phase = "review"
    assert kbd.review_gate_env(gated)["HERMES_KANBAN_RUN_PHASE"] == "review"
    ungated = kb.Task(
        id="t_cafebabe", title="t", body=None, assignee="worker", status="running",
        priority=0, created_by=None, created_at=0, started_at=None, completed_at=None,
        workspace_kind="scratch", workspace_path=None, claim_lock=None,
        claim_expires=None, tenant=None,
    )
    assert kbd.review_gate_env(ungated) == {}


def test_worker_context_states_the_gate_once(board: Path) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
            body="spec goes here",
        )
        context = kb.build_worker_context(conn, tid)
        assert context.count("Required reviewer: reviewer") == 1
        assert "kanban_request_review" in context
        # The gate is context, not body: the description itself is untouched.
        assert kb.get_task(conn, tid).body == "spec goes here"


# ---------------------------------------------------------------------------
# Migration / readback
# ---------------------------------------------------------------------------


def test_migration_adds_required_reviewer_to_a_legacy_board(board: Path) -> None:
    import sqlite3

    db_path = board / "kanban.db"
    conn = sqlite3.connect(db_path)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
        assert "required_reviewer" in cols
        # Simulate a pre-gate board: drop the column, then reopen.
        conn.execute("ALTER TABLE tasks DROP COLUMN required_reviewer")
        conn.execute(
            "INSERT INTO tasks (id, title, status, created_at, workspace_kind) "
            "VALUES ('t_legacy', 'legacy card', 'ready', 0, 'scratch')"
        )
        conn.commit()
    finally:
        conn.close()

    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db(db_path)
    with kbc.connect() as conn:
        task = kb.get_task(conn, "t_legacy")
        assert task is not None
        assert task.required_reviewer is None
        # New rows on the migrated board carry the gate again.
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        assert kb.get_task(conn, tid).required_reviewer == "reviewer"


# ---------------------------------------------------------------------------
# Agent-facing surface: profile wording + zero writes before the board opens
# ---------------------------------------------------------------------------


def test_tool_create_rejects_unknown_profile_with_shared_wording(board: Path) -> None:
    from tools import kanban_tools as kt

    with kbc.connect() as conn:
        before = _counts(conn)
    out = json.loads(kt._handle_create({"title": "typo", "assignee": "nope"}))
    assert out.get("ok") is not True
    message = out["error"]
    assert "profile 'nope' was not found" in message
    assert "not installed" not in message
    assert "Available profiles:" in message
    assert "kanban_discover" in message
    assert message.endswith("Nothing changed.")
    with kbc.connect() as conn:
        assert _counts(conn) == before


def test_tool_create_rejects_missing_skill_before_the_board_opens(board: Path) -> None:
    from tools import kanban_tools as kt

    with kbc.connect() as conn:
        before = _counts(conn)
    out = json.loads(kt._handle_create({
        "title": "specialist", "assignee": "worker",
        "skills": ["definitely-not-a-skill"],
    }))
    assert out.get("ok") is not True
    assert "worker" in out["error"] and "definitely-not-a-skill" in out["error"]
    with kbc.connect() as conn:
        assert _counts(conn) == before


def test_create_schema_declares_the_reviewer_gate() -> None:
    from tools.registry import registry

    properties = registry.get_schema("kanban_create")["parameters"]["properties"]
    assert "reviewer" in properties
    # The gate must not hand an agent the backend's recovery lever.
    complete = registry.get_schema("kanban_complete")["parameters"]["properties"]
    assert "review_gate_override" not in complete
    assert "force" not in complete


def test_delegate_child_cannot_close_a_gated_card(
    board: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegated permission protection still outranks the gate: a delegate
    child inherits HERMES_KANBAN_* but is never a run owner."""
    from agent import delegation_context
    from tools import kanban_tools as kt

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="gated work", assignee="worker", reviewer="reviewer",
        )
        assert kb.claim_task(conn, tid) is not None
        before = _events(conn, tid)

    monkeypatch.setattr(delegation_context, "is_delegated_child_process_context", lambda: True)
    out = json.loads(kt._handle_complete({"task_id": tid, "summary": "done"}))
    assert out.get("ok") is not True
    assert "delegate_task child agents are not Kanban run owners" in out["error"]
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid).status == "running"
        assert _events(conn, tid) == before
