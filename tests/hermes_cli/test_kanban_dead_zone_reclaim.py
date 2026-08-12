"""Tests: dispatcher dead-zone recovery — a card left in a dispatchable
non-running phase ('ready'/'review') with a stale claim_lock, dead worker_pid
and open run must be auto-reclaimed by ``detect_crashed_workers``.

Background (fleetops report, 2026-08-12): when a worker dies mid-API-call
(OOM, SIGKILL, parent signal) the card can end up back in 'ready' while the
claim bookkeeping (claim_lock / claim_expires / worker_pid) and the open run
row are still set. The dispatcher's ready query requires ``claim_lock IS
NULL`` and every reclaim path requires ``status='running'``, so the card sits
invisible forever — the dead zone. Cards t_bc9deca5 (~90 min stuck) and
t_b938d8df (stuck after 24 sec) hit this; both needed a manual
``reclaim_task`` to become visible again.

The fix widens ``detect_crashed_workers`` to also cover 'ready'/'review'
cards that still carry a stale claim + dead pid: the run is closed as
``crashed``, the claim is cleared, and the card returns to the dispatcher's
ready query.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kb.connect() as c:
        yield c


def _dead_pid() -> int:
    """Spawn + reap a child so its pid is guaranteed dead on this host."""
    dead = subprocess.Popen(["true"])
    dead.wait()
    return dead.pid


def test_dead_zone_ready_claimed_card_auto_reclaimed(conn, monkeypatch):
    """A card in 'ready' with a stale claim + dead pid + open run must be
    crash-reclaimed automatically — the dispatcher dead zone."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda pid: False)
    host = _kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="deadzone", assignee="w")

    # Claim opens a real run; then simulate the dead-zone state: the card
    # is flipped back to 'ready' while claim bookkeeping + open run remain.
    kb.claim_task(conn, tid, claimer=f"{host}:dead")
    kb._set_worker_pid(conn, tid, _dead_pid())
    conn.execute(
        "UPDATE tasks SET status='ready' WHERE id=? AND status='running'",
        (tid,),
    )
    conn.commit()

    # Reproduction: before the fix the card is INVISIBLE to the dispatcher —
    # the ready query requires claim_lock IS NULL.
    ready_rows = conn.execute(
        "SELECT id FROM tasks WHERE status='ready' AND claim_lock IS NULL"
    ).fetchall()
    assert tid not in [r["id"] for r in ready_rows], (
        "precondition: dead-zone card must be invisible to the ready query"
    )

    # The fix: the crash pass reclaims it.
    crashed = _kb.detect_crashed_workers(conn)
    assert tid in crashed

    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "ready"
    assert task.claim_lock is None
    assert task.claim_expires is None
    assert task.worker_pid is None
    assert task.current_run_id is None

    # The open run was closed as crashed.
    run = conn.execute(
        "SELECT outcome, status, ended_at FROM task_runs WHERE task_id=?",
        (tid,),
    ).fetchone()
    assert run["outcome"] == "crashed"
    assert run["status"] == "crashed"
    assert run["ended_at"] is not None

    # The card is visible to the dispatcher again.
    ready_rows = conn.execute(
        "SELECT id FROM tasks WHERE status='ready' AND claim_lock IS NULL"
    ).fetchall()
    assert tid in [r["id"] for r in ready_rows]


def test_dead_zone_review_claimed_card_auto_reclaimed(conn, monkeypatch):
    """A card in 'review' with a stale claim + dead pid is crash-reclaimed
    back to the review lane (not ready)."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda pid: False)
    host = _kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="deadzone-review", assignee="w")

    kb.claim_task(conn, tid, claimer=f"{host}:dead")
    kb._set_worker_pid(conn, tid, _dead_pid())
    # Mark the run as a review claim so _retry_status_for_run lands on review.
    claimed_event = conn.execute(
        "SELECT id FROM task_events WHERE task_id=? AND kind='claimed' "
        "ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()
    assert claimed_event is not None
    conn.execute(
        "UPDATE task_events SET payload=? WHERE id=?",
        ('{"source_status": "review"}', claimed_event["id"]),
    )
    conn.execute(
        "UPDATE tasks SET status='review' WHERE id=? AND status='running'",
        (tid,),
    )
    conn.commit()

    crashed = _kb.detect_crashed_workers(conn)
    assert tid in crashed

    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "review", (
        "review-lane dead-zone card must resume from review, "
        f"got {task.status}"
    )
    assert task.claim_lock is None
    assert task.worker_pid is None
    assert task.current_run_id is None


def test_dead_zone_live_pid_never_reclaimed(conn, monkeypatch):
    """A ready card with a stale claim but a LIVE pid must NOT be reclaimed —
    the worker may still be executing and will self-resolve via its own
    terminal kanban call."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda pid: True)
    host = _kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="live-claim", assignee="w")

    kb.claim_task(conn, tid, claimer=f"{host}:live")
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        kb._set_worker_pid(conn, tid, sleeper.pid)
        conn.execute(
            "UPDATE tasks SET status='ready' WHERE id=? AND status='running'",
            (tid,),
        )
        conn.commit()

        assert _kb.detect_crashed_workers(conn) == []

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"
        assert task.claim_lock is not None
        assert task.worker_pid == sleeper.pid
        run = conn.execute(
            "SELECT ended_at FROM task_runs WHERE task_id=?", (tid,),
        ).fetchone()
        assert run["ended_at"] is None, "live run must stay open"
    finally:
        sleeper.terminate()
        sleeper.wait()
