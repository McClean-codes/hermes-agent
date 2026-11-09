"""Kanban workers are finite ``chat -q`` turns but must keep automatic background review,
and the one-shot exit must wait for that review instead of killing its daemon thread."""
import threading
from types import SimpleNamespace

import pytest

from agent.background_review import load_background_review_settings
from gateway.session_finite import finite_turn_scope
from hermes_cli import cli_shutdown


@pytest.fixture(autouse=True)
def _no_safe_worker(monkeypatch):
    monkeypatch.setattr("agent.safe_worker_policy.safe_worker_enabled", lambda: False)


def test_finite_turn_without_kanban_env_stays_disabled(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    with finite_turn_scope(True):
        assert load_background_review_settings()[0] is False


def test_finite_turn_with_kanban_env_is_enabled(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    with finite_turn_scope(True):
        assert load_background_review_settings()[0] is True


def test_non_finite_turn_unchanged(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    with finite_turn_scope(False):
        assert load_background_review_settings()[0] is True


def _cli_with_run(run):
    agent = SimpleNamespace(_background_review_run=run, session_id="s1")
    return SimpleNamespace(agent=agent, session_id="s1")


def test_wait_is_noop_outside_kanban(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    run = SimpleNamespace(request_done=threading.Event())  # never set: would block if waited
    monkeypatch.setattr(cli_shutdown, "_KANBAN_REVIEW_WAIT_SECONDS", 5.0)
    cli_shutdown._wait_for_background_review(_cli_with_run(run))  # returns immediately


def test_wait_returns_when_review_finishes(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    run = SimpleNamespace(request_done=threading.Event())
    threading.Timer(0.05, run.request_done.set).start()
    monkeypatch.setattr(cli_shutdown, "_KANBAN_REVIEW_WAIT_SECONDS", 5.0)
    cli_shutdown._wait_for_background_review(_cli_with_run(run))
    assert run.request_done.is_set()


def test_wait_is_bounded_when_review_hangs(monkeypatch, caplog):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    run = SimpleNamespace(request_done=threading.Event())
    monkeypatch.setattr(cli_shutdown, "_KANBAN_REVIEW_WAIT_SECONDS", 0.1)
    with caplog.at_level("WARNING"):
        cli_shutdown._wait_for_background_review(_cli_with_run(run))
    assert not run.request_done.is_set()
    assert "still running" in caplog.text


def test_wait_without_a_review_run_is_noop(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    monkeypatch.setattr(cli_shutdown, "_KANBAN_REVIEW_WAIT_SECONDS", 5.0)
    cli_shutdown._wait_for_background_review(_cli_with_run(None))
