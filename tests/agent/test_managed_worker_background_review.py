"""A Kanban managed-worker turn keeps the native post-turn background review, and the worker waits
(bounded) for a running review before the turn is torn down.

Regression: ``agent.managed_worker`` hard-coded ``skip_background_review=True`` for every turn, so no
Kanban card ever ran the review. Ordinary (non-Kanban) managed turns keep the skip.
"""
import json
import logging
import threading
from types import SimpleNamespace

from tests.gateway.test_managed_turn_contract import assignment

KANBAN_CONTEXT = {
    'task_id': 't_test', 'board': 'hermes-agent', 'db': '/nonexistent/kanban.db', 'run_id': 1,
    'claim_lock': 'lock', 'goal_mode': False, 'max_runtime_seconds': None, 'branch': None,
    'tenant': None, 'profile': 'fixture', 'workspaces_root': '/tmp', 'skills': [], 'context': '',
    'goal_text': '', 'goal_max_turns': None,
}


def _execute(tmp_path, monkeypatch, *, kanban, run_turns, events):
    """Drive ``managed_worker.execute`` through a real bootstrap frame. Only the agent, the session
    store, the owner controls and the kanban DB binding are stubbed; ``retire_agent`` runs for real."""
    from agent import managed_worker as worker
    frame, _ = assignment(tmp_path, monkeypatch)
    if kanban:
        context = {**KANBAN_CONTEXT, 'workspace': str(tmp_path)}
        # build_kanban_policy: source='kanban' (bind_worker_context checks it), platform='cli'.
        frame = {**frame, 'policy': {**frame['policy'], 'kanban_json': json.dumps(context),
                                     'source': 'kanban', 'platform': 'cli'}}
    built = {}

    class Agent:
        def __init__(self, **kwargs):
            built.update(kwargs)
            self.session_prompt_tokens = self.session_completion_tokens = 0
        def shutdown_memory_provider(self, messages=None): pass
        def release_clients(self): pass

    class Store:
        def __init__(self, *args): pass
        def get_messages_as_conversation(self, sid): return []
        def flush_token_counts(self): events.append(('flush', None))
        def finish(self): pass
        def close(self): pass

    class Controls:
        def __init__(self, *args):
            self.stopped, self.finish = threading.Event(), threading.Event()
            self.finish.set()
        def approval(self, data): pass
        def clarify(self, questions): pass

    monkeypatch.setattr(worker, 'bind_worker_policy', lambda frame: None)
    monkeypatch.setattr(worker, 'discover_profile_mcp', lambda policy: None)
    monkeypatch.setattr(worker, 'WorkerControls', Controls)
    monkeypatch.setattr('agent.runtime_session_store.WorkerRPC', lambda home: lambda *a, **k: {'owner_epoch': 1})
    monkeypatch.setattr('agent.runtime_session_store.RuntimeSessionStore', Store)
    monkeypatch.setattr('tools.process_registry.process_registry.recover_from_checkpoint', lambda: None)
    monkeypatch.setattr('run_agent.AIAgent', Agent)
    monkeypatch.setattr('agent.title_generator.wait_for_title_upgrades', lambda: None)
    monkeypatch.setattr('gateway.session_kanban.bind_worker_context', lambda frame: None)
    monkeypatch.setattr('gateway.session_kanban.run_worker_turns',
                        lambda agent, frame, history: run_turns(agent))
    if kanban:
        # The production binder exports HERMES_KANBAN_TASK; it needs a live claim row, so this unit
        # stubs the binder and sets the one env the gates read, exactly as the binder does.
        monkeypatch.setenv('HERMES_KANBAN_TASK', KANBAN_CONTEXT['task_id'])
    else:
        monkeypatch.delenv('HERMES_KANBAN_TASK', raising=False)
    worker.execute(frame, SimpleNamespace(
        send=lambda kind, **payload: events.append((kind, payload))))
    return built


def test_kanban_turn_builds_agent_with_background_review_enabled(tmp_path, monkeypatch):
    built = _execute(tmp_path, monkeypatch, kanban=True, events=[],
                     run_turns=lambda agent: {'final_response': 'ok', 'completed': True})
    assert built['skip_background_review'] is False


def test_ordinary_managed_turn_keeps_background_review_skipped(tmp_path, monkeypatch):
    built = _execute(tmp_path, monkeypatch, kanban=False, events=[],
                     run_turns=lambda agent: {'final_response': 'ok', 'completed': True})
    assert built['skip_background_review'] is True


def test_kanban_turn_waits_for_background_review_before_reporting_result(tmp_path, monkeypatch):
    from agent import background_review as br
    events = []
    run = br._BackgroundReviewRun()

    def run_turns(agent):
        agent._background_review_run = run
        # The review finishes while the worker is about to tear down: the wait must observe it.
        threading.Timer(0.05, br.finish_background_review_run, args=(agent, run)).start()
        return {'final_response': 'ok', 'completed': True}

    _execute(tmp_path, monkeypatch, kanban=True, run_turns=run_turns, events=events)
    assert run.request_done.is_set()
    kinds = [kind for kind, _ in events]
    assert kinds.index('flush') < kinds.index('result') and 'finished' in kinds


def test_hung_background_review_is_bounded_logged_and_the_turn_still_reports(tmp_path, monkeypatch, caplog):
    from agent import background_review as br
    # retire_agent reads the bound name at call time, so patch the module the wait actually uses.
    monkeypatch.setattr(br, 'REVIEW_TEARDOWN_WAIT_SECONDS', 0.05)
    events = []
    run = br._BackgroundReviewRun()  # never finished: a stuck review

    def run_turns(agent):
        agent._background_review_run = run
        return {'final_response': 'ok', 'completed': True}

    with caplog.at_level(logging.WARNING):
        _execute(tmp_path, monkeypatch, kanban=True, run_turns=run_turns, events=events)
    kinds = [kind for kind, _ in events]
    assert 'result' in kinds and 'finished' in kinds
    assert not run.request_done.is_set()
    assert 'still running' in caplog.text


def test_kanban_turn_keeps_memory_enabled_so_review_writes_reach_the_profile_store(tmp_path, monkeypatch):
    # Item 4: a Kanban turn is not ignore_rules, so the parent owns the profile memory store and the
    # review fork (skip_memory=True) shares it through ``review_agent._memory_store`` rather than
    # opening a second, unscoped one.
    built = _execute(tmp_path, monkeypatch, kanban=True, events=[],
                     run_turns=lambda agent: {'final_response': 'ok', 'completed': True})
    assert built['skip_memory'] is False


def test_review_ending_early_for_a_provider_without_tool_calls_still_completes_its_run(monkeypatch):
    # Regression: this early return never published request completion, so the run stayed open and
    # any wait on it (the teardown wait) stalled for its full timeout.
    from agent import background_review as br
    run = br.prepare_background_review_run(SimpleNamespace())
    agent = SimpleNamespace(client=SimpleNamespace(SUPPORTS_HERMES_TOOL_CALLS=False), provider='x',
                            _background_review_run=run, _background_review_lock=threading.Lock())
    monkeypatch.setattr(br, '_resolve_review_runtime', lambda agent, cfg: {'routed': False})
    br._run_review_in_thread(agent, [], 'prompt', task_cfg={}, review_run=run)
    assert run.request_done.is_set()


def test_kanban_review_is_not_deferred_to_the_process_local_idle_queue(monkeypatch):
    # Regression: the idle queue is process-local, so a deferred review on a one-shot Kanban worker
    # would be dropped at exit. Kanban reviews spawn now; other sessions keep the deferral.
    import agent.review_idle_queue as riq
    from run_agent import _review_should_defer
    monkeypatch.setattr(riq, 'review_targets_managed_local', lambda agent, cfg: True)
    monkeypatch.setenv('HERMES_KANBAN_TASK', 't_test')
    assert _review_should_defer(object(), {}) is False
    monkeypatch.delenv('HERMES_KANBAN_TASK', raising=False)
    assert _review_should_defer(object(), {}) is True


def test_wait_for_background_review_is_a_noop_without_a_review():
    from agent.background_review import wait_for_background_review
    assert wait_for_background_review(SimpleNamespace(_background_review_run=None), timeout=5) is True


def test_wait_for_background_review_returns_false_and_logs_on_timeout(caplog):
    from agent.background_review import wait_for_background_review
    run = SimpleNamespace(request_done=threading.Event())  # never set
    with caplog.at_level(logging.WARNING, logger='agent.background_review'):
        assert wait_for_background_review(SimpleNamespace(_background_review_run=run), timeout=0.01) is False
    assert 'still running' in caplog.text
