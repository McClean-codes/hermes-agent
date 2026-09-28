"""Two lifecycle invariants, using real SQLite and a local GitHub HTTP contract."""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_pr_acceptance as acceptance
from hermes_cli.kanban_db_connect import connect


PLAN_GATE = "Upgrade to GitHub Pro or make this repository public to enable this feature."


@pytest.fixture
def github(tmp_path, monkeypatch):
    state: dict[str, Any] = {"conclusion": "success", "head": "a" * 40, "reads": 0, "requests": [],
                             "private": False,
                             "branch_rule": {"requiredStatusChecks": [{"context": "required", "app": {"databaseId": 1}}]}}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            sha = state["head"]
            if self.path == "/graphql":
                value = {"data": {"repository": {"isPrivate": state["private"], "pullRequest": {
                    "headRefOid": sha, "baseRefName": "main", "state": "OPEN",
                    "baseRef": {"branchProtectionRule": state["branch_rule"]}}}}}
            elif "/rules/branches/" in self.path:
                if state.get("rules_error"):
                    self._fail(*state["rules_error"])
                    return
                value = [[]]
            elif "/check-runs" in self.path:
                run = {"id": 42, "name": "required", "head_sha": sha,
                       "app": {"id": 1}, "status": "in_progress" if state["conclusion"] == "pending" else "completed", "conclusion": state["conclusion"],
                       "html_url": "https://github.com/acme/repo/actions/runs/42"}
                if state.get("stale"):
                    run["head_sha"] = "b" * 40
                runs = [] if state.get("missing") else [run]
                value = [{"total_count": 100 + len(runs), "check_runs": [
                    {**run, "id": 1000 + i, "name": "optional", "conclusion": "skipped"}
                    for i in range(100)]}, {"total_count": 100 + len(runs), "check_runs": runs}]
                if state.get("race"):
                    state["race"]()
                if state.get("head_change"):
                    state["head"] = "b" * 40
            elif "/statuses" in self.path:
                value = [[]]
            elif "/pulls/" in self.path:
                value = {"head": {"sha": sha}, "base": {"ref": "main"}, "state": "open"}
            else:
                self._fail(404, "Not Found")
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def _fail(self, code, message):
            """Serve GitHub's JSON error body so the gh shim can mirror gh's streams."""
            body = json.dumps({"message": message,
                               "documentation_url": "https://docs.github.com/rest",
                               "status": str(code)}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    # gh prints the API error body on stdout and "gh: <message> (HTTP <code>)" on
    # stderr, exiting 1 — the collector classifies plan-gate 403s from both streams.
    gh.write_text(f"#!{sys.executable}\n"
                  "import json,sys,urllib.request,urllib.error\n"
                  f"u='http://127.0.0.1:{server.server_port}/'+sys.argv[2]\n"
                  "try:\n"
                  "    body=urllib.request.urlopen(u).read().decode()\n"
                  "except urllib.error.HTTPError as e:\n"
                  "    text=e.read().decode()\n"
                  "    print(text, end='')\n"
                  "    try:\n"
                  "        msg=json.loads(text).get('message', '')\n"
                  "    except ValueError:\n"
                  "        msg=''\n"
                  "    print('gh: %s (HTTP %s)' % (msg, e.code), file=sys.stderr)\n"
                  "    sys.exit(1)\n"
                  "print(body)\n", encoding="utf-8")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    # Sandbox the shared kanban root: the default resolves to the real ~/.hermes,
    # which the hermetic write guard refuses (the tmp root lives under ~/.hermes).
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "kanban"))
    kb.init_db()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _receipt(conn, tid):
    rows = [json.loads(r[0]) for r in conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,))]
    assert rows, "acceptance receipt was not recorded"
    return rows[-1]


def _status(conn, tid):
    task = kb.get_task(conn, tid)
    assert task is not None, "task is missing from the board"
    return task.status


@pytest.mark.linux_only
def test_pr_completion_requires_current_required_evidence(github):
    with connect() as conn:
        for conclusion in ("failure", "pending", "cancelled", "timed_out", "action_required", "neutral", "skipped", None, "success"):
            github.update(conclusion=conclusion, head="a" * 40)
            tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
            ok = kb.complete_task(conn, tid, metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert ok is (conclusion == "success")
            task = kb.get_task(conn, tid)
            assert (task.status == "done") is ok
            receipts = [json.loads(r[0]) for r in conn.execute(
                "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,))]
            assert receipts and receipts[-1]["head_sha"] == "a" * 40
            # The rulesets endpoint answered 200, so the receipt must say both sources were read.
            assert receipts[-1]["rules_source"] == "graphql_and_rest"
            if not ok:
                assert task.status in {"running", "ready", "blocked", "review"}
                assert "retry" in receipts[-1]["recovery"]
                assert receipts[-1]["checks"][0]["id"] == 42
        for fault in ("missing", "stale", "head_change"):
            github.update(conclusion="success", head="a" * 40)
            github[fault] = True
            tid = kb.create_task(conn, title=fault, completion_contract="acme/repo")
            assert not kb.complete_task(conn, tid, metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert kb.get_task(conn, tid).status != "done"
            github.pop(fault)
        # Omission and a sibling repository cannot downgrade the stored declaration.
        tid = kb.create_task(conn, title="publish", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, summary="local green")
        assert not kb.complete_task(conn, tid, metadata={"published_pr": "https://github.com/other/repo/pull/7"})
        before = len(github["requests"])
        local = kb.create_task(conn, title="local", completion_contract="local-only")
        assert kb.complete_task(conn, local, summary="https://github.com/acme/repo/pull/7 is background context")
        assert len(github["requests"]) == before


@pytest.mark.linux_only
def test_acceptance_receipts_and_terminal_write_share_run_ownership(github):
    with connect() as conn:
        for conclusion in ("success", "failure"):
            tid = kb.create_task(conn, title="race", completion_contract="acme/repo")
            owner = kb.claim_task(conn, tid)
            run_id = owner.current_run_id
            def reclaim():
                with connect() as rival:
                    assert kb.block_task(rival, tid, reason="Reassigned during acceptance")
                    assert kb.unblock_task(rival, tid)
                    github["replacement"] = kb.claim_task(rival, tid).current_run_id
            github.update(conclusion=conclusion, race=reclaim)
            assert not kb.complete_task(conn, tid, expected_run_id=run_id,
                metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert kb.get_task(conn, tid).current_run_id == github["replacement"]
            assert github["replacement"] != run_id
            assert kb.get_task(conn, tid).status != "done"
            assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0] == 0
            github.pop("race")


@pytest.mark.linux_only
def test_private_plan_gate_403_accepts_through_graphql_required_checks(github):
    """The single tolerated failure: private Free-plan repo, verbatim plan-gate 403."""
    github.update(private=True, rules_error=(403, PLAN_GATE))
    with connect() as conn:
        tid = kb.create_task(conn, title="publish", completion_contract="acme/repo")
        assert kb.complete_task(conn, tid, metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = _receipt(conn, tid)
        assert receipt["ok"] is True
        assert receipt["classification"] == "success"
        assert receipt["rules_source"] == "graphql_only"
        assert receipt["head_sha"] == "a" * 40
        assert receipt["required"] == [{"context": "required", "app_id": 1}]
        assert [(c["name"], c["classification"]) for c in receipt["checks"]] == [("required", "success")]
        # REST was still attempted; only its plan-gate answer was tolerated.
        assert any("/rules/branches/" in path for path in github["requests"])
        assert _status(conn, tid) == "done"


@pytest.mark.linux_only
def test_public_repository_plan_gate_body_does_not_fallback(github):
    """The same 403 on a confirmed public repository must stay an infrastructure error."""
    github.update(private=False, rules_error=(403, PLAN_GATE))
    with connect() as conn:
        tid = kb.create_task(conn, title="publish", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = _receipt(conn, tid)
        assert receipt["ok"] is False
        assert receipt["classification"] == "infra"
        assert "rules_source" not in receipt
        assert receipt["checks"] == []
        assert _status(conn, tid) != "done"


@pytest.mark.linux_only
def test_non_plan_rules_failures_stay_infrastructure_errors(github):
    """A different body, an auth failure or a 5xx on the rules call never falls back."""
    github.update(private=True)
    with connect() as conn:
        for code, message in ((403, "Resource not accessible by integration"),
                              (401, "Bad credentials"),
                              (404, "Not Found"),
                              (500, "Server Error")):
            github.update(rules_error=(code, message))
            tid = kb.create_task(conn, title=f"rules-{code}", completion_contract="acme/repo")
            assert not kb.complete_task(conn, tid, metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            receipt = _receipt(conn, tid)
            assert receipt["classification"] == "infra", (code, receipt)
            assert "rules_source" not in receipt
            assert receipt["checks"] == []
            assert _status(conn, tid) != "done"


@pytest.mark.linux_only
def test_plan_gate_fallback_requires_unambiguous_graphql_requirements(github):
    """With REST unreadable, a missing/empty/absent GraphQL required set fails closed."""
    github.update(private=True, rules_error=(403, PLAN_GATE))
    with connect() as conn:
        for label, rule in (("no-rule", None),
                            ("empty-checks", {"requiredStatusChecks": []}),
                            ("null-checks", {"requiredStatusChecks": None})):
            github.update(branch_rule=rule)
            tid = kb.create_task(conn, title=label, completion_contract="acme/repo")
            assert not kb.complete_task(conn, tid, metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            receipt = _receipt(conn, tid)
            assert receipt["classification"] == "infra", (label, receipt)
            assert "rules_source" not in receipt
            assert receipt["checks"] == []
            assert _status(conn, tid) != "done"
        github.update(branch_rule={"requiredStatusChecks": [{"context": "required", "app": {"databaseId": 1}}]})


@pytest.mark.linux_only
def test_plan_gate_fallback_still_rejects_missing_or_failing_checks(github):
    """GraphQL-only rules evidence does not weaken the exact-head check gate."""
    github.update(private=True, rules_error=(403, PLAN_GATE))
    with connect() as conn:
        for fault, expected in (({"missing": True}, "missing"),
                                ({"conclusion": "failure"}, "failure"),
                                ({"conclusion": "pending"}, "pending")):
            github.update(conclusion="success", head="a" * 40)
            github.pop("missing", None)
            github.update(fault)
            tid = kb.create_task(conn, title=next(iter(fault)), completion_contract="acme/repo")
            assert not kb.complete_task(conn, tid, metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            receipt = _receipt(conn, tid)
            assert receipt["classification"] == expected, (fault, receipt)
            assert receipt["rules_source"] == "graphql_only"
            assert _status(conn, tid) != "done"
        github.pop("missing", None)
        github.update(conclusion="success")


def _gate_body(message: str = PLAN_GATE, status: str | None = "403"):
    """The exact stream ``gh api`` prints for a rulesets error."""
    return json.dumps({"message": message,
                       "documentation_url": "https://docs.github.com/rest",
                       **({"status": status} if status is not None else {})})


def _gate_error(stdout=None, stderr=""):
    return subprocess.CalledProcessError(1, ["gh", "api", "repos/acme/repo/rules/branches/main"],
                                         output=stdout, stderr=stderr)


@pytest.mark.linux_only
def test_plan_gate_predicate_requires_the_exact_message():
    """The body is parsed, so only a verbatim ``message`` identifies the plan gate."""
    stderr = f"gh: {PLAN_GATE} (HTTP 403)"
    assert acceptance._is_plan_gate(_gate_error(_gate_body(), stderr)) is True
    assert acceptance._is_plan_gate(_gate_error(_gate_body(), "")) is True
    altered = (f"prefix {PLAN_GATE}", f"{PLAN_GATE} suffix", f"prefix {PLAN_GATE} suffix",
               PLAN_GATE[:-1], PLAN_GATE + " ", " " + PLAN_GATE,
               PLAN_GATE.replace(" ", "  "), PLAN_GATE.replace(".", ""),
               PLAN_GATE.upper(), PLAN_GATE.lower().replace("github", "GitHub", 1),
               json.dumps({"message": PLAN_GATE, "documentation_url": "x", "status": "403"}) * 2)
    for message in altered:
        assert acceptance._is_plan_gate(_gate_error(_gate_body(message), stderr)) is False, message
        assert acceptance._is_plan_gate(_gate_error(_gate_body(message), "")) is False, message


@pytest.mark.linux_only
def test_plan_gate_predicate_fails_closed_on_unparsable_or_missing_message():
    """No body, malformed JSON, a non-object body or a missing message never match."""
    for stdout in (None, "", "   ", "Upgrade to GitHub Pro or make this repository public "
                   "to enable this feature.", "{not json", '{"message":', b"{}",
                   json.dumps([PLAN_GATE]), json.dumps(PLAN_GATE), json.dumps(None),
                   json.dumps({"status": "403"}), json.dumps({"message": 403, "status": "403"}),
                   json.dumps({"message": None, "status": "403"})):
        for stderr in ("", f"gh: {PLAN_GATE} (HTTP 403)", "gh: HTTP 403: " + PLAN_GATE):
            assert acceptance._is_plan_gate(_gate_error(stdout, stderr)) is False, (stdout, stderr)


@pytest.mark.linux_only
def test_plan_gate_predicate_requires_explicit_403_evidence():
    """An exact message on a non-403 response, or with no status at all, stays a failure."""
    exact_no_status = json.dumps({"message": PLAN_GATE})
    assert acceptance._is_plan_gate(_gate_error(exact_no_status, f"gh: {PLAN_GATE} (HTTP 403)")) is True
    assert acceptance._is_plan_gate(_gate_error(exact_no_status,
                                                f"gh: HTTP 403: {PLAN_GATE} (https://docs.github.com)")) is True
    assert acceptance._is_plan_gate(_gate_error(exact_no_status)) is False
    assert acceptance._is_plan_gate(_gate_error(exact_no_status, f"gh: {PLAN_GATE} (HTTP 404)")) is False
    assert acceptance._is_plan_gate(_gate_error(exact_no_status, "HTTP 4030")) is False
    assert acceptance._is_plan_gate(_gate_error(_gate_body(status="404"),
                                                f"gh: {PLAN_GATE} (HTTP 404)")) is False
    assert acceptance._is_plan_gate(_gate_error(_gate_body(status=None),
                                                f"gh: {PLAN_GATE} (HTTP 401)")) is False


@pytest.mark.linux_only
def test_altered_plan_gate_message_never_falls_back(github):
    """A 403 whose message merely contains the plan-gate text is an infrastructure error.

    Prefixing, suffixing or rewording the message must not reach the
    GraphQL-only acceptance path, so no ``rules_source`` is ever recorded.
    """
    github.update(private=True)
    with connect() as conn:
        for label, message in (("prefixed", f"prefix {PLAN_GATE}"),
                               ("suffixed", f"{PLAN_GATE} suffix"),
                               ("wrapped", f"prefix {PLAN_GATE} suffix"),
                               ("period", PLAN_GATE[:-1]),
                               ("case", PLAN_GATE.upper())):
            github.update(rules_error=(403, message))
            tid = kb.create_task(conn, title=f"gate-{label}", completion_contract="acme/repo")
            assert not kb.complete_task(conn, tid,
                                        metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            receipt = _receipt(conn, tid)
            assert receipt["ok"] is False, (label, receipt)
            assert receipt["classification"] == "infra", (label, receipt)
            assert "rules_source" not in receipt, (label, receipt)
            assert receipt["checks"] == [], (label, receipt)
            assert _status(conn, tid) != "done", (label, receipt)
