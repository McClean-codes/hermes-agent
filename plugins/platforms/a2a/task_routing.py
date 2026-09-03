"""A2A task routing, lifecycle, and RPC protocol handlers.

Extracted from ``adapter.py`` to keep the main adapter module focused on
HTTP transport, server infrastructure, and adapter lifecycle.

Uses a mixin pattern: ``TaskRPCHandler`` is a base class whose methods
are mixed into ``A2AAdapter`` via multiple inheritance.  This preserves
the original ``self.*`` attribute access without adapter-passing overhead.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Optional

from gateway.platforms.base import MessageEvent, ProcessingOutcome
from . import protocol, security

logger = logging.getLogger(__name__)

# Keepalive and patience constants matching adapter.py
_SSE_KEEPALIVE = 15
_PATIENCE_MARGIN = 30


def _reply_timeout() -> float:
    """Seconds to wait for the agent to answer an inbound task."""
    import os
    try:
        return max(1.0, float(os.getenv("A2A_REPLY_TIMEOUT", "300")))
    except (ValueError, TypeError):
        return 300.0


class TaskRPCHandler:
    """Mixin providing A2A task lifecycle and JSON-RPC protocol handling.

    Must be mixed into a class that provides (at minimum):
        self.tasks, self._scope_for_agent(), self._agents,
        self._pending, self._pending_lock, self._pending_order,
        self._loop, self._gateway_ready, self._turns,
        self._context_peers, self._context_peers_lock,
        self._context_sessions, self._security_context,
        self._pop_pending(), self._add_pending(), self._resolve_task(),
        self._register_context_peer(), self._refine_peer_identity(),
        self._is_duplicate_inbound(), self._find_existing_nonterminal_task(),
        self._register_inline_push(), self._forward_to_profile(),
        self._wake_origin_session(), self._send_push_notification(),
        self.build_source(), self.handle_message().
    """

    # ── Inbound task handling ─────────────────────────────────────────────

    def _prepare_task_rpc(self, params: dict, peer: str, agent=None):
        """Validate, register, and dispatch an inbound message.

        Returns ``(terminal_task, None)`` when the task ends immediately,
        else ``(None, pending)``.
        """
        # Delegate to the adapter's existing _prepare_task which handles
        # session vars, context peer persistence, profile forwarding, etc.
        return self._prepare_task(params, peer, agent=agent)

    # ── JSON-RPC: message/send ───────────────────────────────────────────

    def _rpc_message_send(self, req_id, params: dict, peer: str,
                          agent=None, v1_response: bool = False,
                          client_alive=None):
        """Handle one blocking message/send JSON-RPC request."""
        terminal, pending = self._prepare_task(params, peer, agent=agent)
        if terminal is not None:
            result = protocol.send_message_response(terminal) if v1_response else terminal
            return protocol.jsonrpc_result(req_id, result)
        assert pending is not None
        pat = self._patience_for(params, pending["peer"])
        if client_alive is not None:
            def _probe():
                if not client_alive():
                    raise ConnectionResetError("A2A client disconnected while awaiting reply")
            state, reply, out_of_band_only, defer_finalization = self._await_reply(
                pending, keepalive=_probe, patience=pat)
        else:
            state, reply, out_of_band_only, defer_finalization = self._await_reply(
                pending, patience=pat)
        if defer_finalization:
            logger.info(
                "A2A: client disconnected for task %s; deferring finalization", pending["task_id"],
            )
            return protocol.jsonrpc_result(
                req_id, {"error": {"code": -32000, "message": "client disconnected, task pending"}}
            )
        state, reply = self._finalize_task(pending, state, reply)
        if out_of_band_only:
            if state in (protocol.STATE_COMPLETED, protocol.STATE_INPUT_REQUIRED) and reply:
                if self._try_push_reply(pending, state, reply):
                    return None
        task = protocol.build_task(
            pending["task_id"], pending["context_id"], state, reply,
            created_at=pending["created_iso"],
        )
        result = protocol.send_message_response(task) if v1_response else task
        return protocol.jsonrpc_result(req_id, result)

    # ── JSON-RPC: message/stream (SSE) ──────────────────────────────────

    @staticmethod
    def _sse_headers(handler):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        handler.close_connection = True

    @staticmethod
    def _sse_write(handler, chunk: str):
        handler.wfile.write(chunk.encode("utf-8"))
        handler.wfile.flush()

    def _emit_terminal(self, handler, task_id: str, context_id: str,
                       state: str, reply: str, req_id=None):
        if reply and state == protocol.STATE_COMPLETED:
            self._sse_write(handler, protocol.sse_data(
                protocol.artifact_update(task_id, context_id, reply), req_id))
            self._sse_write(handler, protocol.sse_data(
                protocol.status_update(task_id, context_id, state), req_id))
        else:
            self._sse_write(handler, protocol.sse_data(
                protocol.status_update(task_id, context_id, state, reply), req_id))
        self._sse_write(handler, protocol.sse_done())

    def _rpc_message_stream(self, handler, req_id, params: dict,
                            peer: str, agent=None):
        protocol.metrics.streams_started += 1
        self._sse_headers(handler)
        try:
            terminal, pending = self._prepare_task(params, peer, agent=agent)
            if terminal is not None:
                self._emit_terminal(
                    handler, terminal["id"], terminal["contextId"],
                    terminal["status"]["state"],
                    protocol.extract_text(terminal.get("status", {}).get("message", {}) or {}),
                    req_id=req_id,
                )
                return
            assert pending is not None
            task_id, context_id = pending["task_id"], pending["context_id"]
            self._sse_write(handler, protocol.sse_data(protocol.stream_task(
                protocol.build_task(task_id, context_id, protocol.STATE_SUBMITTED, created_at=pending["created_iso"])),
                req_id))
            self._sse_write(handler, protocol.sse_data(
                protocol.status_update(task_id, context_id, protocol.STATE_WORKING), req_id))

            state, reply, _, defer_finalization = self._await_reply(
                pending, keepalive=lambda: self._sse_write(handler, ": keepalive\n\n"),
                patience=None,
            )
            if defer_finalization:
                self._emit_terminal(
                    handler, task_id, context_id,
                    protocol.STATE_WORKING, "[client disconnected, task pending]",
                    req_id=req_id,
                )
                return
            state, reply = self._finalize_task(pending, state, reply)
            self._emit_terminal(handler, task_id, context_id, state, reply, req_id=req_id)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("A2A: stream client disconnected")

    # ── JSON-RPC: tasks/subscribe ────────────────────────────────────────

    def _rpc_tasks_subscribe(self, handler, req_id, params: dict, agent=None):
        task_id = str(params.get("taskId") or params.get("id") or "")
        rec = self.tasks.get(task_id, *self._scope_for_agent(agent))
        if not rec:
            handler._json(200, protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND, f"task not found: {task_id}"))
            return
        self._sse_headers(handler)
        try:
            fut = self.tasks.watch(task_id, *self._scope_for_agent(agent))
            if fut is None:
                self._sse_write(handler, protocol.sse_done())
                return
            deadline = time.time() + _reply_timeout()
            while True:
                try:
                    state, reply = fut.result(timeout=_SSE_KEEPALIVE)
                    break
                except FuturesTimeout:
                    if time.time() >= deadline:
                        state, reply = rec["state"], rec.get("reply", "")
                        break
                    self._sse_write(handler, ": keepalive\n\n")
            self._emit_terminal(handler, task_id, rec["context_id"], state, reply, req_id=req_id)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("A2A: subscribe client disconnected")

    # ── JSON-RPC: tasks/get ──────────────────────────────────────────────

    def _rpc_tasks_get(self, req_id, params: dict, agent=None) -> dict:
        task_id = str(params.get("taskId") or params.get("id") or "")
        rec = self.tasks.get(task_id, *self._scope_for_agent(agent))
        if not rec:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND, f"task not found: {task_id}")
        history_len = params.get("historyLength")
        try:
            history_len = int(history_len) if history_len is not None else None
        except (TypeError, ValueError):
            history_len = None
        return protocol.jsonrpc_result(req_id, protocol.TaskStore.to_task(rec, history_length=history_len))

    # ── JSON-RPC: tasks/list ─────────────────────────────────────────────

    def _rpc_tasks_list(self, req_id, params: dict, agent=None) -> dict:
        try:
            offset = int(params.get("pageToken") or 0)
        except (ValueError, TypeError):
            offset = 0
        try:
            page_size = int(params.get("pageSize") or 50)
        except (ValueError, TypeError):
            page_size = 50
        agent_slug, tenant = self._scope_for_agent(agent)
        recs, next_offset, total = self.tasks.list(
            context_id=str(params.get("contextId") or ""),
            state=str(params.get("status") or params.get("state") or ""),
            page_size=page_size,
            offset=max(0, offset),
            agent_slug=agent_slug,
            tenant=tenant,
            with_total=True,
        )
        include_artifacts = bool(params.get("includeArtifacts", False))
        history_len = params.get("historyLength")
        try:
            history_len = int(history_len) if history_len is not None else None
        except (TypeError, ValueError):
            history_len = None
        return protocol.jsonrpc_result(req_id, {
            "tasks": [protocol.TaskStore.to_task(r, history_length=history_len, include_artifacts=include_artifacts) for r in recs],
            "nextPageToken": str(next_offset) if next_offset else "",
            "pageSize": max(1, min(page_size, 100)),
            "totalSize": total,
        })

    # ── JSON-RPC: tasks/cancel ───────────────────────────────────────────

    def _rpc_tasks_cancel(self, req_id, params: dict, agent=None) -> dict:
        task_id = str(params.get("taskId") or params.get("id") or "")
        rec = self.tasks.get(task_id, *self._scope_for_agent(agent))
        if not rec:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND, f"task not found: {task_id}")
        if rec["state"] in protocol.TERMINAL_STATES:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_CANCELABLE,
                f"task {task_id} already {rec['state']}")
        self.tasks.complete(task_id, protocol.STATE_CANCELED, "")
        try:
            from .a2a_persistence import _try_persist_task_ledger, _task_ledger_path
            ok = _try_persist_task_ledger(self.tasks, _task_ledger_path(), f"CANCELED {task_id}")
        except Exception:
            logger.error("A2A: failed to persist task ledger at CANCELED for task %s", task_id, exc_info=True)
            ok = False
        if not ok:
            logger.error("A2A: failed to persist CANCELED state for task %s — returning indeterminate failure", task_id)
            self._turns.reset(rec["context_id"])
            # Fail-closed: do not claim durable cancellation when disk write failed.
            return protocol.jsonrpc_error(req_id, -32000, f"task {task_id} cancellation persistence failed — state indeterminate")
        self._turns.reset(rec["context_id"])
        self._resolve_task(task_id, protocol.STATE_CANCELED, "")
        rec = self.tasks.get(task_id, *self._scope_for_agent(agent)) or rec
        return protocol.jsonrpc_result(req_id, protocol.TaskStore.to_task(rec))

    # ── JSON-RPC: push notification config ───────────────────────────────

    def _register_inline_push(self, task_id: str, params: dict, agent=None):
        cfg = (params.get("configuration") or {}).get("taskPushNotificationConfig") or {}
        if not isinstance(cfg, dict):
            return
        url = cfg.get("url") or (cfg.get("pushNotificationConfig") or {}).get("url") or ""
        if url:
            self.tasks.set_push_config(task_id, str(url), *self._scope_for_agent(agent))

    def _rpc_push_config_create(self, req_id, params: dict, agent=None) -> dict:
        task_id = str(params.get("taskId") or "")
        cfg = params.get("pushNotificationConfig") or params.get("config") or {}
        url = str((cfg or {}).get("url") or "")
        if not task_id or not url:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_INVALID_PARAMS,
                "taskId and pushNotificationConfig.url required")
        stored = self.tasks.set_push_config(task_id, url, *self._scope_for_agent(agent))
        if stored is None:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND, f"task not found: {task_id}")
        return protocol.jsonrpc_result(req_id, stored)

    def _rpc_push_config_get(self, req_id, params: dict, agent=None) -> dict:
        task_id = str(params.get("taskId") or "")
        config_id = str(params.get("id") or params.get("configId") or "")
        if not task_id:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_INVALID_PARAMS, "taskId required")
        cfg = self.tasks.get_push_config(task_id, config_id, *self._scope_for_agent(agent))
        if cfg is None:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND,
                f"push config not found for task: {task_id}")
        return protocol.jsonrpc_result(req_id, cfg)

    def _rpc_push_config_list(self, req_id, params: dict, agent=None) -> dict:
        task_id = str(params.get("taskId") or "")
        if not task_id:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_INVALID_PARAMS, "taskId required")
        configs = self.tasks.list_push_configs(task_id, *self._scope_for_agent(agent))
        return protocol.jsonrpc_result(req_id, {"configs": configs, "nextPageToken": ""})

    def _rpc_push_config_delete(self, req_id, params: dict, agent=None) -> dict:
        task_id = str(params.get("taskId") or "")
        config_id = str(params.get("id") or params.get("configId") or "")
        if not task_id:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_INVALID_PARAMS, "taskId required")
        deleted = self.tasks.delete_push_config(task_id, config_id, *self._scope_for_agent(agent))
        if not deleted:
            return protocol.jsonrpc_error(
                req_id, protocol.ERR_TASK_NOT_FOUND,
                f"push config not found for task: {task_id}")
        return protocol.jsonrpc_result(req_id, {"deleted": True})


    def _finalize_task(self, pending: dict, state: str, reply: str,
                       audit_direction: str = "outbound") -> tuple[str, str]:
        """Record the outcome of a dispatched task. Returns (state, reply) after
        redaction and input-required detection.

        ``audit_direction`` overrides the security audit direction (default
        ``"outbound"``).  Fire-and-forget loopback pushes use ``"push"``.
        """
        task_id = pending["task_id"]
        context_id = pending["context_id"]
        peer = pending["peer"]
        self._pop_pending(task_id)

        reply = security.redact_outbound(reply or "")

        # The agent flags clarification requests with a leading marker; map
        # them to the A2A input-required state so the peer knows to answer.
        if state == protocol.STATE_COMPLETED:
            stripped = reply.lstrip()
            if stripped.upper().startswith(protocol.INPUT_REQUIRED_MARKER):
                state = protocol.STATE_INPUT_REQUIRED
                reply = stripped[len(protocol.INPUT_REQUIRED_MARKER):].strip()

        protocol.persist_message(context_id, "agent", reply, task_id)
        security.audit(audit_direction, peer, task_id, reply, context_id=context_id)

        if state in (protocol.STATE_COMPLETED, protocol.STATE_INPUT_REQUIRED):
            protocol.metrics.outbound_total += 1
            protocol.metrics.tasks_completed += 1
            protocol.metrics.record_latency(time.time() - pending["started"])
        else:
            protocol.metrics.tasks_failed += 1

        self.tasks.complete(task_id, state, reply)
        # Persist the task ledger so GetTask/ListTasks/SubscribeToTask survive a gateway restart.
        # Fail-closed: if durable write fails, caller must not receive a successful terminal result.
        from .a2a_persistence import _try_persist_task_ledger, _task_ledger_path
        ok = _try_persist_task_ledger(self.tasks, _task_ledger_path(), f"terminal {state} {task_id}")
        if not ok:
            logger.error("A2A: failed to persist terminal %s for task %s — returning indeterminate failure", state, task_id)
            # Force memory to FAILED indeterminate so GetTask does not claim durable success.
            # Complete is terminal-idempotent, so we must mutate directly when already terminal.
            try:
                with self.tasks._lock:
                    rec = self.tasks._tasks.get(task_id)
                    if rec is not None:
                        rec["state"] = protocol.STATE_FAILED
                        rec["reply"] = "[terminal persistence failed — state indeterminate]"
                        rec["completed_at"] = rec.get("completed_at") or __import__("time").time()
                # Best-effort second attempt to persist the failure state (may also fail)
                _try_persist_task_ledger(self.tasks, _task_ledger_path(), f"FAILED after terminal persist fail {task_id}")
            except Exception:
                pass
            protocol.metrics.tasks_failed += 1
            return protocol.STATE_FAILED, "[terminal persistence failed — state indeterminate]"
        self._send_push_notification(task_id, context_id, reply, state)
        return state, reply


    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """Resolve the task future when processing ends without a reply send.

        The success path resolves via send(); this hook catches failures,
        cancellations, and empty runs so the HTTP thread returns promptly
        instead of waiting out the reply timeout. For deferred disconnects
        (HTTP waiter already gone), failure/cancellation must immediately
        terminalize the TaskStore record and durably persist — otherwise
        GetTask/SubscribeToTask remain stale WORKING. Success keeps the
        deferred WORKING semantics so a late send() can still finalize the
        original task.
        """
        task_id = str(getattr(event, "message_id", "") or "")
        if not task_id:
            return
        if outcome == ProcessingOutcome.FAILURE:
            self._resolve_task(task_id, protocol.STATE_FAILED, "[agent processing failed]")
            # Deferred failure: if TaskStore still non-terminal (HTTP waiter
            # disconnected, no one will call _finalize_task), terminalize now.
            rec = self.tasks.get(task_id)
            if rec and rec["state"] not in protocol.TERMINAL_STATES:
                self.tasks.complete(task_id, protocol.STATE_FAILED, "[agent processing failed]")
                from .a2a_persistence import _try_persist_task_ledger, _task_ledger_path
                ok = _try_persist_task_ledger(self.tasks, _task_ledger_path(), f"FAILED on_processing_complete {task_id}")
                if not ok:
                    logger.error("A2A: failed to persist FAILED state for task %s after disconnect — state indeterminate", task_id)
                protocol.metrics.tasks_failed += 1
                try:
                    self._send_push_notification(task_id, rec["context_id"], "[agent processing failed]", protocol.STATE_FAILED)
                except Exception:
                    pass
        elif outcome == ProcessingOutcome.CANCELLED:
            self._resolve_task(task_id, protocol.STATE_CANCELED, "")
            rec = self.tasks.get(task_id)
            if rec and rec["state"] not in protocol.TERMINAL_STATES:
                self.tasks.complete(task_id, protocol.STATE_CANCELED, "")
                from .a2a_persistence import _try_persist_task_ledger, _task_ledger_path
                ok = _try_persist_task_ledger(self.tasks, _task_ledger_path(), f"CANCELED on_processing_complete {task_id}")
                if not ok:
                    logger.error("A2A: failed to persist CANCELED state for task %s after disconnect — state indeterminate", task_id)
                protocol.metrics.tasks_failed += 1
                try:
                    self._send_push_notification(task_id, rec["context_id"], "", protocol.STATE_CANCELED)
                except Exception:
                    pass
        else:
            self._resolve_task(task_id, protocol.STATE_COMPLETED, "")

    # ── Push notification delivery ────────────────────────────────────────

    def _send_push_notification(self, task_id: str, context_id: str, reply: str, state: str):
        callback_url = self.tasks.pop_push_url(task_id)
        if not callback_url:
            return
        if not security.is_safe_callback_url(
            callback_url,
            localhost_mode=self._security_context.localhost_only(),
        ):
            logger.warning("A2A: push notification for task %s blocked — unsafe callback URL: %s",
                           task_id, callback_url)
            protocol.metrics.push_failed += 1
            return
        payload = protocol.status_update(task_id, context_id, state, (reply or "")[:2000])
        signature = self._security_context.sign_push_payload(payload)
        headers = {"Content-Type": "application/json"}
        if signature:
            headers["X-A2A-Signature"] = signature
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(callback_url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                if 200 <= resp.status < 300:
                    protocol.metrics.push_sent += 1
                    logger.debug("A2A: push notification sent for task %s", task_id)
                else:
                    protocol.metrics.push_failed += 1
                    logger.warning("A2A: push notification for task %s got HTTP %d", task_id, resp.status)
        except Exception as e:
            protocol.metrics.push_failed += 1
            logger.warning("A2A: push notification for task %s failed: %s", task_id, e)
