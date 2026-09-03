"""
A2A inbound platform adapter — exposes Hermes as an A2A-discoverable agent.

Design (the #11025 insight, done as a plugin with zero core edits):
  - Runs a stdlib http.server in a daemon thread (no a2a-sdk, no asyncio loop
    dependency at register() time — avoids the a2a_fleet "register outside a
    loop" bug class).
  - Serves the A2A v1.0 Agent Card at GET /.well-known/agent-card.json (and legacy agent.json).
  - JSON-RPC at POST /: message/send, message/stream (SSE), tasks/get,
    tasks/list, tasks/cancel, tasks/subscribe, tasks/pushNotificationConfig/create,
    tasks/pushNotificationConfig/get, tasks/pushNotificationConfig/list,
    tasks/pushNotificationConfig/delete.
  - Push notifications: config accepted inline in message/send
    (configuration.taskPushNotificationConfig) or via the create method;
    payloads are v1.0 StreamResponse objects, HMAC-signed.
  - Metrics at GET /metrics.
  - Each inbound task is filtered + framed (security.wrap_inbound) and routed
    into the agent's LIVE gateway session via the normal MessageEvent path, so
    the agent that replies is the same one talking to its user — full memory
    and context, not a throwaway clone.
  - The agent's reply comes back through ``adapter.send()``; we override that to
    fulfil a per-task Future the HTTP handler is blocked on, turning the
    async gateway into a synchronous request/response for the A2A caller.
    ``on_processing_complete`` resolves failures/cancellations promptly.
  - Every exchange is persisted to disk and audit-logged.

Bind safety: with no token configured, the server binds 127.0.0.1 only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import select
import socket
import sqlite3
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import weakref
from collections import deque
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FuturesTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)
from gateway.config import Platform
from .task_routing import TaskRPCHandler

from . import protocol, security

logger = logging.getLogger(__name__)

# _DEFAULT_PORT and _MAX_CONTEXT_PEERS imported from a2a_persistence
_ORPHAN_TIMEOUT = 300  # seconds before a pending task is considered orphaned
_WATCHDOG_INTERVAL = 60  # seconds between orphaned task watchdog runs
_MAX_BODY = 1_048_576  # 1MB max request body — prevents DoS via memory exhaustion
_SSE_KEEPALIVE = 5  # seconds between SSE keepalive comments
# Seconds past the client's advertised read timeout (sender.timeout) before
# the server assumes the client gave up and marks the pending task
# out_of_band_only. The MSG_PEEK probe catches clients that CLOSE; this is
# the deterministic backstop for clients that stay connected but will
# discard the reply (the reply was consumed although the server saw
# "alive").
_PATIENCE_MARGIN = 30

# Portable non-blocking recv flag: MSG_DONTWAIT is a Unix-only socket flag
# that is not exposed by CPython on Windows.  When absent (Winsock), fall
# back to 0 — MSG_PEEK + select() already guarantee non-blocking semantics.
_PORTABLE_NONBLOCK_RECV = getattr(socket, "MSG_DONTWAIT", 0)
# Short-window inbound dedupe: the same wire message
# (contextId, messageId) must not be dispatched twice.
_INBOUND_DEDUPE_WINDOW = 60.0
_INBOUND_DEDUPE_MAX = 1024

# Module-level registry of live A2A adapters (weak refs so a dead gateway
# never pins memory). The outbound client tools (plugins/platforms/a2a/tools.py)
# use this to register the *local* context→peer mapping whenever an agent
# makes an outbound A2A call from ANY platform origin (discord, telegram,
# CLI/ACP, api_server). Without this, _context_peers only ever learns peers
# from inbound A2A tasks, so a completion push for a context that was born on
# another platform finds no peer: no A2A inbound ever touched its contextId,
# so the push had nowhere to go).
_ADAPTERS: "dict[int, weakref.ReferenceType[A2AAdapter]]" = {}
_ADAPTERS_GUARD = threading.Lock()

# Persistence utilities extracted to a2a_persistence.py
from .a2a_persistence import (
    _DEFAULT_PORT,
    _HAS_FCNTL,
    _HAS_MSVCRT,
    _LOOPBACK_ADDRS,
    _MAX_CONTEXT_PEERS,
    _MSVCRT_RETRIES,
    _MSVCRT_RETRY_DELAY,
    _THREAD_FALLBACK_LOCK,
    _active_profile_name,
    _bracket_ipv6,
    _clean_slug,
    _context_peers_path,
    _context_sessions_path,
    _default_agent_name,
    _fanout_children_path,
    _file_lock,
    _file_lock_fcntl,
    _file_lock_msvcrt,
    _file_lock_thread_fallback,
    _is_ipv6_literal,
    _is_own_endpoint,
    _join_url,
    _load_context_peers,
    _load_context_sessions,
    _load_fanout_children,
    _loopback_fallback_url,
    _merge_context_peers,
    _merge_context_sessions,
    _merge_fanout_children,
    _own_a2a_url,
    _persist_context_peers,
    _persist_context_sessions,
    _persist_fanout_children,
    _profile_home,
    _profile_scoped,
    _reply_timeout,
    _reset_worker_session_vars,
    _safe_context_slug,
    _sender_url_acceptable,
    _task_ledger_path,
    _try_persist_task_ledger,
)

def _method_info(method: str) -> tuple[str, bool]:
    """Return (canonical_operation, is_v1_method)."""
    mapping = {
        "SendMessage": ("send", True),
        "message/send": ("send", False),
        "SendStreamingMessage": ("stream", True),
        "message/stream": ("stream", False),
        "GetTask": ("get", True),
        "tasks/get": ("get", False),
        "ListTasks": ("list", True),
        "tasks/list": ("list", False),
        "CancelTask": ("cancel", True),
        "tasks/cancel": ("cancel", False),
        "SubscribeToTask": ("subscribe", True),
        "tasks/subscribe": ("subscribe", False),
        "CreateTaskPushNotificationConfig": ("push_create", True),
        "tasks/pushNotificationConfig/create": ("push_create", False),
        "tasks/pushNotificationConfig/set": ("push_create", False),
        "tasks/pushNotification/set": ("push_create", False),
        "GetTaskPushNotificationConfig": ("push_get", True),
        "tasks/pushNotificationConfig/get": ("push_get", False),
        "ListTaskPushNotificationConfigs": ("push_list", True),
        "tasks/pushNotificationConfig/list": ("push_list", False),
        "DeleteTaskPushNotificationConfig": ("push_delete", True),
        "tasks/pushNotificationConfig/delete": ("push_delete", False),
    }
    return mapping.get(method, ("", False))


def _redacted_jsonrpc_detail(raw_error):
    """Bounded redacted detail for JSON-RPC peer error (Finding 2).

    Uses security.redact_outbound to scrub credential-shaped content,
    retains category jsonrpc via caller, returns (error_str, payload_redacted).
    Bounded to 300 chars plus truncation marker; audit truncates to 500 anyway.
    """
    raw_str = str(raw_error)
    redacted = security.redact_outbound(raw_str)
    if len(redacted) > 300:
        redacted = redacted[:300] + "...[truncated]"
    payload_redacted = None
    if isinstance(raw_error, dict):
        payload_redacted = {}
        for k, v in raw_error.items():
            if isinstance(v, str):
                rv = security.redact_outbound(v)
                if len(rv) > 300:
                    rv = rv[:300] + "...[truncated]"
                payload_redacted[k] = rv
            else:
                payload_redacted[k] = v
    else:
        payload_redacted = redacted
    return redacted, payload_redacted


def _audit_loopback_failure(peer: str, context_id: str, error: str, category: str, task_id: str = "") -> None:
    """Emit exactly one failure audit for loopback (Finding 1, centralized seam).

    Durability -> push_failed; routing -> push_dropped (existing choice) or
    push_failed; transport/invalid -> push_failed. Bounded via security.audit's
    500-char truncation; redact credential shapes as defense-in-depth.
    Guarded best-effort so audit never raises into the push path.
    """
    # Redact as defense-in-depth even for internal errors (no credential expected)
    safe_error = security.redact_outbound(error)
    if len(safe_error) > 500:
        safe_error = safe_error[:500]
    # Map category to existing audit direction per Amendment A
    direction = "push_failed"
    if category == "routing":
        # Preserve existing push_dropped semantics for routing; probe accepts
        # either but push_dropped is the established routing failure audit.
        direction = "push_dropped"
    try:
        security.audit(direction, peer, task_id or context_id, safe_error, context_id=context_id)
    except Exception:
        pass


class _A2AServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that carries a reference to its adapter."""

    daemon_threads = True

    def __init__(self, addr, handler_cls, adapter: "A2AAdapter"):
        super().__init__(addr, handler_cls)
        self.adapter = adapter


class A2ARequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for the A2A JSON-RPC surface."""

    @property
    def adapter(self) -> "A2AAdapter":
        return self.server.adapter  # type: ignore[attr-defined]

    # Silence the default stderr access log.
    def log_message(self, format, *args):  # noqa: A002,N802
        logger.debug("A2A http: " + format, *args)

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        # Flush HERE so a dead socket surfaces OSError AT THE CALL SITE —
        # a buffered write into a half-closed socket "succeeds" silently via
        # TCP buffering (R2: the reply vanished because only the buffer, not
        # the client, received it) and the exception would otherwise only
        # fire later in the base handler's finish(), outside our catch.
        self.wfile.flush()

    def _request_public_url(self) -> str:
        """Derive the routable URL for this request."""
        explicit = os.getenv("A2A_PUBLIC_URL", "").strip()
        if explicit:
            return explicit
        host = self.headers.get("X-Forwarded-Host", "") or self.headers.get("Host", "")
        if not host:
            return ""
        host = host.split(",")[0].strip()
        scheme = (self.headers.get("X-Forwarded-Proto", "") or "http").split(",")[0].strip()
        return f"{scheme}://{host}/"

    def do_GET(self):  # noqa: N802
        route = self.adapter._route_for_path(self.path)
        agent = route["agent"]
        subpath = route["subpath"].rstrip("/") or "/"
        if subpath in ("/.well-known/agent.json", "/.well-known/agent-card.json"):
            public_url = self._request_public_url() or None
            self._json(200, self.adapter._build_card(public_url, agent=agent))
            return
        if subpath in ("/", "/health"):
            payload = {
                "status": "ok",
                "agent": agent.get("name") or self.adapter.agent_name,
            }
            # Do not leak profile/tenant topology on remote unauthenticated GETs.
            # Agent Cards are intentionally public; health topology is not.
            if self.adapter._security_context.localhost_only() or self.adapter._security_context.authenticate(
                self.headers.get("Authorization"),
                self.client_address[0] if self.client_address else "",
            ) is not None:
                payload["served_agents"] = self.adapter._served_agent_summary(
                    public_url=self._request_public_url() or None)
            self._json(200, payload)
            return
        if subpath == "/metrics":
            self._json(200, protocol.metrics.snapshot())
            return
        self._json(404, {"error": "not found"})

    def _a2a_client_alive(self) -> bool:
        """Best-effort liveness probe for the client behind this request."""
        sock = getattr(self, "connection", None)
        if sock is None:
            return True
        try:
            readable, _, _ = select.select([sock], [], [], 0)
            if not readable:
                return True  # no EOF/data pending — assume alive
            # Data or EOF available. b"" (EOF) means the client closed.
            # MSG_DONTWAIT is not a Winsock receive flag; CPython does not
            # expose it on Windows.  MSG_PEEK already prevents consuming the
            # data, and select() has established readability, so the non-
            # blocking flag is a belt-and-suspenders guard on Unix.  On
            # Windows we fall back to MSG_PEEK alone.
            chunk = sock.recv(1, socket.MSG_PEEK | _PORTABLE_NONBLOCK_RECV)
            return bool(chunk)
        except (BlockingIOError, InterruptedError):
            return True
        except OSError:
            return False

    def _handle_send(self, req_id, params, identity, agent, is_v1):
        """Route a message/send with dead-client protection."""
        result = self.adapter._rpc_message_send(
            req_id, params, identity, agent=agent, v1_response=is_v1,
            client_alive=self._a2a_client_alive,
        )
        if result is None:
            # out_of_band_only with a completed reply: already pushed
            # directly — skip the socket write entirely (the client is gone).
            self.close_connection = True
            return
        # Bounded final pre-write liveness probe: the last keepalive probe
        # during _await_reply may have been seconds ago; a client that died
        # in that window would silently lose the reply via TCP buffering.
        # Probe once more before the write; route dead clients through the
        # existing rescue and skip the socket write entirely.
        #
        # RESIDUAL RACE (do NOT claim elimination): a client that dies
        # between THIS probe and the _json write is still lost here —
        # the broad OSError catch below is the final safety net.  A stable
        # delivery ID / application ACK protocol would close this gap but
        # is explicitly future work (see design decision 4).
        if not self._a2a_client_alive():
            self.adapter._push_reply_after_client_gone(req_id, result, is_v1=is_v1)
            self.close_connection = True
            return
        try:
            self._json(200, result)
        except OSError:
            self.adapter._push_reply_after_client_gone(req_id, result, is_v1=is_v1)

    def do_POST(self):  # noqa: N802
        adapter = self.adapter
        client_ip = self.client_address[0] if self.client_address else ""

        # Identity comes from the presented credential (or the socket in
        # localhost-only mode) — never from the request body.
        identity = adapter._security_context.authenticate(
            self.headers.get("Authorization"), client_ip
        )
        if identity is None:
            self._json(401, protocol.jsonrpc_error(None, protocol.ERR_UNAUTHORIZED, "unauthorized"))
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > _MAX_BODY:
                self._json(413, protocol.jsonrpc_error(None, protocol.ERR_PARSE, "payload too large"))
                return
            raw = self.rfile.read(length) if length else b"{}"
            req = json.loads(raw.decode("utf-8"))
        except Exception:
            self._json(400, protocol.jsonrpc_error(None, protocol.ERR_PARSE, "parse error"))
            return

        if not isinstance(req, dict):
            self._json(400, protocol.jsonrpc_error(None, protocol.ERR_INVALID_PARAMS, "JSON-RPC request must be an object"))
            return

        req_id = req.get("id")
        method = str(req.get("method", ""))
        params = req.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            self._json(200, protocol.jsonrpc_error(req_id, protocol.ERR_INVALID_PARAMS, "params must be an object"))
            return

        version = (self.headers.get("A2A-Version") or "").strip()
        if version and version not in {"1.0", "1.0.0"}:
            self._json(200, protocol.jsonrpc_error(req_id, protocol.ERR_INVALID_PARAMS, f"unsupported A2A-Version: {version}"))
            return

        operation, is_v1 = _method_info(method)
        route = adapter._route_for_request(self.path, params)
        if route.get("error"):
            self._json(400, protocol.jsonrpc_error(req_id, protocol.ERR_INVALID_PARAMS, route["error"]))
            return
        agent = route["agent"]

        if not adapter._rate_limiter.allow(identity):
            protocol.metrics.rate_limit_triggers += 1
            self._json(429, protocol.jsonrpc_error(req_id, protocol.ERR_RATE_LIMITED, "rate limit exceeded"))
            return

        if not adapter._security_context.is_trusted_peer(identity):
            self._json(403, protocol.jsonrpc_error(
                req_id, protocol.ERR_UNTRUSTED_PEER, f"peer '{identity}' not trusted"))
            return

        if not operation:
            self._json(200, protocol.jsonrpc_error(
                req_id, protocol.ERR_METHOD_NOT_FOUND, f"method not found: {method}"))
            return

        if operation == "send":
            self._handle_send(req_id, params, identity, agent=agent, is_v1=is_v1)
            return
        if operation == "stream":
            adapter._rpc_message_stream(self, req_id, params, identity, agent=agent)
            return
        if operation == "get":
            self._json(200, adapter._rpc_tasks_get(req_id, params, agent=agent))
            return
        if operation == "list":
            self._json(200, adapter._rpc_tasks_list(req_id, params, agent=agent))
            return
        if operation == "cancel":
            self._json(200, adapter._rpc_tasks_cancel(req_id, params, agent=agent))
            return
        if operation == "subscribe":
            adapter._rpc_tasks_subscribe(self, req_id, params, agent=agent)
            return
        if operation == "push_create":
            self._json(200, adapter._rpc_push_config_create(req_id, params, agent=agent))
            return
        if operation == "push_get":
            self._json(200, adapter._rpc_push_config_get(req_id, params, agent=agent))
            return
        if operation == "push_list":
            self._json(200, adapter._rpc_push_config_list(req_id, params, agent=agent))
            return
        if operation == "push_delete":
            self._json(200, adapter._rpc_push_config_delete(req_id, params, agent=agent))
            return



class A2AAdapter(BasePlatformAdapter, TaskRPCHandler):
    """Inbound A2A server adapter."""

    def __init__(self, config, **kwargs):
        platform = Platform("a2a")
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}
        # Scope-aware: a secondary multiplex profile must not borrow the
        # default profile's bridged A2A_PORT (mirrors the Buzz/SimpleX fix
        # for #98738) — an unconfigured profile falls closed to the module
        # default port instead. (advertised_toolsets has the same env-leak
        # shape but is left unscoped here — see the "Scope note" in this
        # fix's PR description: open PR #98937 is actively rewriting this
        # field's None-vs-empty-list semantics.)
        self._security_context = security.A2ASecurityContext.capture()
        _port_env = None if _profile_scoped() else os.getenv("A2A_PORT")
        self.port = int(_port_env or extra.get("port", _DEFAULT_PORT))
        self.host = self._security_context.resolve_bind_host()
        self.agent_name = _default_agent_name()
        self._advertised_toolsets = [
            t.strip() for t in (
                list(extra.get("advertised_toolsets") or [])
                or os.getenv("A2A_ADVERTISED_TOOLSETS", "").split(",")
            ) if str(t).strip()
        ]
        self._active_profile = _active_profile_name()
        self._agents = self._load_served_agents(extra)

        self._httpd: Optional[_A2AServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Per-adapter protocol state (not module-global): task store, anti-loop
        # turn tracking, and rate limiting.
        self.tasks = protocol.TaskStore()
        self._turns = protocol.TurnTracker()
        self._rate_limiter = protocol.RateLimiter()

        # Forwarded profile sessions: map (profile, agent_slug, context_id) -> session_id.
        self._profile_sessions: Dict[tuple[str, str, str], str] = {}
        self._profile_session_locks: Dict[tuple[str, str, str], threading.Lock] = {}
        self._profile_session_locks_guard = threading.Lock()

        # Pending reply futures, keyed by task_id. Each future resolves to a
        # (state, text) tuple. _pending_order keeps per-context FIFO order so
        # adapter.send() — which only knows the context — resolves the oldest
        # outstanding task for that context (no cross-talk between concurrent
        # requests sharing a context).
        self._pending: Dict[str, tuple[str, Future]] = {}
        self._pending_order: Dict[str, deque[str]] = {}
        self._pending_lock = threading.Lock()

        # Context → peer identity map. Recorded on every inbound task so an
        # out-of-band send (kanban notifier wake reply, late completion) with
        # no pending waiter can be pushed back to the peer that owns the
        # context — reusing the same contextId keeps the caller's session.
        self._context_peers: Dict[str, str] = {}
        self._context_peers_lock = threading.Lock()

        # Context → originating LOCAL session. Recorded by the client tools
        # at a2a_call time (which gateway session created this context), so
        # an inbound push on the context can WAKE that session via the kanban
        # watcher's self-post mechanism — agency (a fresh agent turn) rather
        # than a conversation-store write nobody polls.
        self._context_sessions: Dict[str, dict] = {}
        self._context_sessions_lock = threading.Lock()

        # Fan-out children: parent_context_id → {peer_name: child_context_id}
        # Recorded by a2a_orchestrate so callers can resume a specific
        # child branch with a2a_call(context_id=child_context_id), and
        # late callbacks trace back to the originating session.
        self._fanout_children: Dict[str, Dict[str, str]] = {}
        self._fanout_children_lock = threading.Lock()

        # Short-window inbound dedupe: (contextId, messageId) → first
        # arrival time. The same wire message must not be dispatched twice
        # (duplicate handoffs were observed in testing, and the push+retry
        # paths make double-delivery possible).
        self._inbound_seen: Dict[tuple[str, str], float] = {}
        self._inbound_seen_lock = threading.Lock()

        # Orphaned task watchdog
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: Optional[threading.Thread] = None

        # Register this adapter so the outbound client tools can map local
        # contexts back to this gateway's peer table (see _register_context_peer).
        with _ADAPTERS_GUARD:
            _ADAPTERS[id(self)] = weakref.ref(self)

    # ── Cross-platform context peer registration ─────────────────────────

    @classmethod
    def _register_context_peer(cls, context_id: str, peer: str) -> None:
        """Record ``context_id`` → ``peer`` on every live local A2A adapter."""
        if not context_id or not peer:
            return
        with _ADAPTERS_GUARD:
            refs = list(_ADAPTERS.values())
        union: Dict[str, str] = {}
        for ref in refs:
            adapter = ref()
            if adapter is None:
                continue
            with adapter._context_peers_lock:
                adapter._context_peers[context_id] = peer
                if len(adapter._context_peers) > _MAX_CONTEXT_PEERS:
                    adapter._context_peers.pop(next(iter(adapter._context_peers)), None)
                union.update(adapter._context_peers)
        # Write-through so the registration survives a gateway restart:
        # a restart wipes the in-memory map and no later inbound/outbound
        # task re-registered the context, so the completion push would be
        # dropped before any side effect.  Always include the new
        # context_id→peer mapping directly (not only via union) so a
        # registration made with no live adapters (e.g. a CLI/ACP process)
        # is still persisted for the next gateway start.  Merge with
        # the on-disk state and never clobber existing entries.
        # Serialise the load→merge→write cycle with a file lock so
        # two concurrent registrations (e.g. two outbound a2a_call
        # threads) don't clobber each other's disk state.
        with _file_lock(_context_peers_path().with_suffix(".lock")):
            disk = _load_context_peers()
            disk[context_id] = peer
            disk.update(union)
            _persist_context_peers(_merge_context_peers({}, disk, _MAX_CONTEXT_PEERS))

    @classmethod
    def _register_context_session(cls, context_id: str, origin: dict) -> None:
        """Record ``context_id`` → the LOCAL session that created it."""
        if not context_id or not isinstance(origin, dict) or not origin.get("platform"):
            return
        with _ADAPTERS_GUARD:
            refs = list(_ADAPTERS.values())
        union: Dict[str, dict] = {}
        for ref in refs:
            adapter = ref()
            if adapter is None:
                continue
            with adapter._context_sessions_lock:
                adapter._context_sessions[context_id] = dict(origin)
                if len(adapter._context_sessions) > _MAX_CONTEXT_PEERS:
                    adapter._context_sessions.pop(next(iter(adapter._context_sessions)), None)
                union.update(adapter._context_sessions)
        # Write-through so the mapping survives a gateway restart — a context
        # born before a restart must still wake its session afterwards (the
        # same restart-wipe failure the peer map suffered).
        # setdefault the direct entry too: with no live adapter in this
        # process (CLI/one-shot) union is empty, and the registration must
        # still land on disk for the next gateway start.
        with _file_lock(_context_sessions_path().with_suffix(".lock")):
            disk = _load_context_sessions()
            disk.update(union)
            disk.setdefault(context_id, dict(origin))
            _persist_context_sessions(_merge_context_sessions({}, disk, _MAX_CONTEXT_PEERS))

    @classmethod
    def _own_sender(cls) -> dict:
        """Return this process's A2A AgentName identity for outbound messages."""
        with _ADAPTERS_GUARD:
            refs = list(_ADAPTERS.values())
        for ref in refs:
            adapter = ref()
            if adapter is not None:
                return adapter._sender_identity()
        return cls._sender_from_config()

    @classmethod
    def _sender_from_config(cls) -> dict:
        """Sender identity for processes with no live adapter (CLI/helpers)."""
        name = os.getenv("A2A_AGENT_NAME", "").strip() or _default_agent_name()
        port = _DEFAULT_PORT
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
            a2a_cfg = (cfg.get("platforms") or {}).get("a2a") or {}
            port = int(a2a_cfg.get("port") or _DEFAULT_PORT)
        except Exception:
            logger.debug("A2A: could not read a2a port from config for sender identity", exc_info=True)
        try:
            port = int(os.getenv("A2A_PORT") or port)
        except (ValueError, TypeError):
            pass
        public = os.getenv("A2A_PUBLIC_URL", "").strip()
        if public:
            url = public.rstrip("/")
        else:
            url = _own_a2a_url("127.0.0.1", port)
        return {"agentId": name, "name": name, "url": url}

    def _sender_identity(self) -> dict:
        """This adapter's A2A v1.0 AgentName (``agentId``/``name``/``url``)."""
        return {
            "agentId": self.agent_name,
            "name": self.agent_name,
            "url": _own_a2a_url(self.host, self.port),
        }

    def _refine_peer_identity(self, peer: str, params: dict, context_id: str) -> str:
        """Resolve a port-less ``ip:`` identity to a routable peer.

        Security: never promote an ``ip:`` identity to a configured peer
        from ``agentId`` alone.  Require authenticated binding or exact
        configured endpoint/origin validation before using the peer's
        auth/headers; otherwise retain the authenticated identity and fail
        closed for an unresolvable callback.
        """
        if not peer.startswith("ip:"):
            return peer
        sender = protocol.extract_sender(params)
        if not isinstance(sender, dict):
            return peer
        name = str(sender.get("agentId") or sender.get("name") or "").strip()
        url = str(sender.get("url") or "").strip()
        if not name and not url:
            return peer
        peers_cfg: dict = {}
        try:
            from . import tools as a2a_tools
            peers_cfg = (a2a_tools._load_config() or {}).get("a2a_agents") or {}
        except Exception:
            logger.debug("A2A: could not load a2a_agents for peer refinement", exc_info=True)
        # Do not promote from agentId/name alone — require URL/origin validation.
        if name and name in peers_cfg:
            cfg_entry = peers_cfg.get(name) if isinstance(peers_cfg.get(name), dict) else {}
            cfg_url_str = str((cfg_entry or {}).get("url") or "").strip()
            if url and cfg_url_str and _sender_url_acceptable(url, peers_cfg):
                try:
                    cfg_parsed = urllib.parse.urlparse(cfg_url_str)
                    sender_parsed = urllib.parse.urlparse(url)
                    if (cfg_parsed.hostname and sender_parsed.hostname
                            and cfg_parsed.hostname.lower() == sender_parsed.hostname.lower()
                            and cfg_parsed.port == sender_parsed.port):
                        logger.info(
                            "A2A: refined ip: identity for context %s to configured peer %r (sender agentId+url validated)",
                            context_id, name,
                        )
                        return name
                except Exception:
                    pass
            # No validated URL binding for this name — retain authenticated identity
            logger.info(
                "A2A: refusing to promote ip identity for context %s to %r without URL/origin validation; retaining %r",
                context_id, name, peer,
            )
            return peer
        if url and _sender_url_acceptable(url, peers_cfg):
            # Resolve back to the configured peer key when the sender URL
            # matches a configured peer's URL — returning the URL string
            # loses the bearer auth from a2a_agents config.
            for cfg_key, cfg_entry in peers_cfg.items():
                if not isinstance(cfg_entry, dict):
                    continue
                try:
                    cfg_url = urllib.parse.urlparse(str(cfg_entry.get("url") or ""))
                except Exception:
                    continue
                sender_parsed = urllib.parse.urlparse(url)
                if (cfg_url.hostname and sender_parsed.hostname
                        and cfg_url.hostname.lower() == sender_parsed.hostname.lower()
                        and cfg_url.port == sender_parsed.port):
                    logger.info(
                        "A2A: refined ip: identity for context %s to configured peer %r "
                        "(sender url %s matched config key %s — retaining bearer auth)",
                        context_id, cfg_key, url, cfg_key,
                    )
                    return cfg_key
            logger.info(
                "A2A: refined ip: identity for context %s to sender url %s",
                context_id, url,
            )
            return url
        return peer

    @classmethod
    def _origin_delivery_target(cls, context_id: str, platform_name: str) -> dict:
        """Delivery target of the local session that started this A2A context."""
        origin: dict = {}
        with _ADAPTERS_GUARD:
            refs = list(_ADAPTERS.values())
        for ref in refs:
            adapter = ref()
            if adapter is None:
                continue
            with adapter._context_sessions_lock:
                origin = adapter._context_sessions.get(context_id) or {}
            if origin:
                break
        if not origin:
            origin = (_load_context_sessions() or {}).get(context_id) or {}
        if not origin:
            return {}
        if str(origin.get("platform") or "").strip().lower() != str(platform_name or "").strip().lower():
            return {}
        chat_id = str(origin.get("chat_id") or "").strip()
        if not chat_id:
            return {}
        return {
            "chat_id": chat_id,
            "thread_id": str(origin.get("thread_id") or "").strip(),
            "chat_type": str(origin.get("chat_type") or "group").strip() or "group",
        }

    def _unregister_adapter(self) -> None:
        with _ADAPTERS_GUARD:
            _ADAPTERS.pop(id(self), None)

    @property
    def name(self) -> str:
        return "A2A"

    @property
    def authorization_is_upstream(self) -> bool:
        """A2A authenticates every inbound request via bearer token (or"""
        return True

    # ── Fan-out children registration ────────────────────────────────────

    @classmethod
    def _register_fanout_children(
        cls, parent_context_id: str, peer_children: Dict[str, str],
        origin: Optional[dict] = None,
    ) -> None:
        """Record a fan-out operation: parent → {peer: child_context_id}."""
        if not parent_context_id or not peer_children:
            return
        new_entry = {parent_context_id: dict(peer_children)}
        with _ADAPTERS_GUARD:
            refs = list(_ADAPTERS.values())
        for ref in refs:
            adapter = ref()
            if adapter is None:
                continue
            with adapter._fanout_children_lock:
                adapter._fanout_children = _merge_fanout_children(
                    adapter._fanout_children, new_entry, _MAX_CONTEXT_PEERS,
                )
        # Persist to disk for restart recovery (bounded eviction).
        with _file_lock(_fanout_children_path().with_suffix(".lock")):
            disk = _load_fanout_children()
            merged = _merge_fanout_children(disk, new_entry, _MAX_CONTEXT_PEERS)
            _persist_fanout_children(merged)

    @classmethod
    def _get_fanout_children(cls, parent_context_id: str) -> dict:
        """Return {peer: child_context_id} for a fan-out parent, or {}."""
        if not parent_context_id:
            return {}
        with _ADAPTERS_GUARD:
            refs = list(_ADAPTERS.values())
        for ref in refs:
            adapter = ref()
            if adapter is None:
                continue
            with adapter._fanout_children_lock:
                children = adapter._fanout_children.get(parent_context_id)
                if children:
                    return dict(children)
        disk = _load_fanout_children()
        return dict(disk.get(parent_context_id) or {})

    @classmethod
    def _reject_child_reuse(cls, child_context_id: str, requesting_peer: str) -> str:
        """Check if a child context is already claimed by a different peer."""
        if not child_context_id:
            return ""
        with _ADAPTERS_GUARD:
            refs = list(_ADAPTERS.values())
        for ref in refs:
            adapter = ref()
            if adapter is None:
                continue
            with adapter._fanout_children_lock:
                for _parent, children in adapter._fanout_children.items():
                    if not isinstance(children, dict):
                        continue
                    for peer_name, cid in children.items():
                        if cid == child_context_id:
                            if peer_name != requesting_peer:
                                return peer_name
                            return ""
        # Disk fallback.
        disk = _load_fanout_children()
        for _parent, children in disk.items():
            if not isinstance(children, dict):
                continue
            for peer_name, cid in children.items():
                if cid == child_context_id:
                    if peer_name != requesting_peer:
                        return peer_name
                    return ""
        return ""

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def connect(self, **_kwargs) -> bool:
        # Gateway reconnection plumbing passes adapter-agnostic kwargs such as
        # ``is_reconnect``. A2A does not need them, but accepting them keeps the
        # plugin compatible with the BasePlatformAdapter lifecycle contract.
        # Capture the running gateway loop so the HTTP thread can marshal
        # events onto it via run_coroutine_threadsafe.
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        try:
            self._httpd = _A2AServer((self.host, self.port), A2ARequestHandler, self)
        except OSError as e:
            logger.error("A2A: could not bind %s:%s — %s", self.host, self.port, e)
            self._set_fatal_error("bind_failed", f"A2A bind failed: {e}", retryable=True)
            return False

        self._server_thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="a2a-http",
            daemon=True,
        )
        self._server_thread.start()

        # Reset watchdog state for reconnection (disconnect sets the event)
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="a2a-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

        # Reload context→peer registrations persisted by a previous gateway
        # run. Without this, a restart wipes every registration and out-of-band
        # completion pushes drop until a fresh inbound task re-registers the
        # context: the gateway restarted between the original call and the
        # completion, and the push had no peer.
        with self._context_peers_lock:
            restored = _load_context_peers()
            merged = _merge_context_peers(self._context_peers, restored, _MAX_CONTEXT_PEERS)
            self._context_peers.clear()
            self._context_peers.update(merged)
        if restored:
            logger.info(
                "A2A: restored %d context→peer registration(s) from %s",
                len(restored), _context_peers_path(),
            )

        # Restore context→origin-session registrations persisted by a
        # previous gateway run so pushes still wake their originating
        # sessions after a restart (a2a_call re-registers on the next call,
        # but a completion arriving right after a restart must not drop).
        restored_sessions = self._restore_persisted_context_sessions()
        if restored_sessions:
            logger.info(
                "A2A: restored %d context→origin-session registration(s) from %s",
                restored_sessions, _context_sessions_path(),
            )

        # Restore fan-out children map from disk so callers can still
        # resume child branches after a restart.
        restored_fanout = self._restore_persisted_fanout_children()
        if restored_fanout:
            logger.info(
                "A2A: restored %d fan-out parent→children registration(s) from %s",
                restored_fanout, _fanout_children_path(),
            )

        # Restore task ledger so GetTask/ListTasks/SubscribeToTask survive
        # gateway restarts.  Terminal task records (COMPLETED, FAILED,
        # CANCELED) and recent non-terminal tasks are persisted by
        # _persist_task_ledger on every task completion.
        restored_tasks = self.tasks.restore(_task_ledger_path())
        if restored_tasks:
            logger.info(
                "A2A: restored %d task record(s) from %s",
                restored_tasks, _task_ledger_path(),
            )

        self._mark_connected()

        exposure = (
            "localhost-only"
            if self._security_context.localhost_only()
            else "REMOTE (bearer auth)"
        )
        logger.info(
            "A2A: serving Agent Card + JSON-RPC on http://%s:%s (%s) as %r; %d routed agent(s)",
            self.host, self.port, exposure, self.agent_name, len(self._agents),
        )
        # Plugin-registered native handlers (ctx.register_platform_handler).
        self._wire_plugin_handlers(None)
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()
        self._watchdog_stop.set()
        self._unregister_adapter()
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None
        # Durable per-task shutdown: use the same coordinator as send/complete.
        # Do not resolve Futures or clear state before durable publish.
        pending_snapshot: list[tuple[str, str, Any]] = []
        with self._pending_lock:
            for tid, (ctx, fut) in list(self._pending.items()):
                pending_snapshot.append((tid, ctx, fut))
        # First, attempt durable FAILED publish per pending task; only on success resolve waiter
        for tid, ctx, fut in pending_snapshot:
            rec = self.tasks.get(tid)
            if not rec or rec.get("state") in protocol.TERMINAL_STATES:
                # Task already terminal or missing — still clean pending map but don't claim new terminal
                with self._pending_lock:
                    self._pending.pop(tid, None)
                    order = self._pending_order.get(ctx)
                    if order is not None:
                        try:
                            order.remove(tid)
                        except ValueError:
                            pass
                        if not order:
                            self._pending_order.pop(ctx, None)
                continue
            candidate = dict(rec)
            candidate["state"] = protocol.STATE_FAILED
            candidate["reply"] = "[agent shutting down]"
            candidate["completed_at"] = __import__("time").time()
            try:
                outcome = self.tasks.publish_durable(_task_ledger_path(), tid, candidate)
            except Exception as exc:
                logger.error("A2A: disconnect durable publish exception for task %s: %s", tid, exc, exc_info=True)
                outcome = protocol.DurablePublishOutcome(published=False, newly_published=False, record=rec, durable_state=rec.get("state", ""), error=str(exc))
            if outcome.published and outcome.newly_published:
                # Commit succeeded — now resolve waiter with shutdown outcome and clean pending
                with self._pending_lock:
                    ent = self._pending.get(tid)
                    if ent is not None and ent[1] is fut and not fut.done():
                        try:
                            fut.set_result((protocol.STATE_FAILED, "[agent shutting down]"))
                        except Exception:
                            pass
                    self._pending.pop(tid, None)
                    order = self._pending_order.get(ctx)
                    if order is not None:
                        try:
                            order.remove(tid)
                        except ValueError:
                            pass
                        if not order:
                            self._pending_order.pop(ctx, None)
                # Metrics/audit for successful shutdown terminalization could be added here if desired
            else:
                logger.error("A2A: failed to durably publish FAILED on disconnect for task %s: %s — leaving WORKING", tid, outcome.error)
                # Leave pending and task at prior WORKING for restart recovery; do not resolve Future with terminal
                # Keep pending entry for potential restart handling, but transport is tearing down — waiter will be abandoned
                # We do NOT clear pending for this failed task; but to avoid leaking, we could keep it for restart.
                # For now, keep it so probe sees Future not done with terminal.
                continue
        # Also handle non-pending non-terminal tasks (orphan shutdown) via per-task durable publish
        try:
            # Snapshot of non-terminal tasks not already handled
            remaining_task_ids = []
            with self.tasks._lock:
                for tid, rec in list(self.tasks._tasks.items()):
                    if rec.get("state") not in protocol.TERMINAL_STATES:
                        # Skip those already attempted above
                        if not any(tid == ptid for ptid, _, _ in pending_snapshot):
                            remaining_task_ids.append(tid)
            for tid in remaining_task_ids:
                rec = self.tasks.get(tid)
                if not rec or rec.get("state") in protocol.TERMINAL_STATES:
                    continue
                cand = dict(rec)
                cand["state"] = protocol.STATE_FAILED
                cand["reply"] = "[agent shutting down]"
                cand["completed_at"] = __import__("time").time()
                try:
                    outcome = self.tasks.publish_durable(_task_ledger_path(), tid, cand)
                    if not outcome.published:
                        logger.error("A2A: disconnect orphan durable publish failed for %s: %s", tid, outcome.error)
                except Exception as exc:
                    logger.error("A2A: disconnect orphan publish exception for %s: %s", tid, exc, exc_info=True)
        except Exception:
            logger.error("A2A: failed to persist FAILED state on disconnect/shutdown", exc_info=True)

    # ── Orphaned task watchdog ─────────────────────────────────────────────

    def _watchdog_loop(self) -> None:
        """Background thread that fails orphaned tasks (keeps them queryable)."""
        while not self._watchdog_stop.wait(_WATCHDOG_INTERVAL):
            try:
                failed = self.tasks.fail_orphans(_ORPHAN_TIMEOUT)
                if failed:
                    _try_persist_task_ledger(self.tasks, _task_ledger_path(), f"watchdog {failed}")
                    for tid in failed:
                        logger.warning("A2A: orphaned task %s marked failed (timeout %ds)", tid, _ORPHAN_TIMEOUT)
                        protocol.metrics.tasks_failed += 1
            except Exception:
                logger.debug("A2A: watchdog error", exc_info=True)

    # ── Agent routing + Agent Cards ───────────────────────────────────────

    def _load_global_a2a_config(self) -> dict:
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
            return cfg if isinstance(cfg, dict) else {}
        except Exception:
            return {}

    def _load_served_agents(self, extra: dict) -> dict[str, dict]:
        """Load served-agent routing config."""
        raw = extra.get("agents") or extra.get("served_agents")
        if raw is None:
            cfg = self._load_global_a2a_config()
            raw = cfg.get("a2a_served_agents") or (cfg.get("a2a") or {}).get("served_agents")

        agents: dict[str, dict] = {}
        # Scope-aware for the same reason as port/toolsets above: a secondary
        # profile must not inherit the default profile's A2A_AGENT_DESCRIPTION.
        default_desc = (
            "Hermes Agent — a general-purpose agent reachable over A2A."
            if _profile_scoped()
            else os.getenv(
                "A2A_AGENT_DESCRIPTION",
                "Hermes Agent — a general-purpose agent reachable over A2A.",
            )
        )
        agents[""] = {
            "slug": "",
            "path": "",
            "tenant": "",
            "profile": self._active_profile,
            "local": True,
            "name": self.agent_name,
            "description": default_desc,
            "advertised_toolsets": self._advertised_toolsets,
        }

        reserved = {"health", "metrics", ".well-known"}
        tenants: dict[str, str] = {}
        items = raw.items() if isinstance(raw, dict) else enumerate(raw or []) if isinstance(raw, list) else []
        for key, val in items:
            if not isinstance(val, dict):
                continue
            slug = _clean_slug(str(val.get("slug") or val.get("id") or key))
            if not slug:
                continue
            path_segment = _clean_slug(str(val.get("path") or slug))
            if not path_segment or path_segment in reserved:
                logger.warning("A2A: ignoring served agent %r with reserved/invalid path %r", slug, path_segment)
                continue
            profile = str(val.get("profile") or slug).strip()
            path = "/" + path_segment
            toolsets = val.get("advertised_toolsets") or val.get("toolsets") or val.get("capabilities") or []
            if isinstance(toolsets, str):
                toolsets = [t.strip() for t in toolsets.split(",") if t.strip()]
            local = bool(val.get("local")) or profile in ("", "default", self._active_profile)
            tenant = str(val.get("tenant") or slug).strip()
            if tenant:
                if tenant in tenants:
                    logger.warning(
                        "A2A: ignoring served agent %r with duplicate tenant %r already used by %r",
                        slug, tenant, tenants[tenant],
                    )
                    continue
                tenants[tenant] = slug
            agents[slug] = {
                "slug": slug,
                "path": path,
                "tenant": tenant,
                "profile": profile or slug,
                "local": local,
                "name": str(val.get("name") or f"Hermes {slug}"),
                "description": str(val.get("description") or f"Hermes profile '{profile or slug}' exposed over A2A."),
                "advertised_toolsets": list(toolsets or []),
                "timeout": int(val.get("timeout") or _reply_timeout()),
            }
        return agents

    def _served_agent_summary(self, public_url: Optional[str] = None) -> list[dict]:
        base = (public_url or "").strip() or f"http://{self.host}:{self.port}/"
        return [
            {
                "slug": a["slug"] or "default",
                "name": a.get("name"),
                "url": _join_url(base, a.get("path", "")),
                "tenant": a.get("tenant") or None,
                "profile": a.get("profile"),
                "local": bool(a.get("local")),
            }
            for a in self._agents.values()
        ]

    def _route_for_path(self, raw_path: str) -> dict:
        path = urllib.parse.urlsplit(raw_path or "/").path or "/"
        # Longest prefix wins. Default/root agent is the fallback.
        for agent in sorted(self._agents.values(), key=lambda a: len(a.get("path", "")), reverse=True):
            prefix = agent.get("path", "") or ""
            if prefix and (path == prefix or path.startswith(prefix + "/")):
                subpath = path[len(prefix):] or "/"
                if not subpath.startswith("/"):
                    subpath = "/" + subpath
                return {"agent": agent, "subpath": subpath}
        return {"agent": self._agents[""], "subpath": path}

    def _route_for_request(self, raw_path: str, params: dict) -> dict:
        route = self._route_for_path(raw_path)
        agent = route["agent"]
        tenant = str((params or {}).get("tenant") or "")
        # If no URL prefix chose a non-default agent, allow v1.0 tenant routing.
        if agent.get("slug") == "" and tenant:
            matches = [a for a in self._agents.values() if a.get("tenant") == tenant]
            if matches:
                route = {"agent": matches[0], "subpath": route["subpath"]}
                agent = matches[0]
        expected = str(agent.get("tenant") or "")
        if tenant and expected and tenant != expected:
            return {"error": f"tenant {tenant!r} does not match routed agent {agent.get('slug') or 'default'}"}
        return route

    def _build_card(self, public_url: Optional[str] = None, agent: Optional[dict] = None) -> dict:
        # Prefer per-request public URL (from X-Forwarded-Host / Host /
        # A2A_PUBLIC_URL) over bind host, so peers can call back when we're
        # behind a reverse proxy.
        agent = agent or self._agents[""]
        base = (public_url or "").strip() or f"http://{self.host}:{self.port}/"
        url = _join_url(base, agent.get("path", ""))
        return protocol.build_agent_card(
            name=agent.get("name") or self.agent_name,
            url=url,
            description=agent.get("description") or "Hermes Agent — a general-purpose agent reachable over A2A.",
            skills=self._advertised_skills(agent),
            streaming=bool(agent.get("local", True)),
            push_notifications=True,
            auth_required=not self._security_context.localhost_only(),
            tenant=str(agent.get("tenant") or ""),
        )

    def _advertised_skills(self, agent: Optional[dict] = None) -> list[dict]:
        """Dynamic Agent Card skills from the live tool registry."""
        try:
            from tools.registry import registry as tool_registry
            names = tool_registry.get_registered_toolset_names()
            configured = (agent or {}).get("advertised_toolsets") if agent else self._advertised_toolsets
            allowed = set(configured or []) or None
            mapping = {
                n: tool_registry.get_tool_names_for_toolset(n)
                for n in names
                if allowed is None or n in allowed
            }
            if mapping:
                return protocol.skills_from_toolsets(mapping)
        except Exception:
            logger.debug("A2A: tool registry unavailable for Agent Card", exc_info=True)
        configured = (agent or {}).get("advertised_toolsets") if agent else self._advertised_toolsets
        return protocol.skills_from_toolsets(configured or [])

    # ── Pending reply plumbing ────────────────────────────────────────────

    def _add_pending(self, task_id: str, context_id: str) -> Future:
        fut: Future = Future()
        with self._pending_lock:
            self._pending[task_id] = (context_id, fut)
            self._pending_order.setdefault(context_id, deque()).append(task_id)
        return fut

    def _pop_pending(self, task_id: str) -> None:
        with self._pending_lock:
            entry = self._pending.pop(task_id, None)
            if entry:
                order = self._pending_order.get(entry[0])
                if order:
                    try:
                        order.remove(task_id)
                    except ValueError:
                        pass
                    if not order:
                        self._pending_order.pop(entry[0], None)

    def _resolve_task(self, task_id: str, state: str, text: str) -> bool:
        with self._pending_lock:
            entry = self._pending.get(task_id)
            if entry and not entry[1].done():
                entry[1].set_result((state, text))
                resolved = True
            else:
                resolved = False
        # Pop so resolved entries don't accumulate: HTTP paths pop via
        # _finalize_task, but in-process loopback pushes (and
        # on_processing_complete / cancel) resolve without a finalize call.
        if resolved:
            self._pop_pending(task_id)
        return resolved

    def _resolve_oldest_for_context(self, context_id: str, state: str, text: str) -> bool:
        with self._pending_lock:
            for task_id in list(self._pending_order.get(context_id, ())):
                entry = self._pending.get(task_id)
                if entry and not entry[1].done():
                    entry[1].set_result((state, text))
                    break
            else:
                return False
        self._pop_pending(task_id)
        return True

    def _scope_for_agent(self, agent: Optional[dict]) -> tuple[str, str]:
        agent = agent or self._agents[""]
        return str(agent.get("slug") or ""), str(agent.get("tenant") or "")

    def _forward_lock(self, key: tuple[str, str, str]) -> threading.Lock:
        with self._profile_session_locks_guard:
            lock = self._profile_session_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._profile_session_locks[key] = lock
            return lock

    # ── Inbound task handling ─────────────────────────────────────────────

    def _find_existing_nonterminal_task(self, context_id: str) -> Optional[dict]:
        """Find an existing non-terminal task for ``context_id`` in the task store."""
        recs, _ = self.tasks.list(context_id=context_id)
        for rec in recs:
            if rec["state"] not in protocol.TERMINAL_STATES:
                return rec
        return None

    def _prepare_task(self, params: dict, peer: str, agent: Optional[dict] = None) -> tuple[Optional[dict], Optional[dict]]:
        """Validate, register, and dispatch an inbound message."""
        agent = agent or self._agents[""]
        text = protocol.extract_text(params)
        context_id = protocol.extract_context_id(params) or protocol.new_context_id()
        task_id = protocol.new_task_id()
        # Localhost-only mode authenticates the caller as "ip:<addr>" with no
        # port — unresolvable as a push target when every gateway (including
        # this one) shares one host. Refine the
        # identity from the message's A2A v1.0 sender AgentName so the
        # context→peer registration below routes out-of-band pushes back to
        # the peer's REAL endpoint, port included.
        peer = self._refine_peer_identity(peer, params, context_id)

        # F: inbound dedupe — the same wire message (contextId + messageId)
        # must not be dispatched twice within a short window. Duplicate
        # handoffs were already observed in testing, and the push+retry
        # paths make double-delivery possible. Keyed by the peer-stamped
        # messageId so consecutive turns on one context never collide.
        msg_for_id = params.get("message") if isinstance(params, dict) else None
        message_id = str((msg_for_id.get("messageId") or "") if isinstance(msg_for_id, dict) else "").strip()
        if context_id and message_id and self._is_duplicate_inbound(context_id, message_id):
            # Durable immediate rejection via disk-first primitive (section 5.7)
            agent_slug, tenant = self._scope_for_agent(agent)
            _now = time.time()
            _now_iso = protocol.now_iso()
            _candidate = {
                "task_id": task_id,
                "context_id": context_id,
                "peer": peer,
                "agent_slug": agent_slug,
                "tenant": tenant,
                "state": protocol.STATE_REJECTED,
                "reply": "",
                "created_at": _now,
                "created_iso": _now_iso,
                "push_url": "",
                "push_config_id": "",
            }
            _outcome = self.tasks.publish_durable(_task_ledger_path(), task_id, _candidate)
            if not _outcome.published:
                logger.error("A2A: failed to durably publish REJECTED dedupe for task %s: %s", task_id, _outcome.error)
                raise protocol.DurablePublishError(task_id, context_id, protocol.STATE_REJECTED, _outcome.durable_state, False)
            logger.warning(
                "A2A: duplicate inbound message %s on context %s within the dedupe window; rejecting",
                message_id, context_id,
            )
            rec = _outcome.record or _candidate
            return protocol.build_task(
                task_id, context_id, protocol.STATE_REJECTED,
                "Duplicate message.", created_at=rec["created_iso"],
            ), None

        # Anti-loop ping-pong protection
        turn = self._turns.track(context_id)
        if turn > protocol.max_pingpong_turns():
            protocol.metrics.anti_loop_triggers += 1
            logger.warning("A2A: anti-loop triggered for context %s (turn %d > %d)",
                           context_id, turn, protocol.max_pingpong_turns())
            agent_slug, tenant = self._scope_for_agent(agent)
            _now = time.time()
            _now_iso = protocol.now_iso()
            _candidate = {
                "task_id": task_id,
                "context_id": context_id,
                "peer": peer,
                "agent_slug": agent_slug,
                "tenant": tenant,
                "state": protocol.STATE_REJECTED,
                "reply": "",
                "created_at": _now,
                "created_iso": _now_iso,
                "push_url": "",
                "push_config_id": "",
            }
            _outcome = self.tasks.publish_durable(_task_ledger_path(), task_id, _candidate)
            if not _outcome.published:
                logger.error("A2A: failed to durably publish REJECTED anti-loop for task %s: %s", task_id, _outcome.error)
                raise protocol.DurablePublishError(task_id, context_id, protocol.STATE_REJECTED, _outcome.durable_state, False)
            rec = _outcome.record or _candidate
            return protocol.build_task(
                task_id, context_id, protocol.STATE_REJECTED,
                f"Anti-loop protection: context {context_id} exceeded "
                f"{protocol.max_pingpong_turns()} turns. Start a new context or "
                f"increase A2A_MAX_PINGPONG_TURNS.",
                created_at=rec["created_iso"],
            ), None

        if not text:
            agent_slug, tenant = self._scope_for_agent(agent)
            _now = time.time()
            _now_iso = protocol.now_iso()
            _candidate = {
                "task_id": task_id,
                "context_id": context_id,
                "peer": peer,
                "agent_slug": agent_slug,
                "tenant": tenant,
                "state": protocol.STATE_REJECTED,
                "reply": "",
                "created_at": _now,
                "created_iso": _now_iso,
                "push_url": "",
                "push_config_id": "",
            }
            _outcome = self.tasks.publish_durable(_task_ledger_path(), task_id, _candidate)
            if not _outcome.published:
                logger.error("A2A: failed to durably publish REJECTED empty for task %s: %s", task_id, _outcome.error)
                raise protocol.DurablePublishError(task_id, context_id, protocol.STATE_REJECTED, _outcome.durable_state, False)
            rec = _outcome.record or _candidate
            return protocol.build_task(
                task_id, context_id, protocol.STATE_REJECTED,
                "Empty task — nothing to do.", created_at=rec["created_iso"],
            ), None

        framed = security.wrap_inbound(peer, text)
        security.audit("inbound", peer, task_id, text)
        protocol.persist_message(context_id, "user", text, task_id)
        protocol.metrics.inbound_total += 1

        # Bind the session identity for this A2A context so session-aware
        # tooling (task auto-subscription, notifier routing) can send
        # notifications back to the peer's context. ContextVars (NOT a
        # process-global os.environ write) are the mechanism the tool
        # process reads via get_session_env: the asyncio Task created by
        # run_coroutine_threadsafe below snapshots THIS thread's context, so
        # the bindings ride the whole dispatch chain and stay task-local.
        # os.environ would be last-writer-wins across concurrent A2A
        # contexts and leak into sibling sessions.
        _session_tokens: list = []
        try:
            from gateway.session_context import set_session_vars
            _session_tokens = set_session_vars(
                platform="a2a",
                chat_id=context_id,
                chat_type="dm",
                chat_name=f"a2a:{peer}",
                thread_id=task_id,
                user_id=peer,
                user_name=peer,
                async_delivery=True,
            )
        except Exception as exc:
            logger.warning("A2A: set_session_vars unavailable: %s", exc)
        # Remember which peer owns this context so an out-of-band send with
        # no pending waiter can be pushed back to the caller's session.
        # Bounded: drop the oldest entry past _MAX_CONTEXT_PEERS (dicts keep
        # insertion order) so a long-running gateway can't grow this forever.
        with self._context_peers_lock:
            self._context_peers[context_id] = peer
            if len(self._context_peers) > _MAX_CONTEXT_PEERS:
                self._context_peers.pop(next(iter(self._context_peers)), None)
            # Write-through on inbound registration too: a gateway restart
            # wipes the in-memory map, and the wake self-post path (the task
            # notifier) bypasses this handler — so the disk copy is the only
            # thing that survives to the next start.
            with _file_lock(_context_peers_path().with_suffix(".lock")):
                _persist_context_peers(_merge_context_peers(_load_context_peers(), {context_id: peer}, _MAX_CONTEXT_PEERS))
        self._register_inline_push(task_id, params, agent=agent)

        # Write-ahead: durably create WORKING before any dispatch (disk-first, section 5.7).
        # The task ledger is the authority; memory is updated only after successful disk write.
        # If the write fails, the task remains ABSENT and dispatch is not invoked.
        _agent_slug, _tenant = self._scope_for_agent(agent)
        _now = time.time()
        _now_iso = protocol.now_iso()
        _candidate_working = {
            "task_id": task_id,
            "context_id": context_id,
            "peer": peer,
            "agent_slug": _agent_slug,
            "tenant": _tenant,
            "state": protocol.STATE_WORKING,
            "reply": "",
            "created_at": _now,
            "created_iso": _now_iso,
            "push_url": "",
            "push_config_id": "",
        }
        # If an inline push config was registered, capture it into the candidate
        # (the config is stored via set_push_config which mutates the record; for WORKING
        # we need to include it if present in the store after registration — but since the
        # record is not yet in the store, we keep push_url empty and let the later push_config
        # handling update via separate durable publish if needed. The store's publish will create the record.)
        _outcome_working = self.tasks.publish_durable(_task_ledger_path(), task_id, _candidate_working)
        if not _outcome_working.published:
            logger.error("A2A: failed to durably publish WORKING for task %s: %s", task_id, _outcome_working.error)
            protocol.metrics.tasks_failed += 1
            if _session_tokens:
                try: _reset_worker_session_vars()
                except Exception: pass
            # Fail closed: no dispatch, structured persistence error. The caller (task_routing) will map this to -32603.
            raise protocol.DurablePublishError(task_id, context_id, protocol.STATE_WORKING, _outcome_working.durable_state, False)
        rec = _outcome_working.record or _candidate_working

        if not agent.get("local", True):
            try:
                reply, state = self._forward_to_profile(agent, peer, context_id, framed, task_id)
                # Durable publish for forwarded completion (central commit, section 5.4)
                _candidate_fwd = dict(rec)
                _candidate_fwd["state"] = state
                _candidate_fwd["reply"] = reply
                _candidate_fwd["completed_at"] = time.time()
                _outcome_fwd = self.tasks.publish_durable(_task_ledger_path(), task_id, _candidate_fwd)
                if not _outcome_fwd.published:
                    logger.error("A2A: failed to durably publish forwarded terminal %s for task %s: %s", state, task_id, _outcome_fwd.error)
                    protocol.metrics.tasks_failed += 1
                    raise protocol.DurablePublishError(task_id, context_id, state, _outcome_fwd.durable_state, True)
                # Post-commit side effects only after durable publish (section 5.4)
                if _outcome_fwd.newly_published:
                    protocol.persist_message(context_id, "agent", reply, task_id)
                    security.audit("outbound", peer, task_id, reply, context_id=context_id)
                    if state == protocol.STATE_COMPLETED:
                        protocol.metrics.outbound_total += 1
                        protocol.metrics.tasks_completed += 1
                    else:
                        protocol.metrics.tasks_failed += 1
                    self._send_push_notification(task_id, context_id, reply, state)
                return protocol.build_task(task_id, context_id, state, reply, created_at=rec["created_iso"]), None
            finally:
                if _session_tokens:
                    _reset_worker_session_vars()

        if self._loop is None or self._message_handler is None:
            _candidate_gw = dict(rec)
            _candidate_gw["state"] = protocol.STATE_FAILED
            _candidate_gw["reply"] = ""
            _candidate_gw["completed_at"] = time.time()
            _outcome_gw = self.tasks.publish_durable(_task_ledger_path(), task_id, _candidate_gw)
            if not _outcome_gw.published:
                logger.error("A2A: failed to durably publish FAILED gateway-not-ready for task %s: %s", task_id, _outcome_gw.error)
                raise protocol.DurablePublishError(task_id, context_id, protocol.STATE_FAILED, _outcome_gw.durable_state, True)
            if _outcome_gw.newly_published:
                protocol.metrics.tasks_failed += 1
            return protocol.build_task(
                task_id, context_id, protocol.STATE_FAILED,
                "Agent gateway not ready to accept A2A tasks.",
                created_at=rec["created_iso"],
            ), None

        fut = self._add_pending(task_id, context_id)

        event = MessageEvent(
            text=framed,
            message_type=MessageType.TEXT,
            source=self.build_source(
                chat_id=context_id,
                chat_name=f"a2a:{peer}",
                chat_type="dm",
                user_id=peer,
                user_name=peer,
            ),
            message_id=task_id,
        )

        coro = self.handle_message(event)
        try:
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        except Exception as e:
            # Avoid un-awaited coroutine leak: run_coroutine_threadsafe rejects a
            # closed/stopping loop without consuming the coroutine, which would
            # otherwise emit RuntimeWarning: coroutine was never awaited.
            try:
                coro.close()
            except Exception:
                pass
            self._pop_pending(task_id)
            msg = security.redact_outbound(f"Dispatch failed: {e}")
            _candidate_disp = dict(rec)
            _candidate_disp["state"] = protocol.STATE_FAILED
            _candidate_disp["reply"] = msg
            _candidate_disp["completed_at"] = time.time()
            _outcome_disp = self.tasks.publish_durable(_task_ledger_path(), task_id, _candidate_disp)
            if not _outcome_disp.published:
                logger.error("A2A: failed to durably publish FAILED dispatch for task %s: %s", task_id, _outcome_disp.error)
                raise protocol.DurablePublishError(task_id, context_id, protocol.STATE_FAILED, _outcome_disp.durable_state, True)
            if _outcome_disp.newly_published:
                protocol.metrics.tasks_failed += 1
            return protocol.build_task(
                task_id, context_id, protocol.STATE_FAILED, msg,
                created_at=rec["created_iso"],
            ), None
        finally:
            # The asyncio Task already snapshotted this thread's context
            # (run_coroutine_threadsafe copies it at creation), so the
            # session vars bound above ride the dispatch. Reset the HTTP
            # worker thread's own context so the bindings don't linger on
            # the threadpool thread for the next request.
            if _session_tokens:
                try:
                    _reset_worker_session_vars()
                except Exception:
                    pass

        # Wake the originating local session after durable WORKING so the wake
        # is also ordered after persistence (origin dispatch is a second dispatch).
        try:
            if self._context_sessions.get(context_id):
                asyncio.run_coroutine_threadsafe(
                    self._wake_origin_session(context_id, framed), self._loop
                )
        except Exception as exc:
            logger.debug(
                "A2A: could not schedule origin-session wake for %s: %s",
                context_id, exc,
            )

        return None, {
            "task_id": task_id,
            "context_id": context_id,
            "peer": peer,
            "future": fut,
            "created_iso": rec["created_iso"],
            "started": time.time(),
        }

    def _restore_persisted_context_sessions(self) -> int:
        """Merge persisted context→origin-session registrations into memory."""
        with self._context_sessions_lock:
            restored = _load_context_sessions()
            merged = _merge_context_sessions(self._context_sessions, restored, _MAX_CONTEXT_PEERS)
            self._context_sessions.clear()
            self._context_sessions.update(merged)
        return len(restored)

    def _restore_persisted_fanout_children(self) -> int:
        """Merge persisted fan-out parent→children map into memory."""
        disk = _load_fanout_children()
        if not disk:
            return 0
        with self._fanout_children_lock:
            merged = _merge_fanout_children(self._fanout_children, disk, _MAX_CONTEXT_PEERS)
            self._fanout_children.clear()
            self._fanout_children.update(merged)
        return len(disk)

    async def _wake_origin_session(self, context_id: str, text: str) -> None:
        """Wake the local session that created this A2A context (if any)."""
        with self._context_sessions_lock:
            origin = dict(self._context_sessions.get(context_id) or {})
        if not origin:
            return
        origin_platform = str(origin.get("platform") or "").strip()
        if not origin_platform or origin_platform == "a2a":
            # An a2a-originated context's session IS the session the inbound
            # dispatch above already woke — waking again would double-inject
            # the same message into the same session.
            return

        # Resolve the adapter that owns the originating platform (discord,
        # telegram, api_server, ...). Iterate by platform VALUE so unknown /
        # non-Platform values (cli, tui) simply find no adapter.
        gw = getattr(self, "gateway_runner", None)
        adapter = None
        if gw is not None:
            for _p, _a in (getattr(gw, "adapters", None) or {}).items():
                if str(getattr(_p, "value", _p)) == origin_platform:
                    adapter = _a
                    break
        if adapter is None:
            logger.debug(
                "A2A: no %r adapter to wake origin session for context %s; skipping",
                origin_platform, context_id,
            )
            return

        from gateway.wake import adapter_supports_push, deliver_wake

        if adapter_supports_push(adapter):
            chat_id = str(origin.get("chat_id") or "").strip()
            if not chat_id:
                logger.debug(
                    "A2A: origin session for context %s has no chat_id; cannot wake",
                    context_id,
                )
                return
            from gateway.session import SessionSource

            source = SessionSource(
                platform=adapter.platform,
                chat_id=chat_id,
                chat_type=str(origin.get("chat_type") or "group") or "group",
                thread_id=str(origin.get("thread_id") or "").strip() or None,
                user_id=str(origin.get("user_id") or "").strip() or None,
                profile=str(origin.get("profile") or "").strip() or None,
            )
        else:
            source = None
        session_id = str(origin.get("session_id") or "").strip()

        try:
            await deliver_wake(
                adapter,
                text=text,
                session_id=session_id,
                source=source,
            )
            logger.info(
                "A2A: woke origin %s session for context %s (inbound push)",
                origin_platform, context_id,
            )
        except Exception as exc:
            # Best-effort: the a2a session already processed the message; a
            # broken origin wake must not surface into the task dispatch.
            logger.warning(
                "A2A: wake of origin %s session for context %s failed: %s",
                origin_platform, context_id, exc,
            )

    def _profile_state_db(self, profile: str) -> Optional[str]:
        home = _profile_home(profile)
        if not home:
            return None
        return os.path.join(home, "state.db")

    def _lookup_forward_session(self, profile: str, title: str) -> str:
        db = self._profile_state_db(profile)
        if not db or not os.path.exists(db):
            return ""
        try:
            con = sqlite3.connect(db, timeout=5)
            row = con.execute(
                "SELECT id FROM sessions WHERE title = ? ORDER BY started_at DESC LIMIT 1",
                (title,),
            ).fetchone()
            con.close()
            return str(row[0]) if row else ""
        except Exception:
            logger.debug("A2A: could not lookup forwarded session", exc_info=True)
            return ""

    def _latest_a2a_session(self, profile: str, started_after: float) -> str:
        db = self._profile_state_db(profile)
        if not db or not os.path.exists(db):
            return ""
        try:
            con = sqlite3.connect(db, timeout=5)
            row = con.execute(
                "SELECT id FROM sessions WHERE source = 'a2a' AND started_at >= ? ORDER BY started_at DESC LIMIT 1",
                (started_after - 2.0,),
            ).fetchone()
            con.close()
            return str(row[0]) if row else ""
        except Exception:
            logger.debug("A2A: could not find latest forwarded session", exc_info=True)
            return ""

    def _title_forward_session(self, profile: str, session_id: str, title: str) -> None:
        db = self._profile_state_db(profile)
        if not db or not os.path.exists(db) or not session_id:
            return
        try:
            con = sqlite3.connect(db, timeout=5)
            con.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
            con.commit()
            con.close()
        except Exception:
            logger.debug("A2A: could not title forwarded session", exc_info=True)

    def _forward_to_profile(self, agent: dict, peer: str, context_id: str, framed_text: str, task_id: str) -> tuple[str, str]:
        """Forward a routed A2A task to another local Hermes profile."""
        profile = str(agent.get("profile") or agent.get("slug") or "").strip()
        slug = str(agent.get("slug") or profile or "agent")
        safe_ctx = _safe_context_slug(context_id)
        session_title = f"a2a-{slug}-{safe_ctx}"
        key = (profile or "default", slug, safe_ctx)
        timeout = int(agent.get("timeout") or _reply_timeout())

        lock = self._forward_lock(key)
        with lock:
            session_id = self._profile_sessions.get(key) or self._lookup_forward_session(profile, session_title)
            cmd = ["hermes", "chat", "-q", framed_text, "-Q", "--source", "a2a"]
            if session_id:
                cmd.extend(["--resume", session_id])

            env = os.environ.copy()
            home = _profile_home(profile)
            if home:
                env["HERMES_HOME"] = home
            env["HERMES_A2A_PEER"] = peer
            # Carry the A2A session identity into the forwarded profile's
            # agent subprocess. A CLI process reads these via
            # get_session_env's os.environ fallback, so task
            # auto-subscription + notifier routing can push completions back to
            # this context. Set on the child env only — never on the
            # process-global os.environ (last-writer-wins across concurrent
            # A2A contexts).
            env["HERMES_SESSION_PLATFORM"] = "a2a"
            env["HERMES_SESSION_CHAT_ID"] = context_id
            env["HERMES_SESSION_THREAD_ID"] = task_id
            start = time.time()
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout,
                    env=env, check=False, stdin=subprocess.DEVNULL,
                )
            except subprocess.TimeoutExpired:
                return "[profile did not reply in time]", protocol.STATE_FAILED
            except Exception as e:
                return security.redact_outbound(f"Profile dispatch failed: {e}"), protocol.STATE_FAILED
            if proc.returncode != 0:
                msg = (proc.stderr or proc.stdout or f"profile exited {proc.returncode}").strip()
                return security.redact_outbound(msg[-2000:]), protocol.STATE_FAILED
            if not session_id:
                session_id = self._latest_a2a_session(profile, start)
                if session_id:
                    self._profile_sessions[key] = session_id
                    self._title_forward_session(profile, session_id, session_title)
            return security.redact_outbound((proc.stdout or "").strip()), protocol.STATE_COMPLETED


    def _patience_for(self, params: dict, peer: str) -> float:
        """Client patience for a blocking message/send."""
        _TIMEOUT_CEILING = _ORPHAN_TIMEOUT - _PATIENCE_MARGIN  # 270s
        sender = protocol.extract_sender(params)
        if isinstance(sender, dict):
            try:
                t = float(sender.get("timeout") or 0)
                if math.isfinite(t) and t > 0:
                    return min(t, _TIMEOUT_CEILING)
            except (TypeError, ValueError):
                pass
        try:
            from . import tools as a2a_tools
            entry = a2a_tools._resolve_peer(peer)
            if entry:
                t = float(entry.get("timeout") or 0)
                if math.isfinite(t) and t > 0:
                    return min(t, _TIMEOUT_CEILING)
        except Exception:
            logger.debug("A2A: could not resolve peer timeout for patience", exc_info=True)
        return 120.0

    def _mark_out_of_band(self, pending: dict, reason: str, pop_waiter: bool) -> None:
        """Record that a pending task's client is gone."""
        with self._pending_lock:
            if pending.get("out_of_band_only"):
                return
            pending["out_of_band_only"] = True
            if pop_waiter:
                order = self._pending_order.get(pending["context_id"])
                if order:
                    try:
                        order.remove(pending["task_id"])
                    except ValueError:
                        pass
                    if not order:
                        self._pending_order.pop(pending["context_id"], None)
        logger.info(
            "A2A: %s for task %s (context %s); reply will take the out-of-band push path",
            reason, pending["task_id"], pending["context_id"],
        )
        security.audit(
            "outbound", pending["peer"], pending["task_id"], reason,
            context_id=pending["context_id"],
        )

    def _try_push_reply(self, pending: dict, state: str, reply: str) -> protocol.PushOutcome:
        """Push a completed reply out-of-band, dedupe-guarded. Returns typed PushOutcome."""
        if state not in (protocol.STATE_COMPLETED, protocol.STATE_INPUT_REQUIRED) or not reply:
            return protocol.PushOutcome(success=False, category="routing", error="no reply to push")
        with self._pending_lock:
            if pending.get("pushed"):
                return protocol.PushOutcome(success=True, category="transport", error="")
            pending["pushed"] = True
        try:
            outcome = self._push_out_of_band(pending["context_id"], reply, want_reply=True)
            # Strictly typed: _push_out_of_band now always returns PushOutcome (Amendment B)
            if not outcome.success:
                logger.warning(
                    "A2A: out-of-band push for task %s returned failure %s: %s",
                    pending.get("task_id"), outcome.category, outcome.error,
                )
            return outcome
        except Exception as exc:
            logger.warning(
                "A2A: out-of-band push for task %s failed: %s",
                pending.get("task_id"), exc,
            )
            return protocol.PushOutcome(success=False, category="transport", error=str(exc))

    def _is_duplicate_inbound(self, context_id: str, message_id: str) -> bool:
        """Windowed (contextId, messageId) dedupe."""
        key = (context_id, message_id)
        now = time.time()
        with self._inbound_seen_lock:
            if len(self._inbound_seen) > _INBOUND_DEDUPE_MAX:
                for k, ts in list(self._inbound_seen.items()):
                    if now - ts > _INBOUND_DEDUPE_WINDOW:
                        del self._inbound_seen[k]
                while len(self._inbound_seen) > _INBOUND_DEDUPE_MAX:
                    self._inbound_seen.pop(next(iter(self._inbound_seen)), None)
            seen = self._inbound_seen.get(key)
            if seen is not None and now - seen <= _INBOUND_DEDUPE_WINDOW:
                return True
            self._inbound_seen[key] = now
            return False

    def _await_reply(self, pending: dict, keepalive=None, patience: Optional[float] = None) -> tuple[str, str, bool, bool]:
        """Block until the task's future resolves (or times out)."""
        fut: Future = pending["future"]
        deadline = pending["started"] + _reply_timeout()
        patience_deadline = (
            pending["started"] + patience + _PATIENCE_MARGIN
            if patience is not None else deadline
        )
        while True:
            now = time.time()
            wait = max(0.0, deadline - now)
            if not pending.get("out_of_band_only"):
                wait = min(wait, max(0.0, patience_deadline - now))
            if keepalive:
                wait = min(wait, _SSE_KEEPALIVE)
            try:
                state, reply = fut.result(timeout=wait)
                return state, reply, pending.get("out_of_band_only", False), False
            except FuturesTimeout:
                now = time.time()
                if now >= deadline:
                    return (
                        protocol.STATE_FAILED, "[agent did not reply in time]",
                        pending.get("out_of_band_only", False), False,
                    )
                if keepalive:
                    try:
                        keepalive()
                    except Exception:
                        self._mark_out_of_band(pending, "[client disconnected]", pop_waiter=True)
                        # Task authority: the client is gone but the agent may
                        # still complete this task.  Do NOT finalize the task as
                        # FAILED here — the late agent reply must finalize the
                        # original task record.  The caller skips
                        # _finalize_task and returns a transient error to the
                        # HTTP client.
                        return (protocol.STATE_FAILED, "[client disconnected]", True, True)
                if now >= patience_deadline:
                    self._mark_out_of_band(pending, "[client patience exceeded]", pop_waiter=False)
                    # Keep waiting: when the reply resolves it must be pushed
                    # directly and the socket write skipped — the client is
                    # gone even though the socket may look alive.
            except Exception:
                return (
                    protocol.STATE_FAILED, "[agent did not reply in time]",
                    pending.get("out_of_band_only", False), False,
                )
    # ── Streaming (SSE) ───────────────────────────────────────────────────
    # ── Task queries ──────────────────────────────────────────────────────
    # ── Push notifications ────────────────────────────────────────────────
    # ── Sending (the agent's reply path) ──────────────────────────────────

    def _durable_complete_pending(self, task_id: str, chat_id: str, content: str, message_id: str) -> tuple[bool, str]:
        """Single durable completion coordinator for send paths.

        Stages the terminal candidate, calls TaskStore.publish_durable, and
        only after published=True/newly_published=True resolves/removes the
        Future and returns success.  On failed publish, keeps waiter/task
        coherent, leaves last durable state visible, and returns structured
        failure with no successful side effect.
        """
        # Stage candidate from current durable record — pending map/Future is NOT Task authority (Amendment D)
        rec = self.tasks.get(task_id)
        if rec is None:
            logger.warning("A2A: durable complete for unknown task %s — no authoritative TaskStore record (no fallback, Future unresolved)", task_id)
            return False, "task not found: no authoritative record"
        if rec.get("context_id") != chat_id:
            logger.warning("A2A: context mismatch for task %s: %r != %r", task_id, rec.get("context_id"), chat_id)
            return False, "context mismatch"
        if rec.get("state") in protocol.TERMINAL_STATES:
            # Already terminal — treat as not active for send authority
            return False, "task already terminal"
        candidate = dict(rec)
        candidate["state"] = protocol.STATE_COMPLETED
        candidate["reply"] = content or ""
        candidate["completed_at"] = __import__("time").time()
        try:
            outcome = self.tasks.publish_durable(_task_ledger_path(), task_id, candidate)
        except Exception as exc:
            logger.error("A2A: publish_durable exception for task %s: %s", task_id, exc, exc_info=True)
            return False, "A2A task state could not be durably published"
        if not outcome.published:
            logger.error("A2A: failed to durably publish COMPLETED for task %s: %s", task_id, outcome.error)
            return False, "A2A task state could not be durably published"
        # Publish succeeded — now resolve/remove Future atomically
        with self._pending_lock:
            ent = self._pending.get(task_id)
            if ent is not None and ent[0] == chat_id and not ent[1].done():
                try:
                    ent[1].set_result((protocol.STATE_COMPLETED, content or ""))
                except Exception:
                    pass
                order = self._pending_order.get(chat_id)
                if order is not None:
                    try:
                        order.remove(task_id)
                    except ValueError:
                        pass
                    if not order:
                        self._pending_order.pop(chat_id, None)
                self._pending.pop(task_id, None)
            elif ent is not None:
                # Pending exists but mismatched or already done — do not resolve again
                pass
        return True, ""

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Fulfil the pending reply Future for this context."""
        message_id = str(int(time.time() * 1000))
        # Task-authority: prefer the specific task via thread_id (ContextVar) to avoid cross-talk
        task_id_via_thread = ""
        try:
            from gateway.session_context import get_session_env
            task_id_via_thread = str(get_session_env("HERMES_SESSION_THREAD_ID") or "").strip()
        except Exception:
            task_id_via_thread = ""
        if task_id_via_thread:
            # Exact-thread path: pending or late TaskStore fallback, both disk-first
            with self._pending_lock:
                ent = self._pending.get(task_id_via_thread)
                has_pending = ent is not None and ent[0] == chat_id and not ent[1].done()
            if has_pending:
                ok, err = self._durable_complete_pending(task_id_via_thread, chat_id, content or "", message_id)
                if ok:
                    return SendResult(success=True, message_id=message_id)
                else:
                    return SendResult(success=False, message_id=message_id, error=err)
            # Late completion for disconnected task still WORKING — single durable commit
            try:
                rec_thr = self.tasks.get(task_id_via_thread)
                if rec_thr and rec_thr.get("context_id") == chat_id and rec_thr.get("state") not in protocol.TERMINAL_STATES:
                    logger.info("A2A: late completion for disconnected task %s (context %s) — finalizing original task record (thread_id path)", task_id_via_thread, chat_id)
                    # Use _finalize_task which is disk-first; it raises DurablePublishError on failure
                    try:
                        self._finalize_task({"task_id": task_id_via_thread, "context_id": chat_id, "peer": rec_thr.get("peer", ""), "started": rec_thr.get("created_at", time.time()), "created_iso": rec_thr.get("created_iso", "")}, protocol.STATE_COMPLETED, content or "", audit_direction="push")
                        return SendResult(success=True, message_id=message_id)
                    except protocol.DurablePublishError as dpe:
                        logger.error("A2A: late thread completion durability failed for %s: %s", task_id_via_thread, dpe)
                        return SendResult(success=False, message_id=message_id, error="A2A task state could not be durably published")
            except protocol.DurablePublishError:
                raise
            except Exception:
                pass
        if not task_id_via_thread and reply_to:
            cand = str(reply_to).strip()
            if cand:
                with self._pending_lock:
                    ent2 = self._pending.get(cand)
                    has_pending2 = ent2 is not None and ent2[0] == chat_id and not ent2[1].done()
                if has_pending2:
                    ok2, err2 = self._durable_complete_pending(cand, chat_id, content or "", message_id)
                    if ok2:
                        return SendResult(success=True, message_id=message_id)
                    else:
                        return SendResult(success=False, message_id=message_id, error=err2)
                # TaskStore fallback for reply_to — disk-first via _finalize_task
                try:
                    rec2 = self.tasks.get(cand)
                    if rec2 and rec2.get("context_id") == chat_id and rec2.get("state") not in protocol.TERMINAL_STATES:
                        logger.info("A2A: late completion for disconnected task %s (context %s) — finalizing via reply_to", cand, chat_id)
                        try:
                            self._finalize_task({"task_id": cand, "context_id": chat_id, "peer": rec2.get("peer", ""), "started": rec2.get("created_at", time.time()), "created_iso": rec2.get("created_iso", "")}, protocol.STATE_COMPLETED, content or "", audit_direction="push")
                            return SendResult(success=True, message_id=message_id)
                        except protocol.DurablePublishError as dpe:
                            logger.error("A2A: late reply_to completion durability failed for %s: %s", cand, dpe)
                            return SendResult(success=False, message_id=message_id, error="A2A task state could not be durably published")
                except protocol.DurablePublishError:
                    raise
                except Exception:
                    pass
        if not (metadata or {}).get("notify"):
            logger.debug("A2A: ignoring non-final send for context %s", chat_id)
            return SendResult(success=True, message_id=message_id)
        # Exact task authority per section 6.2/6.3
        if task_id_via_thread:
            logger.warning("A2A: thread_id %s not found/active for context %s — failing without fallback", task_id_via_thread, chat_id)
            return SendResult(success=False, message_id=message_id, error="task not found for thread_id")
        if reply_to and str(reply_to).strip():
            logger.warning("A2A: reply_to %s not found/active for context %s — failing without fallback", reply_to, chat_id)
            return SendResult(success=False, message_id=message_id, error="task not found for reply_to")
        # Context-only selection: count active tasks in this context
        _active_candidates = []
        with self._pending_lock:
            for _tid, (_ctx, _fut) in list(self._pending.items()):
                if _ctx == chat_id and not _fut.done():
                    _active_candidates.append(_tid)
        for _tid, _rec in list(self.tasks._tasks.items()):
            if _rec.get("context_id") == chat_id and _rec.get("state") not in protocol.TERMINAL_STATES and _rec.get("state") != protocol.STATE_SUBMITTED:
                if _tid not in _active_candidates:
                    _active_candidates.append(_tid)
        if len(_active_candidates) == 0:
            pass
        elif len(_active_candidates) == 1:
            _tid = _active_candidates[0]
            with self._pending_lock:
                _ent = self._pending.get(_tid)
                has_pen = _ent is not None and _ent[0] == chat_id and not _ent[1].done()
            if has_pen:
                ok3, err3 = self._durable_complete_pending(_tid, chat_id, content or "", message_id)
                if ok3:
                    return SendResult(success=True, message_id=message_id)
                else:
                    return SendResult(success=False, message_id=message_id, error=err3)
            # TaskStore fallback for the single active task — disk-first
            try:
                _rec = self.tasks.get(_tid)
                if _rec and _rec.get("context_id") == chat_id and _rec.get("state") not in protocol.TERMINAL_STATES:
                    logger.info("A2A: completing single active task %s for context %s via context-only fallback", _tid, chat_id)
                    try:
                        self._finalize_task(
                            {"task_id": _tid, "context_id": chat_id, "peer": _rec.get("peer", ""), "started": _rec.get("created_at", time.time()), "created_iso": _rec.get("created_iso", "")},
                            protocol.STATE_COMPLETED, content or "", audit_direction="push",
                        )
                        return SendResult(success=True, message_id=message_id)
                    except protocol.DurablePublishError as dpe:
                        logger.error("A2A: context fallback durability failed for %s: %s", _tid, dpe)
                        return SendResult(success=False, message_id=message_id, error="A2A task state could not be durably published")
            except protocol.DurablePublishError:
                raise
            except Exception:
                pass
        else:
            logger.warning("A2A: ambiguous task authority for context %s: %d active tasks", chat_id, len(_active_candidates))
            return SendResult(success=False, message_id=message_id, error="ambiguous task authority for context")
        # No waiter (e.g. a late chunk or out-of-band send) — push the message
        # back to the peer that owns this context as a NEW task, reusing the
        # same contextId so it lands in the caller's session (session
        # continuity). Without this, task-notifier wake replies and late
        # completions were silently dropped while reporting success.
        #
        # Loopback self-push guard: in localhost-only mode every inbound
        # caller authenticates as "ip:<addr>" with no port, so the only
        # resolvable target for a loopback identity is THIS gateway's own
        # endpoint. Self-pushing is correct for the notifier's
        # completion delivery — the sub's user_id is the loopback identity
        # and the message must re-enter the owning session (the watcher marks
        # that send with metadata["a2a_push"]=True). A session's own REPLY
        # must never be re-queued into the same session: that produced an
        # unbounded self-ping-pong once the loopback fallback became
        # resolvable — every reply was pushed back,
        # processed, and answered again forever. Unmarked sends to a
        # loopback peer are replies with no external destination — that is
        # a LOUD failure, not a silent success: a
        # helper-sent message refined to "ip:127.0.0.1" and its long reply
        # was dropped here with success=True and no
        # audit. The notifier/engine must rewind instead of advancing past a
        # lost event.
        #
        # Missing-peer guard: if no peer is registered for this context,
        # the push has no destination and MUST fail explicitly — reporting
        # success here would silently advance the Kanban cursor past a
        # lost event (the reviewer's finding: no-peer false success).
        if not (metadata or {}).get("a2a_push"):
            with self._context_peers_lock:
                _loop_peer = self._context_peers.get(chat_id, "")
            if _loop_peer and _loopback_fallback_url(_loop_peer, self.host, self.port):
                security.audit(
                    "push_dropped", _loop_peer, message_id,
                    "peer identity not resolvable", context_id=chat_id,
                )
                logger.warning(
                    "A2A: dropping out-of-band send for %s: loopback peer %r "
                    "is unresolvable and the message is an unmarked session "
                    "reply — no external destination (success=False)",
                    chat_id, _loop_peer,
                )
                return SendResult(
                    success=False, message_id=message_id,
                    error="peer identity not resolvable",
                )
        # Explicit no-peer failure: the context has no registered peer
        # and no loopback identity — there is nowhere to push to.
        with self._context_peers_lock:
            _push_peer = self._context_peers.get(chat_id, "")
        if not _push_peer:
            security.audit(
                "push_dropped", "", message_id,
                "no peer registered for context", context_id=chat_id,
            )
            logger.warning(
                "A2A: out-of-band send for %s has no registered peer; "
                "reporting failure (success=False)",
                chat_id,
            )
            return SendResult(
                success=False, message_id=message_id,
                error="no peer registered for context",
            )
        try:
            outcome = await asyncio.to_thread(
                self._push_out_of_band, chat_id, content or "",
                not (metadata or {}).get("a2a_push"),
            )
            # _push_out_of_band now returns typed PushOutcome (Amendment B)
            if not outcome.success:
                return SendResult(success=False, message_id=message_id, error=f"{outcome.category}: {outcome.error}")
        except Exception as exc:
            logger.warning("A2A: out-of-band push for context %s failed: %s", chat_id, exc)
            return SendResult(success=False, message_id=message_id, error=str(exc))
        return SendResult(success=True, message_id=message_id)
    def _push_out_of_band(self, context_id: str, text: str, want_reply: bool = False) -> protocol.PushOutcome:
        """POST a new message/send to the peer that owns ``context_id``. Returns typed PushOutcome (Amendment A/B)."""
        with self._context_peers_lock:
            peer = self._context_peers.get(context_id, "")
        if not peer:
            logger.debug("A2A: out-of-band send for %s has no known peer; dropping", context_id)
            return protocol.PushOutcome(success=False, category="routing", error="no peer registered for context")
        from . import tools as a2a_tools

        entry = a2a_tools._resolve_peer(peer)
        if not entry or not entry.get("url"):
            # Localhost-only mode records inbound callers as "ip:<addr>" with
            # no port, and there is no a2a_agents key for the raw identity.
            # When the identity is a loopback address, the notifier path
            # falls back to this gateway's own A2A endpoint — the one local
            # endpoint guaranteed to route a same-contextId follow-up into
            # the session that owns the conversation (a registered
            # loopback "ip:" identity carries no port, so without the
            # fallback the push dropped).
            fallback = _loopback_fallback_url(peer, self.host, self.port)
            if fallback:
                if want_reply:
                    self._drop_unresolvable_reply(context_id, peer)
                    return protocol.PushOutcome(success=False, category="routing", error="peer identity not resolvable for reply")
                logger.info(
                    "A2A: out-of-band send for %s: identity %r not in a2a_agents; "
                    "falling back to local endpoint %s",
                    context_id, peer, fallback,
                )
                # The fallback URL is THIS gateway's own endpoint (it is
                # built from self.host/self.port), so an HTTP round-trip
                # would be a synchronous self-call: the inbound handler only
                # answers after the agent session processes the message,
                # which routinely exceeds the client timeout — the audit
                # row + reply log never ran and the notifier logged a false
                # failure. Deliver in-process
                # instead: the exact same code path as an inbound
                # message/send, minus the connection and the wait.
                return self._push_loopback_in_process(context_id, peer, text, want_reply=False)
            else:
                # Stale/unresolvable peer: a registered peer identity that
                # can't be resolved to a URL.  Loud failure so notifier/cursor
                # logic cannot advance over a dropped event.
                security.audit(
                    "push_dropped", peer, context_id,
                    "registered peer not resolvable", context_id=context_id,
                )
                logger.warning(
                    "A2A: out-of-band send for %s: peer %r registered but "
                    "not resolvable — delivery dropped",
                    context_id, peer,
                )
                return protocol.PushOutcome(success=False, category="routing", error="registered peer not resolvable")
        base_url = entry["url"]
        # Own-endpoint guard: if the resolved target is THIS gateway (the
        # context→peer map can be refined to our own URL — an in-process
        # loopback push stamps our own sender, and the inbound refinement
        # accepts it), deliver in-process instead of a synchronous HTTP
        # self-call. The inbound handler only answers after the session
        # processes the message, which routinely exceeds the client timeout.
        # Reply pushes (want_reply=True) refuse this fallback — an
        # unresolvable reply peer fails loudly.
        if _is_own_endpoint(base_url, self.host, self.port):
            if want_reply:
                self._drop_unresolvable_reply(context_id, peer)
                return protocol.PushOutcome(success=False, category="routing", error="peer identity not resolvable for reply (own endpoint)")
            logger.info(
                "A2A: out-of-band send for %s: resolved peer %r is this gateway "
                "(%s); delivering in-process",
                context_id, peer, base_url,
            )
            return self._push_loopback_in_process(context_id, peer, text, want_reply=False)
        headers = {**a2a_tools._auth_header(entry.get("auth") or {}), **(entry.get("headers", {}) or {})}
        timeout = int(entry.get("timeout", 120))
        allowed = tuple(a2a_tools._allowed_rpc_origins(entry))
        card = None
        try:
            card = a2a_tools._fetch_card(base_url, headers, min(timeout, 30), allowed)
        except Exception:
            pass
        rpc_url = a2a_tools._rpc_url(base_url, card)
        if not a2a_tools._origin_allowed(rpc_url, entry):
            logger.warning(
                "A2A: peer '%s' card advertised cross-origin RPC URL %s; not in "
                "peer's allowed_rpc_origins — using configured origin %s instead",
                peer, rpc_url, base_url)
            rpc_url = base_url.rstrip("/")
        rpc_body = {
            "jsonrpc": "2.0",
            "id": protocol.new_task_id(),
            "method": "SendMessage",
            "params": {
                "message": protocol.text_message(
                    protocol.ROLE_USER, text, context_id=context_id, sender=self._sender_identity()
                ),
            },
        }
        tenant = a2a_tools._interface_tenant(card, entry)
        if tenant:
            rpc_body["params"]["tenant"] = tenant
        resp = None
        _push_outcome: protocol.PushOutcome = protocol.PushOutcome(success=False, category="transport", error="unknown")
        try:
            resp = a2a_tools._http_post_json(rpc_url, rpc_body, headers, timeout, allowed_origins=allowed)
            if isinstance(resp, dict) and "error" in resp:
                _redacted, _payload = _redacted_jsonrpc_detail(resp["error"])
                logger.warning("A2A: out-of-band push for context %s got JSON-RPC error: %s", context_id, _redacted)
                _push_outcome = protocol.PushOutcome(success=False, category="jsonrpc", error=_redacted, payload=_payload)
            elif resp is None or not isinstance(resp, dict):
                logger.warning("A2A: out-of-band push for context %s got invalid response: %r", context_id, resp)
                _push_outcome = protocol.PushOutcome(success=False, category="transport", error=f"invalid response: {resp!r}")
            else:
                # Strict V1 parsing for push result
                try:
                    _parsed = protocol.parse_send_message_result(resp.get("result"), "V1_WRAPPED")
                    _push_outcome = protocol.PushOutcome(success=True, category="transport", error="", payload=_parsed.payload)
                except protocol.A2AResultValidationError as ve:
                    logger.warning("A2A: out-of-band push for context %s got malformed/invalid result: %r (%s)", context_id, resp.get("result"), ve.reason)
                    _push_outcome = protocol.PushOutcome(success=False, category="invalid_response", error=f"{ve.reason}: {ve.detail}", payload=None)
        except Exception as exc:
            logger.warning("A2A: out-of-band push for context %s failed: %s", context_id, exc)
            _push_outcome = protocol.PushOutcome(success=False, category="transport", error=str(exc))
        finally:
            # Amendment A: conversation agent entry is evidence of validated successful push only.
            # All failure categories must NOT persist agent entry, must emit exactly one failure audit, no success metric/log.
            if _push_outcome.success:
                protocol.persist_message(context_id, "agent", text)
                security.audit("push", peer, rpc_body["id"], text, context_id=context_id)
                logger.info("A2A: pushed out-of-band reply for context %s to peer %s", context_id, peer)
            else:
                # Failure-only audit with redacted detail; no conversation persist, no success audit/metric/log.
                try:
                    security.audit("push_failed", peer, rpc_body["id"], _push_outcome.error, context_id=context_id)
                except Exception:
                    pass
                logger.warning("A2A: out-of-band push for context %s to peer %s failed or got invalid result: %s", context_id, peer, _push_outcome.error)
        if not _push_outcome.success:
            return _push_outcome
        if want_reply and resp is not None and _push_outcome.payload is not None:
            # Round-trip: the peer answered inside the push's HTTP
            # response — surface non-empty reply into LOCAL gateway.
            # Use validated payload text, not second permissive unwrap.
            try:
                _validated = _push_outcome.payload
                if isinstance(_validated, dict) and "parts" in _validated:
                    reply = protocol.extract_text(_validated)
                else:
                    # Task: extract from status.message or artifacts
                    reply = ""
                    if isinstance(_validated, dict):
                        status = _validated.get("status", {}) or {}
                        msg = status.get("message") if isinstance(status, dict) else None
                        if isinstance(msg, dict):
                            reply = protocol.extract_text(msg)
                        if not reply:
                            for art in _validated.get("artifacts", []) or []:
                                reply = protocol.extract_text(art)
                                if reply:
                                    break
                        if not reply:
                            reply = a2a_tools._reply_text_from_result(_validated)
                if reply:
                    # Loopback for reply surfacing must also be typed but fire-and-forget
                    loop_res = self._push_loopback_in_process(context_id, peer, reply, want_reply=True)
                    if not loop_res.success:
                        logger.warning("A2A: surfaced push reply loopback failed for %s: %s", context_id, loop_res.error)
            except Exception as exc:
                logger.warning(
                    "A2A: could not surface push reply for context %s: %s",
                    context_id, exc,
                )

        return _push_outcome
    def _drop_unresolvable_reply(self, context_id: str, peer: str) -> None:
        """Loud failure for a reply push with no resolvable external target."""
        security.audit(
            "push_dropped", peer, "", "peer identity not resolvable",
            context_id=context_id,
        )
        logger.warning(
            "A2A: out-of-band REPLY for context %s dropped: peer identity %r "
            "is not resolvable (no external destination)",
            context_id, peer,
        )

    def _push_reply_after_client_gone(self, req_id: Any, result: Optional[dict], is_v1: bool = True) -> protocol.PushOutcome:
        """Deliver a completed reply whose HTTP client disconnected first. Returns typed PushOutcome."""
        try:
            inner = (result or {}).get("result")
            _mode = "V1_WRAPPED" if is_v1 else "LEGACY_BARE"
            try:
                _parsed = protocol.parse_send_message_result(inner, _mode)
            except protocol.A2AResultValidationError as ve:
                logger.warning("A2A: rescue found invalid result for req %s: %s (%s)", req_id, ve.reason, ve.detail)
                return protocol.PushOutcome(success=False, category="invalid_response", error=f"{ve.reason}: {ve.detail}")
            if _parsed.kind == "task":
                context_id = _parsed.context_id
                state = _parsed.state
                reply = _parsed.text
            else:
                logger.debug("A2A: rescue got message result, not task terminal, skipping")
                return protocol.PushOutcome(success=False, category="routing", error="message result not pushable via rescue")
            if not context_id or state not in (
                protocol.STATE_COMPLETED, protocol.STATE_INPUT_REQUIRED,
            ):
                logger.debug(
                    "A2A: not pushing reply after client disconnect for %s (state=%r)",
                    context_id, state,
                )
                return protocol.PushOutcome(success=False, category="routing", error=f"state not pushable: {state!r}")
            if not reply:
                return protocol.PushOutcome(success=False, category="routing", error="no reply to push")
            outcome = self._push_out_of_band(context_id, reply, want_reply=True)
            if not outcome.success:
                logger.warning(
                    "A2A: rescue push for context %s failed — reply not delivered (want_reply=True): %s",
                    context_id, outcome.error,
                )
                return outcome
            logger.info(
                "A2A: client disconnected before response write; pushed reply "
                "for context %s out-of-band",
                context_id,
            )
            return outcome
        except Exception as exc:
            logger.warning(
                "A2A: could not push reply after client disconnect (req %s): %s",
                req_id, exc,
            )
            return protocol.PushOutcome(success=False, category="transport", error=str(exc))

    def _push_loopback_in_process(self, context_id: str, peer: str, text: str,
                                  want_reply: bool = False) -> protocol.PushOutcome:
        """Deliver an out-of-band push to this gateway's own session in-process. Returns typed PushOutcome (Amendment B).

        Fire-and-forget loopback durably creates WORKING before dispatch and durably publishes COMPLETED
        before any terminal side effects. Failed WORKING leaves ABSENT; failed COMPLETED leaves WORKING,
        unresolved Future/watcher, no terminal side effects, and category durability.
        """
        params = {
            "message": protocol.text_message(
                protocol.ROLE_USER, text, context_id=context_id, sender=self._sender_identity()
            ),
        }
        try:
            terminal, pending = self._prepare_task(params, peer)
        except protocol.DurablePublishError as dpe:
            logger.error("A2A: loopback WORKING publish failed for context %s: %s", context_id, dpe)
            _audit_loopback_failure(peer, context_id, f"durability failure: {dpe}", "durability", task_id=getattr(dpe, "task_id", "") or "")
            return protocol.PushOutcome(success=False, category="durability", error=f"durability failure: {dpe}")
        except Exception as exc:
            logger.error("A2A: loopback _prepare_task exception for context %s: %s", context_id, exc)
            _audit_loopback_failure(peer, context_id, str(exc), "transport")
            return protocol.PushOutcome(success=False, category="transport", error=str(exc))
        if terminal is not None:
            state = (terminal.get("status") or {}).get("state", "unknown")
            logger.warning("A2A: loopback push for context %s rejected (%s)", context_id, state)
            _audit_loopback_failure(peer, context_id, f"rejected: {state}", "routing", task_id=str(terminal.get("id", "")) if isinstance(terminal, dict) else "")
            return protocol.PushOutcome(success=False, category="routing", error=f"rejected: {state}")
        assert pending is not None  # _prepare_task returns (terminal, None) or (None, pending)
        if want_reply:
            # Session-reply path: the task stays pending for send() to resolve.
            # WORKING already durably created; emit success side effects for the inbound leg.
            try:
                protocol.persist_message(context_id, "agent", text)
                security.audit("push", peer, pending["task_id"], text, context_id=context_id)
                logger.info("A2A: pushed out-of-band reply for context %s to peer %s (want_reply)", context_id, peer)
                return protocol.PushOutcome(success=True, category="transport", error="")
            except Exception as exc:
                logger.error("A2A: loopback want_reply side effect failed for %s: %s", context_id, exc)
                _audit_loopback_failure(peer, context_id, str(exc), "durability", task_id=pending.get("task_id", "") if isinstance(pending, dict) else "")
                return protocol.PushOutcome(success=False, category="durability", error=str(exc))
        else:
            # Fire-and-forget (notifier): complete the task immediately
            # Durable COMPLETED publication must precede terminal effects and success (Amendment B).
            try:
                self._finalize_task(pending, protocol.STATE_COMPLETED, text, audit_direction="push")
                logger.info("A2A: delivered fire-and-forget loopback for context %s (task %s completed)", context_id, pending["task_id"])
                return protocol.PushOutcome(success=True, category="transport", error="")
            except protocol.DurablePublishError as dpe:
                logger.error("A2A: loopback COMPLETED publish failed for context %s task %s: %s", context_id, pending.get("task_id", ""), dpe)
                _audit_loopback_failure(peer, context_id, f"durability failure: {dpe}", "durability", task_id=pending.get("task_id", "") if isinstance(pending, dict) else "")
                return protocol.PushOutcome(success=False, category="durability", error=f"durability failure: {dpe}")
            except Exception as exc:
                logger.error("A2A: loopback finalize exception for context %s: %s", context_id, exc)
                _audit_loopback_failure(peer, context_id, str(exc), "transport", task_id=pending.get("task_id", "") if isinstance(pending, dict) else "")
                return protocol.PushOutcome(success=False, category="transport", error=str(exc))


    async def on_processing_complete(self, event, outcome):
        # Delegate to TaskRPCHandler's implementation (which handles deferred failure/cancel persistence)
        # This wrapper ensures TaskRPCHandler takes precedence over BasePlatformAdapter's no-op hook
        # despite MRO BasePlatformAdapter -> TaskRPCHandler.
        from .task_routing import TaskRPCHandler as _TRH
        return await _TRH.on_processing_complete(self, event, outcome)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": f"a2a:{chat_id}", "type": "dm"}
