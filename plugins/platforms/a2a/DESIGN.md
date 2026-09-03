# A2A Platform Plugin — Design

Consolidates the entire A2A (Agent-to-Agent) feature cluster (#514 and friends)
into one **plugin** with **zero core edits**, built on capabilities the current
codebase already exposes. Implements **A2A Protocol v1.0** (JSON-RPC binding).

## Why a plugin, not a core feature

Earlier A2A attempts (#4135, #4948, #4952, #11025) added a standalone server
package (`a2a_adapter/`) and/or patched `gateway/run.py` + `gateway/config.py`.
Since then the codebase grew `ctx.register_platform()` (the plugin
platform-adapter API — used by irc, line, teams, ntfy, simplex, …) and
`ctx.register_tool()`. That makes the standing policy achievable: **plugins
must not touch core files.** A2A now lives entirely under
`plugins/platforms/a2a/`.

## Two directions

### Outbound — client tools (`a2a` toolset)
- `a2a_discover(url)` — fetch + summarize a peer's Agent Card (v1.0
  `supportedInterfaces` aware, tolerates 0.3 cards).
- `a2a_call(agent, message, context_id?)` — send a JSON-RPC `message/send`
  task to a peer, return the reply. Multi-turn via `context_id` (carried
  inside the Message per v1.0). Surfaces `TASK_STATE_INPUT_REQUIRED` so the
  model knows to answer and continue the context.
- `a2a_list()` — configured peers + persisted conversations + metrics.
- `a2a_history(context_id, limit?)` — recall a persisted conversation
  (this is the production consumer of the persistence layer).
- `a2a_orchestrate(capability, message, mode?)` — fan-out one task to every
  configured peer advertising a capability. Modes: `all` (every reply),
  `first` (first success), `best` (longest successful reply — a deliberately
  coarse heuristic; errors never win, and an all-error fan-out reports the
  failures instead of picking one).

Peers resolved from `config.yaml` → `a2a_agents`, or a direct URL.

### Inbound — platform adapter
- Stdlib `http.server` on a daemon thread (no asyncio loop needed at
  `register()` time — sidesteps the a2a_fleet "register outside a loop" bug
  class that killed inbound serving in forks). The request handler is a
  module-level class (`A2ARequestHandler`) reached through
  `server.adapter`, so RPC handlers are unit-testable without HTTP.
- Agent Card at `GET /.well-known/agent-card.json` (canonical v1.0 path; legacy `agent.json` also answers) (v1.0: `supportedInterfaces[]`,
  `provider`, `capabilities.extendedAgentCard`). **Dynamic**: skills are
  built from the live tool registry at serve time
  (`A2A_ADVERTISED_TOOLSETS` / `extra.advertised_toolsets` restricts them).
- JSON-RPC methods: `message/send`, `message/stream` (SSE), `tasks/get`,
  `tasks/list`, `tasks/cancel`, `tasks/subscribe`,
  `tasks/pushNotificationConfig/create` (legacy `set` names accepted).
- **Live-session injection (the #11025 insight):** inbound tasks route through
  the normal `MessageEvent` → `handle_message` path keyed by the A2A
  `contextId`, so the agent that answers is the same one serving the user —
  full memory/context, not a clone. The reply returns through `adapter.send()`,
  which fulfils the pending per-**task** `Future` the HTTP request is blocked
  on (per-context FIFO, so concurrent same-context requests can't cross-talk);
  `on_processing_complete` resolves failures/cancellations promptly.
- **Task store:** every task (including terminal ones, bounded to the last
  500) stays queryable via `tasks/get` / `tasks/list`, and `tasks/subscribe`
  reattaches to a running task's stream via store watchers. A watchdog fails
  orphaned tasks after 5 minutes (idempotent transitions — no double
  counting in metrics).
- **input-required:** the platform hint tells the agent to start a reply with
  `[INPUT_REQUIRED]` when it needs clarification; the adapter maps that to
  `TASK_STATE_INPUT_REQUIRED` with the question in `status.message`.
- **Push notifications:** config accepted inline in `message/send`
  (`configuration.taskPushNotificationConfig`) or via the create method
  (returns `configId` + `createdAt`). On terminal transition the callback
  receives a v1.0 `StreamResponse` (`statusUpdate`) payload, HMAC-SHA256
  signed (`X-A2A-Signature`, secret `A2A_PUSH_SECRET` falling back to the
  bearer token), with SSRF-guarded callback URLs.

## v1.0 wire format notes
- Task states / roles are SCREAMING_SNAKE_CASE (TASK_STATE_*, ROLE_*).
- Parts are member-presence discriminated — no kind field. All three
  Part types are supported: text (text + mediaType), file
  (url|raw + filename + mediaType), and data (data + mediaType).
  extract_text renders file/data Parts into the text stream (URL +
  filename for files, JSON for data) so the agent sees them; it also
  accepts v0.3 (kind) and pre-0.3 (type) shapes from older peers.
  Outbound replies are still text-only — the agent produces text, and
  file/data Parts are for inbound richness.
- Push notification config: full CRUD — create (inline in message/send
  via configuration.taskPushNotificationConfig, or via the create
  method), get, list, delete. Each config has a configId and createdAt.
  One config per task (v1.0 allows multiple; we keep one).
- SSE events are StreamResponse objects (statusUpdate / artifactUpdate
  members); stream closure signals the terminal state — no final field.
- contextId lives inside the Message (legacy top-level accepted inbound).
- Timestamps are ISO 8601 with millisecond precision; Tasks carry
  createdAt / lastModified.
- Error codes: A2A-reserved codes are used only with their spec semantics
  (`-32001` TaskNotFound, `-32002` TaskNotCancelable); custom errors sit at
  `-32050..-32052` (unauthorized / rate-limited / untrusted).

## Security (on by default)
- **Bind safety:** no token configured (`A2A_BEARER_TOKEN` or
  `A2A_PEER_TOKENS`) ⇒ bind `127.0.0.1` only. A token alone does not widen
  the bind; remote exposure requires token **and** explicit `A2A_HOST`.
- **Peer identity:** `A2A_PEER_TOKENS="alice:tok1,bob:tok2"` gives each peer
  its own credential; the matched name is the authenticated identity used
  for rate limiting, the trust gate, message framing, and audit. A shared
  `A2A_BEARER_TOKEN` authenticates as `ip:<addr>`. Nothing in the request
  body can assert identity. Comparisons are constant-time.
- **Trust gate:** `A2A_TRUSTED_PEERS` (or config `a2a.trusted_peers`)
  optionally restricts which authenticated identities may run tasks.
- **Injection filters:** ALL inbound text (including `/`-prefixed — remote
  peers can never reach operator slash commands) is defanged (ChatML /
  role-prefix / override patterns → `[filtered]`) and framed with a privacy
  prefix marking it untrusted peer input.
- **Outbound redaction:** credential-shaped strings (`sk-…`, `ghp_…`, JWTs,
  bearer tokens, emails) scrubbed before anything leaves.
- **Rate limiting:** sliding window per authenticated identity
  (`A2A_RATE_LIMIT`/min).
- **Anti-loop:** per-context turn cap (`A2A_MAX_PINGPONG_TURNS`, default 5,
  hard max 20) rejects (v1.0 `TASK_STATE_REJECTED`) runaway agent↔agent
  ping-pong; `tasks/cancel` resets the counter for the task's context.
- **Audit log:** append-only `~/.hermes/a2a_audit.jsonl` for every exchange.

## State placement
Task store, turn tracker, rate limiter, and **windowed duplicate suppression**
(`_inbound_seen` map, 60 s window, 1,024 entries, process-local) are
**adapter-instance** objects (classes / maps in `protocol.py` / `adapter.py`).
The metrics counter bag stays a module singleton because it is intentionally
shared between the inbound adapter and the outbound client tools
(`/metrics` and `a2a_list` report both directions).

**Windowed duplicate suppression** is bounded admission control, not durable
idempotency or replay protection: the key is process-local
`(contextId, messageId)`, entries expire after 60 s, the map is capped at
1,024, it is not persisted, restart forgets it, and a duplicate gets a new
`REJECTED` Task rather than the first request's Task/result. It does not
cause request replay, result replay, or exactly-once execution.

## Persistence (survives compaction)
A2A conversations are written to `~/.hermes/a2a_conversations/<context>.jsonl`,
outside the context-compaction pipeline — compaction and restarts can't lose
them (#11025 requirement). The `a2a_history` tool recalls them by context id.

## Requirements traced to the cluster

| Source | Requirement | Where |
|---|---|---|
| #514, #23871, #4135 | Agent Card discovery | `protocol.build_agent_card`, adapter GET |
| #4135, #14559, #8948 | Client: discover / call / list | `tools.py` |
| #11025 | Live-session injection (not a clone) | `adapter._prepare_task` |
| #11025 | Privacy filters + outbound redaction + audit | `security.py` |
| #11025 | Conversation persistence outside compaction | `protocol.persist_message`, `a2a_history` |
| #514, #11025 | Auth, localhost-default | `security.authenticate`, `resolve_bind_host` |
| #56434 | Trusted peer approval | `security.is_trusted_peer` |
| #56435 | Task completion notifications | push notifications (`_send_push_notification`) |
| #25176, #689 | Agent↔agent messaging across machines | client tools + inbound adapter |
| #7517 et al. | Multi-peer orchestration | `a2a_orchestrate` |

## Deliberately out of scope (future, not this pass)
- **a2a-sdk / gRPC + HTTP+JSON bindings.** Only the JSONRPC binding is
  served; the card advertises exactly that.
- **`tenant` field, extended Agent Card, `stateTransitionHistory`.**
- **True task abort:** `tasks/cancel` marks the task canceled and drops the
  reply, but cannot abort the live session's in-flight turn.
- **DID / Ed25519 identity, OAuth2 scopes, x402 micropayments** (#14559
  bindu) — heavy, niche; revisit if there's real demand.

## Edison re-baseline (2026-09-03) — supersession notes

This re-baseline supersedes the following prior assumptions; the
terminology below is canonical:

1. `protocol.is_valid_a2a_result` no longer defines validity by
   meaningful-key presence. The strict parser/schema in this doc's
   §4 (Task/Message/Part/Artifact rules, exact-one wrapper, explicit
   `V1_WRAPPED` vs `LEGACY_BARE`) is authoritative.

2. `unwrap_send_message_response` may not select `task` from a
   both-member wrapper. The oneof contract requires `v1_payload_count`
   failure; production callers use `parse_send_message_result`.

3. The `TaskStore.complete() then persist()` pattern is replaced by the
   disk-first durable publication primitive
   `TaskStore.publish_durable(ledger_path, task_id, candidate_record)`:
   stage clone → atomically replace ledger → update memory → wake
   observers → post-commit audit/metrics/callback/push → return success.
   No `memory terminal → persist → return` path is permitted.

4. `adapter.send()` per-context FIFO does not prevent cross-talk.
   Exact `task_id` (via `HERMES_SESSION_THREAD_ID` or `reply_to`) plus
   `contextId`/`peer`/`agent_slug`/`tenant` verification prevents it;
   context-only selection is valid only when exactly one active task
   exists in the context. Two concurrent same-context tasks via FIFO is
   replaced by exact-ID and ambiguity tests.

5. `DESIGN.md` statements that task transitions are merely “idempotent”
   mean **terminal-state immutability inside one TaskStore**. They do not
   mean durable request idempotency, exactly-once, or at-least-once
   delivery.

6. The `DESIGN.md` Part compatibility statement remains valid for
   tolerant inbound `extract_text` only. It does not loosen successful
   `v1` result validation.

7. The windowed inbound dedupe map is renamed to **windowed duplicate
   suppression** and retained under §8 of the decision: 60 s, 1,024
   entries, process-local `(contextId, messageId)`, bounded admission
   control, not durable idempotency or replay protection.

8. Prior successful transport probes remain authoritative preservation
   evidence, but aggregate green counts do not override the hostile
   predicates in the durability matrix.

## Amendment ac32ee — durability correction (2026-09-03)

This amendment locks the five residual boundaries that the prior
Edison artifact left ambiguous. It does not reopen parser, peer
identity, shutdown, transport trust, or dedupe decisions.

### A. Push result, conversation, and audit ownership

A conversation `persist_message(context_id, "agent", ...)` entry is
evidence of a validated successful push — not transport bookkeeping.

| Outcome | `PushOutcome` | Conversation `persist_message(..., "agent", ...)` | Audit | Success log/metric |
|---|---|---|---|---|
| Valid v1 result | `success=True` | Exactly once | Exactly one success `push` audit | Permitted once |
| JSON-RPC top-level `error` | `success=False, category="jsonrpc"` | Prohibited | Exactly one `push_failed` with redacted peer code/message | Prohibited |
| Malformed/foreign result | `success=False, category="invalid_response"` | Prohibited | Exactly one `push_failed` | Prohibited |
| Transport/timeout/no response | `success=False, category="transport"` | Prohibited | Exactly one `push_failed`; detail says indeterminate | Prohibited |
| Routing failure | `success=False, category="routing"` | Prohibited | Exactly one `push_dropped` or `push_failed` | Prohibited |
| Local durable failure | `success=False, category="durability"` | Prohibited | Exactly one `push_failed` durability audit | Prohibited |

Transport uncertainty permits one failure audit only; it does not
permit a conversation entry, success `push` audit, or success log.
JSON-RPC error is a stronger operation failure and also forbids a
conversation entry.

### B. Typed loopback propagation

Production return contracts are exact:

- `_push_loopback_in_process(...) -> PushOutcome`
- `_push_out_of_band(...) -> PushOutcome`
- `_try_push_reply(...) -> PushOutcome`
- `_push_reply_after_client_gone(...) -> PushOutcome`
- `adapter.send(...) -> SendResult`

No production branch returns `True`/`False`/`None` in place of
`PushOutcome`; no bool-compatibility branch hides failure.

For fire-and-forget loopback: durable WORKING creation precedes
local dispatch; durable COMPLETED publication precedes terminal
conversation/audit/log/watcher/success.  Failed WORKING leaves
ABSENT; failed COMPLETED leaves memory/disk WORKING, unresolved
Future/watcher, no terminal side effect, and `category="durability"`
through every caller. `adapter.send` maps it to
`SendResult(success=False, error=<category plus detail>)`.  A
durable terminal task is never rolled back because later network
delivery fails.

### C. fsync and atomic publication

Under the established lock order, the ledger is written via a
temporary file that is fully flushed and file-fsynced before
`os.replace`.  Serialization, flush, temp-file fsync, and replace
exceptions are publication failures: the temp file is cleaned where
possible, memory/observers are not updated, and the store returns
`DurablePublishOutcome(published=False, newly_published=False, ...)`.

Directory fsync is attempted after replace.  An unsupported
capability (`AttributeError`, `NotImplementedError`, `EINVAL`,
`ENOTSUP`, `EOPNOTSUPP`) falls back once per process to the weaker
guarantee (file-fsync + atomic replace) with a single warning;
it does not claim full directory-entry persistence.  Unexpected I/O
(`EIO`, `ENOSPC`, permission loss, unclassified `OSError`) fails
closed: after a post-replace unexpected error the store returns a
structured durability failure with `safeToRetry=false`, resolves no
watcher/Future, emits no success side effect, and marks the ledger
unavailable until a fresh locked reload or restart re-establishes
authority.  No A2A TaskState `INDETERMINATE` is invented.

### D. Missing authoritative Task record

A pending map/Future is not Task authority.  When
`_durable_complete_pending(task_id, ...)` cannot read an
authoritative `TaskStore` record for `task_id` it returns failure,
`adapter.send` returns `SendResult(success=False)` with
task-authority/durability detail, the Future remains unresolved, the
pending and pending-order entries are retained for reconciliation or
shutdown, no Task is created, no replacement ID is selected, no
context/FIFO fallback follows an explicit task ID, and no terminal
conversation/audit/metric/callback/success log is emitted.  Tests
must create a durable WORKING record before using a pending Future;
production contains no memory-only success fallback.

### E. Same-task terminal authority across TaskStores

The per-ledger interprocess file lock owns same-task serialization.
For every `publish_durable`:

1. Acquire the in-process lock, then the per-ledger file lock in the
   established order, and load the authoritative ledger under that
   lock; an unreadable/unparseable ledger fails closed and is never
   replaced with an empty snapshot.
2. Compare ownership and terminal state against the disk record, not
   only `self._tasks`.
3. Existing terminal + identical state/reply: return the disk record
   with `published=True, newly_published=False`; do not rewrite or
   repeat side effects.
4. Existing terminal + conflicting state/reply: return
   `published=False, newly_published=False` with terminal-conflict
   error; do not rewrite or resolve observers.
5. Reconcile the caller cache to the disk record before both
   terminal returns.
6. Only a nonterminal authoritative record may take a legal candidate
   transition; unrelated IDs may merge without stale same-task
   overwrite.

### F. Local loopback failure audit cardinality (Wave 14 — b7384ce correction)

Every local loopback `PushOutcome` failure emits exactly one failure-only
audit and no success side effect. ` _push_loopback_in_process` is the
central audit seam for loopback: WORKING publish failure and COMPLETED
publish failure emit one `push_failed` with `category="durability"`;
terminal rejection emits one `push_dropped` with `category="routing"`;
want-reply side-effect failure emits one `push_failed` durability audit.
The loopback helper audits before returning; `_push_out_of_band`'s two
in-process branches (`2171`, `2204`) return that outcome unchanged without
re-auditing, and `_try_push_reply` / `_push_reply_after_client_gone` /
rescue / `adapter.send` preserve the typed `PushOutcome`/`SendResult`
propagation without double-auditing. No `persist_message(..., "agent", ...)`,
success-direction `push` audit, success metric, terminal callback, or
success log is emitted on failure. Failed terminal publication leaves
memory/disk WORKING and observers/Futures unresolved, and `adapter.send`
maps durability to `SendResult(success=False, error=<durability plus detail>)`.
This is an audit-event cardinality guarantee, not an exactly-once delivery
guarantee (see §8 bounded duplicate suppression). `_drop_unresolvable_reply`
retains its existing `push_dropped` for reply-path unresolvable peers.

### G. JSON-RPC peer error redaction (Wave 14 — b7384ce correction)

A JSON-RPC top-level `error` is never persisted or returned verbatim.
`_push_out_of_band` builds a bounded redacted detail via
`security.redact_outbound` before constructing `PushOutcome`: raw
`resp["error"]` (code/message, including bearer-shaped sentinels) is
redacted, truncated to 300 chars, and stored as `PushOutcome.error`;
`PushOutcome.payload` carries the same redacted copy (dict values redacted
individually). The warning log for the peer error and the `push_failed`
audit both use the redacted detail; no raw peer code/message reaches
callers, logs, or the audit ledger. `category="jsonrpc"` and the failure
mapping are retained; no `security.py` change was required.

## Files
```
plugins/platforms/a2a/
├── plugin.yaml      # manifest (kind: platform)
├── __init__.py      # register(): platform adapter + client tools
├── adapter.py       # inbound A2A v1.0 server (stdlib http.server)
├── tools.py         # outbound client tools
├── protocol.py      # Agent Card, JSON-RPC framing, task store, persistence + strict parser + durable primitive
├── security.py      # auth/identity, injection filters, redaction, audit
├── DESIGN.md
└── README.md
```
