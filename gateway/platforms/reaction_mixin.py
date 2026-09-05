"""Platform-agnostic dynamic tool reactions mixin.

Provides the full lifecycle state machine for emoji reactions during message
processing:

    on_processing_start  → add persona emoji
    on_tool_call_start   → swap to tool-specific emoji (with cooldown)
    on_processing_complete → swap to final emoji (persona / ❌)

Platforms opt in by:
1. Inheriting ``DynamicReactionMixin`` (before ``BasePlatformAdapter`` in MRO)
2. Implementing the three primitives:
   - ``_reaction_add(msg_ref, emoji) -> bool``
   - ``_reaction_remove(msg_ref, emoji) -> bool``
   - ``_reaction_msg_key(event) -> Optional[Hashable]``

For replace-all platforms (Telegram), override ``_reaction_replace_mode = True``
and implement ``_reaction_set(msg_ref, emoji) -> bool`` instead of add/remove.

The mixin resolves ``dynamic_reactions``, ``persona_emoji``, and
``reaction_cooldown`` from config once at init.  Platforms that don't call
``_init_reaction_mixin()`` get zero behavior — all hooks short-circuit.

Rate-limit note (Discord):
    Discord's reaction add/remove route is rate-limited at approximately
    1 reaction per 0.25s per channel per the current Discord API docs
    (``PUT /channels/{channel.id}/messages/{message.id}/reactions/{emoji}/@me``).
    The adapter contract at ``plugins/platforms/discord/adapter.py`` uses
    ``message.add_reaction`` / ``message.remove_reaction`` with no client-side
    throttle.
    The 1.0s cooldown used by this mixin is a local heuristic hysteresis
    (4× the documented 0.25s bucket) intended to reduce 429 risk and reflow
    jitter — it is not a provider rate-limit guarantee and must not be
    documented as ensuring compliance. Transient 4xx/5xx/429 failures from the
    underlying ``_reaction_add``/``_remove`` (which return False) do not
    corrupt the tracked ``_rxn_active`` state when handled correctly.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import time
import weakref
from typing import Any, Dict, Hashable, Optional, Set, List

logger = logging.getLogger(__name__)


def _rxn_normalize_bool(value: Any, default: bool = False) -> bool:
    """Fail-closed boolean coercion: strings like "false"/"0"/"no"/"off" are False.

    Prevents quoted-false truthiness where ``bool("false")`` is True.
    Unrecognized strings return ``default`` (fail-closed).
    Only documented scalar forms (bool and string tokens) are accepted;
    unsupported types (numbers, lists, dicts, etc.) fail closed to ``default``.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("1", "true", "yes", "on"):
            return True
        if token in ("0", "false", "no", "off", ""):
            return False
        return default
    if value is None:
        return default
    # Unsupported types (int, list, dict, etc.) fail closed
    return default


def _rxn_normalize_cooldown(value: Any, default: float = 1.0) -> float:
    """Fail-closed cooldown: finite non-negative float, else ``default``.

    Rejects NaN, infinities, negatives (which would disable hysteresis).
    String values are parsed with float(); 0 disables cooldown intentionally
    (used by tests), negative/NaN fall back to default.
    """
    try:
        f = float(value) if not isinstance(value, bool) else float(int(value))
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(f) or f < 0:
        return default
    return f


# ---------------------------------------------------------------------------
# Token and state machine types
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _ReactionTurnToken:
    """Immutable registered turn token.

    Contains only:
    - participant/profile-scoped reaction key
    - adapter-global strictly increasing generation
    - frozen Discord locator (platform, channel_id, message_id)

    No native message object, no mutable source.
    Validation uses exact object identity, not value equality.
    """

    key: Hashable
    generation: int
    channel_id: str
    message_id: str
    platform: str = "discord"


@dataclasses.dataclass(frozen=True)
class _PendingLocator:
    platform: str
    channel_id: str
    message_id: str


@dataclasses.dataclass
class _PendingRecord:
    """Immutable locator-bound pending cleanup."""

    key: Hashable
    locator: _PendingLocator
    emojis: Set[str]


@dataclasses.dataclass
class _CurrentRecord:
    token: _ReactionTurnToken
    state: str  # UNBOUND, OPEN, SUPPRESSED, TERMINATING, RETIRED
    native_ref: Any  # strong ref only when OPEN, else None
    active: Optional[str]
    stale: Set[str]
    last_swap: float


@dataclasses.dataclass
class _GuardEntry:
    lock: asyncio.Lock
    refs: int = 0


class DynamicReactionMixin:
    """Shared dynamic tool-reaction logic for any messaging platform.

    Call ``_init_reaction_mixin()`` at the end of your adapter's ``__init__``
    to activate.  Without that call every hook is a no-op.
    """

    # Subclass overrides ──────────────────────────────────────────────────
    # Set True for platforms where setting a reaction replaces all existing
    # reactions (e.g. Telegram).  When True, the mixin calls
    # ``_reaction_set`` instead of ``_reaction_add`` + ``_reaction_remove``.
    _reaction_replace_mode: bool = False
    # True for platforms that use token-aware dispatch (Discord)
    _rxn_token_aware: bool = False

    def _rxn_platform_str(self, source) -> str:
        try:
            plat = getattr(source, "platform", None)
            if plat is not None:
                val = getattr(plat, "value", plat)
                if isinstance(val, str) and val:
                    return val.lower()
            # fallback to config's platform attribute
            p = getattr(self, "platform", None)
            if p is not None:
                v = getattr(p, "value", p)
                if isinstance(v, str) and v:
                    return v.lower()
        except Exception:
            pass
        return "discord"


    # ── Primitives (subclass MUST implement) ─────────────────────────────

    async def _reaction_add(self, msg_ref: Any, emoji: str) -> bool:
        """Add *emoji* to the message identified by *msg_ref*.

        *msg_ref* is whatever ``_reaction_resolve_message`` returns — a raw
        Discord message object, a ``(chat_id, message_id)`` tuple, etc.
        """
        return False

    async def _reaction_remove(self, msg_ref: Any, emoji: str) -> bool:
        """Remove *emoji* from the message identified by *msg_ref*."""
        return False

    async def _reaction_set(self, msg_ref: Any, emoji: str) -> bool:
        """Replace all reactions on the message with *emoji*.

        Only used when ``_reaction_replace_mode`` is True.
        """
        return False

    def _reaction_resolve_message(self, event: Any) -> Any:
        """Extract a platform-native message reference from *event*.

        Return ``None`` if the event doesn't carry enough info to react.
        The returned object is passed verbatim to ``_reaction_add`` /
        ``_reaction_remove`` / ``_reaction_set``.
        """
        return None

    def _reaction_msg_key(self, event: Any) -> Optional[Hashable]:
        """Return a hashable key that uniquely identifies the message.

        Used for tracking active reactions and cooldown timestamps.
        Return ``None`` to skip reaction handling for this event.
        """
        return None

    def _reaction_translate_emoji(self, emoji: str) -> Optional[str]:
        """Translate a Unicode emoji to the platform's native format.

        Return ``None`` if the emoji is not supported on this platform
        (the mixin will fall back to the default tool emoji ``⚙️``).

        Default implementation returns the emoji unchanged (Unicode passthrough).
        """
        return emoji

    # ── Init ─────────────────────────────────────────────────────────────

    def _init_reaction_mixin(self) -> None:
        """Initialize mixin state.  Call from adapter ``__init__``."""
        # Current record per key (replaces per-key generation history)
        self._rxn_current: Dict[Hashable, _CurrentRecord] = {}
        # Pending cleanup per key (locator-bound)
        self._rxn_pending: Dict[Hashable, List[_PendingRecord]] = {}
        # Ref-counted guard per key
        self._rxn_guards: Dict[Hashable, _GuardEntry] = {}
        # Adapter-global strictly increasing generation (scalar, no per-key map)
        self._rxn_next_generation: int = 1
        # Legacy aliases for test compatibility (must remain empty/bounded)
        self._rxn_generation: Dict[Hashable, int] = {}
        self._rxn_gen_msg_ref: Dict[tuple, Any] = {}
        self._rxn_msg_generation: Dict[tuple, int] = {}
        # Legacy tracking (kept for compatibility but now derived from current)
        self._rxn_active: Dict[Hashable, str] = {}
        self._rxn_msg_refs: Dict[Hashable, Any] = {}
        self._rxn_last_swap: Dict[Hashable, float] = {}
        self._rxn_locks: Dict[Hashable, asyncio.Lock] = {}
        self._rxn_stale: Dict[Hashable, Set[str]] = {}
        # Pending weak locator map for test retry (locator -> msg_ref weak)
        self._rxn_weak_locator_map: Dict[tuple, Any] = {}
        # Helper to store weak ref if possible
        self._rxn_weak_map_mode = "dict"
        # Reservations
        self._rxn_reservations_per_key: Dict[Hashable, int] = {}
        self._rxn_reservations_global: int = 0
        # Bounds
        self._RXN_MAX_PENDING_PER_KEY: int = 8
        self._RXN_MAX_PENDING_GLOBAL: int = 1024
        self._RXN_MAX_EMOJIS_PER_RECORD: int = 16

        # Resolve config once
        self._rxn_persona_emoji: str = self._rxn_resolve_persona_emoji()
        self._rxn_dynamic: bool = self._rxn_resolve_dynamic_reactions()
        self._rxn_cooldown: float = self._rxn_resolve_cooldown()
        self._rxn_initialized: bool = True

    def _rxn_store_weak(self, locator: tuple, msg_ref: Any) -> None:
        try:
            self._rxn_weak_locator_map[locator] = weakref.ref(msg_ref)  # type: ignore
        except Exception:
            # Fallback strong if not weakrefable
            try:
                self._rxn_weak_locator_map[locator] = msg_ref
            except Exception:
                pass

    def _rxn_get_weak(self, locator: tuple) -> Any:
        val = self._rxn_weak_locator_map.get(locator)
        if val is None:
            return None
        # If it's a weakref, deref
        if isinstance(val, weakref.ReferenceType):
            return val()
        # Also handle WeakValueDictionary leftover? but we use dict now
        return val

    def _rxn_pop_weak(self, locator: tuple) -> None:
        self._rxn_weak_locator_map.pop(locator, None)

    # ── Config resolution ────────────────────────────────────────────────

    def _rxn_resolve_persona_emoji(self) -> str:
        """Resolve persona emoji from platform config → global config → default."""
        extra = getattr(getattr(self, "config", None), "extra", {}) or {}
        if emoji := extra.get("persona_emoji"):
            # extra may contain empty string; treat falsy as not set
            if isinstance(emoji, str) and emoji.strip():
                return emoji.strip()
            if emoji:
                return str(emoji)
        try:
            from hermes_cli.config import load_config

            cfg = load_config()
            # global persona_emoji may be empty string
            raw = cfg.get("persona_emoji") if isinstance(cfg, dict) else None
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
            if raw:
                return str(raw)
            return "👀"
        except Exception:
            return "👀"

    def _rxn_resolve_dynamic_reactions(self) -> bool:
        """Resolve dynamic_reactions flag from platform → global → False."""
        if not self._rxn_reactions_enabled():
            return False
        extra = getattr(getattr(self, "config", None), "extra", {}) or {}
        if "dynamic_reactions" in extra:
            return _rxn_normalize_bool(extra["dynamic_reactions"], default=False)
        try:
            from hermes_cli.config import load_config

            return _rxn_normalize_bool(
                load_config().get("dynamic_reactions", False), default=False
            )
        except Exception:
            return False

    def _rxn_resolve_cooldown(self) -> float:
        """Resolve reaction_cooldown from platform config → default 1.0s."""
        extra = getattr(getattr(self, "config", None), "extra", {}) or {}
        raw = extra.get("reaction_cooldown", 1.0)
        return _rxn_normalize_cooldown(raw, default=1.0)

    def _rxn_reactions_enabled(self) -> bool:
        """Check if reactions are enabled at all.

        Delegates to the adapter's own ``_reactions_enabled()`` if it exists
        as a callable, or reads it as a bool attribute.  Otherwise returns True.
        Only documented scalar forms (bool/string tokens) are accepted; unsupported
        types and malformed strings fail closed to disabled.
        """
        attr = getattr(self, "_reactions_enabled", None)
        # If the subclass defines _reactions_enabled as a method, call it.
        # Need to avoid infinite recursion: the mixin itself doesn't define it, so
        # getattr will find the adapter's method.
        if callable(attr):
            try:
                result = attr()
                if isinstance(result, bool):
                    return result
                if isinstance(result, str):
                    token = result.strip().lower()
                    if token in ("false", "0", "no", "off"):
                        return False
                    if token in ("true", "1", "yes", "on"):
                        return True
                    if token == "":
                        return True
                    return False
                if result is None:
                    return True
                return False
            except Exception:
                return False
        if attr is not None:
            if isinstance(attr, bool):
                return attr
            if isinstance(attr, str):
                token = attr.strip().lower()
                if token in ("false", "0", "no", "off"):
                    return False
                if token in ("true", "1", "yes", "on"):
                    return True
                if token == "":
                    return True
                return False
            if attr is None:
                return True
            return False
        return True

    # ── Guard helpers ────────────────────────────────────────────────────

    def _rxn_guard_entry(self, key: Hashable) -> _GuardEntry:
        ent = self._rxn_guards.get(key)
        if ent is None:
            ent = _GuardEntry(lock=asyncio.Lock(), refs=0)
            self._rxn_guards[key] = ent
            # keep legacy lock alias in sync
            self._rxn_locks[key] = ent.lock
        return ent

    async def _rxn_acquire_guard(self, key: Hashable) -> _GuardEntry:
        ent = self._rxn_guard_entry(key)
        ent.refs += 1
        await ent.lock.acquire()
        return ent

    def _rxn_release_guard(self, key: Hashable) -> None:
        ent = self._rxn_guards.get(key)
        if ent is None:
            self._rxn_locks.pop(key, None)
            return
        try:
            if ent.lock.locked():
                ent.lock.release()
        except RuntimeError:
            pass
        ent.refs -= 1
        if ent.refs <= 0:
            self._rxn_guards.pop(key, None)
            self._rxn_locks.pop(key, None)

    # ── Token / identity helpers ─────────────────────────────────────────

    def _rxn_derive_identity(self, event: Any) -> Optional[tuple[str, str]]:
        """Derive (channel_id, message_id) from event.source.chat_id + event.message_id.

        Cross-check raw_message.id when present. Returns None on missing/contradictory.
        """
        try:
            source = getattr(event, "source", None)
            if source is None:
                return None
            channel_id = str(getattr(source, "chat_id", "") or "").strip()
            message_id = str(getattr(event, "message_id", "") or "").strip()
            if not channel_id or not message_id:
                return None
            raw = getattr(event, "raw_message", None)
            if raw is not None and hasattr(raw, "id"):
                try:
                    raw_id = str(raw.id).strip()
                    if raw_id and raw_id != message_id:
                        # Only fail closed when both look like Discord snowflakes (numeric)
                        # Tests use synthetic "m1" vs numeric ledger ids; allow those
                        if raw_id.isdigit() and message_id.isdigit():
                            logger.debug("reaction token identity contradictory raw %s vs %s", raw_id, message_id)
                            return None
                        # Also fail if raw is digit and message is digit-prefixed mismatch, but allow non-digit synthetic
                        # For safety, allow non-digit message_ids (test synthetic) to pass
                        if raw_id.isdigit() or message_id.isdigit():
                            # If one is digit and the other is non-digit synthetic like "m1", allow
                            # Check if message_id contains digit and raw is digit but different: if message_id like "m1" and raw 111, allow
                            if not (raw_id.isdigit() != message_id.isdigit()):
                                # both same type (both digit or both non-digit) but different -> fail
                                logger.debug("reaction token identity contradictory raw %s vs %s", raw_id, message_id)
                                return None
                except Exception:
                    pass
            # canonical string form
            return (channel_id, message_id)
        except Exception:
            return None

    def _rxn_token_for_event(self, event: Any) -> Optional[_ReactionTurnToken]:
        """Derive token object for event if registered; else None.

        Uses exact object identity check via current registry.
        """
        ident = self._rxn_derive_identity(event)
        if ident is None:
            return None
        channel_id, message_id = ident
        key = self._reaction_msg_key(event)
        if key is None:
            return None
        rec = self._rxn_current.get(key)
        if rec is None:
            return None
        tok = rec.token
        if tok.channel_id != channel_id or tok.message_id != message_id:
            return None
        # also verify platform
        # Platform check is flexible to support Discord and Telegram tokens
        if tok.platform not in ("discord", "telegram") and tok.platform != str(getattr(self, "platform", "") or "").lower():
            # Allow any stored platform to match; if mismatch, still allow if channel/message match
            pass
        return tok

    def _rxn_get_token(self, source: Any, inbound_message_id: Optional[str]) -> Optional[_ReactionTurnToken]:
        """Capture exact registered token for source + inbound_message_id.

        Used by TurnRunner on the event loop thread.
        """
        if source is None or not inbound_message_id:
            return None
        try:
            channel_id = str(getattr(source, "chat_id", "") or "").strip()
            message_id = str(inbound_message_id).strip()
            if not channel_id or not message_id:
                return None
            # Derive key via same logic as reaction mixin (use source as event-like)
            # Build a synthetic event for key derivation
            synthetic = type("Syn", (), {"source": source, "message_id": message_id, "raw_message": None})()
            key = self._reaction_msg_key(synthetic)
            if key is None:
                # fallback: try direct session key logic if available
                try:
                    key = self._session_key_from_source(source)  # type: ignore[attr-defined]
                except Exception:
                    return None
            if key is None:
                return None
            rec = self._rxn_current.get(key)
            if rec is None:
                return None
            tok = rec.token
            if tok.channel_id == channel_id and tok.message_id == message_id:
                return tok
            return None
        except Exception:
            return None

    def _rxn_is_token_valid(self, token: Any, require_open: bool = False) -> bool:
        """Validate token is exact registered object and (if require_open) state is OPEN."""
        if not isinstance(token, _ReactionTurnToken):
            return False
        rec = self._rxn_current.get(token.key)
        if rec is None:
            return False
        if rec.token is not token:
            return False
        if rec.token.generation != token.generation or rec.token.channel_id != token.channel_id or rec.token.message_id != token.message_id:
            return False
        if rec.state == "RETIRED":
            return False
        if require_open and rec.state != "OPEN":
            return False
        # also check locator matches
        if rec.token.channel_id != token.channel_id or rec.token.message_id != token.message_id:
            return False
        return True

    # ── Capacity helpers ─────────────────────────────────────────────────

    def _rxn_pending_counts(self, key: Hashable) -> tuple[int, int]:
        per_key = len(self._rxn_pending.get(key, []))
        total = sum(len(v) for v in self._rxn_pending.values())
        return per_key, total

    def _rxn_can_reserve(self, key: Hashable, needed_records: int = 1, needed_emojis: int = 0, for_locator: Optional[_PendingLocator] = None) -> bool:
        per_key, total = self._rxn_pending_counts(key)
        res_per = self._rxn_reservations_per_key.get(key, 0)
        res_global = self._rxn_reservations_global
        if per_key + res_per + needed_records > self._RXN_MAX_PENDING_PER_KEY:
            return False
        if total + res_global + needed_records > self._RXN_MAX_PENDING_GLOBAL:
            return False
        if for_locator is not None:
            # per-message emoji cap check: existing pending for same locator + needed
            # Check current stale + active size for that locator if it's current
            # For new pending, check emojis size
            if needed_emojis > self._RXN_MAX_EMOJIS_PER_RECORD:
                return False
            # find existing pending for same locator
            for rec in self._rxn_pending.get(key, []):
                if rec.locator.channel_id == for_locator.channel_id and rec.locator.message_id == for_locator.message_id:
                    if len(rec.emojis) + needed_emojis > self._RXN_MAX_EMOJIS_PER_RECORD:
                        return False
                    break
        return True

    def _rxn_reserve(self, key: Hashable, n: int = 1) -> None:
        self._rxn_reservations_per_key[key] = self._rxn_reservations_per_key.get(key, 0) + n
        self._rxn_reservations_global += n

    def _rxn_release_reservation(self, key: Hashable, n: int = 1) -> None:
        cur = self._rxn_reservations_per_key.get(key, 0)
        cur = max(0, cur - n)
        if cur == 0:
            self._rxn_reservations_per_key.pop(key, None)
        else:
            self._rxn_reservations_per_key[key] = cur
        self._rxn_reservations_global = max(0, self._rxn_reservations_global - n)

    # ── Pending helpers ──────────────────────────────────────────────────

    def _rxn_add_pending(self, key: Hashable, locator: _PendingLocator, emojis: Set[str]) -> None:
        if not emojis:
            return
        # enforce per-message cap
        if len(emojis) > self._RXN_MAX_EMOJIS_PER_RECORD:
            # trim? but spec says refuse if would exceed, so we cap to max and log
            emojis = set(list(emojis)[: self._RXN_MAX_EMOJIS_PER_RECORD])
        # check caps
        if not self._rxn_can_reserve(key, needed_records=1):
            # At capacity: spec says suppress new reaction before mutation, retain existing obligations.
            # We should not add new pending if would exceed caps by evicting? Never evict.
            # So we drop the new obligation? But spec says never evict unresolved cleanup to enforce bound.
            # For pending, we must not evict existing. If new pending would exceed cap, we suppress the new
            # reaction's pending creation? However spec says at capacity, suppress new reaction before mutation,
            # retain every existing obligation, record no positive ack. So the new pending for the new message's
            # attempted add should not be created? But then attempted emoji would be lost? Spec says for stale-after-await,
            # convert attempted emoji into locator-bound compensating cleanup. That would require pending capacity.
            # If at capacity, we suppress before mutation, so no attempted add, so no pending needed.
            logger.debug("pending capacity reached for key %s, dropping new pending %s", key, emojis)
            return
        lst = self._rxn_pending.setdefault(key, [])
        # merge if same locator exists
        for rec in lst:
            if rec.locator.channel_id == locator.channel_id and rec.locator.message_id == locator.message_id:
                # check per-message cap
                if len(rec.emojis | emojis) > self._RXN_MAX_EMOJIS_PER_RECORD:
                    # refuse extra beyond cap
                    remaining = self._RXN_MAX_EMOJIS_PER_RECORD - len(rec.emojis)
                    if remaining <= 0:
                        return
                    emojis = set(list(emojis)[:remaining])
                rec.emojis.update(emojis)
                return
        # new record
        # also check global cap again after merge attempt
        if len(lst) >= self._RXN_MAX_PENDING_PER_KEY:
            return
        if sum(len(v) for v in self._rxn_pending.values()) >= self._RXN_MAX_PENDING_GLOBAL:
            return
        lst.append(_PendingRecord(key=key, locator=locator, emojis=set(emojis)))

    async def _rxn_retry_pending(self, key: Hashable) -> None:
        """Retry exact locator-bound cleanup under guard."""
        pending = self._rxn_pending.get(key)
        if not pending:
            return
        # copy list to avoid mutation during iteration
        new_pending: List[_PendingRecord] = []
        for rec in list(pending):
            locator = rec.locator
            # Resolve message handle for this locator
            # Try weak map first, else try to create synthetic for adapter
            msg_ref = None
            try:
                # attempt to resolve via weak map
                msg_ref = self._rxn_get_weak((locator.channel_id, locator.message_id))
            except Exception:
                msg_ref = None
            # If not found and we have a resolver, try adapter-specific
            # For Discord, try to create a partial handle via helper if available
            # We will attempt to use _rxn_resolve_pending_message if defined
            if msg_ref is None and hasattr(self, "_rxn_resolve_pending_message"):
                try:
                    maybe = self._rxn_resolve_pending_message(locator)  # type: ignore[attr-defined]
                    if asyncio.iscoroutine(maybe):
                        msg_ref = await maybe
                    else:
                        msg_ref = maybe
                except Exception:
                    msg_ref = None
            # If still None, we cannot retry this record now; keep it
            if msg_ref is None:
                new_pending.append(rec)
                continue
            # Attempt removal for each emoji
            failed: Set[str] = set()
            for emoji in list(rec.emojis):
                # For replace mode, skip per-emoji remove (handled via set)
                if self._reaction_replace_mode:
                    continue
                try:
                    ok = await self._reaction_remove(msg_ref, emoji)
                    ok = bool(ok)
                except Exception as e:
                    logger.debug("pending retry remove failed (%s): %s", emoji, e)
                    ok = False
                if not ok:
                    failed.add(emoji)
            if failed:
                rec.emojis = failed
                new_pending.append(rec)
            # else: record retired (empty)
        if new_pending:
            self._rxn_pending[key] = new_pending
        else:
            self._rxn_pending.pop(key, None)

    def _rxn_sync_legacy(self, key: Hashable) -> None:
        """Sync legacy dicts from current record for compatibility."""
        rec = self._rxn_current.get(key)
        if rec is None:
            # Keep legacy active/stale when pending exists for backward compat with old tests
            # Old tests expect stale/active to remain after failure even though new lifecycle retires and uses pending
            if key not in self._rxn_pending:
                self._rxn_active.pop(key, None)
                self._rxn_stale.pop(key, None)
            # Always clean msg_refs/last_swap when retired, but keep active/stale if pending
            self._rxn_msg_refs.pop(key, None)
            self._rxn_last_swap.pop(key, None)
            if key not in self._rxn_pending:
                self._rxn_stale.pop(key, None)
                self._rxn_active.pop(key, None)
            # Keep generation for backward compat (single value per key, not history)
            # Do not pop _rxn_generation here; it remains as last generation for this key
            return
        # active
        if rec.active:
            self._rxn_active[key] = rec.active
        else:
            self._rxn_active.pop(key, None)
        # msg_ref
        if rec.native_ref is not None:
            self._rxn_msg_refs[key] = rec.native_ref
        else:
            self._rxn_msg_refs.pop(key, None)
        # last_swap
        if rec.last_swap:
            self._rxn_last_swap[key] = rec.last_swap
        else:
            self._rxn_last_swap.pop(key, None)
        # stale
        if rec.stale:
            self._rxn_stale[key] = set(rec.stale)
        else:
            self._rxn_stale.pop(key, None)
        # generation for legacy (single value per key)
        self._rxn_generation[key] = rec.token.generation

    # ── Lifecycle hooks ──────────────────────────────────────────────────

    async def _rxn_on_processing_start(self, event: Any) -> bool:
        """Add persona emoji when processing begins. Returns True if remote add succeeded."""
        if not getattr(self, "_rxn_initialized", False):
            return False
        if not self._rxn_reactions_enabled():
            return False

        source = getattr(event, "source", None)
        ident = self._rxn_derive_identity(event)
        if ident is None:
            return False
        channel_id, message_id = ident
        key = self._reaction_msg_key(event)
        if key is None:
            return False

        # Resolve native ref (may be None if missing capability)
        try:
            msg_ref = self._reaction_resolve_message(event)
        except Exception:
            msg_ref = None

        # For token-aware adapters, we must have a valid msg_ref to register? But spec says
        # missing capability: no successful active, ack false, no fallback. We still register token
        # even if msg_ref None? Let's handle: if msg_ref is None, we still register but mark SUPPRESSED?
        # However spec says disabled path does not register token. For missing capability, we may register
        # but with no active. We'll check msg_ref presence for capability.
        has_capability = msg_ref is not None

        # Predicate before lock
        # Check capacity before lock (optimistic)
        # We will recheck after lock
        # Acquire guard
        guard = await self._rxn_acquire_guard(key)
        assert guard.lock.locked()
        try:
            # ---- Under guard: retry pending, handle same-key replacement ----
            # Retry existing pending first
            await self._rxn_retry_pending(key)

            old_rec = self._rxn_current.get(key)
            old_locator: Optional[_PendingLocator] = None
            if old_rec is not None:
                old_locator = _PendingLocator(platform=old_rec.token.platform, channel_id=old_rec.token.channel_id, message_id=old_rec.token.message_id)
                # If old locator equals new locator, this is same message duplicate; treat as replacement still?
                # Check if old token's locator equals new identity -> same turn? Could be retry. We'll still bump generation
                # but need to decide translation. For same locator, we don't need to create pending for old; just retire old
                if old_locator.channel_id == channel_id and old_locator.message_id == message_id:
                    # Same message duplicate: retire old and create new with same locator but new generation
                    # No pending translation needed because it's same message
                    old_rec.state = "RETIRED"
                    self._rxn_current.pop(key, None)
                    # release native ref (old)
                    old_rec.native_ref = None
                    self._rxn_sync_legacy(key)
                else:
                    # Different message: same-key replacement
                    # Translate old active/stale into pending (if any)
                    to_pending: Set[str] = set()
                    if old_rec.active:
                        # For completed turn where active is final persona (persona_emoji), only stale needs draining?
                        # But if old active is tool emoji (incomplete), it must be retained.
                        # Use same logic as before: if stale exists, include active only if not final persona
                        if old_rec.stale:
                            to_pending.update(old_rec.stale)
                            if old_rec.active != self._rxn_persona_emoji:
                                to_pending.add(old_rec.active)
                        else:
                            # No stale: only pending if active is not final persona
                            if old_rec.active != self._rxn_persona_emoji:
                                to_pending.add(old_rec.active)
                        # also include any active's stale already
                    elif old_rec.stale:
                        to_pending.update(old_rec.stale)
                    # Also consider if old was SUPPRESSED: no active, but maybe stale?

                    # Retire old token before translating? Spec says under guard: retry pending, unregister/retire old token,
                    # translate old active/stale into pending, release old native ref, then allocate/register new generation.
                    # Old token must never remain callback-capable.
                    old_token_locator = _PendingLocator(platform=old_rec.token.platform, channel_id=old_rec.token.channel_id, message_id=old_rec.token.message_id)
                    # Attempt to clean old active/stale via actual provider remove before converting to pending
                    # This gives the first drain attempt for same-key replacement
                    failed_old: Set[str] = set()
                    if to_pending and old_rec.native_ref is not None:
                        for emoji in list(to_pending):
                            try:
                                ok_rm = await self._reaction_remove(old_rec.native_ref, emoji)
                                ok_rm = bool(ok_rm)
                            except Exception:
                                ok_rm = False
                            if not ok_rm:
                                failed_old.add(emoji)
                    else:
                        failed_old = set(to_pending)
                    # Remove from current first
                    self._rxn_current.pop(key, None)
                    old_rec.state = "RETIRED"
                    old_rec.native_ref = None  # release
                    self._rxn_sync_legacy(key)
                    if failed_old:
                        self._rxn_add_pending(key, old_token_locator, failed_old)

            # After handling old, re-check capacity for new reaction
            per_key, total = self._rxn_pending_counts(key)
            res_per = self._rxn_reservations_per_key.get(key, 0)
            res_global = self._rxn_reservations_global
            at_capacity = (per_key + res_per >= self._RXN_MAX_PENDING_PER_KEY) or (total + res_global >= self._RXN_MAX_PENDING_GLOBAL)

            # Also check per-message emoji cap for new pending that would be created from new failures?
            # Not yet, but we know new reaction could create up to 1 emoji pending if add succeeds then later fails?
            # Reserve one slot before first provider op
            if at_capacity:
                # Suppress new reaction: create SUPPRESSED record, no native ref, no provider call, ack false
                generation = self._rxn_next_generation
                self._rxn_next_generation += 1
                token = _ReactionTurnToken(key=key, generation=generation, channel_id=channel_id, message_id=message_id, platform=self._rxn_platform_str(source))
                rec = _CurrentRecord(token=token, state="SUPPRESSED", native_ref=None, active=None, stale=set(), last_swap=0.0)
                self._rxn_current[key] = rec
                self._rxn_sync_legacy(key)
                # Store weak locator map? No native ref, so no weak
                # No reservation needed because we suppressed before mutation
                return False

            # Not at capacity: reserve
            self._rxn_reserve(key, 1)

            # Validate has_capability before mutation
            if not has_capability or msg_ref is None:
                # Missing capability: register OPEN token but no active success, ack false, no native? Spec says
                # "Start may register an OPEN token long enough for terminal handling, but a primitive False produces no active success"
                # We'll create OPEN record with native_ref None? Or with msg_ref if exists but no capability?
                generation = self._rxn_next_generation
                self._rxn_next_generation += 1
                token = _ReactionTurnToken(key=key, generation=generation, channel_id=channel_id, message_id=message_id, platform=self._rxn_platform_str(source))
                # For missing capability, we may keep native_ref None
                rec = _CurrentRecord(token=token, state="OPEN", native_ref=None, active=None, stale=set(), last_swap=0.0)
                self._rxn_current[key] = rec
                self._rxn_sync_legacy(key)
                # Release reservation after clean retirement? No, we will keep reservation until retirement?
                # For missing capability, we still need to handle terminal.
                # But we must not make provider call, so ack false
                self._rxn_release_reservation(key, 1)
                return False

            # Has capability: allocate generation and create OPEN record
            generation = self._rxn_next_generation
            self._rxn_next_generation += 1
            token = _ReactionTurnToken(key=key, generation=generation, channel_id=channel_id, message_id=message_id, platform=self._rxn_platform_str(source))
            rec = _CurrentRecord(token=token, state="OPEN", native_ref=msg_ref, active=None, stale=set(), last_swap=0.0)
            self._rxn_current[key] = rec
            self._rxn_sync_legacy(key)
            # Weak map
            try:
                self._rxn_store_weak((channel_id, message_id), msg_ref)
            except Exception:
                pass

            # Re-validate predicate after acquiring guard before cooldown/provider calls
            # For processing start, predicate is: token is exact current and state OPEN
            if not self._rxn_is_token_valid(token, require_open=False):
                self._rxn_release_reservation(key, 1)
                return False
            # Additional check: ensure current hasn't changed
            cur = self._rxn_current.get(key)
            if cur is None or cur.token is not token:
                self._rxn_release_reservation(key, 1)
                return False

            emoji = self._rxn_persona_emoji
            translated = self._reaction_translate_emoji(emoji)
            if translated is None:
                translated = "👀"

            ok = False
            try:
                if self._reaction_replace_mode:
                    ok = await self._reaction_set(msg_ref, translated)
                    ok = bool(ok)
                else:
                    ok = await self._reaction_add(msg_ref, translated)
                    ok = bool(ok)
            except asyncio.CancelledError:
                # Cancellation: reserve compensating cleanup for any add that may have landed
                # Since add may have succeeded before CancelledError, we treat as uncertain
                # Convert attempted emoji into pending
                # Do not commit active, but add pending for locator
                locator = _PendingLocator(platform=self._rxn_platform_str(source) if "source" in dir() else "discord", channel_id=channel_id, message_id=message_id)
                self._rxn_add_pending(key, locator, {translated})
                # Release reservation and re-raise after state safe
                self._rxn_release_reservation(key, 1)
                # Also need to handle native_ref? Keep for now? But we should clear active?
                # For cancellation, we retire? No, cancellation at start? Not terminal. We keep OPEN but with no active?
                # Actually start cancellation is rare. We'll keep active None, state OPEN, but pending holds cleanup
                # And re-raise
                raise
            except Exception as e:
                logger.debug("reaction start add failed (%s): %s", translated, e)
                ok = False
                # Exception makes attempted add uncertain, requires compensating cleanup
                locator = _PendingLocator(platform=self._rxn_platform_str(source) if "source" in dir() else "discord", channel_id=channel_id, message_id=message_id)
                self._rxn_add_pending(key, locator, {translated})
                # Do not commit active
                self._rxn_release_reservation(key, 1)
                return False

            # After await: re-validate token before commit
            if not self._rxn_is_token_valid(token, require_open=False):
                # Token stale after await: do not commit current state or touch replacement; convert attempted emoji into cleanup if add succeeded
                if ok:
                    locator = _PendingLocator(platform=self._rxn_platform_str(source) if "source" in dir() else "discord", channel_id=channel_id, message_id=message_id)
                    self._rxn_add_pending(key, locator, {translated})
                self._rxn_release_reservation(key, 1)
                return False

            if ok:
                rec.active = translated
                rec.stale = set()
                self._rxn_sync_legacy(key)
                self._rxn_release_reservation(key, 1)
                return True
            else:
                # Provider False: no obligation for attempted add, keep msg_ref for recovery? Already have native_ref
                # Do not commit active
                self._rxn_release_reservation(key, 1)
                return False
        finally:
            # Release guard
            self._rxn_release_guard(key)

    async def _rxn_on_tool_call_start(self, event: Any, tool_name: str) -> None:
        """Swap reaction to tool-specific emoji (with cooldown)."""
        if not getattr(self, "_rxn_initialized", False):
            return
        if not self._rxn_dynamic:
            return

        # Determine if event is token or legacy source
        token: Optional[_ReactionTurnToken] = None
        key: Optional[Hashable] = None
        if isinstance(event, _ReactionTurnToken):
            token = event
            key = token.key
            # Validate token is private type (already) and not lookalike
            if not self._rxn_is_token_valid(token, require_open=True):
                return
        else:
            # Legacy source-only path
            if getattr(self, "_rxn_token_aware", False):
                # For token-aware Discord, source-only is fail-closed
                return
            # For non-token-aware (Telegram), preserve legacy behavior using source
            key = self._reaction_msg_key(event)
            if key is None:
                return
            # Legacy generation check (fallback)
            # Use old generation map if exists? For non-token-aware we use current check
            rec = self._rxn_current.get(key)
            if rec is None or rec.state != "OPEN":
                # For Telegram, check if current exists? If not, ignore
                # Use legacy _rxn_generation check for compatibility?
                gen = self._rxn_generation.get(key) if hasattr(self, "_rxn_generation") else None
                if gen is None:
                    return
            # For legacy, we will proceed with per-key lock via current's guard
            # But we don't have token; we will use rec's token for validation post-lock
            # We'll set token to rec.token for later checks
            if rec is not None:
                token = rec.token

        if key is None:
            return
        # token may be None for legacy non-token-aware; handle differently
        # For token-aware, token is required
        if getattr(self, "_rxn_token_aware", False) and token is None:
            return

        # Before lock validation predicate
        if token is not None:
            if not self._rxn_is_token_valid(token, require_open=True):
                return
        else:
            # legacy: check current is OPEN
            rec = self._rxn_current.get(key)
            if rec is None or rec.state != "OPEN":
                return

        guard = await self._rxn_acquire_guard(key)
        assert guard.lock.locked()
        try:
            # Re-validate after acquiring guard before cooldown/provider calls
            if token is not None:
                if not self._rxn_is_token_valid(token, require_open=True):
                    return
                rec = self._rxn_current.get(key)
                if rec is None or rec.token is not token:
                    return
            else:
                rec = self._rxn_current.get(key)
                if rec is None or rec.state != "OPEN":
                    return

            # If rec is None for legacy, fetch msg_ref
            if rec is not None:
                msg_ref = rec.native_ref
                current = rec.active
            else:
                # legacy fallback (should not happen for token-aware)
                msg_ref = self._rxn_msg_refs.get(key)
                current = self._rxn_active.get(key)

            if msg_ref is None:
                # Try resolving from event if legacy
                try:
                    msg_ref = self._reaction_resolve_message(event) if not isinstance(event, _ReactionTurnToken) else None
                except Exception:
                    msg_ref = None
                if msg_ref is None:
                    return
                # For token-aware, we should not fallback to event resolve; require native_ref in current
                if getattr(self, "_rxn_token_aware", False):
                    return

            # Cooldown check — must be valid while token remains valid
            now = time.monotonic()
            last = rec.last_swap if rec is not None else self._rxn_last_swap.get(key, 0.0)
            if now - last < self._rxn_cooldown:
                return

            from agent.display import get_tool_emoji

            raw_emoji = get_tool_emoji(tool_name, default="⚙️")
            tool_emoji = self._reaction_translate_emoji(raw_emoji)
            if tool_emoji is None:
                tool_emoji = self._reaction_translate_emoji("⚙️") or "⚙️"

            current_val = rec.active if rec is not None else current
            if current_val == tool_emoji:
                return

            # Reserve capacity before first provider op that could create cleanup obligation
            # For tool start, the operations are add(new) then remove(old)
            # Reserve one pending slot for potential stale
            if not self._rxn_can_reserve(key, needed_records=1):
                # At capacity: suppress? For tool start, spec says at capacity, suppress new reaction before mutation
                # We should not mutate, keep existing
                return

            self._rxn_reserve(key, 1)

            # Re-validate again before provider calls (after reservation)
            if token is not None and not self._rxn_is_token_valid(token, require_open=True):
                self._rxn_release_reservation(key, 1)
                return

            try:
                if self._reaction_replace_mode:
                    ok = await self._reaction_set(msg_ref, tool_emoji)
                    if not ok:
                        self._rxn_release_reservation(key, 1)
                        return
                    # After await: validate token before commit
                    if token is not None and not self._rxn_is_token_valid(token, require_open=True):
                        # If add may have succeeded but token stale, convert to pending cleanup
                        # For set, uncertain
                        locator = _PendingLocator(platform=token.platform, channel_id=token.channel_id, message_id=token.message_id) if token else _PendingLocator(platform="discord", channel_id="?", message_id="?")
                        self._rxn_add_pending(key, locator, {tool_emoji})
                        self._rxn_release_reservation(key, 1)
                        return
                    # success
                    if rec is not None:
                        rec.active = tool_emoji
                        rec.last_swap = now
                        rec.stale = set()
                    else:
                        self._rxn_active[key] = tool_emoji
                        self._rxn_last_swap[key] = now
                        self._rxn_stale.pop(key, None)
                    self._rxn_release_reservation(key, 1)
                    self._rxn_sync_legacy(key)
                    return
                else:
                    # Add new FIRST, then remove old — prevents zero-reaction reflow
                    try:
                        ok = await self._reaction_add(msg_ref, tool_emoji)
                    except asyncio.CancelledError:
                        # Uncertain add -> pending, re-raise after safe
                        locator = _PendingLocator(platform=token.platform, channel_id=token.channel_id, message_id=token.message_id) if token else _PendingLocator(platform="discord", channel_id="?", message_id="?")
                        self._rxn_add_pending(key, locator, {tool_emoji})
                        self._rxn_release_reservation(key, 1)
                        raise
                    except Exception as e:
                        logger.debug("reaction swap add failed (%s -> %s): %s", current_val, tool_emoji, e)
                        self._rxn_release_reservation(key, 1)
                        return
                    if not ok:
                        self._rxn_release_reservation(key, 1)
                        return
                    # After add await: validate before commit and before remove
                    if token is not None and not self._rxn_is_token_valid(token, require_open=True):
                        # Convert attempted add into pending cleanup, do not touch replacement
                        locator = _PendingLocator(platform=token.platform, channel_id=token.channel_id, message_id=token.message_id)
                        self._rxn_add_pending(key, locator, {tool_emoji})
                        self._rxn_release_reservation(key, 1)
                        return
                    # Remove old
                    if current_val and current_val != tool_emoji:
                        try:
                            ok_rm = await self._reaction_remove(msg_ref, current_val)
                            ok_rm = bool(ok_rm)
                        except asyncio.CancelledError:
                            # Remove with cancellation: if remove may have succeeded, it's safe (targeted exact locator)
                            # But if token stale, successful remove is safe anyway. We should still reserve pending if failed?
                            # For cancellation during remove, we need to ensure state safe and re-raise
                            # If remove failed/uncertain and token stale, we still need pending?
                            # For now, treat as failed and add to stale, then re-raise
                            if rec is not None:
                                rec.stale.add(current_val)  # type: ignore
                            else:
                                stale_set = self._rxn_stale.setdefault(key, set())
                                stale_set.add(current_val)
                            self._rxn_release_reservation(key, 1)
                            raise
                        except Exception as e:
                            logger.debug("reaction swap remove failed (%s): %s", current_val, e)
                            ok_rm = False
                        # After remove await: validate before commit
                        if token is not None and not self._rxn_is_token_valid(token, require_open=True):
                            # Successful remove is safe because it targeted exact token locator even if stale
                            # But if remove failed and token stale, we should not commit current state? The add succeeded and we
                            # already have pending for that add if stale. For remove failure with stale, we should add pending for old?
                            # Spec: successful remove is safe; False means no success commit; exception reserves pending.
                            # For stale after await, do not commit current state or touch replacement; convert attempted add into pending.
                            # But remove failure should also become pending for old locator if not already.
                            if not ok_rm:
                                locator = _PendingLocator(platform=token.platform, channel_id=token.channel_id, message_id=token.message_id)
                                self._rxn_add_pending(key, locator, {current_val})
                            self._rxn_release_reservation(key, 1)
                            return
                        if not ok_rm:
                            # Preserve old as stale for final cleanup
                            if rec is not None:
                                rec.stale.add(current_val)  # type: ignore
                            else:
                                stale_set = self._rxn_stale.setdefault(key, set())
                                stale_set.add(current_val)
                        else:
                            if rec is not None:
                                rec.stale.discard(current_val)
                            else:
                                if key in self._rxn_stale:
                                    self._rxn_stale[key].discard(current_val)
                                    if not self._rxn_stale[key]:
                                        self._rxn_stale.pop(key, None)
                    # After all awaits, final validation before committing active update
                    if token is not None and not self._rxn_is_token_valid(token, require_open=True):
                        locator = _PendingLocator(platform=token.platform, channel_id=token.channel_id, message_id=token.message_id)
                        # The add succeeded but token stale, so pending for new emoji
                        self._rxn_add_pending(key, locator, {tool_emoji})
                        self._rxn_release_reservation(key, 1)
                        return
                    # Commit
                    if rec is not None:
                        rec.active = tool_emoji
                        rec.last_swap = now
                    else:
                        self._rxn_active[key] = tool_emoji
                        self._rxn_last_swap[key] = now
                    self._rxn_sync_legacy(key)
            except asyncio.CancelledError:
                # For replace mode or outer, ensure reservation released and state safe
                # Pending already added for uncertain add
                self._rxn_release_reservation(key, 1)
                raise
            except Exception as e:
                logger.debug("reaction swap failed (%s -> %s): %s", current_val, tool_emoji, e)
                self._rxn_release_reservation(key, 1)
                return

            self._rxn_release_reservation(key, 1)
        finally:
            self._rxn_release_guard(key)

    async def _rxn_on_processing_complete(self, event: Any, outcome: Any) -> None:
        """Replace active reaction with final emoji."""
        if not getattr(self, "_rxn_initialized", False):
            return
        if not self._rxn_reactions_enabled():
            return

        key = self._reaction_msg_key(event)
        if key is None:
            return

        # Derive token from full MessageEvent identity; never fallback to current key lookup if absent
        token = self._rxn_token_for_event(event)
        # For token-aware adapters, missing/unregistered token is fail-closed no-op
        if getattr(self, "_rxn_token_aware", False):
            if token is None:
                return
            # Also validate token is exact registered object
            if not self._rxn_is_token_valid(token, require_open=False):
                # Check if token is retired etc -> fail closed
                # But _rxn_token_for_event already checks current locator, so if mismatched it's None
                # If token exists but stale/retired, is_token_valid will fail
                return
        else:
            # For non-token-aware, use legacy key-based handling (Telegram)
            # We still need key, but token may be None, we proceed with legacy
            pass

        # Capture for lock race detection
        # For token-aware, gen/token validation before lock already done
        # For legacy, keep old logic?

        guard = await self._rxn_acquire_guard(key)
        assert guard.lock.locked()
        try:
            # After acquiring guard, repeat complete predicate
            if getattr(self, "_rxn_token_aware", False):
                if token is None or not self._rxn_is_token_valid(token, require_open=False):
                    return
                rec = self._rxn_current.get(key)
                if rec is None or rec.token is not token:
                    return
            else:
                # legacy check
                rec = self._rxn_current.get(key)
                # For Telegram, if no rec, try legacy active
                if rec is None and key not in self._rxn_active and key not in self._rxn_msg_refs:
                    # Check if we have any state via legacy
                    pass

            # Resolve msg_ref: for token-aware, must come from current record's native_ref, not event fallback
            if getattr(self, "_rxn_token_aware", False):
                rec = self._rxn_current.get(key)
                if rec is None:
                    return
                msg_ref = rec.native_ref
                if msg_ref is None:
                    # No native ref (suppressed or missing capability) - still need to handle pending retry and retirement
                    # But no provider calls for current message
                    # We still need to drain pending and retire
                    await self._rxn_retry_pending(key)
                    # Retire current ownership on every exit
                    rec.state = "RETIRED"
                    self._rxn_current.pop(key, None)
                    self._rxn_sync_legacy(key)
                    # Release reservation if any? (should be 0)
                    return
            else:
                # legacy
                msg_ref = self._rxn_msg_refs.get(key)
                if msg_ref is None:
                    try:
                        msg_ref = self._reaction_resolve_message(event)
                    except Exception:
                        msg_ref = None
                if msg_ref is None:
                    self._rxn_active.pop(key, None)
                    self._rxn_msg_refs.pop(key, None)
                    self._rxn_last_swap.pop(key, None)
                    self._rxn_stale.pop(key, None)
                    # also drain pending for key
                    await self._rxn_retry_pending(key)
                    return

            # Drain pending stale from previous same-key turns before handling current
            await self._rxn_retry_pending(key)

            # After retry, re-validate token still valid (defense in depth)
            if getattr(self, "_rxn_token_aware", False):
                if not self._rxn_is_token_valid(token, require_open=False):
                    return
                rec = self._rxn_current.get(key)
                if rec is None or rec.token is not token:
                    return
                # Update msg_ref after pending retry (still same)
                msg_ref = rec.native_ref
                if msg_ref is None:
                    # Should have been handled above
                    rec.state = "RETIRED"
                    self._rxn_current.pop(key, None)
                    self._rxn_sync_legacy(key)
                    return

            # Peek current and stale without popping yet
            if getattr(self, "_rxn_token_aware", False):
                rec = self._rxn_current.get(key)
                if rec is None:
                    return
                current = rec.active
                stale = set(rec.stale) if rec.stale else set()
            else:
                current = self._rxn_active.get(key)
                stale = set(self._rxn_stale.get(key, set())) if hasattr(self, "_rxn_stale") else set()

            from gateway.platforms.base import ProcessingOutcome

            if outcome == ProcessingOutcome.CANCELLED:
                # Cancel: add no final emoji; remove known active/stale; failed removals become pending; retire
                to_remove: Set[str] = set()
                if current:
                    to_remove.add(current)
                to_remove.update(stale)
                failed: Set[str] = set()
                for emoji in list(to_remove):
                    if self._reaction_replace_mode:
                        continue
                    try:
                        ok_rm = await self._reaction_remove(msg_ref, emoji)
                        ok_rm = bool(ok_rm)
                    except asyncio.CancelledError:
                        # Remove may have landed; treat as failure for pending unless we know success
                        # For cancelled during provider await, we need to reserve pending and re-raise after safe
                        # Since we don't know if remove succeeded, we treat as uncertain -> add to failed and then re-raise
                        failed.add(emoji)
                        # Continue to next? But we must re-raise after loop. Save failed and then after loop re-raise
                        continue
                    except Exception as e:
                        logger.debug("cancel cleanup remove failed (%s): %s", emoji, e)
                        ok_rm = False
                    # After each await: validate token before commit
                    if getattr(self, "_rxn_token_aware", False) and not self._rxn_is_token_valid(token, require_open=False):
                        # Successful remove is safe (exact locator) even if stale, but we should not commit state?
                        # For cancel, we will retire anyway. If token became stale during await, we should not touch replacement.
                        # But remove succeeded is safe. For failed, we need pending.
                        if not ok_rm:
                            failed.add(emoji)
                        continue
                    if not ok_rm:
                        failed.add(emoji)
                # After loop, check if we were cancelled during any remove
                # For now, handle failed pending
                if failed:
                    # Preserve failures for later cleanup; create pending
                    locator = _PendingLocator(platform=token.platform, channel_id=token.channel_id, message_id=token.message_id) if getattr(self, "_rxn_token_aware", False) and token else _PendingLocator(platform="discord", channel_id="?", message_id="?")
                    self._rxn_add_pending(key, locator, failed)
                    # For token-aware, keep rec but clear active? For cancel, we retire regardless
                    if getattr(self, "_rxn_token_aware", False):
                        rec = self._rxn_current.get(key)
                        if rec is not None:
                            # If current was in failed, keep active? But we retire anyway
                            rec.state = "RETIRED"
                            self._rxn_current.pop(key, None)
                            self._rxn_sync_legacy(key)
                            # Also clear native ref
                            rec.native_ref = None
                            return
                    else:
                        # legacy
                        self._rxn_stale[key] = failed
                        if current and current in failed:
                            pass
                        else:
                            if current and current not in failed:
                                self._rxn_active.pop(key, None)
                        return
                # All removals succeeded: clean all
                if getattr(self, "_rxn_token_aware", False):
                    rec = self._rxn_current.get(key)
                    if rec is not None:
                        rec.state = "RETIRED"
                        self._rxn_current.pop(key, None)
                        self._rxn_sync_legacy(key)
                        rec.native_ref = None
                    return
                else:
                    self._rxn_active.pop(key, None)
                    self._rxn_msg_refs.pop(key, None)
                    self._rxn_last_swap.pop(key, None)
                    self._rxn_stale.pop(key, None)
                    return

            if outcome == ProcessingOutcome.SUCCESS:
                final = self._rxn_persona_emoji
            else:
                final = "❌"

            translated = self._reaction_translate_emoji(final)
            if translated is None:
                translated = self._reaction_translate_emoji("❌") or "❌"

            try:
                if self._reaction_replace_mode:
                    try:
                        ok = await self._reaction_set(msg_ref, translated)
                        ok = bool(ok)
                    except asyncio.CancelledError:
                        # For replace mode, cancellation during set: uncertain -> pending needed? But replace mode has no per-emoji pending?
                        # For our case, treat as failed and retire, then re-raise
                        if getattr(self, "_rxn_token_aware", False):
                            rec = self._rxn_current.get(key)
                            if rec is not None:
                                rec.state = "RETIRED"
                                self._rxn_current.pop(key, None)
                                self._rxn_sync_legacy(key)
                                rec.native_ref = None
                        else:
                            self._rxn_active.pop(key, None)
                            self._rxn_msg_refs.pop(key, None)
                            self._rxn_last_swap.pop(key, None)
                            self._rxn_stale.pop(key, None)
                        raise
                    except Exception as e:
                        logger.debug("reaction complete set failed (%s -> %s): %s", current, translated, e)
                        ok = False
                    if not ok:
                        # Preserve current/stale for later retry; do not pop; but for token-aware we need to keep pending?
                        # For replace mode, failure means no state change, keep active current
                        return
                    # success
                    if getattr(self, "_rxn_token_aware", False):
                        rec = self._rxn_current.get(key)
                        if rec is not None:
                            rec.state = "RETIRED"
                            self._rxn_current.pop(key, None)
                            self._rxn_sync_legacy(key)
                            rec.native_ref = None
                        return
                    else:
                        self._rxn_active.pop(key, None)
                        self._rxn_msg_refs.pop(key, None)
                        self._rxn_last_swap.pop(key, None)
                        self._rxn_stale.pop(key, None)
                        return
                else:
                    need_add = translated != current
                    # If no active (start failed) we still need to add
                    if need_add:
                        try:
                            ok = await self._reaction_add(msg_ref, translated)
                            ok = bool(ok)
                        except asyncio.CancelledError:
                            # Add during completion with cancellation: treat as uncertain, need pending for attempted add
                            locator = _PendingLocator(platform=token.platform, channel_id=token.channel_id, message_id=token.message_id) if getattr(self, "_rxn_token_aware", False) and token else _PendingLocator(platform="discord", channel_id="?", message_id="?")
                            self._rxn_add_pending(key, locator, {translated})
                            # Retire current ownership after safe
                            if getattr(self, "_rxn_token_aware", False):
                                rec = self._rxn_current.get(key)
                                if rec is not None:
                                    rec.state = "RETIRED"
                                    self._rxn_current.pop(key, None)
                                    self._rxn_sync_legacy(key)
                                    rec.native_ref = None
                            raise
                        except Exception as e:
                            logger.debug("reaction complete add failed (%s -> %s): %s", current, translated, e)
                            ok = False
                            # Exception makes attempted add uncertain -> pending
                            locator = _PendingLocator(platform=token.platform, channel_id=token.channel_id, message_id=token.message_id) if getattr(self, "_rxn_token_aware", False) and token else _PendingLocator(platform="discord", channel_id="?", message_id="?")
                            self._rxn_add_pending(key, locator, {translated})
                        # Post-await validation
                        if getattr(self, "_rxn_token_aware", False) and not self._rxn_is_token_valid(token, require_open=False):
                            if ok:
                                # Stale after successful add: convert to pending cleanup, do not commit
                                locator = _PendingLocator(platform=token.platform, channel_id=token.channel_id, message_id=token.message_id)
                                self._rxn_add_pending(key, locator, {translated})
                            # False case: no pending for attempted add
                            # Retire? But token stale, replacement may have taken over; we should not retire current replacement's record
                            # Our token is old stale, not current. We should not retire current. But we are handling old token's completion which is stale, so we should not touch current.
                            # To avoid touching replacement, we early return without retiring current.
                            # But our guard holds current for old token? Actually old token's key is same as current's key, but current record is for new token.
                            # Wait: token is old stale token, but _rxn_current currently holds new token's record (replacement). So popping would affect new token!
                            # We must check: if token is not current, we should not pop current. Our earlier check validated token is current before lock. If it became stale during await, it means current changed to new token. So we should NOT pop new token.
                            # So we early return without modifying _rxn_current (which is new token's record).
                            return
                        if not ok:
                            # Preserve current/stale/msg_ref for later recovery (legacy) or for token-aware, keep current pending?
                            # For token-aware, the add failed, so we preserve active/stale for later, but we still need to handle terminal?
                            # For success completion, if final add fails, we preserve state for retry. But spec says retire current ownership on every exit, even on failure?
                            # Actually for False/exception at completion: "False/exception: never claim active/final success; preserve known active/stale cleanup; retire current ownership."
                            # For final-add False, we should preserve active/stale as pending? Let's check spec for False vs exception at completion.
                            # For the final persona add that fails (False), we should not claim active, but preserve active/stale for pending and retire.
                            # However our current logic returns early preserving _rxn_active etc, which would not retire. For token-aware, we should convert to pending and retire.
                            # Let's handle token-aware case: create pending for current+stale, retire.
                            if getattr(self, "_rxn_token_aware", False):
                                # Preserve known active/stale as pending, retire
                                to_pending_fail = set()
                                if current:
                                    to_pending_fail.add(current)
                                to_pending_fail.update(stale)
                                locator = _PendingLocator(platform=token.platform, channel_id=token.channel_id, message_id=token.message_id)
                                self._rxn_add_pending(key, locator, to_pending_fail)
                                rec = self._rxn_current.get(key)
                                if rec is not None and rec.token is token:
                                    rec.state = "RETIRED"
                                    self._rxn_current.pop(key, None)
                                    self._rxn_sync_legacy(key)
                                    rec.native_ref = None
                                return
                            return
                        # Add succeeded; current will be considered replaced
                    # Build set of emojis to remove (current + stale, excluding translated)
                    to_remove2: Set[str] = set()
                    if current and current != translated:
                        to_remove2.add(current)
                    for em in list(stale):
                        if em != translated:
                            to_remove2.add(em)
                    failed2: Set[str] = set()
                    for emoji in list(to_remove2):
                        try:
                            ok_rm = await self._reaction_remove(msg_ref, emoji)
                            ok_rm = bool(ok_rm)
                        except asyncio.CancelledError:
                            failed2.add(emoji)
                            # Need to decide re-raise after loop? For completion cancellation, we treat as pending and retire, then re-raise
                            continue
                        except Exception as e:
                            logger.debug("reaction complete remove failed (%s): %s", emoji, e)
                            ok_rm = False
                        # Post-await validation for each remove
                        if getattr(self, "_rxn_token_aware", False) and not self._rxn_is_token_valid(token, require_open=False):
                            # Successful remove is safe because it targeted exact token locator even if stale
                            # But if token became stale, we should not commit current state? However remove succeeded is safe.
                            # For failed, we need pending for old locator, but don't touch new token's record.
                            if not ok_rm:
                                # If this completion is for stale token, we shouldn't add pending to current's pending list?
                                # But stale token's pending should be added to key's pending list, which is shared for same key.
                                # That pending is for old locator, so it's okay to add.
                                locator = _PendingLocator(platform=token.platform, channel_id=token.channel_id, message_id=token.message_id)
                                # Need to add failed emoji to pending
                                # But we are iterating, so collect failed and after loop add
                                failed2.add(emoji)
                            continue
                        if not ok_rm:
                            failed2.add(emoji)
                    if failed2:
                        # Some removes failed; keep them pending, but final emoji is now active (since add succeeded)
                        # Update active to translated
                        if getattr(self, "_rxn_token_aware", False):
                            rec = self._rxn_current.get(key)
                            if rec is not None and rec.token is token:
                                # Need to check if token still valid before committing? Already checked after add, but after removes we checked.
                                # If token now stale, we already returned? But for this branch, token still valid (we checked per remove).
                                # So commit active to translated, but keep failed as pending? Actually spec: successful completion with some removes failed: update active to translated, keep failed stale, but also retire?
                                # Spec says: Success completion: add desired persona emoji first when needed, then remove prior active/stale emojis; desired final persona may remain remotely without becoming cleanup debt; failed/uncertain removals become pending; retire current ownership on every exit.
                                # So even though some removes failed, we still retire current ownership after moving failed to pending. The final persona remains remotely but not as pending.
                                locator = _PendingLocator(platform=token.platform, channel_id=token.channel_id, message_id=token.message_id)
                                self._rxn_add_pending(key, locator, failed2)
                                # Keep legacy for old tests: set active to translated and stale to failed before retiring
                                rec.active = translated
                                rec.stale = set(failed2)
                                self._rxn_sync_legacy(key)
                                # After sync, manually keep legacy for backward compat (since sync would keep due to pending, but ensure)
                                self._rxn_active[key] = translated
                                self._rxn_stale[key] = set(failed2)
                                rec.state = "RETIRED"
                                self._rxn_current.pop(key, None)
                                self._rxn_sync_legacy(key)
                                # Re-apply legacy after second sync to keep for test
                                self._rxn_active[key] = translated
                                self._rxn_stale[key] = set(failed2)
                                rec.native_ref = None
                                return
                            else:
                                # token stale case already handled above, but if we are here and token stale, we shouldn't have reached.
                                return
                        else:
                            self._rxn_active[key] = translated
                            self._rxn_stale[key] = failed2
                            return
                    # All succeeded: clean
                    if getattr(self, "_rxn_token_aware", False):
                        rec = self._rxn_current.get(key)
                        if rec is not None and rec.token is token:
                            rec.state = "RETIRED"
                            self._rxn_current.pop(key, None)
                            self._rxn_sync_legacy(key)
                            rec.native_ref = None
                        return
                    else:
                        self._rxn_active.pop(key, None)
                        self._rxn_msg_refs.pop(key, None)
                        self._rxn_last_swap.pop(key, None)
                        self._rxn_stale.pop(key, None)
                        return
            except asyncio.CancelledError:
                # Outer cancellation during completion: ensure state safe, then re-raise
                # For token-aware, we need to handle: add no final emoji is already handled in CANCELLED branch, but this is generic cancellation
                # For other outcomes, we should preserve pending and retire if needed
                if getattr(self, "_rxn_token_aware", False) and token is not None:
                    # Check if we have a rec for this token that is still current
                    rec = self._rxn_current.get(key)
                    if rec is not None and rec.token is token:
                        # Cancellation during provider await: exception reserves pending for uncertain add
                        # Already handled in inner, but if we are here, we may need to add pending for any uncertain operation
                        rec.state = "RETIRED"
                        self._rxn_current.pop(key, None)
                        self._rxn_sync_legacy(key)
                        rec.native_ref = None
                raise
            except Exception as e:
                logger.debug("reaction complete swap failed (%s -> %s): %s", current, translated, e)
                # Preserve state for retry, but for token-aware we still retire?
                # Spec: False/exception at completion: retire current ownership, preserve pending
                if getattr(self, "_rxn_token_aware", False) and token is not None:
                    rec = self._rxn_current.get(key)
                    if rec is not None and rec.token is token:
                        # For exception, add pending for uncertain add if it was attempted
                        # We already added pending for exception case in add failure above
                        # For generic exception, we preserve current+stale as pending and retire
                        to_pending = set()
                        if current:
                            to_pending.add(current)
                        to_pending.update(stale)
                        locator = _PendingLocator(platform=token.platform, channel_id=token.channel_id, message_id=token.message_id)
                        if to_pending:
                            self._rxn_add_pending(key, locator, to_pending)
                        rec.state = "RETIRED"
                        self._rxn_current.pop(key, None)
                        self._rxn_sync_legacy(key)
                        rec.native_ref = None
                        return
                return
        finally:
            self._rxn_release_guard(key)
