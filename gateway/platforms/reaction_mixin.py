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
    throttle. To avoid 429s and reflow jitter, this mixin defaults to a
    conservative 1.0s cooldown (4× the documented 0.25s) plus hysteresis:
    repeated identical emoji and rapid intermediate tool events within the
    cooldown are coalesced, and transient 4xx/5xx/429 failures from the
    underlying ``_reaction_add``/``_remove`` (which return False) do not
    corrupt the tracked ``_rxn_active`` state.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Hashable, Optional

logger = logging.getLogger(__name__)


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
        # Per-message tracking: msg_key → currently displayed emoji
        self._rxn_active: Dict[Hashable, str] = {}
        # Per-message tracking: msg_key → resolved message reference
        self._rxn_msg_refs: Dict[Hashable, Any] = {}
        # Cooldown: msg_key → monotonic timestamp of last swap
        self._rxn_last_swap: Dict[Hashable, float] = {}
        # Per-message lock to serialize reaction swaps (prevents stacking)
        self._rxn_locks: Dict[Hashable, asyncio.Lock] = {}
        # Retained unconfirmed terminal authority: keys where terminal
        # completion/cancellation removal was unconfirmed (False/exception).
        # While retained, a new same-key start must not overwrite authority
        # before the old remote mutation is reconciled (fail-closed).
        self._rxn_retained: set = set()

        # Resolve config once
        self._rxn_persona_emoji: str = self._rxn_resolve_persona_emoji()
        self._rxn_dynamic: bool = self._rxn_resolve_dynamic_reactions()
        self._rxn_cooldown: float = self._rxn_resolve_cooldown()
        self._rxn_initialized: bool = True

    # ── Config resolution ────────────────────────────────────────────────

    def _rxn_resolve_persona_emoji(self) -> str:
        """Resolve persona emoji from platform config → global config → default."""
        extra = getattr(getattr(self, "config", None), "extra", {}) or {}
        if emoji := extra.get("persona_emoji"):
            return emoji
        try:
            from hermes_cli.config import load_config
            return load_config().get("persona_emoji") or "👀"
        except Exception:
            return "👀"

    def _rxn_resolve_dynamic_reactions(self) -> bool:
        """Resolve dynamic_reactions flag from platform → global → False."""
        if not self._rxn_reactions_enabled():
            return False

        def _coerce_dynamic(value: Any, default: bool = False) -> bool:
            """Established bool-token coercion (mirrors gateway.config._coerce_bool)."""
            try:
                from gateway.config import _coerce_bool

                return _coerce_bool(value, default)
            except Exception:
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    tok = value.strip().lower()
                    if tok in ("1", "true", "yes", "on"):
                        return True
                    if tok in ("0", "false", "no", "off"):
                        return False
                    return default
                if value is None:
                    return default
                try:
                    return bool(value)
                except Exception:
                    return default

        extra = getattr(getattr(self, "config", None), "extra", {}) or {}
        if "dynamic_reactions" in extra:
            return _coerce_dynamic(extra["dynamic_reactions"], False)
        try:
            from hermes_cli.config import load_config

            return _coerce_dynamic(load_config().get("dynamic_reactions", False), False)
        except Exception:
            return False

    def _rxn_resolve_cooldown(self) -> float:
        """Resolve reaction_cooldown from platform config → default 1.0s."""
        extra = getattr(getattr(self, "config", None), "extra", {}) or {}
        return float(extra.get("reaction_cooldown", 1.0))

    def _rxn_reactions_enabled(self) -> bool:
        """Check if reactions are enabled at all.

        Delegates to the adapter's own ``_reactions_enabled()`` if it exists
        as a callable, or reads it as a bool attribute.  Otherwise returns True.
        """
        attr = getattr(self, "_reactions_enabled", None)
        if callable(attr):
            try:
                return bool(attr())
            except Exception:
                return False
        if attr is not None:
            return bool(attr)
        return True

    # ── Lifecycle hooks ──────────────────────────────────────────────────

    def _rxn_lock(self, key: Hashable) -> asyncio.Lock:
        """Get or create a per-message lock to serialize reaction swaps."""
        if key not in self._rxn_locks:
            self._rxn_locks[key] = asyncio.Lock()
        return self._rxn_locks[key]

    async def _rxn_on_processing_start(self, event: Any) -> bool:
        """Add persona emoji when processing begins.

        Returns True only after the provider confirms success (``_reaction_add``
        or ``_reaction_set`` returns truthy); False on disabled behavior,
        missing capability, provider ``False``, or exception.  Commits
        ``_rxn_active`` and ``_rxn_msg_refs`` only on confirmed success.

        On every new public start attempt for a participant/profile-scoped key,
        any failed/disabled/exception/missing-capability outcome clears prior
        active/message authority for that key before returning, so a stale
        confirmed start for an earlier message cannot authorize later tool or
        completion mutations.
        """
        if not getattr(self, "_rxn_initialized", False):
            return False
        # Derive key early so failed/disabled/missing paths can invalidate stale authority
        key: Optional[Hashable] = None
        try:
            key = self._reaction_msg_key(event)
        except Exception:
            key = None
        # Fail-closed: if a prior terminal removal is unconfirmed for this key,
        # reject the new same-key start before any remote mutation, preserving
        # the old authority/reference for later public reconciliation.
        if key is not None and key in getattr(self, "_rxn_retained", set()):
            logger.debug("reaction start deferred for retained key %s", key)
            return False
        if not self._rxn_reactions_enabled():
            if key is not None:
                self._rxn_active.pop(key, None)
                self._rxn_msg_refs.pop(key, None)
                self._rxn_last_swap.pop(key, None)
                self._rxn_locks.pop(key, None)
                getattr(self, "_rxn_retained", set()).discard(key)
            return False

        msg_ref = self._reaction_resolve_message(event)
        if msg_ref is None:
            if key is not None:
                self._rxn_active.pop(key, None)
                self._rxn_msg_refs.pop(key, None)
                self._rxn_last_swap.pop(key, None)
                self._rxn_locks.pop(key, None)
                getattr(self, "_rxn_retained", set()).discard(key)
            return False
        if key is None:
            return False

        emoji = self._rxn_persona_emoji
        translated = self._reaction_translate_emoji(emoji)
        if translated is None:
            translated = "👀"

        try:
            if self._reaction_replace_mode:
                ok = await self._reaction_set(msg_ref, translated)
            else:
                ok = await self._reaction_add(msg_ref, translated)
            if not ok:
                self._rxn_active.pop(key, None)
                self._rxn_msg_refs.pop(key, None)
                self._rxn_last_swap.pop(key, None)
                self._rxn_locks.pop(key, None)
                getattr(self, "_rxn_retained", set()).discard(key)
                return False
        except Exception as e:
            logger.debug("reaction start add failed (%s): %s", translated, e)
            self._rxn_active.pop(key, None)
            self._rxn_msg_refs.pop(key, None)
            self._rxn_last_swap.pop(key, None)
            self._rxn_locks.pop(key, None)
            getattr(self, "_rxn_retained", set()).discard(key)
            return False

        self._rxn_active[key] = translated
        self._rxn_msg_refs[key] = msg_ref
        return True

    async def _rxn_on_tool_call_start(self, event: Any, tool_name: str) -> None:
        """Swap reaction to tool-specific emoji (with cooldown)."""
        if not getattr(self, "_rxn_initialized", False):
            return
        if not self._rxn_dynamic:
            return

        key = self._reaction_msg_key(event)
        if key is None:
            return
        # Confirmed-start predicate: no active session must not create authority
        # via a raw MessageEvent or raw-cache fallback. A failed/disabled/
        # missing-capability start leaves active/msg_refs empty.
        if key not in self._rxn_active:
            return

        async with self._rxn_lock(key):
            if key not in self._rxn_active:
                return
            msg_ref = self._rxn_msg_refs.get(key)
            if msg_ref is None:
                # Fallback only when a confirmed start exists (active present);
                # source-only callbacks (TurnRunner) still resolve via adapter cache.
                msg_ref = self._reaction_resolve_message(event)
                if msg_ref is None:
                    return
                self._rxn_msg_refs[key] = msg_ref

            # Cooldown check — conservative 1.0s buffer over Discord's 0.25s limit
            now = time.monotonic()
            last = self._rxn_last_swap.get(key, 0.0)
            if now - last < self._rxn_cooldown:
                return

            from agent.display import get_tool_emoji
            raw_emoji = get_tool_emoji(tool_name, default="⚙️")
            tool_emoji = self._reaction_translate_emoji(raw_emoji)
            if tool_emoji is None:
                tool_emoji = self._reaction_translate_emoji("⚙️") or "⚙️"

            current = self._rxn_active.get(key)
            if current == tool_emoji:
                return  # Already showing this emoji

            try:
                if self._reaction_replace_mode:
                    ok = await self._reaction_set(msg_ref, tool_emoji)
                    if not ok:
                        return
                else:
                    # Add new FIRST, then remove old — prevents zero-reaction
                    # reflow jitter on Discord (message shifts when reactions disappear)
                    ok = await self._reaction_add(msg_ref, tool_emoji)
                    if not ok:
                        return
                    if current and current != tool_emoji:
                        remove_ok = await self._reaction_remove(msg_ref, current)
                        if not remove_ok:
                            logger.debug(
                                "reaction swap remove failed (%s -> %s): unconfirmed removal, keeping prior state",
                                current,
                                tool_emoji,
                            )
                            return
            except Exception as e:
                logger.debug("reaction swap failed (%s -> %s): %s", current, tool_emoji, e)
                # Do not corrupt active tracking on transient failure; keep previous
                return

            self._rxn_active[key] = tool_emoji
            self._rxn_last_swap[key] = now

    async def _rxn_on_processing_complete(self, event: Any, outcome: Any) -> None:
        """Replace active reaction with final emoji."""
        if not getattr(self, "_rxn_initialized", False):
            return
        key = self._reaction_msg_key(event)
        if key is None:
            return
        # If reactions are disabled, clear any stale authority for this key
        # but do not attempt remote mutations; this invalidates prior active
        # so later tool/completion cannot reuse it via stale msg_refs.
        if not self._rxn_reactions_enabled():
            # Use lock-protected cleanup to avoid races with concurrent swaps
            async with self._rxn_lock(key):
                self._rxn_active.pop(key, None)
                self._rxn_msg_refs.pop(key, None)
                self._rxn_last_swap.pop(key, None)
                getattr(self, "_rxn_retained", set()).discard(key)
            self._rxn_locks.pop(key, None)
            return

        # Confirmed-start predicate: no active session must not authorize a
        # final reaction. Raw MessageEvent or raw-cache alone is not authority.
        if key not in self._rxn_active:
            return

        async with self._rxn_lock(key):
            if key not in self._rxn_active:
                return
            # Peek without popping: retain authority until remote confirms removal
            msg_ref = self._rxn_msg_refs.get(key)
            if msg_ref is None:
                msg_ref = self._reaction_resolve_message(event)
                if msg_ref is not None:
                    # cache for potential retry while authority retained
                    self._rxn_msg_refs[key] = msg_ref
            current = self._rxn_active.get(key)
            if msg_ref is None:
                self._rxn_active.pop(key, None)
                self._rxn_msg_refs.pop(key, None)
                self._rxn_last_swap.pop(key, None)
                self._rxn_locks.pop(key, None)
                getattr(self, "_rxn_retained", set()).discard(key)
                return

            # Import here to avoid circular imports at module level
            from gateway.platforms.base import ProcessingOutcome

            if outcome == ProcessingOutcome.CANCELLED:
                # Just clean up, don't change the reaction
                if current:
                    if not self._reaction_replace_mode:
                        try:
                            ok = await self._reaction_remove(msg_ref, current)
                        except Exception as e:
                            logger.debug("cancel cleanup remove failed (%s): %s", current, e)
                            getattr(self, "_rxn_retained", set()).add(key)
                            return
                        if not ok:
                            logger.debug("cancel cleanup remove failed (%s): unconfirmed removal", current)
                            getattr(self, "_rxn_retained", set()).add(key)
                            return
                self._rxn_active.pop(key, None)
                self._rxn_msg_refs.pop(key, None)
                self._rxn_last_swap.pop(key, None)
                self._rxn_locks.pop(key, None)
                getattr(self, "_rxn_retained", set()).discard(key)
                return

            if outcome == ProcessingOutcome.SUCCESS:
                final = self._rxn_persona_emoji
            else:
                final = "❌"

            translated = self._reaction_translate_emoji(final)
            if translated is None:
                translated = self._reaction_translate_emoji("❌") or "❌"

            if translated == current:
                self._rxn_active.pop(key, None)
                self._rxn_msg_refs.pop(key, None)
                self._rxn_last_swap.pop(key, None)
                self._rxn_locks.pop(key, None)
                getattr(self, "_rxn_retained", set()).discard(key)
                return

            try:
                if self._reaction_replace_mode:
                    ok = await self._reaction_set(msg_ref, translated)
                    if not ok:
                        logger.debug("reaction complete set failed (%s -> %s): unconfirmed", current, translated)
                        getattr(self, "_rxn_retained", set()).add(key)
                        return
                else:
                    ok = await self._reaction_add(msg_ref, translated)
                    if not ok:
                        logger.debug("reaction complete add failed (%s -> %s): unconfirmed", current, translated)
                        getattr(self, "_rxn_retained", set()).add(key)
                        return
                    if current and current != translated:
                        remove_ok = await self._reaction_remove(msg_ref, current)
                        if not remove_ok:
                            logger.debug(
                                "reaction complete remove failed (%s -> %s): unconfirmed removal, keeping prior state",
                                current,
                                translated,
                            )
                            getattr(self, "_rxn_retained", set()).add(key)
                            return
            except Exception as e:
                logger.debug("reaction complete swap failed (%s -> %s): %s", current, translated, e)
                getattr(self, "_rxn_retained", set()).add(key)
                return

            self._rxn_active.pop(key, None)
            self._rxn_msg_refs.pop(key, None)
            self._rxn_last_swap.pop(key, None)
            getattr(self, "_rxn_retained", set()).discard(key)

        # Clean up lock outside the lock itself
        self._rxn_locks.pop(key, None)
