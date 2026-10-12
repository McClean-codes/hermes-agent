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

State-safety contract (every dict below is keyed per message by
``_reaction_msg_key``, e.g. chat base + message id):
    * Concurrent sessions processed in one chat get distinct keys, so one
      turn's completion can never overwrite or clear another turn's
      raw-message/active/lock state.
    * A FAILED initial add is never marked active — but the message
      association is kept, so a successful completion can retry the final
      reaction against the right message.
    * A false/failed REMOVE never advances state as if the emoji were gone:
      the old emoji is tracked as a residual (``_rxn_residuals``) and
      reconciled at completion; residuals that outlive completion are
      recorded in the bounded ``_rxn_residual_leaks`` map.
    * Every completion path — success, failure, cancel, or a failed reaction
      API call — releases the per-message lock and all per-turn entries.

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
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Optional

logger = logging.getLogger(__name__)


@dataclass
class TurnReactionEvent:
    """MessageEvent-shaped carrier for tool-progress reaction hooks.

    The runner schedules ``on_tool_call_start`` with the turn's ``SessionSource``
    (it carries no inbound payload) plus a ``turn_identity``; adapters re-key onto
    this carrier so reaction, raw-message, cooldown and lock state all resolve the
    same per-message key regardless of which event shape the hook arrived in.
    """

    source: Any
    message_id: Optional[str] = None
    raw_message: Any = None


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

    # Bounded record of reactions that outlived their per-turn state (a
    # removal that kept failing through completion). Oldest entries evicted.
    _RXN_LEAK_LIMIT: int = 64

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
        # Emojis a false/failed remove left on the message: tracked, never
        # advanced away as if removed, reconciled against the final emoji at
        # completion.
        self._rxn_residuals: Dict[Hashable, set] = {}
        # Completion-terminal residuals that could not be removed either — a
        # bounded record of what outlived the per-turn state.
        self._rxn_residual_leaks: Dict[Hashable, set] = {}

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
        extra = getattr(getattr(self, "config", None), "extra", {}) or {}
        if "dynamic_reactions" in extra:
            return bool(extra["dynamic_reactions"])
        try:
            from hermes_cli.config import load_config
            return bool(load_config().get("dynamic_reactions", False))
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

    async def _rxn_show(self, msg_ref: Any, emoji: str) -> bool:
        """Put *emoji* on the message (``_reaction_set`` in replace mode, else add).

        Transport errors are swallowed; True only when the platform confirmed.
        """
        try:
            if self._reaction_replace_mode:
                return bool(await self._reaction_set(msg_ref, emoji))
            return bool(await self._reaction_add(msg_ref, emoji))
        except Exception as e:
            logger.debug("reaction add failed (%s): %s", emoji, e)
            return False

    async def _rxn_unshow(self, msg_ref: Any, emoji: str) -> bool:
        """Remove *emoji*; transport errors swallowed, True only when confirmed removed."""
        try:
            return bool(await self._reaction_remove(msg_ref, emoji))
        except Exception as e:
            logger.debug("reaction remove failed (%s): %s", emoji, e)
            return False

    def _rxn_track_residual_leaks(self, key: Hashable, emojis) -> None:
        """Bounded, inspectable record of reactions that outlived their per-turn state."""
        self._rxn_residual_leaks[key] = set(emojis)
        while len(self._rxn_residual_leaks) > self._RXN_LEAK_LIMIT:
            self._rxn_residual_leaks.pop(next(iter(self._rxn_residual_leaks)))

    async def _rxn_clear(self, key: Hashable, msg_ref: Any, emojis, *, context: str) -> None:
        """Remove every emoji in *emojis*; unconfirmed removals become residual leaks."""
        leftovers = set()
        for emoji in sorted(emojis):
            if not await self._rxn_unshow(msg_ref, emoji):
                leftovers.add(emoji)
        if leftovers:
            logger.warning(
                "reaction %s could not remove residual emoji(s) %s; recording them as residual leaks",
                context, sorted(leftovers),
            )
            self._rxn_track_residual_leaks(key, leftovers)

    async def _rxn_on_processing_start(self, event: Any) -> bool:
        """Add persona emoji when processing begins; True only when confirmed present."""
        if not getattr(self, "_rxn_initialized", False):
            return False
        if not self._rxn_reactions_enabled():
            return False

        msg_ref = self._reaction_resolve_message(event)
        if msg_ref is None:
            return False
        key = self._reaction_msg_key(event)
        if key is None:
            return False

        emoji = self._rxn_persona_emoji
        translated = self._reaction_translate_emoji(emoji)
        if translated is None:
            translated = "👀"

        added = await self._rxn_show(msg_ref, translated)
        if added:
            self._rxn_active[key] = translated
        # Keep the message association even when the add FAILED: a failed ack is
        # never marked active (marking it active made a successful completion skip
        # its retry), but completion still needs this target to retry against.
        self._rxn_msg_refs[key] = msg_ref
        return added

    async def _rxn_on_tool_call_start(self, event: Any, tool_name: str) -> None:
        """Swap reaction to tool-specific emoji (with cooldown)."""
        if not getattr(self, "_rxn_initialized", False):
            return
        if not self._rxn_dynamic:
            return

        key = self._reaction_msg_key(event)
        if key is None:
            return

        async with self._rxn_lock(key):
            msg_ref = self._rxn_msg_refs.get(key)
            if msg_ref is None:
                # Try resolving from event directly (fallback)
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

            if not await self._rxn_show(msg_ref, tool_emoji):
                # Landing the new emoji failed: keep prior state untouched — nothing
                # new is on the message, so nothing may advance.
                return
            if not self._reaction_replace_mode and current and current != tool_emoji:
                if await self._rxn_unshow(msg_ref, current):
                    # Confirmed gone; clear any stale residual for it.
                    residuals = self._rxn_residuals.get(key)
                    if residuals:
                        residuals.discard(current)
                else:
                    # False/failed remove: the OLD emoji is still on the message.
                    # Track it as a residual instead of advancing as if it were
                    # removed; completion reconciles residuals against the final emoji.
                    self._rxn_residuals.setdefault(key, set()).add(current)

            self._rxn_active[key] = tool_emoji
            self._rxn_last_swap[key] = now

    async def _rxn_on_processing_complete(self, event: Any, outcome: Any) -> None:
        """Replace active reaction with final emoji; ALWAYS release per-turn state."""
        if not getattr(self, "_rxn_initialized", False):
            return
        if not self._rxn_reactions_enabled():
            return

        key = self._reaction_msg_key(event)
        if key is None:
            return
        try:
            await self._rxn_complete_locked(event, key, outcome)
        finally:
            # Completion/failure paths release the lock and every per-turn entry even
            # when a reaction API call failed and the body returned early — a leaked
            # lock entry serialized a key no later event would ever touch again.
            self._rxn_locks.pop(key, None)
            self._rxn_active.pop(key, None)
            self._rxn_msg_refs.pop(key, None)
            self._rxn_last_swap.pop(key, None)
            self._rxn_residuals.pop(key, None)

    async def _rxn_complete_locked(self, event: Any, key: Hashable, outcome: Any) -> None:
        """One completion attempt under the per-message lock (cleanup is the caller's)."""
        async with self._rxn_lock(key):
            msg_ref = self._rxn_msg_refs.get(key)
            if msg_ref is None:
                msg_ref = self._reaction_resolve_message(event)
            current = self._rxn_active.get(key)
            residuals = set(self._rxn_residuals.get(key) or ())

            if msg_ref is None:
                if residuals:
                    logger.warning(
                        "reaction completion has no message target; residual reaction(s) "
                        "%s could not be reconciled", sorted(residuals),
                    )
                    self._rxn_track_residual_leaks(key, residuals)
                return

            # Import here to avoid circular imports at module level
            from gateway.platforms.base import ProcessingOutcome

            if outcome == ProcessingOutcome.CANCELLED:
                # Just clean up what we believe is present; don't add a terminal emoji.
                if not self._reaction_replace_mode:
                    targets = {e for e in ((current,) if current else ()) + tuple(residuals) if e}
                    if targets:
                        await self._rxn_clear(key, msg_ref, targets, context="cancel cleanup")
                return

            if outcome == ProcessingOutcome.SUCCESS:
                final = self._rxn_persona_emoji
            else:
                final = "❌"

            translated = self._reaction_translate_emoji(final)
            if translated is None:
                translated = self._reaction_translate_emoji("❌") or "❌"

            if self._reaction_replace_mode:
                if not await self._rxn_show(msg_ref, translated):
                    logger.debug("reaction complete replace failed (%s)", translated)
                return

            # Land the final emoji FIRST, then remove everything else: the tracked
            # emoji plus any residual left by an earlier failed remove — except an
            # emoji that EQUALS the final state, which needs no removal (it already
            # is the correct end state).
            if translated != current and not await self._rxn_show(msg_ref, translated):
                # Final add failed: keep whatever is on the message rather than
                # stripping visible state down to nothing; locks and per-turn state
                # are released by the caller regardless.
                logger.debug(
                    "reaction complete add failed (%s); leaving message reactions untouched",
                    translated,
                )
                if residuals:
                    self._rxn_track_residual_leaks(key, residuals)
                return
            targets = {
                e for e in ((current,) if current else ()) + tuple(residuals)
                if e and e != translated
            }
            if targets:
                await self._rxn_clear(key, msg_ref, targets, context="completion cleanup")
