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
import logging
import math
import time
from typing import Any, Dict, Hashable, Optional

logger = logging.getLogger(__name__)


def _rxn_normalize_bool(value: Any, default: bool = False) -> bool:
    """Fail-closed boolean coercion: strings like "false"/"0"/"no"/"off" are False.

    Prevents quoted-false truthiness where ``bool("false")`` is True.
    Unrecognized strings return ``default`` (fail-closed).
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
    # numbers: 0 => False, non-zero => True, but bool("false") already handled
    try:
        return bool(value)
    except Exception:
        return default


def _rxn_normalize_cooldown(value: Any, default: float = 1.0) -> float:
    """Fail-closed cooldown: finite non-negative float, else ``default``.

    Rejects NaN, infinities, negatives (which would disable hysteresis).
    String values are parsed with float(); 0 disables cooldown intentionally
    (used by tests), negative/NaN fall back to default.
    """
    try:
        f = float(value) if not isinstance(value, bool) else float(int(value))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f) or f < 0:
        return default
    return f


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
        # Stale emojis that failed to be removed; remain cleanup-reachable
        self._rxn_stale: Dict[Hashable, set[str]] = {}

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

            return _rxn_normalize_bool(load_config().get("dynamic_reactions", False), default=False)
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
        """
        attr = getattr(self, "_reactions_enabled", None)
        # If the subclass defines _reactions_enabled as a method, call it.
        # Need to avoid infinite recursion: the mixin itself doesn't define it, so
        # getattr will find the adapter's method.
        if callable(attr):
            try:
                result = attr()
                # Normalize boolean strings fail-closed
                if isinstance(result, str):
                    return _rxn_normalize_bool(result, default=True)
                return bool(result)
            except Exception:
                return False
        if attr is not None:
            if isinstance(attr, str):
                return _rxn_normalize_bool(attr, default=True)
            return bool(attr)
        return True

    # ── Lifecycle hooks ──────────────────────────────────────────────────

    def _rxn_lock(self, key: Hashable) -> asyncio.Lock:
        """Get or create a per-message lock to serialize reaction swaps."""
        if key not in self._rxn_locks:
            self._rxn_locks[key] = asyncio.Lock()
        return self._rxn_locks[key]

    async def _rxn_on_processing_start(self, event: Any) -> None:
        """Add persona emoji when processing begins."""
        if not getattr(self, "_rxn_initialized", False):
            return
        if not self._rxn_reactions_enabled():
            return

        msg_ref = self._reaction_resolve_message(event)
        if msg_ref is None:
            return
        key = self._reaction_msg_key(event)
        if key is None:
            return

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
        except Exception as e:
            logger.debug("reaction start add failed (%s): %s", translated, e)
            ok = False

        if ok:
            self._rxn_active[key] = translated
            self._rxn_msg_refs[key] = msg_ref
            # Successful start clears any prior stale for this key
            self._rxn_stale.pop(key, None)
        else:
            # Cache the msg_ref even on failure so later tool swaps can retry
            # without requiring raw_message on the SessionSource event.
            self._rxn_msg_refs[key] = msg_ref
            # Do NOT mark active; a later no-tool completion will correctly retry.

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

            # Cooldown check — conservative 1.0s buffer over Discord's 0.25s limit (heuristic)
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
                    # success
                    self._rxn_active[key] = tool_emoji
                    self._rxn_last_swap[key] = now
                    self._rxn_stale.pop(key, None)
                    return
                else:
                    # Add new FIRST, then remove old — prevents zero-reaction
                    # reflow jitter on Discord (message shifts when reactions disappear)
                    ok = await self._reaction_add(msg_ref, tool_emoji)
                    if not ok:
                        return
                    if current and current != tool_emoji:
                        try:
                            ok_rm = await self._reaction_remove(msg_ref, current)
                            ok_rm = bool(ok_rm)
                        except Exception as e:
                            logger.debug("reaction swap remove failed (%s): %s", current, e)
                            ok_rm = False
                        if not ok_rm:
                            # Preserve old as stale for final cleanup
                            stale_set = self._rxn_stale.setdefault(key, set())
                            stale_set.add(current)
                        else:
                            # Remove succeeded: clear from stale if present
                            if key in self._rxn_stale:
                                self._rxn_stale[key].discard(current)
                                if not self._rxn_stale[key]:
                                    self._rxn_stale.pop(key, None)
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
        if not self._rxn_reactions_enabled():
            return

        key = self._reaction_msg_key(event)
        if key is None:
            return

        # Use try/finally to guarantee lock cleanup even on early return due to API False
        lock = self._rxn_lock(key)
        try:
            async with lock:
                msg_ref = self._rxn_msg_refs.get(key)
                if msg_ref is None:
                    try:
                        msg_ref = self._reaction_resolve_message(event)
                    except Exception:
                        msg_ref = None
                if msg_ref is None:
                    # No message to act on; clean tracking
                    self._rxn_active.pop(key, None)
                    self._rxn_msg_refs.pop(key, None)
                    self._rxn_last_swap.pop(key, None)
                    self._rxn_stale.pop(key, None)
                    return

                # Peek current and stale without popping yet; only pop after success
                current = self._rxn_active.get(key)
                stale = set(self._rxn_stale.get(key, set())) if hasattr(self, "_rxn_stale") else set()

                # Import here to avoid circular imports at module level
                from gateway.platforms.base import ProcessingOutcome

                if outcome == ProcessingOutcome.CANCELLED:
                    # Just clean up, don't change the reaction to persona
                    to_remove: set[str] = set()
                    if current:
                        to_remove.add(current)
                    to_remove.update(stale)
                    # For replace mode, cancel does not set replacement; just attempt removes
                    failed: set[str] = set()
                    for emoji in list(to_remove):
                        if self._reaction_replace_mode:
                            # replace_mode has no per-emoji remove; skip (already no set)
                            continue
                        try:
                            ok_rm = await self._reaction_remove(msg_ref, emoji)
                            ok_rm = bool(ok_rm)
                        except Exception as e:
                            logger.debug("cancel cleanup remove failed (%s): %s", emoji, e)
                            ok_rm = False
                        if not ok_rm:
                            failed.add(emoji)
                    if failed:
                        # Preserve failures for later cleanup; keep msg_ref/active
                        self._rxn_stale[key] = failed
                        # Keep current if it failed? current is in failed set, so keep active
                        if current and current in failed:
                            pass  # keep _rxn_active
                        else:
                            # current succeeded removed, so clear active
                            if current and current not in failed:
                                self._rxn_active.pop(key, None)
                        # Keep msg_ref for retry, but lock will be released via finally
                        # Keep last_swap?
                        return
                    # All removals succeeded: clean all
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
                        except Exception as e:
                            logger.debug("reaction complete set failed (%s -> %s): %s", current, translated, e)
                            ok = False
                        if not ok:
                            # Preserve current/stale for later retry; do not pop
                            return
                        # success
                        self._rxn_active.pop(key, None)
                        self._rxn_msg_refs.pop(key, None)
                        self._rxn_last_swap.pop(key, None)
                        self._rxn_stale.pop(key, None)
                        return
                    else:
                        need_add = (translated != current)
                        # If no active (start failed) we still need to add
                        if need_add:
                            try:
                                ok = await self._reaction_add(msg_ref, translated)
                                ok = bool(ok)
                            except Exception as e:
                                logger.debug("reaction complete add failed (%s -> %s): %s", current, translated, e)
                                ok = False
                            if not ok:
                                # Preserve current/stale/msg_ref for later recovery
                                return
                            # Add succeeded; current will be considered replaced
                        # Build set of emojis to remove (current + stale, excluding translated)
                        to_remove2: set[str] = set()
                        if current and current != translated:
                            to_remove2.add(current)
                        for em in list(stale):
                            if em != translated:
                                to_remove2.add(em)
                        failed2: set[str] = set()
                        for emoji in list(to_remove2):
                            try:
                                ok_rm = await self._reaction_remove(msg_ref, emoji)
                                ok_rm = bool(ok_rm)
                            except Exception as e:
                                logger.debug("reaction complete remove failed (%s): %s", emoji, e)
                                ok_rm = False
                            if not ok_rm:
                                failed2.add(emoji)
                        if failed2:
                            # Some removes failed; keep them stale for later, but final emoji is now active
                            # Update active to translated (since add succeeded)
                            self._rxn_active[key] = translated
                            # Keep msg_ref for retry
                            self._rxn_stale[key] = failed2
                            # Keep last_swap? Not needed after final, but preserve
                            return
                        # All succeeded: clean
                        self._rxn_active.pop(key, None)
                        self._rxn_msg_refs.pop(key, None)
                        self._rxn_last_swap.pop(key, None)
                        self._rxn_stale.pop(key, None)
                        return
                except Exception as e:
                    logger.debug("reaction complete swap failed (%s -> %s): %s", current, translated, e)
                    # Preserve state for retry
                    return
        finally:
            # Unconditional lock cleanup: pop the per-key lock dict entry
            # This runs even if we returned early inside the lock due to API False
            self._rxn_locks.pop(key, None)
