"""Platform-agnostic dynamic tool reactions mixin.

The mixin owns the reaction lifecycle for adapters that opt in with
``_init_reaction_mixin()``.  Every lifecycle operation is fenced by a per-key
turn generation and message token, and all remote effects are serialized on a
per-key lock.  A delayed callback can therefore never commit state for a
successor turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PERSONA_EMOJI = "👀"
_DEFAULT_TOOL_EMOJI = "⚙️"
_MAX_PERSONA_TEXT = 64


@dataclass(frozen=True)
class ReactionHookContext:
    """Immutable identity carried from one ``TurnContext`` tool callback.

    ``source`` remains the routing object expected by existing adapters.  The
    identity fields are snapshots: they must not be reconstructed from the
    adapter's current per-session cache because that cache may already point at
    a successor message.
    """

    source: Any
    run_generation: Optional[int] = None
    message_id: Optional[str] = None
    message_token: Optional[str] = None
    is_current: Optional[Callable[[], bool]] = None


def _rxn_normalize_bool(value: Any, default: bool = False) -> bool:
    """Normalize documented boolean scalars and fail closed for other types."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("1", "true", "yes", "on"):
            return True
        if token in ("0", "false", "no", "off", ""):
            return False
        return default
    return default


def _rxn_normalize_cooldown(value: Any, default: float = 1.0) -> float:
    """Return a finite non-negative numeric cooldown, else ``default``.

    Booleans are deliberately rejected even though Python treats them as
    integers: ``False`` must not silently disable hysteresis.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) and result >= 0 else default


def _rxn_persona_candidate(value: Any) -> Optional[str]:
    """Accept only bounded, non-empty text suitable for a reaction payload."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > _MAX_PERSONA_TEXT:
        return None
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        return None
    return value


def _rxn_track_callback(method):
    """Track lifecycle tasks so a successor can fence a delayed predecessor."""

    @functools.wraps(method)
    async def wrapped(self, event, *args, **kwargs):
        task = asyncio.current_task()
        key = None
        if task is not None:
            with contextlib.suppress(Exception):
                key = self._reaction_msg_key(event)
            if key is not None:
                self._rxn_callback_tasks.setdefault(key, set()).add(task)
        try:
            return await method(self, event, *args, **kwargs)
        except asyncio.CancelledError:
            if task is not None and task in self._rxn_fenced_tasks:
                return None
            raise
        finally:
            if key is not None:
                callbacks = self._rxn_callback_tasks.get(key)
                if callbacks:
                    callbacks.discard(task)
                    if not callbacks:
                        self._rxn_callback_tasks.pop(key, None)
                self._rxn_fenced_tasks.discard(task)

    return wrapped


class DynamicReactionMixin:
    """Shared dynamic reaction state machine for messaging adapters."""

    _reaction_replace_mode: bool = False

    async def _reaction_add(self, msg_ref: Any, emoji: str) -> bool:
        return False

    async def _reaction_remove(self, msg_ref: Any, emoji: str) -> bool:
        return False

    async def _reaction_set(self, msg_ref: Any, emoji: str) -> bool:
        return False

    def _reaction_resolve_message(self, event: Any) -> Any:
        return None

    def _reaction_msg_key(self, event: Any) -> Optional[Hashable]:
        return None

    def _reaction_translate_emoji(self, emoji: str) -> Optional[str]:
        return emoji

    def _init_reaction_mixin(self) -> None:
        """Initialize all reaction state before lifecycle hooks can run."""
        self._rxn_active: Dict[Hashable, str] = {}
        self._rxn_msg_refs: Dict[Hashable, Any] = {}
        self._rxn_last_swap: Dict[Hashable, float] = {}
        self._rxn_locks: Dict[Hashable, asyncio.Lock] = {}
        self._rxn_lock_users: Dict[Hashable, int] = {}
        self._rxn_callback_tasks: Dict[Hashable, set[asyncio.Task]] = {}
        self._rxn_fenced_tasks: set[asyncio.Task] = set()
        self._rxn_stale: Dict[Hashable, set[str]] = {}
        self._rxn_pending: Dict[Hashable, list[tuple[Any, set[str]]]] = {}
        self._rxn_completion_pending: Dict[Hashable, set[str]] = {}
        self._rxn_generation: Dict[Hashable, int] = {}
        self._rxn_tokens: Dict[Hashable, Optional[str]] = {}
        self._rxn_external_generations: Dict[Hashable, Optional[int]] = {}
        self._rxn_closed: Dict[Hashable, int] = {}
        self._rxn_initialized = False
        self._rxn_persona_emoji = self._rxn_resolve_persona_emoji()
        self._rxn_dynamic = self._rxn_resolve_dynamic_reactions()
        self._rxn_cooldown = self._rxn_resolve_cooldown()
        self._rxn_initialized = True

    def _rxn_resolve_persona_emoji(self) -> str:
        """Resolve platform text, then global text, then the safe default."""
        extra = getattr(getattr(self, "config", None), "extra", {}) or {}
        candidate = _rxn_persona_candidate(extra.get("persona_emoji"))
        if candidate is not None:
            return candidate
        try:
            from hermes_cli.config import load_config

            config = load_config()
            candidate = _rxn_persona_candidate(
                config.get("persona_emoji") if isinstance(config, dict) else None
            )
            return candidate or _DEFAULT_PERSONA_EMOJI
        except Exception:
            return _DEFAULT_PERSONA_EMOJI

    def _rxn_resolve_dynamic_reactions(self) -> bool:
        if not self._rxn_reactions_enabled():
            return False
        extra = getattr(getattr(self, "config", None), "extra", {}) or {}
        if "dynamic_reactions" in extra:
            return _rxn_normalize_bool(extra["dynamic_reactions"], default=False)
        try:
            from hermes_cli.config import load_config

            value = load_config()
            return _rxn_normalize_bool(
                value.get("dynamic_reactions", False)
                if isinstance(value, dict)
                else False,
                default=False,
            )
        except Exception:
            return False

    def _rxn_resolve_cooldown(self) -> float:
        extra = getattr(getattr(self, "config", None), "extra", {}) or {}
        return _rxn_normalize_cooldown(extra.get("reaction_cooldown", 1.0))

    def _rxn_reactions_enabled(self) -> bool:
        """Read an adapter gate using only documented scalar representations."""
        attr = getattr(self, "_reactions_enabled", None)
        if callable(attr):
            try:
                result = attr()
            except Exception:
                return False
        else:
            result = attr
        if result is None:
            return True
        return _rxn_normalize_bool(result, default=False)

    @staticmethod
    def _rxn_source(event: Any) -> Any:
        return getattr(event, "source", event)

    @staticmethod
    def _rxn_message_token(event: Any) -> Optional[str]:
        for name in ("message_token", "message_id"):
            value = getattr(event, name, None)
            if value is not None and str(value):
                return str(value)
        raw = getattr(event, "raw_message", None)
        raw_id = getattr(raw, "id", None)
        if raw_id is not None and str(raw_id):
            return str(raw_id)
        source = getattr(event, "source", None)
        source_id = getattr(source, "message_id", None)
        if source_id is not None and str(source_id):
            return str(source_id)
        return None

    @staticmethod
    def _rxn_external_generation(event: Any) -> Optional[int]:
        value = getattr(event, "run_generation", None)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _rxn_hook_is_current(event: Any) -> bool:
        check = getattr(event, "is_current", None)
        if not callable(check):
            return True
        try:
            return bool(check())
        except Exception:
            return False

    def _rxn_lock(self, key: Hashable) -> asyncio.Lock:
        lock = self._rxn_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._rxn_locks[key] = lock
        return lock

    @contextlib.asynccontextmanager
    async def _rxn_locked(self, key: Hashable):
        """Hold the lifecycle lock without dropping it under a queued waiter."""
        lock = self._rxn_lock(key)
        self._rxn_lock_users[key] = self._rxn_lock_users.get(key, 0) + 1
        acquired = False
        try:
            await lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                lock.release()
            users = self._rxn_lock_users.get(key, 1) - 1
            if users:
                self._rxn_lock_users[key] = users
            else:
                self._rxn_lock_users.pop(key, None)
                if key in self._rxn_closed:
                    self._rxn_locks.pop(key, None)

    def _rxn_reserve_generation(
        self, key: Hashable, token: Optional[str], external_generation: Optional[int]
    ) -> int:
        """Fence older callbacks before waiting on the lifecycle lock.

        Reservation is intentionally synchronous: a successor arriving while a
        predecessor is in an awaited Discord call invalidates that predecessor
        before it can commit state when the call returns.
        """
        generation = self._rxn_generation.get(key, 0) + 1
        self._rxn_generation[key] = generation
        self._rxn_tokens[key] = token
        self._rxn_external_generations[key] = external_generation
        self._rxn_closed.pop(key, None)
        current_task = asyncio.current_task()
        for callback_task in tuple(self._rxn_callback_tasks.get(key, ())):
            if callback_task is not current_task and not callback_task.done():
                self._rxn_fenced_tasks.add(callback_task)
                callback_task.cancel()
        return generation

    def _rxn_identity_current(
        self,
        key: Hashable,
        generation: int,
        token: Optional[str],
        msg_ref: Any = None,
        *,
        allow_closed: bool = False,
    ) -> bool:
        if self._rxn_generation.get(key) != generation:
            return False
        current_token = self._rxn_tokens.get(key)
        if token is not None and current_token != token:
            return False
        current_ref = self._rxn_msg_refs.get(key)
        if (
            msg_ref is not None
            and current_ref is not None
            and current_ref is not msg_ref
        ):
            return False
        return allow_closed or self._rxn_closed.get(key) != generation

    def _rxn_external_generation_matches(
        self, key: Hashable, external_generation: Optional[int]
    ) -> bool:
        current = self._rxn_external_generations.get(key)
        return (
            external_generation is None
            or current is None
            or current == external_generation
        )

    def _rxn_close_generation(self, key: Hashable, generation: int) -> None:
        self._rxn_closed[key] = generation

    def _rxn_enqueue_pending(
        self, key: Hashable, msg_ref: Any, emojis: set[str]
    ) -> None:
        emojis = {emoji for emoji in emojis if emoji}
        if msg_ref is None or not emojis:
            return
        pending = self._rxn_pending.setdefault(key, [])
        for pending_ref, pending_emojis in pending:
            if pending_ref is msg_ref:
                pending_emojis.update(emojis)
                return
        pending.append((msg_ref, emojis))

    async def _rxn_drain_pending(
        self,
        key: Hashable,
        generation: int,
        token: Optional[str],
        *,
        event: Any = None,
        allow_closed: bool = False,
    ) -> bool:
        """Retry old-message cleanup, retaining every failed/unattempted item."""
        pending = self._rxn_pending.get(key)
        if not pending or self._reaction_replace_mode:
            return True
        remaining: list[tuple[Any, set[str]]] = []
        for msg_ref, emojis in list(pending):
            failed: set[str] = set()
            for index, emoji in enumerate(list(emojis)):
                if not self._rxn_identity_current(
                    key, generation, token, allow_closed=allow_closed
                ) or (event is not None and not self._rxn_hook_is_current(event)):
                    failed.update(list(emojis)[index:])
                    remaining.append((msg_ref, failed))
                    self._rxn_pending[key] = remaining
                    return False
                try:
                    ok = bool(await self._reaction_remove(msg_ref, emoji))
                except Exception as exc:
                    logger.debug("pending reaction cleanup failed (%s): %s", emoji, exc)
                    ok = False
                if not self._rxn_identity_current(
                    key, generation, token, allow_closed=allow_closed
                ) or (event is not None and not self._rxn_hook_is_current(event)):
                    if not ok:
                        failed.add(emoji)
                    failed.update(list(emojis)[index + 1 :])
                    remaining.append((msg_ref, failed))
                    self._rxn_pending[key] = remaining
                    return False
                if not ok:
                    failed.add(emoji)
            if failed:
                remaining.append((msg_ref, failed))
        if remaining:
            self._rxn_pending[key] = remaining
            return False
        self._rxn_pending.pop(key, None)
        return True

    def _rxn_translate(self, emoji: str, fallback: str) -> str:
        translated = self._reaction_translate_emoji(emoji)
        return translated or self._reaction_translate_emoji(fallback) or fallback

    @_rxn_track_callback
    async def _rxn_on_processing_start(self, event: Any) -> bool:
        if (
            not getattr(self, "_rxn_initialized", False)
            or not self._rxn_reactions_enabled()
        ):
            return False
        if not self._rxn_hook_is_current(event):
            return False
        key = self._reaction_msg_key(event)
        msg_ref = self._reaction_resolve_message(event)
        if key is None or msg_ref is None:
            return False
        token = self._rxn_message_token(event)
        external_generation = self._rxn_external_generation(event)

        old_ref = self._rxn_msg_refs.get(key)
        old_token = self._rxn_tokens.get(key)
        old_generation = self._rxn_generation.get(key)
        old_closed = (
            old_generation is not None and self._rxn_closed.get(key) == old_generation
        )
        old_active = self._rxn_active.get(key)
        old_stale = set(self._rxn_stale.get(key, set()))
        old_completion_pending = set(self._rxn_completion_pending.get(key, set()))
        generation = self._rxn_reserve_generation(key, token, external_generation)

        async with self._rxn_locked(key):
            if self._rxn_generation.get(key) != generation:
                return False
            replacing = old_ref is not None and (
                old_ref is not msg_ref or old_token != token
            )
            if replacing:
                cleanup = old_stale | old_completion_pending
                if not old_closed and old_active:
                    cleanup.add(old_active)
                if cleanup:
                    self._rxn_enqueue_pending(key, old_ref, cleanup)
                self._rxn_active.pop(key, None)
                self._rxn_msg_refs.pop(key, None)
                self._rxn_stale.pop(key, None)
                self._rxn_completion_pending.pop(key, None)
                self._rxn_last_swap.pop(key, None)
            elif old_ref is None:
                self._rxn_active.pop(key, None)
                self._rxn_stale.pop(key, None)
                self._rxn_completion_pending.pop(key, None)
                self._rxn_last_swap.pop(key, None)

            self._rxn_msg_refs[key] = msg_ref
            if not await self._rxn_drain_pending(key, generation, token, event=event):
                if not self._rxn_identity_current(key, generation, token, msg_ref):
                    return False
            if not self._rxn_identity_current(
                key, generation, token, msg_ref
            ) or not self._rxn_hook_is_current(event):
                return False

            persona = self._rxn_translate(
                self._rxn_persona_emoji, _DEFAULT_PERSONA_EMOJI
            )
            try:
                if self._reaction_replace_mode:
                    ok = bool(await self._reaction_set(msg_ref, persona))
                else:
                    ok = bool(await self._reaction_add(msg_ref, persona))
            except Exception as exc:
                logger.debug("reaction start failed (%s): %s", persona, exc)
                ok = False
            if not self._rxn_identity_current(
                key, generation, token, msg_ref
            ) or not self._rxn_hook_is_current(event):
                if ok and not self._reaction_replace_mode:
                    with contextlib.suppress(Exception):
                        await self._reaction_remove(msg_ref, persona)
                return False
            if ok:
                self._rxn_active[key] = persona
                self._rxn_stale.pop(key, None)
            return ok

    @_rxn_track_callback
    async def _rxn_on_tool_call_start(self, event: Any, tool_name: str) -> None:
        if not getattr(self, "_rxn_initialized", False) or not self._rxn_dynamic:
            return
        if not self._rxn_hook_is_current(event):
            return
        key = self._reaction_msg_key(event)
        if key is None:
            return
        token = self._rxn_message_token(event)
        external_generation = self._rxn_external_generation(event)
        generation = self._rxn_generation.get(key)
        if generation is None or self._rxn_closed.get(key) == generation:
            return
        if token is not None and self._rxn_tokens.get(key) != token:
            return
        if not self._rxn_external_generation_matches(key, external_generation):
            return

        async with self._rxn_locked(key):
            if (
                not self._rxn_identity_current(key, generation, token)
                or not self._rxn_external_generation_matches(key, external_generation)
                or not self._rxn_hook_is_current(event)
            ):
                return
            msg_ref = self._rxn_msg_refs.get(key)
            if msg_ref is None:
                return
            now = time.monotonic()
            if now - self._rxn_last_swap.get(key, 0.0) < self._rxn_cooldown:
                return
            from agent.display import get_tool_emoji

            tool_emoji = self._rxn_translate(
                get_tool_emoji(tool_name, default=_DEFAULT_TOOL_EMOJI),
                _DEFAULT_TOOL_EMOJI,
            )
            current = self._rxn_active.get(key)
            if current == tool_emoji:
                return
            try:
                if self._reaction_replace_mode:
                    ok = bool(await self._reaction_set(msg_ref, tool_emoji))
                    if not ok:
                        return
                    if not self._rxn_identity_current(
                        key, generation, token, msg_ref
                    ) or not self._rxn_hook_is_current(event):
                        return
                    self._rxn_active[key] = tool_emoji
                    self._rxn_last_swap[key] = now
                    return

                ok = bool(await self._reaction_add(msg_ref, tool_emoji))
                if not ok:
                    return
                if not self._rxn_identity_current(
                    key, generation, token, msg_ref
                ) or not self._rxn_hook_is_current(event):
                    with contextlib.suppress(Exception):
                        await self._reaction_remove(msg_ref, tool_emoji)
                    return
                if current and current != tool_emoji:
                    try:
                        removed = bool(await self._reaction_remove(msg_ref, current))
                    except Exception as exc:
                        logger.debug(
                            "reaction swap remove failed (%s): %s", current, exc
                        )
                        removed = False
                    if not self._rxn_identity_current(
                        key, generation, token, msg_ref
                    ) or not self._rxn_hook_is_current(event):
                        return
                    if not removed:
                        self._rxn_stale.setdefault(key, set()).add(current)
                    else:
                        stale = self._rxn_stale.get(key)
                        if stale:
                            stale.discard(current)
                            if not stale:
                                self._rxn_stale.pop(key, None)
                if not self._rxn_identity_current(
                    key, generation, token, msg_ref
                ) or not self._rxn_hook_is_current(event):
                    return
                self._rxn_active[key] = tool_emoji
                self._rxn_last_swap[key] = now
            except Exception as exc:
                logger.debug(
                    "reaction swap failed (%s -> %s): %s", current, tool_emoji, exc
                )

    @_rxn_track_callback
    async def _rxn_on_processing_complete(self, event: Any, outcome: Any) -> None:
        if (
            not getattr(self, "_rxn_initialized", False)
            or not self._rxn_reactions_enabled()
        ):
            return
        key = self._reaction_msg_key(event)
        if key is None:
            return
        token = self._rxn_message_token(event)
        external_generation = self._rxn_external_generation(event)
        generation = self._rxn_generation.get(key)
        if generation is None:
            return
        if token is not None and self._rxn_tokens.get(key) != token:
            return
        if not self._rxn_external_generation_matches(key, external_generation):
            return

        async with self._rxn_locked(key):
            if not self._rxn_identity_current(
                key, generation, token, allow_closed=True
            ):
                return
            if not self._rxn_external_generation_matches(key, external_generation):
                return
            msg_ref = self._rxn_msg_refs.get(key)
            if msg_ref is None:
                resolved = self._reaction_resolve_message(event)
                if resolved is None:
                    return
                msg_ref = resolved
                self._rxn_msg_refs[key] = msg_ref
            event_ref = getattr(event, "raw_message", None)
            if event_ref is not None and event_ref is not msg_ref:
                return
            if not await self._rxn_drain_pending(
                key, generation, token, event=event, allow_closed=True
            ):
                # A failed old-message cleanup must not prevent current finalization.
                pass

            current = self._rxn_active.get(key)
            stale = set(self._rxn_stale.get(key, set()))
            from gateway.platforms.base import ProcessingOutcome

            if outcome == ProcessingOutcome.CANCELLED:
                to_remove = set(stale)
                if current:
                    to_remove.add(current)
                failed: set[str] = set()
                for emoji in to_remove:
                    if self._reaction_replace_mode:
                        continue
                    if not self._rxn_identity_current(
                        key, generation, token, msg_ref, allow_closed=True
                    ) or not self._rxn_hook_is_current(event):
                        return
                    try:
                        removed = bool(await self._reaction_remove(msg_ref, emoji))
                    except Exception as exc:
                        logger.debug("cancel cleanup failed (%s): %s", emoji, exc)
                        removed = False
                    if not self._rxn_identity_current(
                        key, generation, token, msg_ref, allow_closed=True
                    ) or not self._rxn_hook_is_current(event):
                        return
                    if not removed:
                        failed.add(emoji)
                if failed:
                    self._rxn_stale[key] = failed
                    if current and current not in failed:
                        self._rxn_active.pop(key, None)
                    self._rxn_close_generation(key, generation)
                    return
                self._rxn_active.pop(key, None)
                self._rxn_stale.pop(key, None)
                self._rxn_completion_pending.pop(key, None)
                self._rxn_msg_refs.pop(key, None)
                self._rxn_last_swap.pop(key, None)
                self._rxn_close_generation(key, generation)
                return

            final_raw = (
                self._rxn_persona_emoji
                if outcome == ProcessingOutcome.SUCCESS
                else "❌"
            )
            final = self._rxn_translate(
                final_raw,
                _DEFAULT_PERSONA_EMOJI
                if outcome == ProcessingOutcome.SUCCESS
                else "❌",
            )
            need_add = current != final
            if self._reaction_replace_mode:
                try:
                    ok = bool(await self._reaction_set(msg_ref, final))
                except Exception as exc:
                    logger.debug("reaction completion set failed (%s): %s", final, exc)
                    ok = False
                if not self._rxn_identity_current(
                    key, generation, token, msg_ref, allow_closed=True
                ) or not self._rxn_hook_is_current(event):
                    return
                if not ok:
                    self._rxn_completion_pending[key] = (
                        set(stale) | ({current} if current else set()) | {final}
                    )
                    if current:
                        self._rxn_active[key] = current
                    self._rxn_close_generation(key, generation)
                    return
                self._rxn_active.pop(key, None)
                self._rxn_stale.pop(key, None)
                self._rxn_completion_pending.pop(key, None)
                self._rxn_msg_refs.pop(key, None)
                self._rxn_last_swap.pop(key, None)
                self._rxn_close_generation(key, generation)
                return

            if need_add:
                if not self._rxn_identity_current(
                    key, generation, token, msg_ref, allow_closed=True
                ) or not self._rxn_hook_is_current(event):
                    return
                try:
                    added = bool(await self._reaction_add(msg_ref, final))
                except Exception as exc:
                    logger.debug(
                        "reaction completion add failed (%s -> %s): %s",
                        current,
                        final,
                        exc,
                    )
                    added = False
                if not self._rxn_identity_current(
                    key, generation, token, msg_ref, allow_closed=True
                ) or not self._rxn_hook_is_current(event):
                    if added:
                        with contextlib.suppress(Exception):
                            await self._reaction_remove(msg_ref, final)
                    return
                if not added:
                    candidates = stale | ({current} if current else set()) | {final}
                    self._rxn_completion_pending[key] = candidates
                    self._rxn_close_generation(key, generation)
                    return

            to_remove = {emoji for emoji in stale if emoji != final}
            if current and current != final:
                to_remove.add(current)
            failed: set[str] = set()
            for emoji in to_remove:
                if not self._rxn_identity_current(
                    key, generation, token, msg_ref, allow_closed=True
                ) or not self._rxn_hook_is_current(event):
                    return
                try:
                    removed = bool(await self._reaction_remove(msg_ref, emoji))
                except Exception as exc:
                    logger.debug(
                        "reaction completion remove failed (%s): %s", emoji, exc
                    )
                    removed = False
                if not self._rxn_identity_current(
                    key, generation, token, msg_ref, allow_closed=True
                ) or not self._rxn_hook_is_current(event):
                    return
                if not removed:
                    failed.add(emoji)

            if failed:
                self._rxn_active[key] = final
                self._rxn_stale[key] = failed
                self._rxn_completion_pending.pop(key, None)
                self._rxn_close_generation(key, generation)
                return
            self._rxn_active.pop(key, None)
            self._rxn_stale.pop(key, None)
            self._rxn_completion_pending.pop(key, None)
            self._rxn_msg_refs.pop(key, None)
            self._rxn_last_swap.pop(key, None)
            self._rxn_close_generation(key, generation)
