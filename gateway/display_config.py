"""Per-platform display/verbosity resolver (``resolve_display_setting``).

Resolution order, first non-None wins: ``display.platforms.<platform>.<key>`` →
``display.<key>`` → ``_PLATFORM_DEFAULTS[platform][key]`` → ``_GLOBAL_DEFAULTS[key]``.
Exception: ``display.streaming`` is CLI-only; gateway streaming follows the top-level
``streaming`` config unless a per-platform override sets it. Legacy
``display.tool_progress_overrides`` is still read as a ``tool_progress`` fallback.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Settings configurable per-platform; other display settings are CLI-only.
_GLOBAL_DEFAULTS: dict[str, Any] = {
    "tool_progress": "all",
    "tool_progress_grouping": "accumulate",  # "accumulate" = edit one bubble; "separate" = one msg per tool
    "tool_progress_filter": {},  # per-tool/category overrides: {"terminal": "off", "skills": "all", "mcp": "off"}
    "show_reasoning": False,
    "reasoning_style": "code",  # "code" (💭 **Reasoning:** + fence), "blockquote" ("> "), "subtext" ("-# " Discord)
    "tool_preview_length": 0,
    "streaming": None,  # None = follow top-level streaming config
    # Gateway-only assistant/status chatter; mobile platforms opt down to final-answer-first.
    "interim_assistant_messages": True,
    "suppress_warning_notifications": False,
    "long_running_notifications": True,
    "busy_ack_detail": True,
    "busy_steer_ack_enabled": True,  # busy_input_mode=steer echo; the text still lands in the run
    # Delete tool-progress / "⏳ Working" bubbles after a SUCCESSFUL final response where deletion is
    # supported (Telegram); failed runs keep them as breadcrumbs.
    "cleanup_progress": False,
    # Working-state text on text-rendering indicators (Slack assistant status): "full"/true = verb +
    # argument preview, "verb" = verb only (keeps paths out of shared channels), "off"/false = static.
    "live_status": "full",
}

# Tiers: HIGH = editing, personal/team use; MEDIUM = editing but customer-facing;
# LOW = no edit support (progress messages are permanent); MINIMAL = batch delivery.
_TIER_HIGH = {
    "tool_progress": "all", "show_reasoning": False, "tool_preview_length": 40,
    "streaming": None,  # follow global
    "interim_assistant_messages": True, "long_running_notifications": True, "busy_ack_detail": True,
}
_TIER_MEDIUM = {**_TIER_HIGH, "tool_progress": "new"}
_TIER_LOW = {
    **_TIER_HIGH, "tool_progress": "off", "streaming": False,
    "interim_assistant_messages": False, "long_running_notifications": False, "busy_ack_detail": False,
}
_TIER_MINIMAL = {**_TIER_LOW, "tool_preview_length": 0}

_PLATFORM_DEFAULTS: dict[str, dict[str, Any]] = {
    # Mobile inbox: quiet tool_progress / busy-ack, but keep interim commentary and heartbeats so it
    # doesn't look like "typing..." for 30 minutes.
    "telegram": {**_TIER_HIGH, "tool_progress": "off", "busy_ack_detail": False},
    "discord": {**_TIER_HIGH, "reasoning_style": "subtext"},  # "-# " subtext reads as metadata
    # Slack: Bolt posts cannot be edited like CLI; "new"/"all" spam permanent lines.
    "slack": {**_TIER_MEDIUM, "tool_progress": "off", "long_running_notifications": False, "busy_ack_detail": False},
    "mattermost": _TIER_MEDIUM,
    "matrix": _TIER_MEDIUM,
    "feishu": _TIER_MEDIUM,
    "buzz": _TIER_MEDIUM,  # Nostr: edits in place but channels are shared community spaces
    "signal": _TIER_LOW,
    "whatsapp": _TIER_MEDIUM,  # Baileys bridge supports /edit
    "whatsapp_cloud": _TIER_LOW,  # adapter lacks edit_message; promote once it lands
    "photon": _TIER_LOW,  # permanent-message iMessage inboxes (no edit)
    "bluebubbles": _TIER_LOW,
    "weixin": _TIER_LOW,
    # Non-editable, but its native "stream" msgtype gives a typing animation + cumulative updates.
    "wecom": {**_TIER_LOW, "streaming": True},
    "wecom_callback": _TIER_LOW,
    "dingtalk": _TIER_LOW,
    "email": _TIER_MINIMAL,
    "sms": _TIER_MINIMAL,
    "webhook": _TIER_MINIMAL,
    "homeassistant": _TIER_MINIMAL,
    "api_server": {**_TIER_HIGH, "tool_preview_length": 0},
}

# Canonical set of per-platform overrideable keys (for validation).
OVERRIDEABLE_KEYS = frozenset(_GLOBAL_DEFAULTS.keys())


def resolve_display_setting(user_config: dict, platform_key: str, setting: str, fallback: Any = None) -> Any:
    """Resolve a display setting with per-platform override support (see module docstring for order).

    ``platform_key`` is the platform config key (``"telegram"``; see ``_platform_config_key`` in
    gateway/run.py). Returns *fallback* when nothing is configured.
    """
    configured = _configured_display_value(user_config, platform_key, setting)
    if configured is not None:
        return _normalise(setting, configured)
    val = _PLATFORM_DEFAULTS.get(platform_key, {}).get(setting)
    if val is None:
        val = _GLOBAL_DEFAULTS.get(setting)
    return fallback if val is None else val


def _configured_display_value(user_config: dict, platform_key: str, setting: str) -> Any:
    """First non-None operator value, without introducing tier defaults."""
    display_cfg = user_config.get("display")
    if not isinstance(display_cfg, dict):
        return None
    platforms = display_cfg.get("platforms")
    plat_overrides = platforms.get(platform_key) if isinstance(platforms, dict) else None
    if isinstance(plat_overrides, dict) and plat_overrides.get(setting) is not None:
        return plat_overrides[setting]
    if setting == "tool_progress":
        legacy = display_cfg.get("tool_progress_overrides")
        if isinstance(legacy, dict) and legacy.get(platform_key) is not None:
            return legacy[platform_key]
    if setting != "streaming":  # display.streaming is CLI-only
        return display_cfg.get(setting)
    return None


def resolve_tool_progress(user_config: dict, platform_key: str, env_mode: str | None = None) -> tuple[str, bool]:
    """Return (mode, explicit intent) from the same winning source.

    Non-None YAML wins over the legacy env bridge. Null inherits through to env,
    then tier defaults. A tier's off is not an operator request to disable cards.
    """
    configured = _configured_display_value(user_config, platform_key, "tool_progress")
    if configured is not None:
        return _normalise("tool_progress", configured), True
    if env_mode:
        return _normalise("tool_progress", env_mode), True
    return resolve_display_setting(user_config, platform_key, "tool_progress"), False


# --- Normalisation of YAML quirks (bare ``off`` → False in YAML 1.1, etc.) ---

_TRUTHY = {"true", "1", "yes", "on"}
_FALSY = {"false", "0", "no"}


def _norm_tristate(on: str, off: str, choices: set, extra_truthy: set = frozenset()):
    """Normaliser for bool-or-keyword settings: bools/truthy tokens → *on*, falsy → *off*, else a known choice or *on*."""
    def norm(value: Any) -> str:
        if isinstance(value, bool):
            return on if value else off
        val = str(value).strip().lower()
        if val in _FALSY:
            return off
        if val in _TRUTHY | extra_truthy:
            return on
        return val if val in choices else on
    return norm


def _norm_bool(value: Any) -> bool:
    return value.strip().lower() in _TRUTHY | {"raw", "verbose"} if isinstance(value, str) else bool(value)


def _norm_long_running(value: Any) -> Any:
    return "generic" if isinstance(value, str) and value.strip().lower() == "generic" else _norm_bool(value)


def _norm_suppress_warning_notifications(value: Any) -> bool:
    # Only an explicit, recognized opt-in may hide engine diagnostics.
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY
    return False


def _norm_cleanup_progress(value: Any) -> bool:
    return value.lower() in _TRUTHY if isinstance(value, str) else bool(value)


def _norm_choice(choices: tuple[str, ...]) -> Any:
    def norm(value: Any) -> str:
        val = str(value).lower()
        return val if val in choices else choices[0]

    return norm


def _norm_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ``tool_progress_filter`` accepts tool names/categories as keys and the same
# modes as ``tool_progress`` as values. A list is an allowlist shorthand.
_FILTER_VALID_MODES = {"off", "new", "all", "verbose", "log"}


def _norm_tool_progress_filter(value: Any) -> dict[str, str]:
    """Normalize a filter without allowing malformed config to alter visibility."""
    if value is None:
        return {}
    if isinstance(value, list):
        result: dict[str, str] = {}
        for entry in value:
            if not isinstance(entry, str):
                logger.warning("Ignoring non-string tool_progress_filter list entry %r", entry)
                continue
            key = entry.strip().lower()
            if key:
                result[key] = "all"
            else:
                logger.warning("Ignoring empty tool_progress_filter list entry %r", entry)
        return result
    if not isinstance(value, dict):
        logger.warning(
            "Invalid tool_progress_filter (expected dict or list, got %s); ignoring",
            type(value).__name__,
        )
        return {}

    result: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            logger.warning("Ignoring non-string tool_progress_filter key %r", raw_key)
            continue
        key = raw_key.strip().lower()
        if not key:
            logger.warning("Ignoring empty/blank tool_progress_filter key %r", raw_key)
            continue
        if isinstance(raw_value, bool):
            mode = "all" if raw_value else "off"
        elif isinstance(raw_value, str):
            normalized = raw_value.strip().lower()
            if normalized in _FALSY:
                mode = "off"
            elif normalized in _TRUTHY:
                mode = "all"
            elif normalized in _FILTER_VALID_MODES:
                mode = normalized
            else:
                logger.warning("Ignoring unknown tool_progress_filter mode %r for key %r", raw_value, raw_key)
                continue
        elif isinstance(raw_value, int) and raw_value in (0, 1):
            mode = "all" if raw_value else "off"
        else:
            logger.warning("Ignoring malformed tool_progress_filter value %r for key %r", raw_value, raw_key)
            continue
        result[key] = mode
    return result


def resolve_tool_progress_filter(user_config: dict, platform_key: str) -> dict[str, str]:
    """Return global and per-platform filter entries with platform precedence."""
    try:
        display_cfg = user_config.get("display") or {}
        if not isinstance(display_cfg, dict):
            return {}
        global_raw = display_cfg.get("tool_progress_filter")
        platform_raw = None
        platforms = display_cfg.get("platforms")
        if isinstance(platforms, dict):
            platform_cfg = platforms.get(platform_key)
            if isinstance(platform_cfg, dict):
                platform_raw = platform_cfg.get("tool_progress_filter")
        global_filter = _norm_tool_progress_filter(global_raw) if global_raw is not None else {}
        platform_filter = _norm_tool_progress_filter(platform_raw) if platform_raw is not None else None
        if platform_filter is None or not platform_filter:
            return global_filter
        merged = dict(global_filter)
        merged.update(platform_filter)
        return merged
    except Exception as exc:
        logger.debug("resolve_tool_progress_filter failed: %s", exc)
        return {}


_NORMALISERS: dict[str, Any] = {
    "tool_progress": _norm_tristate("all", "off", {"off", "new", "all", "verbose", "log"}),
    "tool_progress_filter": _norm_tool_progress_filter,
    "show_reasoning": _norm_bool,
    "streaming": _norm_bool,
    "interim_assistant_messages": _norm_bool,
    "suppress_warning_notifications": _norm_suppress_warning_notifications,
    "long_running_notifications": _norm_long_running,
    "busy_ack_detail": _norm_bool,
    "busy_steer_ack_enabled": _norm_bool,
    "thinking_progress": _norm_bool,
    "cleanup_progress": _norm_cleanup_progress,
    "live_status": _norm_tristate("full", "off", {"full", "verb", "off"}, extra_truthy={"all"}),
    "tool_progress_grouping": _norm_choice(("accumulate", "separate")),
    "reasoning_style": _norm_choice(("code", "blockquote", "subtext")),
    "tool_preview_length": _norm_int,
}


def _normalise(setting: str, value: Any) -> Any:
    """Normalise a user-supplied value for *setting*; unknown settings pass through."""
    norm = _NORMALISERS.get(setting)
    return norm(value) if norm else value
