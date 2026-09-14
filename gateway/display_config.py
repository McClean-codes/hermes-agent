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
    display_cfg = user_config.get("display") or {}
    plat_overrides = (display_cfg.get("platforms") or {}).get(platform_key)
    if isinstance(plat_overrides, dict) and plat_overrides.get(setting) is not None:
        return _normalise(setting, plat_overrides[setting])
    if setting == "tool_progress":  # legacy display.tool_progress_overrides.<platform>
        legacy = display_cfg.get("tool_progress_overrides")
        if isinstance(legacy, dict) and legacy.get(platform_key) is not None:
            return _normalise(setting, legacy[platform_key])
    if setting != "streaming" and display_cfg.get(setting) is not None:  # display.streaming is CLI-only
        return _normalise(setting, display_cfg[setting])
    val = _PLATFORM_DEFAULTS.get(platform_key, {}).get(setting)
    if val is None:
        val = _GLOBAL_DEFAULTS.get(setting)
    return fallback if val is None else val


def resolve_tool_progress_filter(user_config: dict, platform_key: str) -> dict:
    """Return the merged tool_progress_filter for a platform.

    Merges the global display.tool_progress_filter with the per-platform
    display.platforms.<platform>.tool_progress_filter override, with
    platform entries taking precedence. Handles both dict and list forms
    via normalization, de-duplicates case-insensitively, and fails safe
    to {} on malformed config. Never raises.
    """
    try:
        display_cfg = user_config.get("display") or {}
        if not isinstance(display_cfg, dict):
            return {}
        global_raw = display_cfg.get("tool_progress_filter")
        plat_raw = None
        platforms = display_cfg.get("platforms")
        if isinstance(platforms, dict):
            plat_cfg = platforms.get(platform_key)
            if isinstance(plat_cfg, dict):
                plat_raw = plat_cfg.get("tool_progress_filter")
        global_norm = _norm_tool_progress_filter(global_raw) if global_raw is not None else {}
        plat_norm = _norm_tool_progress_filter(plat_raw) if plat_raw is not None else None
        if plat_norm is None:
            return global_norm
        if not plat_norm and global_raw is None:
            # No platform filter and no global filter → empty (use defaults)
            return {}
        if not plat_norm:
            # Platform filter explicitly empty but global exists → keep global
            # unless platform deliberately set to empty to clear global? We treat empty
            # platform dict as "no override" to avoid surprise clearing.
            # If user wants to clear global for a platform, they should set platform filter
            # to a dict containing the overrides they want (or an explicit empty is ignored).
            return global_norm
        # Merge: global as base, platform overrides
        merged = dict(global_norm)
        merged.update(plat_norm)
        return merged
    except Exception as e:
        logger.debug("resolve_tool_progress_filter failed: %s", e)
        return {}


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


# ---- tool_progress_filter normaliser ---------------------------------------
# Valid modes for the filter are the same as tool_progress but interpreted per tool/category.
# Unknown/malformed entries are skipped (fail-safe: fallback to global mode) rather than
# flooding or hiding. Empty/None → {}. List input is treated as an allowlist (each entry → "all").
# Category aliases are canonicalized before dedup/merge so platform alias precedence is deterministic.

_FILTER_VALID_MODES = {"off", "new", "all", "verbose", "log"}

# Canonical category keys for alias handling; platform alias entries must override
# global canonical entries (platform-wins). Duplicate spellings within one dict use last-wins.
_FILTER_CATEGORY_CANONICAL: dict[str, str] = {
    "skill": "skills",
    "skills": "skills",
    "mcp": "mcp",
    "mcp_tools": "mcp",
    "mcp-tools": "mcp",
    "mcp_tool": "mcp",
    "plugin": "plugins",
    "plugins": "plugins",
}

def _canonical_filter_key(key: str) -> str:
    return _FILTER_CATEGORY_CANONICAL.get(key, key)

def _norm_tool_progress_filter(value: Any) -> dict:
    """Normalize display.tool_progress_filter to a clean {tool_or_category: mode} dict.

    - dict expected: keys are tool names or categories (skills/mcp/plugins), values are modes.
    - list allowed as allowlist shorthand: each string entry maps to "all".
    - None/missing → {}.
    - malformed entries are skipped with a warning; duplicate keys (case-insensitive) last wins.
    - values are normalized: booleans → off/all, truthy/falsy strings → all/off, known modes pass through.
    Unknown mode strings are skipped (warn) rather than defaulting to "all" to avoid flooding.
    Category aliases (skill/skills, mcp_tools/mcp, plugin/plugins) are canonicalized before dedup
    so last-wins across alias spellings is deterministic.
    """
    if value is None:
        return {}
    # Allowlist shorthand: list of tool/category names
    if isinstance(value, list):
        out: dict[str, str] = {}
        for entry in value:
            if not isinstance(entry, str):
                logger.warning("Ignoring non-string tool_progress_filter list entry %r", entry)
                continue
            raw_key = entry.strip().lower()
            if not raw_key:
                logger.warning("Ignoring empty tool_progress_filter list entry %r", entry)
                continue
            key = _canonical_filter_key(raw_key)
            if key in out:
                logger.debug("Duplicate tool_progress_filter key %r (canonical %r), last wins", entry, key)
            out[key] = "all"
        if out:
            logger.debug("Normalized list-style tool_progress_filter to dict with %d entries", len(out))
        return out
    if not isinstance(value, dict):
        logger.warning("Invalid tool_progress_filter (expected dict or list, got %s); ignoring", type(value).__name__)
        return {}
    out = {}
    for raw_k, raw_v in value.items():
        if not isinstance(raw_k, str):
            logger.warning("Ignoring non-string tool_progress_filter key %r", raw_k)
            continue
        raw_key = raw_k.strip().lower()
        if not raw_key:
            logger.warning("Ignoring empty/blank tool_progress_filter key %r", raw_k)
            continue
        key = _canonical_filter_key(raw_key)
        mode = None
        if isinstance(raw_v, bool):
            mode = "all" if raw_v else "off"
        elif isinstance(raw_v, str):
            val = raw_v.strip().lower()
            if val in _FALSY:
                mode = "off"
            elif val in _TRUTHY:
                mode = "all"
            elif val in _FILTER_VALID_MODES:
                mode = val
            else:
                logger.warning("Ignoring unknown tool_progress_filter mode %r for key %r", raw_v, raw_k)
                continue
        elif raw_v is None:
            logger.warning("Ignoring None tool_progress_filter value for key %r", raw_k)
            continue
        else:
            try:
                if isinstance(raw_v, int) and raw_v in (0, 1):
                    mode = "all" if raw_v else "off"
                else:
                    logger.warning("Ignoring malformed tool_progress_filter value %r for key %r", raw_v, raw_k)
                    continue
            except Exception:
                logger.warning("Ignoring malformed tool_progress_filter value %r for key %r", raw_v, raw_k)
                continue
        if key in out:
            logger.debug("Duplicate tool_progress_filter key %r (canonical %r), last wins", raw_k, key)
        out[key] = mode
    return out


_NORMALISERS: dict[str, Any] = {
    "tool_progress": _norm_tristate("all", "off", {"off", "new", "all", "verbose", "log"}),
    "tool_progress_filter": _norm_tool_progress_filter,
    "show_reasoning": _norm_bool,
    "streaming": _norm_bool,
    "interim_assistant_messages": _norm_bool,
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
