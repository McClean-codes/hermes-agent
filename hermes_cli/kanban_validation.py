"""Shared profile + effective-skill validation for Kanban routing decisions.

One kernel for every caller-supplied profile name and every explicitly forced
task skill on the Kanban surface: ``hermes_cli.kanban_db.create_task``
(assignee skills / required reviewer), ``assign_task`` (reassignment that must
preserve skill capability) and ``request_review`` (review routing), plus the
``tools/kanban_tools.py`` handlers that front the same rules for agent-facing
errors.

Two rules the whole surface shares:

* A profile that does not exist is reported as ``profile '<name>' was not
  found`` with the available roster and ``kanban_discover`` guidance — never
  as ``not installed`` (profiles are not installable packages). Every such
  refusal ends with ``Nothing changed`` so a caller can tell a rejected call
  from a partial write.
* A skill is validated against the *assignee profile's* effective skill
  library — its own ``skills/`` tree, the shared root library, the bundled
  checkout skills, that profile's ``skills.external_dirs`` and its plugin
  skills — never against the caller's home. Missing skills name the profile
  and the skill; they never suggest installing the profile.

Nothing here writes: every entry point raises before the caller opens a write
transaction.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# The skill(s) the dispatcher force-loads for a review-lane worker. This is the
# capability a saved reviewer must already have at creation time:
# ``kanban_db_dispatch`` appends ``"sdlc-review"`` to ``claimed.skills`` for the
# review lane (see ``_spawn_one``'s review branch), and
# ``review_dispatch_enabled()`` documents Hermes as shipping it. Keep this tuple
# and that branch in sync — the review-skill gate is only as good as the
# dispatcher contract it mirrors.
REVIEW_SKILL = "sdlc-review"
REVIEW_SKILLS: tuple[str, ...] = (REVIEW_SKILL,)

_NOT_FOUND_HINT = (
    "Call kanban_discover for the roster of profiles this home can spawn."
)


class ProfileNotFoundError(ValueError):
    """Caller named a profile this home cannot spawn."""


class MissingSkillsError(ValueError):
    """Explicitly forced skills that do not resolve for the target profile."""

    def __init__(self, profile: str, missing: Iterable[str]):
        self.profile = profile
        self.missing = list(missing)
        super().__init__(
            f"skill(s) not found for profile {profile!r}: "
            f"{', '.join(self.missing)}. Nothing changed."
        )


# --- Profiles -----------------------------------------------------------------

def _profiles_kernel():
    """``(profile_exists, list_profile_names)`` from ``hermes_cli.profiles``.

    Imported lazily so a non-Kanban import never pays for it. A failure to
    import propagates: refusing to validate must never silently widen to
    "accept anything".
    """
    from hermes_cli.profiles import list_profile_names, profile_exists

    return profile_exists, list_profile_names


def profile_exists(name: str) -> bool:
    exists, _ = _profiles_kernel()
    return bool(exists(str(name)))


def available_profiles() -> list[str]:
    _, list_names = _profiles_kernel()
    try:
        return sorted(list_names())
    except Exception:
        return []


def profile_not_found_message(what: str, name: str) -> str:
    """The one user-facing wording for a profile that does not exist."""
    roster = ", ".join(available_profiles()) or "(none)"
    return (
        f"{what} profile {name!r} was not found. Available profiles: {roster}. "
        f"{_NOT_FOUND_HINT} Nothing changed."
    )


def require_profile(what: str, value, *, hint: str = "") -> str:
    """Strip ``value`` and require it to name a spawnable profile.

    Raises :class:`ProfileNotFoundError` — callers must surface it *before*
    opening a write transaction.
    """
    name = str(value).strip()
    if not name:
        raise ProfileNotFoundError(f"{what} must be a non-empty profile name.")
    if not profile_exists(name):
        message = profile_not_found_message(what, name)
        raise ProfileNotFoundError(f"{message} {hint}".strip() if hint else message)
    return name


# --- Effective skill library ---------------------------------------------------

def _profile_home(profile: str) -> Optional[Path]:
    """``profile``'s home dir, or ``None`` when it cannot be resolved.

    Resolution failures fall back to the *current* home rather than raising:
    a caller that only wants to check skill names must not explode because a
    profile directory is missing (the profile check itself owns that error).
    """
    try:
        from hermes_cli.profiles import resolve_profile_env

        home = resolve_profile_env(profile)
        return Path(home) if home else None
    except Exception:
        return None


def _config_skills_cfg(home: Optional[Path]) -> dict:
    if home is None:
        return {}
    for candidate in (home / "config.yaml", home / "config.yml"):
        try:
            if candidate.is_file():
                import yaml

                data = yaml.safe_load(candidate.read_text(encoding="utf-8-sig")) or {}
                if isinstance(data, dict):
                    cfg = data.get("skills")
                    return cfg if isinstance(cfg, dict) else {}
        except Exception:
            logger.debug("could not read skills config at %s", candidate, exc_info=True)
    return {}


def _expand_external(entry: str, home: Optional[Path]) -> Optional[Path]:
    raw = str(entry).strip()
    if not raw:
        return None
    raw = os.path.expandvars(os.path.expanduser(raw))
    path = Path(raw)
    if not path.is_absolute() and home is not None:
        path = home / path
    try:
        return path.resolve()
    except OSError:
        return path


def _external_dirs(home: Optional[Path]) -> list[Path]:
    cfg = _config_skills_cfg(home)
    out: list[Path] = []
    for entry in cfg.get("external_dirs") or []:
        if not isinstance(entry, str):
            continue
        path = _expand_external(entry, home)
        if path is not None and path.is_dir() and path not in out:
            out.append(path)
    return out


def _disabled_skill_names(home: Optional[Path]) -> set[str]:
    cfg = _config_skills_cfg(home)
    names: set[str] = set()
    disabled = cfg.get("disabled")
    if isinstance(disabled, list):
        names.update(str(n).strip() for n in disabled if str(n).strip())
    platform_disabled = cfg.get("platform_disabled")
    if isinstance(platform_disabled, dict):
        for values in platform_disabled.values():
            if isinstance(values, list):
                names.update(str(n).strip() for n in values if str(n).strip())
    return names


def _skill_dirs(profile: str) -> list[Path]:
    """Search roots for ``profile``, most authoritative first, de-duplicated."""
    home = _profile_home(profile)
    dirs: list[Path] = []
    if home is not None:
        dirs.append(home / "skills")
    try:
        from hermes_constants import get_default_hermes_root

        dirs.append(get_default_hermes_root() / "skills")
    except Exception:
        pass
    try:
        from hermes_constants import get_bundled_skills_dir

        dirs.append(get_bundled_skills_dir())
    except Exception:
        pass
    dirs.extend(_external_dirs(home))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in dirs:
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key in seen or not path.is_dir():
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _plugin_skill_names(profile: str) -> set[str]:
    """Plugin-contributed skill names, best effort (``plugin:skill`` and bare)."""
    names: set[str] = set()
    try:
        from hermes_cli.plugins import discover_plugins, get_plugin_manager

        discover_plugins()
        for skill in get_plugin_manager().list_plugin_skill_metadata():
            name = str(skill.get("name") or "").strip()
            if not name:
                continue
            names.add(name)
            if ":" in name:
                names.add(name.split(":", 1)[1])
    except Exception:
        logger.debug("plugin skill discovery failed for %s", profile, exc_info=True)
    return names


def _recorded_skill_names(skill_md: Path, root: Path) -> list[str]:
    """Names one ``SKILL.md`` resolves as: frontmatter/dir name plus the
    ``category/name`` form ``skill_view`` also accepts."""
    try:
        from agent.skill_utils import parse_frontmatter

        text = skill_md.read_text(encoding="utf-8", errors="replace")[:4000]
        frontmatter, _ = parse_frontmatter(text)
    except Exception:
        frontmatter = {}
    name = str((frontmatter or {}).get("name") or skill_md.parent.name).strip()
    if not name:
        return []
    recorded = [name]
    try:
        rel = skill_md.parent.relative_to(root)
    except ValueError:
        rel = None
    if rel is not None and len(rel.parts) >= 2:
        recorded.append("/".join(rel.parts[:-1]) + "/" + name)
    return recorded


def profile_skill_names(profile: str) -> set[str]:
    """The effective skill library a worker dispatched for ``profile`` can load.

    Profile skills, the shared root library, the checkout's bundled skills,
    that profile's ``skills.external_dirs`` and plugin skills; skills the
    profile's config disables are removed. Read-only.

    A library that resolves to nothing (a home whose skill tree was never
    seeded — ``seed_profile_skills`` does that for every real profile) falls
    back to the checkout's bundled tree, which is the very source
    ``seed_profile_skills`` copies from: with no per-profile evidence either
    way, the bundle is what that profile would carry. A profile that DOES have
    a skill tree is judged strictly against it, so a removed or disabled
    review skill is still rejected.
    """
    from agent.skill_utils import iter_skill_index_files

    roots = _skill_dirs(profile)
    names: set[str] = set()
    for root in roots:
        try:
            for skill_md in iter_skill_index_files(root, "SKILL.md"):
                names.update(_recorded_skill_names(skill_md, root))
        except Exception:
            logger.debug("skill scan failed for %s at %s", profile, root, exc_info=True)
    if not names:
        bundled = _checkout_bundled_skills_dir()
        if bundled is not None:
            try:
                for skill_md in iter_skill_index_files(bundled, "SKILL.md"):
                    names.update(_recorded_skill_names(skill_md, bundled))
            except Exception:
                logger.debug("bundled skill scan failed for %s", profile, exc_info=True)
    names.update(_plugin_skill_names(profile))
    names -= _disabled_skill_names(_profile_home(profile))
    return names


def _checkout_bundled_skills_dir() -> Optional[Path]:
    """This checkout's bundled ``skills/`` tree (``hermes_cli/../skills``)."""
    try:
        path = Path(__file__).resolve().parents[1] / "skills"
        return path if path.is_dir() else None
    except OSError:
        return None


def require_skills(profile: str, skills: Optional[Iterable[str]], *, what: str = "assignee") -> None:
    """Every explicitly forced skill must resolve in ``profile``'s library.

    The whole list is validated as one unit: a mixed valid/missing request is
    rejected atomically naming the profile and every missing skill, so a task
    never lands with half its specialist context.
    """
    wanted = [str(s).strip() for s in (skills or []) if str(s).strip()]
    if not wanted:
        return
    available = profile_skill_names(profile)
    missing = [name for name in wanted if name not in available]
    if missing:
        raise MissingSkillsError(profile, missing)


def require_reviewer(review_profile: str, *, what: str = "reviewer") -> str:
    """Reviewer profile that exists *and* carries the dispatcher's review skill."""
    name = require_profile(what, review_profile)
    require_skills(name, REVIEW_SKILLS, what=what)
    return name
