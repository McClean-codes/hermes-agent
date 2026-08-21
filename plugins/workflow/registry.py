"""Workflow template registry — auto-discovery for dynamic workflows.

Scans two locations for workflow definitions:
  1. ``docs/fleet-pipelines/`` — pre-defined pipeline YAMLs (ships with repo)
  2. ``~/.hermes/workflows/``   — user-saved workflow templates (dynamic mode)

Provides:
  * ``list_workflows()``  — enumerate all available workflows with metadata
  * ``match_workflow_trigger(user_message)`` — keyword-based trigger matching

The registry is the backbone of the "workflows as skills" pattern: workflows
become discoverable and trigger-based, just like skills.  When a user message
matches a workflow trigger, the agent should offer to run that workflow.

No state is cached — callers get a fresh scan on every ``list_workflows()``
so that new YAML files are picked up immediately.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-defined trigger keywords per pipeline name.
#
# These are derived from the SKILL.md Intent → Pipeline Mapping table.
# The registry uses them when the YAML file itself doesn't declare a trigger.
# Format: pipe-separated keywords, case-insensitive substring match.
# ---------------------------------------------------------------------------

_DEFAULT_TRIGGERS: Dict[str, str] = {
    "council":              "council|debate|trade-off|competing priorities|adversarial|perspective",
    "ideation":             "ideation|spec|research|decompose|brainstorm pipeline|architecture",
    "brainstorm":           "brainstorm|collaborative|ideation session|collective thinking|group ideation",
    "feature-dev":          "feature|build|develop|implement|coding|ci|review|merge|pull request|pr",
    "deployment-verify":    "verify deploy|post-deploy|deployment check|deploy probe|smoke test|deploy verify",
    "deployment-revert":    "revert|rollback|undo deploy|deploy failure|auto-rollback|revert deploy|deploy revert",
    "error-response":       "sentry|error alert|incident|triage|error response|fatal error|bug|crash|exception",
    "new-agent-onboarding": "onboard|new agent|commission agent|agent setup|add agent|create agent",
    "report-back":          "report back|deliver results|summary report|wrap up|deliver summary",
}

# ---------------------------------------------------------------------------
# Default description templates (generated when YAML lacks a description)
# ---------------------------------------------------------------------------

_DESCRIPTIONS: Dict[str, str] = {
    "council":              "Use when you need a structured multi-agent debate with adversarial perspectives on trade-offs.",
    "ideation":             "Use when you need research → spec → security → domain decomposition for a project.",
    "brainstorm":           "Use when you need collaborative multi-agent ideation with cooperative framing.",
    "feature-dev":          "Use when you need to build, test, review, merge, and post-merge verification for a feature.",
    "deployment-verify":    "Use when you need a post-deploy adversarial probe to verify a deployment.",
    "deployment-revert":    "Use when you need an auto-rollback on deploy failure.",
    "error-response":       "Use when you need Sentry alert triage and multi-agent dispatch.",
    "new-agent-onboarding": "Use when you need to commission a new agent with the 7-phase onboarding DAG.",
    "report-back":          "Use when you need to wrap a pipeline and deliver results to the origin channel.",
}


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _fleet_pipelines_dirs() -> list[Path]:
    """Resolve all ``docs/fleet-pipelines/`` directories.

    Uses ``HERMES_WORKFLOW_FILES`` env var if set, then falls back to
    scanning profile workspaces for docs repos.
    """
    dirs: list[Path] = []

    # Primary: HERMES_WORKFLOW_FILES env var
    env_path = os.environ.get("HERMES_WORKFLOW_FILES", "")
    if env_path:
        p = Path(env_path).expanduser()
        if p.is_dir():
            dirs.append(p)

    # Fallback: scan profile workspaces for docs repos
    hermes_home = Path(os.environ.get("HERMES_HOME", "")).expanduser()
    if not hermes_home or not hermes_home.is_dir():
        hermes_home = Path.home() / ".hermes"

    profiles_dir = hermes_home / "profiles"
    if profiles_dir.is_dir():
        for profile in profiles_dir.iterdir():
            if not profile.is_dir():
                continue
            for candidate in [
                profile / "workspace" / "docs" / "fleet-pipelines",
                profile / "workspace" / "projects" / "docs" / "fleet-pipelines",
            ]:
                if candidate.is_dir() and candidate not in dirs:
                    dirs.append(candidate)

    return dirs


def _user_workflows_dir() -> Optional[Path]:
    """Resolve ``~/.hermes/workflows/`` for user-saved dynamic templates."""
    hermes_home = Path(os.environ.get("HERMES_HOME", "")).expanduser()
    if not hermes_home or not hermes_home.is_dir():
        hermes_home = Path.home() / ".hermes"
    d = hermes_home / "workflows"
    return d if d.is_dir() else None


# ---------------------------------------------------------------------------
# YAML metadata extraction
# ---------------------------------------------------------------------------

def _extract_yaml_metadata(path: Path) -> Dict[str, Any]:
    """Extract metadata from a workflow YAML file.

    Reads the full file and parses it with PyYAML if available.
    Falls back to comment-based extraction if PyYAML is missing.
    Returns a dict with keys: name, description, trigger, category.
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None  # type: ignore

    text = path.read_text(errors="replace")
    stem = path.stem  # filename without .yaml

    name = stem
    description = ""
    trigger = ""

    if yaml is not None:
        try:
            doc = yaml.safe_load(text)
            if isinstance(doc, dict):
                name = doc.get("name", stem)
                description = doc.get("description", "")
                trigger = doc.get("trigger", "")
        except Exception:
            pass

    # Fallback: extract description from the first comment line if YAML parse
    # didn't yield one.
    if not description:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# ") and len(stripped) > 2:
                description = stripped[2:]
                break

    return {
        "name": name,
        "description": description,
        "trigger": trigger,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_workflows() -> List[Dict[str, Any]]:
    """Return a list of available workflows with metadata.

    Each entry is a dict with keys:
      name, description, trigger, mode, category, path
    """
    workflows: List[Dict[str, Any]] = []

    # --- Pre-defined pipelines from docs/fleet-pipelines/ ---
    for fp_dir in _fleet_pipelines_dirs():
        for yml in sorted(fp_dir.glob("*.yaml")):
            meta = _extract_yaml_metadata(yml)
            stem = yml.stem

            # Merge default trigger / description if YAML didn't provide them
            trigger = meta["trigger"] or _DEFAULT_TRIGGERS.get(stem, "")
            description = meta["description"] or _DESCRIPTIONS.get(
                stem, f"Pre-defined fleet pipeline: {stem}"
            )

            workflows.append({
                "name": meta["name"],
                "description": description,
                "trigger": trigger,
                "mode": "predefined",
                "category": "fleet",
                "path": str(yml),
            })

    # --- User-saved dynamic workflow templates ---
    uw_dir = _user_workflows_dir()
    if uw_dir is not None:
        for yml in sorted(uw_dir.glob("*.yaml")):
            meta = _extract_yaml_metadata(yml)

            workflows.append({
                "name": meta["name"],
                "description": meta["description"],
                "trigger": meta["trigger"],
                "mode": "dynamic",
                "category": "dynamic",
                "path": str(yml),
            })

    return workflows


def match_workflow_trigger(user_message: str) -> Optional[Dict[str, Any]]:
    """Check a user message against all registered workflow triggers.

    Matching is keyword-based: the trigger string is split on ``|`` and each
    keyword is checked as a case-insensitive substring of the user message.

    Returns the best match (the workflow whose trigger keywords produce the
    most hits), or ``None`` if no workflow matches.
    """
    if not user_message:
        return None

    msg_lower = user_message.lower()
    all_workflows = list_workflows()

    best: Optional[Dict[str, Any]] = None
    best_score = 0

    for wf in all_workflows:
        trigger = wf.get("trigger", "")
        if not trigger:
            continue

        keywords = [kw.strip() for kw in trigger.split("|") if kw.strip()]
        if not keywords:
            continue

        score = sum(1 for kw in keywords if kw.lower() in msg_lower)
        if score > best_score:
            best_score = score
            best = wf

    return best
