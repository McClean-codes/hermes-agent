"""Shared LLM utility functions for the Hermes CLI.

These are small, pure functions used across auxiliary modules
(kanban specifier, kanban decomposer, profile describer, workflow analyst).
Centralising them here prevents copy-paste divergence.
"""

from __future__ import annotations

import json
import re
from typing import Optional

# Canonical fence pattern — strips `````json`/````` fences and
# leading/trailing whitespace.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def extract_json_blob(raw: str) -> Optional[dict]:
    """Lenient JSON extraction from an LLM response.

    Tolerates fenced code blocks (`````json ... ````), prose preambles,
    and extra whitespace.  Returns the parsed ``dict``, or ``None`` if no
    valid JSON object could be extracted.

    This is the canonical implementation shared by auxiliary modules.
    """
    if not raw:
        return None
    stripped = _FENCE_RE.sub("", raw.strip())
    # Greedy: find the first ``{`` and last ``}`` and try that slice.
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return None
    candidate = stripped[first : last + 1]
    try:
        val = json.loads(candidate)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(val, dict):
        return None
    return val
