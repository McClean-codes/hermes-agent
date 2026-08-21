"""Shared utilities for the workflow plugin.

Functions that are used by multiple modules within the workflow plugin
live here to avoid duplication.
"""

import sys
from pathlib import Path


def hermes_binary() -> str:
    """Resolve the ``hermes`` CLI binary from the venv.

    The engine spawns ``hermes kanban create/show`` subprocesses to
    interact with the fleet-wide kanban board.  When called from inside a
    Hermes agent process (e.g. via MCP tools), ``hermes`` is not on PATH
    — it only lives in the venv's ``bin/`` directory alongside
    ``sys.executable``.

    Resolution order:
      1. ``sys.executable``'s parent — works when invoked via the venv's
         own python (agent-in-process invocation).
      2. ``sys.prefix/bin/hermes`` — works when invoked via ``python3 -m``
         with a different ``sys.executable`` (CLI invocation outside the
         venv); ``sys.prefix`` always points to the venv root.
      3. Project-level ``.venv/bin/hermes`` — fallback for dev setups.
      4. Bare ``"hermes"`` — last resort for environments with PATH.

    Returns the absolute path to the ``hermes`` binary, which is always
    the correct target for ``subprocess.run`` regardless of how the
    engine is invoked (CLI or in-process).
    """
    candidate = Path(sys.executable).parent / "hermes"
    if candidate.is_file():
        return str(candidate)
    venv_candidate = Path(sys.prefix) / "bin" / "hermes"
    if venv_candidate.is_file():
        return str(venv_candidate)
    project_venv = (
        Path(__file__).resolve().parent.parent.parent / ".venv" / "bin" / "hermes"
    )
    if project_venv.is_file():
        return str(project_venv)
    return "hermes"
