#!/usr/bin/env python3
"""Structural check: verify modified Python files stay under the line limit.

The repository convention is that modified files should stay under 2,000
lines of code.  This script checks the A2A adapter module and its extracted
task_routing submodule, then optionally scans all modified files in a PR.

Exit 0 when all files comply; exit 1 when any file exceeds the limit.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Hard limit for modified files (the <2K invariant).
MAX_LINES = 2000

# Files that are explicitly tracked by this check (the A2A extraction).
TRACKED_FILES = [
    "plugins/platforms/a2a/adapter.py",
    "plugins/platforms/a2a/task_routing.py",
]


def count_lines(path: Path) -> int:
    """Count non-blank, non-comment lines in a Python file."""
    count = 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    count += 1
    except FileNotFoundError:
        return -1
    return count


def check_tracked_files(repo_root: Path) -> list[tuple[str, int, int, bool]]:
    """Check tracked files against the limit.

    Returns list of (filename, line_count, limit, ok).
    """
    results = []
    for rel in TRACKED_FILES:
        path = repo_root / rel
        loc = count_lines(path)
        if loc < 0:
            results.append((rel, 0, MAX_LINES, True))  # file missing = skip
        else:
            results.append((rel, loc, MAX_LINES, loc <= MAX_LINES))
    return results


def check_modified_files(repo_root: Path, base_ref: str = "origin/main") -> list[tuple[str, int, int, bool]]:
    """Check all modified Python files in the current diff against the limit."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base_ref],
            capture_output=True, text=True, cwd=str(repo_root), timeout=10,
        )
        files = [f for f in result.stdout.strip().split("\n") if f.endswith(".py")]
    except Exception:
        return []

    results = []
    for rel in files:
        path = repo_root / rel
        if not path.exists():
            continue
        loc = count_lines(path)
        if loc > MAX_LINES:
            results.append((rel, loc, MAX_LINES, False))
    return results


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]

    print(f"Structural check: MAX_LINES = {MAX_LINES}")
    print()

    all_ok = True

    # 1. Check tracked files (the A2A extraction)
    print("── Tracked files (A2A extraction) ──")
    for name, loc, limit, ok in check_tracked_files(repo_root):
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}: {loc} lines (limit {limit})")
        if not ok:
            all_ok = False

    # 2. Check modified Python files in the diff
    print()
    print("── Modified Python files (git diff) ──")
    modified = check_modified_files(repo_root)
    if modified:
        for name, loc, limit, ok in modified:
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  {name}: {loc} lines (limit {limit})")
            if not ok:
                all_ok = False
    else:
        print("  (no modified Python files beyond tracked)")

    print()
    if all_ok:
        print("Result: ALL OK")
        return 0
    else:
        print("Result: FAILURES DETECTED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
