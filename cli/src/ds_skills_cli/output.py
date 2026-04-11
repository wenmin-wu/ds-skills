"""Structured output helpers. JSON → stdout, human → stderr."""

from __future__ import annotations

import json
import sys


def log(msg: str) -> None:
    """Human-readable message to stderr (visible in both modes)."""
    print(msg, file=sys.stderr)


def emit_json(data: object) -> None:
    """Machine-readable JSON to stdout."""
    print(json.dumps(data, indent=2, ensure_ascii=False))


def emit_table(rows: list[dict], columns: list[str], widths: list[int]) -> None:
    """Print a simple aligned table to stdout."""
    header = "  ".join(c.upper().ljust(w) for c, w in zip(columns, widths))
    print(header)
    print("-" * len(header))
    for row in rows:
        line = "  ".join(str(row.get(c, "")).ljust(w) for c, w in zip(columns, widths))
        print(line)
