"""
Load session CSV rows and group them into sessions (ordered hops per session_id).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterator

SESSION_FIELDS = ("session_id", "persona", "sni", "timestamp", "hop")


def load_session_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a sessions-style CSV; returns flat list of row dicts."""
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(SESSION_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV missing columns {missing}; got {reader.fieldnames!r}")
        return [dict(row) for row in reader]


def group_rows_by_session(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """
    Group rows by session_id. Each value list is sorted by hop (then timestamp).
    """
    by_sid: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        sid = int(row["session_id"])
        by_sid.setdefault(sid, []).append(row)
    for sid in by_sid:
        by_sid[sid].sort(
            key=lambda r: (int(r["hop"]), int(float(r["timestamp"]))),
        )
    return by_sid


def iter_sessions(
    rows: list[dict[str, Any]],
) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    """Yield (session_id, rows_in_hop_order) sorted by session_id."""
    grouped = group_rows_by_session(rows)
    for sid in sorted(grouped.keys()):
        yield sid, grouped[sid]


def session_snis(session_rows: list[dict[str, Any]]) -> list[str]:
    """SNI hostname per hop, in hop order."""
    return [str(r["sni"]) for r in session_rows]
