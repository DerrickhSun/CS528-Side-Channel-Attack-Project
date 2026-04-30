from __future__ import annotations

from typing import Any, Iterable, Mapping


def order_session_rows(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Hop order when available; otherwise preserve incoming order."""
    if not rows:
        return rows
    if not any("hop" in r for r in rows):
        return list(rows)

    def sort_key(r: Mapping[str, Any]) -> tuple[int, float]:
        hop = int(r["hop"]) if "hop" in r else 0
        ts = r.get("timestamp", 0)
        try:
            tsf = float(ts)
        except (TypeError, ValueError):
            tsf = 0.0
        return hop, tsf

    return sorted(rows, key=sort_key)


def last_sni(rows: Iterable[Mapping[str, Any]]) -> str | None:
    """Return last non-empty SNI after applying row ordering."""
    ordered = order_session_rows(list(rows))
    if not ordered:
        return None
    last = str(ordered[-1].get("sni", "")).strip()
    return last or None


def sessions_ordered(rows: Iterable[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group flat training rows into sessions, hops sorted by hop then timestamp."""
    by_sid: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        sid = int(row["session_id"])
        by_sid.setdefault(sid, []).append(dict(row))
    for sid in by_sid:
        by_sid[sid].sort(
            key=lambda r: (
                int(r["hop"]),
                float(r["timestamp"]) if r.get("timestamp") not in (None, "") else 0.0,
            )
        )
    return [by_sid[k] for k in sorted(by_sid.keys())]
