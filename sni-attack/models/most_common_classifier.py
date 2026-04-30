"""
Baseline persona classifier: always predict the training-time mode persona
(one label per session in ``fit``; ``predict`` ignores SNIs).
"""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


class MostCommonClassifier:
    def __init__(self) -> None:
        self._guess: Optional[str] = None

    def fit(self, rows: Iterable[Mapping[str, Any]]) -> MostCommonClassifier:
        """
        Count each ``session_id`` once (persona from hop-ordered rows, same as
        session grouping in training CSV). The predicted persona is the mode over
        sessions; ties break by lexicographic order. ``sni`` is ignored.
        """
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
        persona_by_sid: dict[int, str] = {}
        for sid, sess in sorted(by_sid.items()):
            for r in sess:
                persona = str(r["persona"]).strip()
                if persona:
                    persona_by_sid[sid] = persona
                    break
        counts = Counter(persona_by_sid.values())
        if not counts:
            self._guess = None
        else:
            best = max(counts.values())
            self._guess = sorted(p for p, c in counts.items() if c == best)[0]
        return self

    def most_common_persona(self) -> Optional[str]:
        """The constant persona returned by ``predict`` (``None`` if untrained)."""
        return self._guess

    @staticmethod
    def _order_session_rows(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        """Same hop ordering as ``PopularClassifier``."""
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

    def predict(self, rows: Iterable[Mapping[str, Any]]) -> Optional[str]:
        """
        Same row input as ``fit`` (one session). ``persona`` and ``sni`` are
        ignored; returns the training mode persona, or ``None`` if empty input
        or no persona was learned.
        """
        ordered = self._order_session_rows(list(rows))
        if not ordered:
            return None
        return self._guess

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"guess": self._guess}, f)

    @classmethod
    def load(cls, path: str | Path) -> MostCommonClassifier:
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj._guess = data.get("guess")
        return obj
