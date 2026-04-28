"""
Baseline next-site model: always predict the globally most frequent transition
target (the SNI that most often appears as ``next`` in hop-ordered sessions).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping

from models.first_order_markov import _sessions_ordered


class MostCommonPredictor:
    def __init__(self) -> None:
        self._next_site_counts: dict[str, int] = {}
        self._guess: str | None = None

    def fit(self, rows: Iterable[Mapping[str, Any]]) -> MostCommonPredictor:
        """
        Count how often each hostname appears as the **next** hop after any prior
        hop. ``persona`` is ignored. Same row schema as ``FirstOrderMarkov.fit``.
        """
        self._next_site_counts.clear()
        for session in _sessions_ordered(rows):
            for i in range(len(session) - 1):
                nxt = str(session[i + 1]["sni"]).strip()
                if nxt:
                    self._next_site_counts[nxt] = self._next_site_counts.get(nxt, 0) + 1
        self._guess = self._mode_next_site()
        return self

    def _mode_next_site(self) -> str | None:
        if not self._next_site_counts:
            return None
        best = max(self._next_site_counts.values())
        return sorted(s for s, c in self._next_site_counts.items() if c == best)[0]

    def most_common_next(self) -> str | None:
        """The single next-SNI guess used for every state (``None`` if untrained)."""
        return self._guess

    def next_target_counts(self) -> dict[str, int]:
        """Copy of per-target counts (how often each SNI appeared as **next**)."""
        return dict(self._next_site_counts)

    def next_probabilities(self, _current_sni: str) -> dict[str, float]:
        """Point mass on the global mode; empty if there is no trained guess."""
        if self._guess is None:
            return {}
        return {self._guess: 1.0}

    def next_probabilities_list(self, _current_sni: str) -> list[tuple[str, float]]:
        if self._guess is None:
            return []
        return [(self._guess, 1.0)]

    @staticmethod
    def _order_session_rows(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
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

    def predict(self, rows: Iterable[Mapping[str, Any]]) -> list[tuple[str, float]]:
        """
        Same session-row input as ``FirstOrderMarkov.predict``; the current SNI
        is ignored. Returns ``[(guess, 1.0)]`` when trained with at least one
        transition, else ``[]``.
        """
        ordered = self._order_session_rows(list(rows))
        if not ordered or self._guess is None:
            return []
        return [(self._guess, 1.0)]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"next_site_counts": self._next_site_counts}, f)

    @classmethod
    def load(cls, path: str | Path) -> MostCommonPredictor:
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj._next_site_counts = dict(data.get("next_site_counts", {}))
        obj._guess = obj._mode_next_site()
        return obj
