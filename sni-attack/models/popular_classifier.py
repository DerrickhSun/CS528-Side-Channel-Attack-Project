"""
Frequency-based persona classifier: each SNI votes for the persona it was
most often paired with in training; a session is labeled by majority vote.
"""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from models.model_helper import order_session_rows


class PopularClassifier:
    def __init__(self) -> None:
        # sni -> persona -> co-occurrence count (from training rows)
        self._counts: dict[str, dict[str, int]] = {}

    def fit(self, rows: Iterable[Mapping[str, Any]]) -> PopularClassifier:
        """
        Learn co-occurrence counts from training rows.
        Each row should include ``sni`` and ``persona`` (other fields ignored).
        Rows may span many sessions; session boundaries are not used.
        """
        for row in rows:
            sni = str(row["sni"]).strip()
            persona = str(row["persona"]).strip()
            if not sni or not persona:
                continue
            if sni not in self._counts:
                self._counts[sni] = {}
            self._counts[sni][persona] = self._counts[sni].get(persona, 0) + 1
        return self

    def dominant_persona_for_sni(self, sni: str) -> Optional[str]:
        """Persona with the highest training count for this SNI; None if unseen."""
        sni = sni.strip()
        per = self._counts.get(sni)
        if not per:
            return None
        best = max(per.values())
        # Deterministic tie-break: lexicographically smallest persona name
        winners = sorted(p for p, c in per.items() if c == best)
        return winners[0]

    def _predict_from_snis(self, snis: Iterable[str]) -> Optional[str]:
        votes: Counter[str] = Counter()
        for sni in snis:
            p = self.dominant_persona_for_sni(str(sni))
            if p is not None:
                votes[p] += 1
        if not votes:
            return None
        top = votes.most_common(1)[0][1]
        tied = sorted(p for p, c in votes.items() if c == top)
        return tied[0]

    def predict(self, rows: Iterable[Mapping[str, Any]]) -> Optional[str]:
        """
        Classify one session: same row shape as for ``fit`` (at least ``sni``).
        ``persona`` is ignored if present. Rows are ordered by ``hop`` then
        ``timestamp`` when any row includes ``hop``; otherwise order is the
        iteration order of ``rows``.
        """
        as_list = list(rows)
        ordered = order_session_rows(as_list)
        snis = [str(r["sni"]).strip() for r in ordered if str(r.get("sni", "")).strip()]
        return self._predict_from_snis(snis)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"counts": self._counts}, f)

    @classmethod
    def load(cls, path: str | Path) -> PopularClassifier:
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj._counts = data.get("counts", {})
        return obj
