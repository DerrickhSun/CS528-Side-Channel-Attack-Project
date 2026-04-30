"""
First-order Markov chain over SNI transitions: counts (from_sni -> to_sni),
then next-site probabilities conditioned on the current (last) SNI.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping

from models.model_helper import last_sni, sessions_ordered


class FirstOrderMarkov:
    def __init__(self) -> None:
        # from_sni -> to_sni -> transition count (within sessions, hop order)
        self._counts: dict[str, dict[str, int]] = {}

    def fit(self, rows: Iterable[Mapping[str, Any]]) -> FirstOrderMarkov:
        """
        Count consecutive (current_sni -> next_sni) pairs within each session.
        Same row schema as ``PopularClassifier.fit`` (uses ``session_id``, ``hop``,
        ``timestamp``, ``sni``; ``persona`` ignored).
        """
        self._counts.clear()
        for session in sessions_ordered(rows):
            for i in range(len(session) - 1):
                a = str(session[i]["sni"]).strip()
                b = str(session[i + 1]["sni"]).strip()
                if not a or not b:
                    continue
                if a not in self._counts:
                    self._counts[a] = {}
                self._counts[a][b] = self._counts[a].get(b, 0) + 1
        return self

    def next_probabilities(self, current_sni: str) -> dict[str, float]:
        """
        MLE distribution P(next | current): empty dict if ``current_sni`` never
        appeared as a source state in training.
        """
        cur = current_sni.strip()
        nxt = self._counts.get(cur)
        if not nxt:
            return {}
        total = sum(nxt.values())
        return {s: c / total for s, c in sorted(nxt.items())}

    def next_probabilities_list(self, current_sni: str) -> list[tuple[str, float]]:
        """
        Same as ``next_probabilities``, as a list sorted by descending probability
        then by SNI name (deterministic tie-break).
        """
        d = self.next_probabilities(current_sni)
        return sorted(d.items(), key=lambda x: (-x[1], x[0]))

    def predict(self, rows: Iterable[Mapping[str, Any]]) -> list[tuple[str, float]]:
        """
        Condition on the **last** SNI of the ordered session rows; return the
        distribution over next SNIs as ``(next_sni, probability)`` pairs, sorted
        by descending probability. Empty list if the current state is unknown.
        """
        cur = last_sni(rows)
        if cur is None:
            return []
        return self.next_probabilities_list(cur)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"counts": self._counts}, f)

    @classmethod
    def load(cls, path: str | Path) -> FirstOrderMarkov:
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj._counts = data.get("counts", {})
        return obj
