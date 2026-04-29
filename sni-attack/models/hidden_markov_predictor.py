"""
Hidden-Markov-style next-site predictor.

Hidden state: persona.
Observation: current SNI.

Trains:
- start probability P(persona at session start)
- persona transition P(persona_t+1 | persona_t)
- emission P(sni | persona)

Predicts next-site distribution from observed session prefix by:
1) filtering to infer current persona posterior from observed SNIs,
2) one-step persona transition to next-step persona posterior,
3) mixing persona emissions to get P(next_sni | observed prefix).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from models.first_order_markov import _sessions_ordered


class HiddenMarkovPredictor:
    def __init__(
        self,
        alpha: float = 1.0,
        prediction_target: Literal["next_site", "persona"] = "next_site",
    ) -> None:
        if prediction_target not in {"next_site", "persona"}:
            raise ValueError("prediction_target must be 'next_site' or 'persona'")
        self.alpha = float(alpha)
        self.prediction_target = prediction_target
        self._personas: list[str] = []
        self._snis: list[str] = []
        self._start_prob: dict[str, float] = {}
        self._trans_prob: dict[str, dict[str, float]] = {}
        self._emit_prob: dict[str, dict[str, float]] = {}
        self._emit_default: dict[str, float] = {}

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

    def fit(self, rows: Iterable[Mapping[str, Any]]) -> HiddenMarkovPredictor:
        sessions = _sessions_ordered(rows)
        personas = sorted(
            {
                str(r.get("persona", "")).strip()
                for sess in sessions
                for r in sess
                if str(r.get("persona", "")).strip()
            }
        )
        snis = sorted(
            {
                str(r.get("sni", "")).strip()
                for sess in sessions
                for r in sess
                if str(r.get("sni", "")).strip()
            }
        )
        self._personas = personas
        self._snis = snis
        if not personas:
            self._start_prob = {}
            self._trans_prob = {}
            self._emit_prob = {}
            self._emit_default = {}
            return self

        start_counts = {p: 0 for p in personas}
        trans_counts = {p: {q: 0 for q in personas} for p in personas}
        emit_counts = {p: {s: 0 for s in snis} for p in personas}
        emit_totals = {p: 0 for p in personas}

        for sess in sessions:
            if not sess:
                continue
            first_p = str(sess[0].get("persona", "")).strip()
            if first_p in start_counts:
                start_counts[first_p] += 1
            for i, r in enumerate(sess):
                p = str(r.get("persona", "")).strip()
                s = str(r.get("sni", "")).strip()
                if p in emit_counts and s in emit_counts[p]:
                    emit_counts[p][s] += 1
                    emit_totals[p] += 1
                if i + 1 < len(sess):
                    pn = str(sess[i + 1].get("persona", "")).strip()
                    if p in trans_counts and pn in trans_counts[p]:
                        trans_counts[p][pn] += 1

        k = len(personas)
        v = len(snis)
        alpha = self.alpha

        total_start = sum(start_counts.values()) + alpha * k
        self._start_prob = {
            p: (start_counts[p] + alpha) / total_start for p in personas
        }

        self._trans_prob = {}
        for p in personas:
            row_total = sum(trans_counts[p].values()) + alpha * k
            self._trans_prob[p] = {
                q: (trans_counts[p][q] + alpha) / row_total for q in personas
            }

        self._emit_prob = {}
        self._emit_default = {}
        for p in personas:
            denom = emit_totals[p] + alpha * (v + 1)  # +1 unknown-token bucket
            self._emit_prob[p] = {s: (emit_counts[p][s] + alpha) / denom for s in snis}
            self._emit_default[p] = alpha / denom
        return self

    def _emission(self, persona: str, sni: str) -> float:
        row = self._emit_prob.get(persona)
        if not row:
            return 0.0
        return row.get(sni, self._emit_default.get(persona, 0.0))

    def _normalize(self, dist: dict[str, float]) -> dict[str, float]:
        z = sum(dist.values())
        if z <= 0:
            if not self._personas:
                return {}
            u = 1.0 / len(self._personas)
            return {p: u for p in self._personas}
        return {k: v / z for k, v in dist.items()}

    def _persona_posterior(self, observed_snis: list[str]) -> dict[str, float]:
        if not self._personas:
            return {}
        if not observed_snis:
            return dict(self._start_prob)

        belief = {
            p: self._start_prob.get(p, 0.0) * self._emission(p, observed_snis[0])
            for p in self._personas
        }
        belief = self._normalize(belief)

        for obs in observed_snis[1:]:
            nxt: dict[str, float] = {}
            for q in self._personas:
                prior_q = sum(
                    belief[p] * self._trans_prob.get(p, {}).get(q, 0.0)
                    for p in self._personas
                )
                nxt[q] = prior_q * self._emission(q, obs)
            belief = self._normalize(nxt)
        return belief

    def _next_persona_distribution(self, current_posterior: dict[str, float]) -> dict[str, float]:
        if not self._personas:
            return {}
        nxt = {
            q: sum(
                current_posterior.get(p, 0.0) * self._trans_prob.get(p, {}).get(q, 0.0)
                for p in self._personas
            )
            for q in self._personas
        }
        return self._normalize(nxt)

    def _predict_persona_distribution(
        self, rows: Iterable[Mapping[str, Any]]
    ) -> list[tuple[str, float]]:
        ordered = self._order_session_rows(list(rows))
        observed = [
            str(r.get("sni", "")).strip() for r in ordered if str(r.get("sni", "")).strip()
        ]
        if not self._personas:
            return []

        if observed:
            current_post = self._persona_posterior(observed)
            next_persona = self._next_persona_distribution(current_post)
        else:
            next_persona = dict(self._start_prob)
        next_persona = self._normalize(next_persona)
        return sorted(next_persona.items(), key=lambda x: (-x[1], x[0]))

    def predict_persona(self, rows: Iterable[Mapping[str, Any]]) -> str | None:
        ranked = self._predict_persona_distribution(rows)
        return ranked[0][0] if ranked else None

    def _predict_next_site_distribution(
        self, rows: Iterable[Mapping[str, Any]]
    ) -> list[tuple[str, float]]:
        ordered = self._order_session_rows(list(rows))
        observed = [str(r.get("sni", "")).strip() for r in ordered if str(r.get("sni", "")).strip()]
        if not observed or not self._personas or not self._snis:
            return []

        current_post = self._persona_posterior(observed)
        next_persona = self._next_persona_distribution(current_post)
        next_sni: dict[str, float] = {s: 0.0 for s in self._snis}
        for p, pp in next_persona.items():
            emits = self._emit_prob.get(p, {})
            for s in self._snis:
                next_sni[s] += pp * emits.get(s, 0.0)

        z = sum(next_sni.values())
        if z <= 0:
            return []
        dist = {s: p / z for s, p in next_sni.items()}
        return sorted(dist.items(), key=lambda x: (-x[1], x[0]))

    def predict(self, rows: Iterable[Mapping[str, Any]]) -> list[tuple[str, float]]:
        if self.prediction_target == "persona":
            return self._predict_persona_distribution(rows)
        return self._predict_next_site_distribution(rows)

    def next_probabilities(self, _current_sni: str) -> dict[str, float]:
        """
        Stateless call isn't enough for HMM context. Kept for compatibility.
        Returns empty dict so callers prefer ``predict(session_prefix_rows)``.
        """
        return {}

    def next_probabilities_list(self, _current_sni: str) -> list[tuple[str, float]]:
        return []

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "alpha": self.alpha,
                    "prediction_target": self.prediction_target,
                    "personas": self._personas,
                    "snis": self._snis,
                    "start_prob": self._start_prob,
                    "trans_prob": self._trans_prob,
                    "emit_prob": self._emit_prob,
                    "emit_default": self._emit_default,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> HiddenMarkovPredictor:
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            alpha=float(data.get("alpha", 1.0)),
            prediction_target=str(data.get("prediction_target", "next_site")),
        )
        obj._personas = list(data.get("personas", []))
        obj._snis = list(data.get("snis", []))
        obj._start_prob = dict(data.get("start_prob", {}))
        obj._trans_prob = dict(data.get("trans_prob", {}))
        obj._emit_prob = dict(data.get("emit_prob", {}))
        obj._emit_default = dict(data.get("emit_default", {}))
        return obj
