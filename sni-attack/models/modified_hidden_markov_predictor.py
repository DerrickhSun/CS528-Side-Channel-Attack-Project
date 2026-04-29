"""
Modified HMM-style next-site predictor with observation-dependent dynamics.

State: persona.
Observation: visited SNI.

Learns:
- P(start persona)
- P(next persona | current persona, current site)
  with fallback P(next persona | current site) when (persona, site) is unseen
- P(next site | next persona, current site)
  with fallback P(next site | current site) when (next persona, site) is unseen

When a site-level marginal is unavailable (site never seen), falls back to the
older persona-only distributions.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from models.first_order_markov import _sessions_ordered


class ModifiedHiddenMarkovPredictor:
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

        # P(next persona | persona, current site)
        self._trans_site_prob: dict[str, dict[str, dict[str, float]]] = {}
        # P(next persona | persona) — last-resort if site unknown globally
        self._trans_persona_prob: dict[str, dict[str, float]] = {}
        # P(next persona | current site) — fallback when (persona, site) unseen
        self._trans_site_marginal_prob: dict[str, dict[str, float]] = {}

        # P(next site | next persona, current site)
        self._emit_site_prob: dict[str, dict[str, dict[str, float]]] = {}
        # alpha / denom for each (persona, current_site) emission row
        self._emit_site_row_default: dict[str, dict[str, float]] = {}
        # P(next site | next persona) — last-resort
        self._emit_persona_prob: dict[str, dict[str, float]] = {}
        self._emit_persona_default: dict[str, float] = {}
        # P(next site | current site) — fallback when (persona, site) unseen
        self._emit_site_marginal_prob: dict[str, dict[str, float]] = {}
        self._emit_site_marginal_default: dict[str, float] = {}

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

    def fit(self, rows: Iterable[Mapping[str, Any]]) -> ModifiedHiddenMarkovPredictor:
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
            self._trans_site_prob = {}
            self._trans_persona_prob = {}
            self._trans_site_marginal_prob = {}
            self._emit_site_prob = {}
            self._emit_site_row_default = {}
            self._emit_persona_prob = {}
            self._emit_persona_default = {}
            self._emit_site_marginal_prob = {}
            self._emit_site_marginal_default = {}
            return self

        alpha = self.alpha
        k = len(personas)
        v = len(snis)

        start_counts = {p: 0 for p in personas}

        trans_persona_counts = {p: {q: 0 for q in personas} for p in personas}
        trans_site_counts: dict[str, dict[str, dict[str, int]]] = {
            p: {} for p in personas
        }
        trans_marginal_site_counts: dict[str, dict[str, int]] = {}

        emit_persona_counts = {p: {s: 0 for s in snis} for p in personas}
        emit_persona_totals = {p: 0 for p in personas}
        emit_site_counts: dict[str, dict[str, dict[str, int]]] = {
            p: {} for p in personas
        }
        emit_marginal_site_counts: dict[str, dict[str, int]] = {}

        for sess in sessions:
            if not sess:
                continue
            first_p = str(sess[0].get("persona", "")).strip()
            if first_p in start_counts:
                start_counts[first_p] += 1

            for i in range(len(sess) - 1):
                cur_p = str(sess[i].get("persona", "")).strip()
                nxt_p = str(sess[i + 1].get("persona", "")).strip()
                cur_s = str(sess[i].get("sni", "")).strip()
                nxt_s = str(sess[i + 1].get("sni", "")).strip()
                if (
                    cur_p not in trans_persona_counts
                    or nxt_p not in trans_persona_counts[cur_p]
                    or not cur_s
                    or not nxt_s
                ):
                    continue

                trans_persona_counts[cur_p][nxt_p] += 1
                trans_site_counts[cur_p].setdefault(
                    cur_s, {q: 0 for q in personas}
                )[nxt_p] += 1
                trans_marginal_site_counts.setdefault(cur_s, {q: 0 for q in personas})[
                    nxt_p
                ] += 1

                if nxt_s in emit_persona_counts[nxt_p]:
                    emit_persona_counts[nxt_p][nxt_s] += 1
                    emit_persona_totals[nxt_p] += 1
                    emit_site_counts[nxt_p].setdefault(
                        cur_s, {s: 0 for s in snis}
                    )[nxt_s] += 1
                    emit_marginal_site_counts.setdefault(cur_s, {s: 0 for s in snis})[
                        nxt_s
                    ] += 1

        total_start = sum(start_counts.values()) + alpha * k
        self._start_prob = {
            p: (start_counts[p] + alpha) / total_start for p in personas
        }

        self._trans_persona_prob = {}
        for p in personas:
            denom = sum(trans_persona_counts[p].values()) + alpha * k
            self._trans_persona_prob[p] = {
                q: (trans_persona_counts[p][q] + alpha) / denom for q in personas
            }

        self._trans_site_prob = {}
        for p in personas:
            self._trans_site_prob[p] = {}
            for cur_s, row in trans_site_counts[p].items():
                denom = sum(row.values()) + alpha * k
                self._trans_site_prob[p][cur_s] = {
                    q: (row[q] + alpha) / denom for q in personas
                }

        self._trans_site_marginal_prob = {}
        for cur_s, row in trans_marginal_site_counts.items():
            denom = sum(row.values()) + alpha * k
            self._trans_site_marginal_prob[cur_s] = {
                q: (row[q] + alpha) / denom for q in personas
            }

        self._emit_persona_prob = {}
        self._emit_persona_default = {}
        for p in personas:
            denom = emit_persona_totals[p] + alpha * (v + 1)
            self._emit_persona_prob[p] = {
                s: (emit_persona_counts[p][s] + alpha) / denom for s in snis
            }
            self._emit_persona_default[p] = alpha / denom

        self._emit_site_prob = {}
        self._emit_site_row_default = {}
        for p in personas:
            self._emit_site_prob[p] = {}
            self._emit_site_row_default[p] = {}
            for cur_s, row in emit_site_counts[p].items():
                denom = sum(row.values()) + alpha * (v + 1)
                self._emit_site_prob[p][cur_s] = {
                    s: (row[s] + alpha) / denom for s in snis
                }
                self._emit_site_row_default[p][cur_s] = alpha / denom

        self._emit_site_marginal_prob = {}
        self._emit_site_marginal_default = {}
        for cur_s, row in emit_marginal_site_counts.items():
            denom = sum(row.values()) + alpha * (v + 1)
            self._emit_site_marginal_prob[cur_s] = {
                s: (row[s] + alpha) / denom for s in snis
            }
            self._emit_site_marginal_default[cur_s] = alpha / denom

        return self

    def _normalize(self, dist: dict[str, float]) -> dict[str, float]:
        z = sum(dist.values())
        if z <= 0:
            if not dist:
                return {}
            u = 1.0 / len(dist)
            return {k: u for k in dist}
        return {k: v / z for k, v in dist.items()}

    def _uniform_personas(self) -> dict[str, float]:
        if not self._personas:
            return {}
        u = 1.0 / len(self._personas)
        return {p: u for p in self._personas}

    def _trans_row(self, persona: str, current_site: str) -> dict[str, float]:
        row = self._trans_site_prob.get(persona, {}).get(current_site)
        if row is not None:
            return row
        marginal = self._trans_site_marginal_prob.get(current_site)
        if marginal:
            return marginal
        return self._trans_persona_prob.get(persona, self._uniform_personas())

    def _emit_row(self, persona: str, current_site: str) -> dict[str, float]:
        row = self._emit_site_prob.get(persona, {}).get(current_site)
        if row is not None:
            return row
        marginal = self._emit_site_marginal_prob.get(current_site)
        if marginal:
            return marginal
        return self._emit_persona_prob.get(persona, {})

    def _emit_default_for(self, persona: str, current_site: str) -> float:
        if self._emit_site_prob.get(persona, {}).get(current_site) is not None:
            return self._emit_site_row_default.get(persona, {}).get(
                current_site, self._emit_persona_default.get(persona, 0.0)
            )
        if current_site in self._emit_site_marginal_prob:
            return self._emit_site_marginal_default.get(
                current_site, self._emit_persona_default.get(persona, 0.0)
            )
        return self._emit_persona_default.get(persona, 0.0)

    def _persona_posterior(self, observed: list[str]) -> dict[str, float]:
        if not self._personas:
            return {}
        if not observed:
            return dict(self._start_prob)

        first = observed[0]
        belief = {
            p: self._start_prob.get(p, 0.0)
            * self._emit_persona_prob.get(p, {}).get(
                first, self._emit_persona_default.get(p, 0.0)
            )
            for p in self._personas
        }
        belief = self._normalize(belief)

        for t in range(len(observed) - 1):
            cur_s = observed[t]
            nxt_s = observed[t + 1]
            nxt_belief: dict[str, float] = {}
            for q in self._personas:
                prior_q = 0.0
                for p in self._personas:
                    prior_q += belief.get(p, 0.0) * self._trans_row(p, cur_s).get(q, 0.0)
                emit_q = self._emit_row(q, cur_s).get(
                    nxt_s, self._emit_default_for(q, cur_s)
                )
                nxt_belief[q] = prior_q * emit_q
            belief = self._normalize(nxt_belief)
        return belief

    def _predict_persona_distribution(
        self, rows: Iterable[Mapping[str, Any]]
    ) -> list[tuple[str, float]]:
        ordered = self._order_session_rows(list(rows))
        observed = [
            str(r.get("sni", "")).strip()
            for r in ordered
            if str(r.get("sni", "")).strip()
        ]
        if not self._personas:
            return []

        if not observed:
            next_persona = dict(self._start_prob)
        else:
            current_site = observed[-1]
            current_post = self._persona_posterior(observed)
            next_persona: dict[str, float] = {p: 0.0 for p in self._personas}
            for p, pp in current_post.items():
                for q, pq in self._trans_row(p, current_site).items():
                    next_persona[q] += pp * pq
        next_persona = self._normalize(next_persona)
        return sorted(next_persona.items(), key=lambda x: (-x[1], x[0]))

    def predict_persona(self, rows: Iterable[Mapping[str, Any]]) -> str | None:
        ranked = self._predict_persona_distribution(rows)
        return ranked[0][0] if ranked else None

    def _predict_next_site_distribution(
        self, rows: Iterable[Mapping[str, Any]]
    ) -> list[tuple[str, float]]:
        ordered = self._order_session_rows(list(rows))
        observed = [
            str(r.get("sni", "")).strip()
            for r in ordered
            if str(r.get("sni", "")).strip()
        ]
        if not observed or not self._personas or not self._snis:
            return []

        current_site = observed[-1]
        current_post = self._persona_posterior(observed)

        next_sni: dict[str, float] = {s: 0.0 for s in self._snis}
        for p, pp in current_post.items():
            for q, pq in self._trans_row(p, current_site).items():
                w = pp * pq
                if w == 0.0:
                    continue
                emit_q = self._emit_row(q, current_site)
                dflt = self._emit_default_for(q, current_site)
                for s in self._snis:
                    next_sni[s] += w * emit_q.get(s, dflt)

        next_sni = self._normalize(next_sni)
        return sorted(next_sni.items(), key=lambda x: (-x[1], x[0]))

    def predict(self, rows: Iterable[Mapping[str, Any]]) -> list[tuple[str, float]]:
        if self.prediction_target == "persona":
            return self._predict_persona_distribution(rows)
        return self._predict_next_site_distribution(rows)

    def next_probabilities(self, _current_sni: str) -> dict[str, float]:
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
                    "trans_site_prob": self._trans_site_prob,
                    "trans_persona_prob": self._trans_persona_prob,
                    "trans_site_marginal_prob": self._trans_site_marginal_prob,
                    "emit_site_prob": self._emit_site_prob,
                    "emit_site_row_default": self._emit_site_row_default,
                    "emit_persona_prob": self._emit_persona_prob,
                    "emit_persona_default": self._emit_persona_default,
                    "emit_site_marginal_prob": self._emit_site_marginal_prob,
                    "emit_site_marginal_default": self._emit_site_marginal_default,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> ModifiedHiddenMarkovPredictor:
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            alpha=float(data.get("alpha", 1.0)),
            prediction_target=str(data.get("prediction_target", "next_site")),
        )
        obj._personas = list(data.get("personas", []))
        obj._snis = list(data.get("snis", []))
        obj._start_prob = dict(data.get("start_prob", {}))
        obj._trans_site_prob = dict(data.get("trans_site_prob", {}))
        obj._trans_persona_prob = dict(data.get("trans_persona_prob", {}))
        obj._trans_site_marginal_prob = dict(data.get("trans_site_marginal_prob", {}))
        obj._emit_site_prob = dict(data.get("emit_site_prob", {}))
        obj._emit_site_row_default = dict(data.get("emit_site_row_default", {}))
        obj._emit_persona_prob = dict(data.get("emit_persona_prob", {}))
        obj._emit_persona_default = dict(data.get("emit_persona_default", {}))
        obj._emit_site_marginal_prob = dict(data.get("emit_site_marginal_prob", {}))
        obj._emit_site_marginal_default = dict(
            data.get("emit_site_marginal_default", {})
        )
        return obj
