"""
K-fold validation for models trained on sessions.csv (session-level splits).

* **Classifiers** (e.g. popular, most_common_classifier): session-level persona
  accuracy; per-hop rows repeat the session prediction.
* **Next-site predictors** (Markov, most_common_predictor, …): transition-level
  top-1 next-SNI accuracy; one row per hop edge in the test split.

Output defaults: data/classifier_results/ or data/predictor_results/ (by model type),
each with <model>_validation_results.csv and <model>_validation_summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol

from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from models.first_order_markov import FirstOrderMarkov
from models.most_common_classifier import MostCommonClassifier
from models.most_common_predictor import MostCommonPredictor
from models.popular_classifier import PopularClassifier

from input_processing import group_rows_by_session, load_session_rows

DATA_DIR = ROOT / "data"
CLASSIFIER_RESULTS_DIR = DATA_DIR / "classifier_results"
PREDICTOR_RESULTS_DIR = DATA_DIR / "predictor_results"

# Extend when adding models: persona classifiers vs next-hop / sequence models.
CLASSIFIER_MODELS: frozenset[str] = frozenset({"popular", "most_common_classifier"})
NEXT_SITE_MODELS: frozenset[str] = frozenset(
    {"markov", "first_order_markov", "most_common_predictor"}
)

ModelKind = Literal["classifier", "next_site"]


def model_kind(name: str) -> ModelKind:
    if name in CLASSIFIER_MODELS:
        return "classifier"
    if name in NEXT_SITE_MODELS:
        return "next_site"
    raise KeyError(name)


class SessionClassifier(Protocol):
    def fit(self, rows: list[dict[str, Any]]) -> Any: ...
    def predict(self, rows: list[dict[str, Any]]) -> str | None: ...


def _sessions_list_from_csv(sessions_path: Path) -> list[list[dict[str, Any]]]:
    rows = load_session_rows(sessions_path)
    grouped = group_rows_by_session(rows)
    sids = sorted(grouped.keys())
    return [grouped[sid] for sid in sids]


def session_level_test_accuracy(
    clf: SessionClassifier,
    test_sessions: list[list[dict[str, Any]]],
) -> tuple[int, int, int]:
    """
    Returns (correct, total_with_prediction, skipped_no_prediction).
    """
    correct = 0
    total = 0
    skipped = 0
    for session_rows in test_sessions:
        true_p = str(session_rows[0]["persona"]).strip()
        pred = clf.predict(session_rows)
        if pred is None:
            skipped += 1
            continue
        total += 1
        if pred == true_p:
            correct += 1
    return correct, total, skipped


def _cv_classifier_session(
    sessions_path: Path,
    n_splits: int,
    random_state: int,
    model_cls: type,
) -> dict[str, Any]:
    """Shared K-fold CV for ``fit(rows)`` + ``predict(session_rows)`` classifiers."""
    sessions_list = _sessions_list_from_csv(sessions_path)
    n_sessions = len(sessions_list)
    if n_sessions < n_splits:
        raise SystemExit(
            f"Need at least {n_splits} sessions for {n_splits}-fold CV; got {n_sessions}."
        )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pooled_correct = 0
    pooled_total = 0
    pooled_skipped = 0
    fold_reports: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        kf.split(range(n_sessions)),
        start=1,
    ):
        train_rows: list[dict[str, Any]] = []
        for i in train_idx:
            train_rows.extend(sessions_list[i])
        test_sessions = [sessions_list[i] for i in test_idx]

        clf = model_cls().fit(train_rows)
        correct, total, skipped = session_level_test_accuracy(clf, test_sessions)
        acc = correct / total if total else 0.0
        pooled_correct += correct
        pooled_total += total
        pooled_skipped += skipped

        row_predictions: list[dict[str, Any]] = []
        for session_rows in test_sessions:
            true_p = str(session_rows[0]["persona"]).strip()
            pred = clf.predict(session_rows)
            session_ok = pred is not None and pred == true_p
            for r in session_rows:
                row_predictions.append(
                    {
                        "session_id": int(r["session_id"]),
                        "hop": int(r["hop"]),
                        "timestamp": r.get("timestamp"),
                        "sni": str(r["sni"]),
                        "true_persona": str(r["persona"]).strip(),
                        "predicted_persona": pred,
                        "session_prediction_correct": session_ok,
                    }
                )

        fold_reports.append(
            {
                "fold": fold_idx,
                "session_accuracy": {
                    "correct": correct,
                    "total_with_prediction": total,
                    "skipped_sessions_no_prediction": skipped,
                    "accuracy": acc,
                },
                "row_predictions": row_predictions,
            }
        )

    micro = pooled_correct / pooled_total if pooled_total else 0.0
    return {
        "csv": str(sessions_path.resolve()),
        "n_splits": n_splits,
        "random_state": random_state,
        "folds": fold_reports,
        "pooled": {
            "correct": pooled_correct,
            "total_with_prediction": pooled_total,
            "skipped_sessions_no_prediction": pooled_skipped,
            "micro_accuracy": micro,
        },
    }


def cv_popular(
    sessions_path: Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    return _cv_classifier_session(sessions_path, n_splits, random_state, PopularClassifier)


def cv_most_common_classifier(
    sessions_path: Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    return _cv_classifier_session(sessions_path, n_splits, random_state, MostCommonClassifier)


def _top3_prediction_fields(
    ranked: list[tuple[str, float]],
) -> dict[str, Any]:
    """Six CSV columns: pred_rank{1,2,3}_{sni,prob}; empty strings if fewer than 3."""
    fields: dict[str, Any] = {}
    for k in range(1, 4):
        fields[f"pred_rank{k}_sni"] = ""
        fields[f"pred_rank{k}_prob"] = ""
    for k, (sni, p) in enumerate(ranked[:3], start=1):
        fields[f"pred_rank{k}_sni"] = sni
        fields[f"pred_rank{k}_prob"] = round(float(p), 6)
    return fields


def _cv_next_site_model(
    sessions_path: Path,
    n_splits: int,
    random_state: int,
    model_cls: type,
) -> dict[str, Any]:
    """Shared K-fold CV for models with ``fit(rows)``, ``next_probabilities(s)``."""
    sessions_list = _sessions_list_from_csv(sessions_path)
    n_sessions = len(sessions_list)
    if n_sessions < n_splits:
        raise SystemExit(
            f"Need at least {n_splits} sessions for {n_splits}-fold CV; got {n_sessions}."
        )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pooled_correct = 0
    pooled_total = 0
    pooled_skipped = 0
    fold_reports: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        kf.split(range(n_sessions)),
        start=1,
    ):
        train_rows: list[dict[str, Any]] = []
        for i in train_idx:
            train_rows.extend(sessions_list[i])
        test_sessions = [sessions_list[i] for i in test_idx]

        m = model_cls().fit(train_rows)
        correct = 0
        total = 0
        skipped = 0
        edge_predictions: list[dict[str, Any]] = []

        for session_rows in test_sessions:
            for i in range(len(session_rows) - 1):
                r_from, r_to = session_rows[i], session_rows[i + 1]
                from_sni = str(r_from["sni"]).strip()
                actual_next = str(r_to["sni"]).strip()
                from_hop = int(r_from["hop"])
                to_hop = int(r_to["hop"])
                sid = int(r_from["session_id"])
                dist = m.next_probabilities(from_sni)
                if not dist:
                    skipped += 1
                    edge_predictions.append(
                        {
                            "session_id": sid,
                            "from_hop": from_hop,
                            "to_hop": to_hop,
                            "from_sni": from_sni,
                            "actual_next_sni": actual_next,
                            **_top3_prediction_fields([]),
                            "prob_actual_next": "",
                            "top1_correct": False,
                            "evaluated": False,
                        }
                    )
                    continue
                total += 1
                ranked = m.next_probabilities_list(from_sni)
                top1 = ranked[0][0] if ranked else ""
                top1_ok = top1 == actual_next
                if top1_ok:
                    correct += 1
                prob_actual = float(dist.get(actual_next, 0.0))
                edge_predictions.append(
                    {
                        "session_id": sid,
                        "from_hop": from_hop,
                        "to_hop": to_hop,
                        "from_sni": from_sni,
                        "actual_next_sni": actual_next,
                        **_top3_prediction_fields(ranked),
                        "prob_actual_next": round(prob_actual, 6),
                        "top1_correct": top1_ok,
                        "evaluated": True,
                    }
                )

        acc = correct / total if total else 0.0
        pooled_correct += correct
        pooled_total += total
        pooled_skipped += skipped

        fold_reports.append(
            {
                "fold": fold_idx,
                "transition_accuracy": {
                    "correct": correct,
                    "total_evaluated": total,
                    "skipped_unknown_state": skipped,
                    "accuracy": acc,
                },
                "edge_predictions": edge_predictions,
            }
        )

    micro = pooled_correct / pooled_total if pooled_total else 0.0
    return {
        "csv": str(sessions_path.resolve()),
        "n_splits": n_splits,
        "random_state": random_state,
        "folds": fold_reports,
        "pooled": {
            "correct": pooled_correct,
            "total_evaluated": pooled_total,
            "skipped_unknown_state": pooled_skipped,
            "micro_accuracy": micro,
        },
    }


def cv_first_order_markov(
    sessions_path: Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    return _cv_next_site_model(sessions_path, n_splits, random_state, FirstOrderMarkov)


def cv_most_common_predictor(
    sessions_path: Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    return _cv_next_site_model(sessions_path, n_splits, random_state, MostCommonPredictor)


CV_RUNNERS: dict[str, Callable[[Path, int, int], dict[str, Any]]] = {
    "popular": cv_popular,
    "most_common_classifier": cv_most_common_classifier,
    "markov": cv_first_order_markov,
    "first_order_markov": cv_first_order_markov,
    "most_common_predictor": cv_most_common_predictor,
}

CLASSIFIER_RESULTS_FIELDS = [
    "fold",
    "session_id",
    "hop",
    "timestamp",
    "sni",
    "true_persona",
    "predicted_persona",
    "session_prediction_correct",
]

CLASSIFIER_SUMMARY_FIELDS = [
    "n_splits",
    "random_state",
    "fold",
    "correct_sessions",
    "total_with_prediction",
    "skipped_sessions",
    "session_accuracy",
]

NEXT_SITE_RESULTS_FIELDS = [
    "fold",
    "session_id",
    "from_hop",
    "to_hop",
    "from_sni",
    "actual_next_sni",
    "pred_rank1_sni",
    "pred_rank1_prob",
    "pred_rank2_sni",
    "pred_rank2_prob",
    "pred_rank3_sni",
    "pred_rank3_prob",
    "prob_actual_next",
    "top1_correct",
    "evaluated",
]

NEXT_SITE_SUMMARY_FIELDS = [
    "n_splits",
    "random_state",
    "fold",
    "correct_transitions",
    "total_transitions_evaluated",
    "skipped_unknown_from_state",
    "top1_accuracy",
]


def write_classifier_results_csv(report: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CLASSIFIER_RESULTS_FIELDS)
        w.writeheader()
        for fr in report["folds"]:
            fold = fr["fold"]
            for rp in fr["row_predictions"]:
                w.writerow(
                    {
                        "fold": fold,
                        "session_id": rp["session_id"],
                        "hop": rp["hop"],
                        "timestamp": rp["timestamp"],
                        "sni": rp["sni"],
                        "true_persona": rp["true_persona"],
                        "predicted_persona": rp["predicted_persona"] or "",
                        "session_prediction_correct": rp["session_prediction_correct"],
                    }
                )


def write_classifier_summary_csv(report: dict[str, Any], out: Path) -> None:
    n_splits = report["n_splits"]
    seed = report["random_state"]
    pool = report["pooled"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CLASSIFIER_SUMMARY_FIELDS)
        w.writeheader()
        for fr in report["folds"]:
            sa = fr["session_accuracy"]
            w.writerow(
                {
                    "n_splits": n_splits,
                    "random_state": seed,
                    "fold": fr["fold"],
                    "correct_sessions": sa["correct"],
                    "total_with_prediction": sa["total_with_prediction"],
                    "skipped_sessions": sa["skipped_sessions_no_prediction"],
                    "session_accuracy": round(sa["accuracy"], 6),
                }
            )
        w.writerow(
            {
                "n_splits": n_splits,
                "random_state": seed,
                "fold": "pooled",
                "correct_sessions": pool["correct"],
                "total_with_prediction": pool["total_with_prediction"],
                "skipped_sessions": pool["skipped_sessions_no_prediction"],
                "session_accuracy": round(pool["micro_accuracy"], 6),
            }
        )


def write_next_site_results_csv(report: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=NEXT_SITE_RESULTS_FIELDS)
        w.writeheader()
        for fr in report["folds"]:
            fold = fr["fold"]
            for ep in fr["edge_predictions"]:
                w.writerow({**ep, "fold": fold})


def write_next_site_summary_csv(report: dict[str, Any], out: Path) -> None:
    n_splits = report["n_splits"]
    seed = report["random_state"]
    pool = report["pooled"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=NEXT_SITE_SUMMARY_FIELDS)
        w.writeheader()
        for fr in report["folds"]:
            ta = fr["transition_accuracy"]
            w.writerow(
                {
                    "n_splits": n_splits,
                    "random_state": seed,
                    "fold": fr["fold"],
                    "correct_transitions": ta["correct"],
                    "total_transitions_evaluated": ta["total_evaluated"],
                    "skipped_unknown_from_state": ta["skipped_unknown_state"],
                    "top1_accuracy": round(ta["accuracy"], 6),
                }
            )
        w.writerow(
            {
                "n_splits": n_splits,
                "random_state": seed,
                "fold": "pooled",
                "correct_transitions": pool["correct"],
                "total_transitions_evaluated": pool["total_evaluated"],
                "skipped_unknown_from_state": pool["skipped_unknown_state"],
                "top1_accuracy": round(pool["micro_accuracy"], 6),
            }
        )


def print_classifier_summary(report: dict[str, Any], n_splits: int) -> None:
    for fr in report["folds"]:
        fold_idx = fr["fold"]
        sa = fr["session_accuracy"]
        c, t = sa["correct"], sa["total_with_prediction"]
        sk = sa["skipped_sessions_no_prediction"]
        acc = sa["accuracy"]
        extra = f", {sk} test session(s) with no prediction" if sk else ""
        print(
            f"Fold {fold_idx}/{n_splits}: "
            f"{c}/{t} correct ({acc:.1%}) on test sessions{extra}"
        )
    p = report["pooled"]
    micro = p["micro_accuracy"]
    print(
        f"\nPooled (micro) over all test sessions with predictions: "
        f"{p['correct']}/{p['total_with_prediction']} ({micro:.1%})"
    )
    if p["skipped_sessions_no_prediction"]:
        print(
            f"Skipped across folds (no prediction): "
            f"{p['skipped_sessions_no_prediction']} session(s)"
        )


def print_next_site_summary(report: dict[str, Any], n_splits: int) -> None:
    for fr in report["folds"]:
        fold_idx = fr["fold"]
        ta = fr["transition_accuracy"]
        c, t = ta["correct"], ta["total_evaluated"]
        sk = ta["skipped_unknown_state"]
        acc = ta["accuracy"]
        extra = f", {sk} edge(s) skipped (unknown from-state)" if sk else ""
        print(
            f"Fold {fold_idx}/{n_splits}: "
            f"{c}/{t} top-1 next-SNI correct ({acc:.1%}){extra}"
        )
    p = report["pooled"]
    micro = p["micro_accuracy"]
    print(
        f"\nPooled (micro) over evaluated transitions: "
        f"{p['correct']}/{p['total_evaluated']} ({micro:.1%})"
    )
    if p["skipped_unknown_state"]:
        print(
            f"Skipped across folds (no training data for from-SNI): "
            f"{p['skipped_unknown_state']} edge(s)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "K-fold CV on sessions.csv (session-level splits). "
            "Classifiers: persona accuracy. Next-site models: top-1 transition accuracy."
        ),
    )
    parser.add_argument(
        "model",
        choices=sorted(CV_RUNNERS.keys()),
        metavar="MODEL",
        help="model to validate (%s)" % ", ".join(sorted(CV_RUNNERS.keys())),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DATA_DIR / "sessions.csv",
        help="path to sessions CSV (default: data/sessions.csv)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help=(
            "detailed results CSV (default: data/classifier_results/ or "
            "data/predictor_results/<MODEL>_validation_results.csv)"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help=(
            "per-fold + pooled summary CSV (default: same folder as default results, "
            "<MODEL>_validation_summary.csv)"
        ),
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="number of folds (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for shuffling sessions before splitting (default: 42)",
    )
    args = parser.parse_args()
    path = args.csv
    if not path.is_file():
        raise SystemExit(f"Missing {path}")
    if args.folds < 2:
        raise SystemExit("--folds must be at least 2")

    report = CV_RUNNERS[args.model](path, n_splits=args.folds, random_state=args.seed)
    kind = model_kind(args.model)
    default_dir = (
        CLASSIFIER_RESULTS_DIR if kind == "classifier" else PREDICTOR_RESULTS_DIR
    )

    results_path = args.results or (
        default_dir / f"{args.model}_validation_results.csv"
    )
    summary_path = args.summary or (
        default_dir / f"{args.model}_validation_summary.csv"
    )

    if kind == "classifier":
        write_classifier_results_csv(report, results_path)
        write_classifier_summary_csv(report, summary_path)
        print_summary = print_classifier_summary
    else:
        write_next_site_results_csv(report, results_path)
        write_next_site_summary_csv(report, summary_path)
        print_summary = print_next_site_summary

    print(f"Wrote detailed results to {results_path.resolve()}")
    print(f"Wrote summary to       {summary_path.resolve()}\n")
    print_summary(report, args.folds)


if __name__ == "__main__":
    main()
