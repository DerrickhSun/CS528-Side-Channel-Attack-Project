"""
5-fold cross-validation for session-level persona classifiers.
Splits by session_id so all hops from a session stay in train or test together.
Writes two CSVs under data/: <model>_validation_results.csv (per-hop predictions)
and <model>_validation_summary.csv (per-fold and pooled metrics).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from models.popular_classifier import PopularClassifier

from input_processing import group_rows_by_session, load_session_rows

DATA_DIR = ROOT / "data"


class SessionClassifier(Protocol):
    def fit(self, rows: list[dict[str, Any]]) -> Any: ...
    def predict(self, rows: list[dict[str, Any]]) -> str | None: ...


def session_level_test_accuracy(
    clf: SessionClassifier,
    test_sessions: list[list[dict[str, Any]]],
) -> tuple[int, int, int]:
    """
    Returns (correct, total_with_prediction, skipped_no_prediction).
    Same counting rule as train.py: sessions where predict returns None are skipped.
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


def cv_popular(
    sessions_path: Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    rows = load_session_rows(sessions_path)
    grouped = group_rows_by_session(rows)
    sids = sorted(grouped.keys())
    sessions_list = [grouped[sid] for sid in sids]
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

        clf = PopularClassifier().fit(train_rows)
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


CV_RUNNERS: dict[str, Callable[[Path, int, int], dict[str, Any]]] = {
    "popular": cv_popular,
}

RESULTS_FIELDNAMES = [
    "fold",
    "session_id",
    "hop",
    "timestamp",
    "sni",
    "true_persona",
    "predicted_persona",
    "session_prediction_correct",
]

SUMMARY_FIELDNAMES = [
    "n_splits",
    "random_state",
    "fold",
    "correct_sessions",
    "total_with_prediction",
    "skipped_sessions",
    "session_accuracy",
]


def write_validation_results_csv(report: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_FIELDNAMES)
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


def write_validation_summary_csv(report: dict[str, Any], out: Path) -> None:
    n_splits = report["n_splits"]
    seed = report["random_state"]
    pool = report["pooled"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
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


def print_summary(report: dict[str, Any], n_splits: int) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="K-fold session-level CV for persona classifiers (sessions.csv).",
    )
    parser.add_argument(
        "model",
        choices=sorted(CV_RUNNERS.keys()),
        metavar="MODEL",
        help="classifier to validate (%s)" % ", ".join(sorted(CV_RUNNERS.keys())),
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
        help="per-row predictions CSV (default: data/<MODEL>_validation_results.csv)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="per-fold + pooled summary CSV (default: data/<MODEL>_validation_summary.csv)",
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

    results_path = args.results or (DATA_DIR / f"{args.model}_validation_results.csv")
    summary_path = args.summary or (DATA_DIR / f"{args.model}_validation_summary.csv")
    write_validation_results_csv(report, results_path)
    write_validation_summary_csv(report, summary_path)
    print(f"Wrote predictions to {results_path.resolve()}")
    print(f"Wrote summary to    {summary_path.resolve()}\n")

    print_summary(report, args.folds)


if __name__ == "__main__":
    main()
