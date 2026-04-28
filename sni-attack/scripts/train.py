"""
Train a selected model on sessions.csv (and related artifacts).
Add new trainers by defining a function and registering it in TRAINERS.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from models.popular_classifier import PopularClassifier

from input_processing import iter_sessions, load_session_rows

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


def session_accuracy_popular(
    clf: PopularClassifier,
    rows_path: Path,
) -> tuple[int, int, float]:
    """Returns (correct, total_sessions_with_prediction, accuracy)."""
    rows = load_session_rows(rows_path)
    correct = 0
    total = 0
    for _, session_rows in iter_sessions(rows):
        true_p = str(session_rows[0]["persona"]).strip()
        pred = clf.predict(session_rows)
        if pred is None:
            continue
        total += 1
        if pred == true_p:
            correct += 1
    acc = correct / total if total else 0.0
    return correct, total, acc


def train_popular() -> None:
    """SNI vote / co-occurrence persona classifier → popular_classifier.pkl"""
    sessions_csv = DATA_DIR / "sessions.csv"
    if not sessions_csv.is_file():
        raise SystemExit(f"Missing {sessions_csv}; run generate.py first.")

    rows = load_session_rows(sessions_csv)
    clf = PopularClassifier().fit(rows)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "popular_classifier.pkl"
    clf.save(out_path)
    print(f"Saved {out_path}")

    c, n, acc = session_accuracy_popular(clf, sessions_csv)
    print(f"sessions.csv: {c}/{n} sessions correct ({acc:.1%})")

    demo_csv = DATA_DIR / "demo.csv"
    if demo_csv.is_file():
        c2, n2, acc2 = session_accuracy_popular(clf, demo_csv)
        print(f"demo.csv:     {c2}/{n2} sessions correct ({acc2:.1%})")


# Register each trainable model: name shown on CLI -> no-arg trainer
TRAINERS: dict[str, Callable[[], None]] = {
    "popular": train_popular,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a persona / traffic model from data/sessions.csv.",
    )
    parser.add_argument(
        "model",
        choices=sorted(TRAINERS.keys()),
        metavar="MODEL",
        help="which model to train (%s)" % ", ".join(sorted(TRAINERS.keys())),
    )
    args = parser.parse_args()
    TRAINERS[args.model]()


if __name__ == "__main__":
    main()
