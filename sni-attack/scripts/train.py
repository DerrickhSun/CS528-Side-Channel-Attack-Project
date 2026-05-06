"""
Train a selected model on sessions.csv (and related artifacts).
Add new trainers by defining a function and registering it in TRAINERS.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from models.first_order_markov import FirstOrderMarkov
from models.llm_predictor import LLMPredictor
from models.hidden_markov_predictor import HiddenMarkovPredictor
from models.modified_hidden_markov_predictor import ModifiedHiddenMarkovPredictor
from models.most_common_classifier import MostCommonClassifier
from models.most_common_predictor import MostCommonPredictor
from models.popular_classifier import PopularClassifier

from input_processing import iter_sessions, load_session_rows

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# Persona classifiers vs next-hop / sequence models — extend when adding trainers.
CLASSIFIER_MODELS: frozenset[str] = frozenset({"popular", "most_common_classifier"})
NEXT_SITE_MODELS: frozenset[str] = frozenset(
    {
        "markov",
        "first_order_markov",
        "hidden_markov_predictor",
        "modified_hidden_markov_predictor",
        "most_common_predictor",
        "llm_predictor",
    }
)


def session_accuracy_classifier(
    clf: Any,
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

    c, n, acc = session_accuracy_classifier(clf, sessions_csv)
    print(f"sessions.csv: {c}/{n} sessions correct ({acc:.1%})")

    demo_csv = DATA_DIR / "demo.csv"
    if demo_csv.is_file():
        c2, n2, acc2 = session_accuracy_classifier(clf, demo_csv)
        print(f"demo.csv:     {c2}/{n2} sessions correct ({acc2:.1%})")


def train_markov() -> None:
    """First-order SNI transition model → markov.pkl"""
    sessions_csv = DATA_DIR / "sessions.csv"
    if not sessions_csv.is_file():
        raise SystemExit(f"Missing {sessions_csv}; run generate.py first.")

    rows = load_session_rows(sessions_csv)
    m = FirstOrderMarkov().fit(rows)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "markov.pkl"
    m.save(out_path)
    print(f"Saved {out_path}")

    # Quick sanity: show next-site distribution from one common tail state
    sample = "paypal.com"
    dist = m.next_probabilities_list(sample)
    if dist:
        top = ", ".join(f"{s} ({p:.2%})" for s, p in dist[:5])
        print(f"Example P(next | {sample!r}): {top}{' …' if len(dist) > 5 else ''}")
    else:
        print(f"No outgoing transitions from {sample!r} in training.")


def train_most_common_predictor() -> None:
    """Global most-frequent next SNI baseline → most_common_predictor.pkl"""
    sessions_csv = DATA_DIR / "sessions.csv"
    if not sessions_csv.is_file():
        raise SystemExit(f"Missing {sessions_csv}; run generate.py first.")

    rows = load_session_rows(sessions_csv)
    p = MostCommonPredictor().fit(rows)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "most_common_predictor.pkl"
    p.save(out_path)
    print(f"Saved {out_path}")
    g = p.most_common_next()
    if g:
        n = p.next_target_counts().get(g, 0)
        print(f"Always predicts next = {g!r} ({n} occurrences as next-hop in training)")
    else:
        print("No transitions in training; model has no guess.")


def train_hidden_markov_predictor() -> None:
    """Persona-transition HMM-style next-site predictor → hidden_markov_predictor.pkl"""
    sessions_csv = DATA_DIR / "sessions.csv"
    if not sessions_csv.is_file():
        raise SystemExit(f"Missing {sessions_csv}; run generate.py first.")

    rows = load_session_rows(sessions_csv)
    m = HiddenMarkovPredictor().fit(rows)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "hidden_markov_predictor.pkl"
    m.save(out_path)
    print(f"Saved {out_path}")

    # Example next-site distribution from an early session prefix
    first_session = next(iter_sessions(rows))[1]
    prefix = first_session[: min(3, len(first_session))]
    dist = m.predict(prefix)
    if dist:
        top = ", ".join(f"{s} ({p:.2%})" for s, p in dist[:5])
        print(f"Example HMM next-site distribution: {top}{' ...' if len(dist) > 5 else ''}")
    else:
        print("No HMM prediction available for sample prefix.")


def train_modified_hidden_markov_predictor() -> None:
    """Observation-conditioned HMM next-site predictor → modified_hidden_markov_predictor.pkl"""
    sessions_csv = DATA_DIR / "sessions.csv"
    if not sessions_csv.is_file():
        raise SystemExit(f"Missing {sessions_csv}; run generate.py first.")

    rows = load_session_rows(sessions_csv)
    m = ModifiedHiddenMarkovPredictor().fit(rows)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "modified_hidden_markov_predictor.pkl"
    m.save(out_path)
    print(f"Saved {out_path}")

    # Example next-site distribution from an early session prefix
    first_session = next(iter_sessions(rows))[1]
    prefix = first_session[: min(3, len(first_session))]
    dist = m.predict(prefix)
    if dist:
        top = ", ".join(f"{s} ({p:.2%})" for s, p in dist[:5])
        print(
            "Example modified-HMM next-site distribution: "
            f"{top}{' ...' if len(dist) > 5 else ''}"
        )
    else:
        print("No modified-HMM prediction available for sample prefix.")


def train_most_common_classifier() -> None:
    """Majority-session persona baseline → most_common_classifier.pkl"""
    sessions_csv = DATA_DIR / "sessions.csv"
    if not sessions_csv.is_file():
        raise SystemExit(f"Missing {sessions_csv}; run generate.py first.")

    rows = load_session_rows(sessions_csv)
    clf = MostCommonClassifier().fit(rows)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "most_common_classifier.pkl"
    clf.save(out_path)
    print(f"Saved {out_path}")
    g = clf.most_common_persona()
    if g:
        print(f"Always predicts persona = {g!r}")
    else:
        print("No labeled sessions in training.")

    c, n, acc = session_accuracy_classifier(clf, sessions_csv)
    print(f"sessions.csv: {c}/{n} sessions correct ({acc:.1%})")

    demo_csv = DATA_DIR / "demo.csv"
    if demo_csv.is_file():
        c2, n2, acc2 = session_accuracy_classifier(clf, demo_csv)
        print(f"demo.csv:     {c2}/{n2} sessions correct ({acc2:.1%})")


def train_llm_predictor() -> None:
    """Fine-tuned DistilBERT next-SNI classifier → models/llm_predictor/"""
    sessions_csv = DATA_DIR / "sessions.csv"
    if not sessions_csv.is_file():
        raise SystemExit(f"Missing {sessions_csv}; run generate.py first.")

    rows = load_session_rows(sessions_csv)
    m = LLMPredictor().fit(rows)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = MODELS_DIR / "llm_predictor"
    m.save(out_dir)
    print(f"Saved {out_dir}")

    first_session = next(iter_sessions(rows))[1]
    prefix = first_session[: min(3, len(first_session))]
    dist = m.predict(prefix)
    if dist:
        top = ", ".join(f"{s} ({p:.2%})" for s, p in dist[:5])
        print(
            f"Example transformer next-site distribution: {top}"
            f"{' ...' if len(dist) > 5 else ''}"
        )
    else:
        print("No LLM predictor output for sample prefix (empty model?).")


# Register each trainable model: name shown on CLI -> no-arg trainer
TRAINERS: dict[str, Callable[[], None]] = {
    "first_order_markov": train_markov,
    "markov": train_markov,
    "most_common_classifier": train_most_common_classifier,
    "popular": train_popular,
    "hidden_markov_predictor": train_hidden_markov_predictor,
    "modified_hidden_markov_predictor": train_modified_hidden_markov_predictor,
    "most_common_predictor": train_most_common_predictor,
    "llm_predictor": train_llm_predictor,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a model from data/sessions.csv. "
            "Classifiers: %s. Next-site: %s."
            % (
                ", ".join(sorted(CLASSIFIER_MODELS)),
                ", ".join(sorted(NEXT_SITE_MODELS)),
            )
        ),
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
