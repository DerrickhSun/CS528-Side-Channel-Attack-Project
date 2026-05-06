"""
Next-site predictor using a fine-tuned Hugging Face transformer.

Treats next-hop prediction as multi-class classification: the input text is the
visited SNI sequence up to the current hop (joined with ``" | "``), and the
label is the next SNI (restricted to SNIs that occur as a ``next`` hop in
training). Matches the ``fit(rows)`` / ``predict(rows)`` contract used by
``FirstOrderMarkov``.

Requires: ``torch``, ``transformers``, ``datasets`` (see requirements.txt).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from models.model_helper import order_session_rows, sessions_ordered

_METADATA_NAME = "llm_predictor_meta.json"


class LLMPredictor:
    """
    Fine-tune ``AutoModelForSequenceClassification`` on (prefix text → next SNI).

    Default base checkpoint is ``distilbert-base-uncased`` (small, runs on CPU
    for modest data sizes; use a GPU for faster training).
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        *,
        max_length: int = 128,
        epochs: float = 15.0,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed

        self._tokenizer = None
        self._model = None
        self._sorted_labels: list[str] = []
        self._label2id: dict[str, int] = {}
        self._device: Any = None

    @staticmethod
    def _align_tokenizer_truncation(tokenizer: Any) -> None:
        """Keep the **recent** SNIs when truncating long prefixes.

        Prefix text is chronological (``early | … | latest``). The HF default is
        ``truncation_side="right"``, which drops trailing tokens—the wrong end
        for next-hop prediction. ``"left"`` removes the **start**, preserving
        the last hop Markov chains condition on.
        """
        setattr(tokenizer, "truncation_side", "left")

    def _ensure_torch(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "LLMPredictor requires PyTorch. Install with: pip install torch"
            ) from e

    def _build_transition_examples(
        self, rows: Iterable[Mapping[str, Any]]
    ) -> tuple[list[str], list[int], list[str]]:
        """Returns (prefix_texts, label_ids, sorted_label_strings)."""
        sessions = sessions_ordered(rows)
        next_snis: set[str] = set()
        for session in sessions:
            snis = [
                str(r.get("sni", "")).strip()
                for r in session
                if str(r.get("sni", "")).strip()
            ]
            for i in range(len(snis) - 1):
                next_snis.add(snis[i + 1])

        if not next_snis:
            return [], [], []

        sorted_labels = sorted(next_snis)
        label2id = {s: i for i, s in enumerate(sorted_labels)}

        texts: list[str] = []
        labels: list[int] = []
        for session in sessions:
            snis = [
                str(r.get("sni", "")).strip()
                for r in session
                if str(r.get("sni", "")).strip()
            ]
            for i in range(len(snis) - 1):
                texts.append(" | ".join(snis[: i + 1]))
                labels.append(label2id[snis[i + 1]])

        return texts, labels, sorted_labels

    def fit(self, rows: Iterable[Mapping[str, Any]]) -> LLMPredictor:
        self._ensure_torch()
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )

        texts, labels, sorted_labels = self._build_transition_examples(rows)
        self._sorted_labels = sorted_labels
        self._label2id = {s: i for i, s in enumerate(sorted_labels)}

        if not texts:
            self._tokenizer = None
            self._model = None
            return self

        num_labels = len(sorted_labels)
        torch.manual_seed(self.seed)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._align_tokenizer_truncation(tokenizer)
        id2label = {i: s for i, s in enumerate(sorted_labels)}
        label2id = {s: i for i, s in enumerate(sorted_labels)}

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        ds = Dataset.from_dict({"text": texts, "label": labels})

        def tokenize(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )

        ds_tok = ds.map(tokenize, batched=True)
        drop_cols = [c for c in ds_tok.column_names if c not in ("label", "input_ids", "attention_mask")]
        if drop_cols:
            ds_tok = ds_tok.remove_columns(drop_cols)

        ds_tok = ds_tok.rename_column("label", "labels")

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        scratch_dir = tempfile.mkdtemp(prefix="llm_predictor_train_")
        training_args = TrainingArguments(
            output_dir=str(scratch_dir),
            num_train_epochs=float(self.epochs),
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            logging_steps=max(1, len(ds_tok) // max(self.batch_size, 1)),
            save_strategy="no",
            report_to="none",
            seed=self.seed,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds_tok,
            data_collator=data_collator,
        )
        trainer.train()

        shutil.rmtree(scratch_dir, ignore_errors=True)

        self._tokenizer = tokenizer
        self._model = model
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()
        return self

    def predict(self, rows: Iterable[Mapping[str, Any]]) -> list[tuple[str, float]]:
        if self._model is None or self._tokenizer is None or not self._sorted_labels:
            return []

        self._ensure_torch()
        import torch

        ordered = order_session_rows(list(rows))
        snis = [
            str(r.get("sni", "")).strip()
            for r in ordered
            if str(r.get("sni", "")).strip()
        ]
        if not snis:
            return []

        text = " | ".join(snis)
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            truncation_side="left",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits[0], dim=-1)

        n = min(len(self._sorted_labels), int(probs.shape[-1]))
        pairs = [
            (self._sorted_labels[i], float(probs[i].item())) for i in range(n)
        ]
        pairs.sort(key=lambda x: (-x[1], x[0]))
        return pairs

    def next_probabilities(self, current_sni: str) -> dict[str, float]:
        return {}

    def next_probabilities_list(self, current_sni: str) -> list[tuple[str, float]]:
        return []

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_name": self.model_name,
            "sorted_labels": self._sorted_labels,
            "label2id": self._label2id,
            "max_length": self.max_length,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
        }
        with open(path / _METADATA_NAME, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if self._model is None or self._tokenizer is None:
            return

        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str | Path) -> LLMPredictor:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(path)
        meta_path = path / _METADATA_NAME
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing {meta_path}")

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        obj = cls(
            model_name=meta.get("model_name", "distilbert-base-uncased"),
            max_length=int(meta.get("max_length", 128)),
            epochs=float(meta.get("epochs", 3)),
            batch_size=int(meta.get("batch_size", 8)),
            learning_rate=float(meta.get("learning_rate", 2e-5)),
            seed=int(meta.get("seed", 42)),
        )
        obj._sorted_labels = list(meta.get("sorted_labels", []))
        obj._label2id = dict(meta.get("label2id", {}))

        if not obj._sorted_labels:
            return obj

        obj._tokenizer = AutoTokenizer.from_pretrained(str(path))
        cls._align_tokenizer_truncation(obj._tokenizer)
        obj._model = AutoModelForSequenceClassification.from_pretrained(str(path))
        obj._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obj._model.to(obj._device)
        obj._model.eval()
        return obj


class LLMPredictorFastCV(LLMPredictor):
    """Smaller training budget for K-fold CV (single epoch, default batch)."""

    def __init__(self) -> None:
        super().__init__(epochs=1.0, batch_size=8)
