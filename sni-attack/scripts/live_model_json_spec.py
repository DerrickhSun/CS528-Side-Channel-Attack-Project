# -*- coding: utf-8 -*-
"""
Portable JSON snapshot of trained models for ``attack.py`` / ``live_json_models.py``.

Used by ``train.py`` (writes sibling ``.json``) and ``export_live_json.py`` (pickle → JSON).
Must stay aligned with ``FORMAT_VERSION`` in ``live_json_models.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from models.first_order_markov import FirstOrderMarkov
from models.hidden_markov_predictor import HiddenMarkovPredictor
from models.modified_hidden_markov_predictor import ModifiedHiddenMarkovPredictor
from models.most_common_classifier import MostCommonClassifier
from models.most_common_predictor import MostCommonPredictor
from models.popular_classifier import PopularClassifier

FORMAT_VERSION = 1


def model_to_live_json_spec(obj: object) -> Dict[str, Any]:
    """Turn an in-memory trained model into the dict ``live_json_models`` expects."""
    if isinstance(obj, PopularClassifier):
        return {
            "format_version": FORMAT_VERSION,
            "model_type": "popular_classifier",
            "counts": obj._counts,
        }
    if isinstance(obj, MostCommonClassifier):
        return {
            "format_version": FORMAT_VERSION,
            "model_type": "most_common_classifier",
            "guess": obj._guess,
        }
    if isinstance(obj, FirstOrderMarkov):
        return {
            "format_version": FORMAT_VERSION,
            "model_type": "first_order_markov",
            "counts": obj._counts,
        }
    if isinstance(obj, MostCommonPredictor):
        return {
            "format_version": FORMAT_VERSION,
            "model_type": "most_common_predictor",
            "next_site_counts": obj._next_site_counts,
        }
    if isinstance(obj, HiddenMarkovPredictor):
        return {
            "format_version": FORMAT_VERSION,
            "model_type": "hidden_markov_predictor",
            "alpha": obj.alpha,
            "prediction_target": obj.prediction_target,
            "personas": obj._personas,
            "snis": obj._snis,
            "start_prob": obj._start_prob,
            "trans_prob": obj._trans_prob,
            "emit_prob": obj._emit_prob,
            "emit_default": obj._emit_default,
        }
    if isinstance(obj, ModifiedHiddenMarkovPredictor):
        return {
            "format_version": FORMAT_VERSION,
            "model_type": "modified_hidden_markov_predictor",
            "alpha": obj.alpha,
            "prediction_target": obj.prediction_target,
            "personas": obj._personas,
            "snis": obj._snis,
            "start_prob": obj._start_prob,
            "trans_site_prob": obj._trans_site_prob,
            "trans_persona_prob": obj._trans_persona_prob,
            "trans_site_marginal_prob": obj._trans_site_marginal_prob,
            "emit_site_prob": obj._emit_site_prob,
            "emit_site_row_default": obj._emit_site_row_default,
            "emit_persona_prob": obj._emit_persona_prob,
            "emit_persona_default": obj._emit_persona_default,
            "emit_site_marginal_prob": obj._emit_site_marginal_prob,
            "emit_site_marginal_default": obj._emit_site_marginal_default,
        }
    raise TypeError("Unsupported model type for live JSON export: %r" % (type(obj),))


def write_live_model_json_spec(spec: Dict[str, Any], out_path: Path) -> None:
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, sort_keys=True)
        f.write("\n")


def export_trained_model_to_json(obj: object, json_path: Path) -> None:
    """Serialize ``obj`` to ``json_path`` for use with ``attack.py`` ``*.json`` paths."""
    spec = model_to_live_json_spec(obj)
    write_live_model_json_spec(spec, json_path)
