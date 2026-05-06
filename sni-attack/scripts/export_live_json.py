#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export classifier/predictor pickles (Python 3) to JSON for ``attack.py`` on Python 2.7.

Normally ``train.py`` writes sibling ``.json`` files automatically. Use this script only
when you have an existing ``.pkl`` and need JSON::

    python3 scripts/export_live_json.py ../models/popular_classifier.pkl
    python3 scripts/export_live_json.py ../models/markov.pkl

Then on the attacker VM (Python 2.7)::

    sudo python attack.py \\
        --classifier-pkl ../models/popular_classifier.json \\
        --predictor-pkl ../models/markov.json

Despite the CLI flag names ``--*-pkl``, paths ending in ``.json`` are loaded via
``live_json_models`` (stdlib JSON only).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from live_model_json_spec import export_trained_model_to_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pkl", type=Path, help="Input .pkl from train.py")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .json path (default: same basename with .json)",
    )
    args = ap.parse_args()

    pkl_path = args.pkl.resolve()
    if not pkl_path.is_file():
        raise SystemExit("Not found: %s" % pkl_path)

    out_path = args.output
    if out_path is None:
        out_path = pkl_path.with_suffix(".json")
    else:
        out_path = out_path.resolve()

    with pkl_path.open("rb") as f:
        obj = pickle.load(f)

    export_trained_model_to_json(obj, out_path)
    print("Wrote %s" % out_path)


if __name__ == "__main__":
    main()
