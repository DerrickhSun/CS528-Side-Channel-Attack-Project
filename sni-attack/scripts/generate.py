"""
generate.py — synthesizes SNI browsing sessions for three user personas.
Writes data to sessions.csv (150 sessions) and demo.csv (15 sessions).
The sessions.csv is used to traint he model. 
The demo.csv is used during demo, its a new set of data but same schema as sessions.csv
"""

import csv
import random
import time
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Persona config
# Each persona has a list of start sites (with weights) and a transition table: {current_sni: [(next_sni, weight), ...]}
# For example: student starts 40% at google.com, 40% at purdue.edu, and 20% at mail.purdue.edu. 
# If currently at google.com, next hop is 40% purdue.edu, 30% reddit.com, etc.
# ---------------------------------------------------------------------------

PERSONAS = {
    "student": {
        "starts": [
            ("google.com", 0.4),
            ("purdue.edu", 0.4),
            ("mail.purdue.edu", 0.2),
        ],
        "transitions": {
            "google.com":             [("purdue.edu", 0.4), ("reddit.com", 0.3), ("mail.purdue.edu", 0.2), ("lib.purdue.edu", 0.1)],
            "purdue.edu":             [("purdue.brightspace.com", 0.5), ("mail.purdue.edu", 0.2), ("mypurdue.purdue.edu", 0.2), ("lib.purdue.edu", 0.1)],
            "mail.purdue.edu":        [("gradescope.com", 0.4), ("piazza.com", 0.3), ("purdue.brightspace.com", 0.2), ("edstem.org", 0.1)],
            "purdue.brightspace.com": [("piazza.com", 0.3), ("edstem.org", 0.3), ("gradescope.com", 0.3), ("reddit.com", 0.1)],
            "piazza.com":             [("edstem.org", 0.4), ("gradescope.com", 0.3), ("purdue.brightspace.com", 0.2), ("reddit.com", 0.1)],
            "gradescope.com":         [("vocareum.com", 0.4), ("edstem.org", 0.3), ("piazza.com", 0.2), ("reddit.com", 0.1)],
            "edstem.org":             [("vocareum.com", 0.5), ("gradescope.com", 0.2), ("piazza.com", 0.2), ("reddit.com", 0.1)],
            "vocareum.com":           [("edstem.org", 0.3), ("gradescope.com", 0.3), ("lib.purdue.edu", 0.2), ("reddit.com", 0.2)],
            "lib.purdue.edu":         [("purdue.edu", 0.3), ("google.com", 0.3), ("mypurdue.purdue.edu", 0.2), ("reddit.com", 0.2)],
            "mypurdue.purdue.edu":    [("purdue.edu", 0.4), ("mail.purdue.edu", 0.3), ("purdue.brightspace.com", 0.2), ("reddit.com", 0.1)],
            "reddit.com":             [("google.com", 0.4), ("purdue.edu", 0.3), ("lib.purdue.edu", 0.2), ("piazza.com", 0.1)],
        },
    },

    "shopper": {
        "starts": [
            ("google.com", 0.4),
            ("amazon.com", 0.4),
            ("ebay.com", 0.2),
        ],
        "transitions": {
            "google.com":     [("amazon.com", 0.5), ("ebay.com", 0.2), ("walmart.com", 0.2), ("reddit.com", 0.1)],
            "amazon.com":     [("paypal.com", 0.35), ("bestbuy.com", 0.25), ("ebay.com", 0.2), ("walmart.com", 0.1), ("target.com", 0.1)],
            "ebay.com":       [("paypal.com", 0.5), ("craigslist.org", 0.2), ("amazon.com", 0.2), ("venmo.com", 0.1)],
            "walmart.com":    [("target.com", 0.3), ("amazon.com", 0.3), ("paypal.com", 0.2), ("google.com", 0.2)],
            "paypal.com":     [("amazon.com", 0.3), ("ebay.com", 0.3), ("venmo.com", 0.2), ("google.com", 0.2)],
            "bestbuy.com":    [("amazon.com", 0.3), ("paypal.com", 0.3), ("walmart.com", 0.2), ("target.com", 0.2)],
            "target.com":     [("walmart.com", 0.3), ("amazon.com", 0.3), ("paypal.com", 0.2), ("etsy.com", 0.2)],
            "etsy.com":       [("paypal.com", 0.4), ("amazon.com", 0.2), ("target.com", 0.2), ("venmo.com", 0.2)],
            "craigslist.org": [("ebay.com", 0.4), ("paypal.com", 0.3), ("google.com", 0.2), ("venmo.com", 0.1)],
            "venmo.com":      [("paypal.com", 0.4), ("amazon.com", 0.2), ("ebay.com", 0.2), ("google.com", 0.2)],
            "reddit.com":     [("google.com", 0.4), ("amazon.com", 0.3), ("ebay.com", 0.2), ("craigslist.org", 0.1)],
        },
    },

    "news_reader": {
        "starts": [
            ("google.com", 0.5),
            ("cnn.com", 0.3),
            ("bbc.com", 0.2),
        ],
        "transitions": {
            "google.com":      [("cnn.com", 0.4), ("reddit.com", 0.3), ("bbc.com", 0.2), ("nytimes.com", 0.1)],
            "cnn.com":         [("reddit.com", 0.3), ("bbc.com", 0.2), ("foxnews.com", 0.2), ("nytimes.com", 0.2), ("apnews.com", 0.1)],
            "bbc.com":         [("reuters.com", 0.4), ("theguardian.com", 0.3), ("nytimes.com", 0.2), ("cnn.com", 0.1)],
            "reuters.com":     [("apnews.com", 0.3), ("bbc.com", 0.3), ("nytimes.com", 0.2), ("google.com", 0.2)],
            "nytimes.com":     [("theguardian.com", 0.4), ("bbc.com", 0.2), ("reuters.com", 0.2), ("reddit.com", 0.2)],
            "apnews.com":      [("reuters.com", 0.3), ("cnn.com", 0.3), ("bbc.com", 0.2), ("npr.org", 0.2)],
            "theguardian.com": [("bbc.com", 0.3), ("nytimes.com", 0.3), ("reuters.com", 0.2), ("reddit.com", 0.2)],
            "foxnews.com":     [("cnn.com", 0.3), ("reddit.com", 0.3), ("google.com", 0.2), ("apnews.com", 0.2)],
            "npr.org":         [("apnews.com", 0.3), ("reuters.com", 0.3), ("bbc.com", 0.2), ("google.com", 0.2)],
            "reddit.com":      [("google.com", 0.3), ("cnn.com", 0.3), ("nytimes.com", 0.2), ("bbc.com", 0.2)],
        },
    },
}


# ---------------------------------------------------------------------------
# Session generation
# ---------------------------------------------------------------------------

def pick(weighted_list):
    """Choose from [(item, weight), ...] using weighted random selection."""
    items, weights = zip(*weighted_list)
    return random.choices(items, weights=weights)[0]


def build_a_session(session_id, persona, start_time):
    """
    Build one browsing session of 4–6 hops for the given persona.
    Returns (list of row dicts, end_timestamp).
    """
    config = PERSONAS[persona]
    hops = random.randint(4, 6)
    current = pick(config["starts"]) # pick a start site based on the persona's start distribution

    rows = []
    t = start_time
    for hop in range(hops):
        rows.append({
            "session_id": session_id,
            "persona": persona,
            "sni": current,
            "timestamp": int(t),
            "hop": hop,
        })
        t += random.uniform(1, 5)
        current = pick(config["transitions"][current])
    return rows, t


def build_all_sessions(num_sessions, start_time):
    """
    Generate num_sessions of sessions for each persona.
    Returns a flat list of row dicts with continuous session_ids.
    """
    raw = []
    sid = 0
    t = start_time
    for persona in PERSONAS:
        for _ in range(num_sessions):
            rows, t = build_a_session(sid, persona, t)
            raw.append(rows)
            sid += 1
            t += random.uniform(10, 30)

    # Shuffle the sessions to mix personas and make it less predictable, 
    # then flatten to a single list of rows with new continuous session_ids.
    random.shuffle(raw)
    flat = []
    for new_id, rows in enumerate(raw):
        for row in rows:
            row["session_id"] = new_id
        flat.extend(rows)
    return flat


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

FIELDS = ["session_id", "persona", "sni", "timestamp", "hop"]

def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    n_sessions = len(set(r["session_id"] for r in rows))
    print(f"  wrote {path.name}: {len(rows)} rows, {n_sessions} sessions")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    base_time = time.time() - 86400  # start from 24 hours ago

    print("Generating sessions.csv (150 sessions, 50 per persona)...")
    sessions = build_all_sessions(num_sessions=50, start_time=base_time)
    write_csv(DATA_DIR / "sessions.csv", sessions)

    print("Generating demo.csv (15 sessions, 5 per persona)...")
    demo_start = sessions[-1]["timestamp"] + random.uniform(30, 60) # start demo sessions after the last training session
    demo = build_all_sessions(num_sessions=5, start_time=demo_start)
    write_csv(DATA_DIR / "demo.csv", demo)


if __name__ == "__main__":
    main()
