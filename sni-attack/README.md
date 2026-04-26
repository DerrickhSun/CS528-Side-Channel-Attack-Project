# CS 528 — SNI Traffic Analysis Attack

## Overview

This project demonstrates a privacy attack against TLS Server Name Indication (SNI), a field transmitted in cleartext during the TLS handshake. SNI was designed to allow servers to host multiple domains on one IP address, but its unencrypted nature exposes a user's browsing destinations to any passive observer on the network.

The project is structured in three parts:

### Part 1 — The Obvious Leak
Capture live TLS traffic between two VMs and show that SNI reveals the destination hostname in plaintext. No decryption required. A passive attacker running tcpdump can trivially read every site a user visits in real time.

### Part 2 — The Deeper Leak
Train a Markov-based probability model on SNI sequences to learn user behavioral patterns. The model classifies which type of user is browsing (Student, Shopper, or News Reader) based on early SNI observations, then predicts where they will go next — before they get there. This enables a pre-emptive phishing attack: the attacker prepares a fake login page for the predicted destination before the victim even types the URL.

### Part 3 — Mitigation: ESNI and ECH
Discuss Encrypted Client Hello (ECH) and its predecessor ESNI as the community's response to SNI leakage. Covers what ECH is, how it works, references to research on its effectiveness, and the current deployment landscape. The live demo shows that enabling ECH on the victim VM causes the capture to go blind — the attack from Parts 1 and 2 collapses completely.

---

## Project Structure

```
sni-attack/
├── data/
│   ├── sessions.csv          # 150 labeled training sessions (generated)
│   ├── demo.csv              # 15 fresh sessions for live demo (generated)
│   ├── capture.pcap          # Raw packet dump from tcpdump (kept for debugging)
│   └── captured_sni.csv      # Parsed SNI hostnames + timestamps [attacker VM]
├── models/
│   ├── markov.pkl            # Markov transition tables per persona
│   └── classifier.pkl        # Persona classifier
├── scripts/
│   ├── generate.py           # Build sessions.csv + demo.csv [dev machine]
│   ├── train.py              # Train + evaluate models [dev machine]
│   ├── victim.sh             # curl sessions from CSV [victim VM]
│   ├── capture.sh            # tcpdump → capture.pcap, then calls parse_sni.py [attacker VM]
│   ├── parse_sni.py          # Pure Python 2.7 pcap parser — extracts SNI from TLS ClientHellos [attacker VM]
│   └── attack.py             # Live classify + predict [attacker VM]
├── README.md
└── requirements.txt
```

---

## VM Setup

| Machine | Role | Scripts |
|---|---|---|
| Victim VM | Browses sites via curl following sessions.csv or demo.csv | `victim.sh` |
| Attacker VM | Sniffs traffic with tcpdump, parses SNI, runs live prediction | `capture.sh`, `parse_sni.py`, `attack.py` |
| Dev Machine | Generates data, trains models before the demo | `generate.py`, `train.py` |

Both VMs must be on the same network segment so the attacker can observe the victim's TLS handshakes.

> **Note on tooling:** The lab VMs run Ubuntu 12.04 / tshark 1.6.7 (2012), which predates TLS SNI dissection support. The capture pipeline was redesigned to use `tcpdump -w` for raw packet collection and `parse_sni.py` — a pure Python 2.7 stdlib parser — to extract SNI fields directly from the pcap binary.

---

## Personas

Three user archetypes are used to generate realistic, distinguishable SNI sequences. Each persona has a defined vocabulary of SNI hostnames. Sessions sample 4–6 hops from that vocabulary using weighted transition probabilities.

### Shared SNIs (all personas)
These appear across all personas and have low discriminative value on their own. The classifier waits for at least one non-shared SNI before assigning a persona.

| SNI | Notes |
|---|---|
| `google.com` | Universal starting point |
| `reddit.com` | Used by students and news readers alike |

---

### Persona 1 — Student

Models a Purdue University student accessing academic resources.

| SNI | Description |
|---|---|
| `purdue.edu` | Main university portal |
| `mail.purdue.edu` | Student email |
| `purdue.brightspace.com` | Learning management system (LMS) |
| `piazza.com` | Course Q&A platform |
| `gradescope.com` | Assignment submission and grading |
| `edstem.org` | Course discussion and coding platform |
| `vocareum.com` | Cloud-based lab environment |
| `lib.purdue.edu` | Purdue library |
| `mypurdue.purdue.edu` | Student portal (registration, records) |

**Characteristic transitions:** `purdue.edu → brightspace.com`, `mail.purdue.edu → gradescope.com`, `edstem.org → vocareum.com`

---

### Persona 2 — Shopper

Models a user doing online comparison shopping and completing purchases.

| SNI | Description |
|---|---|
| `amazon.com` | Primary online retailer |
| `ebay.com` | Auction and secondary market |
| `walmart.com` | Big-box online retail |
| `paypal.com` | Payment processor |
| `bestbuy.com` | Electronics retail |
| `target.com` | General merchandise |
| `etsy.com` | Handmade and specialty goods |
| `craigslist.org` | Local classifieds |
| `venmo.com` | Peer payment |

**Characteristic transitions:** `amazon.com → paypal.com`, `google.com → amazon.com → bestbuy.com`, `ebay.com → paypal.com`

---

### Persona 3 — News Reader

Models a user following current events across multiple news outlets.

| SNI | Description |
|---|---|
| `cnn.com` | Cable news |
| `bbc.com` | International news |
| `reuters.com` | Wire news service |
| `nytimes.com` | Newspaper of record |
| `apnews.com` | Associated Press |
| `theguardian.com` | UK/international news |
| `foxnews.com` | Cable news |
| `npr.org` | Public radio news |

**Characteristic transitions:** `google.com → cnn.com → reddit.com`, `bbc.com → reuters.com`, `nytimes.com → theguardian.com`

---

## Data Generation

`generate.py` synthesizes realistic SNI sessions by:

1. Randomly selecting a persona for each session
2. Sampling a sequence of 4–6 SNI hops using the persona's weighted transition probabilities
3. Adding realistic timing gaps between hops (1–5 seconds) and between sessions (10–30 seconds)
4. Writing two output files:
   - `sessions.csv` — 150 sessions (50 per persona) for training
   - `demo.csv` — 15 sessions (5 per persona) for the live demo

### sessions.csv schema

| Column | Description |
|---|---|
| `session_id` | Unique integer per browsing session |
| `persona` | Ground truth label: `student`, `shopper`, `news_reader` |
| `sni` | Hostname visited |
| `timestamp` | Simulated Unix timestamp |
| `hop` | Position within the session (0-indexed) |

---

## Run Order

### Before the demo (dev machine)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate training and demo sessions
python scripts/generate.py

# 3. Train and evaluate the model
python scripts/train.py
```

### Part 1 — Live SNI capture (both VMs)

```bash
# Attacker VM — start capture first (requires sudo)
sudo bash scripts/capture.sh

# Victim VM — browse through sessions to generate TLS traffic
bash scripts/victim.sh data/sessions.csv
```

Press Ctrl-C on the attacker when done. `capture.sh` automatically calls `parse_sni.py` on exit and writes `data/captured_sni.csv`. Open it to show the plaintext SNI leak — every hostname the victim visited, no decryption required.

To manually test the victim side without a full session CSV, you can run curl directly:

```bash
for host in github.com google.com purdue.edu; do
    curl -sI https://$host > /dev/null
    sleep 2
done
```

### Part 2 — Live prediction demo (both VMs)

```bash
# Attacker VM — start live attack
python scripts/attack.py

# Victim VM — browse demo sessions
bash scripts/victim.sh data/demo.csv
```

The attacker terminal prints persona classification and next-site predictions in real time.

### Part 3 — ECH mitigation demo

Enable ECH on the victim VM, re-run a few demo sessions, and show that `captured_sni.csv` comes back empty — tcpdump captures packets but `parse_sni.py` finds no readable SNI fields.

---

## Model Design

### Stage 1 — Persona classifier
Observes the first 2–3 non-shared SNIs in a session and classifies the user as Student, Shopper, or News Reader. Uses a simple frequency-based approach: each SNI votes for the persona it belongs to exclusively.

### Stage 2 — Markov predictor
A first-order Markov chain per persona. Trained on SNI transition pairs from `sessions.csv`. Given the current SNI, outputs a probability distribution over the next SNI. Reports top-1 and top-3 predictions.

### Evaluation metrics
- Top-1 accuracy vs random baseline
- Top-1 accuracy vs most-frequent baseline (always predict `google.com`)
- Top-3 accuracy
- Confusion matrix across personas

---

## Attack Output Example

```
[SESSION 7]
  Seen:      google.com → amazon.com
  Persona:   Shopper (confidence 91%)
  Predicts:  paypal.com (58%) | ebay.com (24%) | walmart.com (12%)
  → ACTION: Prepare fake paypal.com credential page
```

---

## Dependencies

See `requirements.txt`. Key packages:

- `pandas` — session data handling
- `scikit-learn` — classifier
- `numpy` — transition matrix math

`parse_sni.py` uses only the Python 2.7 standard library — no external packages required for packet capture and parsing.

---

## Notes

- `sessions.csv` contains ground truth persona labels. The attacker never has access to these — `captured_sni.csv` contains only raw SNIs and timestamps.
- Session boundaries in `captured_sni.csv` are inferred from timing gaps (>8 seconds between SNIs = new session).
- `reddit.com` and `google.com` are intentionally shared across personas to test classifier robustness on ambiguous inputs.
