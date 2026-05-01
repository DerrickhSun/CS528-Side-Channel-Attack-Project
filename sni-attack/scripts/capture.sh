#!/bin/bash
#
# capture.sh — passive SNI capture on the attacker VM
#
# Captures TLS traffic with tcpdump, then parses SNIs out of the pcap using parse_sni.py
#
# Output (written to the same scripts/ directory):
#   capture.pcap        — raw packet dump (kept for debugging)
#   captured_sni.csv    — parsed timestamp,sni rows
#
# Stop: Ctrl-C (the parser runs automatically on exit)
# From vitctim run "curl https://google.com" 
# --- Config -----------------------------------------------------------

INTERFACE="eth14"
BPF="tcp port 443"
PCAP_FILE="capture.pcap"
CSV_FILE="captured_sni.csv"
PARSER="parse_sni.py"

# --- Pre-flight -------------------------------------------------------

if ! command -v tcpdump >/dev/null 2>&1; then
    echo "ERROR: tcpdump not installed." >&2
    exit 1
fi

if [[ ! -f "$PARSER" ]]; then
    echo "ERROR: parse_sni.py not found at $PARSER" >&2
    exit 1
fi

echo "[capture.sh] Interface: $INTERFACE"
echo "[capture.sh] Pcap:      $PCAP_FILE"
echo "[capture.sh] CSV:       $CSV_FILE"
echo "[capture.sh] Press Ctrl-C to stop and parse."
echo ""

# --- Capture ----------------------------------------------------------

# tcpdump flags:
#   -i  interface
#   -w  write raw packets to pcap (no parsing — fastest, lossless)
#   -n  no DNS resolution
#   -s 0  capture full packets (default snaplen on old tcpdump is 68 bytes,
#         which truncates ClientHellos — must override)
#
# Trap EXIT so the parser runs automatically when tcpdump stops.
on_exit() {
    echo ""
    echo "[capture.sh] Capture stopped. Parsing SNIs..."
    python "$PARSER" "$PCAP_FILE" "$CSV_FILE"
    rows=$(($(wc -l < "$CSV_FILE") - 1))
    echo "[capture.sh] Done. Wrote $rows SNI rows to $CSV_FILE"
}
trap on_exit EXIT

tcpdump -i "$INTERFACE" -n -s 0 -w "$PCAP_FILE" "$BPF"
