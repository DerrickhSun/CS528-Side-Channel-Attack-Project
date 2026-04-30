#!/bin/bash
# Runs on dns_user (victim VM).
# Reads a session CSV (sessions.csv or demo.csv), curls each SNI over HTTPS,
#
# Usage: bash victim.sh <csvfile>
#   e.g. bash victim.sh ../data/demo.csv

CSV="${1:-}"
if [ -z "$CSV" ]; then
    echo "Usage: $0 <csvfile>"
    exit 1
fi

if [ ! -f "$CSV" ]; then
    echo "File not found: $CSV"
    exit 1
fi

PREV_SESSION=""

# Skip the header line, then process each row
tail -n +2 "$CSV" | while IFS=, read -r session_id persona sni timestamp hop; do

    # Between sessions: longer pause so the attacker's boundary detector fires
    if [ -n "$PREV_SESSION" ] && [ "$session_id" != "$PREV_SESSION" ]; then
        echo ""
        echo "--- end of session $PREV_SESSION, sleeping 5s ---"
        sleep 5
    fi

    # New session header
    if [ "$session_id" != "$PREV_SESSION" ]; then
        echo ""
        echo "=== SESSION $session_id ($persona) ==="
    fi

    echo "  curl https://$sni"
    curl -sk --max-time 5 "https://$sni" > /dev/null

    PREV_SESSION="$session_id"
    sleep 2

done

echo ""
echo "Done."
