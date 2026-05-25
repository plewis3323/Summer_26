#!/usr/bin/env bash
set -euo pipefail

COUNTER_FILE="$(dirname "$0")/.git_push_counter"

if [[ -f "$COUNTER_FILE" ]]; then
    n=$(<"$COUNTER_FILE")
else
    n=0
fi
n=$((n + 1))
echo "$n" > "$COUNTER_FILE"

git add .
git commit -m "Change #${n}"
git push origin main



