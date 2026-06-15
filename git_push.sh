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

# Don't fail the script if there's nothing to commit
if git diff --cached --quiet; then
    echo "Nothing to commit (Change #${n})"
else
    git commit -m "Change #${n}"
fi

git push origin main
