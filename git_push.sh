#!/usr/bin/env bash
set -euo pipefail

# .git_push_counter is intentionally untracked (see .gitignore). Each machine
# (this box + the nomachine/mylab clone) keeps its own local counter, so the
# file can never cause a push conflict the way it used to.
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

# Integrate any remote work (e.g. pushes from the nomachine clone) before
# pushing, so diverged history rebases automatically instead of being rejected.
git pull --rebase origin main

git push origin main
