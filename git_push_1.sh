#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# git_push_1.sh  --  run this on the SECOND machine (the nomachine / work box).
#
# The local terminal uses git_push.sh and pushes straight to `main`.
# This machine instead pushes its work to its OWN per-machine branch, e.g.
#     work-<hostname>-<N>
# so the two machines can never diverge on origin/main and you stop getting
# the "Change #N" counter / merge-conflict collisions.
#
# Because this machine never pushes to origin/main, `git pull --rebase` of main
# always stays clean: main only ever moves forward from the local terminal.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(dirname "$0")"

# Per-machine local counter (untracked via .gitignore -> never shared, never
# conflicts). Kept separate from git_push.sh's counter.
COUNTER_FILE="$SCRIPT_DIR/.git_push_1_counter"

if [[ -f "$COUNTER_FILE" ]]; then
    n=$(<"$COUNTER_FILE")
else
    n=0
fi
n=$((n + 1))
echo "$n" > "$COUNTER_FILE"

# Branch name unique to this machine, e.g. work-mylab-5
host=$(hostname -s)
branch="work-${host}-${n}"

git add .

# Don't fail the script if there's nothing to commit
if git diff --cached --quiet; then
    echo "Nothing to commit (Change #${n})"
else
    git commit -m "Change #${n} (${host})"
fi

# Pick up the local terminal's latest main first, so pulling is never a conflict.
git pull --rebase origin main

# Push this machine's work to its OWN branch (created automatically on first push).
git push origin "HEAD:${branch}"

echo "Pushed to branch: ${branch}"
