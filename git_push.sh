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

# Push whatever branch is currently checked out, not a hardcoded "main".
# (Committing to a feature branch but pushing "main" silently dropped work.)
branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$branch" == "HEAD" ]]; then
    echo "Detached HEAD -- checkout a branch before pushing." >&2
    exit 1
fi

git add .

# Don't fail the script if there's nothing to commit
if git diff --cached --quiet; then
    echo "Nothing to commit (Change #${n})"
else
    git commit -m "Change #${n}"
fi

# Integrate any remote work on this same branch before pushing, so diverged
# history rebases automatically instead of being rejected. Only rebase if the
# branch already exists on origin (a brand-new branch has nothing to pull).
if git ls-remote --exit-code --heads origin "$branch" >/dev/null 2>&1; then
    git pull --rebase origin "$branch"
fi

# Push the current branch to the same-named branch on origin, and set upstream.
git push -u origin "$branch"
