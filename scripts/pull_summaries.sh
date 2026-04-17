#!/bin/bash
# Pull retrospective summaries from VPS to local, commit + push.
#
# The VPS runs scripts/retrospective.py via systemd timers (daily,
# weekly, monthly, yearly) and writes under /opt/arbo/summaries/ with
# --no-commit. This script is the laptop-side counterpart that
# syncs those files into the local repo and ships them to GitHub.
#
# Safe to run anytime — rsync is idempotent, git commit only happens
# if new content arrived.
#
# Usage:
#   bash scripts/pull_summaries.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "[pull] syncing from arbo-dublin:/opt/arbo/summaries/"
rsync -av --quiet \
    arbo-dublin:/opt/arbo/summaries/ \
    "$REPO_ROOT/summaries/"

echo "[pull] checking for changes"
if git diff --quiet --exit-code summaries/ && git diff --cached --quiet --exit-code summaries/; then
    # No tracked changes — but maybe new untracked files
    if [ -z "$(git ls-files --others --exclude-standard summaries/)" ]; then
        echo "[pull] no new summaries"
        exit 0
    fi
fi

echo "[pull] committing new summaries"
git add summaries/
# Produce a brief list of what's new for the commit message
newfiles="$(git diff --cached --name-only summaries/ | sed 's|^|  |')"
git commit -m "docs(summaries): sync retrospectives from VPS

$(echo "$newfiles")"

echo "[pull] pushing to origin"
git push origin main
echo "[pull] done"
