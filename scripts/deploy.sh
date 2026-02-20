#!/usr/bin/env bash
# Deploy Arbo to VPS
# Usage: ./scripts/deploy.sh [user@host]
set -euo pipefail

VPS="${1:-arbo@your-vps-ip}"
REMOTE_DIR="/opt/arbo"

echo "=== Deploying Arbo to ${VPS}:${REMOTE_DIR} ==="

# Sync code (exclude dev/local files)
rsync -avz --delete \
    --exclude '.env' \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '.mypy_cache' \
    --exclude '.pytest_cache' \
    --exclude '.ruff_cache' \
    --exclude 'models/*.joblib' \
    --exclude '.git' \
    --exclude 'docs/' \
    --exclude 'Arbo_DevBrief_FINAL.md' \
    --exclude 'arbo_devbrief_v4_final.md' \
    --exclude 'arbo_cto_handoff_memo.md' \
    ./ "${VPS}:${REMOTE_DIR}/"

echo "=== Installing dependencies ==="
ssh "${VPS}" "cd ${REMOTE_DIR} && .venv/bin/pip install -e '.[dev]'"

echo "=== Running migrations ==="
ssh "${VPS}" "cd ${REMOTE_DIR} && .venv/bin/alembic upgrade head"

echo "=== Restarting service ==="
ssh "${VPS}" "sudo systemctl restart arbo"

echo "=== Checking status ==="
ssh "${VPS}" "sudo systemctl status arbo --no-pager"

echo "=== Deploy complete ==="
