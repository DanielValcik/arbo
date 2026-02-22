#!/usr/bin/env bash
# Deploy Arbo to AWS Lightsail (London eu-west-2)
# Usage: ./scripts/deploy.sh [user@host]
set -euo pipefail

VPS="${1:-arbo}"
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
    --exclude '/models/' \
    --exclude 'logs/' \
    --exclude '.git' \
    --exclude '.claude' \
    --exclude 'docs/' \
    --exclude '_archive/' \
    --exclude 'Screenshot*' \
    --exclude 'ARBO_CTO_Development_Brief_v3.md' \
    --exclude 'Arbo_CTO_Handoff_Memo.md' \
    ./ "${VPS}:${REMOTE_DIR}/"

echo "=== Installing dependencies ==="
ssh "${VPS}" "cd ${REMOTE_DIR} && .venv/bin/pip install -e '.[dev]' --quiet"

echo "=== Running migrations ==="
ssh "${VPS}" "cd ${REMOTE_DIR} && .venv/bin/python -m alembic upgrade head"

echo "=== Restarting service ==="
ssh "${VPS}" "sudo systemctl restart arbo"

# Wait for service to stabilize
sleep 3

echo "=== Checking status ==="
ssh "${VPS}" "sudo systemctl status arbo --no-pager"

echo ""
echo "=== Deploy complete ==="
echo "Logs:    ssh ${VPS} 'journalctl -u arbo -f'"
echo "Status:  ssh ${VPS} 'sudo systemctl status arbo'"
echo "Restart: ssh ${VPS} 'sudo systemctl restart arbo'"
