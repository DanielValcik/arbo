#!/bin/bash
# Install Arbo retrospective systemd timers + services.
# Idempotent — safe to re-run after script/unit updates.
#
# Usage on VPS (arbo-dublin):
#   sudo bash /opt/arbo/scripts/systemd/install_summaries.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# Sanity
[ -f /opt/arbo/.env ] || { echo "[install] /opt/arbo/.env missing"; exit 1; }
[ -x /opt/arbo/.venv/bin/python ] || { echo "[install] arbo venv missing"; exit 1; }

echo "[install] copying units into /etc/systemd/system/"
for unit in arbo-summary-daily.service arbo-summary-daily.timer \
            arbo-summary-weekly.service arbo-summary-weekly.timer \
            arbo-summary-monthly.service arbo-summary-monthly.timer \
            arbo-summary-yearly.service arbo-summary-yearly.timer \
            arbo-parallel-digest.service arbo-parallel-digest.timer; do
    cp "$HERE/$unit" "/etc/systemd/system/$unit"
    chmod 0644 "/etc/systemd/system/$unit"
done

echo "[install] reloading systemd"
systemctl daemon-reload

echo "[install] enabling + starting timers"
for t in arbo-summary-daily.timer arbo-summary-weekly.timer \
         arbo-summary-monthly.timer arbo-summary-yearly.timer \
         arbo-parallel-digest.timer; do
    systemctl enable --now "$t"
done

echo "[install] done. Timer status:"
systemctl list-timers 'arbo-summary-*' 'arbo-parallel-*' --no-pager
