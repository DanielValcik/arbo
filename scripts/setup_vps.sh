#!/usr/bin/env bash
# One-time VPS setup for Arbo (Polymarket trading system)
# Run as root on fresh Ubuntu 24.04 Hetzner CX22
set -euo pipefail

echo "=== Arbo VPS Setup ==="

# System updates
apt-get update && apt-get upgrade -y

# PostgreSQL 16
apt-get install -y postgresql-16 postgresql-client-16

# Python 3.12
apt-get install -y python3.12 python3.12-venv python3.12-dev python3-pip

# Build dependencies (needed for asyncpg, web3, etc.)
apt-get install -y build-essential libpq-dev

# Create arbo user
useradd -r -m -s /bin/bash arbo || true

# Create project directory
mkdir -p /opt/arbo
chown arbo:arbo /opt/arbo

# Setup Python venv
su - arbo -c "python3.12 -m venv /opt/arbo/.venv"

# Setup PostgreSQL
sudo -u postgres createuser arbo || true
sudo -u postgres createdb -O arbo arbo || true
sudo -u postgres psql -c "ALTER USER arbo WITH PASSWORD 'password';" || true

# Enable and start PostgreSQL
systemctl enable postgresql
systemctl start postgresql

# Install systemd service
cp /opt/arbo/arbo.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable arbo

echo "=== VPS Setup Complete ==="
echo "Next steps:"
echo "  1. Copy .env to /opt/arbo/.env (update DATABASE_URL password!)"
echo "  2. Run: ./scripts/deploy.sh arbo@<vps-ip>"
echo "  3. Verify: ssh arbo@<vps-ip> 'journalctl -u arbo -f'"
