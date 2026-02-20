#!/usr/bin/env bash
# One-time VPS setup for Arbo
# Run as root on fresh Ubuntu 24.04 Hetzner CX22
set -euo pipefail

echo "=== Arbo VPS Setup ==="

# System updates
apt-get update && apt-get upgrade -y

# PostgreSQL 16
apt-get install -y postgresql-16 postgresql-client-16

# Redis 7
apt-get install -y redis-server

# Python 3.12
apt-get install -y python3.12 python3.12-venv python3.12-dev python3-pip

# Build dependencies
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

# Enable and start services
systemctl enable postgresql redis-server
systemctl start postgresql redis-server

# Install systemd service
cp /opt/arbo/arbo.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable arbo

echo "=== VPS Setup Complete ==="
echo "Next steps:"
echo "  1. Copy .env to /opt/arbo/.env"
echo "  2. Run: cd /opt/arbo && .venv/bin/pip install -e '.[dev]'"
echo "  3. Run: cd /opt/arbo && .venv/bin/alembic upgrade head"
echo "  4. Run: systemctl start arbo"
