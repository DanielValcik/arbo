#!/usr/bin/env bash
# One-time VPS setup for Arbo (Polymarket trading system)
# Target: AWS Lightsail eu-west-2 (London), Ubuntu 24.04, 2 vCPU / 4 GB RAM
# Run as root: ssh ubuntu@<ip> 'sudo bash -s' < scripts/setup_vps.sh
set -euo pipefail

echo "=== Arbo VPS Setup (AWS Lightsail London) ==="

# ---- System updates ----
apt-get update && apt-get upgrade -y

# ---- PostgreSQL 16 ----
apt-get install -y postgresql-16 postgresql-client-16

# ---- Python 3.12 ----
apt-get install -y python3.12 python3.12-venv python3.12-dev python3-pip

# ---- Build dependencies (asyncpg, web3, sentence-transformers, etc.) ----
apt-get install -y build-essential libpq-dev

# ---- Firewall: SSH + dashboard port 8080 ----
ufw allow OpenSSH
ufw allow 8080/tcp
ufw --force enable

# ---- Create arbo user ----
useradd -r -m -s /bin/bash arbo || true

# ---- Project directory ----
mkdir -p /opt/arbo/{models,logs}
chown -R arbo:arbo /opt/arbo

# ---- Python venv ----
su - arbo -c "python3.12 -m venv /opt/arbo/.venv"

# ---- PostgreSQL: create user + database with secure password ----
DB_PASSWORD=$(openssl rand -base64 24 | tr -d '/+=' | head -c 32)
sudo -u postgres createuser arbo 2>/dev/null || true
sudo -u postgres createdb -O arbo arbo 2>/dev/null || true
sudo -u postgres psql -c "ALTER USER arbo WITH PASSWORD '${DB_PASSWORD}';" 2>/dev/null || true

# Enable and start PostgreSQL
systemctl enable postgresql
systemctl start postgresql

# ---- Install systemd service ----
cp /opt/arbo/arbo.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable arbo

# ---- Allow arbo user to manage its own service (for deploy.sh) ----
cat > /etc/sudoers.d/arbo << 'SUDOERS'
arbo ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart arbo
arbo ALL=(ALL) NOPASSWD: /usr/bin/systemctl status arbo
arbo ALL=(ALL) NOPASSWD: /usr/bin/systemctl stop arbo
SUDOERS
chmod 440 /etc/sudoers.d/arbo

echo ""
echo "=== VPS Setup Complete ==="
echo ""
echo "Generated DB password: ${DB_PASSWORD}"
echo ""
echo "Next steps:"
echo "  1. Copy .env to /opt/arbo/.env"
echo "  2. Update DATABASE_URL in .env:"
echo "     DATABASE_URL=postgresql+asyncpg://arbo:${DB_PASSWORD}@localhost:5432/arbo"
echo "  3. Deploy: ./scripts/deploy.sh arbo@<lightsail-ip>"
echo "  4. Verify: ssh arbo@<lightsail-ip> 'journalctl -u arbo -f'"
echo ""
echo "IMPORTANT: Save the DB password above — it won't be shown again!"
