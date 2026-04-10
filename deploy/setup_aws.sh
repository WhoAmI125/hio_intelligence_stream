#!/usr/bin/env bash
# HiO v2 AWS g4dn Deployment Script
# Usage: sudo deploy/setup_aws.sh
set -euo pipefail

PROJECT_DIR="/home/ubuntu/hio_v2"
VENV="$PROJECT_DIR/venv"
SERVICE_USER="hioapp"
DEPLOY_DIR="$PROJECT_DIR/deploy"

echo "═══════════════════════════════════════════════════"
echo "  HiO v2 — AWS g4dn Deployment"
echo "═══════════════════════════════════════════════════"

# ── 1. Preflight ──────────────────────────────────────
echo ""
echo "▸ Stage 1: Preflight checks"
if ! command -v nvidia-smi &>/dev/null; then
    echo "  ✗ nvidia-smi not found. Install NVIDIA driver first."
    echo "    sudo apt install -y nvidia-driver-570-server && sudo reboot"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "  ✓ GPU available"
df -h / | tail -1 | awk '{print "  Disk: " $3 " used / " $2 " total (" $5 " used)"}'
free -h | grep Mem | awk '{print "  RAM:  " $3 " used / " $2 " total"}'

# ── 2. Service account ───────────────────────────────
echo ""
echo "▸ Stage 2: Service account"
if id "$SERVICE_USER" &>/dev/null; then
    echo "  ✓ User $SERVICE_USER exists"
else
    useradd --system --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER"
    usermod -aG video,render "$SERVICE_USER"
    echo "  ✓ Created $SERVICE_USER (video,render groups)"
fi

# ── 3. Runtime directories ───────────────────────────
echo ""
echo "▸ Stage 3: Runtime directories"
for d in data data/clips data/thumbnails data/logs data/events db; do
    mkdir -p "$PROJECT_DIR/$d"
done
echo "  ✓ Directories created"

# ── 4. Environment file ──────────────────────────────
echo ""
echo "▸ Stage 4: Environment file"
if [ ! -f "$PROJECT_DIR/.env" ]; then
    cp "$DEPLOY_DIR/.env.aws" "$PROJECT_DIR/.env"
    echo "  ✓ .env created from .env.aws (edit GEMINI_API_KEY!)"
else
    echo "  ✓ .env already exists (skipped)"
fi

# ── 5. Permissions ────────────────────────────────────
echo ""
echo "▸ Stage 5: Permissions"
chown -R "$SERVICE_USER:$SERVICE_USER" "$PROJECT_DIR/data" "$PROJECT_DIR/db"
chmod -R 755 "$PROJECT_DIR"
echo "  ✓ Ownership set"

# ── 6. systemd services ──────────────────────────────
echo ""
echo "▸ Stage 6: systemd services"
cp "$DEPLOY_DIR/hio-model.service" /etc/systemd/system/
cp "$DEPLOY_DIR/hio-frontend.service" /etc/systemd/system/
cp "$DEPLOY_DIR/hio-boot-recover.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable hio-model hio-frontend hio-boot-recover
echo "  ✓ Services installed + enabled"

# ── 7. nginx ──────────────────────────────────────────
echo ""
echo "▸ Stage 7: nginx"
cp "$DEPLOY_DIR/nginx.conf" /etc/nginx/sites-available/hio-cctv
ln -sf /etc/nginx/sites-available/hio-cctv /etc/nginx/sites-enabled/hio-cctv
rm -f /etc/nginx/sites-enabled/default
nginx -t
echo "  ✓ nginx configured"

# ── 8. Start services ────────────────────────────────
echo ""
echo "▸ Stage 8: Starting services"
systemctl start hio-model
echo "  ⏳ Waiting for model server (up to 120s)..."
for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:8000/api/status &>/dev/null; then
        echo "  ✓ Model server ready (${i}s)"
        break
    fi
    sleep 1
done

systemctl start hio-frontend
sleep 2
echo "  ✓ Frontend server started"

systemctl restart nginx
echo "  ✓ nginx started"

# ── Done ──────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✓ Deployment complete!"
echo ""
echo "  Internal:  http://127.0.0.1:8002/monitor"
echo "  External:  http://<your-domain>/"
echo ""
echo "  Next steps:"
echo "    1. Edit .env → set GEMINI_API_KEY"
echo "    2. sudo certbot --nginx -d <your-domain>"
echo "    3. Add cameras via UI"
echo "═══════════════════════════════════════════════════"
