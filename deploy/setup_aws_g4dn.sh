#!/usr/bin/env bash
# ============================================================
# VLM Intelligent CCTV System — AWS g4dn Setup Script
# ============================================================
#
# Target: Ubuntu 24.04 LTS, g4dn.xlarge (Tesla T4)
# Project: /home/ubuntu/hio_intelligence_stream
#
# Prerequisites (do these BEFORE running this script):
#   1. EC2 instance running (g4dn.xlarge, 30GB+ root volume)
#   2. Volume expanded: sudo growpart /dev/nvme0n1 1 && sudo resize2fs /dev/nvme0n1p1
#   3. chmod 755 /home/ubuntu
#   4. NVIDIA driver installed: sudo apt-get install -y nvidia-driver-570-server && sudo modprobe nvidia
#   5. CUDA torch pre-installed:
#        cd /home/ubuntu/hio_intelligence_stream
#        python3 -m venv venv
#        sudo venv/bin/pip install --no-cache-dir -r requirements_gpu.txt
#        sudo rm -rf /root/.cache/pip
#
# Usage:
#   cd /home/ubuntu/hio_intelligence_stream
#   sudo bash deploy/setup_aws_g4dn.sh
#
# ============================================================

set -euo pipefail

# ---- Config ----
PROJECT_DIR="/home/ubuntu/hio_intelligence_stream"
VENV_DIR="${PROJECT_DIR}/venv"
SERVICE_USER="vlmapp"
DEPLOY_DIR="${PROJECT_DIR}/deploy"
SKIP_MODEL_PRELOAD="${SKIP_MODEL_PRELOAD:-1}"

print_resource_snapshot() {
    echo "  Resource check:"
    df -h / | awk 'NR==1 || NR==2 {print "    " $0}'
    free -h | awk 'NR==1 || NR==2 {print "    " $0}'
}

echo "============================================================"
echo "  VLM CCTV System — AWS g4dn Setup"
echo "============================================================"
echo "  Project:  ${PROJECT_DIR}"
echo "  User:     ${SERVICE_USER}"
echo "============================================================"

# ---- 0. Preflight ----
echo ""
echo "[0/8] Running preflight checks..."
chmod 755 /home/ubuntu
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "  ERROR: nvidia-smi not found. Install NVIDIA driver first."
    exit 1
fi
if ! nvidia-smi >/dev/null 2>&1; then
    echo "  ERROR: nvidia-smi command failed. Verify driver/module first."
    exit 1
fi
print_resource_snapshot

# ---- 1. System packages ----
echo ""
echo "[1/8] Installing system packages..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev \
    nginx certbot python3-certbot-nginx \
    ffmpeg \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    curl jq
print_resource_snapshot

# ---- 2. Service account ----
echo ""
echo "[2/8] Creating service account '${SERVICE_USER}'..."
if ! id "${SERVICE_USER}" &>/dev/null; then
    useradd --system --no-create-home --shell /usr/sbin/nologin "${SERVICE_USER}"
    echo "  Created user: ${SERVICE_USER}"
else
    echo "  User already exists: ${SERVICE_USER}"
fi

# ---- 3. Virtual environment + dependencies ----
echo ""
echo "[3/8] Setting up Python virtual environment..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "  Created venv at ${VENV_DIR}"
fi
echo "  Upgrading pip/setuptools/wheel..."
"${VENV_DIR}/bin/pip" install --no-cache-dir --upgrade pip setuptools wheel
print_resource_snapshot

# Check if CUDA torch is already installed
if "${VENV_DIR}/bin/python" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  CUDA torch already installed, skipping GPU requirements."
else
    echo "  WARNING: CUDA torch not detected. Installing CUDA torch packages sequentially..."
    CUDA_REQ_FILE="${PROJECT_DIR}/requirements_gpu.txt"
    CUDA_INDEX_URL="$(awk '/^--index-url/{print $2; exit}' "${CUDA_REQ_FILE}")"
    mapfile -t CUDA_PACKAGES < <(grep -E '^[[:space:]]*[A-Za-z0-9._-]+==[^[:space:]]+$' "${CUDA_REQ_FILE}" | sed 's/^[[:space:]]*//')
    for pkg in "${CUDA_PACKAGES[@]}"; do
        echo "    Installing ${pkg}..."
        "${VENV_DIR}/bin/pip" install --no-cache-dir --index-url "${CUDA_INDEX_URL}" "${pkg}"
        print_resource_snapshot
    done
fi

echo "  Installing remaining dependencies..."
"${VENV_DIR}/bin/pip" install --no-cache-dir -r "${PROJECT_DIR}/requirements.txt"
print_resource_snapshot

# ---- 4. Florence-2 model cache ----
echo ""
echo "[4/8] Florence-2 model preload..."
if [ "${SKIP_MODEL_PRELOAD}" = "1" ]; then
    echo "  SKIP_MODEL_PRELOAD=1 -> skipping model pre-download for low-disk safety."
    echo "  Model will be downloaded on first inference request."
else
    echo "  Pre-downloading Florence-2 model (SKIP_MODEL_PRELOAD=0)..."
    "${VENV_DIR}/bin/python" -c "
from transformers import AutoModelForCausalLM, AutoProcessor
import os
cache_dir = '${PROJECT_DIR}/models'
os.makedirs(cache_dir, exist_ok=True)
model_name = 'microsoft/Florence-2-large'
print(f'  Downloading {model_name} to {cache_dir}...')
AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
print('  Model cached successfully.')
" || echo "  WARNING: Model download failed. Will download on first request."
fi
print_resource_snapshot

# ---- 5. Runtime directories ----
echo ""
echo "[5/8] Creating runtime directories..."
mkdir -p "${PROJECT_DIR}/data/events"
mkdir -p "${PROJECT_DIR}/data/clips"
mkdir -p "${PROJECT_DIR}/data/thumbnails"
mkdir -p "${PROJECT_DIR}/data/logs"
mkdir -p "${PROJECT_DIR}/data/media_archive"
mkdir -p "${PROJECT_DIR}/data/shadow_feedback"
mkdir -p "${PROJECT_DIR}/data/critic_models"
mkdir -p "${PROJECT_DIR}/data/rule_versions"
mkdir -p "${PROJECT_DIR}/data/lora_training"
mkdir -p "${PROJECT_DIR}/data/lora_output"
mkdir -p "${PROJECT_DIR}/models"
mkdir -p "${PROJECT_DIR}/models/hf/modules"
mkdir -p "${PROJECT_DIR}/models/hf/transformers"
print_resource_snapshot

# ---- 6. Environment file ----
echo ""
echo "[6/8] Setting up environment file..."
if [ ! -f "${PROJECT_DIR}/.env" ]; then
    cp "${PROJECT_DIR}/.env.aws" "${PROJECT_DIR}/.env"
    echo "  Copied .env.aws → .env"
    echo "  ⚠  IMPORTANT: Edit .env and set GEMINI_API_KEY!"
else
    echo "  .env already exists, skipping."
fi

# ---- 7. Permissions ----
echo ""
echo "[7/8] Setting permissions..."
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${PROJECT_DIR}/data"
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${PROJECT_DIR}/models"
chown "${SERVICE_USER}:${SERVICE_USER}" "${PROJECT_DIR}/.env" 2>/dev/null || true

# venv needs to be readable by service user
chmod -R o+rX "${VENV_DIR}"
print_resource_snapshot

# ---- 8. systemd + nginx ----
echo ""
echo "[8/8] Installing systemd services and nginx config..."

# systemd services
for svc in vlm-model vlm-db vlm-frontend; do
    cp "${DEPLOY_DIR}/${svc}.service" "/etc/systemd/system/${svc}.service"
    echo "  Installed: ${svc}.service"
done

systemctl daemon-reload
systemctl enable vlm-model vlm-db vlm-frontend

# nginx
if [ -f /etc/nginx/sites-available/vlm-cctv ] && grep -q "listen 443" /etc/nginx/sites-available/vlm-cctv; then
    echo "  Existing SSL nginx config detected. Skipping nginx.conf overwrite."
else
    cp "${DEPLOY_DIR}/nginx.conf" /etc/nginx/sites-available/vlm-cctv
fi
ln -sf /etc/nginx/sites-available/vlm-cctv /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

nginx -t
systemctl reload nginx

# Start services
echo ""
echo "Starting services..."
systemctl start vlm-db
sleep 1
systemctl start vlm-model
sleep 2
systemctl start vlm-frontend

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "  Service status:"
for svc in vlm-model vlm-db vlm-frontend; do
    status=$(systemctl is-active "${svc}" 2>/dev/null || echo "unknown")
    echo "    ${svc}: ${status}"
done
echo ""
echo "  Quick verification:"
echo "    curl -s http://127.0.0.1:8000/        # model_server health"
echo "    curl -s http://127.0.0.1:8001/        # db_server health"
echo "    curl -s http://127.0.0.1:8002/status  # frontend health"
echo "    curl -s http://localhost/              # nginx → frontend"
echo ""
echo "  GPU verification:"
echo "    nvidia-smi"
echo "    ${VENV_DIR}/bin/python -c \"import torch; print('CUDA:', torch.cuda.is_available())\""
echo ""
echo "  Logs:"
echo "    sudo journalctl -u vlm-model -f"
echo "    sudo journalctl -u vlm-db -f"
echo "    sudo journalctl -u vlm-frontend -f"
echo ""
echo "  Next steps:"
echo "    1. Edit .env: nano ${PROJECT_DIR}/.env  (set GEMINI_API_KEY)"
echo "    2. Restart: sudo systemctl restart vlm-model vlm-db vlm-frontend"
echo "    3. SSL: sudo certbot --nginx -d YOUR_DOMAIN --non-interactive --agree-tos -m YOUR_EMAIL"
echo "============================================================"
