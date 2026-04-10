#!/usr/bin/env bash
# HiO v2 Safe Recovery Script
# Usage: sudo deploy/hio-safe-recover.sh [status|stop|start|recover|boot-start]
set -euo pipefail

MODEL_URL="http://127.0.0.1:8000"
FRONTEND_URL="http://127.0.0.1:8002"
MODEL_TIMEOUT=180
FRONTEND_TIMEOUT=40

wait_for_http() {
    local url="$1" timeout="$2" label="$3"
    for i in $(seq 1 "$timeout"); do
        if curl -sf "$url" &>/dev/null; then
            echo "  ✓ $label ready (${i}s)"
            return 0
        fi
        sleep 1
    done
    echo "  ✗ $label timeout after ${timeout}s"
    return 1
}

cmd_status() {
    echo "═══ HiO v2 Service Status ═══"
    for svc in hio-model hio-frontend nginx; do
        state=$(systemctl is-active "$svc" 2>/dev/null || echo "inactive")
        printf "  %-20s %s\n" "$svc" "$state"
    done
    echo ""
    echo "═══ Endpoints ═══"
    for url in "$MODEL_URL/api/status" "$FRONTEND_URL/monitor"; do
        code=$(curl -sf -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || echo "ERR")
        printf "  %-45s %s\n" "$url" "$code"
    done
    echo ""
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader 2>/dev/null || echo "  GPU: N/A"
}

cmd_stop() {
    echo "▸ Stopping services..."
    systemctl stop hio-frontend 2>/dev/null || true
    systemctl stop hio-model 2>/dev/null || true
    sleep 2
    echo "  ✓ All services stopped"
}

cmd_start() {
    echo "▸ Starting services (safe order)..."

    echo "  ⏳ Starting hio-model..."
    systemctl start hio-model
    wait_for_http "$MODEL_URL/api/status" "$MODEL_TIMEOUT" "Model server"

    echo "  ⏳ Starting hio-frontend..."
    systemctl start hio-frontend
    wait_for_http "$FRONTEND_URL/monitor" "$FRONTEND_TIMEOUT" "Frontend"

    systemctl restart nginx
    echo "  ✓ nginx restarted"
    echo ""
    echo "  ✓ All services running"
}

cmd_recover() {
    echo "═══ HiO v2 Full Recovery ═══"
    cmd_stop
    sleep 2
    cmd_start
}

cmd_boot_start() {
    echo "═══ HiO v2 Boot Recovery ═══"
    sleep 5  # wait for GPU driver
    cmd_start
}

case "${1:-status}" in
    status)     cmd_status ;;
    stop)       cmd_stop ;;
    start)      cmd_start ;;
    recover)    cmd_recover ;;
    boot-start) cmd_boot_start ;;
    *)
        echo "Usage: $0 [status|stop|start|recover|boot-start]"
        exit 1
        ;;
esac
