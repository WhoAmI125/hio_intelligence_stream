#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/ubuntu/hio_intelligence_stream"
LOG_DIR="${PROJECT_DIR}/data/recovery_logs"
ENV_FILE="${PROJECT_DIR}/.env"

DB_URL="${DB_URL:-http://127.0.0.1:8001/}"
MODEL_URL="${MODEL_URL:-http://127.0.0.1:8000/}"
MODEL_STATUS_URL="${MODEL_STATUS_URL:-http://127.0.0.1:8000/status}"
FRONTEND_URL="${FRONTEND_URL:-http://127.0.0.1:8002/monitor/adhoc}"
PUBLIC_URL="${PUBLIC_URL:-https://dev-cctv.hio.ai.kr/monitor/adhoc/}"

DB_TIMEOUT_SEC="${DB_TIMEOUT_SEC:-25}"
MODEL_TIMEOUT_SEC="${MODEL_TIMEOUT_SEC:-180}"
FRONTEND_TIMEOUT_SEC="${FRONTEND_TIMEOUT_SEC:-40}"
NGINX_TIMEOUT_SEC="${NGINX_TIMEOUT_SEC:-40}"
CAMERA_TIMEOUT_SEC="${CAMERA_TIMEOUT_SEC:-150}"
STEP_SLEEP_SEC="${STEP_SLEEP_SEC:-2}"

SERVICES=(vlm-db vlm-model vlm-frontend nginx)
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/recover_${TS}.log"

mkdir -p "${LOG_DIR}"

log() {
  printf "%s %s\n" "$(date +'%F %T %Z')" "$*" | tee -a "${LOG_FILE}"
}

json_get() {
  local expr="$1"
  python3 -c '
import json
import sys

expr = sys.argv[1]
data = json.load(sys.stdin)
parts = [p for p in expr.split(".") if p]
cur = data
for part in parts:
    if isinstance(cur, dict):
        cur = cur.get(part)
    else:
        cur = None
        break
if isinstance(cur, bool):
    print("true" if cur else "false")
elif cur is None:
    print("")
else:
    print(cur)
' "$expr"
}

http_code() {
  local url="$1"
  curl -k -sS -L -o /dev/null -w '%{http_code}' --max-time 8 "$url" 2>/dev/null || true
}

wait_for_http_code() {
  local url="$1"
  local timeout_sec="$2"
  local expected="${3:-200}"
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    local code
    code="$(http_code "$url")"
    if [[ "$code" == "$expected" ]]; then
      return 0
    fi
    if (( $(date +%s) - start_ts >= timeout_sec )); then
      log "Health check timeout url=${url} expected=${expected} got=${code:-ERR}"
      return 1
    fi
    sleep 2
  done
}

wait_for_model_ready() {
  local timeout_sec="$1"
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    local body
    body="$(curl -sS --max-time 8 "$MODEL_URL" 2>/dev/null || true)"
    if [[ -n "$body" ]]; then
      local status initialized
      status="$(printf '%s' "$body" | json_get "status" || true)"
      initialized="$(printf '%s' "$body" | json_get "florence_initialized" || true)"
      if [[ "$status" == "ok" && "$initialized" == "true" ]]; then
        return 0
      fi
    fi
    if (( $(date +%s) - start_ts >= timeout_sec )); then
      log "Model health timeout body=${body:-<empty>}"
      return 1
    fi
    sleep 3
  done
}

wait_for_camera_restore() {
  local timeout_sec="$1"
  if [[ ! -f "$ENV_FILE" ]]; then
    return 0
  fi

  local auto_restore
  auto_restore="$(awk -F= '/^AUTO_RESTORE_CAMERAS_ON_BOOT=/{print tolower($2)}' "$ENV_FILE" | tail -n1)"
  if [[ "$auto_restore" != "true" && "$auto_restore" != "1" && "$auto_restore" != "yes" ]]; then
    log "Camera restore check skipped: AUTO_RESTORE_CAMERAS_ON_BOOT is disabled."
    return 0
  fi

  local cameras_json
  cameras_json="$(curl -sS --max-time 8 http://127.0.0.1:8001/api/cameras 2>/dev/null || true)"
  if [[ -z "$cameras_json" ]]; then
    log "Camera restore check skipped: camera list unavailable."
    return 0
  fi

  local camera_ids
  camera_ids="$(python3 - <<'PY' "$cameras_json"
import json
import sys

try:
    data = json.loads(sys.argv[1])
except Exception:
    print("")
    raise SystemExit(0)

ids = []
for item in data.get("cameras", []):
    if isinstance(item, dict):
        cam = str(item.get("camera_id", "")).strip()
        if cam:
            ids.append(cam)
print("\n".join(ids))
PY
)"
  if [[ -z "$camera_ids" ]]; then
    log "Camera restore check skipped: no saved cameras."
    return 0
  fi

  local start_ts
  start_ts="$(date +%s)"
  while true; do
    local all_ok="true"
    while IFS= read -r camera_id; do
      [[ -z "$camera_id" ]] && continue
      local status_json
      status_json="$(curl -sS --get --data-urlencode "camera_id=${camera_id}" http://127.0.0.1:8000/api/vlm/status/ 2>/dev/null || true)"
      local running last_error
      running="$(printf '%s' "$status_json" | json_get "running" || true)"
      last_error="$(printf '%s' "$status_json" | json_get "last_error" || true)"
      if [[ "$running" != "true" ]]; then
        all_ok="false"
        break
      fi
      if [[ -n "$last_error" && "$last_error" != "Server shutting down" ]]; then
        all_ok="false"
        break
      fi
    done <<< "$camera_ids"

    if [[ "$all_ok" == "true" ]]; then
      return 0
    fi
    if (( $(date +%s) - start_ts >= timeout_sec )); then
      log "Camera restore timeout after ${timeout_sec}s"
      return 1
    fi
    sleep 4
  done
}

service_state() {
  systemctl is-active "$1" 2>/dev/null || true
}

show_status() {
  log "Service status snapshot:"
  for svc in "${SERVICES[@]}"; do
    log "  ${svc}=$(service_state "$svc")"
  done
  log "  db_code=$(http_code "$DB_URL") model_code=$(http_code "$MODEL_URL") frontend_code=$(http_code "$FRONTEND_URL") public_code=$(http_code "$PUBLIC_URL")"
}

stop_all() {
  log "Stopping services..."
  systemctl stop nginx vlm-frontend vlm-model vlm-db || true
  sleep 2
  pkill -f 'uvicorn .*model_server.main:app' || true
  pkill -f 'uvicorn .*db_server.main:app' || true
  pkill -f 'uvicorn .*frontend_server.main:app' || true
  sleep 2
  log "Services stopped."
}

start_all_safely() {
  log "Starting vlm-db..."
  systemctl start vlm-db
  wait_for_http_code "$DB_URL" "$DB_TIMEOUT_SEC" "200"
  log "vlm-db healthy."
  sleep "$STEP_SLEEP_SEC"

  log "Starting vlm-model..."
  systemctl start vlm-model
  wait_for_model_ready "$MODEL_TIMEOUT_SEC"
  log "vlm-model healthy and Florence initialized."
  sleep "$STEP_SLEEP_SEC"

  log "Waiting for saved camera restore..."
  wait_for_camera_restore "$CAMERA_TIMEOUT_SEC"
  log "Camera restore verification complete."
  sleep "$STEP_SLEEP_SEC"

  log "Starting vlm-frontend..."
  systemctl start vlm-frontend
  wait_for_http_code "$FRONTEND_URL" "$FRONTEND_TIMEOUT_SEC" "200"
  log "vlm-frontend healthy."
  sleep "$STEP_SLEEP_SEC"

  log "Starting nginx..."
  systemctl start nginx
  wait_for_http_code "$PUBLIC_URL" "$NGINX_TIMEOUT_SEC" "200"
  log "nginx/public URL healthy."
}

recover_stack() {
  local reason="$1"
  log "Recovery triggered. reason=${reason}"
  stop_all
  start_all_safely
  show_status
  log "Recovery completed."
}

check_public_health() {
  local code
  code="$(http_code "$PUBLIC_URL")"
  [[ "$code" == "200" ]]
}

usage() {
  cat <<'EOF'
Usage:
  vlm-safe-recover.sh status
  vlm-safe-recover.sh stop
  vlm-safe-recover.sh start
  vlm-safe-recover.sh recover
  vlm-safe-recover.sh recover-if-web-down
  vlm-safe-recover.sh boot-start
EOF
}

cmd="${1:-status}"

case "$cmd" in
  status)
    show_status
    ;;
  stop)
    stop_all
    show_status
    ;;
  start)
    start_all_safely
    show_status
    ;;
  recover)
    recover_stack "manual"
    ;;
  recover-if-web-down)
    if check_public_health; then
      log "Public URL healthy; no recovery needed."
      show_status
      exit 0
    fi
    recover_stack "public_url_down"
    ;;
  boot-start)
    recover_stack "boot_start"
    ;;
  *)
    usage
    exit 1
    ;;
esac
