#!/usr/bin/env bash
set -euo pipefail

# Lightweight 2-hour incident tracker for unexpected disconnects.
# Default behavior:
# - Duration: 7200 sec (2h)
# - Snapshot interval: 60 sec
# - Captures warning/error journald logs + periodic health/resource snapshots

PROJECT_DIR="/home/ubuntu/hio_intelligence_stream"
DURATION_SEC="${DURATION_SEC:-7200}"
INTERVAL_SEC="${INTERVAL_SEC:-60}"

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  DURATION_SEC="$1"
fi
if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
  INTERVAL_SEC="$2"
fi

RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
OUT_ROOT="${PROJECT_DIR}/data/incident_watch"
OUT_DIR="${OUT_ROOT}/${RUN_ID}"
LATEST_LINK="${OUT_ROOT}/latest"
mkdir -p "${OUT_DIR}"
ln -sfn "${OUT_DIR}" "${LATEST_LINK}"

SNAPSHOT_LOG="${OUT_DIR}/snapshots.log"
TRACKER_LOG="${OUT_DIR}/tracker.log"
WARN_LOG="${OUT_DIR}/journal_warn.log"
INCIDENT_LOG="${OUT_DIR}/incidents.log"
CONTEXT_LOG="${OUT_DIR}/incident_context.log"
PID_FILE="${OUT_DIR}/tracker.pid"

touch "${SNAPSHOT_LOG}" "${TRACKER_LOG}" "${WARN_LOG}" "${INCIDENT_LOG}" "${CONTEXT_LOG}"
echo "$$" > "${PID_FILE}"

log() {
  printf "%s %s\n" "$(date -u +'%Y-%m-%d %H:%M:%S UTC')" "$*" | tee -a "${TRACKER_LOG}" >/dev/null
}

capture_context() {
  {
    echo "================================================================"
    echo "CONTEXT_AT=$(date -u +'%Y-%m-%d %H:%M:%S UTC')"
    echo "--- systemctl status (short) ---"
    systemctl --no-pager --full status vlm-model vlm-db vlm-frontend nginx || true
    echo "--- recent warnings/errors (last 3 min) ---"
    sudo journalctl -u vlm-model -u vlm-db -u vlm-frontend -u nginx \
      --since "3 minutes ago" -p warning --no-pager || true
    echo "--- recent kernel alerts (last 10 min) ---"
    sudo journalctl -k --since "10 minutes ago" --no-pager \
      | grep -Ei "oom|out of memory|killed process|xid|nvrm|segfault|reset|hung task" || true
    echo "================================================================"
  } >> "${CONTEXT_LOG}" 2>&1
}

# Background log followers (warning/error only).
sudo journalctl -f -u vlm-model -u vlm-db -u vlm-frontend -u nginx \
  -p warning --output=short-iso >> "${WARN_LOG}" 2>&1 &
JOURNAL_PID=$!

cleanup() {
  kill "${JOURNAL_PID}" >/dev/null 2>&1 || true
  log "Tracker stopped."
}
trap cleanup EXIT INT TERM

END_TS=$(( $(date +%s) + DURATION_SEC ))
log "Tracker started. duration=${DURATION_SEC}s interval=${INTERVAL_SEC}s out=${OUT_DIR}"

while (( $(date +%s) < END_TS )); do
  TS="$(date -u +'%Y-%m-%d %H:%M:%S UTC')"

  STATUS_MODEL="$(systemctl is-active vlm-model 2>/dev/null || echo unknown)"
  STATUS_DB="$(systemctl is-active vlm-db 2>/dev/null || echo unknown)"
  STATUS_FE="$(systemctl is-active vlm-frontend 2>/dev/null || echo unknown)"
  STATUS_NGX="$(systemctl is-active nginx 2>/dev/null || echo unknown)"

  CODE_8000="$(curl -sS -m 3 -o /dev/null -w '%{http_code}' 'http://127.0.0.1:8000/' || echo ERR)"
  CODE_8001="$(curl -sS -m 3 -o /dev/null -w '%{http_code}' 'http://127.0.0.1:8001/' || echo ERR)"
  CODE_8002="$(curl -sS -m 3 -o /dev/null -w '%{http_code}' 'http://127.0.0.1:8002/monitor/adhoc' || echo ERR)"
  CODE_HTTPS="$(curl -sS -m 4 -o /dev/null -w '%{http_code}' 'https://dev-cctv.hio.ai.kr/monitor/adhoc' || echo ERR)"

  MEM_LINE="$(free -m | awk 'NR==2{printf("mem_used_mb=%s mem_total_mb=%s mem_avail_mb=%s", $3, $2, $7)}')"
  DISK_LINE="$(df -h / | awk 'NR==2{printf("disk_used=%s disk_avail=%s disk_use=%s", $3, $4, $5)}')"
  GPU_LINE="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 'gpu_na')"
  TOP_PROC="$(ps -eo pid,comm,%cpu,%mem --sort=-%cpu | head -n 6 | tr '\n' ';' | sed 's/;*$//')"

  {
    echo "TS=${TS} model=${STATUS_MODEL} db=${STATUS_DB} fe=${STATUS_FE} nginx=${STATUS_NGX} code8000=${CODE_8000} code8001=${CODE_8001} code8002=${CODE_8002} codeHttps=${CODE_HTTPS}"
    echo "  ${MEM_LINE}"
    echo "  ${DISK_LINE}"
    echo "  gpu=${GPU_LINE}"
    echo "  top=${TOP_PROC}"
  } >> "${SNAPSHOT_LOG}"

  # Incident trigger: unhealthy service or endpoint failure.
  if [[ "${STATUS_MODEL}" != "active" || "${STATUS_DB}" != "active" || "${STATUS_FE}" != "active" || "${STATUS_NGX}" != "active" \
     || "${CODE_8000}" == "ERR" || "${CODE_8001}" == "ERR" || "${CODE_8002}" == "ERR" || "${CODE_HTTPS}" == "ERR" \
     || "${CODE_8000}" == "000" || "${CODE_8001}" == "000" || "${CODE_8002}" == "000" || "${CODE_HTTPS}" == "000" ]]; then
    {
      echo "INCIDENT_AT=${TS} model=${STATUS_MODEL} db=${STATUS_DB} fe=${STATUS_FE} nginx=${STATUS_NGX} code8000=${CODE_8000} code8001=${CODE_8001} code8002=${CODE_8002} codeHttps=${CODE_HTTPS}"
    } >> "${INCIDENT_LOG}"
    log "Incident trigger detected. Capturing context."
    capture_context
  fi

  sleep "${INTERVAL_SEC}"
done

{
  echo "RUN_ID=${RUN_ID}"
  echo "START_UTC=$(head -n 1 "${TRACKER_LOG}" | sed 's/ Tracker started.*//')"
  echo "END_UTC=$(date -u +'%Y-%m-%d %H:%M:%S UTC')"
  echo "DURATION_SEC=${DURATION_SEC}"
  echo "INTERVAL_SEC=${INTERVAL_SEC}"
  echo "SNAPSHOT_COUNT=$(grep -c '^TS=' "${SNAPSHOT_LOG}" || true)"
  echo "INCIDENT_COUNT=$(wc -l < "${INCIDENT_LOG}" 2>/dev/null || echo 0)"
  echo "WARN_LINES=$(wc -l < "${WARN_LOG}" 2>/dev/null || echo 0)"
  echo "RECENT_INCIDENTS:"
  tail -n 10 "${INCIDENT_LOG}" 2>/dev/null || true
} > "${OUT_DIR}/summary.txt"

log "Tracker finished after ${DURATION_SEC}s."
