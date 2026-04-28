# HIO v3 Run Guide

이 문서는 현재 `hio_intelligence_stream_v3` 실행 기준만 정리합니다. 상세 아키텍처와 알고리즘 설명은 [README.md](README.md)를 기준으로 봅니다.

최신 런타임:

```text
YOLO26 pose/fire tier1 -> SigLIP2 frame/full-clip tier2 -> EpisodeManager -> minimal clips -> Gemini hard gate
```

활성 시나리오:

```text
cash
fire
violence
```

기본 candidate artifact:

```text
raw              = val_{event_id}.mp4 재사용
context_overlay  = cashier/exchange/staff zones + skeleton/SoM full-frame overlay 1개
skeleton_json    = skeleton summary JSON
```

기본 생성하지 않는 artifact:

```text
{event_id}_skeleton_overlay.mp4
{event_id}_cashier_zone_overlay.mp4
cashier crop video
```

## 1. Environment

PowerShell:

```powershell
cd E:\02_StayG\00_CCTV_Motion_Detection\github\hio_intelligence_stream_v3
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements_gpu.txt`는 기존 명령 호환용 shim이며 실제 의존성은 `requirements.txt` 하나에서 관리합니다. PyTorch CUDA가 별도 필요하면:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 2. Required Models

```text
models/yolo26s-pose.pt
models/yolov26_fire_detection_best.pt
```

optional:

```text
YOLO26_CASH_WEIGHTS=
```

custom cash detector가 없으면 비워둡니다.

## 3. Required .env Keys

민감한 값은 실제 `.env`에만 둡니다.

```env
GEMINI_MODEL=gemini-3.1-flash-lite-preview
GEMINI_MAX_CONCURRENT=2

HIO_V3_ENABLED=true
HIO_V3_PIPELINE_VERSION=v3-yolo26-tier1-siglip2classifier-episode-gemini
V3_SCENARIOS=cash,fire,violence

YOLO26_POSE_WEIGHTS=models/yolo26s-pose.pt
YOLO26_DETECT_WEIGHTS=
YOLO26_CASH_WEIGHTS=
YOLO26_FIRE_WEIGHTS=models/yolov26_fire_detection_best.pt
YOLO26_DEVICE=cuda
ALLOW_CPU_FALLBACK=false

V3_VALIDATION_CLIP_SEC=15
V3_GEMINI_ALWAYS_VALIDATE=true
V3_CLIP_ARTIFACT_MODE=minimal
V3_EPISODE_COOLDOWN_SEC=20
V3_EPISODE_MAX_GAP_SEC=6

V3_SEMANTIC_FILTER_ENABLED=true
V3_SEMANTIC_MODEL=google/siglip2-base-patch16-224
V3_SEMANTIC_DEVICE=cuda
V3_CASH_SIGLIP_CLIP_ENABLED=true
V3_CASH_SIGLIP_CLIP_FRAMES=12
V3_CASH_SIGLIP_CLIP_BATCH_SIZE=4
V3_CASH_SIGLIP_CLIP_WINDOW_SEC=15
V3_CASH_SIGLIP_CLIP_PEAK_WINDOW_SEC=5
V3_CASH_SIGLIP_CLIP_MIN_SCORE=0.50
V3_CASH_SIGLIP_CLIP_FRAME_POSITIVE=0.48
V3_CASH_SIGLIP_CLIP_MIN_POSITIVE_FRAMES=2
V3_CASH_SIGLIP_CLIP_COOLDOWN_SEC=2.0
V3_FIRE_SIGLIP_MIN_SCORE=0.52
V3_FIRE_NEUTRALIZER_THRESHOLD=0.58
```

## 4. Start

통합 실행:

```powershell
.\.venv\Scripts\python.exe start_local.py
```

개별 실행:

```powershell
.\.venv\Scripts\python.exe -m uvicorn model_server.main:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m uvicorn db_server.main:app --host 127.0.0.1 --port 8001
.\.venv\Scripts\python.exe -m uvicorn frontend_server.main:app --host 127.0.0.1 --port 8002
```

## 5. URLs

```text
http://127.0.0.1:8002/dashboard
http://127.0.0.1:8002/monitor/adhoc
http://127.0.0.1:8002/monitor/v3-proposal-logs
http://127.0.0.1:8002/monitor/gemini-logs
http://127.0.0.1:8002/monitor/labeling
http://127.0.0.1:8000/docs
http://127.0.0.1:8001/docs
```

## 6. First Camera Flow

1. `/monitor/adhoc` 접속
2. RTSP URL 입력
3. camera 추가
4. cashier zone + exchange_band polygon 그리기
5. `validation_clip_sec=15`, `event_cooldown_sec=20` 확인
6. start
7. `/monitor/v3-proposal-logs`에서 proposal 확인
8. `/monitor/gemini-logs`에서 raw/context overlay 확인
9. `/monitor/labeling`에서 TP/FP feedback 저장

## 7. Verification

컴파일:

```powershell
.\.venv\Scripts\python.exe -m compileall model_server db_server frontend_server tools
```

Smoke test:

```powershell
.\.venv\Scripts\python.exe tools\v3_smoke_test.py
```

정상 contract:

```json
{
  "required_present": ["context_overlay", "raw", "skeleton_json"],
  "missing": [],
  "forbidden_present": []
}
```

## 8. Failure Policy

```text
Gemini disabled/API error -> validation_error, TP 처리 안 함
YOLO CUDA unavailable    -> model_health error/degraded
SigLIP load failure      -> exponential backoff retry
CPU fallback             -> ALLOW_CPU_FALLBACK=true일 때만 허용
queue full               -> data/dead_letter/events_dropped.jsonl 기록
duplicate camera/scenario-> duplicate_pending drop
```

## 9. Monitor MJPEG Stutter

`/monitor/adhoc` preview가 초 단위로 멈췄다가 한 번에 점프하면 먼저 MJPEG 설정을 확인합니다.

```env
FRONTEND_MJPEG_FPS=3
FRONTEND_MJPEG_BURST_FPS=12
FRONTEND_MJPEG_QUALITY=55
FRONTEND_MJPEG_WIDTH=854
FRONTEND_MJPEG_DEDUP_FRAMES=true
FRONTEND_MJPEG_IDLE_PAUSE_SEC=5
```

조사 결과, local full-res `15fps` + quality `70` + width `0` 조합은 frontend -> model MJPEG proxy에서 JPEG encode와 browser decode 부하가 커져 preview가 밀릴 수 있습니다. 동일 frame dedup 시 idle heartbeat가 안 나가던 문제도 수정했으므로, 설정 변경 후에는 서버 재시작이 필요합니다.

## 10. Notes

서버가 이미 실행 중이면 파일 수정만으로 적용되지 않습니다. 새 코드 적용은 서버 재시작 후 반영됩니다.

`uvicorn --reload` 가 워커 재기동에서 멈추는 경우(긴 backround 스레드 때문)에는 launcher 자체를 죽였다 다시 띄워야 합니다.

## 11. Time / KST

- 모든 운영 시각은 KST 기준입니다. 자세한 규칙은 [README.md §18](README.md#18-time--locale-handling-kst).
- 날짜별 조회는 `GET /api/vlm/events/?kst_date=YYYYMMDD` 사용 (운영 화면이 사용하는 경로).
- 기존 `date=YYYYMMDD` 는 UTC 버킷 단위 raw 조회로 남아 있습니다.
- 신규 이벤트의 `at`, `saved_at` 은 `+09:00` suffix 가 붙은 tz-aware ISO 입니다. 화면 정렬·표시는 `event_id` 의 epoch ms 가 기준이라 과거 naive 데이터도 정확히 KST 로 노출됩니다.
