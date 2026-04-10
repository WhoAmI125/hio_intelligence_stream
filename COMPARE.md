# 03_CCTV_Final vs hio_v2 — 전체 비교

## 아키텍처 개요

| | 03_CCTV_Final (A) | hio_v2 (B) |
|---|---|---|
| **철학** | "모든 것을 캡처하고, uncertainty gate로 라우팅" | "Simple is best: 경량 Tier 1 → 로컬 Tier 2 → 필수 Tier 3" |
| **LOC** | ~17,000+ (model/db/frontend 합산) | ~2,280 |
| **서버 구조** | 3개 (model:8000, db:8001, frontend:8002) | 2개 (model:8000, frontend:8002) |
| **Tier 1** | Florence-2 (캡션→키워드 매칭) | YOLO-Pose (wrist fallback) + CLIP zero-shot |
| **Tier 2** | Gemini 2.5 Flash (클라우드) | Qwen2.5-VL-3B (로컬, lazy load) |
| **Tier 3** | 없음 (Tier 2가 곧 클라우드) | Gemini 2.5 Flash (필수 통과, 독립 판단) |
| **Tier 3 결과** | — | Confirmed / Decline (둘 다 DB 저장) |

---

## RTSP / 스트리밍

| | A | B |
|---|---|---|
| **라이브러리** | cv2.VideoCapture | cv2.VideoCapture (PyAV에서 전환) |
| **프레임 버퍼** | Deque[(frame, mono_ts)] ~450+ | Deque[BufferFrame] maxlen=720 (~60초) |
| **해상도** | 원본 그대로 | **720p 강제 정규화** (1440p/4K → 720p) |
| **디스플레이** | 매 cap.read() → stream FPS | 매 cap.read() → stream FPS |
| **추론 샘플링** | base 1.5 FPS / burst 4 FPS | 1 FPS (normal) / **4 FPS (burst)** |
| **Burst Mode** | 이벤트 감지 시 3초간 4 FPS | **이벤트 감지 시 3초간 4 FPS** |
| **MJPEG** | 15 FPS 직접 스트림 | **10 FPS 직접 스트림** (프록시 우회) |
| **재연결** | Exponential backoff + jitter (1→30s) | Exponential backoff + jitter (1→30s) |
| **Stale 감지** | 2.5초 | **2.5초** + 40회 연속 실패 → 재연결 |
| **FFmpeg 옵션** | nobuffer, low_delay, 8s timeout | **nobuffer, low_delay, 8s timeout** |
| **클립 FPS** | 15 FPS | **8 FPS** (96프레임/12초) |
| **클립 인코딩** | cv2 mp4v | **ffmpeg libx264 H.264** (브라우저 재생) |

---

## GPU / 모델 관리

| | A | B |
|---|---|---|
| **Tier 1 VRAM** | ~3.5GB (Florence-2 large) | ~2.5GB (YOLO 1GB + CLIP 1.5GB) |
| **Tier 2 VRAM** | 0 (Gemini는 클라우드) | ~7-9GB (Qwen, **lazy load**) |
| **상시 VRAM** | ~5-6GB | **~4GB** (Tier 1 + CUDA) |
| **최대 VRAM** | ~5-6GB | ~13GB (첫 이벤트 후) |
| **FP16** | Florence config 설정 가능 | Qwen auto mixed precision |
| **병렬 추론** | ThreadPoolExecutor 3워커 | **ThreadPoolExecutor 2워커** (YOLO+CLIP) |

---

## Tier 1 감지 로직

| | A | B |
|---|---|---|
| **Cash 모델** | Florence-2 캡션 → 키워드 매칭 | **YOLO-Pose skeleton → wrist 거리** |
| **Cash 로직** | 캡션에서 "cash" 등 키워드 검색 | wrist 근접 < 250px (720p 기준 ~20% 프레임폭) |
| **부분 skeleton** | Florence는 캡션 기반 (무관) | **wrist만 보여도 감지** (hip>shoulder>wrist fallback) |
| **Zone 역할** | ROI 크롭 + 분석 영역 제한 (필수) | **보조** (staff/customer 분류 도움, 없어도 작동) |
| **Zone 없을 때** | 감지 불가 | **모든 2인 조합 전수 체크** |
| **Fire/Violence** | Florence 캡션 → 키워드/regex | **CLIP zero-shot** text-image 유사도 |
| **프레임당 지연** | ~10-50ms | **~12ms** (YOLO+CLIP 병렬) |
| **실시간 로그** | Florence 캡션 텍스트 | **wrist 거리 바 + CLIP % + Accumulator + zone 상태** |
| **VLM 시각 힌트** | 없음 | **Zone 테두리를 프레임에 그려서 Tier 2/3에 전달** |

---

## Tier 2 분석

| | A | B |
|---|---|---|
| **모델** | Gemini 2.5 Flash (클라우드 API) | **Qwen2.5-VL-3B (로컬 GPU, lazy load)** |
| **입력** | 키프레임 6-12장 + ROI 크롭 | 12프레임 (96프레임 중 샘플링) + zone 시각 힌트 |
| **프롬프트** | 통합 하드코드 (H1-H3 + S1-S3) | **시나리오별 5Q 프롬프트** (한국어) |
| **지연** | ~2-5초 (네트워크) | ~5-15초 (GPU, T4 기준) |
| **비용** | 매 이벤트 API → 월 $30-50 | **$0 (로컬)** |
| **라우팅** | 복잡한 policy_scores | **≥0.3 → Tier 3 (필수), <0.3 → 기각** |

---

## Tier 3 검증

| | A | B |
|---|---|---|
| **존재 여부** | Tier 2가 곧 클라우드 (별도 Tier 3 없음) | **Gemini 2.5 Flash (독립 Tier 3)** |
| **호출 조건** | — | conf ≥ 0.3이면 **무조건 호출** |
| **입력** | — | **풀 12프레임** (4장 키프레임 아님) |
| **판단 방식** | — | **1단계: 독립 판단 → 2단계: Qwen 참고** |
| **결과** | — | Confirmed / **Decline** (둘 다 DB 저장) |
| **알림** | 모든 이벤트 | **Confirmed만** 알림 발송 |

---

## Trigger / Episode 관리

| | A | B |
|---|---|---|
| **구조** | Episode 상태 머신 (IDLE→ACTIVE→VALIDATING→DONE) | Stateless 누적 카운터 |
| **Stability 점수** | label_history mode ratio (10프레임 이력) | 없음 |
| **시간 윈도우** | 30초 episode timeout | cash 2/30s, fire 2/15s, violence 2/10s |
| **쿨다운** | per-episode + per-type 60초 | per-(cam, scenario) 60초 |
| **Burst** | burst FPS 전환 | **burst FPS 전환 (3초 4FPS)** |

---

## Event Pipeline / 저장

| | A | B |
|---|---|---|
| **흐름** | Detect → Episode → Tier 2 → Router → Store → Alert | Tier 1 → Accumulate → Clip → Tier 2 → **Tier 3 (필수)** → Store → Alert |
| **클립 종류** | global + ROI (2종) | **full + zone** (원본 + VLM이 본 zone 그려진 버전) |
| **클립 FPS** | 15 FPS | **8 FPS H.264** (브라우저 재생 가능) |
| **썸네일** | 키프레임 기반 | 클립 중간 프레임 (full + zone) |
| **FALSE_ALARM** | 저장 여부 불명 | **DB 저장 (Decline으로 표시), 알림 안 보냄** |
| **DB 테이블** | events, episode_reviews, cameras, gemini_logs, worker_leases (5+) | events, cameras (2개) |
| **DB 접근** | 동기 sqlite3 | **비동기 aiosqlite** |
| **보관 정책** | LOCAL_RETENTION_DAYS (3일) | **LOCAL_RETENTION_DAYS (7일)** 매시간 자동 정리 |

---

## 프론트엔드

| | A | B |
|---|---|---|
| **페이지** | 6+ (adhoc_rtsp, clip_review, florence_logs, gemini_logs, shadow) | **6** (monitor, events, tier1_logs, tier2_logs, clip_review, dashboard) |
| **라이브 뷰** | MJPEG 15 FPS | **MJPEG 10 FPS 직접** (프록시 우회) |
| **ROI 에디터** | 캔버스 폴리곤 (cashier + drawer) | **캔버스 폴리곤** (cashier zone, 테두리만) |
| **실시간 추론** | Florence 캡션 텍스트 + Gemini 큐 | **YOLO wrist 거리 바 + CLIP % 프로그레스 + Accumulator** |
| **Clip 리뷰** | 업로드 → Florence + Gemini 평가 | **업로드 → 3시나리오 전체 Tier 1/2/3 자동 평가 (DB 저장)** |
| **Feedback UI** | Shadow Agent 리뷰 | 없음 |
| **이벤트 상세** | 테이블 뷰 | **디테일 모달 (썸네일 + Tier 2/3 전체 reason + 클립 링크)** |
| **Tier 3 표시** | Gemini verdict | **Confirmed (초록) / Decline (빨간)** |
| **미디어 서빙** | /media/ 정적 마운트 | **양쪽 서버 /media/ 정적 마운트** |

---

## API 엔드포인트

| | A | B |
|---|---|---|
| **Model Server** | 20+ | **16** (cameras CRUD/zones/tier1/history/snapshot/mjpeg/clip-review/clip-reviews/events/stats/ws) |
| **DB Server** | 10+ | 없음 (model server 통합) |
| **Frontend** | 페이지 + /api/vlm/* | 페이지 + /api/proxy/* + /media/ |

---

## 자동 개선 / 학습

| | A | B |
|---|---|---|
| **Shadow Agent** | 있음 | 없음 |
| **Critic Model** | DualHeadCritic | 없음 |
| **Rule Updater** | 동적 프롬프트 | 없음 (정적 프롬프트) |
| **LoRA** | DataCollector + train_lora.py | 없음 |
| **Feedback Loop** | episode review → critic 학습 | 없음 |

---

## 운영

| | A | B |
|---|---|---|
| **카메라 자동 복원** | DB 복원 | **DB 복원 (5초 딜레이)** |
| **Backoff 재연결** | 1s→30s + jitter | **1s→30s + jitter** |
| **데이터 보관** | 3일 | **7일 자동 정리** |
| **환경변수** | 260+ | ~15 |

---

## 성능

| 지표 | A | B |
|---|---|---|
| **Tier 1 지연** | ~10-50ms (Florence) | **~12ms** (YOLO+CLIP 병렬) |
| **Tier 2 지연** | ~2-5초 (Gemini API) | ~5-15초 (Qwen 로컬) |
| **Tier 2 비용** | ~$30-50/월 | **$0** (로컬) |
| **Tier 3** | 없음 | ~$0.30/일 |
| **상시 VRAM** | ~5-6GB | **~4GB** (lazy load) |
| **MJPEG FPS** | 15 | **10 (직접)** |
| **클립 FPS** | 15 | **8 (H.264)** |
| **해상도 정규화** | 없음 (원본) | **720p 강제** |

---

## 리소스 예산 (카메라 3대 기준)

| 리소스 | A | B |
|---|---|---|
| **RAM** | ~8GB (프레임 버퍼) | **~13GB** (720p ring 5.7GB + Qwen 4GB + OS 3GB) |
| **VRAM 상시** | ~5-6GB | ~4GB |
| **VRAM 최대** | ~5-6GB | ~13GB (Qwen 로드 시) |
| **CPU (4vCPU)** | ~50-60% | **~60-70%** (MJPEG + 추론 + ffmpeg) |
| **g4dn.xlarge** | 여유 | **3대까지 OK** (4대부터 RAM 빡빡) |
| **g4dn.2xlarge** | 여유 | **6대+ OK** |

---

## B에 있고 A에 없는 것

| 기능 | 설명 |
|---|---|
| 시나리오 특화 Tier 1 | YOLO-Pose (물리) + CLIP (시각) 분리 |
| Wrist fallback | 손만 보여도 감지 (hip>shoulder>wrist) |
| 로컬 VLM Tier 2 | Qwen2.5-VL-3B → API 비용 $0 |
| 독립 Tier 3 | Gemini가 풀 영상 보고 독립 판단, Qwen은 참고만 |
| Tier 3 필수 통과 | conf ≥ 0.3이면 무조건 Gemini 검증 |
| Decline 기록 | FALSE_ALARM도 DB 저장 (알림은 안 보냄) |
| Lazy Loading | 첫 이벤트까지 Qwen 안 로드 (VRAM 4GB 운영) |
| Zone 시각 힌트 | VLM 프레임에 zone 테두리 그려서 전달 |
| 720p 정규화 | 모든 카메라 720p 강제 (OOM 방지) |
| H.264 클립 | ffmpeg libx264 → 브라우저 직접 재생 |
| Clip Review | MP4 업로드 → 3시나리오 전체 자동 평가 (DB 저장, 이력 유지) |
| Qwen Lock | asyncio.Lock으로 GPU 동시 호출 방지 (직렬화) |
| Decline 기록 | FALSE_ALARM도 DB 저장 + UI 표시 (알림은 안 보냄) |
| 7일 보관 정책 | 매시간 자동 정리 |

## A에 있고 B에 없는 것

| 기능 | 중요도 | 비고 |
|---|---|---|
| Shadow Agent | 중 | v3 예정 |
| Critic Model | 중 | v3 예정 |
| LoRA Fine-tuning | 중 | 운영 데이터 축적 후 |
| Rule Updater | 낮 | 프롬프트 안정화 후 |
| Episode 상태 머신 | 중 | 현재 Accumulator로 대체 |
| Feedback UI | 중 | v3 예정 |
| Motion Gate | 낮 | Tier 1이 이미 경량 |
| NVENC HW 가속 | 낮 | ffmpeg libx264로 충분 |
