# HiO Intelligence Stream v2

**3-Tier CCTV 이상 탐지 시스템**

YOLO-Pose (현금거래 감지) + CLIP (화재/폭력 감지) → Qwen2.5-VL-3B (12초 영상 분석) → Gemini 2.5 Flash (풀 영상 독립 판정)

---

## 설계 철학

> "Simple is best. Sometimes throw away and reconstruct again from start."

v1(Florence-2 캡션 → 키워드 매칭)의 근본적 실패 원인은 **간접 판단 구조**였다.
v2는 이 교훈에서 출발한다:

- **감지(Detection)와 이해(Understanding)를 분리.** 감지는 빠르고 단순한 CV 모델이, 이해는 VLM이 담당
- **시나리오별 최적 감지 방식.** 현금거래는 사람 간 물리적 근접성(YOLO-Pose), 화재/폭력은 시각 패턴(CLIP)
- **영상(Video) 분석.** 단일 프레임이 아닌 12초 클립을 VLM에 넘겨 시간적 맥락 이해
- **3단 Cascade 필수 통과.** Tier 2에서 기각 아닌 이상 반드시 Tier 3 검증

---

## 전체 아키텍처

```
RTSP Cameras (720p~1440p, 25-30fps)
│
├──→ cv2.VideoCapture (H.264 자동 디코딩, 깨짐 없음)
│    → 720p 강제 리사이즈 (모든 카메라 정규화)
│
├──→ display_frame: 매 cap.read() (~stream FPS) — UI 스냅샷
├──→ ring_buffer: ~12 FPS decoded frames (maxlen=720) — 클립 추출
└──→ inference_frame: 1 FPS (normal) / 4 FPS (burst) — Tier 1
      │
      ▼
┌─────────────────────────────────────────────────────┐
│          Tier 1: Task-Specific Triggers              │
│          GPU ~2.5GB, ~12ms/프레임 (병렬)             │
│                                                      │
│  Cash Trigger (YOLO-Pose)          ~4ms              │
│  ├ 카메라별 trigger_mode 선택 가능 (UI 드롭다운)      │
│  │                                                    │
│  ├ [Mode 1: wrist] 손목 근접 (기본)                   │
│  │  ├ Zone 있으면: 안=staff, 밖=customer 분류          │
│  │  ├ Zone 없으면: 모든 2인 조합 전수 체크             │
│  │  └ 손목 거리 < threshold → Cash 플래그              │
│  │                                                    │
│  ├ [Mode 2: zone_intrusion] 캐셔존 침입               │
│  │  ├ 캐셔가 카운터 뒤에 가려진 카메라용 (예: 일산)    │
│  │  ├ 사람 body center가 캐셔존 밖에 있는데            │
│  │  └ 손목/팔꿈치가 캐셔존 안으로 들어오면 → 플래그    │
│  │                                                    │
│  ※ Zone은 보조(wrist) / 필수(zone_intrusion)          │
│                                                      │
│  Fire/Violence Trigger (CLIP ViT-L/14)    ~12ms      │
│  ├ text prompts vs 프레임 유사도                      │
│  └ positive score > threshold? → 플래그               │
│                                                      │
│  ※ YOLO FP32 + ThreadPoolExecutor 4워커 병렬           │
│  ※ Trigger 감지 시 Burst Mode (3초간 4 FPS)          │
└──────────────────────┬───────────────────────────────┘
                       │ 플래그 (~5-15%)
                       ▼
            ┌──────────────────────┐
            │  Trigger Accumulator  │
            │  cash 2/30s           │
            │  fire 2/15s           │
            │  violence 2/10s       │
            └──────────┬───────────┘
                       │ 누적 충족
                       ▼
              Ring Buffer → 12초 클립 추출 (pre 6s + post 6s)
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│         Phase 1: 클립 저장 (Lock 없음, ~7초)         │
│                                                      │
│  1. ring buffer → 8fps ~74프레임 추출                 │
│  2. ffmpeg → _full.mp4 (H.264, 12초 영상)             │
│  3. Cash + Zone 있으면:                               │
│     전체 프레임에 노란 테두리 → _zone.mp4 (12초 영상)  │
│  4. 썸네일 2종 (full + zone)                          │
└──────────────────────┬───────────────────────────────┘
                       │ MP4 파일 경로
                       ▼
┌─────────────────────────────────────────────────────┐
│     Phase 2: Qwen2.5-VL-3B 4-bit — 증거 추출기         │
│     MP4에서 12프레임 샘플 → 시나리오별 구조화 분석       │
│                                                      │
│  Cash: 6개 증거 슬롯 추출 (판정은 하지 않음)            │
│    cash_like_object / hand_to_hand_transfer            │
│    counter_context / staff_customer_roles_clear         │
│    drawer_or_counting / non_cash_object                 │
│  Fire/Violence: 5질문 yes_count (기존 방식 유지)        │
│                                                      │
│  Cash 라우팅:                                          │
│    non_cash 명확 → dismiss                              │
│    cash_like + hand_transfer 둘 다 없음 → dismiss       │
│    그 외 → Tier 3 전달                                  │
│  Fire/Violence: confidence ≥ 0.3 → Tier 3              │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│     Phase 3: Gemini 2.5 Flash — Hard Gate 판정기     │
│                                                      │
│  동일 MP4 파일을 Gemini에 업로드                       │
│  Qwen 증거는 "unreliable hints"로만 참고               │
│                                                      │
│  Cash: 3 Hard Gates (완화 기준)                        │
│    H1. 현금일 수 있는 물체 전달 (얇은 물체 포함)        │
│    H2. 소유권 이전 (손→손 또는 카운터 위 전달)          │
│    H3. 거래 맥락 (카운터에서 대면 상호작용)             │
│    + 3 Soft Strong (서랍/세기/거스름돈)                 │
│                                                      │
│  판정: 서버단 강제 (Gemini 답변과 무관하게 적용)        │
│    3/3 Hard + Soft ≥ 1  → CONFIRMED (알림)             │
│    3/3 Hard + Soft = 0  → UNCERTAIN (로그만)            │
│    2/3 Hard             → UNCERTAIN (로그만)            │
│    0~1/3 Hard           → FALSE_ALARM                  │
│    non-cash 명확        → FALSE_ALARM                  │
│                                                      │
│  Fire/Violence: CONFIRMED / FALSE_ALARM (단순)         │
│    빈 응답/에러 시 → CONFIRMED (safety first)           │
│                                                      │
│  + SQLite 이벤트 기록 (모든 verdict 저장)               │
│  + 알림: CONFIRMED만 발송                              │
│  + 쿨다운 60초 / 7일 자동 보관                          │
└─────────────────────────────────────────────────────┘
```

---

## 스트리밍 아키텍처

### cv2.VideoCapture 기반

PyAV에서 발생한 H.264 부분 디코딩 → 화면 깨짐 문제를 해결.
cv2.VideoCapture는 내부적으로 모든 H.264 프레임을 자동 디코딩.

### 720p 강제 정규화

```python
if w > 1280 or h > 720:
    scale = min(1280 / w, 720 / h)
    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
```

1440p/4K 카메라도 720p로 정규화 → RAM/VRAM 예산 통제.
720p 12프레임이면 Qwen이 정상 처리. 1440p면 47GB VRAM 요구 → OOM.

### Burst Mode

Tier 1 trigger 감지 시 `reader.trigger_burst()` → **3초간 inference 4 FPS**.
이벤트 전후 프레임 밀도 향상 → 클립 품질 개선.

### Stale 감지 + Exponential Backoff

- 2.5초간 새 프레임 없으면 stale → None 반환
- 40회 연속 read 실패 → 자동 재연결
- 재연결: 1s → 2s → 4s → ... → 30s + jitter

---

## Cash Trigger: Skeleton 기반 손목 근접 감지

### Keypoint 감지 전략

카메라 각도에 따라 사람의 전신이 안 보일 수 있다 (예: 일산 — 캐셔가 손만 보임).
이를 해결하기 위해 **wrist(손목) 하나만 보여도 사람으로 인식**한다.

```
사람 판별: wrist가 1개 이상 보이면 유효한 사람
위치 판별: hip > shoulder > wrist 순 fallback
          → hip 안 보이면 shoulder로, shoulder도 없으면 wrist로 zone 판별
```

| 보이는 부위 | 이전 | 이후 |
|---|---|---|
| 전신 (hip 포함) | 감지됨 | 감지됨 |
| 상반신만 (shoulder) | **미감지** | 감지됨 (shoulder로 위치) |
| **손만** (wrist) | **미감지** | **감지됨** (wrist로 위치) |

### Wrist Threshold (250px, 720p 기준)

```
720p 프레임 (1280x720)
├── 250px = 프레임 폭의 ~20%
├── 카운터에서 손 맞닿는 거리: ~200-400px
├── 떨어져 있는 사람: 500px+
└── .env에서 조정: CASH_WRIST_THRESHOLD_PX=300
```

### Trigger Mode 선택 (카메라별)

UI 설정 모달에서 카메라마다 trigger mode를 선택 가능. DB에 저장되어 서버 재시작 후에도 유지.

| Mode | 설명 | 적합한 카메라 |
|------|------|-------------|
| **wrist** (기본) | 두 사람의 손목 거리 < threshold | 캐셔와 고객이 모두 보이는 카메라 (예: 금촌) |
| **zone_intrusion** | 캐셔존 밖 사람의 손목/팔꿈치가 캐셔존 안으로 침입 | 캐셔가 카운터 뒤에 가려진 카메라 (예: 일산) |

### Cashier Zone

**wrist 모드**: Zone은 보조. staff/customer 분류에 활용. 없어도 모든 2인 조합 전수 체크로 동작.

**zone_intrusion 모드**: Zone 필수. 사람의 body center(hip>shoulder>wrist)로 안/밖 판별 후, 밖에 있는 사람의 wrist/elbow가 zone 안으로 들어오면 트리거.

```
zone_intrusion 동작 원리:
┌──────────────────────────────┐
│ 캐셔존 (안쪽)                  │
│   [캐셔 — 카운터 뒤 가려짐]    │
│ ─────── 카운터 ───────────── │
│         ← 손목이 여기 넘어옴   │  ← 트리거!
│   [고객] body center는 밖     │
└──────────────────────────────┘
```

### VLM 시각 힌트

cashier zone 설정 시, Tier 2/3 MP4에 **노란색 폴리곤 테두리 + "CASHIER ZONE" 라벨**.
VLM이 "이 영역 안에서 캐셔와 고객 사이 거래가 이루어지고 있다"고 추론 가능.

### 카메라별 Graceful Degradation

| 카메라 상황 | Cash (wrist) | Cash (zone_intrusion) | Fire/Violence |
|---|---|---|---|
| 캐셔+고객 모두 보임 | zone staff/customer 분류 | 손목 zone 침입 감지 | CLIP |
| 캐셔 가려짐 (일산) | wrist fallback (불안정) | **손목/팔꿈치 침입 감지** | CLIP |
| 사람 1명만 보임 | 불가 (2명 필요) | 밖→안 침입 가능 시 감지 | CLIP |
| 사람 0명 | 불가 | 불가 | CLIP |

---

## Tier 3: Gemini 독립 판단

### 이전 (v1)
- Qwen 고신뢰(≥0.7) → Tier 3 스킵
- 4장 키프레임만 전송
- Gemini에 "Qwen이 이렇게 판단했는데 맞나?" 질문

### 현재 (v2)
- Tier 2 기각(conf < 0.3) 아닌 이상 **무조건 Tier 3 통과**
- **풀 12프레임 전송** (12초 전체 영상)
- **1단계: Gemini 독립 판단** ("이 영상에서 현금거래가 보이는가?")
- **2단계: Qwen 분석 참고** (보조 자료로만 제공)
- 최종: CONFIRMED / FALSE_ALARM

---

## VRAM 전략: Lazy Loading + 양자화

| 단계 | 로드 모델 | VRAM |
|---|---|---|
| 서버 시작 | YOLO (FP32) + CLIP + CUDA | ~4.0GB |
| 평소 운영 | ↑ 동일 | ~4.0GB |
| Qwen 모델 로드 | + Qwen2.5-VL-3B (4-bit NF4) | ~6.8GB |
| Qwen 추론 피크 | + KV 캐시 + 입력 텐서 | ~12-13GB |

> **알려진 이슈:** Qwen 비디오 입력 시 720p 원본 기준 토큰 생성으로
> 추론 피크 VRAM이 높음. 프레임 수동 리사이즈로 개선 예정.

### 양자화 적용 내역
| 모델 | 이전 | 이후 | 절감 | 적용 방식 |
|------|------|------|------|----------|
| YOLO-Pose | FP32 (~1GB) | **FP32 유지** | — | FP16은 CLIP 동시 실행 시 CUDA assert 발생 → 비활성화 |
| Qwen2.5-VL-3B | auto/BF16 (~7-9GB) | **4-bit NF4** (~2.8GB) | -4~6GB, 정확도 1-3% 감소 | `BitsAndBytesConfig(load_in_4bit=True)` |
| CLIP ViT-L/14 | FP32 (~1.5GB) | 변경 없음 | — | — |

> **참고:** YOLO FP16(`half=True`)은 ThreadPoolExecutor에서 CLIP과 동시 GPU 접근 시
> CUDA device-side assert를 유발하여 전체 GPU 컨텍스트가 오염됨. FP32로 유지.

---

## 3-Phase Event Pipeline + Qwen Lock

이벤트 처리를 3단계로 분리하여 GPU lock 점유 시간을 최소화.

```
Phase 1: 클립 저장 (~7초) ── GPU Lock 없음
  └── ring buffer → 12초 클립 추출 (pre 6s + post 6s 대기)
  └── ffmpeg → _full.mp4 + _zone.mp4 (cash일 때) + 썸네일

Phase 2: Qwen 추론 (수 초) ── _qwen_lock 보유
  └── MP4 파일에서 12프레임 샘플링 → 4-bit NF4 추론 → JSON
  └── route_result → tier3 / dismiss

Phase 3: Tier 3 + 알림 ── GPU Lock 없음
  └── 동일 MP4 파일을 Gemini에 업로드
  └── DB insert + 알림 (CONFIRMED만)
  └── DB insert + 알림 발송
```

**이전**: lock이 전체 파이프라인(클립 추출 6초 + Qwen + Gemini)을 감싸서 이벤트당 20~50초 점유
**현재**: lock은 Qwen GPU 추론만 감싸서 이벤트당 수 초 점유

| 대상 | Lock 적용 | 영향 |
|---|---|---|
| Tier 1 (YOLO+CLIP) | **없음** | 항상 실시간 |
| 클립 추출 (Phase 1) | **없음** | 병렬 가능 |
| Tier 2 (Qwen) | **Lock 직렬화** | GPU 순차 처리 |
| Tier 3 (Gemini) | **없음** | Lock 해제 후 실행 |
| Clip Review Tier 2 | **Lock 직렬화** | 3시나리오 Qwen만 Lock |
| Clip Review Tier 3 | **없음** | Lock 해제 후 실행 |
| 스트리밍/UI | **없음** | 항상 실시간 |

대기(waiting)지 거부(reject)가 아님. 순서대로 전부 처리됨.

---

## 프로젝트 구조

```
hio_v2/                                      ~2,400 LOC
├── main.py                   (530)  # FastAPI + 추론 루프 + API + clip review + qwen lock
├── config.py                  (84)  # 환경변수 + YAML
│
├── tier1/                             # Tier 1: 경량 트리거
│   ├── cash_trigger.py       (200)  # YOLO-Pose wrist + zone_intrusion 2모드
│   ├── clip_trigger.py        (87)  # CLIP ViT-L/14 zero-shot
│   └── trigger_accumulator.py (79)  # 시간 윈도우 누적
│
├── tier2/                             # Tier 2: 증거 추출기 (판정 안 함)
│   ├── video_analyzer.py     (180)  # Qwen2.5-VL-3B 4-bit, 증거 슬롯 + JSON 파싱
│   └── agent_prompts.py      (110)  # cash=6슬롯 증거, fire/violence=5Q
│
├── tier3/                             # Tier 3: Hard Gate 최종 판정
│   └── tier3_verifier.py     (260)  # Gemini Hard Gate 3단 + UNCERTAIN + 서버 강제
│
├── stream/                            # RTSP 스트림
│   ├── stream_reader.py      (200)  # cv2 + 720p + burst + MJPEG 10fps
│   └── clip_extractor.py      (75)  # ring buffer → H.264 8fps (ffmpeg)
│
├── event/                             # 이벤트 파이프라인
│   ├── event_pipeline.py     (150)  # finalize_event: Tier3 + 저장 + 알림 (Phase 3)
│   └── alert_sender.py        (87)  # Webhook / Slack / LINE
│
├── storage/                           # 저장소
│   ├── db.py                 (280)  # SQLite WAL 싱글톤 커넥션 + 7일 보관 + busy_timeout
│   └── s3_uploader.py         (29)  # 선택적 S3
│
├── frontend_server/                   # UI (port 8002)
│   ├── main.py               (230)  # FastAPI + Jinja2 + 프록시 + /media
│   └── templates/hio_v2/
│       ├── base_public.html           # 사이드바 레이아웃
│       ├── monitor.html               # 카메라 그리드 + 존 에디터 + Skeleton 오버레이 + Tier 1 라이브
│       ├── events.html                # 이벤트 로그 (필터 + 디테일 모달 + 미디어)
│       ├── tier1_logs.html            # 트리거 이력 (임계값 초과만)
│       ├── tier2_logs.html            # Tier 2 + Tier 3 전체 reason
│       └── clip_review.html           # MP4 업로드 → 3시나리오 전체 평가
│
├── configs/cameras.yaml
├── .gitignore                         # .env, cameras.yaml, db/, data/clips 등 제외
├── deploy/                            # Dockerfile, systemd, setup.sh
├── db/                                # DB (미디어 경로 밖, /media 노출 방지)
│   └── events.db                      # SQLite WAL (events + cameras + clip_reviews)
├── data/                              # 런타임 미디어 (/media로 서빙)
│   ├── clips/YYYYMMDD/               # H.264 MP4 (_full + _zone)
│   └── thumbnails/YYYYMMDD/          # JPG (full + zone)
├── COMPARE.md                         # 03_CCTV_Final 대비 비교표
└── README.md
```

---

## 프론트엔드

### CCTV Monitor (`/monitor`)
- 카메라 그리드 (스냅샷 5초 폴링)
- 설정 모달:
  - MJPEG 라이브 (10fps 직접)
  - 🦴 **Skeleton 토글** — YOLO 키포인트 오버레이 (1.5초 폴링, 추가 GPU 0)
    - 초록=staff, 주황=customer, 손목 강조, 스켈레톤 연결선
    - 캐셔존 + trigger_mode + threshold 표시
  - 폴리곤 존 에디터 + **Trigger Mode 선택** (같은 줄에서 Apply)
  - Tier 1 라이브 (Cash/Fire/Violence) + Accumulator

### Events (`/events`)
- 필터 (카메라, 시나리오, 페이지당)
- 통계 카드 (Total/Cash/Fire/Violence)
- Detail 모달: 전체 Tier 2/3 reason + 썸네일 미리보기 + Full/Zone 클립 링크

### Tier 1 Logs (`/tier1-logs`)
- **임계값 초과 트리거만** 기록
- 카메라/시나리오 필터 + 통계

### Tier 2 Logs (`/tier2-logs`)
- Tier 2 전체 reason + Tier 3 verdict/reason 동시 표시
- Routing (Tier 3 / Dismissed)

### Clip Review (`/clip-review`)
- MP4 드래그&드롭 → **3개 시나리오 전부 자동 평가**
- 시나리오별 카드: Tier 2 (Detected/Confidence/Yes Count/Reason) + Tier 3 (Verdict/Reason)
- Raw JSON 출력
- **결과 DB 저장** — 페이지 떠났다 와도 이력 유지 (`clip_reviews` 테이블)

---

## API (모델 서버 port 8000)

| Method | Endpoint | 설명 |
|---|---|---|
| GET | `/api/status` | 시스템 상태 |
| GET | `/api/cameras` | 카메라 목록 |
| POST | `/api/cameras/{id}/start` | 카메라 추가 + DB 저장 |
| POST | `/api/cameras/{id}/stop` | 중지 (DB 유지) |
| DELETE | `/api/cameras/{id}` | 완전 삭제 |
| PUT | `/api/cameras/{id}/zones` | Cashier Zone + Trigger Mode + Wrist Threshold 저장 |
| GET | `/api/cameras/{id}/tier1` | 실시간 Tier 1 (디버그 포함) |
| GET | `/api/cameras/{id}/tier1/history` | 추론 이력 (200건) |
| GET | `/api/cameras/{id}/snapshot` | JPEG 스냅샷 |
| GET | `/api/cameras/{id}/skeleton` | YOLO 스켈레톤 오버레이 스냅샷 (추가 GPU 0) |
| GET | `/api/cameras/{id}/mjpeg` | MJPEG 스트림 (10 FPS) |
| POST | `/api/clip-review` | MP4 → 3시나리오 Tier 1/2/3 평가 (DB 저장) |
| GET | `/api/clip-reviews` | Clip review 이력 조회 |
| GET | `/api/events` | 이벤트 조회 |
| GET | `/api/stats` | 시나리오별 통계 |
| WS | `/ws/events` | 실시간 이벤트 |

---

## 핵심 소프트웨어 패턴

### 1. cv2.VideoCapture + 720p 정규화
H.264 자동 디코딩 → 깨짐 없음. 1440p/4K → 720p 강제.

### 2. Cashier Zone = 보조
트리거: zone 없이도 모든 2인 조합 체크. 분석: zone 테두리를 프레임에 그려 VLM 시각 힌트.

### 3. Lazy Pipeline — 클립 저장 후 VLM 분석
클립 MP4 파일을 먼저 저장하고, 저장된 파일을 Qwen/Gemini에 전달.
프레임 메모리 전달 대신 파일 경로 1개 → VRAM 절약 + 임시 JPEG 불필요.

### 4. Tier 3 독립 판단
Gemini가 12초 MP4 영상 전체를 보고 독립 판단 → Qwen 분석은 참고만.

### 5. Tier 3 필수 통과
conf < 0.3 기각 외에는 무조건 Gemini 검증.

### 6. Qwen Lazy Loading + 3-Phase Lock
첫 이벤트까지 Tier 1만 로드 (~4GB). Qwen은 필요 시 로드.
`_qwen_lock`은 Qwen GPU 추론만 직렬화 (클립 추출/Tier 3는 lock 밖).

### 7. Burst Mode
Tier 1 trigger → 3초간 4 FPS 추론.

### 8. H.264 클립 (ffmpeg)
MJPG temp → ffmpeg libx264 → 브라우저 직접 재생.

### 9. 2종 클립 저장 — VLM도 동일 파일 사용
`_full.mp4` (원본 12초 8fps) + `_zone.mp4` (원본 + 캐셔존 테두리 12초 8fps).
Qwen과 Gemini는 `_zone.mp4`(cash) 또는 `_full.mp4`(fire/violence) 파일을 직접 입력받아 분석.

### 10. 7일 데이터 보관
매시간 `cleanup_old_data()`. 오래된 DB rows + clips + thumbnails 자동 삭제.

### 11. 카메라 자동 복원
SQLite cameras 테이블 저장 → 서버 재시작 시 5초 후 자동 복원.

---

## 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `GEMINI_API_KEY` | (필수) | Tier 3 API 키 |
| `YOLO_MODEL` | `yolo26s-pose.pt` | YOLO Pose 모델 |
| `CLIP_MODEL` | `ViT-L/14` | CLIP 모델 |
| `QWEN_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` | Qwen VLM |
| `CASH_WRIST_THRESHOLD_PX` | `250` | 손목 근접 임계값 (720p 기준 ~20% 프레임폭) |
| `CLIP_FIRE_THRESHOLD` | `0.25` | 화재 감지 임계값 |
| `CLIP_VIOLENCE_THRESHOLD` | `0.30` | 폭력 감지 임계값 |
| `QWEN_CONFIDENCE_LOW` | `0.3` | 이하 기각, 이상 Tier 3 통과 |
| `SAMPLE_FPS` | `1.0` | 추론 샘플링 (burst 시 4.0) |
| `TRIGGER_COOLDOWN_SECONDS` | `60` | 이벤트 쿨다운 |
| `LOCAL_RETENTION_DAYS` | `7` | 데이터 보관 기간 |
| `AUTO_RESTORE_DELAY` | `5` | 카메라 복원 대기 (초) |

---

## 스트리밍 성능 개선 이력

### 문제 → 원인 → 해결 (v1 → v2 순)

| # | 문제 | 원인 | 해결 |
|---|---|---|---|
| 1 | 화면 깨짐/찢어짐 | PyAV 부분 디코딩 → H.264 참조 프레임 손실 | cv2.VideoCapture로 전환 |
| 2 | 10초에 1초씩 전진 | 디코딩 1 FPS, MJPEG 5 FPS (5:1 불일치) | 전체 디코딩 + 디스플레이 스로틀 |
| 3 | 프록시 MJPEG 깨짐 | multipart 경계가 chunk로 쪼개짐 | **프록시 우회, 모델 서버 직접 MJPEG** |
| 4 | 클립 슬라이드쇼 | 1 FPS 12프레임 저장 | **8 FPS 96프레임 저장** |
| 5 | 카드 뷰 끊김 | 스냅샷 5초 폴링 (0.2 FPS) | **MJPEG 직접 10 FPS** |
| 6 | 1440p OOM | Ring buffer 900프레임 × 11MB | **720p 강제 + maxlen 720** |
| 7 | FMP4 브라우저 불가 | cv2 mp4v 코덱 | **ffmpeg libx264 H.264** |

### 현재 스트리밍 수치

| 항목 | 수치 |
|---|---|
| 카드 라이브 | MJPEG 직접 **10 FPS** (모델 서버 → 브라우저) |
| 설정 모달 라이브 | MJPEG 직접 **10 FPS** |
| 링 버퍼 | ~12 FPS (maxlen=720, ~60초) |
| 추론 샘플링 | 1 FPS (normal) / 4 FPS (burst) |
| 클립 저장 | **8 FPS** (96프레임/12초, H.264) |
| VLM 입력 | 12프레임 (96프레임 중 샘플링) |

---

## 리소스 예산

### GPU VRAM

| 단계 | 로드 모델 | VRAM |
|---|---|---|
| 서버 시작 | YOLO + CLIP + CUDA | ~4.0GB |
| 첫 이벤트 | + Qwen2.5-VL-3B | ~11-13GB |

### RAM (카메라 수별)

| 카메라 | Ring Buffer | 총 RAM 예상 | 16GB 기준 | 32GB 기준 |
|---|---|---|---|---|
| 1대 | 1.9GB | ~9GB | 여유 | 여유 |
| 2대 | 3.8GB | ~11GB | 여유 | 여유 |
| 3대 | 5.7GB | ~13GB | 빡빡 (3GB 여유) | 여유 |
| 4대 | 7.6GB | ~15GB | 위험 | 여유 |

### CPU (4 vCPU 기준)

| 작업 | 사용량 |
|---|---|
| RTSP cv2.read() × 3 스트림 | ~0.5 vCPU |
| MJPEG imencode × 30fps | ~0.3 vCPU |
| YOLO + CLIP 추론 | ~0.5 vCPU |
| FastAPI + asyncio | ~0.3 vCPU |
| ffmpeg 클립 인코딩 (간헐적) | ~0.5 vCPU |
| 합계 | ~2.4 (평소) / ~3.5 (인코딩 중) |

### 환경별 판정

| 환경 | 판정 | 병목 |
|---|---|---|
| **로컬 (32GB RAM)** | 문제없음 | — |
| **g4dn.xlarge (16GB, 4vCPU)** | 3대까지 OK | RAM (ring buffer) |
| **g4dn.xlarge + 4대** | 위험 | RAM → ring buffer maxlen 축소 필요 |
| **g4dn.2xlarge (32GB, 8vCPU)** | 6대+ OK | — |

### AWS에서 카메라 4대 이상 시

ring buffer maxlen을 줄이면 RAM 절약 가능:

```python
# config.py 또는 .env에서 조정
RING_BUFFER_MAXLEN=360  # 30초 (기본 720=60초)
```

3대 × 360 = 2.85GB → 총 ~10GB → 16GB에서 6GB 여유.

---

## 배포

### 최소 요구사항

| 항목 | 최소 | 권장 |
|---|---|---|
| GPU | T4 16GB | L4 24GB |
| CPU | 4 vCPU | 8 vCPU |
| RAM | 16GB (3대까지) | 32GB |
| Python | 3.10+ | 3.12 |
| CUDA | 12.1+ | — |
| ffmpeg | 필수 | — |

### 실행

```bash
# 터미널 1: 모델 서버
cd hio_v2 && venv/Scripts/python main.py

# 터미널 2: 프론트엔드
cd hio_v2 && python frontend_server/main.py
```

http://localhost:8002

---

## v1 대비

| 항목 | v1 (Florence) | v2 (YOLO+CLIP+Qwen+Gemini) |
|---|---|---|
| Cash 탐지 | 캡션→키워드 (0-58%) | Pose wrist 근접 + VLM 질문 (wrist fallback) |
| VLM 입력 | 단일 프레임 캡션 | **12초 MP4 영상 직접 입력** (Qwen+Gemini 동일 파일) |
| Tier 3 | 스킵 가능 | **무조건 통과** (독립 판단, MP4 업로드) |
| Zone 역할 | 필수 ROI | **보조** (없어도 작동) |
| 스트리밍 | PyAV → 깨짐 → 스냅샷 폴링 0.2fps | **cv2 MJPEG 직접 10 FPS** |
| 클립 FPS | 1 FPS 슬라이드쇼 | **8 FPS 부드러운 영상** |
| 클립 재생 | FMP4 (브라우저 불가) | **H.264** (브라우저 직접 재생) |
| VLM 시각 힌트 | 없음 | **Zone 테두리 + 라벨** |
| Tier 2 비용 | $30-50/월 | $0 (로컬) |
| 코드 | 14,000+ LOC | ~2,280 LOC |

---

## 코드 품질 개선 이력 (2026-04-10)

5개 전문 에이전트(리소스 누수, 동시성, 에러 핸들링, 아키텍처, 보안)로 전수 감사 후 수정.

### 아키텍처 개선
| 항목 | 이전 | 이후 |
|---|---|---|
| VLM 입력 | 12프레임 JPEG 메모리 전달 | **저장된 MP4 파일 경로 전달** (Lazy Pipeline) |
| Qwen 입력 | 임시 JPEG 12장 생성→삭제 | **MP4에서 OpenCV 12프레임 샘플 → PIL → processor** |
| Gemini 입력 | base64 이미지 12장 API 전송 | **MP4 파일 업로드** (client.files.upload) |
| YOLO 정밀도 | FP32 (~1GB) | **FP32 유지** (FP16은 CLIP 동시 실행 시 CUDA assert) |
| Qwen 양자화 | auto/BF16 (~7-9GB) | **4-bit NF4** (~2.8GB, BitsAndBytes) |
| VRAM 합계 | ~13GB (12GB GPU에서 OOM) | **~6.35GB** (5.65GB 여유) |
| 이벤트 파이프라인 | `_qwen_lock`이 전체 감싸 (20~50초) | 3-Phase: 클립저장→Lock(Qwen만)→Tier3+DB |
| clip_review Lock | Tier3 API가 lock 내부 | Tier2만 lock, Tier3는 lock 해제 후 실행 |
| ThreadPool | `max_workers=2` → GPU 큐 대기 TimeoutError | `max_workers=4` |
| DB 커넥션 | 매 쿼리마다 새 커넥션 | 싱글톤 + `asyncio.Lock` 이중 초기화 방지 |
| DB 위치 | `data/events.db` (`/media`로 직접 다운로드 가능) | `db/events.db` (미디어 경로 밖) |

### 리소스 누수 수정
| 항목 | 수정 |
|---|---|
| GPU OOM | `torch.cuda.empty_cache()` + 텐서 정리 (video_analyzer except 블록) |
| VideoCapture 누수 | clip_review `cap.release()` try/finally 보호 |
| VideoWriter 누수 | save_clip_mp4 try/finally + ffmpeg TimeoutExpired 처리 |
| 임시 파일 | tmp_dir 생성을 try 블록 안으로 이동 |
| _handle_event 태스크 | `_event_tasks` set 추적 + 종료 시 cancel |
| _retention_loop 태스크 | 핸들 저장 + 종료 시 cancel |

### 동시성/데이터 수정
| 항목 | 수정 |
|---|---|
| `ws_clients` 반복 중 변경 | `list(ws_clients)` 스냅샷 + remove 전 존재 확인 |
| 중복 프레임 처리 | `_last_consumed_time` 추적으로 이미 처리한 프레임 스킵 |
| DB `row_factory` 오염 | 각 Row 쿼리 후 `conn.row_factory = None` 리셋 |
| DB 스키마 | `tier2_yes_count` + `trigger_mode` 컬럼 추가 + ALTER TABLE 마이그레이션 |
| `busy_timeout` | 5초 설정 (SQLITE_BUSY 방지) |

### 기능 추가
| 항목 | 내용 |
|---|---|
| Cash Trigger Mode | 카메라별 `wrist` / `zone_intrusion` 선택 — UI 드롭다운 + DB 저장 |
| Zone Intrusion | 캐셔 가려진 카메라용: 밖 사람의 손목/팔꿈치가 캐셔존 침입 시 트리거 |

### 프롬프트 재설계 (Tier 2/3 판정 정확도 개선)
| 항목 | 이전 | 이후 |
|------|------|------|
| Qwen 역할 | 5질문 yes_count → 판정자 | **6증거 슬롯 추출기** (판정 안 함) |
| Qwen cash 프롬프트 | "불확실하면 yes쪽" | **"불확실하면 false"** (보수적) |
| Gemini 역할 | "CONFIRMED / FALSE_ALARM" 2택 | **Hard Gate 3단 + UNCERTAIN** |
| Gemini 모호한 경우 | "lean toward CONFIRMED" | **UNCERTAIN** (알림 없음) |
| Qwen→Gemini 전달 | detected/confidence/reason | **"unreliable hints"로 개별 슬롯만** (앵커링 방지) |
| 서버단 강제 | Gemini 답변 그대로 사용 | **Hard Gate 카운트 + non-cash override** (Gemini 무시) |
| Hard Gate 기준 | N/A | H1 완화 (얇은 물체 포함) + H3 완화 (대면 상호작용) |
| 2/3 Hard Gate | N/A | **UNCERTAIN** (이전엔 FALSE_ALARM) |
| non-cash 감지 | 없음 | smartphone/card/receipt/document/envelope → 즉시 dismiss/FALSE_ALARM |
| JSON 파싱 | 단순 `text.index("{")` | 코드펜스 제거 + 필드 검증 + 폴백 |
| max_new_tokens | 100 | **200** (7필드 JSON 잘림 방지) |
| fire/violence 에러 시 | CONFIRMED (fail-open) | **fire=CONFIRMED (safety), cash=UNCERTAIN** |
| 빈 응답 | CONFIRMED | **FALSE_ALARM / UNCERTAIN** (절대 CONFIRMED 안 됨) |

### 보안
| 항목 | 수정 |
|---|---|
| `.gitignore` | `.env`, `cameras.yaml`, `db/`, `data/clips` 등 제외 |
| `events.db` 노출 | `/media` 경로 밖(`db/`)으로 이동 + 자동 마이그레이션 |
| DB_PATH | `.env` 상대경로 → `Path.resolve()` 절대경로 정규화 |
| WAL 마이그레이션 | `-wal`, `-shm` 파일도 함께 이동 |

### 프론트엔드
| 항목 | 수정 |
|---|---|
| 타임스탬프 NaN | `timestamp * 1000` → `new Date(timestamp)` (ISO 문자열 호환) |
| zone 클립 | 12프레임 샘플만 저장 (1.5초) → **풀 clip_frames에 zone 그려서 12초 저장** |

### Tier 3 (Gemini)
| 항목 | 수정 |
|---|---|
| 한글 경로 ASCII 에러 | `client.files.upload`에 한글 경로 전달 시 httpx ASCII 실패 → 임시 파일 복사 후 업로드 |
| 파일 ACTIVE 대기 | 업로드 직후 `FAILED_PRECONDITION` → `client.files.get()` 폴링으로 ACTIVE 상태 대기 (최대 30초) |

### Dead code 제거
- `process_event` 함수 삭제 (`_qwen_lock` 없는 Qwen 호출 경로 차단)
- `event_pipeline.py`에서 `extract_clip_frames` 데드 임포트 제거
- `import collections` 순서 정리 (파일 최상단으로)

---

## 향후 확장

1. **LoRA Fine-tuning**: 운영 데이터로 Qwen 호텔 도메인 특화
2. **Shadow Agent**: 백그라운드 재분석 + 피드백 루프
3. **Critic Trainer**: LightGBM false alarm 예측
4. **Episode Manager**: 다중 프레임 temporal reasoning + stability scoring
5. **프롬프트 버전 관리**: Qwen/Gemini 프롬프트 A/B 테스트
6. **멀티 GPU**: 10대+ 카메라 시 확장
