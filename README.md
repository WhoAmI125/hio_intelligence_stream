# HiO Intelligence Stream — 실시간 CCTV 이상 탐지 시스템

> Florence-2 (Tier-1, On-Device GPU) + Gemini (Tier-2, Cloud API) + 인간 라벨링 기반 GT 수집 파이프라인
> **3-서버 마이크로서비스 · 실시간 RTSP 분석 · g4dn.2xlarge 1대로 멀티 카메라 운영**

---

## 최근 업데이트 (2026-04-17)

- **Shadow/Critic 런타임 경로 제거** — 181건 기록에도 downstream 0 artifact로 Dead 확인, UI/API/설정/메뉴까지 완전 제거
- **Labeling UI 재설계** (`/monitor/labeling`) — 블로킹 `prompt()` 제거, 웰컴 카드 + 좌측 이벤트 리스트 패널(직접 선택) + 한국어 UI + TP/FP/Unclear 기준 가이드
- **`display_name` 카메라 라벨 편집** — `camera_id`(내부 식별자) 고정, `display_name`만 UI에서 편집 가능. 일산/금촌 같은 오배정 이름 UI에서 바로 수정
- **TP 이벤트 clip race fix** (`vlm_api.py`) — validation clip 유지, 영구 clip 생성 시 `val_entries` 재사용으로 ring buffer evict 경합 제거. TP 50% clip 유실 문제 해결
- **Gemini 로그 미디어 strip 통일** — `clip | ROI | gemini` 3열 고정 (기존 `ROI clip / clip / thumb` 불일치 제거)
- **Florence 튜닝 실측** — 1000 캡션 분석 결과 cash 어휘 실측 **0.00%**, latency p50 **1002ms**, 이중 resize 경로 확인. 상세: `scratch/tuning_bundle_20260417.tar.gz`

> **현재 강화 방향**: Gemini는 이미 잘 기능. Florence의 Tier-1 스크리너 역할과 GT 라벨 수집이 병목. **Motion gating + GT 수집**이 다음 우선순위.

---

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [전체 파일 구조](#전체-파일-구조)
3. [현재 아키텍처](#현재-아키텍처)
4. [아키텍처 다이어그램](#아키텍처-다이어그램)
5. [모델 아키텍처 상세](#모델-아키텍처-상세)
6. [추론 파이프라인 흐름](#추론-파이프라인-흐름)
7. [GPU/CPU 리소스 실측](#gpucpu-리소스-실측)
8. [핵심 컴포넌트](#핵심-컴포넌트)
9. [Labeling UI (GT 수집)](#labeling-ui-gt-수집)
10. [UI 구성](#ui-구성)
11. [데이터베이스 스키마](#데이터베이스-스키마)
12. [데이터 저장 구조](#데이터-저장-구조)
13. [API 요약](#api-요약)
14. [환경 변수](#환경-변수)
15. [배포 구조](#배포-구조)
16. [로컬 실행](#로컬-실행)
17. [트러블슈팅](#트러블슈팅)
18. [실측 기반 한계와 다음 단계](#실측-기반-한계와-다음-단계)
19. [Repo Hygiene](#repo-hygiene)
20. [로컬 개발 상세 가이드](#로컬-개발-상세-가이드)

---

## 프로젝트 개요

RTSP CCTV 스트림을 실시간으로 분석하여 3개 시나리오를 탐지합니다:

- `cash` — 현금 거래/금전 전달
- `fire` — 화재/연기
- `violence` — 폭력/충돌

### 2-Tier 설계

| Tier | 역할 | 모델 | 비용 |
|------|-----|------|------|
| Tier-1 | 고속 스크리닝 (장면 캡션 → 키워드 soft score) | Florence-2-large (GPU, float16) | GPU 추론당 ~1s |
| Tier-2 | 경계선 케이스 최종 판정 (video 기반) | Gemini 3.1 Flash Lite (Cloud API) | API 호출당 ~$0.001 + 지연 4-6s |

### 핵심 설계 원칙

- **Caption Sharing**: 프레임당 Florence-2 추론 1회로 3개 시나리오 동시 분석 (GPU 효율)
- **Escalation by threshold**: Tier-1 confidence `< TIER2_CASH_THRESHOLD(0.55)` 이면 자동 Gemini로 전송
- **Gemini가 cash 판정 전담**: Florence 캡션에는 cash 어휘 실측 **0.00%**. Tier-1은 "사람+카운터" 같은 soft trigger만 제공, 실제 판정은 Gemini video가 담당
- **인간 라벨링으로 GT 구축**: `/monitor/labeling` UI로 이벤트 단위 TP/FP/Unclear 수동 라벨링 → 후속 튜닝·A/B 측정 근거

### 현재 운영 범위

- g4dn.2xlarge 1대, 카메라 2대 (일산·금촌), T4 15GB GPU 중 ~2.2GB 사용, ~1 fps 실효 처리
- 하루 이벤트 수십~수백 건, Gemini 호출 건당 ~6초, 월 API 비용 수만원 수준

---

## 전체 파일 구조

```
hio_intelligence_stream/
├── model_server/                          # Tier-1/Tier-2 추론 서버 (:8000)
│   ├── main.py                            #   FastAPI 앱, 전체 컴포넌트 라이프사이클
│   ├── vlm_api.py                         #   VLM API 라우터 + 추론 루프 + val_clip race fix
│   ├── config.py                          #   환경변수 기반 중앙 설정
│   ├── stream_manager.py                  #   멀티카메라 RTSP 리더 + ring buffer + burst FPS
│   ├── pipeline_orchestrator.py           #   3개 시나리오 caption 공유 오케스트레이터
│   ├── evidence_router.py                 #   Tier-2 에스컬레이션 라우터 (threshold + Q-value)
│   ├── gemini_validator.py                #   Tier-2 Gemini Vision API 검증기
│   ├── inference_scheduler.py             #   중앙 추론 스케줄러 (디스패처 + 워커 풀 1)
│   ├── event_postprocessor.py             #   비동기 후처리 (클립 저장 + Gemini 호출)
│   ├── local_storage.py                   #   이벤트 JSON/클립 MP4/썸네일 JPG 저장
│   ├── flush_worker.py                    #   Model→DB 배치 플러시 (HTTP POST)
│   ├── logger.py                          #   구조화 JSONL 로거
│   ├── adapters/
│   │   ├── base_adapter.py                #     VLM 어댑터 ABC (preprocess/crop 포함)
│   │   └── florence_adapter.py            #     Florence-2 PyTorch 어댑터 (+LoRA 로드 지원)
│   ├── scenarios/
│   │   ├── base_scenario.py               #     CaptionAnalyzer (word-boundary regex 매칭)
│   │   ├── cash_scenario.py
│   │   ├── fire_scenario.py
│   │   └── violence_scenario.py
│   ├── agents/
│   │   ├── dynamic_agent.py               #     [Dead Code] 초기화만, 호출 0
│   │   └── prompts/                       #     cash.md / fire.md / violence.md
│   ├── evolution/                         # [Dead Code — 다음 라운드 제거 예정]
│   │   ├── critic_trainer.py              #     LightGBM 크리틱. 호출 경로 Shadow 제거로 죽음
│   │   └── rule_updater.py                #     프롬프트 자동 진화. 실측 Auto-Added 0건
│   └── lora/
│       ├── data_collector.py              #     LoRA 학습 데이터 수집기 (활성)
│       ├── dataset.py                     #     PyTorch Dataset + Collate
│       └── train_lora.py                  #     LoRA 파인튜닝 스크립트 (오프라인)
│
├── db_server/                             # 이벤트 DB 서버 (:8001)
│   ├── main.py                            #   FastAPI + SQLite WAL + cameras display_name 마이그레이션
│   └── models.py                          #   [Legacy] Django ORM 스키마 (참조용, 미사용)
│
├── frontend_server/                       # 모니터링 UI + 리버스 프록시 (:8002)
│   ├── main.py                            #   FastAPI + Jinja2 + /api/proxy/*
│   └── templates/vlm_pipeline/
│       ├── base_public.html               #     사이드바 레이아웃 (5개 메뉴)
│       ├── adhoc_rtsp.html                #     CCTV Live + 설정 모달 (display_name 편집)
│       ├── florence_logs.html             #     Florence Tier-1 추론 로그 + LoRA 피드백
│       ├── gemini_logs.html               #     Gemini Tier-2 검증 로그 (clip | ROI | gemini)
│       └── labeling.html                  #     GT 라벨링 UI (웰컴 + 이벤트 리스트 패널)
│
├── deploy/                                # AWS 배포 자동화
│   ├── setup_aws_g4dn.sh                  #   원스텝 배포 스크립트
│   ├── nginx.conf                         #   nginx 리버스 프록시
│   ├── vlm-{model,db,frontend}.service    #   systemd 유닛
│   ├── vlm-boot-recover.service           #   부팅 복구 유닛
│   ├── vlm-safe-recover.sh                #   복구 오케스트레이션
│   └── track_disconnect_2h.sh             #   인시던트 모니터링
│
├── data/                                  # 런타임 데이터 (gitignore)
│   ├── events/YYYYMMDD/*.json             #   이벤트 JSON (canonical)
│   ├── clips/YYYYMMDD/*.mp4               #   H.264 영구 clip (plain)
│   ├── clips/YYYYMMDD/*_roi.mp4           #   H.264 overlay clip (cash only)
│   ├── clips/YYYYMMDD/val_*.mp4           #   Gemini 입력 원본 (유지됨)
│   ├── thumbnails/YYYYMMDD/*.jpg          #   썸네일
│   ├── florence_logs/YYYYMMDD/*.jsonl     #   Florence 캡션 raw 로그
│   ├── lora_training/                     #   LoRA 수집 데이터
│   ├── shadow_feedback/*.jsonl            #   [Legacy] 2026-04-17 이전 기록, 유지만
│   ├── critic_models/                     #   [Empty] 삭제 대기
│   ├── rule_versions/                     #   [Empty] 삭제 대기
│   └── cctv_events.db                     #   SQLite
│
├── models/                                # HuggingFace 캐시 (gitignore, ~1.9GB)
│   └── hf/ + models--microsoft--Florence-2-large/
│
├── scratch/                               # 실험/분석 bundle (gitignore)
│   └── tuning_bundle_*.tar.gz             #   튜닝 분석용 번들
│
├── .env / .env.example / .env.aws         # 환경 설정
├── requirements.txt                       # Python 의존성 (CPU)
├── requirements_gpu.txt                   # CUDA 12.1 PyTorch
├── start_local.py                         # 로컬 3-서버 런처
├── AWS_G4DN_DEPLOY_GUIDE.md
└── VLM_INFERENCE_REFACTOR_GUIDE.md
```

---

## 현재 아키텍처

### 3-서버 구조

| 서버 | 포트 | 프로세스 | 역할 |
|------|-----|---------|------|
| `model_server` | `:8000` | uvicorn worker 1 | 추론, 스트림, 이벤트 생성, GPU 점유 |
| `db_server` | `:8001` | uvicorn worker 1 | SQLite WAL, 이벤트/카메라/검증 로그 |
| `frontend_server` | `:8002` | uvicorn worker 2 | Jinja2 UI + reverse proxy (vlm/db) |

### 런타임 컴포넌트

| 컴포넌트 | 역할 | GPU/CPU |
|---------|-----|--------|
| `StreamManager` | RTSP 리더 + ring buffer + burst FPS | CPU (OpenCV), GPU 선택 (NVDEC) |
| `InferenceScheduler` | 중앙 디스패처, 카메라별 최신 프레임만 | CPU |
| `FlorenceAdapter` | Tier-1 캡션 (Florence-2-large fp16) | **GPU** (CUDA) |
| `PipelineOrchestrator` | Caption Sharing + 시나리오 분석 | GPU 1회 + CPU regex |
| `CaptionAnalyzer` | word-boundary regex 키워드 매칭 | CPU |
| `EvidenceRouter` | Tier-2 에스컬레이션 판단 (threshold 우선) | CPU |
| `GeminiValidator` | Tier-2 검증 (Cloud API) | CPU (JPEG/MP4 인코딩) |
| `EventPostProcessor` | 비동기 후처리 (clip/thumb/Gemini) | CPU (FFmpeg) |
| `LocalStorage` | 이벤트/clip/썸네일 파일 저장 | CPU I/O |
| `FlushWorker` | Model→DB 배치 동기화 | CPU (HTTP) |
| `DataCollector` | LoRA 학습 데이터 수집 (passive) | CPU |

### Dead Code (제거 대기)

다음 모듈은 Shadow 경로 제거(2026-04-17)로 호출 경로가 끊김. 유지보수 라운드에서 제거 예정:

- `model_server/evolution/critic_trainer.py` — Shadow에서만 호출되던 `train()` 트리거 사라짐
- `model_server/evolution/rule_updater.py` — `apply_feedback_to_rules()` 호출 경로 없음
- `model_server/agents/dynamic_agent.py` — 초기화만 존재
- `model_server/base_detector.py` — 정의만, 호출 0

---

## 아키텍처 다이어그램

```text
                         ┌─────────────────────────────────┐
                         │   nginx (:80/443)                │
                         │   dev-cctv.hio.ai.kr             │
                         └──────────────┬──────────────────┘
                                        │
                         ┌──────────────▼──────────────────┐
                         │  Frontend Server (:8002)         │
                         │  Jinja2 + reverse proxy          │
                         │  /monitor/adhoc                  │
                         │  /monitor/florence-logs          │
                         │  /monitor/gemini-logs            │
                         │  /monitor/labeling   ← GT 수집    │
                         │  /dashboard                      │
                         └─────┬───────────────┬────────────┘
                               │               │
                  /api/vlm/*   │               │  /api/cameras|events|stats
                               ▼               ▼
┌──────────────────────────────────────┐  ┌──────────────────────┐
│      Model Server (:8000)            │  │ DB Server (:8001)    │
│                                      │  │ SQLite WAL           │
│ ┌────────────┐   ┌────────────────┐  │  │                      │
│ │StreamMgr   │   │InferScheduler  │  │  │ events               │
│ │ RTSP rd    │   │ Dispatcher     │  │  │ cameras (+display_name)
│ │ RingBuf    │   │ Worker×1       │  │  │ gemini_logs          │
│ └─────┬──────┘   └───────┬────────┘  │  │ episode_reviews      │
│       │                  │            │  │ worker_leases        │
│       │             ┌────▼──────┐    │  └──────────────────────┘
│       │             │Florence-2 │    │
│       │             │Tier-1 GPU │    │
│       │             │fp16 CUDA  │    │
│       │             └─────┬─────┘    │
│       │    caption (free-form text)  │
│       │             ┌─────▼─────────┐ │
│       │             │PipelineOrch + │ │
│       │             │CaptionAnalyzer│ │
│       │             │  ×3 scenarios │ │
│       │             └─────┬─────────┘ │
│       │              borderline conf  │
│       │             ┌─────▼─────────┐ │
│       │             │EvidenceRouter │ │
│       │             │< 0.55 → Gemini│ │
│       │             └─────┬─────────┘ │
│       │              escalate (video) │
│       │             ┌─────▼─────────┐ │
│       │             │GeminiValidator│ │
│       │             │ video clip    │ │
│       │             │ hard gate +   │ │
│       │             │ soft score    │ │
│       │             └─────┬─────────┘ │
│       │     ┌─────────────▼─────────┐ │
│       └─────│ EventPostProcessor    │ │
│             │ val_clip (retained)   │ │
│             │ ev_{id}.mp4 (plain)   │ │
│             │ ev_{id}_roi.mp4(over) │ │
│             │ thumbnail.jpg         │ │
│             └──────────┬────────────┘ │
│                        │              │
│             ┌──────────▼────────────┐ │
│             │ FlushWorker → DB ─────────→ DB Server
│             └───────────────────────┘ │
│                                       │
│   [LoRA DataCollector — passive]       │
└───────────────────────────────────────┘

Human feedback loop:
  Events → /monitor/labeling UI → POST /api/vlm/feedback → event.human_feedback
  (지금 자동 학습 루프 없음. 쌓인 라벨은 수동 분석·튜닝·A/B에 활용)
```

---

## 모델 아키텍처 상세

### Tier-1: Florence-2 (On-Device GPU)

| 항목 | 기본값 | 운영값 (.env) | 설명 |
|------|-------|-------------|------|
| 모델 | `microsoft/Florence-2-large` | 동일 | ~770M 파라미터 VLM encoder-decoder |
| 로드 | HF `AutoModelForCausalLM` | `trust_remote_code=True` | 캐시 `MODEL_SERVER_MODELS_DIR` (기본 `models/`) |
| 백엔드 | `pytorch` | `pytorch` | OpenVINO stub 존재 (미완성) |
| 정밀도 | GPU fp16 / CPU fp32 | fp16 CUDA | `torch.autocast` 적용 |
| 입력 크기 | 448×448 | **320×320** | 속도 우선. ⚠️ HF processor가 내부 768 upscale — 이중 resize |
| max_tokens | 512 | **96** | 25.9% 캡션이 90+ 토큰 근접 (truncation 위험) |
| num_beams | 3 | **1** | Greedy. do_sample=False |
| caption_detail | `more` | **`detailed`** | `<DETAILED_CAPTION>` 사용 |
| LoRA | 비활성 | `LORA_ENABLED=false` | `peft.PeftModel` 런타임 로드 지원 |

#### Florence-2 추론 Hot Path

```
BGR numpy (카메라 프레임 ~1280×720)
  ↓ cv2.resize(320×320, INTER_AREA) + BGR→RGB           [이중 resize 시작]
  ↓ PIL Image 변환
  ↓ AutoProcessor (internal 768 upscale, tokenize)       [이중 resize 끝]
  ↓ .to(cuda) → GPU 텐서
  ↓ torch.inference_mode() + torch.autocast(cuda, fp16)
  ↓ model.generate(max_new_tokens=96, num_beams=1)
  ↓ post_process_generation() → 자유 캡션 텍스트
  ↓ CaptionAnalyzer.analyze() → 키워드 매칭 (CPU, μs)
```

#### 지원 task tokens

| Task | 운영 사용? | 설명 |
|------|---------|------|
| `<DETAILED_CAPTION>` | ✅ (기본) | 현재 운영 경로 |
| `<CAPTION>` / `<MORE_DETAILED_CAPTION>` | - | 설정으로 전환 가능 |
| `<OD>` | - | 미연결 (`detect_objects` 메서드 존재) |
| `<DENSE_REGION_CAPTION>` | - | 미연결 |
| `<CAPTION_TO_PHRASE_GROUNDING>` | - | `ground_phrase` 메서드 존재, 파이프라인 미연결 |
| `<OPEN_VOCABULARY_DETECTION>` | - | 실험 완료: **hallucination 확인** (존재 여부 무관하게 bbox 반환). 단독 사용 불가 |

### Tier-2: Gemini Vision (Cloud API)

| 항목 | 값 |
|------|-----|
| 모델 | `gemini-3.1-flash-lite-preview` |
| Temperature | 0.1 |
| max output tokens | 1500 |
| 타임아웃 | 180초 (.env) / 기본 30초 |
| 동시 호출 | 1 (`BoundedSemaphore`) |
| 기본 모드 | `video_only` |
| 입력 clip | 1280×720 H.264 CRF23 ~3Mbps, 10초 |
| 실패 정책 | Fail-Open (API 오류 시 승인) |

#### 검증 모드

| 모드 | 동작 |
|------|------|
| `hybrid` | cash는 storyboard 우선, 그 외는 video 우선 |
| `video_first` | 비디오 먼저, 실패 시 storyboard |
| **`video_only`** | **운영 기본값** — 비디오만 |
| `images_first` | 키프레임 먼저 |
| `storyboard` | 최대 12장 키프레임 |
| `image` | 단일 프레임 |

#### Gemini 통합 프롬프트 구조

```
[시나리오별 이벤트 정의]
  ├── hard-gate 규칙 (즉시 판정 — non_cash_penalty 40+ 등)
  ├── soft-score 규칙 (money_likelihood/hand_to_hand/safe_drawer 가중)
  └── 정책 우선순위
[Tier-1 업스트림 컨텍스트] (soft hints only)
  ├── Florence confidence/stability
  ├── matched_keywords, object_hints
  └── Router action 이유
[응답 포맷] (JSON)
  ├── event_policy, is_valid_event, decision
  ├── severity_label, confidence
  ├── policy_scores, reason_bullets
  └── corrected_event_type
```

### LoRA 파인튜닝 (오프라인, 현재 비활성)

| 항목 | 값 |
|------|-----|
| 베이스 | `microsoft/Florence-2-large` |
| Rank / Alpha / Dropout | 8 / 16 / 0.05 |
| 타겟 레이어 | `q_proj`, `v_proj`, `k_proj`, `out_proj` |
| Optimizer | AdamW (lr=1e-4, wd=0.01) |
| 배치 / 에폭 | 4 / 3 |
| 최소 샘플 | 50 |

> LoRA는 현재 **data collection만 활성**. 학습은 수동 트리거. GT TP 30-50건 이상 쌓이면 실행 고려.

---

## 추론 파이프라인 흐름

### Caption Sharing 최적화

```
┌──────────────────────────────────────────────────────────────┐
│ 비효율 (안 씀):                                                │
│   Frame → Florence(cash) → Florence(fire) → Florence(viol)   │
│   = GPU 3회/프레임                                             │
│                                                              │
│ Caption Sharing (운영):                                        │
│   Frame → Florence(1회) → caption 텍스트 공유                  │
│         ├→ CaptionAnalyzer(cash)     ← CPU regex (μs)         │
│         ├→ CaptionAnalyzer(fire)     ← CPU regex (μs)         │
│         └→ CaptionAnalyzer(violence) ← CPU regex (μs)         │
│   = GPU 1회/프레임                                             │
└──────────────────────────────────────────────────────────────┘
```

### CaptionAnalyzer 매칭 엔진

#### 키워드 계층 (시나리오별)

| 계층 | 가중치 | 역할 | Cash 예시 |
|-----|-------|-----|----------|
| `strong_positive` | 0.3~0.4 | 직접 증거 | money, cash, banknote |
| `moderate_positive` | 0.1~0.15 | 간접 증거 | counter, holding, handing |
| `context_phrases` | 0.3~0.5 | 복합 구문 | "handing over", "cash register" |
| `negative` | -0.2~-0.3 | 반증 | phone, card, receipt |
| `neutralizing_phrases` | 무효화 | 강한 키워드 제거 | "fire extinguisher" → fire 무효 |
| `object_hints` | 0 | Tier-2 힌트만 | paper, envelope, wallet |

#### 실측 행동 (cash 이벤트 186건 분석)

- `matched_keywords=[]` — **186/186 전부 빈 배열** (strong_positive 매칭 0건)
- `holding` 캡션 포함 — **186/186 (100%)** ← cash 이벤트의 실질 트리거
- `desk` 79%, `bank` 20%, `counter` 19%, `teller` 12%
- **캡션에 `cash`/`bill`/`banknote`/`money`/`wallet`/`drawer` 매칭 — 0건 (0.00%)**
- Tier-1 confidence p50 = **0.40** (모두 0.30~0.70 경계)

#### Cash H2H (Hand-to-Hand) 탐지 (설계)

```
H2H 양성 = 위치 키워드 (counter/cashier/desk/teller/register/drawer)
         AND
         행동 키워드 (handing/holding/passing/reaching/exchanging)
```

실측상 이 경로로 대부분 이벤트가 conf 0.30~0.55 달성 → 전부 Gemini로 에스컬레이션.

### EvidenceRouter 판정 로직

```python
# evidence_router.py:1119~1141 (단순화)
if event_type in {'fire', 'violence'} and avg_conf < 0.95:
    action = GEMINI_VIDEO   # hard-risk escalation
elif avg_conf < scenario_threshold:  # cash: 0.55
    action = GEMINI_VIDEO   # force_tier2
elif avg_conf >= 0.85 and stability >= 0.90:
    action = SKIP
else:
    action = max_q_action   # margin gate 적용
```

실측 영향:
- cash 이벤트 92%가 conf < 0.55 → `force_tier2` 경로 (margin gate 무관)
- SKIP 게이트에 도달하는 경우는 거의 없음

**액션 공간**: `SKIP`, `GEMINI_IMG`, `GEMINI_VIDEO`, `HUMAN_QUEUE`

---

## GPU/CPU 리소스 실측

### 운영 환경 (g4dn.2xlarge, 2026-04-17 측정)

| 리소스 | 스펙 | 실측 사용량 |
|--------|-----|----------|
| GPU | Tesla T4 15GB VRAM | **2,165 MiB / 15GB (14%)** |
| GPU SM 활용률 | - | 평균 **65%** (45~83% 범위) |
| GPU mem BW | - | 33~59% |
| GPU 온도 | - | 39°C |
| CPU | Intel Xeon 8 vCPU | user 21%, load avg 2.13/8 |
| RAM | 30 GB | **11.6 GB (37%)** |
| model_server 프로세스 | - | 144% CPU, RSS 9.2 GB |
| 디스크 (data/) | - | ~10 GB |
| 디스크 (models/) | - | 1.9 GB |

### GPU 메모리 내역

```
Florence-2-large (fp16):                  ~1.5 GB
PyTorch CUDA context + 커널 캐시:          ~0.3 GB
Activation peaks (320 input, tok 96):      ~0.2-0.4 GB
RTSP HW 디코딩 버퍼 (NVDEC, 카메라×2):      ~0.1 GB
────────────────────────────────────────
합계:                                      ~2.1 GB (15 GB 중 14%)
여유:                                      ~13 GB
```

### Latency 실측 (Florence 1000 캡션)

| 지표 | ms |
|------|-----|
| mean | 994 |
| p50 | **1002** |
| p90 | 1123 |
| p99 | 1472 |
| max | 1754 |
| >667ms (BASE_FPS 1.5 예산) 초과 | **98.6%** |
| >1000ms (1.0 fps 예산) 초과 | 50.8% |

**결론**: `BASE_FPS=1.5`는 설정값일 뿐, 실효 처리율 **~1 fps**. GPU compute-bound 확정. `INFERENCE_WORKERS=2`는 throughput 개선 효과 없음 (이미 saturate 경계).

### 직렬화 메커니즘 (3중 락)

1. `GLOBAL_INFERENCE_LOCK=true` — 모든 Florence 추론 직렬
2. `INFERENCE_WORKERS=1` — 스케줄러 워커 1개
3. `GEMINI_MAX_CONCURRENT=1` — Tier-2 API 동시 1회

### 멀티 카메라 확장 예측

| 카메라 | GPU VRAM | RAM | Florence fps | 비고 |
|-------|---------|-----|--------------|------|
| 1 | ~2.1 GB | ~8 GB | 1.0 | 여유 충분 |
| 2 (현재) | ~2.2 GB | ~11.6 GB | 0.5/cam (총 1.0) | compute-bound |
| 4 | ~2.3 GB | ~15 GB | 0.25/cam (총 1.0) | 프레임 drop 심해짐 |
| 8+ | ~2.5 GB | >15 GB | 0.125/cam | Ring buffer/디코딩 부족 |

> 병목은 GPU compute (Florence 지연), 확장 전 **해상도 튜닝 또는 multi-GPU** 필요.

---

## 핵심 컴포넌트

### StreamManager

- 카메라별 전용 `_reader_loop` daemon thread
- Ring buffer: `collections.deque(maxlen ≈ effective_fps × 30)` — `{"frame": np.ndarray, "mono_ts": float}`
- Sampling: base ~12fps, burst ~15fps
- Burst mode: 탐지 후 3초간 FPS 상승
- 재연결: 지수 백오프 1s→2s→4s→8s + jitter
- HW 가속: `RTSP_HWACCEL=cuda` (NVDEC), CPU fallback 자동
- 중복 방지: 동일 RTSP URL 중복 오픈 차단

### InferenceScheduler

- Dispatcher: 20ms 간격 카메라 순회, 최신 프레임으로 `InferenceJob` 생성
- Worker 1개, 큐에서 잡 꺼내 `_run_inference_once()` 실행
- `pending/inflight` 플래그로 카메라당 max 1 job
- Active burst: 탐지 후 `INFERENCE_ACTIVE_BURST_SEC=3` 동안 FPS 증가
- 스테일 잡: `run_id` 버전으로 이전 세션 잡 폐기
- 큐: `INFERENCE_QUEUE_SIZE=128`

### PipelineOrchestrator

- `process_frame_sequential()` — Florence 1회 + 3 시나리오 CaptionAnalyzer (운영 경로)
- `process_frame()` — `ThreadPoolExecutor(3)` 병렬 (미사용)
- `CASH_DUAL_PATH_ENABLED` — ROI crop + 전체 프레임 2회 추론 (현재 **OFF**)
- `CASH_ROI_INFER_ENABLED` — ROI 분리 추론 (현재 **OFF**)

### Val_clip 유지 + 영구 clip 생성 (2026-04-17 fix)

```
1. validation clip 생성 (val_ev_{id}.mp4) — Gemini 입력용, overlay 포함(cash 시)
2. Gemini API 호출 (video_only) — 4-6초 대기
3. val_clip 유지 (삭제 안 함. 과거: os.remove)
4. 영구 clip 생성:
   - val_entries 재사용 (동일 프레임 셋) ← race 제거
   - validation_clip_sec 로 duration 통일
   - ev_{id}.mp4 (plain, clip_url)
   - ev_{id}_roi.mp4 (overlay, cash에만)
   - thumbnail.jpg
5. Fallback: 영구 clip 생성 실패 시 event.clip_url = val_clip_path 자동 승계
```

결과: TP 이벤트 clip 유실 버그 해결, Gemini 본 원본과 UI 표시 영상 1:1 일치.

---

## Labeling UI (GT 수집)

### 개요

`/monitor/labeling` — 이벤트 단위 TP/FP/Unclear 인간 라벨링.

누적된 라벨은:
- 모델 정확도 A/B 측정의 근거
- Florence/Gemini/CaptionAnalyzer 튜닝 방향 결정의 유일한 수단
- LoRA 학습 데이터의 품질 게이트 (GT TP 30-50건 이상 필요)

### UI 구성 (2026-04-17 재설계)

**웰컴 카드** (처음 방문 시)
- 블로킹 `prompt()` 없음. 인라인 이름 입력 + Enter 지원
- 3단계 워크플로 설명 (영상→판정→저장)
- TP/FP/Unclear 기준 박스 (색상 구분)

**좌측 이벤트 리스트 패널** (320px 고정, 스크롤)
- 각 행: 시간·카메라·판정 배지(TP/FP/Unclear/unlabeled)·Gemini 배지(G:accept/G:decline)·t1 conf·"no clip" 마커
- **클릭 → 해당 이벤트 즉시 이동**
- N/P 키보드 네비게이션 시 active 행 자동 스크롤
- 저장 시 배지 즉시 갱신

**우측 판정 패널**
- 영상 (자동재생, 오버레이 토글)
- Gemini 판정 참고 정보 (가리지 않고 표시)
- TP/FP/Unclear 큰 버튼 + 키보드 `1/2/3`
- FP 세부 유형 (폰/영수증/카드/빈 장면/직원만/전달 없음/기타)
- 메모 (FP/Unclear는 필수)
- 저장 & 다음 (`N`), 이전 (`P`), 스킵 (`⇧S`)

**카메라 필터** — `display_name`이 있으면 `표시명 (id)` 형식으로 표시

---

## UI 구성

### 1. CCTV Live (`/monitor/adhoc`)

- 멀티 카메라 카드 그리드 (실시간 MJPEG)
- 카드 클릭 → 설정 모달
- **표시명 편집** (`display_name`) — camera_id는 readonly (내부 식별자), 표시명만 자유 편집
- ROI zone 편집 (cashier/drawer 폴리곤)
- Start/Stop, Full Screen, ROI Only

### 2. Florence 로그 (`/monitor/florence-logs`)

- Tier-1 추론 결과 테이블 (카메라/시나리오/탐지여부 필터)
- 캡션 preview + 상세 JSON 모달
- LoRA 피드백 버튼 (accept/decline/unsure)
- 통계: 총 행, 탐지 행, 평균 추론 시간

### 3. Gemini 로그 (`/monitor/gemini-logs`)

- Tier-2 검증 결과 + 사유
- 카메라 필터 (display_name 우선 표시)
- 시나리오/결정 상태 필터
- **미디어 strip 통일**: `clip | ROI | gemini` — overlay 유무 무관하게 라벨 고정
- 판정 상세 모달
- 6초 자동 갱신

### 4. Labeling (`/monitor/labeling`)

위 "Labeling UI" 섹션 참조.

### 5. System Dashboard (`/dashboard`)

- CPU/RAM/GPU/VRAM 실시간 (5초 갱신)
- 모델 상태
- 최근 이벤트

---

## 데이터베이스 스키마

### SQLite WAL — `data/cctv_events.db`

#### cameras 테이블 (2026-04-17 `display_name` 추가)

| 컬럼 | 타입 | 설명 |
|------|-----|------|
| `camera_id` | TEXT UNIQUE | 내부 식별자 (파일명·이벤트 ID에 박힘, **읽기 전용**) |
| `display_name` | TEXT DEFAULT '' | UI 표시 라벨 (자유 편집) |
| `rtsp_url` | TEXT | RTSP 스트림 URL |
| `base_fps` | REAL | 기본 FPS |
| `rtsp_transport` | TEXT | tcp/udp |
| `event_cooldown_sec` | REAL | 이벤트 쿨다운 |
| `clip_duration_sec` | REAL | 클립 길이 (권장: validation_clip_sec과 동일) |
| `validation_clip_sec` | REAL | Gemini 입력 클립 길이 |
| `evidence_mode` | TEXT | Gemini 모드 (video_only 등) |
| `cashier_zone` | TEXT JSON | 캐셔 존 폴리곤 (정규화 좌표) |
| `drawer_zone` | TEXT JSON | 서랍 존 폴리곤 |

마이그레이션은 `db_server/main.py`에서 idempotent 실행 (`PRAGMA table_info` 후 없으면 `ALTER TABLE`).

#### events 테이블

| 컬럼 | 타입 | 설명 |
|------|-----|------|
| `event_id` | TEXT UNIQUE | 이벤트 고유 ID (`ev_{ts_ms}_{scenario}_{camera_id}`) |
| `camera_id` | TEXT | 카메라 내부 식별자 |
| `event_type` / `scenario` | TEXT | cash/fire/violence |
| `confidence` | REAL | Tier-1 신뢰도 |
| `tier` | INTEGER | 1 / 2 |
| `is_detected` | BOOLEAN | 최종 탐지 여부 |
| `gemini_validated` / `gemini_confidence` / `gemini_reason` | - | Tier-2 결과 |
| `caption` | TEXT | Florence 캡션 |
| `matched_keywords` | TEXT | (실측: 대부분 `[]`) |
| `clip_path` / `clip_url` | TEXT | 영구 clip 경로 |
| `human_feedback` | TEXT JSON | **Labeling UI 라벨 (canonical GT)** |
| `event_data` | TEXT JSON | 전체 메타 |
| `created_at` | TIMESTAMP | |

> **참고**: SQLite `events` 테이블은 현재 flush_worker가 실제 쓰기 안 하는 상태. 진실의 원본은 `data/events/YYYYMMDD/*.json`.

#### gemini_logs 테이블

`event_id`, `gemini_state`, `gemini_validated`, `input_mode`, `processing_time_ms`, `prompt_version`, `log_data` (전체 JSON)

#### episode_reviews 테이블

에피소드 인간 리뷰 큐 (현재 미사용, 구조만 유지)

#### worker_leases 테이블

크로스 프로세스 워커 중복 방지 (`camera_id` UNIQUE)

---

## 데이터 저장 구조

루트는 `MODEL_SERVER_DATA_DIR` (기본 `data/`):

```
data/
├── events/YYYYMMDD/*.json              # ★ 이벤트 JSON (canonical 진실)
├── clips/YYYYMMDD/
│   ├── ev_{id}_{scenario}_{cam}.mp4    # plain 영구 clip (clip_url)
│   ├── ev_{id}_{scenario}_{cam}_roi.mp4 # overlay 영구 clip (overlay_clip_url, cash only)
│   └── val_ev_{id}_{scenario}_{cam}.mp4 # Gemini 입력 원본 (2026-04-17부터 유지)
├── thumbnails/YYYYMMDD/*.jpg           # 썸네일
├── florence_logs/YYYYMMDD/{cam}.jsonl  # Florence 캡션 raw 로그
├── lora_training/                      # LoRA 학습 데이터 (수집 중)
│   ├── images/*.jpg
│   ├── annotations.jsonl
│   └── LoRa_Flourence_feedback/
├── shadow_feedback/*.jsonl             # ★ Legacy (2026-04-17 이전), 보존
├── critic_models/                      # Empty, 삭제 대기
├── rule_versions/                      # Empty, 삭제 대기
├── cctv_events.db                      # SQLite WAL
├── media_archive/                      # DB 서버 미디어 보관
├── recovery_logs/
└── incident_watch/
```

### 클립 저장 경로

```
1. 카메라 RTSP → StreamManager ring buffer (numpy frames)
2. 이벤트 발생 → event_postprocessor 큐 enqueue
3. val_clip 생성:
   - ring buffer에서 anchor_mono_ts 기준 10초 프레임 추출 (val_entries)
   - cv2.VideoWriter로 임시 AVI (MJPG)
   - FFmpeg: libx264, CRF 23, preset fast, yuv420p, +faststart → val_ev_*.mp4
   - overlay 있으면 zone polygon을 프레임에 burn
4. Gemini API 호출 (video_only 모드, val_ev_*.mp4 전송)
5. 영구 clip 생성 (val_entries 재사용):
   - ev_*.mp4 (plain, 항상 생성)
   - ev_*_roi.mp4 (overlay, cash + zone 있을 때)
   - thumbnail.jpg
6. FlushWorker가 주기적으로 DB로 메타 POST
```

---

## API 요약

### Frontend Server (`:8002`)

| 메서드 | 경로 | 설명 |
|-------|-----|------|
| GET | `/monitor/adhoc` | CCTV Live |
| GET | `/monitor/florence-logs` | Florence 추론 로그 |
| GET | `/monitor/gemini-logs` | Gemini 검증 로그 |
| GET | `/monitor/labeling` | **GT 라벨링 UI** |
| GET | `/dashboard` | 시스템 대시보드 |
| ANY | `/api/vlm/{path}` | Model Server 프록시 |
| GET | `/api/proxy/status` | 모델 서버 상태 |
| GET | `/api/proxy/events` | 이벤트 목록 |
| GET | `/api/proxy/stats` | 통계 |
| CRUD | `/api/proxy/cameras[/{camera_id}]` | 카메라 설정 (display_name 포함) |
| GET | `/api/proxy/system` | 시스템 메트릭 |

### Model Server (`:8000`)

| 메서드 | 경로 | 설명 |
|-------|-----|------|
| POST | `/api/vlm/start/` | RTSP 스트림 시작 |
| POST | `/api/vlm/stop/` | 스트림 중지 |
| GET | `/api/vlm/video/` | MJPEG 스트리밍 |
| GET | `/api/vlm/status/` | 카메라 상태 |
| GET | `/api/vlm/events/` | 이벤트 목록 |
| POST | `/api/vlm/zones/` | ROI 존 설정 |
| GET | `/api/vlm/crop/` | ROI crop 미리보기 |
| POST | `/api/vlm/feedback/` | 인간 피드백 (human_feedback 저장) |

### DB Server (`:8001`)

| 메서드 | 경로 | 설명 |
|-------|-----|------|
| POST | `/api/flush` | 배치 이벤트 수신 (multipart) |
| GET | `/api/events` | 페이지네이션 이벤트 |
| GET | `/api/events/{event_id}` | 단건 조회 |
| POST | `/api/feedback` | 피드백 저장 |
| GET | `/api/stats` | 집계 통계 |
| GET/POST/PUT/DELETE | `/api/cameras[/{camera_id}]` | 카메라 CRUD (`display_name` 포함) |

---

## 환경 변수

### AI 모델

| 변수 | 기본 | 운영 | 설명 |
|------|-----|-----|------|
| `FLORENCE_MODEL` | `microsoft/Florence-2-large` | 동일 | |
| `FLORENCE_BACKEND` | `pytorch` | `pytorch` | OpenVINO stub |
| `FLORENCE_DEVICE` | `cuda` | `cuda` | |
| `FLORENCE_INPUT_SIZE` | `448` | **`320`** | ⚠️ HF processor가 768 upscale — 이중 resize |
| `FLORENCE_MAX_TOKENS` | `512` | **`96`** | 25.9% 캡션 cap 근접 |
| `FLORENCE_NUM_BEAMS` | `3` | **`1`** | Greedy |
| `FLORENCE_CAPTION_DETAIL` | `more` | **`detailed`** | `<DETAILED_CAPTION>` |
| `FLORENCE_LOG_PERSIST` | `false` | **`true`** | raw 캡션 JSONL 저장 |
| `GEMINI_API_KEY` | (필수) | 설정됨 | |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite-preview` | 동일 | |
| `GEMINI_TIMEOUT_SEC` | `30` | **`180`** | |
| `GEMINI_MAX_CONCURRENT` | `1` | `1` | |
| `LORA_ENABLED` | `false` | `false` | |

### 탐지 임계값

| 변수 | 기본 | 설명 |
|------|-----|------|
| `CASH_THRESHOLD` | `0.30` | Tier-1 cash 탐지 경계 (이벤트 생성 최저) |
| `VIOLENCE_THRESHOLD` | `0.30` | |
| `FIRE_THRESHOLD` | `0.30` | |
| `TIER2_CASH_THRESHOLD` | **`0.55`** | 이하 → `force_tier2` Gemini 전송 |
| `TIER2_VIOLENCE_THRESHOLD` | `0.70` | |
| `TIER2_FIRE_THRESHOLD` | `0.60` | |
| `SKIP_CONFIDENCE` | `0.85` | 이 이상 + 안정성 높으면 Tier-2 스킵 |
| `SKIP_STABILITY` | `0.90` | |
| `CASH_DUAL_PATH_ENABLED` | `false` | ROI + Global 이중 추론 |
| `CASH_ROI_INFER_ENABLED` | `false` | ROI 분리 추론 |

### 스트림/추론

| 변수 | 기본 | 설명 |
|------|-----|------|
| `BASE_FPS` | `1.5` | 겉보기 설정 (실효 ~1 fps) |
| `BURST_FPS` | `4.0` | 탐지 후 |
| `GLOBAL_INFERENCE_LOCK` | `true` | Florence 추론 직렬화 |
| `INFERENCE_WORKERS` | `1` | GPU compute-bound이라 2로 늘려도 개선 없음 |
| `INFERENCE_QUEUE_SIZE` | `128` | |
| `INFERENCE_ACTIVE_BURST_SEC` | `3.0` | |
| `RTSP_TRANSPORT` | `tcp` | |
| `RTSP_HWACCEL` | `cuda` | NVDEC (CPU fallback) |

### 저장/기타

| 변수 | 기본 | 설명 |
|------|-----|------|
| `MODEL_SERVER_DATA_DIR` | `data` | |
| `DB_PATH` | `data/cctv_events.db` | |
| `FFMPEG_PATH` | `ffmpeg` | |
| `EVIDENCE_MODE` | `video_only` | |
| `LOCAL_RETENTION_DAYS` | `3` | |
| `TZ` | `Asia/Seoul` | |
| `LOG_LEVEL` | `INFO` | |
| `AUTO_RESTORE_CAMERAS_ON_BOOT` | `true` | |

> **제거됨 (2026-04-17)**: `SHADOW_BATCH_SIZE`, `SHADOW_PERSIST_DIR`, `SHADOW_MAX_QUEUE`, `SHADOW_DISAGREE_THRESHOLD`, `CRITIC_SHADOW_MODE`

---

## 배포 구조

### AWS g4dn.2xlarge 타겟

| 리소스 | 스펙 |
|-------|-----|
| CPU | Intel Xeon 8 vCPU |
| RAM | 32 GB |
| GPU | NVIDIA Tesla T4 15 GB VRAM |
| 스토리지 | EBS 30GB+ (gp3 권장) |
| OS | Ubuntu 24.04 LTS |
| CUDA | 12.1+ |

### systemd 서비스

```
vlm-boot-recover.service (oneshot, 부팅 시)
  └→ vlm-safe-recover.sh boot-start
      ├→ vlm-db.service (worker 1, :8001)
      ├→ vlm-model.service (worker 1, :8000, GPU)
      └→ vlm-frontend.service (worker 2, :8002)

nginx.service (:80/443 → :8002)
```

### 보안

- 8000/8001/8002 포트는 `127.0.0.1` 바인딩 (외부 접근 차단)
- nginx만 80/443 외부 노출
- `.env` gitignore
- RTSP credential 로그에 노출됨 — journalctl 로그 접근 통제 필요 (알려진 이슈)

### 복구 (`vlm-safe-recover.sh`)

| 단계 | 타임아웃 | 검증 |
|------|---------|------|
| vlm-db | 25초 | HTTP 200 |
| vlm-model | 180초 | `florence_initialized=true` |
| 카메라 자동 복원 | 150초 | 모든 카메라 `running=true` |
| vlm-frontend | 40초 | HTTP 200 |
| nginx + 공개 URL | 40초 | 접근 확인 |

---

## 로컬 실행

### 1. 가상환경/의존성

```bash
cd /path/to/hio_intelligence_stream
python3 -m venv venv
source venv/bin/activate       # Linux/Mac
# .\venv\Scripts\Activate.ps1  # Windows

# GPU (CUDA 12.1) — requirements.txt보다 먼저
pip install --no-cache-dir -r requirements_gpu.txt
pip install -r requirements.txt

# CPU-only
pip install -r requirements.txt
```

### 2. 환경 파일

```bash
cp .env.example .env
# GEMINI_API_KEY=your_key
# FLORENCE_DEVICE=cuda  (또는 cpu)
```

### 3. 실행

```bash
python start_local.py          # 3서버
python start_local.py model db # 일부
```

접속:
- CCTV Live: http://localhost:8002/monitor/adhoc
- Labeling: http://localhost:8002/monitor/labeling
- Florence 로그: http://localhost:8002/monitor/florence-logs
- Gemini 로그: http://localhost:8002/monitor/gemini-logs
- Dashboard: http://localhost:8002/dashboard

---

## 트러블슈팅

### TP 이벤트에 clip이 없는 경우

2026-04-17 fix 이전 이벤트일 수 있음. 신규 이벤트는 val_clip 유지 + race 제거로 항상 `clip_url` 존재. journalctl 에서 `Permanent clip skipped` WARNING 있으면 ring buffer 문제 가능성.

### Labeling UI 진입 시 팝업만 뜨는 경우

2026-04-17 재설계 이전 브라우저 캐시. 강제 새로고침 (`Ctrl+Shift+R`) 후 확인. 웰컴 카드가 보여야 정상.

### 카메라 이름 바꾸기

Camera ID는 고정 (파일명·이벤트 ID에 박힘). **표시명**(`display_name`)만 편집:
1. CCTV Live → 카드 Settings 클릭 → 표시명 입력 → Save
2. 모든 UI에서 새 이름 반영, 내부 ID는 기존 이벤트와 연속

### `Assertion fctx->async_lock failed` / 시작 500

동일 RTSP 중복 오픈으로 인한 디코더 충돌. `/api/vlm/start/`에서 중복 차단하지만 이미 열린 프로세스를 `Ctrl+C`로 정리 못 했을 때 발생. 서비스 재시작 필요.

### Gemini 로그의 `SKIP`

Tier-1 신뢰도가 충분히 높아 Tier-2를 생략한 상태. **오탐 아님**.

### GPU OOM

- `FLORENCE_INPUT_SIZE` 320 이하로
- `FLORENCE_MAX_TOKENS` 96 이하
- 카메라 수 줄이기 (Ring buffer RAM)

### 모델 서버 시작이 느림

최초 Florence-2 다운로드 ~1.9GB (2-5분). `setup_aws_g4dn.sh`의 `SKIP_MODEL_PRELOAD=0`으로 사전 다운로드.

---

## 실측 기반 한계와 다음 단계

### 실측 확인된 한계

| 항목 | 실측 | 의미 |
|------|-----|------|
| Florence 캡션 cash 어휘 | **0.00%** (1000건) | Florence는 cash 직접 인식 안 함 |
| Tier-1 matched_keywords | **186/186 빈 배열** | strong_positive 매칭 0 |
| Cash 이벤트 triggering | "holding" 100% + "desk" 79% | soft score로 경계선 진입 |
| Tier-1 confidence 분포 | p50 0.40, 92%가 <0.55 | 대부분 force_tier2 → Gemini 직행 |
| Florence latency | p50 1002ms | BASE_FPS 1.5 불가능, 실효 ~1 fps |
| Gemini accept (cash 2일치) | 2 / 186 (~1%) | FP 필터 잘 작동, recall은 미측정 |
| Human labels | 1 / 186 | **GT 부족이 가장 큰 병목** |

### 튜닝 우선순위

| 우선 | 작업 | Recall 리스크 | 비용/지연 |
|-----|------|------------|----------|
| ★★★★★ | GT 라벨 수집 (Labeling UI 운영) | 0 | 운영 공수 |
| ★★★★★ | Gemini hard-gate 확장 (no physical/no exchange 조기 종료) | 0 | 5분, post-Gemini |
| ★★★★ | Dead code 2차 제거 | 0 | 유지보수성 |
| ★★★★ | 모니터링 자동화 (캡션 어휘 drift, conf 버킷 추적) | 0 | 측정 도구 |
| ★★★ | Motion gating (빈 데스크 차단, "no customer" 27.4% 즉시 컷) | 낮음 | Gemini 호출 -30% |
| ★★★ | Florence `INPUT_SIZE` 320→512 (이중 resize 제거, cash 해상도 2배) | **중간** — "bank" 오인식이 현재 TP trigger | latency +50%, GT 확보 후 A/B |
| ★★ | `CASH_DUAL_PATH_ENABLED` ON | 낮음 | latency 2× (Motion gating과 세트) |
| ★★ | `max_tokens` 96→256 + `MORE_DETAILED_CAPTION` | 낮음 | latency 소폭 ↑ |
| ✗ | `INFERENCE_WORKERS=2` | - | GPU compute-bound, 효과 없음 |
| ✗ | LoRA 학습 | - | GT 1/186 → 데이터 부족 |
| ✗ | Phrase grounding 단독 사용 | - | 실험 완료: hallucination |

### 향후 작업

- Motion gating 구현 (~150 LoC, 1일)
- 운영 대시보드 자동 리포트 (florence 어휘 drift, Gemini reject 사유 분포)
- Dead code 제거 라운드 (`critic_trainer`, `rule_updater`, `dynamic_agent`, `base_detector`)
- GT 30건 이상 확보 후 Florence 512px A/B
- GT 50건 이상 확보 후 LoRA 1차 시도
- RTSP credential 로그 마스킹 (journalctl에 credential 평문 노출)
- SQLite events 테이블 flush 경로 복구 or deprecation 결정

---

## Repo Hygiene

`.gitignore` 제외:
- `.env`, `venv/`, `data/`, `models/`, `scratch/`, `model_cache/`
- `*.log`, 미디어 파일, 모델 가중치 (`.pt`, `.bin`, `.onnx`, `.safetensors`)
- `_tests_archive/`

공유용 템플릿: `.env.example`, `.env.aws`

---

## 로컬 개발 상세 가이드

### 사전 요구사항

| 항목 | 필수 | 확인 |
|------|-----|-----|
| Python 3.10+ (권장 3.12) | ✓ | `python3 --version` |
| FFmpeg | ✓ | `ffmpeg -version` |
| NVIDIA GPU + CUDA | 권장 | `nvidia-smi` |
| Gemini API Key | ✓ (Tier-2용) | [Google AI Studio](https://aistudio.google.com/) |
| RTSP 카메라 | 테스트용 | 없어도 UI 기동 가능 |

### Step-by-Step

```bash
# 1. 클론
git clone https://github.com/WhoAmI125/hio_intelligence_stream.git
cd hio_intelligence_stream

# 2. 가상환경
python3 -m venv venv
source venv/bin/activate

# 3. 의존성
pip install --no-cache-dir -r requirements_gpu.txt   # GPU
pip install -r requirements.txt

# 4. 환경
cp .env.example .env
# GEMINI_API_KEY=your_key 설정
# GPU 없으면 FLORENCE_DEVICE=cpu, RTSP_HWACCEL=
```

### CPU-only 로컬 최적화

```bash
FLORENCE_INPUT_SIZE=256     # 속도 우선
FLORENCE_MAX_TOKENS=64
AUTO_RESTORE_CAMERAS_ON_BOOT=false
# GEMINI_API_KEY 비우면 Tier-2 비활성
```

### 실행

```bash
python start_local.py
```

기동 순서:
- `:8001` DB (즉시)
- `:8000` Model (**최초 실행 시 Florence-2 다운로드 ~1.9GB, 2-5분**)
- `:8002` Frontend (즉시)

### 흔한 문제

| 증상 | 해결 |
|-----|------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` 재실행 |
| `CUDA out of memory` | `.env`에서 `FLORENCE_DEVICE=cpu` |
| Florence 초기화 실패 | `python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)"` |
| 포트 충돌 | `lsof -i :8000` 후 `kill -9 PID` |
| FFmpeg 미설치 | `sudo apt install ffmpeg` / `brew install ffmpeg` |
| RTSP 카메라 없음 | 서버는 정상 기동, 카메라 추가 안 하면 추론 미실행 |

### 코드 경로 참고

- 데이터 경로는 상대경로 (`data/`, `models/`), 프로젝트 루트 기준 자동 생성
- `deploy/` 내 systemd 파일에만 `/home/ubuntu/hio_intelligence_stream` 절대경로 (AWS 전용)
- Windows 호환: `start_local.py`가 `Scripts/python.exe` / `bin/python` 자동 감지. RTSP HW 가속은 Linux NVIDIA 전용

### 최소 하드웨어

| 모드 | CPU | RAM | GPU | 디스크 |
|------|-----|-----|-----|-------|
| GPU (권장) | 4코어+ | 8GB+ | CUDA 4GB VRAM | 10GB+ |
| CPU (테스트용) | 4코어+ | 8GB+ | 불필요 | 10GB+ |

CPU 모드 Florence 추론 ~2-5초/프레임 (GPU 대비 ~10× 느림). 실시간엔 부적합, UI/기능 테스트엔 충분.

---

## 라이선스/주의

사내/프로젝트 목적 운영 코드 문서. 실서버 적용 전 RTSP 접근권한, 개인정보/보안 정책, 저장 보존 정책 점검 필수.
