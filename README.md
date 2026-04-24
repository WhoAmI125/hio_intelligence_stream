# HiO Intelligence Stream — 실시간 CCTV 이상 탐지 시스템

> Florence-2 (Tier-1, On-Device GPU) + Gemini (Tier-2, Cloud API) + 인간 라벨링 기반 GT 수집 파이프라인
> **3-서버 마이크로서비스 · 실시간 RTSP 분석 · g4dn.2xlarge 1대로 멀티 카메라 운영**

---

## 최근 업데이트 (2026-04-17 ~ 현재)

- **Florence 파라미터 재튜닝 (2026-04-18 이후)** — `FLORENCE_INPUT_SIZE=448` (이중 resize 완화), `FLORENCE_MAX_TOKENS=200` (25.9% truncation 위험 해소), `FLORENCE_CAPTION_DETAIL=more` (`<MORE_DETAILED_CAPTION>` task). 속도 위주에서 recall 위주로 전환.
- **Shadow/Critic 런타임 경로 제거 (2026-04-17)** — 181건 기록에도 downstream 0 artifact로 Dead 확인, UI/API/설정/메뉴까지 완전 제거. `config.SHADOW_*` 4개 상수, `main.shadow_agents`, `/api/vlm/shadow/*` 라우트 2개, `/monitor/shadow`, shadow HTML 템플릿, base sidebar 메뉴, `gemini_logs.html` shadow_feedback fallback 모두 정리.
- **Labeling UI 재설계** (`/monitor/labeling`) — 블로킹 `prompt()` 제거, 웰컴 카드 + 좌측 이벤트 리스트 패널(직접 선택) + 한국어 UI + TP/FP/Unclear 기준 가이드 + 단축키 힌트 패널 상시 노출.
- **`display_name` 카메라 라벨 편집** — `camera_id`(내부 식별자, 파일명·이벤트 ID·DB 키에 박힘) 고정, `display_name`만 UI에서 편집 가능. idempotent `ALTER TABLE cameras ADD COLUMN display_name` 마이그레이션 내장.
- **TP 이벤트 clip race fix** (`vlm_api.py`) — validation clip 디스크 유지, 영구 clip 생성 시 `val_entries` 재사용으로 ring buffer evict 경합 제거, `clip_duration_sec` → `validation_clip_sec` 통일. TP 50% clip 유실 문제 해결. `finally` fallback으로 영구 clip 실패 시 `val_clip_path` 자동 승계.
- **Gemini 로그 미디어 strip 통일** — `clip | ROI | gemini` 3열 고정 (fire/violence도 overlay 없을 때 plain/thumbnail fallback하되 라벨은 불변).
- **Florence 튜닝 실측 번들** — 1000 캡션 분석 결과 cash 어휘 실측 **0.00%**, latency p50 **1002ms**, 이중 resize 경로 확인. `matched_keywords` 186/186 빈 배열. 트리거 실체는 `holding` 100% + `desk` 79% + `bank` 20% soft-score 누적. 상세: `scratch/tuning_bundle_20260417.tar.gz`

> **현재 강화 방향**: Gemini는 hard-gate v.26.04.17로 잘 기능(총 186건 중 accept 약 1%). Florence의 Tier-1 스크리너 역할과 GT 라벨 수집이 병목. **Motion gating + GT 수집**이 다음 우선순위. LoRA는 GT 1/186으로 데이터 부족해 비활성 유지.

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
17. [코드 규모](#코드-규모-2026-04-21-실측)
18. [트러블슈팅](#트러블슈팅)
19. [이벤트 생명주기 (End-to-End Trace)](#이벤트-생명주기-end-to-end-trace)
20. [실측 기반 한계와 다음 단계](#실측-기반-한계와-다음-단계)
21. [Repo Hygiene](#repo-hygiene)
22. [로컬 개발 상세 가이드](#로컬-개발-상세-가이드)

---

## 프로젝트 개요

RTSP CCTV 스트림을 실시간으로 분석하여 3개 시나리오를 탐지합니다:

- `cash` — 현금 거래/금전 전달
- `fire` — 화재/연기
- `violence` — 폭력/충돌

### 2-Tier 설계

| Tier | 역할 | 모델 | 비용 |
|------|-----|------|------|
| Tier-1 | 고속 스크리닝 (장면 캡션 → 키워드 soft score) | **Florence-2-large 단일** (PyTorch fp16, CUDA autocast) | 추론당 p50 1848ms/1299ms (T4, 448px, tok=200, `<MORE_DETAILED_CAPTION>`, beams=1) |
| Tier-2 | 경계선 케이스 최종 판정 (Korean-aware hard/soft rules) | **Gemini 3.1 Flash Lite Preview 단일** (Cloud API) | API 호출당 ~$0.001, video_only 모드 p50 ~4-6s |

> **중요**: Tier-1은 **Florence-2만** 사용. YOLO/YOLO-pose/CLIP/MediaPipe/Grounding DINO/Qwen-VL 등은 **현재 코드베이스에 import/호출 0건**. `other_v2/` 설계 문서의 3-Tier 계획(YOLO26s-Pose + CLIP + Qwen2.5-VL-3B)은 **미실현 프로토타입**이며 본 저장소와 무관합니다. 자세한 내역은 [현재 구현되어 있지 않은 것](#현재-구현되어-있지-않은-것-오해-방지) 섹션 참조.

### 핵심 설계 원칙

- **Caption Sharing**: 프레임당 Florence-2 추론 1회 → 자유 캡션 텍스트를 3개 시나리오(`CaptionAnalyzer.analyze()`)가 CPU regex로 공유 분석. GPU round-trip 3→1 감소.
- **Word-boundary regex matching**: `re.compile(r'\b{kw}\b', re.IGNORECASE)` 패턴을 전역 캐시(`_compiled_patterns`)에 lazy 컴파일. substring 오탐(`billboard`가 `bill`로 매칭되는 현상) 방지.
- **4-layer scoring**: `strong_positive` × 0.3 + `moderate_positive` × 0.1 + `context_phrases` × 0.15 + `negative` × -0.3 → clamp(0, 1). 시나리오별 weight 미세 조정(violence strong 0.35 / fire strong 0.4).
- **Neutralizing phrases**: "fire extinguisher"는 `fire` strong 신호를 무효화 (mask-then-search 2-pass). cash의 경우 "cash"는 "cash register" 안에 있을 때만 neutralize, `cash register` 자체는 강한 신호로 보존.
- **H2H fallback detection**: `strong_positive` 매치가 0이어도 `LOCATION {counter, cashier, checkout, front desk, reception, drawer, bank, teller, store, lobby} ∩ ACTION {handing, holding, passing, reaching, exchanging, giving, receiving, placing, picking}`이 동시에 매칭되면 `is_detected=True` 승격. cash 이벤트의 대부분이 이 경로.
- **Escalation by threshold**: Tier-1 confidence `< TIER2_CASH_THRESHOLD(0.55)` 이면 `evidence_router.py:1126` `force_tier2=True`로 설정되어 margin gate 무시하고 자동 `GEMINI_VIDEO`.
- **Gemini가 cash 판정 전담**: Florence 캡션에는 cash 어휘 실측 **0.00%** (1000건/0건). Tier-1은 "사람+카운터" 같은 soft trigger만 제공, 실제 판정은 Gemini video + hard rule H1/H2/H3 + soft rule S_STRONG_1~3이 담당.
- **인간 라벨링으로 GT 구축**: `/monitor/labeling` UI로 이벤트 단위 TP/FP/Unclear 수동 라벨링 → `event.human_feedback` JSON 필드에 canonical 저장. 후속 튜닝·A/B 측정·LoRA 품질 게이트 근거.
- **Fail-open API 정책**: Gemini API 오류 시 `True, 1.0, "Validation disabled", event_type` 반환. 이벤트를 놓치는 것보다 overconfirm이 낫다는 판단.

### 현재 운영 범위 (2026-04-17 실측)

| 항목 | 값 |
|------|-----|
| 인스턴스 | g4dn.2xlarge 1대 (T4 15GB + Xeon 8 vCPU + 32GB RAM) |
| 카메라 | 2대 (일산·금촌) |
| GPU VRAM 사용 | **2,165 MiB / 15GB (14%)** — Florence-2 fp16 ~1.5GB + CUDA context + NVDEC |
| GPU SM 활용률 | 평균 **65%** (45~83% 범위), 온도 39°C |
| CPU | `model_server` 프로세스 144% CPU, RSS 9.2GB, system user 21%, load avg 2.13/8 |
| RAM 사용 | 11.6 GB / 30 GB (37%) |
| 실효 처리율 | **~1.0 fps/camera** (Florence 1002ms p50로 compute-bound) |
| 이벤트 생성 | 하루 수십~수백 건, cash 2일치 186건 기준 |
| Gemini 호출 | 호출당 ~4-6초, accept 비율 2/186 (~1%) |
| 월 비용 | GPU 인스턴스 약 $400, Gemini API 수만원 |

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
│   ├── base_detector.py                   # [Dead Code] YOLO 기반 detector 구현체. 호출 경로 0
│   ├── episode_manager.py                 # [Dead Code] import 없음, throwaway Episode만 사용
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

### 현재 구현되어 있지 않은 것 (오해 방지)

README와 저장소를 훑으면 다음 키워드가 눈에 띄지만, **모두 운영 파이프라인에 연결되어 있지 않습니다**:

| 파일 / 외부 경로 | 실제 상태 | 왜 보이는가 |
|----------------|---------|-----------|
| `model_server/base_detector.py` (302 LoC) | **Dead Code** — `from ultralytics import YOLO`, `_load_yolo_model()`, `_yolo_to_detections()` 전부 구현돼 있음. `main.py`에서 import 0건, 호출 0건. `requirements.txt`에 `ultralytics` 없음 (이미 정리) → import 시 ModuleNotFoundError | 과거 YOLO 기반 Tier-0 detector 실험 흔적 |
| `/home/ubuntu/other_v2/` (이 repo **밖**) | **v2 설계 문서 + HTML 프로토타입만 존재**. 코드 없음. YOLO26s-Pose, CLIP ViT-L/14, Qwen2.5-VL-3B 같은 수치가 적힌 `ARCHITECTURE.md`/`COMPARE.md`는 계획서일 뿐 | 이전 분석(`analysis_deploy.md`, Opus 4.6)이 v2를 v1으로 착각한 전력 있음 |
| `<OD>` / `<OPEN_VOCABULARY_DETECTION>` / `<CAPTION_TO_PHRASE_GROUNDING>` (Florence-2 task tokens) | Florence 어댑터에 메서드만 존재, 파이프라인 미연결. **실험 완료: phrase grounding은 "best-match localizer"로 hallucination** | Florence-2가 자체 제공하는 기능 |
| `peft` / LoRA | `LORA_ENABLED=false`. `peft.PeftModel` 런타임 로드 지원 코드는 존재하나 어댑터 파일 없음 | 학습 파이프라인은 오프라인 수동 트리거 |
| `critic_trainer.py` + `rule_updater.py` | **Dead Code** — Shadow 제거(2026-04-17) 이후 호출 경로 단절 | 과거 "자율 진화" 실험 흔적 |
| `scenarios/prompts.py`의 `*_SCENARIO` 문자열 상수 | `get_scenario_prompt()`는 호출되나 반환된 prompt 문자열이 Florence 어댑터에서 무시됨 (task token만 사용) | 레거시 API 호환용 |

**실제로 GPU 추론에 연관된 모델은 Florence-2-large 하나뿐**. Tier-2 Gemini는 Cloud API. 그 외 모든 VLM/detector/grounding 모델은 미연결이거나 미구현입니다.

### Dead Code (제거 대기, 2026-04-17 Shadow 제거 후 경로 단절)

| 파일 | LoC | 상태 | 비고 |
|------|-----|------|------|
| `model_server/evolution/critic_trainer.py` | 191 | Dead | Shadow에서만 호출되던 `train()` 트리거 사라짐 |
| `model_server/evolution/rule_updater.py` | 337 | Dead | `apply_feedback_to_rules()` 호출 경로 없음 |
| `model_server/agents/dynamic_agent.py` | 295 | Dead | 초기화만 존재 |
| `model_server/base_detector.py` | 302 | Dead | **YOLO 기반 detector**, 정의만 존재. `main.py`에서 호출 0 |
| `model_server/episode_manager.py` | 421 | Dead | `main.py`에서 import 없음. throwaway Episode만 생성 |
| `db_server/models.py` | 102 | Dead | Django ORM 스키마. SQLite WAL로 대체됨 |
| `frontend_server/views.py` | 1 | Empty | |
| `fix_html.py` | - | Dead | 일회성 마이그레이션 스크립트 |

**제거 시 예상 LoC 감소**: ~1,800 lines (~9%).

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

| 항목 | config.py 기본값 | 운영 `.env` | 설명 |
|------|-------|-------------|------|
| 모델 | `microsoft/Florence-2-large` | 동일 | ~770M 파라미터 VLM encoder-decoder (DaViT vision tower + BART-like decoder) |
| 로드 | HF `AutoModelForCausalLM` | `trust_remote_code=True` | 캐시 `MODEL_SERVER_MODELS_DIR` (기본 `models/hf/`) |
| 백엔드 | `pytorch` | `pytorch` | OpenVINO backend은 `_init_openvino` stub 존재, `_infer_openvino`는 PyTorch로 fallback |
| 정밀도 | GPU fp16 / CPU fp32 | fp16 CUDA | `torch_dtype=torch.float16` + 추론 시 `torch.autocast(device_type='cuda', dtype=torch.float16)` |
| 입력 크기 | 448 | **448×448** | `FLORENCE_INPUT_SIZE=448`. ⚠️ HF processor가 내부 768 bilinear upscale — 이중 resize 경로는 여전함 |
| max_tokens | 512 | **200** | 이전 96 → 200으로 승격 (25.9% 캡션 cap 근접 문제 해소). 여전히 cap 아래 95th percentile |
| num_beams | 3 | **1** | Greedy decoding. `do_sample=False`, `model.generate(max_new_tokens=200, num_beams=1)` |
| caption_detail | `more` | **`more`** | `<MORE_DETAILED_CAPTION>` task token 사용 |
| LoRA | `LORA_ENABLED=false` | `false` | `peft.PeftModel` 런타임 로드 지원 (`adapter_config.json` 존재 + 플래그 ON이면 `PeftModel.from_pretrained`) |

#### Florence-2 추론 Hot Path (`florence_adapter.py:313~400`)

```
BGR numpy (카메라 프레임 최대 1280×720)
  ↓ BaseVLMAdapter.preprocess_image(image)
  ↓   cv2.resize(448×448, INTER_AREA) + BGR→RGB             [1차 resize]
  ↓   → np.uint8 RGB array
  ↓ PIL.Image.fromarray(image_rgb)
  ↓ AutoProcessor(text=task_token, images=pil_image, return_tensors="pt")
  ↓   internal bilinear upscale → 768×768                    [2차 resize]
  ↓   text tokenize (task prefix + optional text_input)
  ↓ inputs.to(cuda) → pixel_values + input_ids
  ↓ with torch.inference_mode() + torch.autocast(cuda, fp16):
  ↓   model.generate(
  ↓     **inputs,
  ↓     max_new_tokens=200, num_beams=1, do_sample=False
  ↓   )
  ↓ processor.batch_decode(generated_ids, skip_special_tokens=False)
  ↓ processor.post_process_generation(
  ↓     generated_text, task=task_token, image_size=pil_image.size
  ↓ )
  ↓ → 자유 캡션 텍스트 (task 따라 dict{task: str|dict} 파싱)
  ↓ CaptionAnalyzer.analyze(caption, ScenarioType) × 3      [CPU regex μs]
  ↓ → {is_detected, confidence, matched_keywords, object_hints, evidence}
```

- **`_run_task(image, task, text_input)`**: 내부 helper로 `<OD>`/`<OCR>`/`<CAPTION_TO_PHRASE_GROUNDING>`/`<OPEN_VOCABULARY_DETECTION>` 등도 호출 가능.
- **`preprocess_image`**: `BaseVLMAdapter`에서 crop polygon zone 지원 (ROI crop 경로는 dual-path OFF라 현재 미호출).

#### 지원 task tokens

| Task | 운영 사용 | 메서드 | 설명 |
|------|---------|--------|------|
| `<MORE_DETAILED_CAPTION>` | ✅ 기본 | `infer()` | `FLORENCE_CAPTION_DETAIL=more` → `task_map['more']` |
| `<DETAILED_CAPTION>` | 전환 가능 | `infer()` | `FLORENCE_CAPTION_DETAIL=detailed` |
| `<CAPTION>` | 전환 가능 | `infer()` | `FLORENCE_CAPTION_DETAIL=basic|caption` |
| `<OD>` | 미연결 | `detect_objects()` | COCO 91 클래스 detection. 실험 시 person/chair/monitor만 반환 |
| `<DENSE_REGION_CAPTION>` | 미연결 | `_run_task()` | 씬 수준 region captioning |
| `<CAPTION_TO_PHRASE_GROUNDING>` | 미연결 | `ground_phrase()` | 자유 phrase → bbox. **hallucination 확인** — 존재 여부 무관하게 가장 유사한 영역 반환 |
| `<OPEN_VOCABULARY_DETECTION>` | 미연결 | `_run_task()` | "banknote / cash / wallet / korean banknote"가 전부 같은 영역을 서로 다른 라벨로 반환 (bbox 영역 1.7~1.9% 동일) |
| `<OCR>` / `<OCR_WITH_REGION>` | 미연결 | - | 텍스트 추출 |
| `<REGION_PROPOSAL>` | 미연결 | - | 객체 제안 |

> **결정적 증거 (2026-04-17 실험)**: FP 5건의 cash 이벤트 thumbnail에 `<OPEN_VOCABULARY_DETECTION>` 테스트 결과 — `banknote` 1.87%, `paper money` 1.84%, `cash` 1.75%, `wallet` 1.72%, `korean banknote` 1.88%가 **전부 1.7-1.9% 동일 영역**을 반환. Florence는 abstention 없이 "best-match localizer"로 동작. GT calibration 없이는 noise 증폭기.

### Tier-2: Gemini Vision (Cloud API)

| 항목 | 값 | 근거 |
|------|-----|-----|
| 모델 | `gemini-3.1-flash-lite-preview` | `GEMINI_MODEL` .env (config.py 기본 `gemini-2.5-flash-lite`) |
| 프롬프트 버전 | `evidence-v1.1` / 본문 `v.26.04.17` | `EVIDENCE_PROMPT_VERSION` 상수 + reason_bullets 첫 줄 표식 |
| Temperature | 0.1 | deterministic 경향 |
| top_k / top_p | 1 / 1.0 | 거의 greedy |
| max output tokens | 1500 | `GenerateContentConfig` |
| response mime | `application/json` | JSON 강제 |
| 타임아웃 | 180초 (`GEMINI_TIMEOUT_SEC=180`) | config 기본 30초보다 6× |
| 동시 호출 | 1 (`GEMINI_MAX_CONCURRENT=1`) | `asyncio.BoundedSemaphore` |
| 기본 모드 | `EVIDENCE_MODE=video_only` | storyboard/image fallback 비활성 |
| 입력 clip | 1280×720 H.264 CRF23, preset fast, `+faststart`, yuv420p | FFmpeg, ~3Mbps, 10초 |
| 실패 정책 | Fail-Open | `not enabled or not client` → `return True, 1.0, "Validation disabled"` |

#### 검증 모드 (`validate_event_evidence(packet, mode)`)

| 모드 | 우선순위 | 비고 |
|------|---------|------|
| `hybrid` | cash: storyboard → video → image. 그 외: video → storyboard → image | 기본이 아님 |
| `video_first` | video → storyboard → image | |
| **`video_only`** | video 단독 | **운영 기본값**, fallback 없음 |
| `images_first` / `storyboard` | storyboard (최대 12 keyframes) → image → video | |
| `image` | 단일 프레임 → storyboard → video | |

#### Gemini 통합 프롬프트 (`DEFAULT_UNIFIED_PROMPT`, `gemini_validator.py:37~301`)

251줄 규모의 통합 프롬프트로 **4가지 event_policy**를 판정:

1. `CASH_TRANSACTION` — 현금거래
2. `THREAT_TO_CASHIER` — 캐셔 위협/폭력
3. `FIRE_ALERT` — 실제 화재
4. `STAFF_CASH_THEFT_SUSPECT` — 내부자 현금 절도 의심
5. `NONE` — 해당 없음

##### 한국 원화 맥락 주입

```
Currency context: This environment uses Korean Won (KRW).
Korean banknotes have distinctive colors
(blue 1000, green 5000, orange 10000, yellow 50000)
and are larger than receipts. Do not claim "Korean cash"
unless you see these traits.
```

##### Visual hint overlay (주의사항)

Gemini는 yellow polygon(`CASHIER ZONE`) + cyan polygon(`DRAWER`)을 **attention hint**로만 보고, polygon 자체는 reason_bullets에 묘사 금지. "rectangle/frame overlay/yellow box" 같은 표현 금지.

##### CASH_TRANSACTION Hard Rules (H1 ∧ H2 ∧ H3 전부 PASS 필요)

| 규칙 | PASS 조건 | FAIL 조건 |
|------|---------|---------|
| **H1. CASH_VISUAL_CONFIRM** | banknote-like printing/color/pattern, 분명한 지폐 1장+, 또는 지폐 counting/peeling | plain white slip, 단단/반사 객체(카드/기기), smartphone 화면, 모호함 |
| **H2. OWNERSHIP_TRANSFER** | 객체가 한 사람 손 → 다른 사람 손으로 이동. 교환 방향 가시. video면 짧은 가림 허용 단 전후로 객체 보여야 함 | 한 사람만 다룸, 심한 가림 |
| **H3. ACTIVE_TRANSACTION_CONTEXT** | counter/register 근처. 직원이 register 조작/drawer 오픈/상품 처리/결제/상호작용 | counter/register 없음, 직원이 개인 행동(식사/개인폰/잡담) 중 |

##### CASH_TRANSACTION Soft Rules (H1-H3 통과 후 평가)

| 등급 | 규칙 | 설명 |
|------|-----|------|
| **S_STRONG_1** | Cash drawer 명확히 열림 or 객체가 cash slot/till에 삽입 | 40점 → `safe_drawer` |
| **S_STRONG_2** | 직원이 지폐 counting/peeling/aligning | 40점 → `money_likelihood` |
| **S_STRONG_3** | 직원이 거스름돈(지폐·동전) 반환 | 40점 → `hand_to_hand` |
| S_WEAK_1 | 전달 순간이 명확히 가시 | |
| S_WEAK_2 | 손님이 교환 후 돌아서거나 떠남 | |
| S_WEAK_3 | 양쪽 모두 객체를 봄 | |

##### Policy Scores

```json
{
  "money_likelihood":  0 | 25 | 40,   // 0=H1 fail, 25=banknote traits, 40=S_STRONG_2
  "hand_to_hand":      0 | 35 | 40,   // 0=H2 fail, 35=transfer visible, 40=S_STRONG_3
  "safe_drawer":       0 | 40,        // 0=S_STRONG_1 fail, 40=drawer open/insert
  "non_cash_penalty":  -30 | -15 | 0, // -30=phone/card/tablet, -15=white slip, 0=OK
  "total_score": sum
}
```

##### Decision Rule (모두 참일 때만 `CASH_TRANSACTION` 허용)

1. H1, H2, H3 전부 PASS
2. S_STRONG_1 / S_STRONG_2 / S_STRONG_3 중 최소 1 PASS
3. `safe_drawer=40` OR `money_likelihood=40` OR `hand_to_hand=40` 중 하나 이상

→ False면 `event_policy=NONE`, `decision=FALSE_POSITIVE` (upstream이 cash-like인 경우) or `NOT_APPLICABLE`.

##### THREAT_TO_CASHIER Scores

```json
{
  "mandatory_score": 0, "supporting_score": 0, "negative_score": 0,
  "total_score": 0,
  "threat_level": 0-4,
  "threat_label": "CLEAR | TENSE | INTIMIDATION | PHYSICAL | WEAPON"
}
```

Severity 매핑: `CLEAR=none`, `TENSE=low`, `INTIMIDATION=medium`, `PHYSICAL=high`, `WEAPON=critical`.

##### FIRE_ALERT Scores

```json
{"fire_confidence": 0.0-1.0, "smoke_confidence": 0.0-1.0}
```

실제 flame / smoke 가시 시에만 valid. 소화기/알람/표지만으로는 NONE.

##### STAFF_CASH_THEFT_SUSPECT Scores

```json
{
  "suspicion_level": 0-3,
  "suspicion_label": "none | low | medium | high",
  "cash_box_access": true|false,
  "looks_around": true|false,
  "moves_cash_to_personal_area": true|false,
  "customer_present": true|false,
  "paperwork_or_reconciliation": true|false
}
```

##### 응답 파싱 (`_parse_new_response_format`)

| 필드 | 타입 | 역할 |
|------|-----|------|
| `event_policy` | str | 5중 선택 |
| `event_type_detected` | str | `cash|violence|fire|staff_cash_theft|none` |
| `is_valid_event` | bool | 최상위 판정 |
| `decision` | str | `TRUE_POSITIVE|FALSE_POSITIVE|NOT_APPLICABLE` |
| `severity_label` | str | `none|low|medium|high|critical` |
| `confidence` | float | 0.0-1.0 |
| `policy_scores` | dict | 위 4개 정책별 score 구조 |
| `reason_bullets` | list[str] | 첫 줄에 `- [PROMPT_VERSION] v.26.04.17` 강제, "appears / likely / seems" 같은 hedge 금지 (있으면 NONE) |

##### Event type correction

Gemini가 업스트림 `original_event_type`과 다른 `event_type_detected`를 내면 `corrected_event_type`로 교정. 단 `violence → cash` 방향은 중복 방지 목적으로 **차단** (is_valid=False로 강제).

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

### CaptionAnalyzer 매칭 엔진 (`scenarios/base_scenario.py:240~547`)

#### 키워드 계층 (시나리오별)

| 계층 | Cash 가중치 | Violence 가중치 | Fire 가중치 | 역할 |
|-----|-----------|--------------|-----------|------|
| `strong_positive` | **0.3** | 0.35 | 0.4 | 직접 증거 키워드 (cash/money/banknote, fight/punch, fire/flame) |
| `moderate_positive` | **0.1** | 0.15 | 0.15 | 간접 증거 (counter, teller, drawer, holding, bank) |
| `context_phrases` | **0.15** | 0.45 | 0.5 | 복합 구문 ("handing to customer", "hitting someone", "on fire") |
| `negative` | **-0.3** | -0.25 | -0.2 | 반증 (credit card, handshake/hugging, sunset/candle/fireplace) |
| `neutralizing_phrases` | 무효화 | 무효화 | 무효화 | `strong` 매치를 컨텍스트 보고 무효화 (e.g. `fire extinguisher`, `billboard`) |
| `object_hints` | 0점 | - | - | 점수 0, Tier-2 metadata로만 전달 (Florence가 본 객체 추론 힌트) |

#### 신뢰도 계산 (`analyze()`)

```python
score = (
    len(strong_matches)   * weights['strong']   +
    len(moderate_matches) * weights['moderate'] +
    len(context_matches)  * weights['context']  +
    len(negative_matches) * weights['negative']
)
confidence = max(0.0, min(1.0, score))  # clamp
```

`is_detected` 판정 트리거 (`is_detected = confidence > 0 and has_signal`):
- `len(strong_matches) > 0` OR
- `len(context_matches) > 0` OR
- **H2H fallback**: `has_location ∧ has_action` (moderate 매치 안의 LOCATION set과 ACTION set 교집합)

```python
_LOCATION_KEYWORDS = {'counter', 'cashier', 'checkout', 'front desk',
                      'reception', 'drawer', 'bank', 'bank teller',
                      'teller', 'store', 'lobby'}
_ACTION_KEYWORDS   = {'handing', 'holding', 'passing', 'reaching',
                      'exchanging', 'giving', 'receiving',
                      'placing', 'picking'}
has_h2h = bool(_LOCATION_KEYWORDS & set(moderate_matches)) \
          and bool(_ACTION_KEYWORDS & set(moderate_matches))
```

#### Neutralizing phrases (2-pass masking)

```python
# 1차: strong_matches에 해당 키워드 있는지 확인
# 2차: 해당 키워드 주변에 neutralizing 구문이 모두 덮고 있다면
#      캡션 복사본에서 neutralizing phrase 영역을 공백으로 mask
# 3차: masked 캡션에 키워드가 여전히 보이면 neutralize 안 함,
#      그렇지 않으면 strong_matches에서 제거
```

예시:
- `fire`: `fire extinguisher`, `fire exit`, `fire escape`, `fire alarm`, `fire department`, `fire hydrant`, `fire truck`, `fire safety`, `fire door`, `fire hose`, `fire prevention`
- `cash`: `cash register`만 (그러나 `cash register` 자체는 강한 cash 신호라 neutralize 안 됨 — 주석에 명시)
- `bill`: `billboard`

#### Regex caching

전역 class 변수 `_compiled_patterns: Dict[str, re.Pattern] = {}`. `\b{escape(kw)}\b` + `re.IGNORECASE`로 lazy 컴파일. word-boundary로 substring 오탐(`billing`이 `bill` 매칭되는 문제) 차단.

#### Cash H2H (Hand-to-Hand) 키워드 상세

**strong_positive** (9개):
```
cash, money, banknote(s), currency, dollar, won, coins,
paying, payment, transaction, cash register, paper money, cash payment
```
*제거됨*: `bill/bills/notes/change` — 다의어 FP 심각 (`restaurant bill`, `taking notes`, `change clothes`). `bill`만 moderate로 강등.

**moderate_positive** (~30개): 장소 + 동작 2축으로 구성
- 장소: `counter, cashier, checkout, front desk, reception, drawer, receipt, bank, bank teller, teller, store, lobby, customer`
- 동작: `handing, holding, passing, reaching, exchanging, giving, receiving, placing, picking`
- 소지품: `wallet, purse, envelope`
- 강등: `bill, bills`
- H2H: `two men, two people, two women, two persons, facing each other, across the counter, across the desk, leaning over, leaning forward`

**context_phrases** (~50개): H2H 전달 구문
- 최상위: `from one person to another`, `between two people`, `staff to customer`, `customer to staff`, `handing/giving/receiving to/from customer/staff`, `hand to hand`, `hand-to-hand`
- 전달: `handing a piece of paper`, `passing something`, `giving an object`, `handing over`, `reaching across`
- 현금 직접: `giving/receiving/counting money`, `counting bills`, `holding cash`, `cash drawer`
- 서랍: `opening drawer`, `putting into drawer`, `taking from drawer`
- 지갑: `reaching into wallet`, `pulling out wallet`, `opening wallet`, `reaching into pocket`
- 접객: `face to face`, `talking to customer`, `helping a customer`, `serving a customer`, `waiting at the counter`, `standing at the counter`

*제거됨* (과도한 FP):
- `holding a small/black/brown/white/blue/green/red ...` — 모든 물체에 매칭
- `holding an object/objects` — 3794건 중 대부분 비현금
- `holding a paper/papers/cover/file/bag` — 서류/가방 오탐
- `placing something`, `picking something`, `picking up` — 일반 동작

**object_hints** (~25개): 점수 0점, Gemini에 `detection_metadata.object_hints`로 전달
- 종이 계열: `paper, piece of paper, folded paper, envelope, receipt, document`
- 현금: `cash, money, bill, bills, banknote, coin, coins, change`
- 카드: `card, credit card, debit card`
- 지갑: `wallet, purse`
- Florence 오인 대상: `phone, mobile, remote, remote control, small object, black object, object, cover, bag, file, papers, yellow/blue/brown object`

**negative** (현재 cash는 축소됨):
```
credit card, debit card, card reader, swipe, contactless, terminal
```
*제거됨*: `phone/mobile` — Florence가 현금을 "phone"으로 자주 오인하므로 Tier-2가 판별 담당.

#### Violence 키워드 (요약)
- strong: `fight(ing), punch(ing), attack(ing), struggle(ing), violent, violence, assault(ing), shove, slap, kick`
- moderate: `aggressive, angry, confrontation, conflict, restrain, threaten(ing), yelling, screaming, falling down, knocked`
- context: `hitting someone/person/him/her, pushing someone, grabbing someone, pulling hair, throwing punch, physical altercation`
- negative: `handshake, hug(ging), friendly, greeting, playing, children, laughing, smiling, waving, keyboard, button, typing`

#### Fire 키워드 (요약)
- strong: `fire, flame(s), burning, blaze, smoke, ignite, combustion, inferno`
- moderate: `orange/red glow, haze, hazy, emergency, sprinkler, charred, scorched, smoldering`
- context: `on fire, catching fire, thick smoke, smoke rising/coming/billowing, flames spreading`
- neutralizing: 위 목록 참조 (소화기/비상구/경보 등 12개)
- negative: `lamp, screen, monitor, reflection, sunset, warm lighting, neon, candle, fireplace, cigarette, no smoking sign, extinguisher`

#### 실측 행동 (cash 이벤트 186건 분석, 2026-04-17 기준)

- `matched_keywords=[]` — **186/186 전부 빈 배열** (strong_positive 매칭 0건)
- `holding` 캡션 포함 — **186/186 (100%)** ← cash 이벤트의 실질 트리거
- `desk` 147건 (79.0%), `bank` 38건 (20.4%), `counter` 35건 (18.8%), `teller` 22건 (11.8%)
- `standing` 98건, `sitting` 97건
- **캡션에 `cash`/`bill`/`banknote`/`money`/`wallet`/`drawer` 매칭 — 0건 (0.00%)**
- Tier-1 confidence p50 = **0.40** (모두 0.30~0.70 경계)
- Human labeled — **1/186** (GT 부족이 가장 큰 병목)

전체 1000건 캡션 기준 어휘 분포:
- `desk` 932 (전체), `bank` 312 (31.2% — Florence가 호텔 데스크를 "bank"로 부름), `teller` 59, `counter` 26, `standing` 37, `holding/hand` 12
- Unique caption 382/1000 (38.2%) — 상위 3개가 234건(23.4%) 반복 (정적 호텔 프론트 반복 추론)

#### Cash H2H (Hand-to-Hand) 탐지 (설계)

```
H2H 양성 = 위치 키워드 (counter/cashier/desk/teller/register/drawer)
         AND
         행동 키워드 (handing/holding/passing/reaching/exchanging)
```

실측상 이 경로로 대부분 이벤트가 conf 0.30~0.55 달성 → 전부 Gemini로 에스컬레이션.

### EvidenceRouter 판정 로직 (`evidence_router.py:1100~1170`)

```python
# 1. hard-risk escalation (fire/violence 전용)
if event_type in self.hard_tier2_events and avg_conf < self.hard_tier2_max_conf:
    action = ACTION_GEMINI_VIDEO
    force_tier2 = True
    baseline_reason = f"Hard-risk escalation for {event_type}: conf={avg_conf:.2f} < {hard_tier2_max_conf:.2f}"

# 2. 시나리오 threshold 미달 (cash: 0.55)
elif avg_conf < scenario_threshold:
    action = ACTION_GEMINI_VIDEO
    force_tier2 = True
    baseline_reason = f"Below Tier2 threshold for {event_type}: conf={avg_conf:.2f} < {scenario_threshold:.2f}"

# 3. SKIP gate (고신뢰 + 고안정)
elif avg_conf >= self.skip_confidence and stability >= self.skip_stability:
    action = ACTION_SKIP
    baseline_reason = f"High confidence/stability skip: conf={avg_conf:.2f}>= {skip_confidence:.2f}, stab={stability:.2f}>= {skip_stability:.2f}"

# 4. Q-learning 기반 max-action + margin gate
else:
    action = max(ACTIONS, key=lambda x: q.get(x, -9999.0))
    baseline_reason = f"max_q_action[{q_source}]"  # q_source = 'learned+heuristic' or 'heuristic'

# 5. Margin gate: Gemini 액션이 SKIP보다 router_margin 이상 우월해야 유지
if action in TIER2_ACTIONS and not force_tier2:
    margin = q.get(action, 0.0) - q.get(ACTION_SKIP, 0.0)
    if margin < self.router_margin and event_type not in hard_tier2_events:
        action = ACTION_SKIP
        baseline_reason = f"Margin gate: best_gemini_margin={margin:.3f} < router_margin={router_margin:.3f}"
```

#### 액션 공간

| 액션 | 비용 (Q-table) | 의미 |
|------|------|-----|
| `ACTION_SKIP` | 0.0 | Tier-2 생략, Tier-1만으로 판정 |
| `ACTION_GEMINI_IMG` | 0.10 | 단일 프레임 Gemini 호출 (저비용) |
| `ACTION_GEMINI_VIDEO` | 0.20 | 비디오 clip Gemini 호출 (운영 기본) |
| `ACTION_HUMAN_QUEUE` | 0.35 | 운영자 큐로 올림 (현재 연결된 UI 없음) |

#### Q-value 스코어링

- `score_actions(episode, state)` — 11차원 feature vector(avg_conf, stability, prior_calls, durations, detection_count, ...) → 4개 action별 Q-value
- Heuristic 기본값 + optional `self.policy_model` (joblib으로 로드 가능한 classifier)
- `q_source = 'learned+heuristic'` if `policy_model else 'heuristic'`

#### Critic (shadow-only, 현재 비활성)

- `LightGBM` critic trainer는 `CRITIC_ENABLED=false` 기본값으로 꺼짐
- Shadow 제거로 training 트리거 없음 → artifact 0

#### 실측 영향

- cash 이벤트 **92%가 conf < 0.55** → `force_tier2` 경로 (margin gate 무관)
- SKIP 게이트(`≥ 0.85 conf ∧ ≥ 0.90 stability`)에 도달하는 경우는 거의 없음 — `p50=0.40`이라 도달 자체 어려움
- `hard_tier2_events = {'fire', 'violence'}`, `hard_tier2_max_conf = 0.95` — 사실상 모든 fire/violence가 Tier-2로
- Router log는 `router_steps.jsonl`에 append-only (critic training data 목적)

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

### StreamManager (`stream_manager.py`, 683 LoC)

- **Thread model**: 카메라별 전용 `_reader_loop` daemon thread + ring buffer
- **Ring buffer**: `collections.deque(maxlen)` 각 요소는 `{"frame": np.ndarray, "mono_ts": float}` tuple
  - `maxlen = effective_fps × buffer_seconds` (기본 `12 × 30 = 360 frames`)
  - `mono_ts = time.monotonic()` — 시스템 시간 변경에 내성
- **Sampling rate**:
  - 메인 추론 샘플링: `BASE_FPS=1.5` → `BURST_FPS=4.0` (탐지 후 `BURST_DURATION_SEC=3.0`)
  - Clip buffer 저장: `clip_buffer_fps=12.0` → `clip_buffer_burst_fps=15.0` (고해상도 clip 재현 목적)
- **Reconnection**: 지수 백오프 `1s → 2s → 4s → 8s` + `±jitter` (OpenCV `VideoCapture` 재오픈)
- **HW 가속** (`RTSP_HWACCEL=cuda`):
  - FFmpeg `-hwaccel cuda -hwaccel_device 0`로 NVDEC 사용
  - `RTSP_HWACCEL_ALLOW_FALLBACK=true`로 디코더 미지원 시 CPU로 자동 전환
- **중복 방지** (`_rtsp_key` canonical 정규화): 동일 RTSP URL이 이미 열린 상태면 새 `start` 요청 무시 (`Assertion fctx->async_lock failed` 디코더 충돌 방지)
- **API**: `get_frame(camera_id) → latest frame`, `get_clip_frames(camera_id, anchor_mono_ts, duration) → List[entries]`

### InferenceScheduler (`inference_scheduler.py`, 211 LoC)

- **Dispatcher 스레드** (`_dispatch_loop`):
  - `dispatcher_sleep_sec = 0.02` (20ms 폴링)
  - 각 카메라 순회, `state.running=True ∧ pending=False ∧ inflight=False`면 job 후보
  - Target FPS: `target_fps = max(base_fps, burst_fps if active)`, `interval = 1.0 / target_fps`
  - `now - last_submit_ts < interval`이면 skip (rate limit)
  - `stream_manager.get_frame(camera_id)` 최신 프레임 가져와 `InferenceJob` 생성
  - `queue.put_nowait(job)` (Full이면 `jobs_dropped++`)
- **Worker 스레드** (`_worker_loop`, `INFERENCE_WORKERS=1`):
  - `_queue.get(timeout=0.2)` blocking
  - **Stale job 방어**: `state.run_id != job.run_id`이면 drop (카메라 재시작으로 이전 세션 잡 폐기)
  - `process_fn(camera_id, frame, state, started_at)` 콜백 호출 (→ `vlm_api._run_inference_once`)
  - `pending`/`inflight` 플래그 다음 라운드를 위해 False로 reset
- **큐 크기**: `INFERENCE_QUEUE_SIZE=128`
- **Runtime 상태 추적** (`_camera_runtime`): `jobs_enqueued`, `jobs_completed`, `jobs_dropped`, `last_submit_ts`, `last_finish_ts`, `last_active_ts`, `workers_alive`, `dispatcher_alive`
- **Active burst** (`INFERENCE_ACTIVE_BURST_SEC=3.0`, `INFERENCE_ACTIVE_BURST_FPS=3.0`): `mark_camera_active()` 호출 후 3초 동안 FPS 승격

### PipelineOrchestrator (`pipeline_orchestrator.py`, 644 LoC)

- **`process_frame_sequential()`** — Florence 1회 + 3 시나리오 `CaptionAnalyzer.analyze()` (운영 경로, Caption Sharing)
- **`process_frame()`** — `ThreadPoolExecutor(max_workers=3, inference_timeout=10.0s)` 병렬 경로 (미사용, legacy)
- **Cash Dual Path** (전부 현재 OFF):
  - `CASH_DUAL_PATH_ENABLED=false` — ROI crop + 전체 프레임 2회 추론 통합
  - `CASH_ROI_INFER_ENABLED=false` — ROI 분리 추론
  - `CASH_GLOBAL_ASSIST_THRESHOLD=0.30` — ROI 점수가 이 이상이면 global도 요청
- **EVA Q2E 5-stage** 구조 (classification → decomposition → enrichment → clustering → parallel inference) — 현재 실제로 1 stage로 수행. legacy 스타일 보존.

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

### SQLite WAL — `data/cctv_events.db` (`db_server/main.py:54~167`)

전체 5개 테이블 + 7개 인덱스. `PRAGMA journal_mode=WAL`로 read/write 동시성 확보.

#### cameras 테이블 (2026-04-17 `display_name` 추가)

| 컬럼 | 타입 | 기본값 | 설명 |
|------|-----|------|-----|
| `id` | INTEGER PK AUTOINCREMENT | - | 내부 PK |
| `camera_id` | TEXT UNIQUE NOT NULL | - | 내부 식별자 (파일명·이벤트 ID에 박힘, **읽기 전용**) |
| `rtsp_url` | TEXT NOT NULL | - | RTSP 스트림 URL (credential 포함) |
| `base_fps` | REAL | 1.5 | 추론 샘플링 FPS |
| `rtsp_transport` | TEXT | tcp | `tcp` / `udp` |
| `open_timeout_ms` | INTEGER | 8000 | cv2 connect timeout |
| `read_timeout_ms` | INTEGER | 8000 | cv2 read timeout |
| `event_cooldown_sec` | INTEGER | 20 | 동일 시나리오 이벤트 간 최소 간격 |
| `clip_duration_sec` | INTEGER | 10 | 영구 clip 길이 |
| `validation_clip_sec` | INTEGER | 10 | Gemini 입력 clip 길이 (`clip_duration_sec`과 통일 권장) |
| `evidence_mode` | TEXT | video_only | Gemini 모드 |
| `use_video_validation` | INTEGER | 1 | 비디오 검증 활성화 플래그 |
| `cashier_zone` | TEXT JSON | '[]' | 캐셔 존 정규화 좌표 폴리곤 `[[x,y],...]` |
| `drawer_zone` | TEXT JSON | '[]' | 서랍 존 폴리곤 |
| `display_name` | TEXT | '' | UI 표시 라벨 (자유 편집, 2026-04-17 추가) |
| `created_at` / `updated_at` | TEXT | `datetime('now','localtime')` | |

마이그레이션: `PRAGMA table_info(cameras)` 검사 후 `display_name` 없으면 `ALTER TABLE cameras ADD COLUMN display_name TEXT DEFAULT ''` idempotent 실행.

#### events 테이블

| 컬럼 | 타입 | 설명 |
|------|-----|------|
| `id` | INTEGER PK AUTOINCREMENT | 내부 PK |
| `event_id` | TEXT UNIQUE NOT NULL | 이벤트 고유 ID (`ev_{ts_ms}_{scenario}_{camera_id}`) |
| `camera_id` | TEXT | 카메라 내부 식별자 |
| `event_type` / `scenario` | TEXT | cash / fire / violence |
| `confidence` | REAL | Tier-1 신뢰도 (`CaptionAnalyzer` 산출) |
| `tier` | INTEGER | 1 / 2 |
| `is_detected` | INTEGER | 최종 탐지 여부 (0/1) |
| `gemini_validated` | INTEGER NULLABLE | Gemini 판정 결과 (null=미호출) |
| `gemini_confidence` | REAL NULLABLE | Gemini confidence |
| `gemini_reason` | TEXT | reason_bullets 조인 |
| `caption` | TEXT | Florence 원문 캡션 |
| `matched_keywords` | TEXT JSON | (실측: 대부분 `[]`) |
| `evidence` | TEXT | `"Keywords: ... | Objects: ... | Caption: ..."` |
| `clip_path` | TEXT | 영구 clip 경로 |
| `human_feedback` | TEXT JSON NULLABLE | **Labeling UI 라벨 (canonical GT)** — `{"decision":"accept|decline|unsure","note":"","labeler":"","error_type":"","created_at":"..."}` |
| `event_data` | TEXT JSON | 전체 메타 (Tier-1/Tier-2 raw 원본, router snapshot) |
| `created_at` | TEXT | `datetime('now','localtime')` |

인덱스: `event_id`, `camera_id`, `event_type`, `created_at`

> **참고**: SQLite `events` 테이블은 현재 flush_worker가 실제 쓰기 안 하는 상태. 진실의 원본은 `data/events/YYYYMMDD/*.json` (canonical GT는 JSON 파일의 `human_feedback` 필드).

#### gemini_logs 테이블

| 컬럼 | 타입 | 설명 |
|------|-----|------|
| `id` | INTEGER PK | |
| `event_id` | TEXT UNIQUE NOT NULL | |
| `camera_id`, `event_type` | TEXT | |
| `gemini_state` | TEXT | validated / declined / skipped / error |
| `gemini_validated` | INTEGER | |
| `gemini_confidence` | REAL | |
| `gemini_reason` | TEXT | |
| `validation_type` | TEXT | cash / violence / fire / staff_cash_theft / none |
| `input_mode` | TEXT | video_only / storyboard / image 등 |
| `prompt_version` | TEXT | `evidence-v1.1` |
| `processing_time_ms` | INTEGER | API round-trip |
| `media_ref` | TEXT | `video:<path>` / `image:<idx>` / `packet_keyframes` |
| `log_data` | TEXT JSON | 전체 prompt + response + packet_summary |
| `created_at` | TEXT | |

인덱스: `event_id`, `created_at`

#### episode_reviews 테이블 (현재 미사용, 구조 유지)

| 컬럼 | 설명 |
|------|-----|
| `episode_id`, `event_id`, `camera_id`, `event_type` | 식별자 |
| `final_policy`, `is_valid_event` | 최종 정책 |
| `review_status` | `queued` / `in_review` / `resolved` |
| `reviewer` | 리뷰어 이름 |
| `gemini_validated/confidence/reason` | Tier-2 결과 스냅샷 |
| `tier1_snapshot`, `router_snapshot`, `florence_signals` | JSON snapshot |
| `feedback_suggestion` | 큐레이션 힌트 |

#### worker_leases 테이블

크로스 프로세스 워커 중복 방지:

| 컬럼 | 설명 |
|------|-----|
| `camera_id` UNIQUE | 리스 키 |
| `instance_id` | 워커 인스턴스 식별자 |
| `pid` | 프로세스 ID |
| `rtsp_url` | 리스 대상 URL |
| `acquired_at`, `last_heartbeat` | TTL 관리 |
| `lease_ttl_sec` | 기본 60초 |

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

| 변수 | config.py 기본 | 운영 `.env` | 설명 |
|------|-----|-----|------|
| `FLORENCE_MODEL` | `microsoft/Florence-2-large` | 동일 | HF 캐시 `models/hf/` |
| `FLORENCE_BACKEND` | `pytorch` | `pytorch` | `openvino` backend는 stub — PyTorch로 fallback |
| `FLORENCE_DEVICE` | `cuda` | `cuda` | `cpu` / `auto`도 허용. `cuda` 명시인데 CUDA 없으면 RuntimeError |
| `FLORENCE_INPUT_SIZE` | `448` | **`448`** | `cv2.resize(INTER_AREA)` → HF processor 내부 768 bilinear upscale (이중 resize) |
| `FLORENCE_DTYPE` | `float32` | `float32` | GPU 사용 시 어댑터가 자동 `torch.float16`로 승격 |
| `FLORENCE_MAX_TOKENS` | `512` | **`200`** | 2026-04-17 이후 96 → 200 승격 (truncation 방지) |
| `FLORENCE_NUM_BEAMS` | `3` | **`1`** | Greedy. `do_sample=False` |
| `FLORENCE_CAPTION_DETAIL` | `more` | **`more`** | `<MORE_DETAILED_CAPTION>` task token |
| `FLORENCE_LOG_PERSIST` | `true` | `true` | `data/florence_logs/YYYYMMDD/{cam}.jsonl` raw 캡션 저장 |
| `FLORENCE_LOG_DIR` | `data/florence_logs` | 동일 | |
| `GEMINI_API_KEY` | (필수) | 설정됨 | Google AI Studio key |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | `gemini-3.1-flash-lite-preview` | |
| `GEMINI_TEMPERATURE` | `0.1` | - | |
| `GEMINI_MAX_OUTPUT_TOKENS` | `1500` | - | |
| `GEMINI_TIMEOUT_SEC` | `30` | **`180`** | 6× 타임아웃 확장 |
| `GEMINI_MAX_CONCURRENT` | `1` | `1` | `asyncio.BoundedSemaphore` |
| `LORA_ENABLED` | `false` | `false` | `peft` 어댑터 로드 |
| `LORA_DATA_COLLECTION` | `true` | `true` | 학습 데이터 수동 수집 지속 |
| `LORA_COLLECT_NORMAL_RATIO` | `0.05` | `0.0` | 정상 프레임 샘플링 비율 |
| `LORA_MAX_SAMPLES` | `50000` | - | |

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
| `BASE_FPS` | `1.5` | 겉보기 설정 (실효 ~1 fps, Florence latency bound) |
| `BURST_FPS` | `4.0` | 탐지 후 |
| `BURST_DURATION_SEC` | `3.0` | burst 지속 시간 |
| `SINGLE_CAMERA_MODE` | `false` | 멀티카메라 vs 단일카메라 강제 |
| `GLOBAL_INFERENCE_LOCK` | `true` | Florence 추론 직렬화 (모든 카메라 공유 lock) |
| `INFERENCE_WORKERS` | `1` | GPU compute-bound이라 2로 늘려도 개선 없음 |
| `INFERENCE_QUEUE_SIZE` | `128` | job queue 크기, 초과 시 drop |
| `INFERENCE_ACTIVE_BURST_SEC` | `3.0` | mark_camera_active 후 FPS 승격 지속 시간 |
| `INFERENCE_ACTIVE_BURST_FPS` | `3.0` | burst FPS (스케줄러 단) |
| `RTSP_TRANSPORT` | `tcp` | `tcp`/`udp`, 기본 tcp로 안정성 우선 |
| `RTSP_OPEN_TIMEOUT_MS` | `8000` | cv2 connect timeout |
| `RTSP_READ_TIMEOUT_MS` | `8000` | cv2 read timeout |
| `RTSP_HWACCEL` | `cuda` | NVDEC. `none`으로 CPU 강제 |
| `RTSP_HWACCEL_DEVICE` | `0` | GPU 인덱스 |
| `RTSP_HWACCEL_DECODER` | (자동) | `h264_cuvid` 등 강제 지정 가능 |
| `RTSP_HWACCEL_ALLOW_FALLBACK` | `true` | HW 디코더 실패 시 CPU fallback |
| `STALE_THRESHOLD_SEC` | `2.5` | 프레임 stale 판정 임계 |
| `CLIP_BUFFER_SECONDS` | `30` | ring buffer 최대 유지 시간 (deque maxlen 계산 기준) |

### 에피소드 / 라우터

| 변수 | 기본 | 설명 |
|------|-----|------|
| `EPISODE_MIN_DETECTIONS` | `2` | episode 시작 최소 연속 탐지 수 |
| `EPISODE_STABILITY_THRESHOLD` | `0.65` | episode 안정성 minimum |
| `EPISODE_COOLDOWN_SEC` | `60` | 동일 type 재시작 쿨다운 |
| `EPISODE_MAX_PER_TYPE` | `3` | 동시 활성 episode 상한 |
| `GEMINI_TARGET_RATIO` | `0.30` | Gemini 호출 목표 비율 |
| `GEMINI_RATIO_PENALTY` | `0.25` | 비율 초과 시 Q-score penalty |
| `VIDEO_CLIP_SECONDS` | `10` | Gemini 비디오 clip 길이 |
| `CRITIC_ENABLED` | `false` | LightGBM critic 활성 |
| `CRITIC_MIN_SAMPLES` | `30` | 최소 학습 샘플 |

### 저장/플러시

| 변수 | 기본 | 설명 |
|------|-----|------|
| `MODEL_SERVER_DATA_DIR` | `data` | 런타임 데이터 루트 |
| `MODEL_SERVER_MODELS_DIR` | `models` | HF 캐시 |
| `MODEL_SERVER_LOG_DIR` | `data/logs` | |
| `DB_PATH` | `data/cctv_events.db` | SQLite WAL |
| `DB_SERVER_URL` | `http://localhost:8001` | flush 대상 |
| `FLUSH_ENDPOINT` | `/api/flush` | |
| `FLUSH_INTERVAL_SEC` | `3600` | 1시간 간격 배치 |
| `FLUSH_MAX_RETRIES` | `3` | |
| `LOCAL_RETENTION_DAYS` | `5` | 로컬 파일 보존 기간 |
| `FFMPEG_PATH` | `ffmpeg` | |
| `EVIDENCE_MODE` | `video_only` | Gemini 기본 모드 |
| `CLIP_SAVE_MAX_CONCURRENT` | `1` | 동시 clip 저장 한도 |
| `POSTPROCESS_WORKERS` | `1` | event_postprocessor 큐 워커 |
| `POSTPROCESS_QUEUE_SIZE` | `128` | |
| `USE_S3` | `false` | S3 업로드 |
| `AWS_REGION` | `ap-northeast-2` | |
| `ROUTER_STEPS_PATH` | `data/router_steps.jsonl` | append-only router log |

### 부팅/복구

| 변수 | 기본 | 설명 |
|------|-----|------|
| `AUTO_RESTORE_CAMERAS_ON_BOOT` | `true` | 부팅 시 마지막 세션 카메라 자동 복원 |
| `AUTO_RESTORE_DELAY_SEC` | `4` | 복원 시작 전 대기 |
| `AUTO_RESTORE_DB_RETRIES` | `20` | DB 연결 재시도 횟수 |
| `AUTO_RESTORE_DB_RETRY_SEC` | `3` | |
| `AUTO_RESTORE_FRAME_WAIT_SEC` | `20` | 첫 프레임 기다림 |
| `AUTO_RESTORE_BETWEEN_CAM_SEC` | `1.5` | 카메라 간격 |

### 기타

| 변수 | 기본 | 설명 |
|------|-----|------|
| `TZ` | `Asia/Seoul` | |
| `LOG_LEVEL` | `INFO` | |
| `LOG_FORMAT` | 표준 | `%(asctime)s [%(name)s] %(levelname)s: %(message)s` |

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

## 코드 규모 (2026-04-21 실측)

### LoC 분해 (19,334 lines 총)

| 영역 | LoC | 주요 파일 |
|------|-----|----------|
| model_server (core) | 11,951 | `evidence_router.py` (1,799), `vlm_api.py` (1,784), `gemini_validator.py` (1,316), `main.py` (853), `scenarios/base_scenario.py` (775), `stream_manager.py` (683), `pipeline_orchestrator.py` (644), `local_storage.py` (602) |
| model_server (adapters) | 798 | `florence_adapter.py` (533), `base_adapter.py` (265) |
| model_server (scenarios) | 1,037 | `base_scenario.py` (775), `prompts.py` (235), `__init__.py` (27) |
| model_server (lora) | 1,259 | `data_collector.py` (631), `train_lora.py` (402), `dataset.py` (226) |
| model_server (evolution) | 528 | `rule_updater.py` (337), `critic_trainer.py` (191) — **Dead code, 제거 대기** |
| model_server (agents) | 295 | `dynamic_agent.py` — **Dead code, 제거 대기** |
| model_server (dead) | 302 | `base_detector.py` — **Dead code, 제거 대기** |
| db_server | 777 | `main.py` (675), `models.py` (102 Django legacy) |
| frontend_server | 590 | `main.py` 단일 FastAPI 파일 |
| frontend templates | 3,422 | `adhoc_rtsp.html` (1,081), `labeling.html` (844), `gemini_logs.html` (779), `florence_logs.html` (483), `base_public.html` (235) |
| deploy | 838 | `vlm-safe-recover.sh` (293), `setup_aws_g4dn.sh` (239), `track_disconnect_2h.sh` (125) + 4 systemd unit |

### Dead code 후보 (2026-04-17 Shadow 제거 후 경로 끊김)

| 파일 | LoC | 상태 | 비고 |
|------|-----|------|-----|
| `evolution/critic_trainer.py` | 191 | Dead | Shadow에서만 호출되던 `train()` 트리거 사라짐 |
| `evolution/rule_updater.py` | 337 | Dead | `apply_feedback_to_rules()` 호출 경로 없음 |
| `agents/dynamic_agent.py` | 295 | Dead | 초기화만 존재 |
| `base_detector.py` | 302 | Dead | 정의만 존재 (YOLO `ultralytics` import는 남아있으나 호출 0) |
| `episode_manager.py` | 421 | Dead | `main.py`에서 import 없음, throwaway Episode만 생성 |
| `db_server/models.py` | 102 | Dead | Django ORM 스키마, SQLite WAL로 대체됨 |
| `scenarios/prompts.py` 문자열 상수 | ~150 | Dead | `get_scenario_prompt()`는 호출되나 반환 문자열이 Florence에서 무시됨 |
| `frontend_server/views.py` | 1 | Empty | |
| `fix_html.py` | - | Dead | 일회성 마이그레이션 스크립트 |

**제거 시 예상 LoC 감소**: ~1,800 lines (~9%).

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

## 이벤트 생명주기 (End-to-End Trace)

```
[1] RTSP 프레임 캡처 (StreamManager._reader_loop)
     │  cv2.VideoCapture(rtsp_url) + NVDEC (h264_cuvid)
     │  frame (BGR numpy) + mono_ts = time.monotonic()
     │  ↓ deque.append({"frame": frame, "mono_ts": mono_ts})
     ▼
[2] Ring buffer (deque maxlen = 360 frames ≈ 30초 × 12fps)

[3] InferenceScheduler dispatcher (20ms 폴링)
     │  state.running=True ∧ no pending/inflight
     │  interval 경과 여부 확인 (1.0 / target_fps)
     │  stream_manager.get_frame(camera_id) → 최신 프레임
     │  ↓ InferenceJob 생성 + queue.put_nowait()
     ▼
[4] Worker (1개) ← queue.get(timeout=0.2)
     │  run_id stale 가드 (다른 세션 잡 drop)
     │  ↓ process_fn(camera_id, frame, state, started_at)
     │     = vlm_api._run_inference_once
     ▼
[5] PipelineOrchestrator.process_frame_sequential
     │  Florence-2 infer 1회:
     │    preprocess_image (448×448) → PIL
     │    AutoProcessor (내부 768 upscale)
     │    autocast(cuda, fp16) + generate(max_tokens=200, beams=1)
     │    post_process_generation → caption (free-form text)
     │  ↓
     │  CaptionAnalyzer.analyze(caption, CASH) × 3 scenarios
     │  word-boundary regex → strong/moderate/context/negative matches
     │  neutralizing phrases 무효화 → confidence (0-1 clamp)
     │  ↓ (optional) florence_logs/{camera}.jsonl append (JSONL)
     ▼
[6] EvidenceRouter.select_action(episode, state)
     │  1. hard-risk? (fire/violence & conf<0.95) → GEMINI_VIDEO
     │  2. below threshold? (cash<0.55) → GEMINI_VIDEO (force_tier2)
     │  3. high-conf+stability? (conf≥0.85, stab≥0.90) → SKIP
     │  4. max-Q + margin gate
     │  ↓ router_steps.jsonl append-only log
     ▼
[7] Event 생성 (Detection → Event)
     │  event_id = f"ev_{int(time.time()*1000)}_{scenario}_{camera_id}"
     │  detection metadata + Tier-1 결과 + Router 판정
     │  ↓ event_postprocessor queue enqueue
     ▼
[8] EventPostProcessor (CPU, 별도 thread)
     │
     ├─ val_clip 생성 (Gemini 입력 원본):
     │     ring buffer에서 anchor_mono_ts 기준 10초 frames (val_entries)
     │     cv2.VideoWriter → 임시 AVI (MJPG)
     │     FFmpeg: libx264 CRF23 preset fast yuv420p +faststart
     │     cashier/drawer zone polygon overlay burn-in (cash만)
     │     → data/clips/YYYYMMDD/val_ev_*.mp4
     │
     ├─ Gemini API 호출 (video_only 모드):
     │     validate_event_evidence(packet, mode='video_only',
     │                             video_path='val_ev_*.mp4')
     │     prompt = DEFAULT_UNIFIED_PROMPT.replace('{event_type}', ...)
     │     generate_content(model, contents=[prompt + video_bytes],
     │                      config=GenerateContentConfig(temp=0.1,
     │                        top_k=1, max_output_tokens=1500,
     │                        response_mime_type="application/json"))
     │     → JSON {event_policy, is_valid_event, decision,
     │             severity_label, confidence, policy_scores,
     │             reason_bullets, event_type_detected}
     │     processing_time_ms 기록
     │     _parse_new_response_format → (is_valid, conf, reason, corrected_type)
     │     H2H correction (violence→cash 차단)
     │
     ├─ 영구 clip 생성 (val_entries 재사용, race 제거):
     │     ev_{id}.mp4 (plain, clip_url)
     │     ev_{id}_roi.mp4 (overlay, cash+zone 있을 때만)
     │     thumbnails/{id}.jpg
     │
     ├─ LocalStorage:
     │     events/YYYYMMDD/{event_id}.json (canonical)
     │     clips/YYYYMMDD/{ev_id}.mp4 + _roi.mp4
     │     thumbnails/YYYYMMDD/{ev_id}.jpg
     │
     └─ LoRA DataCollector (passive):
          LORA_DATA_COLLECTION=true면 detection → images/ + annotations.jsonl
     ▼
[9] FlushWorker (1시간 간격 또는 수동 트리거)
     │  data/events/YYYYMMDD/*.json 스캔
     │  multipart POST /api/flush (DB Server :8001)
     │  DB Server: SQLite INSERT OR REPLACE INTO events + gemini_logs
     ▼
[10] Frontend UI
     GET /api/proxy/events → 목록
     GET /monitor/labeling → GT 라벨링
     POST /api/vlm/feedback → event.human_feedback 업데이트
     (Labeling UI에서 저장한 GT는 event JSON의 human_feedback 필드에 기록)
```

### 주요 타임라인 (cash 이벤트 기준 실측)

| 단계 | 시간 | 누적 |
|-----|-----|-----|
| 프레임 캡처 → ring buffer | <1ms | <1ms |
| 스케줄러 dispatch 대기 | 0-20ms | ~10ms |
| Florence-2 추론 (p50) | 1002ms | ~1012ms |
| CaptionAnalyzer × 3 | <1ms | 1012ms |
| Router + Event 생성 | <10ms | 1022ms |
| val_clip 생성 (10초 clip + FFmpeg) | ~500-1500ms | ~2.5s |
| Gemini API 호출 | 4000-6000ms | ~8s |
| 영구 clip + thumbnail | ~300-800ms | ~9s |
| Event → JSON 저장 | <50ms | ~9s |
| FlushWorker → DB | 1시간 지연 | +3600s |

**첫 이벤트 발생 → UI 가시화 latency**: 약 **9초** (Gemini 모드 video_only 기준).

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

### Gemini 거절 사유 분포 (cash 186건, 2026-04-17 기준)

| 사유 | 건수 | 비율 | Motion gating 차단 가능? |
|-----|-----|------|------------------------|
| `no cash` | 127 | **68.3%** | × |
| `no physical` | 62 | 33.3% | × |
| `no customer` (present) | 51 | **27.4%** | ✓ (빈 데스크 즉시 컷) |
| `no exchange` | 50 | 26.9% | × |
| `phone` | 44 | 23.7% | × |
| `receipt` | 36 | 19.4% | × |
| `smartphone` | 34 | 18.3% | × |
| `no banknote` | 28 | 15.1% | × |
| `no hand` | 1 | 0.5% | - |
| `no drawer` | 0 | 0.0% | - |

Motion gating으로 하한 27.4% (`no customer`) 즉시 차단 가능. 상한은 정적 장면 중복 (상위 3 캡션이 전체의 23.4%) 포함 시 30-40%.

### 튜닝 우선순위

| 우선 | 작업 | Recall 리스크 | 비용/지연 |
|-----|------|------------|----------|
| ★★★★★ | GT 라벨 수집 (Labeling UI 운영) | 0 | 운영 공수 (1건 10-15초, 전수 라벨링 ~30분) |
| ★★★★★ | Gemini hard-gate 확장 (no physical/no exchange 조기 종료) | 0 | 5분, post-Gemini |
| ★★★★★ | Motion gating (빈 데스크 차단, `no customer` 27.4% + 중복 23% 차단) | 낮음 | Gemini 호출 -27~-40%, ~150 LoC 1일 |
| ★★★★ | Dead code 2차 제거 (`critic_trainer`, `rule_updater`, `dynamic_agent`, `base_detector`, `episode_manager`) | 0 | 유지보수성, ~1,800 LoC 감소 |
| ★★★★ | 모니터링 자동화 (캡션 어휘 drift, conf 버킷 추적, Gemini reject 사유 분포) | 0 | 측정 도구 |
| ★★★ | RTSP credential 로그 마스킹 (journalctl에 credential 평문 노출) | 0 | 보안 |
| ★★★ | Florence input size 448→768 (이중 resize 완전 제거, cash 해상도 5.7×) | **중간** — "bank" 오인식이 현재 TP trigger이므로 영향 모니터링 필요 | latency +50-100%, GT 확보 후 A/B |
| ★★ | `CASH_DUAL_PATH_ENABLED` ON | 낮음 | latency 2× (Motion gating과 세트) |
| ★★ | Repetition penalty 추가 (`repetition_penalty=1.3, no_repeat_ngram_size=3`) | 낮음 | 30분 구현, 중복 캡션 다양화 |
| ★ | SQLite events 테이블 flush 경로 복구 or deprecation | 0 | 현재 JSON canonical이라 긴급도 낮음 |
| ✗ | `INFERENCE_WORKERS=2` | - | GPU compute-bound 확정 (p50 1002ms), throughput 개선 없음 |
| ✗ | LoRA 학습 | - | GT 1/186 → 학습 데이터 부족 |
| ✗ | Phrase grounding 단독 사용 | - | 실험 완료: hallucination (best-match localizer). GT calibration 선행 필수 |
| ✗ | Qwen2.5-VL-3B 로컬 Tier-2 이식 | - | Gemini가 잘 작동, 로컬 Qwen은 VRAM 2.8GB + 5-15초 latency + 유지보수 부담 |

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
