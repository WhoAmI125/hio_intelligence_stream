# HiO Intelligence Stream - 자율 진화형 CCTV 이상 탐지 시스템

> Florence-2 (Tier-1, On-Device GPU) + Gemini (Tier-2, Cloud API) + Shadow/Evolution + LoRA 데이터 수집
> **코드 규모:** Python ~15,320 LOC | **3-서버 마이크로서비스** | **실시간 RTSP 분석**

---

## 최근 업데이트 (2026-03-06)

- 멀티 카메라 그리드 + 팝업 기반 설정 UI 반영 (`/monitor/adhoc`)
- ROI 편집/미리보기/ROI Only 팝업 반영
- 검증 로그 화면 분리 및 카메라 필터 지원 (`/monitor/validation-logs`)
- `PipelineOrchestrator` + `EvidenceRouter`를 실제 추론 루프에 연결
- Tier-2 결과가 `REJECT`면 이벤트 저장 스킵 처리
- 클립 저장: FFmpeg(H.264) 우선 + OpenCV fallback 경로
- 썸네일 저장, `last_validation`/`last_clip_path` 상태 반영
- Tier-2 검증 메타를 DB `validation_logs` 테이블에 적재
- 동일 RTSP 중복 실행 방지(충돌/디코더 오류 완화)
- LoRA 수집 정책 강화: cash + Tier-2 승인 + clip 기반 샘플만 수집

---

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [전체 파일 구조](#전체-파일-구조)
3. [현재 아키텍처](#현재-아키텍처)
4. [아키텍처 다이어그램 (ASCII)](#아키텍처-다이어그램-ascii)
5. [모델 아키텍처 상세](#모델-아키텍처-상세)
6. [추론 파이프라인 상세 흐름](#추론-파이프라인-상세-흐름)
7. [GPU/CPU 리소스 사용 분석](#gpucpu-리소스-사용-분석)
8. [핵심 컴포넌트 상세](#핵심-컴포넌트-상세)
9. [자율 진화 루프 (Shadow/Evolution/LoRA)](#자율-진화-루프-shadowevolutionlora)
10. [탐지 파이프라인](#탐지-파이프라인)
11. [UI 구성](#ui-구성)
12. [UI/UX 상세 설계](#uiux-상세-설계)
13. [데이터베이스 스키마](#데이터베이스-스키마)
14. [데이터 저장 구조](#데이터-저장-구조)
15. [LoRA 데이터 수집 정책](#lora-데이터-수집-정책)
16. [API 요약](#api-요약)
17. [환경 변수 전체](#환경-변수-전체)
18. [배포 구조](#배포-구조)
19. [로컬 실행 방법](#로컬-실행-방법)
20. [트러블슈팅](#트러블슈팅)
21. [Repo Hygiene](#repo-hygiene)
22. [향후 작업](#향후-작업)

---

## 프로젝트 개요

이 프로젝트는 RTSP CCTV 스트림을 실시간으로 분석하여 아래 3개 시나리오를 탐지합니다.

- `cash`: 현금 거래/금전 전달
- `fire`: 화재/연기
- `violence`: 폭력/충돌

핵심 철학은 다음과 같습니다.

- **Tier-1 고속 탐지 + Tier-2 정밀 검증**: Florence-2로 빠른 캡션 생성 → 키워드 분석 → 불확실한 경우만 Gemini Cloud API로 검증
- **Caption Sharing 최적화**: 프레임당 Florence-2 추론 1회로 3개 시나리오 동시 분석 (GPU 효율 극대화)
- **운영 중 피드백/로그 축적**: 모든 추론 결과, Gemini 검증, 인간 피드백을 구조화된 JSONL로 기록
- **Shadow/Critic/RuleUpdater를 통한 점진적 자율 개선**: 백그라운드에서 탐지 품질을 자동 평가하고 프롬프트를 진화시킴

---

## 전체 파일 구조

```
hio_intelligence_stream/
├── model_server/                          # Tier-1/Tier-2 추론 서버 (:8000)
│   ├── main.py                            #   FastAPI 앱, 전체 컴포넌트 초기화/종료 라이프사이클
│   ├── vlm_api.py                         #   VLM API 라우터 (/api/vlm/*), 핵심 추론 루프
│   ├── config.py                          #   환경변수 기반 중앙 설정 (Florence/Gemini/임계값/스트림)
│   ├── stream_manager.py                  #   멀티카메라 RTSP 리더 + 링버퍼 + burst FPS
│   ├── pipeline_orchestrator.py           #   멀티시나리오 병렬/순차 추론 오케스트레이터
│   ├── evidence_router.py                 #   Tier-2 에스컬레이션 라우터 (Q-value + 정책 기반)
│   ├── gemini_validator.py                #   Tier-2 Gemini Vision API 검증기
│   ├── inference_scheduler.py             #   중앙 추론 스케줄러 (디스패처+워커 풀)
│   ├── episode_manager.py                 #   시간축 에피소드 상태머신 (IDLE→ACTIVE→VALIDATING→DONE)
│   ├── event_postprocessor.py             #   비동기 후처리 워커 풀 (클립저장/Gemini호출)
│   ├── local_storage.py                   #   로컬 파일 저장 (이벤트JSON/클립MP4/썸네일JPG) + S3 옵션
│   ├── flush_worker.py                    #   주기적 Model→DB 배치 플러시 (HTTP POST)
│   ├── base_detector.py                   #   YOLO 기반 탐지기 ABC + 키포인트/존 유틸
│   ├── logger.py                          #   구조화된 JSONL 로거 (에이전트/오케스트레이터/에피소드/라우터)
│   ├── adapters/
│   │   ├── base_adapter.py                #     VLM 어댑터 ABC (전처리/crop/추론래퍼)
│   │   └── florence_adapter.py            #     Florence-2 어댑터 (PyTorch/OpenVINO 백엔드, LoRA 로드)
│   ├── scenarios/
│   │   ├── prompts.py                     #     시나리오별 프롬프트 템플릿 + 존 컨텍스트
│   │   ├── base_scenario.py               #     CaptionAnalyzer (키워드 매칭 엔진) + ScenarioResult
│   │   ├── cash_scenario.py               #     현금 탐지 시나리오
│   │   ├── violence_scenario.py           #     폭력 탐지 시나리오
│   │   └── fire_scenario.py               #     화재 탐지 시나리오
│   ├── agents/
│   │   ├── dynamic_agent.py               #     2-Tier 탐지 에이전트 (Florence→UncertaintyGate→Gemini)
│   │   ├── shadow_agent.py                #     비동기 백그라운드 재평가 에이전트
│   │   └── prompts/                       #     시나리오별 운영/쉐도우 프롬프트 (.md)
│   │       ├── cash.md / cash_shadow.md
│   │       ├── fire.md / fire_shadow.md
│   │       └── violence.md / violence_shadow.md
│   ├── evolution/
│   │   ├── critic_trainer.py              #     LightGBM 이진분류 크리틱 (Tier-1 정확도 예측)
│   │   └── rule_updater.py                #     프롬프트 자동 진화 (Gemini Pro 메타리파인 + 롤백)
│   └── lora/
│       ├── data_collector.py              #     LoRA 학습 데이터 자동 수집기
│       ├── dataset.py                     #     Florence-2 LoRA PyTorch Dataset + Collate
│       └── train_lora.py                  #     LoRA 파인튜닝 학습 스크립트 (rank-8 어댑터)
│
├── db_server/                             # 이벤트/통계/피드백 DB 서버 (:8001)
│   ├── main.py                            #   FastAPI 앱, SQLite WAL, 5개 테이블 관리
│   ├── models.py                          #   레거시 Django ORM 모델 (참조용)
│   └── api/
│       └── flush.py                       #   초기 flush 라우터 (main.py 배치 flush로 대체됨)
│
├── frontend_server/                       # 모니터링 UI + 리버스 프록시 (:8002)
│   ├── main.py                            #   FastAPI 앱, Jinja2 렌더, VLM/DB 프록시, 시스템 메트릭
│   ├── views.py                           #   (deprecated)
│   └── templates/vlm_pipeline/
│       ├── base_public.html               #     Jinja2 베이스 레이아웃 (사이드바 내비게이션)
│       ├── adhoc_rtsp.html                #     실시간 CCTV 모니터링 대시보드
│       ├── monitor_shadow.html            #     Shadow 에이전트 리뷰 UI
│       ├── gemini_logs.html               #     Gemini Tier-2 검증 로그 뷰어
│       └── florence_logs.html             #     Florence Tier-1 추론 로그 뷰어 + LoRA 피드백
│
├── deploy/                                # AWS 배포 자동화
│   ├── setup_aws_g4dn.sh                  #   원스텝 배포 스크립트 (8단계)
│   ├── nginx.conf                         #   nginx 리버스 프록시 설정
│   ├── vlm-model.service                  #   model_server systemd 유닛
│   ├── vlm-db.service                     #   db_server systemd 유닛
│   ├── vlm-frontend.service               #   frontend_server systemd 유닛 (worker 2)
│   ├── vlm-boot-recover.service           #   부팅 시 안전 복구 유닛
│   ├── vlm-safe-recover.sh                #   복구 오케스트레이션 스크립트
│   └── track_disconnect_2h.sh             #   2시간 인시던트 모니터링 도구
│
├── data/                                  # 런타임 데이터 (gitignore)
│   ├── events/YYYYMMDD/*.json             #   이벤트 JSON
│   ├── clips/YYYYMMDD/*.mp4               #   비디오 클립 (H.264)
│   ├── thumbnails/YYYYMMDD/*.jpg          #   이벤트 썸네일
│   ├── lora_training/                     #   LoRA 학습 데이터 (이미지 + annotations.jsonl)
│   ├── shadow_feedback/                   #   Shadow 에이전트 피드백 JSONL
│   ├── critic_models/                     #   LightGBM 크리틱 모델 (critic_v*.txt)
│   ├── rule_versions/                     #   프롬프트 버전 히스토리
│   ├── cctv_events.db                     #   SQLite 데이터베이스
│   └── media_archive/                     #   DB 서버 미디어 아카이브
│
├── models/                                # HuggingFace 모델 캐시 (gitignore)
│   └── hf/                                #   Florence-2 모델 가중치 (~1.9GB)
│
├── .env / .env.example / .env.aws         # 환경 설정
├── requirements.txt                       # Python 의존성 (CPU)
├── requirements_gpu.txt                   # CUDA 12.1 PyTorch (GPU)
├── start_local.py                         # 로컬 개발 3-서버 런처
├── VLM_INFERENCE_REFACTOR_GUIDE.md        # 추론 파이프라인 리팩토링 가이드
└── AWS_G4DN_DEPLOY_GUIDE.md               # AWS g4dn 배포 가이드
```

---

## 현재 아키텍처

### 3-서버 구조

| 서버 | 포트 | 프로세스 | 역할 |
|---|---|---|---|
| `model_server` | `:8000` | uvicorn worker 1 | 추론, 스트림 제어, 이벤트 생성, GPU 점유 |
| `db_server` | `:8001` | uvicorn worker 1 | SQLite WAL, 이벤트/통계/피드백/검증로그 저장 |
| `frontend_server` | `:8002` | uvicorn worker 2 | Jinja2 UI + reverse proxy (VLM/DB) |

### 런타임 핵심 컴포넌트

| 컴포넌트 | 역할 | GPU/CPU |
|---|---|---|
| `StreamManager` | RTSP 입력 + ring buffer + burst FPS | CPU (OpenCV + numpy) |
| `InferenceScheduler` | 중앙 디스패처 + 워커 풀 (카메라별 최신 프레임만) | CPU (스케줄링) |
| `FlorenceAdapter` | Tier-1 캡션 생성 (Florence-2-large, float16) | **GPU** (CUDA) |
| `PipelineOrchestrator` | Caption Sharing + 시나리오 병렬 분석 | GPU 1회 + CPU 키워드 |
| `CaptionAnalyzer` | 캡션 키워드 매칭 (word-boundary regex) | CPU only |
| `EvidenceRouter` | Tier-2 에스컬레이션 판단 (Q-value + 정책) | CPU (numpy) |
| `GeminiValidator` | Tier-2 검증 (Cloud API) | CPU (이미지 인코딩만) |
| `EpisodeManager` | 시간축 상태머신 (안정성/신뢰도 추적) | CPU only |
| `EventPostProcessor` | 비동기 후처리 (클립/썸네일/Gemini) | CPU (FFmpeg) |
| `LocalStorage` | 이벤트/클립/썸네일 파일 저장 | CPU (I/O) |
| `FlushWorker` | Model→DB 배치 동기화 | CPU (네트워크) |
| `ShadowAgent` | 백그라운드 재평가 | CPU + Cloud API |
| `CriticTrainer` | LightGBM 크리틱 학습 | CPU only |
| `RuleUpdater` | 프롬프트 자동 진화 | CPU + Cloud API |
| `DataCollector` | LoRA 학습 데이터 수집 | CPU (JPEG 인코딩) |

---

### 아키텍처 다이어그램 (ASCII)

```text
                         ┌─────────────────────────────────┐
                         │    nginx (:80/443)               │
                         │    dev-cctv.hio.ai.kr            │
                         └──────────────┬──────────────────┘
                                        │
                         ┌──────────────▼──────────────────┐
                         │   Frontend Server (:8002)        │
                         │   Jinja2 Templates + Proxy       │
                         │   /monitor/adhoc                 │
                         │   /monitor/florence-logs          │
                         │   /monitor/gemini-logs            │
                         │   /monitor/shadow                 │
                         │   /dashboard (시스템 메트릭)       │
                         └─────┬───────────────┬────────────┘
                               │               │
                  /api/vlm/*   │               │  /api/events, /api/flush
                               ▼               ▼
┌──────────────────────────────────────┐  ┌──────────────────────┐
│       Model Server (:8000)           │  │ DB Server (:8001)    │
│                                      │  │ SQLite WAL           │
│ ┌──────────────┐  ┌───────────────┐  │  │                      │
│ │StreamManager │  │InferenceScheduler│ │ │ events               │
│ │ RTSP Reader  │  │ Dispatcher    │  │  │ episode_reviews      │
│ │ Ring Buffer  │  │ Worker Pool(1)│  │  │ gemini_logs          │
│ │ Burst FPS    │  └───────┬───────┘  │  │ cameras              │
│ └──────┬───────┘          │          │  │ worker_leases        │
│        │            ┌─────▼────────┐ │  └──────────────────────┘
│        │            │Florence-2    │ │
│        │            │(Tier-1 GPU)  │ │
│        │            │float16 CUDA  │ │
│        │            └─────┬────────┘ │
│        │                  │ caption   │
│        │     ┌────────────▼────────┐ │
│        │     │PipelineOrchestrator │ │
│        │     │ CaptionAnalyzer ×3  │ │
│        │     │ (cash/fire/violence)│ │
│        │     └────────┬───────────┘  │
│        │              │ detections   │
│        │     ┌────────▼───────────┐  │
│        │     │ EvidenceRouter     │  │
│        │     │ Q-value scoring    │  │
│        │     │ Skip gate          │  │
│        │     └────┬──────┬────────┘  │
│        │     skip │      │ escalate  │
│        │          │ ┌────▼────────┐  │
│        │          │ │GeminiValidator│ │
│        │          │ │(Tier-2 Cloud)│  │
│        │          │ │gemini-2.5-   │  │
│        │          │ │flash-lite    │  │
│        │          │ └────┬────────┘  │
│        │          │      │           │
│        │  ┌───────▼──────▼────────┐  │
│        └──│ LocalStorage          │  │
│           │ events/clips/thumb    │  │
│           │ FFmpeg H.264          │  │
│           └───────┬───────────────┘  │
│                   │                  │
│           ┌───────▼───────────────┐  │
│           │ FlushWorker → DB ──────────→ DB Server
│           └───────────────────────┘  │
│                                      │
│  ┌─────── 자율 진화 루프 ──────────┐  │
│  │ ShadowAgent ←→ CriticTrainer   │  │
│  │                ↕               │  │
│  │            RuleUpdater         │  │
│  │            (Gemini Pro 메타)    │  │
│  └────────────────────────────────┘  │
│                                      │
│  ┌────────────────────────────────┐  │
│  │ LoRA DataCollector             │  │
│  │ (cash + tier2 validated only)  │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

---

## 모델 아키텍처 상세

### Tier-1: Florence-2 (On-Device GPU 추론)

| 항목 | 기본값 | 운영값 (.env) | 설명 |
|---|---|---|---|
| **모델** | `microsoft/Florence-2-large` | 동일 | ~770M 파라미터 Vision-Language 인코더-디코더 |
| **로드 방식** | HuggingFace `AutoModelForCausalLM` | `trust_remote_code=True` | 자동 모델 다운로드 + 캐시 |
| **백엔드** | `pytorch` | `pytorch` | OpenVINO 인프라 존재하나 PyTorch 사용 |
| **정밀도** | GPU: `float16` / CPU: `float32` | `float16` (CUDA) | `torch.autocast(cuda, float16)` 적용 |
| **입력 크기** | 448×448px | **320×320px** | 속도 최적화를 위해 축소 |
| **최대 토큰** | 512 | **96** | 생성 길이 제한으로 추론 속도 향상 |
| **빔 수** | 3 | **1** | Greedy decoding으로 전환 (속도 우선) |
| **캡션 상세도** | `more` (`<MORE_DETAILED_CAPTION>`) | **`detailed`** (`<DETAILED_CAPTION>`) | 중간 상세도 |
| **LoRA** | 비활성 | `LORA_ENABLED=false` | `peft.PeftModel` 런타임 로드 지원 |
| **양자화** | 없음 | float16만 | INT8/INT4 미적용 |

#### Florence-2 추론 흐름 (Hot Path)

```
BGR numpy (카메라 프레임)
  ↓ cv2.resize(320×320) + BGR→RGB
  ↓ PIL Image 변환
  ↓ AutoProcessor (토크나이즈: input_ids, pixel_values, attention_mask)
  ↓ .to(device) → GPU 텐서 이동
  ↓ torch.inference_mode() + torch.autocast(cuda, float16)
  ↓ model.generate(max_new_tokens=96, num_beams=1, do_sample=False)
  ↓ processor.batch_decode() → 텍스트
  ↓ processor.post_process_generation() → 구조화된 딕셔너리
  ↓ CaptionAnalyzer.analyze() → 키워드 매칭 (CPU, 마이크로초)
```

#### Florence-2 지원 태스크

| 태스크 | 토큰 | 용도 |
|---|---|---|
| `<CAPTION>` | 기본 캡션 | 빠른 장면 요약 |
| `<DETAILED_CAPTION>` | 상세 캡션 | **운영 기본값** |
| `<MORE_DETAILED_CAPTION>` | 최상세 캡션 | 코드 기본값 |
| `<OD>` | 객체 탐지 | bbox + label 반환 |
| `<CAPTION_TO_PHRASE_GROUNDING>` | 문구 그라운딩 | 텍스트→bbox 매핑 |
| `<DENSE_REGION_CAPTION>` | 밀집 영역 캡션 | 영역별 설명 |
| `<OCR>` / `<OCR_WITH_REGION>` | 문자 인식 | 텍스트 추출 |

### Tier-2: Gemini Vision (Cloud API 검증)

| 항목 | 값 | 설명 |
|---|---|---|
| **모델** | `gemini-2.5-flash-lite` | 경량 멀티모달 LLM |
| **Temperature** | 0.1 | 낮은 Temperature로 결정적 출력 |
| **최대 출력 토큰** | 1500 | 검증 결과 + 근거 |
| **타임아웃** | 90초 | 네트워크 지연 허용 |
| **동시 호출** | 1 (`BoundedSemaphore`) | 직렬화된 API 호출 |
| **프롬프트** | ~360줄 통합 프롬프트 | hard-gate + soft-score 하이브리드 |
| **실패 정책** | Fail-Open | API 오류 시 이벤트 승인 (가용성 우선) |

#### Gemini 검증 모드

| 모드 | 동작 | 적용 시나리오 |
|---|---|---|
| `hybrid` (기본) | cash: 스토리보드 우선 → video / 기타: video 우선 → 스토리보드 | 시나리오별 최적화 |
| `video_first` | 비디오 클립 우선 전송 | 동작 맥락 중시 |
| `video_only` | **운영 기본값** - 비디오만 전송 | 효율성 우선 |
| `images_first` | 키프레임 이미지 우선 | 빠른 검증 |
| `storyboard` | 최대 12장 키프레임 | 상세 분석 |

#### Gemini 통합 프롬프트 구조

```
[시나리오별 이벤트 정의]
  ├── hard-gate 규칙 (즉시 판정)
  ├── soft-score 규칙 (가중 점수)
  └── 정책 우선순위
[Tier-1 업스트림 컨텍스트] (soft hints)
  ├── Florence 신뢰도, 안정성
  ├── 매칭된 키워드, 객체 힌트
  └── Router 액션 이유
[응답 포맷] (JSON)
  ├── event_policy, is_valid_event, decision
  ├── severity_label, confidence
  ├── policy_scores, reason_bullets
  └── corrected_event_type (이벤트 유형 교정 가능)
```

### 크리틱 모델: LightGBM (CPU)

| 항목 | 값 |
|---|---|
| **알고리즘** | LightGBM Binary Classifier |
| **피처** | `tier1_confidence`, `keyword_count`, `object_hint_count` (3개) |
| **잎 노드** | 15 |
| **학습률** | 0.05 |
| **부스팅 라운드** | 100 |
| **최소 학습 샘플** | 30 |
| **저장** | `data/critic_models/critic_v{N}.txt` (버전 관리) |

### LoRA 파인튜닝 설정

| 항목 | 값 |
|---|---|
| **베이스 모델** | `microsoft/Florence-2-large` |
| **LoRA Rank** | 8 |
| **LoRA Alpha** | 16 |
| **Dropout** | 0.05 |
| **타겟 레이어** | `q_proj`, `v_proj`, `k_proj`, `out_proj` (어텐션) |
| **태스크 유형** | `CAUSAL_LM` |
| **옵티마이저** | AdamW (lr=1e-4, weight_decay=0.01) |
| **그래디언트 클리핑** | 1.0 |
| **그래디언트 누적** | 4 스텝 |
| **에폭** | 3 |
| **배치 크기** | 4 |
| **정밀도** | GPU: float16 / CPU: float32 |

---

## 추론 파이프라인 상세 흐름

### Caption Sharing 최적화 (핵심 설계)

이 시스템의 핵심 최적화는 **프레임당 Florence-2 추론을 1회만 수행**하는 것입니다.

```
┌────────────────────────────────────────────────────────────────┐
│ 기존 방식 (비효율):                                              │
│   Frame → Florence(cash) → Florence(fire) → Florence(violence) │
│   = GPU 추론 3회/프레임                                          │
│                                                                │
│ Caption Sharing (현재):                                          │
│   Frame → Florence(1회) → caption 텍스트 공유                    │
│        ├→ CaptionAnalyzer(cash)     ← CPU 키워드 매칭 (~μs)     │
│        ├→ CaptionAnalyzer(fire)     ← CPU 키워드 매칭 (~μs)     │
│        └→ CaptionAnalyzer(violence) ← CPU 키워드 매칭 (~μs)     │
│   = GPU 추론 1회/프레임 + CPU 키워드 매칭 3회                     │
└────────────────────────────────────────────────────────────────┘
```

### CaptionAnalyzer 키워드 매칭 엔진

Florence-2는 자유 형식 캡션을 생성합니다 (예: "A person standing at a counter holding a piece of paper").
`CaptionAnalyzer`는 이 캡션을 word-boundary regex (`\b...\b`)로 분석하여 탐지 여부를 결정합니다.

#### 키워드 계층 구조 (시나리오별)

| 계층 | 가중치 | 역할 | 예시 (cash) |
|---|---|---|---|
| `strong_positive` | 0.3~0.4 | 직접 증거 | money, cash, currency, banknote |
| `moderate_positive` | 0.1~0.15 | 간접/맥락 증거 | counter, holding, handing, reaching |
| `context_phrases` | 0.3~0.5 | 복합 구문 | "handing over", "cash register" |
| `negative` | -0.2~-0.3 | 반증 키워드 | phone, card, receipt |
| `neutralizing_phrases` | (무효화) | 강한 키워드 무효화 | "fire extinguisher" → fire 무효 |
| `object_hints` | 0 (점수 없음) | Tier-2 전달용 | paper, envelope, wallet |

#### Cash H2H (Hand-to-Hand) 탐지 전략

Florence-2는 현금 자체를 정확히 분류하지 못합니다. 대신 **손-대-손 상호작용**을 탐지합니다:

```
H2H 탐지 조건:
  위치 키워드 (counter, cashier, checkout, front desk, reception, drawer)
  AND
  행동 키워드 (handing, holding, passing, reaching, exchanging, giving, receiving)
  → 두 조건 모두 충족 시 H2H 양성 판정
```

#### 탐지 최종 게이트

```python
is_detected = confidence > 0 AND (has_strong OR has_context OR has_h2h)
```
- 점수가 0보다 크고, 최소 하나의 의미있는 신호가 있어야 탐지 판정

### EvidenceRouter 에스컬레이션 판단

| 조건 | 동작 |
|---|---|
| 신뢰도 ≥ 0.85 AND 안정성 ≥ 0.90 | **SKIP** (Tier-2 생략) |
| fire/violence (안전 위험) | **항상 ESCALATE** |
| cash: 신뢰도 < 0.55 | Tier-2 생략 (너무 약한 신호) |
| cash: 0.55 ≤ 신뢰도 < 0.85 | **ESCALATE to Gemini** |
| Gemini 호출 비율 > 30% | 소프트 패널티 적용 (비용 제어) |

**액션 공간:** `SKIP`, `GEMINI_IMG`, `GEMINI_VIDEO`, `HUMAN_QUEUE`

---

## GPU/CPU 리소스 사용 분석

### 현재 운영 환경 (AWS g4dn.xlarge)

| 리소스 | 스펙 | 현재 사용량 |
|---|---|---|
| **GPU** | Tesla T4 15GB VRAM | **2,165 MiB (~14%)** |
| **GPU 활용률** | - | **33%** |
| **GPU 온도** | - | 36°C |
| **GPU 전력** | 70W TDP | 46W (66%) |
| **CPU** | Intel Xeon 8259CL 2.50GHz × 4코어 | model_server 151% (멀티스레드) |
| **RAM** | 15GB | **12GB 사용 (80%)** |
| **디스크 (data/)** | - | 8.2GB |
| **디스크 (models/)** | - | 1.9GB (Florence-2 가중치) |

### GPU 메모리 내역 분석

```
Florence-2-large 모델 가중치 (float16):     ~1.5 GB
PyTorch CUDA 컨텍스트 + 커널 캐시:            ~300 MB
추론 시 임시 텐서 (입력/중간/출력):            ~200 MB
RTSP HW 디코딩 버퍼 (NVDEC):                 ~100 MB
────────────────────────────────────────────────────
총 GPU VRAM 사용:                            ~2.1 GB / 15 GB
여유 VRAM:                                   ~12.8 GB
```

### CPU 메모리 내역 분석

```
Florence-2 모델 (시스템 RAM 미러):             ~3 GB
Python 런타임 + 라이브러리:                     ~1.5 GB
OpenCV RTSP 버퍼 (카메라당 ~50MB):             ~100-500 MB
Ring Buffer (카메라당 ~30초 × 12fps):          ~200-800 MB
DB Server + Frontend Server:                 ~200 MB
OS + 기타:                                    ~2 GB
────────────────────────────────────────────────────
총 RAM 사용:                                  ~12 GB / 15 GB
```

### GPU 직렬화 메커니즘 (3중 잠금)

GPU 추론은 3가지 메커니즘으로 완전히 직렬화됩니다:

1. **`GLOBAL_INFERENCE_LOCK`** (`threading.Lock`) — 모든 Florence-2 추론 호출을 직렬화
2. **`INFERENCE_WORKERS=1`** — InferenceScheduler 워커 스레드 1개
3. **`GEMINI_MAX_CONCURRENT=1`** — Tier-2 API 호출도 직렬 (GPU 무관, 비용 제어)

### 컴포넌트별 리소스 사용 요약

| 컴포넌트 | GPU | CPU | RAM | I/O |
|---|---|---|---|---|
| Florence-2 추론 | **Heavy** (float16) | Low | ~1.5GB (가중치) | - |
| CaptionAnalyzer | - | **Micro** (regex) | ~1MB | - |
| StreamManager | **Light** (NVDEC) | **Medium** (OpenCV) | **Heavy** (버퍼) | RTSP 네트워크 |
| GeminiValidator | - | Low (JPEG 인코딩) | ~50MB | Cloud API |
| EvidenceRouter | - | Low (numpy dot) | ~10MB | - |
| LocalStorage | - | **Medium** (FFmpeg) | ~100MB | 디스크 쓰기 |
| FlushWorker | - | Low | ~50MB | HTTP POST |
| ShadowAgent | - | Low | ~20MB | Cloud API |
| CriticTrainer | - | **Medium** (LightGBM) | ~50MB | 디스크 |
| LoRA 학습 (오프라인) | **Very Heavy** | Medium | **~4-6GB** | 디스크 |

### 멀티카메라 확장성 예측

| 카메라 수 | GPU VRAM | RAM | 초당 Florence 추론 | 비고 |
|---|---|---|---|---|
| 1대 | ~2.1 GB | ~8 GB | 1.5 | 여유 충분 |
| 2대 | ~2.2 GB | ~10 GB | 1.5 (직렬) | 현재 운영 기준 |
| 4대 | ~2.3 GB | ~13 GB | 1.5 (직렬) | RAM 한계 근접 |
| 8대+ | ~2.5 GB | **>15 GB** | 1.5 (직렬) | **RAM 부족 가능** |

> **참고:** GPU VRAM은 카메라 수에 거의 영향받지 않습니다 (모델 가중치 공유). 병목은 RAM (Ring Buffer)과 CPU (RTSP 디코딩)입니다.

---

## 핵심 컴포넌트 상세

### StreamManager (RTSP 스트림 관리)

- **카메라별 전용 Reader 스레드**: `_reader_loop()` 데몬 스레드가 `cv2.VideoCapture`로 연속 프레임 읽기
- **Ring Buffer**: `collections.deque(maxlen=N)` — `{"frame": np.ndarray, "mono_ts": float}` 형태로 ~30초 분량 저장
- **프레임 샘플링**: N번째 프레임만 버퍼에 저장 (base ~12fps, burst ~15fps)
- **Burst 모드**: 탐지 후 3초간 샘플링 레이트 상승
- **재연결**: 지수 백오프 (1s → 2s → 4s → 8s max) + 지터
- **HW 가속**: `RTSP_HWACCEL=cuda` 옵션으로 NVIDIA NVDEC 디코딩 (CPU fallback 자동)
- **중복 방지**: 동일 RTSP URL 중복 오픈 차단 (디코더 충돌 방지)

### InferenceScheduler (추론 스케줄러)

- **Dispatcher 스레드**: 20ms 간격으로 등록된 카메라를 순회, 최신 프레임으로 InferenceJob 생성
- **Worker 스레드**: 기본 1개, 큐에서 잡을 꺼내 `_run_inference_once()` 실행
- **카메라별 중복 방지**: `pending`/`inflight` 플래그로 카메라당 최대 1개 잡만 큐에 존재
- **Active Burst**: 탐지 후 `active_burst_sec`(3초)간 FPS 증가 (3fps)
- **Stale Job 보호**: `run_id` 버전 관리로 이전 세션의 잡 자동 폐기
- **큐 크기**: `INFERENCE_QUEUE_SIZE=128` (초과 시 드롭)

### PipelineOrchestrator (파이프라인 오케스트레이터)

- **`process_frame_sequential()`** (운영 경로): Florence-2 1회 추론 → 3개 시나리오 CaptionAnalyzer 분석
- **`process_frame()`** (병렬 경로): `ThreadPoolExecutor(3)`으로 시나리오별 병렬 추론 (미사용)
- **Cash Dual-Path**: 설정 시 ROI crop + 전체 프레임 2회 추론으로 현금 탐지율 향상 (`CASH_DUAL_PATH_ENABLED`)
- **ROI 처리**: cashier zone 정의 시 프레임을 zone 영역으로 crop → 별도 추론

### EpisodeManager (에피소드 상태머신)

```
IDLE ──[detection_count ≥ 2]──→ ACTIVE ──[stability ≥ 0.6 AND confidence ≥ 0.7]──→ VALIDATING ──→ DONE
                                                                                       │
                                                                          cooldown (60s) ←─┘
```

- **안정성 점수**: 최근 10개 라벨 중 최다 라벨 비율
- **에피소드 타임아웃**: 30초 무활동 시 자동 종료
- **시나리오별 쿨다운**: 60초 (동일 이벤트 반복 방지)

---

## 자율 진화 루프 (Shadow/Evolution/LoRA)

```
Real-time Pipeline                    Background Evolution Loop
─────────────────                    ──────────────────────────
Detection Event ──enqueue──→ ShadowAgent (daemon thread)
                                  │
                                  ├→ Shadow Prompt로 재평가
                                  │   (Gemini API, 적대적 관점)
                                  │
                                  ├→ Tier-1 vs Shadow 비교
                                  │
                                  └→ FeedbackBuffer (batch 50개)
                                          │
                               ┌──────────┴──────────┐
                               ▼                      ▼
                        CriticTrainer            RuleUpdater
                        (LightGBM 학습)          (불일치율 >30% 시)
                        ├─ 3 피처               ├─ Gemini Pro 메타리파인
                        ├─ 15 잎 노드           ├─ 프롬프트 버전 관리
                        └─ critic_v{N}.txt      ├─ changelog.jsonl
                                                └─ 롤백 지원

Separately:
Detection + Gemini Validated ──→ DataCollector ──→ annotations.jsonl + images/
Human Feedback (UI) ──→ DataCollector ──→ LoRA 학습 데이터
                                              │
                                    train_lora.py (오프라인)
                                              │
                                    LoRA Adapter → FlorenceAdapter 런타임 로드
```

### Shadow 프롬프트 설계

각 시나리오별 **운영 프롬프트**(탐지 규칙)와 **쉐도우 프롬프트**(적대적 검증)가 별도 존재합니다:

- 쉐도우 프롬프트는 원래 탐지를 **도전**하는 관점으로 설계됨
- 5개 검증 체크리스트 + 오탐 카탈로그 + 판정 프레임워크 포함
- `<!-- RuleUpdater appends learned rules below this line -->` 마커를 통해 자동 규칙 추가
- 핫 리로드: 파일 시스템 mtime 변경 시 자동 재로드 (서버 재시작 불필요)

---

## 원본 아키텍처 이미지 대비 차이

원본 이미지의 큰 축(Tier1 → Tier2 → 행동 인식, 하단 Shadow/Evolve)은 유지됩니다.
다만 현재 구현은 운영 안정성과 추적성 중심으로 아래가 추가/변형되었습니다.

| 원본 블록 | 현재 구현 | 비고 |
|---|---|---|
| Tier 1: Florence 2 Visual Detector | `florence_adapter.py` + `PipelineOrchestrator` | Caption Sharing 최적화 적용 |
| Global & Local Information Fusion | `vlm_api._inference_loop`의 ROI+GLOBAL caption 결합 | `"[ROI] ... [GLOBAL] ..."` 형태 |
| Expert MLP | `EvidenceRouter.select_action()` | 정책/Q 기반 액션 선택으로 대체 |
| Tier 2: High-Inference VLM | `GeminiValidator.validate_event_evidence()` | hybrid/video_first/video_only 모드 |
| Behavioral Pattern Recognition | `cash/fire/violence` 시나리오 + `CaptionAnalyzer` | 키워드 매칭 기반 |
| Shadow Agent Layer | `agents/shadow_agent.py` | 비동기 백그라운드 분리 실행 |
| Auto Evolve Training | `evolution/critic_trainer.py`, `rule_updater.py` | LightGBM + Gemini Pro 메타 |
| LoRA Based Finetuning | `lora/data_collector.py`, `train_lora.py` | 수집 중심, 자동 스왑은 정책 단계 |

---

## 탐지 파이프라인

1. RTSP 프레임 수집 (`StreamManager` — CPU, Ring Buffer)
2. InferenceScheduler가 카메라별 최신 프레임을 큐에 등록 (20ms 폴링)
3. Worker 스레드가 `_run_inference_once()` 실행
4. Tier-1 Florence-2 분석 (`process_frame_sequential` — **GPU 1회**)
5. CaptionAnalyzer가 캡션을 3개 시나리오로 키워드 분석 (CPU)
6. Cash 시나리오는 선택적으로 ROI + Global 캡션 융합
7. `EvidenceRouter.select_action()`로 Tier-2 필요 여부 판단 (CPU)
8. 필요 시 EventPostProcessor가 비동기로 Gemini 검증 (`validate_event_evidence`)
9. Tier-2 거부 시 이벤트 저장 스킵 (`is_detected=False`)
10. 승인 이벤트만 이벤트/클립/썸네일 저장 (FFmpeg H.264)
11. FlushWorker가 DB로 주기적 동기화

### Tier-2 관련 동작

| 상태 | 의미 | 후속 동작 |
|---|---|---|
| `state=skipped` | Tier-1에서 충분히 확신 → Tier-2 호출 생략 | 바로 저장 |
| `state=done, validated=true` | 검증 통과 | 이벤트 저장 |
| `state=done, validated=false` | 검증 거부 | **저장 스킵** |
| `state=error` | Tier-2 오류 | Fail-open (저장) |

---

## UI 구성

### 1) CCTV 모니터 (`/monitor/adhoc`)

- 멀티 카메라 카드 그리드 (실시간 MJPEG 미리보기)
- 카드 클릭 시 설정 팝업 오픈
- 팝업에서 Start/Stop, Full Screen, ROI Only 제어
- ROI zone 편집(cashier/drawer), 즉시 적용
- ROI crop 실시간 미리보기
- 상태 패널: FPS, Last Event, Validation, Error
- Florence 입력 설명: ROI + Global 캡션 확인

### 2) Florence 로그 (`/monitor/florence-logs`)

- Tier-1 추론 결과 테이블 (카메라/시나리오/탐지여부 필터)
- 캡션 미리보기 + 상세 JSON 모달
- **LoRA 피드백 버튼**: accept/decline/unsure → 학습 데이터 자동 수집
- 통계 바: 총 행, 탐지 행, 평균 추론 시간

### 3) Gemini 로그 (`/monitor/gemini-logs`)

- Tier-2 상태/결정/사유 조회
- 카메라 이름 기준 필터(정확 일치 + 포함 검색)
- 시나리오/결정 상태별 필터
- `SKIP` 배지 안내 포함 (오탐 의미 아님)
- 6초마다 자동 갱신

### 4) Shadow 모니터 (`/monitor/shadow`)

- Shadow Agent 큐/통계 모니터링
- Gemini 검증 이벤트 테이블 (수락/거부/스킵)
- 인간 피드백 모달 (accept/decline/unsure + 구조화 노트)

### 5) 시스템 대시보드 (`/dashboard`)

- CPU/RAM/GPU/VRAM 실시간 메트릭 (5초 갱신)
- 모델 상태 확인
- 최근 이벤트 목록

---

## UI/UX 상세 설계

### CCTV 페이지 정보 구조

- 상단 제어: `Add Camera`, `Start All`, `Stop All`
- 본문: 카메라 카드 그리드(각 카드에 live 미리보기 + 간단 상태)
- 카드 클릭: 설정 팝업 오픈(메인 조작은 팝업에서 수행)

### 설정 팝업 구성

- 좌측: 라이브 화면 + `zoneCanvas` 오버레이
- 우측: 상태 박스(`Status/FPS/Last Event/Validation/Error`)
- 중단: ROI Zone Edit (`cashier`/`drawer` 모드 전환, 점 추가/되돌리기/초기화/적용)
- 하단: `Save + Start`, `Start`, `Stop`, `Full Screen`, `ROI Only`
- ROI 패널: zone 선택 후 crop 이미지 실시간 갱신

### ROI 편집 동작 규칙

- ROI 점은 정규화 좌표(0~1)로 localStorage에 저장
- 서버 적용 시 현재 캔버스 크기에 맞춰 픽셀 좌표로 변환해 `/api/vlm/zones/`로 전송
- `ROI Only`는 별도 모달로 열리며 `/api/vlm/crop/`를 주기 호출해 zone crop만 표시

### Florence 입력 가시화

- Cash 판단은 ROI와 Global을 결합한 캡션을 사용
- 팝업의 `ROI + Global 설명`과 `Florence ROI Input (Cash)` 영역에서 입력 텍스트 확인

### 검증 로그 상세 뷰

- 카메라 필터: 정확 일치 셀렉트 + 부분 검색 동시 지원
- 결정 필터: `approved/rejected/skip/error/pending`
- `SKIP` 의미: Tier-1 신뢰도 조건으로 Tier-2 호출 생략 상태
- 상세 버튼으로 reason/prompt/input_mode/processing_time 확인

---

## 데이터베이스 스키마

### SQLite (WAL 모드) — `data/cctv_events.db`

#### events 테이블

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `id` | INTEGER PK | 자동 증가 |
| `event_id` | TEXT UNIQUE | 이벤트 고유 ID |
| `camera_id` | TEXT | 카메라 식별자 |
| `event_type` | TEXT | 이벤트 유형 |
| `scenario` | TEXT | 시나리오 (cash/fire/violence) |
| `confidence` | REAL | Tier-1 신뢰도 |
| `tier` | INTEGER | 탐지 티어 (1 or 2) |
| `is_detected` | BOOLEAN | 최종 탐지 여부 |
| `gemini_validated` | BOOLEAN | Gemini 검증 통과 여부 |
| `gemini_confidence` | REAL | Gemini 신뢰도 |
| `gemini_reason` | TEXT | Gemini 판정 사유 |
| `caption` | TEXT | Florence 캡션 |
| `matched_keywords` | TEXT | 매칭된 키워드 |
| `clip_path` | TEXT | 비디오 클립 경로 |
| `human_feedback` | TEXT | 인간 피드백 (accept/decline/unsure) |
| `event_data` | TEXT (JSON) | 전체 이벤트 메타데이터 |
| `created_at` | TIMESTAMP | 생성 시각 |

#### cameras 테이블

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `camera_id` | TEXT UNIQUE | 카메라 식별자 |
| `rtsp_url` | TEXT | RTSP 스트림 URL |
| `base_fps` | REAL | 기본 FPS |
| `rtsp_transport` | TEXT | 전송 프로토콜 (tcp/udp) |
| `event_cooldown_sec` | REAL | 이벤트 쿨다운 시간 |
| `clip_duration_sec` | REAL | 클립 녹화 시간 |
| `evidence_mode` | TEXT | 증거 캡처 모드 |
| `cashier_zone` | TEXT (JSON) | 캐셔 존 폴리곤 (정규화 좌표) |
| `drawer_zone` | TEXT (JSON) | 서랍 존 폴리곤 (정규화 좌표) |

#### gemini_logs 테이블

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `event_id` | TEXT UNIQUE | 이벤트 ID |
| `gemini_state` | TEXT | 검증 상태 |
| `gemini_validated` | BOOLEAN | 검증 결과 |
| `validation_type` | TEXT | 검증 유형 |
| `input_mode` | TEXT | 입력 모드 (video/storyboard/image) |
| `prompt_version` | TEXT | 프롬프트 버전 |
| `processing_time_ms` | REAL | 처리 시간 |
| `log_data` | TEXT (JSON) | 전체 로그 데이터 |

#### episode_reviews 테이블

에피소드 인간 리뷰 큐 (`episode_id`, `event_id`, `camera_id`, `review_status`, `final_policy`, `is_valid_event` 등)

#### worker_leases 테이블

크로스 프로세스 워커 중복 방지 (`camera_id` UNIQUE, `instance_id`, `pid`, `last_heartbeat`)

---

## 데이터 저장 구조

기본 루트는 `MODEL_SERVER_DATA_DIR`(기본 `data/`)입니다.

```
data/
├── events/YYYYMMDD/*.json          # 이벤트 메타데이터
├── clips/YYYYMMDD/*.mp4            # H.264 비디오 클립
├── thumbnails/YYYYMMDD/*.jpg       # 이벤트 썸네일
├── validation_logs/*               # 검증 로그 (로컬)
├── lora_training/                  # LoRA 학습 데이터
│   ├── images/*.jpg                #   프레임 이미지 (JPEG q85-90)
│   ├── annotations.jsonl           #   캡션 + 라벨 주석
│   └── LoRa_Flourence_feedback/    #   Florence UI 피드백 로그
├── shadow_feedback/*.jsonl         # Shadow 에이전트 피드백
├── critic_models/critic_v*.txt     # LightGBM 모델 버전
├── rule_versions/{scenario}/       # 프롬프트 버전 히스토리
├── cctv_events.db                  # SQLite 데이터베이스
├── media_archive/                  # DB 서버 미디어 보관
├── recovery_logs/                  # 복구 스크립트 로그
└── incident_watch/                 # 인시던트 모니터링 결과
```

### 클립 저장 경로 설명

```
1. 임시 AVI(MJPG) 생성 (OpenCV)
2. FFmpeg로 최종 MP4(H.264) 변환 (libx264, preset fast, CRF 23, yuv420p, faststart)
3. 변환 실패 시에만 OpenCV(mp4v)로 최종 MP4 fallback 생성
→ 정상 시 최종 산출물은 MP4 하나 (임시 AVI는 삭제)
```

---

## LoRA 데이터 수집 정책

현재 수집 정책은 의도적으로 엄격합니다.

| 항목 | 값 |
|---|---|
| **수집 대상 시나리오** | `cash` only |
| **수집 조건** | `Tier-2 validated == true` |
| **소스** | 이벤트 clip 프레임 균등 샘플링 (기본 3장) |
| **JPEG 품질** | 일반 85, Gemini 검증 88, 인간 피드백 90 |
| **일반 프레임 수집** | 비활성 (`LORA_COLLECT_NORMAL_RATIO=0.0`) |
| **최대 샘플 수** | 50,000 (초과 시 oldest 20% 삭제) |
| **최소 학습 요건** | 50 샘플 |

### 데이터 수집 경로

```
① Gemini 검증 통과 클립 → collect_gemini_validated_clip()
② UI Florence 로그에서 인간 피드백 → collect_florence_feedback()
③ 매 추론 후 자동 수집 (탐지=항상, 일반=5%) → collect()
④ 이벤트 피드백 (accept/decline) → collect_feedback()
```

### 중요

- Shadow Agent는 LoRA 수집 주체가 아닙니다.
- LoRA 학습 자동 스왑은 완전 자동 운영 단계가 아니라, 수집/검증 기반의 운영 적용 단계입니다.

---

## API 요약

### Frontend Server (`:8002`)

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/monitor/adhoc` | CCTV 실시간 모니터링 |
| GET | `/monitor/florence-logs` | Florence 추론 로그 |
| GET | `/monitor/gemini-logs` | Gemini 검증 로그 |
| GET | `/monitor/shadow` | Shadow 에이전트 리뷰 |
| GET | `/dashboard` | 시스템 대시보드 |
| ANY | `/api/vlm/{path}` | Model Server 리버스 프록시 |
| GET | `/api/proxy/status` | 모델 서버 상태 프록시 |
| GET | `/api/proxy/events` | DB 이벤트 프록시 |
| GET | `/api/proxy/stats` | DB 통계 프록시 |
| CRUD | `/api/proxy/cameras` | 카메라 설정 CRUD 프록시 |
| GET | `/api/proxy/system` | 시스템 메트릭 (CPU/RAM/GPU) |

### Model Server (`:8000`)

| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/api/vlm/start/` | RTSP 스트림 시작 |
| POST | `/api/vlm/stop/` | 스트림 중지 |
| GET | `/api/vlm/video/` | MJPEG 스트리밍 (15fps) |
| GET | `/api/vlm/status/` | 카메라 상태 조회 |
| GET | `/api/vlm/events/` | 이벤트 목록 |
| POST | `/api/vlm/zones/` | ROI 존 설정 |
| GET | `/api/vlm/crop/` | ROI crop 미리보기 |
| POST | `/api/vlm/feedback/` | 인간 피드백 제출 |
| GET | `/api/vlm/shadow/recent/` | 최근 Shadow 결과 |

### DB Server (`:8001`)

| 메서드 | 경로 | 설명 |
|---|---|---|
| POST | `/api/flush` | 배치 이벤트/클립 수신 (multipart) |
| GET | `/api/events` | 페이지네이션 이벤트 조회 |
| GET | `/api/events/{event_id}` | 단건 이벤트 상세 |
| POST | `/api/feedback` | 인간 피드백 제출 |
| GET | `/api/stats` | 집계 통계 |
| CRUD | `/api/cameras` | 카메라 설정 CRUD |

---

## 환경 변수 전체

### AI 모델

| 변수 | 기본값 | 운영값 | 설명 |
|---|---|---|---|
| `FLORENCE_MODEL` | `microsoft/Florence-2-large` | 동일 | Florence 모델 이름 |
| `FLORENCE_BACKEND` | `pytorch` | `pytorch` | 추론 백엔드 |
| `FLORENCE_DEVICE` | `cuda` | `cuda` | 디바이스 |
| `FLORENCE_INPUT_SIZE` | `448` | **`320`** | 입력 이미지 크기 |
| `FLORENCE_MAX_TOKENS` | `512` | **`96`** | 최대 생성 토큰 |
| `FLORENCE_NUM_BEAMS` | `3` | **`1`** | 빔 서치 폭 |
| `FLORENCE_CAPTION_DETAIL` | `more` | **`detailed`** | 캡션 상세도 |
| `FLORENCE_DTYPE` | `float32` | float16(자동) | 정밀도 (CUDA시 자동 fp16) |
| `FLORENCE_LOG_PERSIST` | `false` | **`true`** | 추론 로그 디스크 저장 |
| `GEMINI_API_KEY` | (필수) | 설정됨 | Gemini API 키 |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | 동일 | Gemini 모델 |
| `GEMINI_TIMEOUT_SEC` | `30` | **`90`** | API 타임아웃 |
| `GEMINI_MAX_CONCURRENT` | `1` | `1` | 동시 호출 수 |
| `GEMINI_TEMPERATURE` | `0.1` | `0.1` | Temperature |

### 탐지 임계값

| 변수 | 기본값 | 설명 |
|---|---|---|
| `CASH_THRESHOLD` | `0.30` | Tier-1 현금 탐지 |
| `VIOLENCE_THRESHOLD` | `0.30` | Tier-1 폭력 탐지 |
| `FIRE_THRESHOLD` | `0.30` | Tier-1 화재 탐지 |
| `TIER2_CASH_THRESHOLD` | `0.55` | Tier-2 에스컬레이션 (cash) |
| `TIER2_VIOLENCE_THRESHOLD` | `0.70` | Tier-2 에스컬레이션 (violence) |
| `TIER2_FIRE_THRESHOLD` | `0.60` | Tier-2 에스컬레이션 (fire) |
| `SKIP_CONFIDENCE` | `0.85` | Tier-2 스킵 신뢰도 |
| `SKIP_STABILITY` | `0.90` | Tier-2 스킵 안정성 |

### 스트림/추론

| 변수 | 기본값 | 설명 |
|---|---|---|
| `BASE_FPS` | `1.5` | 기본 프레임 캡처 레이트 |
| `BURST_FPS` | `4.0` | 탐지 후 버스트 FPS |
| `GLOBAL_INFERENCE_LOCK` | `true` | 글로벌 추론 직렬화 |
| `INFERENCE_WORKERS` | `1` | 추론 워커 스레드 수 |
| `INFERENCE_QUEUE_SIZE` | `128` | 추론 큐 크기 |
| `INFERENCE_ACTIVE_BURST_SEC` | `3.0` | 버스트 모드 지속 시간 |
| `INFERENCE_ACTIVE_BURST_FPS` | `3.0` | 버스트 모드 FPS |
| `RTSP_TRANSPORT` | `tcp` | RTSP 전송 프로토콜 |
| `RTSP_HWACCEL` | `cuda` | HW 가속 디코딩 |
| `SINGLE_CAMERA_MODE` | `false` | 단일 카메라 제한 |

### 저장/연동

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MODEL_SERVER_DATA_DIR` | `data` | 데이터 루트 |
| `DB_PATH` | `data/cctv_events.db` | SQLite 경로 |
| `DB_MEDIA_ROOT` | `data/media_archive` | 미디어 아카이브 |
| `LOCAL_RETENTION_DAYS` | `3` | 로컬 보관 기간 |
| `USE_S3` | `false` | S3 업로드 여부 |
| `FFMPEG_PATH` | `ffmpeg` | FFmpeg 바이너리 경로 |
| `EVIDENCE_MODE` | `video_only` | 증거 캡처 모드 |

### LoRA

| 변수 | 기본값 | 설명 |
|---|---|---|
| `LORA_ENABLED` | `false` | LoRA 어댑터 활성화 |
| `LORA_DATA_COLLECTION` | `true` | 학습 데이터 수집 |
| `LORA_COLLECT_NORMAL_RATIO` | `0.0` | 일반 프레임 수집 비율 |
| `LORA_ADAPTER_PATH` | - | LoRA 어댑터 경로 |

### 서버 포트/URL

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MODEL_SERVER_PORT` | `8000` | Model Server |
| `DB_SERVER_PORT` | `8001` | DB Server |
| `FRONTEND_SERVER_PORT` | `8002` | Frontend Server |
| `MODEL_SERVER_URL` | `http://localhost:8000` | 내부 통신용 |
| `DB_SERVER_URL` | `http://localhost:8001` | 내부 통신용 |

### 부팅/복구

| 변수 | 기본값 | 설명 |
|---|---|---|
| `AUTO_RESTORE_CAMERAS_ON_BOOT` | `true` | 부팅 시 카메라 자동 복원 |
| `AUTO_RESTORE_DELAY_SEC` | `4` | 복원 대기 시간 |
| `AUTO_RESTORE_DB_RETRIES` | `20` | DB 재시도 횟수 |
| `AUTO_RESTORE_FRAME_WAIT_SEC` | `20` | 프레임 대기 시간 |
| `AUTO_RESTORE_BETWEEN_CAM_SEC` | `1.5` | 카메라간 대기 |
| `TZ` | `Asia/Seoul` | 타임존 |
| `LOG_LEVEL` | `INFO` | 로그 레벨 |

---

## 배포 구조

### AWS g4dn.xlarge 타겟 스펙

| 리소스 | 스펙 |
|---|---|
| CPU | Intel Xeon (Cascade Lake) 4 vCPU |
| RAM | 16 GB |
| GPU | NVIDIA Tesla T4 15 GB VRAM |
| 스토리지 | EBS 30GB+ (gp3 권장) |
| OS | Ubuntu 24.04 LTS |
| CUDA | 12.1+ (PyTorch CUDA) |

### systemd 서비스 구조

```
vlm-boot-recover.service (oneshot, 부팅 시)
  └→ vlm-safe-recover.sh boot-start
      ├→ vlm-db.service (worker 1, :8001)
      ├→ vlm-model.service (worker 1, :8000, GPU)
      └→ vlm-frontend.service (worker 2, :8002)

nginx.service (:80/443 → :8002)
```

### 보안

- 8000/8001/8002 포트는 `127.0.0.1`에만 바인딩 (외부 직접 접근 차단)
- nginx만 80/443 외부 노출
- RTSP 자격증명 로그 마스킹 (`rtsp://***:***@...`)
- `.env` gitignore 처리

### 복구 메커니즘 (`vlm-safe-recover.sh`)

| 단계 | 타임아웃 | 검증 방법 |
|---|---|---|
| vlm-db 시작 | 25초 | HTTP 200 응답 |
| vlm-model 시작 | 180초 | `florence_initialized=true` |
| 카메라 자동 복원 | 150초 | 모든 저장된 카메라 `running=true` |
| vlm-frontend 시작 | 40초 | HTTP 200 응답 |
| nginx + 공개 URL | 40초 | 공개 URL 접근 확인 |

---

## 로컬 실행 방법

### 1) 가상환경/의존성

```bash
# 프로젝트 디렉토리로 이동 (절대경로는 본인 환경에 맞게 변경)
cd /path/to/hio_intelligence_stream
python3 -m venv venv
source venv/bin/activate       # Linux/Mac
# .\venv\Scripts\Activate.ps1  # Windows PowerShell

# GPU 환경 (CUDA 12.1) — 반드시 requirements.txt보다 먼저 설치
pip install --no-cache-dir -r requirements_gpu.txt
pip install -r requirements.txt

# CPU-only 환경
pip install -r requirements.txt
```

### 2) 환경 파일

```bash
cp .env.example .env
# .env를 열어 아래 항목 설정:
#   GEMINI_API_KEY=your_actual_key
#   FLORENCE_DEVICE=cuda  (GPU) 또는 cpu (CPU-only)
```

### 3) 실행

```bash
python start_local.py          # 3개 서버 모두 시작
python start_local.py model db # 일부만 시작
```

접속:

- CCTV 모니터: `http://localhost:8002/monitor/adhoc`
- Florence 로그: `http://localhost:8002/monitor/florence-logs`
- Gemini 로그: `http://localhost:8002/monitor/gemini-logs`
- Shadow 모니터: `http://localhost:8002/monitor/shadow`
- 대시보드: `http://localhost:8002/dashboard`

---

## 트러블슈팅

### 1) Ctrl+C 후 즉시 종료가 안 되는 경우

`uvicorn --reload` 환경에서는 reloader 프로세스와 장시간 스트림 응답 정리 때문에 종료 로그가 지연될 수 있습니다.

### 2) `Assertion fctx->async_lock failed` / 시작 500

주 원인은 동일 RTSP를 중복으로 열 때 디코더 충돌입니다.
현재는 `/api/vlm/start/`에서 동일 RTSP active 카메라를 감지해 차단합니다.

### 3) 검증 로그의 `SKIP` 의미

`SKIP`은 Tier-2 미호출 상태입니다.
Tier-1 신뢰도가 충분해 Tier-2를 생략한 것이며, 오탐 의미가 아닙니다.

### 4) ROI 점이 안 찍히거나 어긋나는 경우

- 팝업에서 카메라 프리뷰가 완전히 로드된 뒤 클릭
- zone 적용 후 `ROI preview` 토글로 crop 화면 재확인

### 5) GPU OOM (Out of Memory)

- `FLORENCE_INPUT_SIZE`를 320 이하로 줄이기
- `FLORENCE_MAX_TOKENS`를 96 이하로 줄이기
- `FLORENCE_NUM_BEAMS=1`로 설정
- 카메라 수 줄이기 (Ring Buffer RAM 절약)

### 6) 모델 서버 시작이 느린 경우

- 최초 실행 시 Florence-2 모델 다운로드 (~1.9GB) 소요
- `deploy/setup_aws_g4dn.sh`에서 `SKIP_MODEL_PRELOAD=0`으로 사전 다운로드 가능

---

## Repo Hygiene

`.gitignore`는 대용량/민감/실행 산출물을 제외하도록 설정되어 있습니다.

- 제외: `.env`, `venv/`, `data/`, `models/`, `model_cache/`, `*.log`, 미디어 파일, 모델 가중치 (`.pt`, `.bin`, `.onnx`, `.safetensors`)
- 테스트 임시 파일은 `_tests_archive/`로 이동 후 ignore 처리
- 공유용 환경 템플릿: `.env.example`, `.env.aws`

---

## 향후 작업

- LoRA 자동 학습/자동 스왑의 운영 정책 확정
- Tier-2 검증 로그 전용 DB 조회 API 추가
- 운영 배포 프로파일(무중단 재기동, 헬스체크) 강화
- S3 경로 활성화/운영 전환
- Florence-2 OpenVINO 백엔드 완성 (현재 stub)
- INT8/INT4 양자화 도입으로 VRAM 절감
- 멀티 GPU 분산 추론 (카메라 수 확장 시)

---

## 로컬 개발 환경 세팅 상세 가이드

GitHub에서 클론 후 로컬 머신에서 바로 실행하기 위한 체크리스트입니다.

### 사전 요구사항

| 항목 | 필수 | 설명 |
|---|---|---|
| **Python** | 3.10+ (권장 3.12) | `python3 --version`으로 확인 |
| **FFmpeg** | 필수 | 클립 H.264 변환용. `ffmpeg -version`으로 확인 |
| **NVIDIA GPU + CUDA** | 권장 (없으면 CPU 모드) | `nvidia-smi`로 확인 |
| **Gemini API Key** | 필수 (Tier-2 검증용) | [Google AI Studio](https://aistudio.google.com/)에서 발급 |
| **RTSP 카메라** | 테스트용 | 없으면 UI만 확인 가능 (스트림 없이 서버 기동 가능) |

### Step-by-Step

```bash
# 1. 클론
git clone https://github.com/WhoAmI125/hio_intelligence_stream.git
cd hio_intelligence_stream

# 2. 가상환경
python3 -m venv venv
source venv/bin/activate

# 3. 의존성 설치
# GPU가 있는 경우 (CUDA 12.1):
pip install --no-cache-dir -r requirements_gpu.txt
pip install -r requirements.txt

# GPU가 없는 경우 (CPU-only):
pip install -r requirements.txt

# 4. 환경 설정
cp .env.example .env
```

### `.env` 필수 수정 항목

```bash
# 반드시 설정해야 하는 것
GEMINI_API_KEY=your_actual_gemini_api_key

# GPU가 없는 경우 아래로 변경
FLORENCE_DEVICE=cpu
RTSP_HWACCEL=
# (RTSP_HWACCEL을 빈 값으로 두면 CPU 디코딩 사용)
```

### `.env` 선택 수정 항목 (로컬 최적화)

```bash
# CPU-only 환경에서 추론 속도 개선
FLORENCE_INPUT_SIZE=256          # 320 → 256 (더 작은 입력)
FLORENCE_MAX_TOKENS=64           # 96 → 64
FLORENCE_NUM_BEAMS=1             # 이미 1이지만 확인

# 카메라 자동복원 끄기 (로컬 개발 시 불필요)
AUTO_RESTORE_CAMERAS_ON_BOOT=false

# Gemini 없이 테스트하려면 (Tier-2 검증 비활성)
# GEMINI_API_KEY를 빈 값으로 두면 자동 비활성
# → Tier-1 결과만으로 이벤트가 저장됨
```

### 실행

```bash
python start_local.py
```

서버 3개가 순차적으로 기동됩니다:
- `:8001` DB 서버 (즉시 준비)
- `:8000` 모델 서버 (**최초 실행 시 Florence-2 모델 다운로드 ~1.9GB, 2-5분 소요**)
- `:8002` 프론트엔드 (즉시 준비)

**모델 다운로드 위치**: `models/hf/` 또는 HuggingFace 기본 캐시 (`~/.cache/huggingface/`)

### 로컬에서 흔히 발생하는 문제와 해결

#### 1) `ModuleNotFoundError: No module named 'xxx'`

```bash
# 의존성 누락 — 전체 재설치
pip install -r requirements.txt
# lightgbm 관련 오류 시
pip install lightgbm scikit-learn pandas
```

#### 2) `RuntimeError: CUDA out of memory` 또는 GPU 없음

```bash
# .env에서 CPU 모드로 전환
FLORENCE_DEVICE=cpu
RTSP_HWACCEL=
```

#### 3) `Florence-2 모델 초기화 실패`

```bash
# HuggingFace 캐시 경로 확인
ls ~/.cache/huggingface/hub/
# 또는 수동 다운로드
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)"
```

#### 4) 포트 충돌 (`Address already in use`)

```bash
# 8000/8001/8002 포트 사용 중인 프로세스 확인
lsof -i :8000
lsof -i :8001
lsof -i :8002
# 필요시 종료
kill -9 <PID>
```

#### 5) FFmpeg 미설치

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows — ffmpeg.org에서 다운로드 후 PATH에 추가
# 또는 .env에 절대경로 지정:
FFMPEG_PATH=C:\tools\ffmpeg\bin\ffmpeg.exe
```

#### 6) RTSP 카메라 없이 테스트

카메라 없이도 서버는 정상 기동됩니다. UI에서 카메라를 추가하지 않으면 추론이 실행되지 않을 뿐입니다.
테스트용 RTSP 스트림이 필요하면:

```bash
# 로컬 비디오 파일을 RTSP로 변환 (ffmpeg + mediamtx 등 사용)
# 또는 공개 RTSP 테스트 스트림 사용
```

### 코드 내 경로 관련 참고사항

- **모든 데이터 경로는 상대경로**: `data/`, `models/` 등은 프로젝트 루트 기준 상대경로로 자동 생성됨
- **절대경로 하드코딩 없음**: Python 코드에는 절대경로가 없음. 경로는 모두 환경변수 또는 `os.path.join()`으로 구성
- **`deploy/` 디렉토리**: systemd 서비스 파일과 쉘 스크립트에 `/home/ubuntu/hio_intelligence_stream` 절대경로가 있으나, 이는 **AWS 배포 전용**. 로컬 개발 시 `start_local.py`를 사용하므로 무관
- **Windows 호환**: `start_local.py`가 Windows(`Scripts/python.exe`)와 Linux(`bin/python`) 모두 감지. 단, RTSP HW 가속은 Linux NVIDIA 전용

### 최소 하드웨어 요구사항

| 모드 | CPU | RAM | GPU | 디스크 |
|---|---|---|---|---|
| **GPU 모드** (권장) | 4코어+ | 8GB+ | CUDA GPU 4GB+ VRAM | 10GB+ |
| **CPU 모드** (개발/테스트) | 4코어+ | 8GB+ | 불필요 | 10GB+ |

> **참고:** CPU 모드에서 Florence-2 추론은 프레임당 2-5초 소요 (GPU 대비 ~10배 느림). 실시간 분석에는 부적합하지만 기능 테스트/UI 개발에는 충분합니다.

---

## 라이선스/주의

사내/프로젝트 목적의 운영 코드 기준 문서입니다.
실서버 적용 전 RTSP 접근권한, 개인정보/보안 정책, 저장 보존 정책을 반드시 점검하세요.
