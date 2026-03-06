# Intelligent CCTV System - VLM New Deploy

> 자율 진화형 CCTV 이상 탐지 시스템  
> Florence-2 (Tier-1) + Gemini (Tier-2) + Shadow/Evolution + LoRA 데이터 수집

---

## 최근 업데이트 (2026-03-06)

- 멀티 카메라 그리드 + 팝업 기반 설정 UI 반영 (`/monitor/adhoc`)
- ROI 편집/미리보기/ROI Only 팝업 반영
- Gemini 로그 화면 분리 및 카메라 필터 지원 (`/monitor/gemini-logs`)
- `PipelineOrchestrator` + `EvidenceRouter`를 실제 추론 루프에 연결
- Gemini Tier-2 결과가 `REJECT`면 이벤트 저장 스킵 처리
- 클립 저장: FFmpeg(H.264) 우선 + OpenCV fallback 경로
- 썸네일 저장, `last_validation`/`last_clip_path` 상태 반영
- Gemini 검증 메타를 DB `gemini_logs` 테이블에 적재
- 동일 RTSP 중복 실행 방지(충돌/디코더 오류 완화)
- LoRA 수집 정책 강화: cash + Gemini 승인 + clip 기반 샘플만 수집

---

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [현재 아키텍처](#현재-아키텍처)
3. [아키텍처 다이어그램 (ASCII)](#아키텍처-다이어그램-ascii)
4. [원본 아키텍처 이미지 대비 차이](#원본-아키텍처-이미지-대비-차이)
5. [원본 이미지 매핑 상세 (Image #1)](#원본-이미지-매핑-상세-image-1)
6. [탐지 파이프라인](#탐지-파이프라인)
7. [UI 구성](#ui-구성)
8. [UI/UX 상세 설계](#uiux-상세-설계)
9. [데이터 저장 구조](#데이터-저장-구조)
10. [LoRA 데이터 수집 정책](#lora-데이터-수집-정책)
11. [API 요약](#api-요약)
12. [로컬 실행 방법](#로컬-실행-방법)
13. [환경 변수](#환경-변수)
14. [트러블슈팅](#트러블슈팅)
15. [Repo Hygiene](#repo-hygiene)
16. [향후 작업](#향후-작업)

---

## 프로젝트 개요

이 프로젝트는 RTSP CCTV 스트림을 실시간으로 분석하여 아래 3개 시나리오를 탐지합니다.

- `cash`: 현금 거래/금전 전달
- `fire`: 화재/연기
- `violence`: 폭력/충돌

핵심 철학은 다음과 같습니다.

- Tier-1 고속 탐지 + Tier-2 정밀 검증
- 운영 중 피드백/로그 축적
- Shadow/Critic/RuleUpdater를 통한 점진적 개선

---

## 현재 아키텍처

### 3-서버 구조

- `model_server` (`:8000`): 추론, 스트림 제어, 이벤트 생성
- `db_server` (`:8001`): 이벤트/통계/피드백/검증로그 저장
- `frontend_server` (`:8002`): 모니터링 UI + reverse proxy

### 런타임 핵심 컴포넌트

- `StreamManager`: RTSP 입력 + ring buffer + burst FPS
- `FlorenceAdapter`: Tier-1 캡션/시그널
- `PipelineOrchestrator`: 시나리오 병렬 처리
- `EvidenceRouter`: evidence 선택 및 Tier-2 에스컬레이션 판단
- `GeminiValidator`: Tier-2 검증 (`validate_event_evidence`)
- `LocalStorage`: 이벤트/클립/썸네일 저장 (S3 옵션 포함)
- `FlushWorker`: Model → DB 배치 flush
- `ShadowAgent`, `CriticTrainer`, `RuleUpdater`: 진화 루프
- `DataCollector`: LoRA 학습 데이터 수집

### 아키텍처 다이어그램 (ASCII)

```text
+----------------------------------------------------------------------------------+
|                           Frontend Server (:8002)                               |
|                    Jinja2 Templates + Reverse Proxy                             |
|            /monitor/adhoc  /monitor/shadow  /monitor/gemini-logs               |
+-----------------------------------+----------------------------------------------+
                                    |
                                    | /api/vlm/*  (HTTP Proxy)
                                    v
+----------------------------------------------------------------------------------+
|                            Model Server (:8000)                                 |
|                                                                                  |
|  +--------------------+      +----------------------+     +-------------------+  |
|  | StreamManager      | ---> | FlorenceAdapter      | --> | PipelineOrchestr.|  |
|  | - RTSP Reader      |      | (Tier-1)             |     | (cash/fire/viol.)|  |
|  | - Ring Buffer      |      +----------------------+     +---------+---------+  |
|  | - Burst FPS        |                                         ROI+GLOBAL |      |
|  +---------+----------+                                                    v      |
|            |                                           +-------------------+----+ |
|            |                                           | EvidenceRouter         | |
|            |                                           | - select_action()      | |
|            |                                           +-----------+------------+ |
|            |                                                       |              |
|            |                                   escalate (uncertain)|              |
|            |                                                       v              |
|            |                                           +-----------+------------+ |
|            |                                           | GeminiValidator        | |
|            |                                           | (Tier-2)              | |
|            |                                           +-----------+------------+ |
|            |                                                       |              |
|            +------------------------ save approved events ---------+              |
|                                                                    v              |
|  +----------------------+      +----------------------+     +-------------------+ |
|  | LocalStorage         | ---> | FlushWorker          | --> | DB Server (:8001) | |
|  | - events/clips/thumb |      | periodic /api/flush |     | events/gemini_logs| |
|  | - FFmpeg H.264       |      +----------------------+     +-------------------+ |
|  +----------------------+                                                     ^    |
|                                                                                |    |
|  +----------------------+   +----------------------+   +---------------------+ |    |
|  | ShadowAgent          |<->| CriticTrainer        |<->| RuleUpdater         | |    |
|  | background analysis  |   | lightgbm critic      |   | prompt evolution    | |    |
|  +----------------------+   +----------------------+   +---------------------+ |    |
|                                                                                |    |
|  +----------------------+                                                       |    |
|  | LoRA DataCollector   |  (cash + gemini validated + clip frames only)       |    |
|  +----------------------+                                                       |    |
+----------------------------------------------------------------------------------+
```

---

## 원본 아키텍처 이미지 대비 차이

원본 이미지의 큰 축(Tier1 → Tier2 → 행동 인식, 하단 Shadow/Evolve)은 유지됩니다.  
다만 현재 구현은 운영 안정성과 추적성 중심으로 아래가 추가/변형되었습니다.

- `3-서버 분리`가 명확함 (모델/DB/프론트)
- `EvidenceRouter`가 MLP 역할 일부를 대체 (정책 기반 라우팅)
- `Gemini 검증 메타 DB 적재`로 운영 추적 강화 (`gemini_logs`)
- `FFmpeg(H.264)+fallback` 저장 경로로 웹 호환성 강화
- `S3 코드 경로`는 존재하지만 기본 운영값은 `USE_S3=false`
- `LoRA 수집`은 Shadow가 아니라 메인 추론/피드백 경로에서 수행

---

## 원본 이미지 매핑 상세 (Image #1)

아래는 원본 아키텍처 그림의 블록과 현재 구현 모듈의 1:1 매핑입니다.

| 원본 블록 | 현재 구현 | 비고 |
|---|---|---|
| Tier 1: Florence 2 Visual Detector | `model_server/adapters/florence_adapter.py` + `PipelineOrchestrator` | 실제 모델 기본값은 `microsoft/Florence-2-large` |
| Global & Local Information Fusion | `vlm_api._inference_loop`의 ROI+GLOBAL caption 결합 | Cash에서 `\"[ROI] ... [GLOBAL] ...\"` 형태로 결합 |
| Expert MLP | `EvidenceRouter.select_action()` | 정책/Q 기반 액션 선택으로 대체 |
| Tier 2: High Computing Power VLM | `gemini_validator.validate_event_evidence()` | mode: `hybrid/video_first/video_only/images_first` |
| Behavioral Pattern Recognition | `cash/fire/violence` 시나리오 분기 + `CaptionAnalyzer` | 이벤트 생성/통계/로그로 연결 |
| Shadow Agent Layer | `agents/shadow_agent.py` | 운영 추론과 분리된 백그라운드 분석 |
| Auto Evolve Training | `evolution/critic_trainer.py`, `evolution/rule_updater.py` | critic/rule update 루프 |
| LoRA Based Finetuning | `lora/data_collector.py`, `lora/train_lora.py` | 현재는 데이터 수집 중심, 자동 스왑은 정책 단계 |

운영 경로 기준 핵심 포인트:

- `PipelineOrchestrator`와 `EvidenceRouter`는 실 추론 루프에서 호출됩니다.
- Tier-2가 `validated=false`면 이벤트를 저장하지 않고 건너뜁니다.
- Shadow는 LoRA 수집 주체가 아니라 보조 진화/평가 레이어입니다.

---

## 탐지 파이프라인

1. RTSP 프레임 수집 (`StreamManager`)
2. Tier-1 Florence 분석 (`PipelineOrchestrator`)
3. Cash 시나리오는 ROI + Global 캡션 융합
4. `EvidenceRouter.select_action()`로 Tier-2 필요 여부 판단
5. 필요 시 Gemini 검증 (`validate_event_evidence`)
6. Gemini 거부 시 이벤트 저장 스킵
7. 승인 이벤트만 이벤트/클립/썸네일 저장
8. flush worker가 DB로 주기적 동기화

### Tier-2 관련 동작

- `state=skipped`: Tier-1에서 충분히 확신하여 Gemini 호출 생략
- `state=done + validated=true`: 검증 통과
- `state=done + validated=false`: 검증 거부(저장 스킵)
- `state=error`: Gemini 오류 시 fail-open (운영 가용성 우선)

---

## UI 구성

### 1) CCTV 모니터 (`/monitor/adhoc`)

- 멀티 카메라 카드 그리드
- 카드 클릭 시 설정 팝업 오픈
- 팝업에서 Start/Stop, Full Screen, ROI Only 제어
- ROI zone 편집(cashier/drawer), 즉시 적용
- ROI crop 실시간 미리보기
- 상태 패널: FPS, Last Event, Gemini, Error
- Florence 입력 설명: ROI + Global 캡션 확인

### 2) Shadow 모니터 (`/monitor/shadow`)

- Shadow Agent 큐/통계 모니터링
- 운영 판단과 분리된 백그라운드 분석 확인

### 3) Gemini 로그 (`/monitor/gemini-logs`)

- Gemini 상태/결정/사유 조회
- 카메라 이름 기준 필터(정확 일치 + 포함 검색)
- 시나리오/결정 상태별 필터
- `SKIP` 배지 안내 포함 (오탐 의미 아님)

---

## UI/UX 상세 설계

현재 UI는 `Hotel-Cash-Detector` 스타일의 \"미리보기 중심 + 팝업 상세 설정\" 흐름으로 구성됩니다.

### CCTV 페이지 정보 구조

- 상단 제어: `Add Camera`, `Start All`, `Stop All`
- 본문: 카메라 카드 그리드(각 카드에 live 미리보기 + 간단 상태)
- 카드 클릭: 설정 팝업 오픈(메인 조작은 팝업에서 수행)

### 설정 팝업 구성

- 좌측: 라이브 화면 + `zoneCanvas` 오버레이
- 우측: 상태 박스(`Status/FPS/Last Event/Gemini/Error`)
- 중단: ROI Zone Edit (`cashier`/`drawer` 모드 전환, 점 추가/되돌리기/초기화/적용)
- 하단: `Save + Start`, `Start`, `Stop`, `Full Screen`, `ROI Only`
- ROI 패널: zone 선택 후 crop 이미지 실시간 갱신

### ROI 편집 동작 규칙

- ROI 점은 정규화 좌표(0~1)로 localStorage에 저장됩니다.
- 서버 적용 시 현재 캔버스 크기에 맞춰 픽셀 좌표로 변환해 `/api/vlm/zones/`로 전송됩니다.
- `ROI Only`는 별도 모달로 열리며 `/api/vlm/crop/`를 주기 호출해 zone crop만 표시합니다.

### Florence 입력 가시화

- Cash 판단은 ROI와 Global을 결합한 캡션을 사용합니다.
- 팝업의 `ROI + Global 설명`과 `Florence ROI Input (Cash)` 영역에서 입력 텍스트를 확인할 수 있습니다.

### Gemini 로그 상세 뷰

- 카메라 필터: 정확 일치 셀렉트 + 부분 검색 동시 지원
- 결정 필터: `approved/rejected/skip/error/pending`
- `SKIP` 의미: Tier-1 신뢰도 조건으로 Tier-2 호출 생략 상태
- 상세 버튼으로 reason/prompt/input_mode/processing_time 확인

---

## 데이터 저장 구조

기본 루트는 `MODEL_SERVER_DATA_DIR`(기본 `data/`)입니다.

- 이벤트 JSON: `data/events/YYYYMMDD/*.json`
- 클립 MP4: `data/clips/YYYYMMDD/*.mp4`
- 썸네일 JPG: `data/thumbnails/YYYYMMDD/*.jpg`
- Gemini 로그 파일(로컬): `data/gemini_logs/*`
- LoRA 데이터: `data/lora_training/`

### 클립 저장 경로 설명

클립은 두 번 저장되지 않습니다.

- 1차: 임시 AVI(MJPG) 생성
- 2차: FFmpeg로 최종 MP4(H.264) 변환
- 변환 실패 시에만 OpenCV(mp4v)로 최종 MP4 fallback 생성

즉, 정상 시 최종 산출물은 MP4 하나입니다(임시 AVI는 삭제).

---

## LoRA 데이터 수집 정책

현재 수집 정책은 의도적으로 엄격합니다.

- 수집 대상: `cash` 시나리오만
- 조건: `Gemini validated == true`
- 소스: 이벤트 clip 프레임 샘플링(기본 3장)
- 일반 프레임 랜덤 수집: 비활성 (`LORA_COLLECT_NORMAL_RATIO=0.0` 권장)

### 중요

- Shadow Agent는 LoRA 수집 주체가 아닙니다.
- LoRA 학습 자동 스왑은 완전 자동 운영 단계가 아니라, 수집/검증 기반의 운영 적용 단계입니다.

---

## API 요약

### Frontend Server (`:8002`)

- `GET /monitor/adhoc`
- `GET /monitor/shadow`
- `GET /monitor/gemini-logs`
- `GET|POST /api/vlm/{path}` (Model Server reverse proxy)

### Model Server (`:8000`)

- `POST /api/vlm/start/`
- `POST /api/vlm/stop/`
- `GET /api/vlm/video/`
- `GET /api/vlm/status/`
- `GET /api/vlm/events/`
- `POST /api/vlm/zones/`
- `GET /api/vlm/crop/`
- `POST /api/vlm/feedback/`
- `GET /api/vlm/shadow/recent/`

### DB Server (`:8001`)

- `POST /api/flush`
- `GET /api/events`
- `GET /api/events/{event_id}`
- `POST /api/feedback`
- `GET /api/stats`

---

## 로컬 실행 방법

### 1) 가상환경/의존성

```powershell
cd E:\02_StayG\00_CCTV_Motion_Detection\vlm_gravity\vlm_new_deploy
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

필요 시(모델 쪽 의존성 분리 설치):

```powershell
pip install -r model_server\requirements.txt
```

### 2) 환경 파일

```powershell
copy .env.example .env
```

`.env`에 최소 아래를 채우세요.

- `GEMINI_API_KEY`
- `FLORENCE_DEVICE` (`cuda`/`cpu`)
- 포트/URL (`MODEL_SERVER_PORT`, `DB_SERVER_URL` 등)

### 3) 실행

```powershell
python start_local.py
```

접속:

- CCTV 모니터: `http://localhost:8002/monitor/adhoc`
- Shadow 모니터: `http://localhost:8002/monitor/shadow`
- Gemini 로그: `http://localhost:8002/monitor/gemini-logs`
- Dashboard: `http://localhost:8002/dashboard`

---

## 환경 변수

대표 변수만 정리합니다. 전체는 `.env.example` 참고.

- 모델
  - `FLORENCE_MODEL=microsoft/Florence-2-large`
  - `FLORENCE_DEVICE=cuda`
  - `GEMINI_MODEL=gemini-2.5-flash-lite`
- 임계값
  - `CASH_THRESHOLD`, `VIOLENCE_THRESHOLD`, `FIRE_THRESHOLD`
- 스트림
  - `BASE_FPS`, `BURST_FPS`, `RTSP_TRANSPORT`
- 저장/연동
  - `MODEL_SERVER_DATA_DIR`, `DB_PATH`
  - `USE_S3=false` (기본)
  - `FFMPEG_PATH=ffmpeg`
- LoRA
  - `LORA_DATA_COLLECTION=true`
  - `LORA_COLLECT_NORMAL_RATIO=0.0`

---

## 트러블슈팅

### 1) Ctrl+C 후 즉시 종료가 안 되는 경우

`uvicorn --reload` 환경에서는 reloader 프로세스와 장시간 스트림 응답 정리 때문에 종료 로그가 지연될 수 있습니다.

- `Waiting for connections to close`가 잠깐 보일 수 있음
- 동일 RTSP 중복 실행이 있으면 종료/재시작이 더 느려질 수 있음

### 2) `Assertion fctx->async_lock failed` / 시작 500

주 원인은 동일 RTSP를 중복으로 열 때 디코더 충돌입니다.  
현재는 `/api/vlm/start/`에서 동일 RTSP active 카메라를 감지해 차단합니다.

### 3) Gemini 로그의 `SKIP` 의미

`SKIP`은 Gemini 미호출 상태입니다.  
Tier-1 신뢰도가 충분해 Tier-2를 생략한 것이며, 오탐 의미가 아닙니다.

### 4) ROI 점이 안 찍히거나 어긋나는 경우

- 팝업에서 카메라 프리뷰가 완전히 로드된 뒤 클릭
- zone 적용 후 `ROI preview` 토글로 crop 화면 재확인
- 필요 시 브라우저 새로고침 후 재설정

---

## Repo Hygiene

`.gitignore`는 대용량/민감/실행 산출물을 제외하도록 설정되어 있습니다.

- 제외 예시: `.env`, `venv/`, `data/`, `models/`, `model_cache/`, `*.log`, 미디어 파일
- 테스트 임시 파일은 `_tests_archive/`로 이동 후 ignore 처리
- 공유용 환경 템플릿은 `.env.example` 사용

---

## 향후 작업

- LoRA 자동 학습/자동 스왑의 운영 정책 확정
- Gemini 로그 전용 DB 조회 API 추가(현재는 이벤트 기반 조회 중심)
- 운영 배포 프로파일(무중단 재기동, 헬스체크) 강화
- 필요 시 S3 경로 활성화/운영 전환

---

## 라이선스/주의

사내/프로젝트 목적의 운영 코드 기준 문서입니다.  
실서버 적용 전 RTSP 접근권한, 개인정보/보안 정책, 저장 보존 정책을 반드시 점검하세요.
