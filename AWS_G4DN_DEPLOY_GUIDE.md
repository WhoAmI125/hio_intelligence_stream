# AWS g4dn 배포 가이드 (통합)

Last Updated: 2026-02-27
Target: Ubuntu 24.04 LTS, g4dn.xlarge (Tesla T4), `/home/ubuntu/vlm_pipipeline`

---

## 현재 배포 상태

| 항목 | 상태 |
|---|---|
| 인스턴스 | g4dn.xlarge / Tesla T4 15GB |
| NVIDIA 드라이버 | 570.211.01 (DKMS, kernel 6.14 + 6.17 양쪽 설치됨) |
| CUDA 버전 | 12.8 (드라이버 제공) / torch cu121 |
| Florence-2 | GPU 실행 중 (VRAM ~982MiB 점유) |
| torch | 2.5.1+cu121, `cuda.is_available()=True` |
| 서비스 | `hotel-cctv-vlm.service` active (running) |
| 바인드 | `127.0.0.1:8010` (gunicorn) |
| 프로젝트 경로 | `/home/ubuntu/vlm_pipipeline` |

> **과거 문제 (해결됨):** 초기 배포 시 NVIDIA 드라이버가 설치되지 않아 Florence가 CPU로 동작했음.
> 드라이버 설치 후 서비스 재시작으로 GPU 전환 완료. 아래 3-2단계 참고.

---

## 시스템 아키텍처

### 배포 구조

```
인터넷
  |
nginx (80/443)
  |
gunicorn (127.0.0.1:8010, workers=1, threads=4)
  |
Django VLM app (hotel_cctv.settings_vlm_pipeline)
  |
  +--> Tier1: Florence-2 (GPU, PyTorch backend)
  +--> Router / Critic (action selection)
  +--> Tier2: Gemini validation (image/video)
  +--> Gemini Shadow Runtime (parallel, non-intervening)
  +--> LightGBM Shadow Collector (parallel, non-intervening)
  +--> DB (SQLite or PostgreSQL)
```

### 설계 원칙

1. 프로덕션 판정 경로는 안정적으로 유지한다.
2. 개선 실험은 shadow/collector 경로에서만 수행한다.
3. 사람 피드백을 1급 학습 데이터로 영속 저장한다.
4. 로그는 감사/재현을 위해 append-only JSONL로 유지한다.

### 파이프라인 다이어그램

```
RTSP Stream
   |
   v
AdhocRTSPVLMWorker (web/services/adhoc_worker.py)
   |
   +--> Tier1: VLMUnifiedDetector (Florence-2, GPU)
   |      - 이벤트 후보 생성
   |      - tier1 confidence/stability 산출
   |
   +--> EvidenceRouter (+ Critic)
   |      - action: SKIP | GEMINI_IMG | GEMINI_VIDEO | HUMAN_QUEUE
   |      - router_steps chain 기록
   |
   +--> Tier2: Gemini Validation
   |      - image/video 증거 검증
   |      - confidence/reason/state 반환
   |
   +--> Gemini Shadow Runtime (병렬, 비개입)
   |      - shadow 판정
   |      - 피드백으로부터 prompt refine
   |
   +--> LightGBM Shadow Collector (병렬, 비개입)
          - decision snapshot 스트림
          - feedback 스트림
          - 학습 행 JSONL 영속화
```

### 구현 범위

| 영역 | 상태 | 비고 |
|---|---|---|
| Tier1 Florence GPU 추론 | 구현됨 | 프로덕션 경로 |
| Tier2 Gemini 검증 | 구현됨 | image/video |
| Router/Critic 판정 | 구현됨 | shadow-safe 기본값 |
| Main feedback API + DB | 구현됨 | `EpisodeReview` 영속화 |
| Shadow feedback API + override UI | 구현됨 | main review 상태와 분리 |
| Gemini shadow runtime | 구현됨 | prompt refine 포함 |
| LightGBM shadow collector | 구현됨 (수집 전용) | 프로덕션 개입 없음 |
| LightGBM live scoring | **미구현** | 의도적 제외 |

---

## 배포 절차 (신규 인스턴스 기준)

### 전체 순서 요약

```
1.  EC2 생성 (Ubuntu 24.04, g4dn.xlarge, 30GB+)
2.  Security Group: 22, 80, 443 오픈
3.  DNS A 레코드 → EC2 Public IP
4.  볼륨 크기 변경 후: growpart + resize2fs
5.  sudo chmod 755 /home/ubuntu
6.  NVIDIA 드라이버 설치 (★ 반드시 선행)
7.  sudo venv/bin/pip install --no-cache-dir -r requirements_gpu.txt
8.  sudo rm -rf /root/.cache/pip
9.  sudo NGINX_MODE=bootstrap bash deploy/setup_aws_g4dn.sh
10. GPU 동작 확인 (nvidia-smi, torch.cuda.is_available)
11. curl -s http://127.0.0.1:8010/api/vlm/status/ 확인
12. dig +short dev-cctv.hio.ai.kr 로 DNS 전파 확인
13. sudo certbot --nginx -d dev-cctv.hio.ai.kr --non-interactive --agree-tos -m your@email.com
14. curl -s https://dev-cctv.hio.ai.kr/api/vlm/status/ 최종 확인
```

---

### 1단계: EC2 인스턴스 준비

**인스턴스 사양**
- AMI: Ubuntu 24.04 LTS (일반 Ubuntu, Deep Learning AMI 불필요)
- 인스턴스 타입: `g4dn.xlarge`
- 루트 볼륨: **30GB 이상** (필수)
  - torch CUDA 패키지 ~3GB, 모델 캐시 ~1GB, 드라이버 ~500MB 필요
  - 기본 8GB는 절대 부족

**Security Group 인바운드 규칙**

| 포트 | 프로토콜 | Source | 용도 |
|---|---|---|---|
| 22 | TCP | 관리자 IP | SSH |
| 80 | TCP | 0.0.0.0/0 | HTTP (certbot 인증 + 리다이렉트) |
| 443 | TCP | 0.0.0.0/0 | HTTPS |

> 포트 81은 사용하지 않음. 80과 443만 열면 됨.

**DNS 설정**
도메인 A 레코드를 EC2 Public IPv4로 변경. certbot 실행 전에 반드시 완료.

```bash
dig +short dev-cctv.hio.ai.kr
# → EC2 Public IP가 나와야 함
```

---

### 2단계: 볼륨 확장

AWS 콘솔에서 볼륨 크기 변경 후 OS에서도 반드시 확장해야 적용된다.

```bash
df -h /
lsblk

sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/nvme0n1p1

df -h /   # 변경된 용량 확인
```

---

### 3단계: /home/ubuntu 권한 설정

서비스 계정(`vlmapp`)이 프로젝트 폴더에 접근할 수 있도록 **미리** 설정.
빠뜨리면 서비스가 `Permission denied (CHDIR)` 오류로 시작되지 않음.

```bash
sudo chmod 755 /home/ubuntu
```

---

### 4단계: NVIDIA 드라이버 설치 ★

> **이 단계를 누락하면 Florence가 CPU로 동작한다.**
> 드라이버가 없을 때 `torch.cuda.is_available()`은 False를 반환하고,
> `VLM_DEVICE=auto` 설정이 CPU로 폴백됨.

```bash
# 커널 헤더 (이미 설치됐을 수 있음)
sudo apt-get install -y linux-headers-$(uname -r)

# NVIDIA 서버 드라이버 설치 (Tesla T4 권장 버전)
sudo apt-get install -y nvidia-driver-570-server

# 커널 모듈 로드 (재부팅 없이 즉시 적용)
sudo modprobe nvidia

# 확인
nvidia-smi
# Tesla T4 / Driver 570.x.x / CUDA 12.x 가 표시되어야 함
```

**DKMS 상태 확인** (양쪽 커널에 컴파일됐는지)

```bash
dkms status
# nvidia-srv/570.x.x, 6.14.x-aws, x86_64: installed
# nvidia-srv/570.x.x, 6.17.x-aws, x86_64: installed  ← 양쪽 모두 있어야 함
```

---

### 5단계: torch CUDA 선설치

setup 스크립트 내부에서 torch를 설치하면 pip 캐시 ~2.7GB가 추가로 사용되어 디스크가 가득 찬다.
**반드시 `--no-cache-dir`로 먼저 설치**한 후 setup 스크립트를 실행.

```bash
cd /home/ubuntu/vlm_pipipeline
sudo venv/bin/pip install --no-cache-dir -r requirements_gpu.txt
```

설치 후 pip 캐시 정리:
```bash
sudo rm -rf /root/.cache/pip
df -h /   # 여유 공간 확인 (최소 3GB 이상 필요)
```

> `requirements_gpu.txt`는 `--index-url https://download.pytorch.org/whl/cu121` 기반.
> `requirements.txt`보다 **반드시 먼저** 설치해야 CPU torch가 CUDA 버전을 덮어쓰지 않음.

---

### 6단계: setup 스크립트 실행

torch가 설치된 상태에서 실행하면 CUDA torch 단계를 건너뛰고 나머지만 진행한다.

```bash
cd /home/ubuntu/vlm_pipipeline
sudo NGINX_MODE=bootstrap bash deploy/setup_aws_g4dn.sh
```

스크립트가 하는 일:
1. 시스템 패키지 설치 (nginx, ffmpeg, certbot 등)
2. `vlmapp` 서비스 계정 생성
3. venv 생성 및 의존성 설치
4. Florence-2 모델 캐시 다운로드
5. 런타임 디렉토리 생성
6. DB migrate, collectstatic
7. nginx 설정 (bootstrap = HTTP only)
8. systemd 서비스 등록 및 시작

> Ubuntu 24.04에서 `libgl1-mesa-glx` 패키지명이 `libgl1`로 변경됨. 스크립트에 이미 반영됨.

---

### 7단계: GPU 동작 확인

서비스 시작 후 Florence가 GPU에 올라갔는지 반드시 확인한다.

```bash
# GPU VRAM 확인 (Florence 로드 시 ~1GB 점유)
nvidia-smi
# Memory-Usage: ~1000MiB / 15360MiB 이상이면 정상

# Python에서 직접 확인
source /home/ubuntu/vlm_pipipeline/venv/bin/activate
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '/', torch.cuda.get_device_name(0))"
# CUDA: True / Tesla T4 가 나와야 함
```

Florence는 lazy load이므로 첫 요청이 들어와야 VRAM에 올라간다.
로드 전 nvidia-smi에 프로세스가 없는 것은 정상.

---

### 8단계: 서비스 동작 확인

```bash
# 서비스 상태
sudo systemctl status hotel-cctv-vlm --no-pager
sudo systemctl status nginx --no-pager

# API 응답 확인
curl -s http://127.0.0.1:8010/api/vlm/status/
# → {"running": false, "status": "stopped", ...} 가 나와야 함

# nginx 경유 확인
curl -s http://localhost/api/vlm/status/
```

**서비스 시작 안 될 때**

| 증상 | 원인 | 해결 |
|---|---|---|
| `status=200/CHDIR` | `/home/ubuntu` 권한 750 | `sudo chmod 755 /home/ubuntu` |
| gunicorn 워커 응답 없음 | 이전 프로세스 잔존 | `sudo systemctl restart hotel-cctv-vlm` |
| Florence CPU 동작 | 드라이버 미설치 또는 서비스가 드라이버 설치 전에 시작됨 | 드라이버 설치 후 `sudo systemctl restart hotel-cctv-vlm` |

---

### 8-1단계: 실전 오류 요약 (빠른 해결용)

이번 배포에서 실제로 발생했던 오류를 다음 표로 정리한다.

| 증상 | 원인 | 즉시 해결 | 재발 방지 |
|---|---|---|---|
| `ssh ... Permission denied (publickey)` | 잘못된 키 사용 또는 서버에 공개키 미등록 | 올바른 `IdentityFile`로 접속, 서버 `~/.ssh/authorized_keys` 확인 | 인스턴스 생성 시 키페어 이름/파일을 문서화 |
| 81 포트를 열어야 하는지 혼선 | 서비스는 81 미사용 | SG에서 81 제거, 80/443만 유지 | 포트 정책 고정: 22, 80, 443 |
| 도메인 접속 타임아웃 | DNS가 이전 IP를 가리킴 | `dig +short dev-cctv.hio.ai.kr` 확인 후 A 레코드 수정 | 배포 시작 전 DNS 확인을 필수 게이트로 둠 |
| HTTP는 되는데 HTTPS 실패 (`443 refused`) | 443 미개방 또는 certbot 미실행 | SG 443 오픈 → `sudo certbot --nginx -d dev-cctv.hio.ai.kr` | bootstrap -> certbot -> SSL 전환 순서 고정 |
| `DisallowedHost: Invalid HTTP_HOST header` | IP 직접 접근 시 `ALLOWED_HOSTS`에 IP 누락 | `.env`의 `ALLOWED_HOSTS`에 서버 IP 추가 후 서비스 재시작 | 운영은 도메인 기준, IP 접근 필요 시만 명시적으로 추가 |
| `/api/vlm/status`에서 `running=false` | 웹 서비스는 정상이나 RTSP 워커 미시작 | `/monitor/adhoc/`에서 시작 또는 `POST /api/vlm/start/` 호출 | 배포 검증 시 워커 시작/중지 시나리오 포함 |
| `No space left on device` (torch 설치) | 루트 볼륨 부족 + pip 캐시 과다 | 볼륨 확장, `--no-cache-dir`, pip 캐시 정리 | 최초 볼륨 30GB+, CUDA torch 선설치 규칙 준수 |
| `python3 -m venv` 실패 (`ensurepip`) | `python3-venv` 누락 | `sudo apt-get install -y python3-venv` | 베이스 패키지 체크리스트에 포함 |
| `ImportError: libGL.so.1` | OpenCV 시스템 라이브러리 누락 | `sudo apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg` | setup 스크립트의 시스템 패키지 단계 선확인 |

#### 5분 진단 커맨드 세트

```bash
# 1) DNS/포트
dig +short dev-cctv.hio.ai.kr
nc -zv <PUBLIC_IP> 22
nc -zv <PUBLIC_IP> 80
nc -zv <PUBLIC_IP> 443

# 2) 서비스/포트
sudo systemctl status hotel-cctv-vlm --no-pager
sudo systemctl status nginx --no-pager
ss -lntp | rg ':8010|:80|:443'

# 3) API 상태
curl -s http://127.0.0.1:8010/api/vlm/status/
curl -I http://dev-cctv.hio.ai.kr/
curl -I https://dev-cctv.hio.ai.kr/

# 4) 최근 오류 로그
sudo journalctl -u hotel-cctv-vlm -n 200 --no-pager
sudo tail -n 200 /var/log/gunicorn/vlm_error.log
```

---

### 9단계: SSL 인증서 발급

DNS가 이 서버의 IP를 가리키고, 포트 80이 열려있는 상태에서 실행.

```bash
dig +short dev-cctv.hio.ai.kr   # IP가 맞아야 함

sudo certbot --nginx -d dev-cctv.hio.ai.kr --non-interactive --agree-tos -m your@email.com
```

발급 후 nginx가 자동으로 HTTPS로 전환되고 80→443 리다이렉트가 설정됨.

```bash
curl -s https://dev-cctv.hio.ai.kr/api/vlm/status/
```

---

## 환경 변수

### 필수

| 변수 | 설명 |
|---|---|
| `SECRET_KEY` | Django 시크릿 키 |
| `ALLOWED_HOSTS` | 예: `localhost,127.0.0.1,dev-cctv.hio.ai.kr` (IP 직접 접근 테스트가 필요하면 `,15.165.161.171` 추가) |
| `GEMINI_API_KEY` | Gemini API 키 |
| `VLM_DEVICE` | `auto` 권장 (GPU 있으면 자동으로 cuda 선택) |

### DB

| 변수 | 기본값 | 설명 |
|---|---|---|
| `VLM_PIPELINE_DB_ENGINE` | `sqlite3` | `sqlite3` 또는 `postgresql` |
| `VLM_PIPELINE_DB_NAME` | - | DB 이름 (PostgreSQL 시) |
| `VLM_PIPELINE_DB_USER` | - | |
| `VLM_PIPELINE_DB_PASSWORD` | - | |
| `VLM_PIPELINE_DB_HOST` | - | |
| `VLM_PIPELINE_DB_PORT` | `5432` | |
| `VLM_PIPELINE_DB_CONN_MAX_AGE` | `60` | |

### VLM 모델

| 변수 | 기본값 | 설명 |
|---|---|---|
| `VLM_MODEL` | `florence-2-base` | |
| `VLM_BACKEND` | `pytorch` | |
| `VLM_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `VLM_INPUT_SIZE` | `448` | |
| `VLM_MODEL_CACHE` | `cctv/model_examine/model_cache` | |

### Gemini Shadow

| 변수 | 기본값 |
|---|---|
| `VLM_GEMINI_SHADOW_ENABLED` | - |
| `VLM_GEMINI_SHADOW_PROMPT_DIR` | - |
| `VLM_GEMINI_SHADOW_REFINE_ENABLED` | - |
| `VLM_GEMINI_SHADOW_REFINE_THRESHOLD` | `10` |
| `VLM_GEMINI_SHADOW_REFINE_DAILY_CAP` | - |

### LightGBM Collector

| 변수 | 설명 |
|---|---|
| `VLM_LGBM_SHADOW_ENABLED` | 수집 활성화 |
| `VLM_LGBM_SHADOW_LOG_MODE` | `audit` (기본) / `legacy` |
| `VLM_LGBM_SHADOW_LOG_DIR` | `LOG_MODE=legacy` 시 사용 |
| `VLM_LGBM_SHADOW_QUEUE_MAXSIZE` | 큐 최대 크기 |
| `VLM_LGBM_SHADOW_EVENT_CACHE_MAXLEN` | 이벤트 캐시 크기 |

---

## API 레퍼런스

### 워커 제어

```
POST /api/vlm/start/
POST /api/vlm/stop/
GET  /api/vlm/status/
```

**`/api/vlm/start/` 요청 예시**
```json
{
  "rtsp_url": "rtsp://...",
  "base_fps": 1.5,
  "burst_fps": 4.0,
  "burst_duration_sec": 3.0
}
```

**`/api/vlm/status/` 주요 필드**

| 필드 | 설명 |
|---|---|
| `running`, `status`, `last_error` | 워커 상태 |
| `router_step_counts`, `chain_completeness` | 라우터 통계 |
| `shadow.enabled`, `shadow.queue_size`, `shadow.last_error` | Gemini shadow 상태 |
| `lgbm_shadow.enabled`, `lgbm_shadow.queue_size` | LightGBM collector 상태 |

### 피드백

**Main 피드백 (DB 영속화)**

```
POST /api/vlm/feedback/
```

```json
{
  "event_id": "ev_...",
  "decision": "decline",
  "note": "not a cash transaction",
  "error_type": "false_positive",
  "missed_focus": ["hands outside cashier zone", "receipt exchange only"],
  "suggestion": "strengthen hand-to-hand constraint for cash"
}
```

`decision` 허용값: `accept` / `decline` / `unsure` (또는 `true` / `false` 호환)

**Shadow 피드백 (DB 비저장, shadow 전용)**

```
POST /api/vlm/shadow/feedback/
```

```json
{
  "event_id": "ev_...",
  "decision": "decline",
  "note": "shadow correction",
  "error_type": "false_positive",
  "missed_focus": ["receipt_only"],
  "suggestion": "",
  "override": true
}
```

- 최초 제출 후 재제출 시 `"override": true` 필요
- `/monitor/shadow/`에서 `reviewed` 상태 확인 가능

### 피드백 범위 분리 (중요)

| 구분 | 엔드포인트 | DB 저장 | EpisodeReview | Router 기록 |
|---|---|---|---|---|
| Main feedback | `POST /api/vlm/feedback/` | O | O | O |
| Shadow feedback | `POST /api/vlm/shadow/feedback/` | X | X | X |

- `/monitor/adhoc/`, `/monitor/simple/` → main feedback 상태 읽음
- `/monitor/shadow/` → shadow feedback 상태 읽음, Edit/Override 지원

### 모니터 페이지

| URL | 설명 |
|---|---|
| `/monitor/adhoc/` | 실시간 이벤트 모니터 |
| `/monitor/shadow/` | Shadow 판정 리뷰 패널 |
| `/monitor/simple/` | 간단 모니터 |

---

## 데이터 저장소

### DB (SQLite 기본)

테이블: `vlm_episode_reviews`

| 필드 그룹 | 주요 필드 |
|---|---|
| 식별자 | `episode_id`, `event_id`, `camera_id`, `event_type` |
| 최종 판정 | `final_policy`, `is_valid_event`, `review_status` |
| Gemini 컨텍스트 | `gemini_state`, `gemini_validated`, `gemini_confidence`, `gemini_reason` |
| 스냅샷 | `tier1_snapshot`, `router_snapshot`, `packet_summary`, `florence_signals` |
| 구조화 피드백 | `feedback_error_type`, `feedback_missed_focus`, `feedback_suggestion` |

> Shadow feedback 엔드포인트는 이 테이블에 쓰지 않음.

### JSONL 로그

경로: `media_vlm_pipeline/vlm_logs/YYYYMMDD/{camera_id}/`

| 파일 | 설명 |
|---|---|
| `adhoc_audit.jsonl` | 프레임 처리 이력 |
| `orchestrator.jsonl` | 오케스트레이터 결정 |
| `router_steps.jsonl` | 라우터 단계 체인 |
| `agent_cash.jsonl` / `agent_violence.jsonl` / `agent_fire.jsonl` | 에이전트별 로그 |
| `shadow_gemini_decisions.jsonl` | Gemini shadow 판정 |
| `lgbm_shadow/lgbm_decisions.jsonl` | LightGBM 수집 스냅샷 |
| `lgbm_shadow/lgbm_training_rows.jsonl` | 학습 행 (피드백 수신 후 확정) |
| `lgbm_shadow/lgbm_collector_errors.jsonl` | 수집기 오류 |

---

## 사전 배포 검증 체크리스트

배포 전 로컬/CI 에서 수행:

```bash
# Django 설정 검증
python manage_vlm.py check --settings=hotel_cctv.settings_vlm_pipeline

# 마이그레이션 계획 확인
python manage_vlm.py migrate --settings=hotel_cctv.settings_vlm_pipeline --plan

# 스탠드얼론 무결성 확인
python deploy/verify_standalone.py

# LightGBM 스모크 테스트
# collector instantiate → start → enqueue → feedback → stop
```

배포 후 운영 서버에서 확인:

```bash
# GPU 상태
nvidia-smi

# CUDA 인식 확인
source venv/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 서비스 API 응답
curl -s http://127.0.0.1:8010/api/vlm/status/ | python3 -m json.tool

# shadow, lgbm_shadow 필드 확인
curl -s http://127.0.0.1:8010/api/vlm/status/ | python3 -c \
  "import sys,json; s=json.load(sys.stdin); print('shadow:', s.get('shadow',{}).get('enabled')); print('lgbm:', s.get('lgbm_shadow',{}).get('enabled'))"

# 피드백 제출 후 JSONL 성장 확인
ls -lh media_vlm_pipeline/vlm_lgbm_shadow/*.jsonl
```

---

## 운영 명령어

```bash
# 서비스 재시작
sudo systemctl restart hotel-cctv-vlm
sudo systemctl restart nginx

# 로그 확인
sudo journalctl -u hotel-cctv-vlm -n 100 --no-pager
sudo tail -f /var/log/gunicorn/vlm_error.log
sudo tail -f /var/log/gunicorn/vlm_access.log

# GPU 상태
nvidia-smi
watch -n 2 nvidia-smi   # 실시간 모니터링

# 디스크 여유 공간
df -h /

# pip 캐시 정리 (디스크 부족 시)
sudo rm -rf /root/.cache/pip

# SSL 인증서 수동 갱신
sudo certbot renew

# Florence VRAM 점유 확인
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

---

## 알려진 문제 및 해결법

| 증상 | 원인 | 해결 |
|---|---|---|
| Florence CPU 동작 (`CUDA: False`) | NVIDIA 드라이버 미설치 | 드라이버 설치(4단계) 후 `sudo systemctl restart hotel-cctv-vlm` |
| Florence CPU 동작 (드라이버는 있음) | 서비스가 드라이버 설치 전에 시작된 채로 유지 | `sudo systemctl restart hotel-cctv-vlm` 으로 재시작 |
| 서비스 `CHDIR` 오류 (`status=200/CHDIR`) | `/home/ubuntu` 권한 750 | `sudo chmod 755 /home/ubuntu` |
| torch 설치 중 디스크 꽉 참 | pip 캐시 ~2.7GB 동시 사용 | `--no-cache-dir` 플래그 사용 후 `/root/.cache/pip` 삭제 |
| `libgl1-mesa-glx` 설치 실패 | Ubuntu 24.04 패키지명 변경 | `libgl1` 사용 (스크립트에 이미 반영됨) |
| gunicorn 워커 응답 없음 | 이전 실행 프로세스 잔존 | `sudo systemctl restart hotel-cctv-vlm` |
| certbot 실패 | DNS 미전파 또는 80 포트 닫힘 | `dig +short` 확인 후 재시도 |
| 볼륨 늘렸는데 `df -h` 변화 없음 | OS 파티션 미확장 | `growpart` + `resize2fs` 실행 |
| opencv + ultralytics 의존성 경고 | ultralytics가 `opencv-python` 명시하지만 contrib로 대체 가능 | 런타임 정상 동작, 무시 가능 |
| Bad Request `/api/vlm/shadow/recent/` | ALLOWED_HOSTS에 없는 HOST 헤더 | `.env`의 `ALLOWED_HOSTS` 확인 |

---

## 버전 및 런타임 제약

```
transformers==4.46.3       # Florence 호환성
torch==2.5.1+cu121         # CUDA 12.1 기반
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
```

- `requirements_gpu.txt` → `requirements.txt` 순서 반드시 지킬 것 (역순 시 CPU torch로 덮어씀)
- gunicorn `workers=1` 필수 (메모리 안전)
- `VLM_DEVICE=auto` 권장 (GPU 있으면 자동 CUDA, 없으면 CPU 폴백)

---

## 위험 요소 및 운영 주의사항

| 위험 | 내용 |
|---|---|
| `.env` 설정 오류 | `SECRET_KEY`, `GEMINI_API_KEY` 잘못되면 기동 실패 |
| DNS/SSL 불일치 | 외부 HTTPS 차단 |
| GPU cold start | Florence 첫 로드 시 높은 지연 발생 (VRAM 로딩 시간) |
| LightGBM 범위 | collect-only. 프로덕션 라우터/critic에 live scoring 없음 |
| Docker GPU 모드 | NVIDIA Container Toolkit 호스트 설치 필요 |
| 피드백 중복 레이블 | 동일 이벤트에 main+shadow 피드백 모두 제출 시 collector 행 2개 생성 (설계상 의도됨) |

---

## LightGBM Collector 동작 방식

### 학습 행 생성 시점

1. 이벤트 판정 시점에 decision snapshot 기록
2. `POST /api/vlm/feedback/` 또는 `POST /api/vlm/shadow/feedback/` 수신 시 학습 행 확정

### 라벨 매핑

| decision | 라벨 |
|---|---|
| `accept` | 1 (양성) |
| `decline` | 0 (음성) |
| `unsure` | unlabeled (오프라인 처리) |

> 피드백 없으면 labeled 학습 행은 늘어나지 않음.

---

## 다음 단계 (권장)

1. collector JSONL → parquet/CSV 자동 export 추가
2. 학습 재현성을 위한 feature schema 버전 관리 도입
3. LightGBM live scoring 도입 전 canary 정책 수립
