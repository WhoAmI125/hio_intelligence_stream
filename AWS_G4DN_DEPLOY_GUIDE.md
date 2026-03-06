# AWS g4dn 배포 가이드

Last Updated: 2026-03-06
Target: Ubuntu 24.04 LTS, g4dn.xlarge (Tesla T4), `/home/ubuntu/hio_intelligence_stream`

---

## 현재 배포 아키텍처

### 3-서버 FastAPI 구조

```
인터넷
  │
nginx (80/443)
  │
  └── frontend_server (:8002, uvicorn)
        ├── Jinja2 UI (/monitor/adhoc, /monitor/shadow, /monitor/validation-logs)
        ├── /api/vlm/* → proxy → model_server (:8000)
        └── /api/*     → proxy → db_server (:8001)

systemd services (모두 127.0.0.1에만 바인딩):
  ├── vlm-model.service    → model_server:8000 (Florence GPU, Tier-2, 추론)
  ├── vlm-db.service       → db_server:8001    (SQLite 이벤트 저장)
  └── vlm-frontend.service → frontend_server:8002 (UI + reverse proxy)
```

### 인스턴스 사양

| 항목 | 값 |
|---|---|
| 인스턴스 | g4dn.xlarge (4 vCPU, 16GB RAM) |
| GPU | Tesla T4 15GB VRAM |
| 루트 볼륨 | **30GB 이상** 필수 |
| Security Group | 22 (SSH), 80 (HTTP), 443 (HTTPS) |

> 8000/8001/8002 포트는 외부에 열 필요 없음. nginx → frontend 한 곳만 통과.

---

## 배포 절차

### 전체 순서 요약

```
1.  EC2 생성 (Ubuntu 24.04, g4dn.xlarge, 30GB+)
2.  Security Group: 22, 80, 443 오픈
3.  DNS A 레코드 → EC2 Public IP
4.  볼륨 확장: growpart + resize2fs
5.  sudo chmod 755 /home/ubuntu
6.  NVIDIA 드라이버 설치 (★ 반드시 선행)
7.  프로젝트 클론 + venv 생성
8.  CUDA torch 선설치: pip install --no-cache-dir -r requirements_gpu.txt
9.  sudo bash deploy/setup_aws_g4dn.sh
10. .env 편집 (GEMINI_API_KEY 설정)
11. 서비스 재시작 + GPU 동작 확인
12. SSL 인증서 발급 (certbot)
13. 최종 확인
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

> 포트 8000/8001/8002는 열지 않음. 내부 바인딩(127.0.0.1)만 사용.

**DNS 설정**
도메인 A 레코드를 EC2 Public IPv4로 설정. certbot 실행 전에 반드시 완료.

```bash
dig +short dev-cctv.hio.ai.kr
# → EC2 Public IP가 나와야 함
```

---

### 2단계: 볼륨 확장

AWS 콘솔에서 볼륨 크기 변경 후 OS에서도 반드시 확장.

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

```bash
sudo chmod 755 /home/ubuntu
```

---

### 4단계: NVIDIA 드라이버 설치 ★

> **이 단계를 누락하면 Florence가 CPU로 동작한다.**

```bash
# 커널 헤더
sudo apt-get install -y linux-headers-$(uname -r)

# NVIDIA 서버 드라이버 설치 (Tesla T4 권장)
sudo apt-get install -y nvidia-driver-570-server

# 커널 모듈 로드
sudo modprobe nvidia

# 확인
nvidia-smi
# Tesla T4 / Driver 570.x.x / CUDA 12.x 가 표시되어야 함
```

**DKMS 상태 확인**

```bash
dkms status
# nvidia-srv/570.x.x, 6.14.x-aws: installed ← 있어야 함
```

---

### 5단계: 프로젝트 클론 + CUDA torch 선설치

```bash
cd /home/ubuntu

# Git 클론 (또는 scp 업로드)
git clone https://github.com/YOUR_REPO/hio_intelligence_stream.git
cd hio_intelligence_stream

# venv 생성
python3 -m venv venv

# CUDA torch 먼저 설치 (반드시 --no-cache-dir!)
sudo venv/bin/pip install --no-cache-dir -r requirements_gpu.txt

# pip 캐시 정리
sudo rm -rf /root/.cache/pip
df -h /   # 여유 공간 확인 (최소 3GB 이상 필요)
```

> `requirements_gpu.txt` → `requirements.txt` 순서 **반드시** 지킬 것.
> 역순 시 CPU torch가 CUDA 버전을 덮어씀.

---

### 6단계: setup 스크립트 실행

```bash
cd /home/ubuntu/hio_intelligence_stream
sudo bash deploy/setup_aws_g4dn.sh
```

> 저디스크 안전 모드 기본값: `SKIP_MODEL_PRELOAD=1` (Florence 사전 다운로드 스킵).
> 사전 다운로드가 필요하면 `sudo SKIP_MODEL_PRELOAD=0 bash deploy/setup_aws_g4dn.sh` 사용.

스크립트가 하는 일:
1. 시스템 패키지 설치 (nginx, ffmpeg, certbot 등)
2. `vlmapp` 서비스 계정 생성
3. 남은 Python 의존성 설치 (`requirements.txt`)
4. Florence-2 모델 캐시 다운로드 (저디스크 모드에서는 기본 스킵)
5. 런타임 디렉토리 생성 (data/, logs/, clips/ 등)
6. `.env.aws` → `.env` 복사 (없을 때만)
7. 권한 설정
8. nginx 설정 + systemd 서비스 3개 등록 및 시작

---

### 7단계: 환경 변수 설정

```bash
nano /home/ubuntu/hio_intelligence_stream/.env
# GEMINI_API_KEY=실제_API_키_입력
```

설정 후 서비스 재시작:

```bash
sudo systemctl restart vlm-model vlm-db vlm-frontend
```

---

### 8단계: GPU 동작 확인

```bash
# GPU VRAM 확인 (Florence 로드 시 ~1GB 점유)
nvidia-smi

# Python에서 직접 확인
source /home/ubuntu/hio_intelligence_stream/venv/bin/activate
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '/', torch.cuda.get_device_name(0))"
# CUDA: True / Tesla T4 가 나와야 함
```

---

### 9단계: 서비스 동작 확인

```bash
# 서비스 상태
sudo systemctl status vlm-model --no-pager
sudo systemctl status vlm-db --no-pager
sudo systemctl status vlm-frontend --no-pager
sudo systemctl status nginx --no-pager

# API 응답 확인 (각 서버 직접)
curl -s http://127.0.0.1:8000/         # model_server
curl -s http://127.0.0.1:8001/         # db_server
curl -s http://127.0.0.1:8002/         # frontend (redirect)

# nginx 경유 확인
curl -s http://localhost/
```

---

### 10단계: SSL 인증서 발급

DNS가 이 서버의 IP를 가리키고, 포트 80이 열려있는 상태에서 실행.

```bash
dig +short dev-cctv.hio.ai.kr   # IP가 맞아야 함

sudo certbot --nginx -d dev-cctv.hio.ai.kr --non-interactive --agree-tos -m your@email.com
```

최종 확인:

```bash
curl -s https://dev-cctv.hio.ai.kr/monitor/adhoc
```

---

## 환경 변수

`.env.aws` 기본값 기준. 전체는 `.env.aws` 파일 참고.

### 필수

| 변수 | 설명 |
|---|---|
| `GEMINI_API_KEY` | Gemini API 키 (Tier-2 검증) |
| `FLORENCE_DEVICE` | `cuda` 권장 (GPU 있으면 자동 선택) |

### 서버 포트/URL (기본값으로 충분)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MODEL_SERVER_PORT` | `8000` | model_server 포트 |
| `DB_SERVER_PORT` | `8001` | db_server 포트 |
| `FRONTEND_SERVER_PORT` | `8002` | frontend_server 포트 |
| `MODEL_SERVER_URL` | `http://127.0.0.1:8000` | 내부 통신용 |
| `DB_SERVER_URL` | `http://127.0.0.1:8001` | 내부 통신용 |

### 모델

| 변수 | 기본값 |
|---|---|
| `FLORENCE_MODEL` | `microsoft/Florence-2-large` |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` |

### 저장소

| 변수 | 기본값 |
|---|---|
| `DB_PATH` | `data/cctv_events.db` |
| `MODEL_SERVER_DATA_DIR` | `data` |
| `USE_S3` | `false` |
| `FFMPEG_PATH` | `ffmpeg` |

---

## 5분 진단 커맨드 세트

```bash
# 1) DNS/포트
dig +short dev-cctv.hio.ai.kr
nc -zv <PUBLIC_IP> 22
nc -zv <PUBLIC_IP> 80
nc -zv <PUBLIC_IP> 443

# 2) 서비스 상태
sudo systemctl status vlm-model --no-pager
sudo systemctl status vlm-db --no-pager
sudo systemctl status vlm-frontend --no-pager
sudo systemctl status nginx --no-pager
ss -lntp | grep -E ':8000|:8001|:8002|:80|:443'

# 3) API 상태
curl -s http://127.0.0.1:8000/ | jq .
curl -s http://127.0.0.1:8001/ | jq .
curl -s http://127.0.0.1:8002/api/vlm/status/ | jq .

# 4) GPU
nvidia-smi
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# 5) 최근 오류 로그
sudo journalctl -u vlm-model -n 100 --no-pager
sudo journalctl -u vlm-db -n 50 --no-pager
sudo journalctl -u vlm-frontend -n 50 --no-pager
```

---

## 운영 명령어

```bash
# 서비스 재시작 (전체)
sudo systemctl restart vlm-model vlm-db vlm-frontend

# 개별 재시작
sudo systemctl restart vlm-model

# 로그 실시간 추적
sudo journalctl -u vlm-model -f
sudo journalctl -u vlm-db -f
sudo journalctl -u vlm-frontend -f

# nginx 재시작
sudo systemctl restart nginx

# GPU 실시간 모니터링
watch -n 2 nvidia-smi

# 디스크 여유 공간
df -h /

# pip 캐시 정리 (디스크 부족 시)
sudo rm -rf /root/.cache/pip

# SSL 인증서 수동 갱신
sudo certbot renew
```

---

## 실전 오류 요약 (빠른 해결용)

이전 배포 + 현재 아키텍처 기반 오류 정리.

| 증상 | 원인 | 즉시 해결 |
|---|---|---|
| Florence CPU 동작 (`CUDA: False`) | NVIDIA 드라이버 미설치 | 드라이버 설치(4단계) 후 `sudo systemctl restart vlm-model` |
| Florence CPU 동작 (드라이버는 있음) | 서비스가 드라이버 설치 전에 시작됨 | `sudo systemctl restart vlm-model` |
| 서비스 `CHDIR` 오류 | `/home/ubuntu` 권한 750 | `sudo chmod 755 /home/ubuntu` |
| torch 설치 중 디스크 꽉 참 | pip 캐시 ~2.7GB | `--no-cache-dir` + `rm -rf /root/.cache/pip` |
| `libgl1-mesa-glx` 설치 실패 | Ubuntu 24.04 패키지명 변경 | `libgl1` 사용 (스크립트에 이미 반영) |
| `python3 -m venv` 실패 | `python3-venv` 누락 | `sudo apt-get install -y python3-venv` |
| `ImportError: libGL.so.1` | OpenCV 라이브러리 누락 | `sudo apt install -y libgl1 libglib2.0-0` |
| certbot 실패 | DNS 미전파 또는 80 포트 닫힘 | `dig +short` 확인 후 재시도 |
| 볼륨 늘렸는데 `df -h` 변화 없음 | OS 파티션 미확장 | `growpart` + `resize2fs` |
| curl 127.0.0.1:8000 connection refused | vlm-model 서비스 안 뜸 | `journalctl -u vlm-model -n 100` 확인 |
| nginx 502 Bad Gateway | frontend_server 미시작 | `sudo systemctl start vlm-frontend` |

---

## 파일 구조 (deploy/)

```
hio_intelligence_stream/
├── deploy/
│   ├── setup_aws_g4dn.sh       # 원스탑 배포 스크립트
│   ├── nginx.conf               # nginx 리버스 프록시
│   ├── vlm-model.service        # systemd: model_server (:8000)
│   ├── vlm-db.service           # systemd: db_server (:8001)
│   └── vlm-frontend.service     # systemd: frontend_server (:8002)
├── requirements.txt             # Python 의존성 (CPU 호환)
├── requirements_gpu.txt         # CUDA torch 고정 (★ 먼저 설치)
├── .env.aws                     # AWS 전용 환경 변수 템플릿
├── .env                         # 실사용 환경 변수 (gitignore)
├── model_server/                # Tier-1/2 추론 서버
├── db_server/                   # 이벤트 DB 서버
└── frontend_server/             # UI + reverse proxy 서버
```

---

## 버전 및 런타임 제약

```
transformers==4.46.3       # Florence 호환성
torch==2.5.1+cu121         # CUDA 12.1 기반
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
```

- `requirements_gpu.txt` → `requirements.txt` 순서 반드시 지킬 것
- model_server `workers=1` 필수 (GPU VRAM 공유 불가)
- frontend_server `workers=2` 권장 (UI + proxy 동시 처리)
- db_server `workers=1` 충분 (SQLite 단일 writer)

---

## 이전 배포(Django) 대비 변경 사항

| 항목 | 이전 (Django) | 현재 (FastAPI) |
|---|---|---|
| 프레임워크 | Django + gunicorn | FastAPI + uvicorn |
| 서비스 파일 | `hotel-cctv-vlm.service` 1개 | `vlm-model/db/frontend.service` 3개 |
| 포트 | gunicorn → 8010 | uvicorn → 8000, 8001, 8002 |
| nginx upstream | `127.0.0.1:8010` | `127.0.0.1:8002` |
| DB migrate | `python manage_vlm.py migrate` | 자동 (SQLite init on startup) |
| collectstatic | 필요 | 불필요 (Jinja2 직접 서빙) |
| ALLOWED_HOSTS | `.env`에 필수 설정 | 불필요 (CORS middleware) |
| SECRET_KEY | 필수 | 불필요 |
| 외부 포트 | 22, 80, 443 | 22, 80, 443 (동일) |
