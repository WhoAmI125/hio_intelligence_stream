# HiO v2 — AWS g4dn 배포 가이드

## 대상 스펙

| 항목 | 값 |
|------|-----|
| AMI | Ubuntu 24.04 LTS |
| Instance | g4dn.xlarge (4 vCPU, 16GB RAM, T4 16GB VRAM) |
| Root Volume | 30GB 이상 |
| Security Group | 22 (SSH), 80 (HTTP), 443 (HTTPS) |
| NVIDIA Driver | 570-server |
| CUDA | 12.1 (PyTorch 번들) |
| Python | 3.12 |

### VRAM 예산 (4-bit 양자화 적용)

| 단계 | VRAM |
|------|------|
| YOLO FP16 + CLIP + CUDA | ~3.55GB |
| + Qwen 4-bit NF4 (이벤트 시) | ~6.35GB |
| **여유** | **~9.65GB** |

---

## 배포 순서 (13단계)

### 1. EC2 인스턴스 생성

```
AMI: Ubuntu 24.04 LTS
Instance type: g4dn.xlarge
Root volume: 30GB gp3
Key pair: 기존 또는 새로 생성
```

### 2. Security Group 설정

| Port | Protocol | Source | 용도 |
|------|----------|--------|------|
| 22 | TCP | My IP | SSH |
| 80 | TCP | 0.0.0.0/0 | HTTP (nginx) |
| 443 | TCP | 0.0.0.0/0 | HTTPS (certbot) |

### 3. DNS A 레코드 (SSL용)

도메인 → EC2 Public IP. 예: `cctv.example.com → 3.35.xxx.xxx`

### 4. SSH 접속 + 볼륨 확장

```bash
ssh -i key.pem ubuntu@<EC2_PUBLIC_IP>

# 디스크 확장 (30GB 활성화)
sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/nvme0n1p1
df -h /  # 30GB 확인

chmod 755 /home/ubuntu
```

### 5. NVIDIA 드라이버 설치

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y nvidia-driver-570-server
sudo reboot
```

재부팅 후:
```bash
nvidia-smi  # Tesla T4 확인
```

### 6. 시스템 패키지

```bash
sudo apt install -y nginx certbot python3-certbot-nginx ffmpeg \
  python3.12-venv python3.12-dev build-essential \
  libgl1-mesa-glx libglib2.0-0
```

### 7. 프로젝트 업로드

```bash
cd /home/ubuntu

# Option A: git clone
git clone <your-repo-url> hio_v2

# Option B: scp 업로드
# scp -i key.pem -r hio_v2/ ubuntu@<IP>:/home/ubuntu/hio_v2
```

### 8. Python 가상환경 + 의존성

```bash
cd /home/ubuntu/hio_v2
python3.12 -m venv venv

# CUDA PyTorch 먼저 (순서 중요!)
venv/bin/pip install --no-cache-dir -r deploy/requirements_gpu.txt

# 나머지 의존성
venv/bin/pip install --no-cache-dir -r requirements.txt

# bitsandbytes (Qwen 4-bit)
venv/bin/pip install --no-cache-dir bitsandbytes
```

### 9. 환경 설정

```bash
cp deploy/.env.aws .env
nano .env  # GEMINI_API_KEY 입력
```

### 10. 배포 스크립트 실행

```bash
chmod +x deploy/setup_aws.sh
sudo deploy/setup_aws.sh
```

이 스크립트가 하는 일:
- `hioapp` 서비스 계정 생성
- 런타임 디렉토리 생성 (data/, db/)
- systemd 서비스 3개 설치 (hio-model, hio-frontend, hio-boot-recover)
- nginx 설정 복사 + 활성화
- 권한 설정 + 서비스 시작

### 11. 서비스 확인

```bash
sudo systemctl status hio-model hio-frontend nginx

# 로그 확인
sudo journalctl -u hio-model -f
sudo journalctl -u hio-frontend -f

# 내부 포트 확인
curl http://127.0.0.1:8000/api/status
curl http://127.0.0.1:8002/monitor
```

### 12. SSL 인증서 (Let's Encrypt)

```bash
sudo certbot --nginx -d cctv.example.com
```

### 13. 최종 확인

```bash
# 외부 접속
curl https://cctv.example.com/api/status

# GPU 상태
nvidia-smi

# 디스크 여유
df -h /
```

---

## 서비스 아키텍처

```
┌──────────────────────────────────────┐
│ Internet → nginx (cctv.example.com)  │
│           Port 80/443                │
└──────────────┬───────────────────────┘
               │ reverse proxy
        ┌──────▼──────────────┐
        │ hio-frontend :8002   │  Jinja2 UI + API proxy
        │ (2 workers)          │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │ hio-model :8000      │  YOLO + CLIP + Qwen + Gemini
        │ (1 worker, GPU)      │  SQLite DB (내장)
        └─────────────────────┘
        ※ 모든 내부 서비스 127.0.0.1만 바인딩
```

---

## 서비스 시작 순서

```
1. hio-boot-recover.service (oneshot) → hio-safe-recover.sh boot-start
2. hio-model.service (GPU 모델 로드, 카메라 자동 복원)
3. hio-frontend.service (UI + proxy)
4. nginx (외부 접속)
```

---

## 운영 명령어

```bash
# 전체 재시작
sudo deploy/hio-safe-recover.sh recover

# 상태 확인
sudo deploy/hio-safe-recover.sh status

# 모델 서버만 재시작
sudo systemctl restart hio-model

# 로그 실시간
sudo journalctl -u hio-model -f --no-pager

# GPU 모니터링
watch -n 1 nvidia-smi

# 인시던트 트래킹 (2시간)
deploy/track_disconnect_2h.sh
```

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `nvidia-smi` 안됨 | 드라이버 미설치/재부팅 필요 | `sudo apt install nvidia-driver-570-server && sudo reboot` |
| Qwen OOM | VRAM 부족 | 4-bit 양자화 확인, `MAX_VLM_FRAMES=12` |
| RTSP 연결 실패 | Security Group | EC2 → RTSP 카메라 IP 아웃바운드 허용 확인 |
| 502 Bad Gateway | hio-model 미시작 | `sudo systemctl start hio-model && journalctl -u hio-model -f` |
| certbot 실패 | DNS 미설정 | A 레코드 확인 (`dig cctv.example.com`) |
| 디스크 풀 | 클립 축적 | `LOCAL_RETENTION_DAYS=3` 줄이기 |
| torch CPU 설치됨 | requirements 순서 오류 | `requirements_gpu.txt` **먼저** 설치 |
| Permission denied | 권한 문제 | `sudo chown -R hioapp:hioapp /home/ubuntu/hio_v2` |
| 한글 파일명 Gemini 에러 | httpx ASCII | 코드에서 자동 처리됨 (임시 파일 복사) |
| Cash trigger 30초 timeout | Qwen 로딩 중 GPU 경합 | 초기 1회만 발생, 이후 정상 |
| Cash 과잉 CONFIRMED | Gemini over-confirm | Hard Gate 서버단 강제 (코드에 내장) |
| UNCERTAIN 이벤트 쌓임 | Hard Gate 2/3만 통과 | 정상 동작, 24시간 단위 일괄 검토 권장 |

---

## GitHub 연동 + 업데이트 배포

### 초기 GitHub 연동 (EC2에서)

```bash
cd /home/ubuntu/hio_v2

# git 초기화 (이미 clone한 경우 스킵)
git init
git remote add origin git@github.com:<owner>/hio-v2.git

# .gitignore 확인 (.env, db/, data/clips 등 제외됨)
cat .gitignore

# 초기 push
git add -A
git commit -m "feat: initial hio_v2 deployment"
git push -u origin main
```

### 코드 업데이트 배포

```bash
cd /home/ubuntu/hio_v2

# 코드 풀
git pull origin main

# 의존성 변경 시
venv/bin/pip install -r requirements.txt

# 서비스 재시작
sudo deploy/hio-safe-recover.sh recover
```

### 3-Tier 판정 체계

현재 시스템은 Cash 시나리오에 대해 **증거 기반 3단 판정**을 사용합니다:

```
Tier 2 (Qwen 4-bit): 증거 추출기 — 6개 슬롯 채우기
  cash_like_object / hand_to_hand_transfer / counter_context
  staff_customer_roles_clear / drawer_or_counting / non_cash_object

Tier 3 (Gemini): Hard Gate 최종 판정 — 서버단 강제
  3/3 Hard + Soft ≥ 1 → CONFIRMED (알림 발송)
  3/3 Hard + Soft = 0 → UNCERTAIN (로그만)
  2/3 Hard            → UNCERTAIN (로그만)
  0~1/3 Hard          → FALSE_ALARM
  non-cash 명확       → FALSE_ALARM

Fire/Violence: CONFIRMED / FALSE_ALARM (단순, safety first)
```
