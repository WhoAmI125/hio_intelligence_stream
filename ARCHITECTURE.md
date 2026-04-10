# HiO Intelligence Stream v2 — Architecture Design

> **3-Tier CCTV 이상 탐지 시스템**
> YOLO26-Pose (현금거래 감지) + CLIP (화재/폭력 감지) → Qwen2.5-VL-3B (12초 영상 분석) → Gemini 2.5 Flash (최종 판정)

---

## 설계 철학

> "Simple is best. Sometimes throw away and reconstruct again from start, like a Falcon 1, 2, 4 rocket."

v1(Florence-2 캡션 → 키워드 매칭)의 근본적 실패 원인은 **간접 판단 구조**였다.
Florence-2가 생성한 캡션에 "cash"라는 단어가 등장하지 않아 실제 현금 거래 4건 중 0건을 탐지(0%)했다.

v2는 이 교훈에서 출발한다:

- **감지(Detection)와 이해(Understanding)를 분리한다.** 감지는 빠르고 단순한 CV 모델이, 이해는 VLM이 담당한다.
- **시나리오별 최적 감지 방식을 사용한다.** 현금거래는 사람 간 물리적 근접성, 화재/폭력은 시각 패턴.
- **영상(Video)을 분석한다.** 단일 프레임이 아닌 12초 클립을 VLM에 넘겨 시간적 맥락을 이해시킨다.
- **3단 Cascade로 비용을 제어한다.** 초경량 Tier 1 → 로컬 VLM Tier 2 → 클라우드 API Tier 3.

---

## 전체 아키텍처

```text
3 RTSP Cameras (각 호텔, 1080p, 30fps)
│
├──→ Encoded Packet Ring Buffer × 3 (항상 60초 보관, 각 ~30MB)
│    H.264 패킷 그대로 저장 (디코딩 안 함)
│
└──→ 1 FPS 프레임 샘플링 (초당 3프레임)
      │
      ▼
┌─────────────────────────────────────────────────────┐
│          Tier 1: Task-Specific Triggers              │
│          GPU ~2.5GB VRAM, ~15ms/프레임               │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │ Cash Trigger: YOLO26s-pose (~4ms)           │    │
│  │                                              │    │
│  │ 1. 프레임에서 사람 skeleton 추출 (17 keypoints) │  │
│  │ 2. 2명 이상 감지?                            │    │
│  │ 3. 카운터 ROI 영역 내?                       │    │
│  │ 4. 양측 손목(wrist) 거리 < 임계값?           │    │
│  │    → YES: Cash 의심 프레임 플래그             │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │ Fire/Violence Trigger: CLIP ViT-L/14 (~12ms)│    │
│  │                                              │    │
│  │ text prompts:                                │    │
│  │   fire: ["fire burning", "smoke in room",   │    │
│  │          "normal room", "steam or fog"]      │    │
│  │   violence: ["people fighting violently",   │    │
│  │          "physical assault",                 │    │
│  │          "normal interaction",               │    │
│  │          "people playing or exercising"]     │    │
│  │                                              │    │
│  │ fire/violence 유사도 > threshold?             │    │
│  │   → YES: 해당 시나리오 플래그                 │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  결과: 프레임별 {cash: bool, fire: bool, violence: bool} │
└──────────────────────┬──────────────────────────────┘
                       │
                       │ 플래그된 프레임만 (전체의 ~5-15%)
                       │
            ┌──────────▼──────────┐
            │  Trigger Accumulator │
            │                     │
            │  30초 윈도우 내      │
            │  동일 시나리오       │
            │  N회 이상 트리거?    │
            │  (cash: 2회,        │
            │   fire: 2회,        │
            │   violence: 2회)    │
            └──────────┬──────────┘
                       │
                       │ 누적 조건 충족
                       ▼
              Ring Buffer에서 12초 클립 추출
              (pre 6초 + post 6초, ffmpeg remux)
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│         Tier 2: Qwen2.5-VL-3B 영상 분석              │
│         GPU ~7-9GB VRAM, ~5-15초/클립                │
│                                                     │
│  12초 클립에서 1 FPS로 12프레임 추출                   │
│  → Qwen2.5-VL-3B에 video input으로 전달              │
│                                                     │
│  시나리오별 전문 에이전트 프롬프트:                     │
│                                                     │
│  ┌─ Cash Agent ──────────────────────────────┐      │
│  │ "이 12초 영상에서 현금 거래가 발생하는지 분석해. │  │
│  │  다음을 확인해:                              │      │
│  │  1) 손-대-손 물체 전달이 보이는가?           │      │
│  │  2) 지폐 형태의 물체가 보이는가?             │      │
│  │  3) 서비스 카운터에서 발생하는가?             │      │
│  │  4) 직원-고객 역할 구분이 되는가?            │      │
│  │  5) 지갑/서랍 활동이 있는가?                 │      │
│  │  JSON: {detected, confidence, reason}"       │      │
│  └───────────────────────────────────────────┘      │
│                                                     │
│  ┌─ Fire Agent ──────────────────────────────┐      │
│  │ "이 12초 영상에서 화재/연기가 발생하는지 분석해. │  │
│  │  다음을 확인해:                              │      │
│  │  1) 연기/안개가 보이는가?                    │      │
│  │  2) 오렌지/적색 화염이 보이는가?             │      │
│  │  3) 시간에 따라 확산되는가?                  │      │
│  │  4) 대피 행동이 보이는가?                    │      │
│  │  5) 구조물 손상이 보이는가?                  │      │
│  │  JSON: {detected, confidence, reason}"       │      │
│  └───────────────────────────────────────────┘      │
│                                                     │
│  ┌─ Violence Agent ──────────────────────────┐      │
│  │ "이 12초 영상에서 폭력이 발생하는지 분석해.    │  │
│  │  다음을 확인해:                              │      │
│  │  1) 공격적 신체 접촉이 보이는가?             │      │
│  │  2) 공격 자세(주먹, 발차기)가 보이는가?      │      │
│  │  3) 쓰러진 사람이 있는가?                    │      │
│  │  4) 주변인의 놀람/회피 반응이 있는가?        │      │
│  │  5) 무기가 보이는가?                        │      │
│  │  JSON: {detected, confidence, reason}"       │      │
│  └───────────────────────────────────────────┘      │
│                                                     │
│  결과 판단:                                          │
│    confidence >= 0.7  → Gemini 없이 바로 알림        │
│    confidence 0.3-0.7 → Tier 3 Gemini 검증          │
│    confidence < 0.3   → 기각                        │
└──────────────────────┬──────────────────────────────┘
                       │
                       │ 의심 클립 (하루 ~50-200건)
                       ▼
┌─────────────────────────────────────────────────────┐
│         Tier 3: Gemini 2.5 Flash (클라우드)           │
│         API 호출, ~1-2초/건                          │
│                                                     │
│  Qwen이 의심한 프레임 + Qwen의 분석 결과를 함께 전송  │
│                                                     │
│  "Qwen2.5-VL이 이 프레임을 분석한 결과:              │
│   {qwen_result}                                     │
│   이 프레임에서 {scenario}가 실제로 발생하고 있는가?  │
│   CONFIRMED / FALSE_ALARM 중 하나로 답해."           │
│                                                     │
│  → CONFIRMED: 이벤트 확정                            │
│  → FALSE_ALARM: 기각                                │
└──────────────────────┬──────────────────────────────┘
                       │
                       │ 확정 이벤트 (하루 ~10-50건)
                       ▼
┌─────────────────────────────────────────────────────┐
│                  Event Pipeline                      │
│                                                     │
│  1. 12초 클립 → S3 업로드                            │
│  2. 이벤트 메타데이터 → SQLite 저장                   │
│  3. 관리자 알림 (Webhook / Slack / LINE)              │
│  4. 쿨다운 적용 (동일 카메라+시나리오 60초)           │
└─────────────────────────────────────────────────────┘
```

---

## 모델 선정 근거

### Tier 1: YOLO26s-pose + CLIP ViT-L/14

#### YOLO26s-pose (현금거래 감지)

| 모델 | Params | mAPpose (COCO) | T4 TensorRT FP16 | 용도 |
|---|---|---|---|---|
| yolo26n-pose | ~3M | 57.2% | ~1.7ms | 가벼우나 정밀도 부족 |
| **yolo26s-pose** | ~7M | ~63% | **~2.5ms** | **정밀도-속도 최적 균형** |
| yolo26m-pose | ~16M | ~67% | ~4ms | 더 정확하지만 불필요한 여유 |

YOLO26은 2026년 1월 출시된 최신 모델로, NMS-free end-to-end 추론과 RLE(Residual Log-Likelihood Estimation) 기반 고정밀 keypoint 추정을 지원한다. Pose estimation은 17개 관절 좌표를 반환하며, 이 중 양쪽 손목(left_wrist=9, right_wrist=10) 좌표 간 거리를 계산하여 현금 전달 동작을 감지한다.

**yolo26s-pose를 선정한 이유**: n(nano)은 mAPpose 57.2%로 손목 좌표의 정밀도가 부족할 수 있고, m(medium)은 성능 대비 속도 이득이 적다. s(small)가 ~63% mAPpose에 ~2.5ms로 가장 효율적이다. VRAM은 ~1GB 수준.

#### CLIP ViT-L/14 (화재/폭력 감지)

CLIP은 이미지와 텍스트 간 유사도를 계산하는 모델로, 별도 학습 없이 자연어 프롬프트만으로 시각 패턴을 분류할 수 있다. "fire burning"이라는 텍스트와 프레임의 유사도가 높으면 화재 의심으로 분류.

- **VRAM**: ~1.5GB
- **추론 속도**: ~12ms/프레임 (T4 FP16)
- **장점**: 학습 없이 zero-shot 분류, 프롬프트 변경만으로 새 시나리오 추가 가능

**왜 fire/violence에는 CLIP이고 cash에는 YOLO-Pose인가?**

| 시나리오 | 시각적 특성 | 최적 감지 방식 |
|---|---|---|
| Cash | 2명의 손이 가까워지는 **물리적 동작** | Skeleton 기반 거리 계산 |
| Fire | 불꽃/연기라는 **시각 패턴** | CLIP text-image 유사도 |
| Violence | 싸움이라는 **시각 패턴** | CLIP text-image 유사도 |

현금거래는 CLIP으로 감지하기 어렵다. "person handing cash"라는 텍스트와 실제 호텔 CCTV 프레임의 유사도가 낮기 때문. 반면 2명의 skeleton에서 손목 거리를 계산하는 것은 명확한 물리적 신호.

### Tier 2: Qwen2.5-VL-3B (영상 에이전트)

Qwen2.5-VL-3B는 **동영상 입력을 네이티브로 지원**하는 3B 파라미터 VLM이다. Dynamic FPS Sampling과 mRoPE(Multimodal Rotary Position Embedding)를 통해 시간 축 정보를 이해하며, 영상 내 특정 시점의 이벤트를 localize할 수 있다.

**12초 클립 분석 방식:**

```python
# 12초 클립에서 1 FPS로 12프레임 추출
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": [
                    "frame_t0.jpg",   # t=0초
                    "frame_t1.jpg",   # t=1초
                    ...
                    "frame_t11.jpg",  # t=11초
                ],
                "fps": 1.0,
            },
            {
                "type": "text",
                "text": CASH_AGENT_PROMPT  # 시나리오별 전문가 프롬프트
            },
        ],
    }
]
```

**왜 단일 프레임이 아닌 12초 영상인가?**

- 현금거래는 시간적 시퀀스다: "접근 → 손 뻗기 → 전달 → 확인 → 분리"
- 단일 프레임에서는 "손을 뻗고 있는 사람"만 보이지만, 12프레임 시퀀스에서는 "물체를 건네는 행위"가 보인다
- Qwen2.5-VL의 temporal reasoning이 이 시퀀스를 이해한다

**왜 Tier 1에서 바로 Gemini가 아닌가?**

Tier 1 트리거가 하루 ~3,000-5,000회 발생할 수 있다. 이를 전부 Gemini로 보내면 월 $30-50. Qwen이 로컬에서 90%를 걸러주면 Gemini 호출이 ~50-200회로 줄어 월 $0.5-2 수준으로 관리 가능하다. 또한 Qwen의 영상 분석 결과(reason)를 Gemini에 context로 함께 보내면 Gemini의 판단 정확도도 높아진다.

### Tier 3: Gemini 2.5 Flash

- **비용**: $0.30/1M input tokens, $2.50/1M output tokens
- **지연 시간**: 0.5-1.2초 (time-to-first-token P50 기준)
- **역할**: Qwen이 confidence 0.3-0.7으로 판단한 애매한 케이스에 대해 2차 의견 제공
- **Structured Output**: `response_mime_type: "text/x.enum"`으로 `CONFIRMED` / `FALSE_ALARM` 강제

---

## VRAM 및 리소스 예산

### GPU (T4 16GB 기준)

| 컴포넌트 | VRAM | 동시 로드 |
|---|---|---|
| YOLO26s-pose (TensorRT FP16) | ~1.0GB | ✅ 상시 |
| CLIP ViT-L/14 (FP16) | ~1.5GB | ✅ 상시 |
| Qwen2.5-VL-3B (FP16) | ~7-9GB | ✅ 상시 |
| CUDA context + overhead | ~1.5GB | ✅ 상시 |
| **합계** | **~11-13GB** | **여유 3-5GB** |

### GPU 업그레이드 시 (L4 24GB, g6.xlarge)

L4는 BF16과 FlashAttention 2를 지원하여 Qwen2.5-VL-3B 추론이 T4 대비 ~2-3배 빨라진다. g6.xlarge Spot 가격은 서울 리전 기준 ~$0.25-0.35/hr, 월 ~$180-250 (25-35만원).

| 항목 | T4 (g4dn.xlarge) | L4 (g6.xlarge) |
|---|---|---|
| BF16 | ❌ | ✅ |
| FlashAttention 2 | ❌ | ✅ |
| VRAM | 16GB | 24GB |
| Qwen 추론 속도 | ~5-15초/클립 | ~2-5초/클립 |
| Spot 가격 | ~$0.16-0.20/hr | ~$0.25-0.35/hr |
| 월 비용 | ~15-20만원 | ~25-35만원 |

**T4로도 작동 가능하다.** Tier 2(Qwen)는 Tier 1 트리거가 발생할 때만 호출되므로, 클립 분석에 10-15초 걸려도 실시간성에 영향 없다. 다만 L4를 사용하면 응답 속도가 빨라져 이벤트 확인까지의 총 지연시간이 줄어든다.

### CPU (4 vCPU)

| 작업 | CPU 사용률 |
|---|---|
| RTSP 디코딩 (3 스트림, 1 FPS) | ~3-5% |
| Ring Buffer 관리 | ~1% |
| Motion pre-check (optional) | ~1% |
| 이벤트 후처리 + ffmpeg remux | ~2% (간헐적) |
| FastAPI 서버 | ~5% |
| **합계** | **~12-14%** |

### RAM (16GB)

| 항목 | 사용량 |
|---|---|
| Qwen2.5-VL-3B (시스템 메모리) | ~4GB |
| Ring Buffer (3 × 60초) | ~90MB |
| 프레임 버퍼 + OpenCV | ~500MB |
| Python + FastAPI + 기타 | ~1GB |
| OS | ~1.5GB |
| **합계** | **~7GB (여유 9GB)** |

---

## Tier 1 상세: Cash Trigger (YOLO26s-pose)

### 감지 로직

```python
from ultralytics import YOLO
import numpy as np

class CashTrigger:
    def __init__(self, cam_config):
        self.pose_model = YOLO("yolo26s-pose.pt")
        self.counter_roi = cam_config["counter_roi"]     # [x1, y1, x2, y2]
        self.counter_line_y = cam_config["counter_y"]     # 직원/손님 구분선
        self.wrist_threshold = cam_config.get("wrist_px", 80)  # 픽셀 거리
    
    def detect(self, frame) -> dict:
        results = self.pose_model(frame, verbose=False)
        keypoints = results[0].keypoints
        
        if keypoints is None or len(keypoints) < 2:
            return {"triggered": False}
        
        # ROI 내 인물만 필터링
        persons_in_roi = []
        for i, kp in enumerate(keypoints.xy):
            hip_center = (kp[11] + kp[12]) / 2  # left_hip + right_hip
            if self._in_roi(hip_center):
                persons_in_roi.append({
                    "index": i,
                    "side": "staff" if hip_center[1] < self.counter_line_y else "guest",
                    "left_wrist": kp[9],
                    "right_wrist": kp[10],
                })
        
        # 직원-손님 쌍의 손목 거리 확인
        staff = [p for p in persons_in_roi if p["side"] == "staff"]
        guests = [p for p in persons_in_roi if p["side"] == "guest"]
        
        for s in staff:
            for g in guests:
                min_dist = self._min_wrist_distance(s, g)
                if min_dist < self.wrist_threshold:
                    return {
                        "triggered": True,
                        "wrist_distance": float(min_dist),
                        "staff_count": len(staff),
                        "guest_count": len(guests),
                    }
        
        return {"triggered": False}
    
    def _min_wrist_distance(self, person_a, person_b):
        dists = []
        for wa in [person_a["left_wrist"], person_a["right_wrist"]]:
            for wb in [person_b["left_wrist"], person_b["right_wrist"]]:
                if wa.sum() > 0 and wb.sum() > 0:  # keypoint 감지됨
                    dists.append(np.linalg.norm(wa - wb))
        return min(dists) if dists else float("inf")
    
    def _in_roi(self, point):
        x, y = point
        x1, y1, x2, y2 = self.counter_roi
        return x1 <= x <= x2 and y1 <= y <= y2
```

### 카메라별 설정 파일

```yaml
# configs/cameras.yaml
cameras:
  - id: "ilsan_hotel"
    rtsp_url: "rtsp://..."
    counter_roi: [100, 200, 500, 450]   # 카운터 영역 (x1, y1, x2, y2)
    counter_y: 320                       # 이 Y좌표 위=직원, 아래=손님
    wrist_px: 80                         # 손목 근접 판단 임계값 (픽셀)
    
  - id: "geumchon_hotel"
    rtsp_url: "rtsp://..."
    counter_roi: [150, 180, 550, 420]
    counter_y: 300
    wrist_px: 90

  - id: "paju_hotel"
    rtsp_url: "rtsp://..."
    counter_roi: [120, 210, 480, 440]
    counter_y: 310
    wrist_px: 75
```

### YOLO26s vs YOLO26m 선택 기준

| 상황 | 추천 모델 |
|---|---|
| 카메라가 카운터와 가까움 (사람 크게 보임) | yolo26s-pose (충분) |
| 카메라가 멀리 설치됨 (사람 작게 보임) | yolo26m-pose (더 정확한 keypoint) |
| 향후 카메라 10대 이상 확장 예정 | yolo26n-pose (속도 우선) |

---

## Tier 1 상세: Fire/Violence Trigger (CLIP)

### 감지 로직

```python
import torch
import clip
from PIL import Image

class CLIPTrigger:
    def __init__(self):
        self.model, self.preprocess = clip.load("ViT-L/14", device="cuda")
        
        # 시나리오별 텍스트 프롬프트 (사전 인코딩)
        self.prompts = {
            "fire": {
                "positive": ["fire burning in a room", "smoke rising indoors"],
                "negative": ["normal room", "steam from cooking", "fog or mist"],
            },
            "violence": {
                "positive": ["people fighting violently", "physical assault or attack"],
                "negative": ["normal interaction", "people playing", "friendly hug"],
            },
        }
        self.text_features = self._encode_texts()
    
    def _encode_texts(self):
        features = {}
        for scenario, prompts in self.prompts.items():
            all_texts = prompts["positive"] + prompts["negative"]
            tokens = clip.tokenize(all_texts).to("cuda")
            with torch.no_grad():
                features[scenario] = {
                    "features": self.model.encode_text(tokens),
                    "n_positive": len(prompts["positive"]),
                }
        return features
    
    def detect(self, frame_rgb) -> dict:
        image = self.preprocess(Image.fromarray(frame_rgb)).unsqueeze(0).to("cuda")
        
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        
        results = {}
        for scenario, data in self.text_features.items():
            similarity = (image_features @ data["features"].T).softmax(dim=-1)[0]
            positive_score = similarity[:data["n_positive"]].max().item()
            results[scenario] = {
                "triggered": positive_score > self._threshold(scenario),
                "score": positive_score,
            }
        
        return results
    
    def _threshold(self, scenario):
        # 낮은 threshold = 높은 recall (놓치지 않기 우선)
        return {"fire": 0.25, "violence": 0.30}[scenario]
```

---

## Trigger Accumulator: 단발성 오탐 방지

Tier 1의 단일 트리거만으로 Tier 2를 호출하면 오탐이 많아진다. 예: 손님이 서류를 건네는 순간 손목이 가까워져 cash 트리거 발생. 이를 방지하기 위해 **시간 윈도우 내 반복 트리거**를 요구한다.

```python
from collections import defaultdict
import time

class TriggerAccumulator:
    def __init__(self):
        self.triggers = defaultdict(list)  # key: (cam_id, scenario)
        self.thresholds = {
            "cash": {"count": 2, "window": 30},      # 30초 내 2회
            "fire": {"count": 2, "window": 15},       # 15초 내 2회
            "violence": {"count": 2, "window": 10},   # 10초 내 2회
        }
        self.cooldowns = {}  # 동일 이벤트 재트리거 방지
    
    def add(self, cam_id, scenario, timestamp) -> bool:
        """트리거 추가. 누적 조건 충족 시 True 반환."""
        key = (cam_id, scenario)
        
        # 쿨다운 확인
        if key in self.cooldowns and timestamp - self.cooldowns[key] < 60:
            return False
        
        # 오래된 트리거 제거
        window = self.thresholds[scenario]["window"]
        self.triggers[key] = [
            t for t in self.triggers[key] 
            if timestamp - t < window
        ]
        
        # 새 트리거 추가
        self.triggers[key].append(timestamp)
        
        # 누적 조건 확인
        if len(self.triggers[key]) >= self.thresholds[scenario]["count"]:
            self.cooldowns[key] = timestamp
            self.triggers[key] = []
            return True  # → Tier 2 호출
        
        return False
```

---

## Tier 2 상세: Qwen2.5-VL-3B 영상 에이전트

### 에이전트 프롬프트 설계

```python
AGENT_PROMPTS = {
    "cash": """당신은 호텔 CCTV 현금 거래 탐지 전문가입니다.
이 12초 영상을 분석하여 현금 거래 발생 여부를 판단하세요.

다음 5가지 질문에 대해 각각 yes/no로 판단하세요:
1. 한 사람이 다른 사람에게 물체를 손에서 손으로 전달하는 장면이 보이는가?
2. 전달되는 물체가 지폐 또는 지폐 형태의 평면 사각형으로 보이는가? 불확실하면 yes.
3. 이 상호작용이 서비스 카운터/데스크 근처에서 발생하는가?
4. 참여자 간 직원-고객 역할 구분이 가능한가?
5. 지갑에서 꺼내기, 지폐 세기, 서랍 열기 등의 행위가 보이는가?

위 질문 중 2개 이상 yes이면 detected=true로 판단하세요.
불확실하지만 가능성이 있으면 yes쪽으로 판단하세요 (놓친 거래의 비용 > 오탐 비용).

반드시 아래 JSON 형식으로만 응답하세요:
{"detected": bool, "confidence": 0.0-1.0, "yes_count": int, "reason": "한줄 설명"}""",

    "fire": """당신은 호텔 CCTV 화재/연기 탐지 전문가입니다.
이 12초 영상을 분석하여 화재 또는 연기 발생 여부를 판단하세요.

다음 5가지 질문에 대해 각각 yes/no로 판단하세요:
1. 떠다니는 연기 또는 안개 같은 물질이 보이는가?
2. 오렌지/적색 화염 또는 비정상적 발광이 보이는가?
3. 시간에 따라 연기/화염이 확산 또는 강화되는가?
4. 사람들의 대피/긴급 행동이 보이는가?
5. 열에 의한 구조물 변색이나 변형이 보이는가?

위 질문 중 2개 이상 yes이면 detected=true로 판단하세요.
안전 최우선: 불확실하면 yes로 판단하세요.

False Positive 주의: 주방 수증기, 조명 반사, 석양빛은 화재가 아닙니다.

반드시 아래 JSON 형식으로만 응답하세요:
{"detected": bool, "confidence": 0.0-1.0, "yes_count": int, "reason": "한줄 설명"}""",

    "violence": """당신은 호텔 CCTV 폭력 탐지 전문가입니다.
이 12초 영상을 분석하여 폭력 발생 여부를 판단하세요.

다음 5가지 질문에 대해 각각 yes/no로 판단하세요:
1. 한 사람이 다른 사람을 밀거나 때리는 공격적 접촉이 보이는가?
2. 주먹을 쥐거나 발을 올리는 공격 자세가 보이는가?
3. 바닥에 쓰러진 사람이 있는가?
4. 주변 사람들이 놀라거나 물러서는 반응이 보이는가?
5. 무기로 사용될 수 있는 물체가 보이는가?

위 질문 중 2개 이상 yes이면 detected=true로 판단하세요.

False Positive 주의: 악수, 포옹, 장난, 운동은 폭력이 아닙니다.
의도(intent)와 반응(reaction)을 함께 확인하세요.

반드시 아래 JSON 형식으로만 응답하세요:
{"detected": bool, "confidence": 0.0-1.0, "yes_count": int, "reason": "한줄 설명"}""",
}
```

### 영상 분석 코드

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import json, cv2

class VideoAnalyzer:
    def __init__(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype="auto",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct"
        )
    
    def analyze_clip(self, clip_frames: list, scenario: str) -> dict:
        """
        clip_frames: 12초 클립에서 1 FPS로 추출한 12장의 RGB numpy 배열
        scenario: "cash" | "fire" | "violence"
        """
        # 프레임을 임시 파일로 저장 (Qwen video input 형식)
        frame_paths = []
        for i, frame in enumerate(clip_frames):
            path = f"/tmp/clip_frame_{i:03d}.jpg"
            cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                       [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_paths.append(f"file://{path}")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frame_paths,
                        "fps": 1.0,
                    },
                    {
                        "type": "text",
                        "text": AGENT_PROMPTS[scenario],
                    },
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], videos=[clip_frames],
            padding=True, return_tensors="pt"
        ).to(self.model.device)
        
        output_ids = self.model.generate(**inputs, max_new_tokens=128)
        response = self.processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]
        
        return self._parse_json(response)
    
    def _parse_json(self, text):
        try:
            # JSON 블록 추출
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return {"detected": False, "confidence": 0.0, "reason": "parse_error"}
```

---

## 클립 추출: Encoded Packet Ring Buffer

### 원리

RTSP 스트림의 H.264 인코딩된 패킷을 **디코딩하지 않고 그대로** 순환 버퍼에 저장한다. 탐지 시 해당 구간의 패킷을 꺼내 MP4 컨테이너로 감싸기만 하면(remux) 12초 클립이 완성된다.

| 방식 | 메모리 (60초, 1080p) | 클립 생성 시간 |
|---|---|---|
| 디코딩된 프레임 저장 | ~11GB (30fps×RGB) | 수초 (재인코딩) |
| **인코딩 패킷 저장** | **~30MB** (4Mbps) | **수십ms** (remux) |

### 구현

```python
import av
import collections
import threading
import time

class StreamReader:
    def __init__(self, rtsp_url, buffer_seconds=60):
        self.url = rtsp_url
        self.buffer_seconds = buffer_seconds
        self.packet_buffer = collections.deque()
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
    
    def _read_loop(self):
        while self.running:
            try:
                container = av.open(self.url, options={
                    "rtsp_transport": "tcp",
                    "stimeout": "5000000",
                })
                stream = container.streams.video[0]
                
                for packet in container.demux(stream):
                    if not self.running:
                        break
                    
                    # 인코딩된 패킷 저장
                    with self.lock:
                        self.packet_buffer.append({
                            "data": bytes(packet),
                            "pts": packet.pts,
                            "dts": packet.dts,
                            "is_keyframe": packet.is_keyframe,
                            "time": time.monotonic(),
                            "time_base": stream.time_base,
                        })
                        
                        # 오래된 패킷 제거
                        cutoff = time.monotonic() - self.buffer_seconds
                        while (self.packet_buffer and 
                               self.packet_buffer[0]["time"] < cutoff):
                            self.packet_buffer.popleft()
                    
                    # 1 FPS 프레임 디코딩 (분석용)
                    # 별도 로직으로 매초 1프레임만 디코딩
                    
                container.close()
            except Exception as e:
                print(f"RTSP reconnecting: {e}")
                time.sleep(5)
    
    def extract_clip(self, trigger_time, pre_sec=6, post_sec=6) -> str:
        """Ring buffer에서 12초 클립 추출하여 MP4 파일로 저장."""
        time.sleep(post_sec)  # post 시간 대기
        
        start_time = trigger_time - pre_sec
        end_time = trigger_time + post_sec
        
        with self.lock:
            packets = [
                p for p in self.packet_buffer
                if start_time <= p["time"] <= end_time
            ]
        
        if not packets:
            return None
        
        # 가장 가까운 keyframe부터 시작
        first_keyframe = None
        for i, p in enumerate(packets):
            if p["is_keyframe"]:
                first_keyframe = i
                break
        
        if first_keyframe is not None:
            packets = packets[first_keyframe:]
        
        # ffmpeg remux (재인코딩 없음)
        output_path = f"/tmp/clip_{int(trigger_time)}.mp4"
        # ... PyAV output container로 패킷 복사
        
        return output_path
```

---

## 파일 구조

```
hio_v2/
├── main.py                        # FastAPI 앱, 전체 라이프사이클
├── config.py                      # 환경변수 + cameras.yaml 로딩
│
├── tier1/
│   ├── cash_trigger.py            # YOLO26s-pose 기반 현금거래 감지
│   ├── clip_trigger.py            # CLIP ViT-L/14 기반 화재/폭력 감지
│   └── trigger_accumulator.py     # 시간 윈도우 누적 트리거
│
├── tier2/
│   ├── video_analyzer.py          # Qwen2.5-VL-3B 영상 분석
│   └── agent_prompts.py           # 시나리오별 전문가 프롬프트
│
├── tier3/
│   └── gemini_verifier.py         # Gemini 2.5 Flash 최종 검증
│
├── stream/
│   ├── stream_reader.py           # RTSP 입력 + Ring Buffer + 1 FPS 샘플링
│   └── clip_extractor.py          # Ring Buffer → MP4 클립 추출
│
├── event/
│   ├── event_pipeline.py          # 이벤트 확정 → 저장 → 알림
│   └── alert_sender.py            # Webhook / Slack / LINE 알림
│
├── storage/
│   ├── db.py                      # SQLite WAL
│   └── s3_uploader.py             # 클립 S3 업로드
│
├── configs/
│   └── cameras.yaml               # 카메라별 설정 (ROI, 임계값)
│
├── deploy/
│   ├── setup.sh                   # 원스텝 배포
│   ├── Dockerfile                 # 컨테이너 빌드
│   └── hio-v2.service             # systemd 유닛
│
├── .env.example
├── requirements.txt
└── README.md
```

**목표 코드 규모: ~3,000 LOC** (v1의 14,000 LOC에서 ~80% 감소)

---

## 예상 성능

### Tier 1 → Tier 2 → Tier 3 Cascade 효과

```
                    프레임 수/일     처리 비용
전체 프레임:         259,200        -
Tier 1 트리거:       ~13,000 (5%)   GPU 15ms × 259K = ~1시간 (분산)
Trigger 누적:        ~500 (0.2%)    -
Tier 2 분석:         ~500           GPU 10초 × 500 = ~1.4시간
Tier 3 검증:         ~100 (0.04%)   API $0.003 × 100 = $0.30/일
확정 이벤트:         ~20-50         -
```

### 예상 정확도 (추가 학습 없음, 원본 모델)

| 시나리오 | Tier 1 Recall | Tier 2 Precision | 최종 정확도 (예상) |
|---|---|---|---|
| Cash | 85-92% (YOLO-Pose) | 75-85% (Qwen 영상) | **80-88%** |
| Fire | 80-90% (CLIP) | 85-92% (Qwen 영상) | **85-93%** |
| Violence | 75-85% (CLIP) | 82-90% (Qwen 영상) | **80-90%** |

**v1 대비 개선 포인트:**

| 항목 | v1 (Florence) | v2 (YOLO+CLIP+Qwen) |
|---|---|---|
| Cash 탐지율 | 0-58% | 80-88% (예상) |
| 판단 방식 | 캡션 → 키워드 매칭 (간접) | Pose 거리 + VLM 직접 질문 |
| 영상 이해 | 단일 프레임 | 12초 영상 시퀀스 |
| False Positive | 키워드 확장으로 증가 | Trigger 누적 + VLM + Gemini 3중 필터 |
| 코드 복잡도 | 14,000 LOC | ~3,000 LOC |

---

## 배포

### 최소 요구사항

| 항목 | 스펙 |
|---|---|
| GPU | NVIDIA T4 16GB (최소) / L4 24GB (권장) |
| CPU | 4+ vCPU |
| RAM | 16GB+ |
| 스토리지 | 50GB+ (gp3) |
| OS | Ubuntu 22.04+ |
| Python | 3.10+ |
| CUDA | 12.1+ |

### 환경 변수

```bash
# 필수
GEMINI_API_KEY=your_key

# GPU
CUDA_VISIBLE_DEVICES=0

# 모델
YOLO_MODEL=yolo26s-pose.pt
CLIP_MODEL=ViT-L/14
QWEN_MODEL=Qwen/Qwen2.5-VL-3B-Instruct

# 임계값
CASH_WRIST_THRESHOLD_PX=80
CLIP_FIRE_THRESHOLD=0.25
CLIP_VIOLENCE_THRESHOLD=0.30
QWEN_CONFIDENCE_HIGH=0.7      # 이상이면 Gemini 스킵
QWEN_CONFIDENCE_LOW=0.3       # 이하이면 기각

# 스트림
SAMPLE_FPS=1.0
RING_BUFFER_SECONDS=60
CLIP_PRE_SECONDS=6
CLIP_POST_SECONDS=6
TRIGGER_COOLDOWN_SECONDS=60

# 서버
SERVER_PORT=8000
DB_PATH=data/events.db
```

---

## 향후 확장

1. **LoRA Fine-tuning**: 운영 데이터 축적 후 Qwen2.5-VL-3B를 호텔 CCTV 도메인에 특화
2. **YOLO26s-pose → YOLO26m-pose**: 카메라 거리가 먼 경우 keypoint 정밀도 개선
3. **Qwen3-VL 업그레이드**: 출시 시 drop-in replacement로 정확도 향상
4. **멀티 GPU / 카메라 확장**: 10대 이상 시 g6.2xlarge 또는 DeepStream 전환
5. **자동 ROI 캘리브레이션**: 최초 설치 시 카운터 영역 자동 감지
6. **A/B 테스트 프레임워크**: 프롬프트 변경의 정확도 영향 정량 측정
