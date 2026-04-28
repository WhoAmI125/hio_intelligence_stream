# Future Plan: Shared Backbone Multi-Head SigLIP2

나중에 호텔 특화 GT가 충분히 쌓인 뒤 학습할 계획. 지금 운영에는 적용하지 않는다.

작성일: 2026-04-24
상태: **Deferred** (GT 수집 후 재평가)

---

## 1. 동기

현재 `prithivMLmods/Fire-Detection-Siglip2`와 `prithivMLmods/Human-Action-Recognition`은 둘 다
`google/siglip2-base-patch16-224`를 base로 하지만, **공개 checkpoint는 full fine-tune이라 backbone 가중치가
서로 다르다**. 따라서 한 번의 image encode로 두 head에 공급하는 "shared backbone"이 공개 weights로는 불가.

이걸 자체 학습으로 해결하면 compute/VRAM 절감 효과가 크다.

| 옵션 | backbone | head | 프레임당 ms | VRAM |
|---|---|---|---:|---:|
| 2 classifier 독립 로드 | 2개 | 2개 | 160~170 | ~740 MB |
| **shared backbone multi-head** | **1개** | **2개** | **80~90** | **~400 MB** |

---

## 2. 학습 옵션 (난이도 오름차순)

### Option 1 — Frozen backbone + head-only (가장 쉬움)

```python
class MultiHeadSiglip2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SiglipVisionModel.from_pretrained(
            "google/siglip2-base-patch16-224"
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.fire_head = nn.Linear(768, 3)   # Fire / Normal / Smoke
        self.fight_head = nn.Linear(768, 2)  # Fight / No-fight
```

- 데이터: task당 5,000~10,000 라벨 이미지
- 시간: T4 1대, 2~3 epoch, 2~5시간
- 기대 F1: Fire 0.94~0.97, Violence 0.73~0.80
- 장점: 가장 안전
- 단점: domain fit 제한적

### Option 2 — LoRA backbone + heads (권장)

```python
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj","v_proj","k_proj","out_proj"],
    lora_dropout=0.05, bias="none",
)
backbone = get_peft_model(backbone, lora_config)
```

- 데이터: task당 8,000~15,000
- 시간: 4~8시간
- 기대 F1: Fire 0.96~0.98, Violence 0.78~0.85
- 장점: 백본 적응 + 파라미터 효율

### Option 3 — Full multi-task fine-tune (가장 강력)

- 데이터: task당 20,000~30,000
- 시간: 1~2일
- 기대 F1: Fire 0.99, Violence 0.85~0.90
- 단점: catastrophic forgetting 위험, 비용

---

## 3. 데이터 소스

### Fire/Smoke

| 데이터셋 | 규모 | 호텔 실내 적합도 | 비고 |
|---|---:|---|---|
| FASDD (FASDD_CV ground subset) | 120,000+ | ★★ | ESSDD, 3 sub |
| sagecontinuum/smokedataset | 41,000 | ★ | wildfire |
| touati-kamel/forest-fire-dataset | ~69,000 | ★ | forest fire video frames |
| Shravanig/fire_detection_final | 7,580 | ★★ | Forest-Fire-Detection 학습 데이터 |
| Kaggle DataCluster fire-and-smoke | ~7,000 | ★★★ | 다양한 환경 |
| UniDataPro/fire-and-smoke | 85 videos + bbox | ★★★ | 일부 실내 |

### Violence

| 데이터셋 | 규모 | 호텔 적합도 | 비고 |
|---|---:|---|---|
| RWF-2000 | 2,000 video × 150 frame ≈ 300k | ★★★★ | **CCTV 5sec 30fps**, fight/non-fight 1:1 |
| UCF-Crime | 1,900 video, 60~600s | ★★★★★ | **13 anomaly class** (fighting/robbery/shooting...) |
| SCVD/CCTV-Fights | ~1,000 clips | ★★★★ | CCTV fighting |
| Hockey Fight | 1,000 clips 1.6s | ★★ | 특수 환경 |
| Movies Fight | 200 clips | ★ | 영화 |

---

## 4. 데이터 조립 목표

| 클래스 | 소스 | 샘플 수 |
|---|---|---:|
| Fire | FASDD_CV 실내 + Shravanig + DataCluster | ~10,000 |
| Smoke | 동일 소스 smoke label | ~10,000 |
| Normal (no fire, no fight) | 호텔 CCTV 정상 + RWF-2000 non-violent | ~15,000 |
| Fight | RWF-2000 violence + UCF-Crime fighting/assault | ~10,000 |
| **Total** | | **~45,000** |

Train/Val/Test split: 80/10/10.

---

## 5. 권장 학습 설정 (Option 2 LoRA 기준)

```python
TrainingArguments(
    learning_rate=2e-4,
    per_device_train_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.02,
    gradient_accumulation_steps=2,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
)
```

Multi-task dataloader: 한 step에 fire batch 또는 action batch를 랜덤 샘플 (task 비율 1:1 균형).

Loss: `CrossEntropyLoss` per task, weighted sum `loss = 0.5 * L_fire + 0.5 * L_fight`.

---

## 6. 호텔 특화 단계 (Phase B)

Phase A 공개 데이터 모델 배포 → 호텔 운영 → labeling UI로 TP/FP 수집 →

- 100~200건: fine-tune 준비
- 300~500건: LoRA re-fine-tune (domain gap 20~30% 복구 기대)
- 1000건+: full fine-tune, Fire F1 0.98+, Violence F1 0.87+ 목표

---

## 7. v3 통합 스케치

```python
# model_server/proposal/multihead_siglip.py
class MultiHeadClassifier:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.model = MultiHeadSiglip2.from_pretrained(checkpoint_path).to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(
            "google/siglip2-base-patch16-224"
        )
        self.device = device

    def score(self, frame_bgr) -> dict[str, dict[str, float]]:
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feat = self.model.backbone(**inputs).pooler_output
            fire_logits = self.model.fire_head(feat)
            fight_logits = self.model.fight_head(feat)
        fire = torch.softmax(fire_logits[0], dim=0).cpu().tolist()
        fight = torch.softmax(fight_logits[0], dim=0).cpu().tolist()
        return {
            "fire":   {"fire": fire[0], "normal": fire[1], "smoke": fire[2]},
            "fight":  {"fight": fight[0], "no_fight": fight[1]},
        }
```

`semantic_filter.SemanticPrefilter`를 이 multi-head 클래스로 교체.
backbone 1회 forward → fire/fight 두 head 결과 동시 반환.

---

## 8. Trigger 조건

- v3 로컬 GT feedback 300건 이상 누적
- 또는 공개 데이터로 먼저 baseline 학습 (Option 1 frozen)해서 2개 독립 SigLIP 대비 성능 동등 이상 검증

---

## 9. 참고 문헌

- FASDD: https://essd.copernicus.org/preprints/essd-2023-73/
- RWF-2000: https://arxiv.org/abs/1911.05913
- DVD (2025): https://arxiv.org/html/2506.05372v1
- SigLIP 2: https://huggingface.co/blog/siglip2
- Fine-Tuning SigLIP2 공식 블로그: https://huggingface.co/blog/prithivMLmods/siglip2-finetune-image-classification
- PEFT LoRA: https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
