# HIO Intelligence Stream v3

호텔 CCTV에서 `cash`, `fire`, `violence` 3개 시나리오를 감지하고, Gemini가 원본 full-frame 영상과 1개 overlay 영상을 보고 최종 검증하는 v3 런타임입니다.

최신 반영일: **2026-04-29** — cash exchange-band proposal + full-clip SigLIP2 cash aggregation + unified requirements + MJPEG 안정화 반영.

주요 변경 요약:

- **Cash 후보 로직 재구성** — `cashier_zone` 안 손목만으로는 emit 금지. `exchange_band` 근처 손목/움직임/crossing/proximity와 5초 rolling window 내 2회 hit를 cash 후보 기준으로 사용.
- **`exchange_band`, `staff_work_zone` 연결** — DB, model API, frontend zone editor, auto-restore, event metadata, candidate overlay, Gemini packet까지 연결. `exchange_band`는 선이 아니라 얇은 polygon band로 3점 이상 필요.
- **Cash full-clip SigLIP2 aggregation 추가** — hard pass 전이라도 cash `soft_hit`이 안정적으로 잡히면 최근 15초 clip에서 12 full-frame을 샘플링하고 `google/siglip2-base-patch16-224`로 cash score를 집계해 후보를 승격.
- **requirements 단일화** — `requirements.txt` 하나에 CUDA torch/cu121 + runtime dependency를 고정. `requirements_gpu.txt`는 호환용 shim으로 `requirements.txt`만 참조.
- **MJPEG preview 안정화** — 로컬 full-res 15fps preview를 제거하고 `3fps base / 12fps burst / width 854 / quality 55`로 통일. dedup heartbeat가 동일 프레임 장기 유지 시 브라우저 스트림을 굶기지 않도록 수정.
- **Tier-2에 fine-tuned SigLIP2 분류기 2개 추가** — `Fire-Detection-Siglip2` (99.4% acc 3-class), `Human-Action-Recognition` (fighting F1 0.84, 15-class, 14 neutral class 내장). 둘 다 Apache 2.0 오픈 웨이트.
- **YOLO26 generic detect 드롭** — COCO class에 한국 원화·weapon class가 없어 실측 기여 0. `YOLO26_DETECT_WEIGHTS=""`가 기본.
- **8 vCPU 스팟 대응 안정화** — 720p ingest 다운샘플, FFmpeg `ultrafast` / `crf 23` (+ `FFMPEG_ENCODER=h264_nvenc` 옵션), overlay fps `/1`, MJPEG stable preview (`3fps`, width `854`, dedup heartbeat), uvicorn 1 worker, NVDEC `h264_cuvid` 명시.
- **Episode 1-per-emission 재의미** — 조건 지속 시 같은 episode = 1회만 emit. "20초 metronome" 문제 해소. `max_gap_sec=6` 이상 non-detected 프레임 후 새 episode.
- **Classifier permanent_failure 감지** — 8회 연속 load 실패 시 ERROR 로그 + `/api/vlm/config/`의 `*.permanent_failure=true`로 운영자 알림.
- **SigLIP health runtime 노출** — `/api/vlm/config/`에 `semantic_filter / fire_classifier / action_classifier` 각각 `loaded`, `failure_count`, `last_error`, `permanent_failure` 필드.
- **FlushWorker lock + today-skip** — 동시 flush race 방지, `archive_date(today)` 스킵으로 같은 날 재시작 데이터 보존.
- **Labeling FP subtype 강제** — 7종 (phone/receipt/card/empty_scene/staff_only/no_transfer/other) UI + 서버 validation. 키보드 Q/W/E/R/T/Y/U (FP 모드 시만 활성).
- **스팟 인터럽션 대응** — `FLUSH_INTERVAL_SEC=120` + `tools/spot_interruption_watcher.py` (pre-stop `/flush-now/` POST) + systemd units (`deploy/`).
- Multi-head 공유 backbone 학습 계획은 `docs/FUTURE_MULTIHEAD_SIGLIP_PLAN.md`에 보관 (현재 배포 미포함).

기준 설계 문서:

```text
E:\02_StayG\00_CCTV_Motion_Detection\github\HIO_V2_YOLO26_GEMINI_ARCHITECTURE.md
```

현재 구현 폴더:

```text
E:\02_StayG\00_CCTV_Motion_Detection\github\hio_intelligence_stream_v3
```

## 1. Architecture Diagram

```text
+-------------------+
| RTSP CCTV Camera  |
+---------+---------+
          |
          v
+---------+-----------------------------------------------+
| StreamManager                                           |
| - RTSP reconnect (exponential backoff + jitter)         |
| - NVDEC (h264_cuvid) -> CPU fallback allowed            |
| - ingest downsample to INGEST_DOWNSAMPLE_HEIGHT (720)   |
| - ring buffer (deque + monotonic ts)                    |
+---------+-----------------------------------------------+
          |
          v
+---------+---------+
| InferenceScheduler|
| - per-camera FPS  |
| - latest sampling |
| - burst handling  |
+---------+---------+
          |
          v
+---------+---------+
| HioV3Pipeline     |
+---------+---------+
          |
          v
+---------+-----------------------------------------------+
| Tier 1: YOLO26 / pose / ROI                             |
| - YOLO26 pose: person bbox + keypoints + wrists         |
| - YOLOv26 fire: fire/smoke candidate                    |
| - CashierTrackerV3: cashier/customer role + linger      |
| (generic YOLO26 detect disabled by default)             |
+---------+-----------------------------------------------+
          |
          | only tier1-candidate scenarios pass through
          v
+---------+-----------------------------------------------+
| Tier 2: SigLIP semantic layer (gated)                   |
| - cash  : SigLIP2 frame/clip scoring                    |
|           (google/siglip2-base-patch16-224)             |
|           positive vs neutral prompt                    |
| - fire  : Fire-Detection-Siglip2 (fire/normal/smoke)    |
|         + SigLIP zero-shot neutralizer prompts          |
| - viol. : Human-Action-Recognition (fighting class)     |
|           minus max(hugging, clapping, laughing, dance) |
|           * V3_ACTION_NEUTRAL_DAMPEN                    |
+---------+-----------------------------------------------+
          |
          v
+---------+-----------------------------------------------+
| TemporalEventEngine                                     |
| - exchange_band + pose + SigLIP weighted proposal       |
| - cash/fire/violence only                               |
| - polygon_coords + skeleton_summary metadata            |
+---------+-----------------------------------------------+
          |
          v
+---------+-----------------------------------------------+
| EpisodeManager                                          |
| - repeated-frame proposal merge                         |
| - min hits / min duration                               |
| - event cooldown default 20 sec                         |
+---------+-----------------------------------------------+
          |
          v
+---------+-----------------------------------------------+
| Event Admission                                         |
| - event JSON pending                                    |
| - capture clip_entries immediately from ring buffer     |
| - duplicate camera/scenario job merge/drop              |
+---------+-----------------------------------------------+
          |
          v
+---------+-----------------------------------------------+
| EventPostProcessor                                      |
| - async Gemini/clip worker (POSTPROCESS_WORKERS=2)      |
| - queue full -> dead_letter/events_dropped.jsonl        |
| - queue latency > validation window -> validation_error |
+---------+-----------------------------------------------+
          |
          v
+---------+-----------------------------------------------+
| CandidateClipBuilder                                    |
| - raw: reuse val_{event_id}.mp4                         |
| - context_overlay.mp4: red cashier ROI + skeleton/SoM   |
|   (rendered at clip_fps / V3_OVERLAY_FPS_DIVISOR)       |
| - skeleton.json                                         |
| - FFmpeg preset=ultrafast, crf=28                       |
+---------+-----------------------------------------------+
          |
          v
+---------+-----------------------------------------------+
| Tier 3: GeminiTemporalValidatorV3                       |
| - raw + context_overlay only                            |
| - cash hard gates + KRW context                         |
| - fire hard rules (sunlight/LED/extinguisher reject)    |
| - fail-closed on API/disabled errors                    |
+---------+-----------------------------------------------+
          |
          v
+---------+---------+         +----------------------+
| LocalStorage     | ------>  | FlushWorker / DB     |
| events/clips/log |          | SQLite event storage |
+---------+--------+          +----------+-----------+
          |                              |
          v                              v
+---------+-----------------------------------------------+
| Frontend :8002                                          |
| dashboard / adhoc monitor / proposal logs / Gemini logs |
| labeling feedback (MJPEG 3 fps cap)                     |
+---------------------------------------------------------+
```

## 2. Current Runtime Summary

| 항목 | 현재 값 |
|---|---|
| 활성 시나리오 | `cash`, `fire`, `violence` |
| pipeline version | `v3-yolo26-tier1-siglip2classifier-episode-gemini` |
| Tier 1 | YOLO26 pose tracking + fire/smoke YOLO + exchange-band pose rules (generic detect disabled) |
| Tier 2 | SigLIP2 cash frame/clip scoring + Fire-Detection-Siglip2 + Human-Action-Recognition |
| Event stabilization | `EpisodeManager` |
| Final validator | Gemini temporal video validator |
| Candidate clip policy | minimal: `raw`, `context_overlay`, `skeleton_json` |
| Gemini API 오류 정책 | TP 처리 금지, `validation_error` |
| CPU fallback | 기본 금지: `ALLOW_CPU_FALLBACK=false` |
| Gemini concurrency | `GEMINI_MAX_CONCURRENT=2` |
| BASE_FPS / BURST_FPS | `3.0 / 3.0` (pose 3fps, 4 camera budget) |
| Ingest downsample | `INGEST_DOWNSAMPLE_HEIGHT=720` |
| Overlay fps | `clip_fps / V3_OVERLAY_FPS_DIVISOR` (현재 /1) |
| FFmpeg 인코딩 | `preset=ultrafast, crf=23` |
| DB flush 주기 | `FLUSH_INTERVAL_SEC=120` (스팟 대응) |

## 3. Model Layer

| 모델 | 파일/설정 | 역할 | Tier | 라이선스 |
|---|---|---|---|---|
| YOLO26 pose | `models/yolo26s-pose.pt` | 사람 bbox, keypoints, wrist 좌표 (cash ROI 필수) | 1 | - |
| YOLOv26 fire/smoke | `models/yolov26_fire_detection_best.pt` | fire/smoke YOLO gate | 1 | HF `SalahALHaismawi/yolov26-fire-detection` |
| YOLO26 generic detect | `YOLO26_DETECT_WEIGHTS` (**empty, disabled**) | 4 camera 배포에서는 드롭 (COCO class 한계) | - | - |
| SigLIP2 base zero-shot | `google/siglip2-base-patch16-224` | cash 현재 프레임 semantic + cash full-clip aggregation + fire neutralizer | 2 | Apache 2.0 |
| **Fire-Detection-Siglip2** | `prithivMLmods/Fire-Detection-Siglip2` | SigLIP2 fine-tuned 3-class (Fire / Normal / Smoke, 99.41% acc) | 2 | Apache 2.0 |
| **Human-Action-Recognition** | `prithivMLmods/Human-Action-Recognition` | SigLIP2 fine-tuned 15-class (fighting F1 0.84 + 14 neutral class) | 2 | Apache 2.0 |
| Gemini | `.env`의 `GEMINI_MODEL` | 최종 temporal validation (KRW hard gate + fire hard rule) | 3 | - |

모든 모델이 **오픈 웨이트**입니다. Fine-tuned SigLIP2 2개 base는 `google/siglip2-base-patch16-224`
(92.9M params, 224×224). YOLO fire 모델의 `other`, `background`, `none` 계열 label은
fire signal로 사용하지 않습니다.

### 3.1 Tier-2 classifier head 동작

Tier-1 (YOLO pose / YOLO fire / pose rules)에서 후보 scenario가 발생한 프레임에만
classifier가 돌아갑니다. **매 프레임 상시 실행 아님**.

| 시나리오 | Tier-1 trigger | Tier-2 모델 | 수식 |
|---|---|---|---|
| cash | exchange_band soft/hard hit, handover, motion/crossing | SigLIP2 frame + full-clip aggregation | positive-vs-neutral prompt softmax over sampled full frames |
| fire | YOLO fire bbox | Fire-Detection-Siglip2 | `max(fire_prob, smoke_prob)`, zero-shot neutralizer로 감쇠 |
| violence | pose `close_person_pair` / `cross_person_wrist_near_body` | Human-Action-Recognition | `max(0, fighting - V3_ACTION_NEUTRAL_DAMPEN × max(hugging, clapping, laughing, dancing))` |

Classifier는 직접 `AutoModel` logits → softmax → 확률을 쓰므로 zero-shot 대비 **text encoder forward가 없어 프레임당 20~30% 빠름**. 로드 실패 시 `SemanticPrefilter`의 exponential backoff retry 경로를 따라갑니다.

## 4. Runtime Flow

```text
1. StreamManager가 RTSP 프레임을 720p로 다운샘플 (INGEST_DOWNSAMPLE_HEIGHT) 후 ring buffer에 유지
2. InferenceScheduler가 최신 프레임을 샘플링 (BASE_FPS=3.0, V3_POSE_FPS=3.0)
3. YOLO26Runner가 pose + fire 모델 실행 (generic detect 드롭)
4. TemporalEventEngine이 Tier 1 후보를 계산
5. Tier 1 후보가 있는 scenario만 SigLIP 경로 실행
     - cash  : zero-shot SigLIP2 (google/siglip2-base-patch16-224)
     - fire  : Fire-Detection-Siglip2 + SigLIP zero-shot neutralizer
     - viol. : Human-Action-Recognition (fighting - neutral dampen)
     - cash soft_hit: hard pass 전이라도 5초 window 내 2회 hit면 최근 15초 full-clip SigLIP2 aggregation 실행
6. SigLIP/classifier score와 neutralizer를 반영해 proposal 재계산
7. EpisodeManager가 반복 프레임 proposal을 하나의 episode로 안정화 (cooldown 20s)
8. 안정화된 이벤트만 EventPostProcessor queue에 등록
9. queue 등록 시점에 validation clip entries를 즉시 확보
10. CandidateClipBuilder가 overlay fps = clip_fps / V3_OVERLAY_FPS_DIVISOR 로 렌더
11. Gemini가 raw + context_overlay를 보고 hard gate 검증
12. LocalStorage에 event, clip, thumbnail, logs 저장 (FFmpeg ultrafast / crf 28)
13. FlushWorker가 DB Server로 2분 간격 batch sync (FLUSH_INTERVAL_SEC=120)
14. Frontend에서 모니터링/라벨링 (MJPEG 3 fps cap)
```

## 5. Scenario Algorithms

### 5.1 Cash

cash는 단순히 손목이 ROI 안에 들어간다고 확정하지 않습니다. Tier 1에서 아래 신호를 조합합니다.

| 신호 | 의미 |
|---|---|
| `wrist_inside_cashier_zone` | 손목이 cashier ROI 안에 있음 |
| `handover_like_pose` | 복수 사람 손목/손 위치가 거래 동작처럼 가까움 |
| `cashier_tracker_customer_staff_wrist_proximity` | CashierTrackerV3가 직원/고객 역할과 손목 근접을 감지 |
| `multiple_persons_near_counter` | 카운터 주변에 여러 사람 |
| `siglip_cash_context` | Tier 2 SigLIP이 결제/현금거래 장면과 유사하다고 판단 |

`CashierTrackerV3`는 원본 v1의 cashier/customer 역할 추적 개념을 v3 pose 구조에 맞게 이식한 모듈입니다. 직원은 cashier zone에서 반복 관측되는 사람으로 추정하고, 너무 오래 같은 위치에 머무는 사람은 고객 linger로 재분류합니다.

> cash는 별도 bbox detector를 쓰지 않습니다. 한국 원화/손 전달은 YOLO class로 안정화하기 어렵기 때문에, Tier-1은 `exchange_band` 근처 pose/motion/crossing으로 후보를 만들고 Tier-2는 SigLIP2 full-frame / full-clip semantic aggregation으로 보강합니다. 최종 판정은 Gemini hard gate (KRW + ownership transfer)가 담당합니다.

### 5.2 Fire

fire는 YOLO fire 후보가 Tier-1 gate입니다. Tier-2에서 Fire-Detection-Siglip2가 직접 3-class 확률을 내고, SigLIP zero-shot neutralizer가 조명/반사/소화기 장면을 감쇠합니다.

Fire-Detection-Siglip2 출력:

```text
fire    : 불꽃이 실제로 보이는 정도
smoke   : 연기 plume이 보이는 정도
normal  : 정상 실내 조명
```

SigLIP zero-shot neutralizer prompt (`semantic_filter.PROMPTS["fire"]`):

```text
# positive
visible fire flames
visible smoke filling the scene
a fire emergency in an indoor camera view

# neutralizer
sunlight glare or bright reflection with no fire
TV / LED screen / sign that looks bright
red or orange sign lamp or decorative light
fire extinguisher with no smoke or flame
steam fog blur or camera artifact
```

`siglip_neutralizer_score`가 높고 `siglip_fire_score`가 낮으면 (Tier-2 gate `V3_FIRE_SIGLIP_MIN_SCORE=0.52` 미만) fire score가 `× 0.35` 감쇠됩니다. SigLIP fire ≥ 0.65이면 YOLO 점수보다 SigLIP * 0.8이 우선 (작은 불씨도 놓치지 않도록).

### 5.3 Violence

violence는 YOLO weapon class가 사실상 없어서 pose + Tier-2 classifier 조합으로 proposal을 만듭니다. 최종 판단은 Gemini가 원본 영상을 보고 합니다.

| 신호 | 의미 |
|---|---|
| `close_person_pair` | 사람 bbox/center가 비정상적으로 가까움 |
| `cross_person_wrist_near_body` | 한 사람의 손목이 다른 사람 bbox 근처 |
| `siglip_violence_context` | Human-Action-Recognition의 `fighting - dampen × max(neutral)` |

Human-Action-Recognition 15-class 중 다음을 활용합니다:

| 용도 | 클래스 |
|---|---|
| positive (violence) | `fighting` |
| **neutral dampen** (friendly interaction 차감) | `hugging`, `clapping`, `laughing`, `dancing` |
| unused (현재 노드에 영향 없음) | calling, cycling, drinking, eating, listening_to_music, running, sitting, sleeping, texting, using_laptop |

수식: `violence_score = max(0, fighting - V3_ACTION_NEUTRAL_DAMPEN × max(hugging, clapping, laughing, dancing))`

`V3_ACTION_FIGHT_MIN_SCORE=0.40`이 Tier-2 semantic gate의 임계값이며 TemporalEventEngine에서 violence bonus cap은 0.45입니다.

## 6. EpisodeManager

`EpisodeManager`는 같은 사건이 여러 프레임에서 반복 proposal로 올라오는 문제를 줄입니다.

| scenario | min hits | min duration | high confidence |
|---|---:|---:|---:|
| cash | 2 | 1.0 sec | 0.85 |
| fire | 2 | 1.0 sec | 0.80 |
| violence | 3 | 2.0 sec | 0.90 |

episode 정책:

```text
- 아직 안정화되지 않은 proposal은 이벤트로 내보내지 않음
- 같은 camera/scenario가 최근 방출됐으면 cooldown 동안 suppress
- 기본 cooldown은 V3_EPISODE_COOLDOWN_SEC=20
- UI의 event_cooldown_sec도 기본 20초로 동작
```

## 7. Candidate Clip Policy

v3는 더 이상 이벤트 1건마다 여러 overlay mp4를 만들지 않습니다.

기본 생성 artifact:

| key | 실제 파일 | 설명 |
|---|---|---|
| `raw` | `val_{event_id}.mp4` | Gemini가 보는 원본 full-frame validation clip |
| `context_overlay` | `{event_id}_context_overlay.mp4` | full-frame overlay with cashier/exchange/staff zones + skeleton/SoM marker, fps=`clip_fps / V3_OVERLAY_FPS_DIVISOR` |
| `skeleton_json` | `{event_id}_skeleton.json` | skeleton summary metadata |

기본 생성하지 않는 것:

```text
{event_id}_raw.mp4
{event_id}_skeleton_overlay.mp4
{event_id}_cashier_zone_overlay.mp4
cashier ROI crop video
```

중요 원칙:

```text
- crop 금지
- Gemini는 원본 CCTV 구도 전체를 봐야 함
- overlay는 1개 context_overlay만 저장
- raw는 validation clip을 재사용
- overlay 렌더링은 save_clip_stream으로 frame-by-frame 처리
- overlay fps는 clip_fps / V3_OVERLAY_FPS_DIVISOR (현재 /1, Gemini 전달 전 FPS 다운샘플 없음)
- FFmpeg: preset=ultrafast, crf=23 (FFMPEG_PRESET / FFMPEG_CRF로 조정)
```

## 8. Gemini Validation

Gemini 입력은 최대 2개 영상입니다.

```text
1. raw
2. context_overlay
```

`context_overlay`의 빨간 박스는 cashier/counter area를 표시합니다. 현금거래는 이 빨간 overlay box가 표시한 cashier 영역 안 또는 근처에서 일어나는지 봅니다. 단, 빨간 박스 자체, skeleton line, SoM marker는 사건 증거가 아니라 위치/사람 식별용 힌트입니다.

### 8.1 Cash Hard Gate

cash TRUE 조건은 Gemini prompt와 parser에서 둘 다 강제합니다.

```text
H1: visible Korean cash / banknotes
H2: ownership transfer or payment movement
H3: cashier / counter / register context
S_STRONG: 명확한 결제/수납 증거
no_hedging: appears / likely / maybe / probably 류 표현 없음
```

KRW 지폐 색상 문맥:

```text
1,000 KRW: blue
5,000 KRW: green
10,000 KRW: orange / red-orange
50,000 KRW: yellow
```

다음은 cash로 인정하지 않습니다.

```text
receipt
white paper
card
phone
menu
form
envelope
room key
```

Gemini가 cash hard gate를 충족하지 못하면 confidence가 높아도 `FALSE_POSITIVE`로 강제됩니다.

### 8.2 Fire Hard Rule

fire TRUE 조건:

```text
visible flame
visible smoke plume
temporally persistent smoke
```

다음은 reject 대상입니다.

```text
sunlight / glare / reflection
TV / LED screen / signage
red or orange signs
lamps / decorative lights
fire extinguisher without smoke/flame
fog / steam / blur / camera artifact
```

### 8.3 Fail-Closed

Gemini가 꺼져 있거나 API 오류가 나면 이벤트를 TP로 올리지 않습니다.

```text
Gemini disabled -> validation_error
API error       -> validation_error
no clip         -> validation_error
confidence      -> 0.0
is_detected     -> false
```

## 9. Queue and Ring Buffer Safety

기존 문제는 worker queue가 밀린 뒤 clip을 뜨면 ring buffer에서 프레임이 사라질 수 있다는 점이었습니다.

현재 정책:

```text
- 이벤트 admission 시점에 admission_clip_entries 확보
- worker는 가능하면 admission_clip_entries를 그대로 사용
- 같은 camera/scenario job이 이미 queue에 있으면 duplicate_pending으로 drop
- queue full이면 dead_letter/events_dropped.jsonl에 기록
- queue latency가 validation_clip_sec보다 길면 Gemini 검증 중단
```

dead letter 위치:

```text
data/dead_letter/events_dropped.jsonl
```

## 10. GPU and Runtime Health

silent failure를 막기 위해 다음 정책을 사용합니다.

| 상황 | 현재 정책 |
|---|---|
| YOLO CUDA 불가 | `ALLOW_CPU_FALLBACK=false`면 error/degraded로 기록 |
| YOLO inference 실패 | 빈 결과로 조용히 넘기지 않고 `model_health.errors`에 기록 |
| SigLIP load 실패 | 영구 비활성화하지 않고 exponential backoff 재시도 (5→10→…→300s) |
| SigLIP CUDA 불가 | CPU fallback 기본 금지 |
| Fire classifier / Action classifier load 실패 | exponential backoff 재시도, 실패 중엔 Tier-2 zero-shot만 사용 |
| Gemini disabled/API error | TP 처리 금지, `validation_error` |

status API와 UI는 `/api/vlm/config/`의 `model_health` 필드로 runtime 상태를 확인할 수 있습니다. v3 pipeline은 `HioV3Pipeline.health()`로 `fire_classifier`/`action_classifier`의 loaded/device/failure_count를 노출합니다.

### 10.1 CPU budget (g4dn.2xlarge 8 vCPU)

| 스레드 | 예산 |
|---|---:|
| RTSP reader (4 cam, NVDEC) | 0.4 |
| Frame preprocess | 0.3 |
| Inference dispatcher + worker | 0.3 |
| Postprocess workers (2) + overlay | 0.8 |
| MJPEG live (3 fps cap, idle-pause) | 0.6 |
| uvicorn × 3 (model+db+frontend, 1 worker each) | 0.5 |
| FlushWorker + 시스템 | 0.7 |
| **합계 (평균)** | **~3.6 vCPU / 8** |

NVDEC이 작동하지 않으면 SW 디코드로 RTSP당 ~1.5 vCPU가 더 붙습니다. 반드시 확인:

```bash
nvidia-smi --query-gpu=utilization.decoder --format=csv -l 1
```

`decoder > 0%`이어야 정상.

### 10.2 VRAM budget (T4 15GB)

```
YOLO26 pose (+ batch acts)      : ~380 MB
YOLOv26 fire                    : ~300 MB
SigLIP base (zero-shot)         : ~400 MB
Fire-Detection-Siglip2          : ~370 MB
Human-Action-Recognition        : ~370 MB
CUDA context + caches           : ~500 MB
NVDEC buffers (4 stream)        : ~200 MB
--------------------------------
합계                            : ~2.5 GB / 15 GB
```

## 11. Storage and Retention

`LocalStorage`는 이벤트 JSON, clips, thumbnails를 날짜별로 저장합니다.

```text
data/
  events/YYYYMMDD/*.json
  clips/YYYYMMDD/*.mp4
  thumbnails/YYYYMMDD/*.jpg
  dead_letter/events_dropped.jsonl
  smoke/clips/YYYYMMDD/*
```

human feedback이 있는 이벤트는 retention cleanup에서 보호합니다. TP/FP/unclear 라벨이 붙은 이벤트의 JSON, clip, thumbnail은 일반 보존 기간이 지나도 삭제하지 않습니다.

스팟 운영 시 `data/`는 별도 EBS 볼륨 (`/srv/hio-data`)에 마운트 권장. 인스턴스 교체에도 보존.

## 12. File Structure

```text
hio_intelligence_stream_v3/
  .env
  .env.example
  start_local.py
  requirements.txt
  requirements_gpu.txt                  # compatibility shim -> requirements.txt
  README.md
  docs/
    FUTURE_MULTIHEAD_SIGLIP_PLAN.md    # 나중에 호텔 GT 누적 후 학습할 계획
  deploy/
    README.md                           # AWS g4dn.2xlarge spot setup guide
    vlm-model.service                   # systemd unit (Restart=always, TimeoutStopSec=120)
    vlm-db.service
    vlm-frontend.service
    vlm-spot-watcher.service            # IMDS poll + graceful stop
  models/
    yolo26s-pose.pt
    yolo26s.pt                          # 참고용. 기본 .env에서 disable
    yolov26_fire_detection_best.pt
  model_server/
    main.py
    vlm_api.py
    v3_pipeline.py                      # HioV3Pipeline + health()
    event_postprocessor.py
    local_storage.py                    # FFmpeg preset/crf env-driven
    stream_manager.py                   # INGEST_DOWNSAMPLE_HEIGHT, NVDEC
    config.py                           # 모든 env 플래그 중앙화
    proposal/
      yolo26_runner.py
      temporal_engine.py                # violence_semantic_gate env-driven
      semantic_filter.py                # fire_head/action_head 주입
      episode_manager.py
      cashier_tracker_v3.py
      classifier_heads.py               # Fire/Action SigLIP2 fine-tuned heads
      feedback_collector.py
    candidates/
      clip_builder.py                   # V3_OVERLAY_FPS_DIVISOR 반영
    validators/
      gemini_temporal_validator_v3.py
    skeleton/
      pose_features.py
  db_server/
    main.py
  frontend_server/
    main.py
    templates/vlm_pipeline/
      adhoc_rtsp.html
      v3_proposal_logs.html
      gemini_logs.html
      labeling.html
  tools/
    v3_smoke_test.py
    spot_interruption_watcher.py        # AWS IMDS poll -> systemctl stop
```

## 13. Environment Variables

핵심 `.env` 값:

```env
# Gemini (Tier-3)
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.1-flash-lite-preview
GEMINI_MAX_CONCURRENT=2

# v3 runtime
HIO_V3_ENABLED=true
HIO_V3_PIPELINE_VERSION=v3-yolo26-tier1-siglip2classifier-episode-gemini
V3_SCENARIOS=cash,fire,violence
ALLOW_CPU_FALLBACK=false

# Tier-1 YOLO
YOLO26_POSE_WEIGHTS=models/yolo26s-pose.pt
YOLO26_DETECT_WEIGHTS=                           # empty, disabled
YOLO26_FIRE_WEIGHTS=models/yolov26_fire_detection_best.pt
YOLO26_CASH_WEIGHTS=
YOLO26_DEVICE=cuda
V3_POSE_FPS=3.0

# Tier-2 SigLIP + classifier heads
V3_SEMANTIC_FILTER_ENABLED=true
V3_SEMANTIC_MODEL=google/siglip2-base-patch16-224
V3_SEMANTIC_DEVICE=cuda
V3_CASH_SIGLIP_CLIP_ENABLED=true
V3_CASH_SIGLIP_CLIP_FRAMES=12
V3_CASH_SIGLIP_CLIP_BATCH_SIZE=4
V3_CASH_SIGLIP_CLIP_WINDOW_SEC=15
V3_CASH_SIGLIP_CLIP_PEAK_WINDOW_SEC=5
V3_CASH_SIGLIP_CLIP_MIN_SCORE=0.50
V3_CASH_SIGLIP_CLIP_FRAME_POSITIVE=0.48
V3_CASH_SIGLIP_CLIP_MIN_POSITIVE_FRAMES=2
V3_CASH_SIGLIP_CLIP_COOLDOWN_SEC=2.0
V3_FIRE_CLASSIFIER_ENABLED=true
V3_FIRE_CLASSIFIER_MODEL=prithivMLmods/Fire-Detection-Siglip2
V3_ACTION_CLASSIFIER_ENABLED=true
V3_ACTION_CLASSIFIER_MODEL=prithivMLmods/Human-Action-Recognition
V3_CLASSIFIER_DEVICE=cuda

# Thresholds
V3_VALIDATION_CLIP_SEC=15
V3_CASH_PREFILTER_THRESHOLD=0.45
V3_FIRE_PREFILTER_THRESHOLD=0.35
V3_VIOLENCE_PREFILTER_THRESHOLD=0.45
V3_GEMINI_ALWAYS_VALIDATE=true
V3_FIRE_SIGLIP_MIN_SCORE=0.52
V3_FIRE_NEUTRALIZER_THRESHOLD=0.58
V3_ACTION_FIGHT_MIN_SCORE=0.40
V3_ACTION_NEUTRAL_DAMPEN=0.50

# Tier-2 SigLIP SUPPRESSION GATE (block Gemini call if SigLIP disagrees with pose)
V3_CASH_SIGLIP_GATE=0.30
V3_VIOLENCE_SIGLIP_GATE=0.25
V3_FIRE_SIGLIP_FLOOR=0.15

# Skeleton overlay frames in context_overlay clip (0 = ROI only, no ghost)
V3_OVERLAY_SKELETON_FRAMES=0

# Episode & clip policy
V3_CLIP_ARTIFACT_MODE=minimal
V3_EPISODE_COOLDOWN_SEC=20
V3_EPISODE_MAX_GAP_SEC=6

# 4 camera spot budget (CPU 압축)
BASE_FPS=3.0
BURST_FPS=3.0
INGEST_DOWNSAMPLE_HEIGHT=720
RTSP_HWACCEL=cuda
RTSP_HWACCEL_DECODER=h264_cuvid
RTSP_HWACCEL_ALLOW_FALLBACK=true
POSTPROCESS_WORKERS=2
POSTPROCESS_QUEUE_SIZE=256
CLIP_SAVE_MAX_CONCURRENT=1
CLIP_BUFFER_SECONDS=20

# FFmpeg (CPU-light encode)
FFMPEG_PATH=ffmpeg
FFMPEG_PRESET=ultrafast
FFMPEG_CRF=23
V3_OVERLAY_FPS_DIVISOR=1

# Frontend MJPEG stable preview
#   Keep browser preview light. Full-res 15fps can stall local monitoring.
#   DEDUP_FRAMES keeps encode load low; IDLE_PAUSE still sends heartbeat frames.
FRONTEND_MJPEG_FPS=3
FRONTEND_MJPEG_BURST_FPS=12
FRONTEND_MJPEG_QUALITY=55
FRONTEND_MJPEG_WIDTH=854
FRONTEND_MJPEG_DEDUP_FRAMES=true
FRONTEND_MJPEG_IDLE_PAUSE_SEC=5

# FFmpeg encoder: libx264 (CPU) | h264_nvenc (GPU, 10x faster on T4+)
FFMPEG_ENCODER=libx264
FFMPEG_NVENC_PRESET=p3
FFMPEG_NVENC_CQ=28

# Spot-safe flush
FLUSH_INTERVAL_SEC=120
LOCAL_RETENTION_DAYS=3
```

### 13.1 `/monitor/adhoc` MJPEG 안정화 메모

미리보기 초가 멈췄다가 한 번에 점프하는 현상은 RTSP 원본보다 MJPEG preview 경로 부하가 먼저 의심됩니다. local full-res `15fps`, quality `70`, width `0` 조합은 frontend -> model proxy에서 JPEG encode와 browser decode 부하가 커져 지연이 누적될 수 있습니다.

현재 권장값은 `FRONTEND_MJPEG_FPS=3`, `FRONTEND_MJPEG_BURST_FPS=12`, `FRONTEND_MJPEG_QUALITY=55`, `FRONTEND_MJPEG_WIDTH=854`, `FRONTEND_MJPEG_DEDUP_FRAMES=true`, `FRONTEND_MJPEG_IDLE_PAUSE_SEC=5`입니다. 동일 frame dedup 시 idle heartbeat가 끊기던 문제도 수정했으므로, 설정 변경 후 서버 재시작이 필요합니다.

`GEMINI_API_KEY`는 `.env`에만 두고 문서나 코드에 커밋하지 않습니다.

## 14. Setup and Run

### 14.1 Windows 로컬 개발

PowerShell 기준:

```powershell
cd E:\02_StayG\00_CCTV_Motion_Detection\github\hio_intelligence_stream_v3
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements_gpu.txt`는 기존 명령 호환용 shim이며, 실제 의존성은 `requirements.txt` 하나에서 관리합니다.

### 14.2 HuggingFace 모델 사전 캐시 (배포 전 권장)

```powershell
.\.venv\Scripts\python.exe - <<'PY'
from transformers import AutoImageProcessor, SiglipForImageClassification, AutoModel
for m in [
    "google/siglip2-base-patch16-224",
    "prithivMLmods/Fire-Detection-Siglip2",
    "prithivMLmods/Human-Action-Recognition",
]:
    AutoImageProcessor.from_pretrained(m)
    try:
        SiglipForImageClassification.from_pretrained(m)
    except Exception:
        AutoModel.from_pretrained(m)
    print("OK", m)
PY
```

### 14.3 통합 실행

```powershell
.\.venv\Scripts\python.exe start_local.py
```

개별 실행:

```powershell
.\.venv\Scripts\python.exe -m uvicorn model_server.main:app --host 127.0.0.1 --port 8000 --workers 1
.\.venv\Scripts\python.exe -m uvicorn db_server.main:app --host 127.0.0.1 --port 8001 --workers 1
.\.venv\Scripts\python.exe -m uvicorn frontend_server.main:app --host 127.0.0.1 --port 8002 --workers 1
```

주요 URL:

```text
http://127.0.0.1:8002/dashboard
http://127.0.0.1:8002/monitor/adhoc
http://127.0.0.1:8002/monitor/v3-proposal-logs
http://127.0.0.1:8002/monitor/gemini-logs
http://127.0.0.1:8002/monitor/labeling
http://127.0.0.1:8000/docs
http://127.0.0.1:8001/docs
```

### 14.4 AWS g4dn.2xlarge spot 배포

`deploy/README.md` 참고. systemd 4개 unit (`vlm-model`, `vlm-db`, `vlm-frontend`, `vlm-spot-watcher`)을 `/etc/systemd/system/`에 복사하고 `systemctl enable --now`.

## 15. UI Workflow

1. `/monitor/adhoc` 접속
2. RTSP camera 추가
3. cashier zone + exchange_band polygon 그리기
4. drawer zone은 필요하면 추가
5. `validation_clip_sec=15`, `event_cooldown_sec=20` 확인
6. camera start
7. `/monitor/v3-proposal-logs`에서 Tier 1/2 proposal 확인 (`siglip_fire_score`, `classifier_fighting` 등 상세 필드)
8. `/monitor/gemini-logs`에서 Gemini 결과와 raw/context overlay 확인
9. `/monitor/labeling`에서 TP/FP/unclear feedback 입력 (키보드 1/2/3 단축키)

## 16. API Summary

| Method | Path | 역할 |
|---|---|---|
| `POST` | `/api/vlm/start/` | RTSP camera start |
| `POST` | `/api/vlm/stop/` | camera stop |
| `GET` | `/api/vlm/status/` | camera runtime status |
| `POST` | `/api/vlm/zones/` | cashier/drawer polygon 저장 |
| `GET` | `/api/vlm/config/` | v3 runtime config (classifier head health 포함) |
| `GET` | `/api/vlm/events/` | local events |
| `GET` | `/api/vlm/gemini-logs/` | Gemini validation logs |
| `GET` | `/api/vlm/v3-proposal-logs/` | proposal logs |
| `POST` | `/api/vlm/feedback/` | human feedback 저장 |

## 17. Verification

컴파일:

```powershell
.\.venv\Scripts\python.exe -m compileall model_server db_server frontend_server tools
```

Smoke test:

```powershell
.\.venv\Scripts\python.exe tools\v3_smoke_test.py
```

정상 smoke output의 핵심:

```json
{
  "pipeline_version": "v3-yolo26-tier1-siglip2classifier-episode-gemini",
  "candidate_clip_contract": {
    "required_present": ["context_overlay", "raw", "skeleton_json"],
    "missing": [],
    "forbidden_present": []
  }
}
```

## 18. Troubleshooting

### 이벤트가 너무 많이 생김

확인 순서:

```text
1. V3_EPISODE_COOLDOWN_SEC=20 확인
2. UI event_cooldown_sec=20 확인
3. POSTPROCESS_QUEUE_SIZE와 dropped_total 확인
4. proposal log에서 episode.stable / episode.suppressed 확인
5. cash threshold를 0.45보다 올릴지 검토
```

### Gemini가 clip을 못 받음

확인할 key:

```text
candidate_clip_paths.raw
candidate_clip_paths.context_overlay
candidate_clip_paths.skeleton_json
```

`context_overlay`가 없으면 `data/clips/YYYYMMDD/` 쓰기 권한, FFmpeg, postprocess queue 상태를 확인합니다.

### CUDA가 안 잡힘

현재 기본은 CPU fallback 금지입니다.

```powershell
.\.venv\Scripts\python.exe - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

운영에서 CPU fallback을 허용하려면 `.env`에서 명시적으로 바꿉니다.

```env
ALLOW_CPU_FALLBACK=true
```

### NVDEC 작동 안 함 (4 cam에서 vCPU 폭주)

```bash
nvidia-smi --query-gpu=utilization.decoder --format=csv -l 1
```

`decoder 0%`면 FFmpeg가 `--enable-cuda --enable-cuvid`로 빌드됐는지 `ffmpeg -version | grep cuvid`로 확인. 미지원이면 `RTSP_HWACCEL_DECODER`를 비우거나 `h264` 로 변경해 OpenCV 기본 경로 사용.

### fire false positive가 많음

확인할 metadata:

```text
local_prefilter.siglip_fire_score
local_prefilter.siglip_neutralizer_score
local_prefilter.classifier_fire / classifier_smoke / classifier_normal
matched_keywords
```

햇빛, 화면, 간판, 조명 오탐이면:

```env
V3_FIRE_NEUTRALIZER_THRESHOLD=0.50      # 낮춰서 neutralizer 자주 활성
V3_FIRE_SIGLIP_MIN_SCORE=0.60           # 올려서 classifier 확신 높을 때만 통과
V3_FIRE_PREFILTER_THRESHOLD=0.45        # 올려서 후보 자체 줄이기
```

### violence false positive가 많음 (hugging/clapping 오탐)

확인:

```text
local_prefilter.classifier_fighting
local_prefilter.classifier_hugging / classifier_clapping / classifier_laughing / classifier_dancing
local_prefilter.classifier_neutral_max
```

악수·포옹을 싸움으로 보면:

```env
V3_ACTION_NEUTRAL_DAMPEN=0.70           # neutral dampen 강하게
V3_ACTION_FIGHT_MIN_SCORE=0.55          # semantic gate 올림
V3_VIOLENCE_PREFILTER_THRESHOLD=0.55    # 전체 threshold 올림
```

### cash false positive가 많음

확인할 metadata:

```text
cashier_tracker.triggered
cashier_tracker.cashier_count
cashier_tracker.customer_count
cashier_tracker.lingering_customer_count
cash_hard_gates
```

영수증/종이/카드 오탐은 Gemini hard gate에서 reject되어야 합니다. reject되지 않으면 Gemini response의 `cash_hard_gates`와 `reason_bullets`를 확인합니다.

### queue drop이 있음

확인 위치:

```text
data/dead_letter/events_dropped.jsonl
```

`duplicate_pending`은 같은 camera/scenario 작업이 이미 처리 중이라는 뜻입니다. `queue_full`이면 `POSTPROCESS_WORKERS`, `POSTPROCESS_QUEUE_SIZE`, Gemini latency를 같이 봐야 합니다.

### classifier head 로드 실패

`/api/vlm/config/`의 `fire_classifier.loaded` 또는 `action_classifier.loaded`가 false면
`failure_count`, `last_error`, `permanent_failure`를 확인. 주요 원인:

```text
- SentencePiece / protobuf 미설치    -> pip install sentencepiece protobuf + restart
- HF hub 접근 실패 (네트워크)        -> 사전 캐시 (14.2)로 해결
- CUDA OOM                          -> V3_CLASSIFIER_DEVICE=cpu로 임시 우회
- ALLOW_CPU_FALLBACK=false 위반       -> CUDA 복구 또는 명시적으로 true
- transformers 버전 불일치            -> SiglipForImageClassification 지원 버전 설치
```

`permanent_failure=true`가 표시되면 `MAX_LOAD_FAILURES=8`회 연속 실패로 운영자 조치 + 재시작이 필요한 상태입니다. classifier가 죽어도 zero-shot SigLIP 경로는 계속 돌아가므로 recall은 떨어져도 TP는 여전히 잡힙니다 (graceful degrade).

## 19. Implementation Notes

### SigLIP은 Tier 2

SigLIP은 모든 프레임에 항상 돌지 않습니다. YOLO/pose Tier 1에서 후보가 생긴 scenario만 SigLIP으로 보강합니다. 이는 비용 문제가 아니라 GPU 병목과 latency를 줄이기 위한 구조입니다. Fire-Detection-Siglip2 / Human-Action-Recognition classifier도 같은 gate를 따릅니다.

### Classifier permanent failure 감지

SigLIP / Fire classifier / Action classifier는 load 실패 시 exponential backoff (5→10→20→…→300s)로 재시도합니다. `MAX_LOAD_FAILURES=8`회 (~10분) 초과 시 `permanent_failure=true`로 표시되고 ERROR 로그 1회 발행. `/api/vlm/config/`의 각 `*_classifier.permanent_failure` 필드로 운영자가 확인 후 환경 수정 (예: `pip install sentencepiece protobuf`) + 프로세스 재시작 필요.

### MJPEG live preview 튜닝 (A1 / A2 / A3)

- **A1 WIDTH 다운샘플**: `FRONTEND_MJPEG_WIDTH`로 미리보기만 리사이즈. YOLO ingest path는 `INGEST_DOWNSAMPLE_HEIGHT` 기준 독립. `cv2.resize`는 새 ndarray라 원본 ring buffer 안 건드림.
- **A2 Frame dedup**: `stream_manager.get_frame()`이 같은 ndarray 참조를 돌려주면 encode/send 스킵. 정적 장면 CPU ≈ 0.
- **A3 Burst FPS**: 감지 직후 `INFERENCE_ACTIVE_BURST_SEC` 동안 `FRONTEND_MJPEG_BURST_FPS`로 승격. 평시 3 fps 절전, 이벤트 순간 12 fps smooth.

### FFmpeg encoder (libx264 vs NVENC)

`FFMPEG_ENCODER=h264_nvenc`로 전환하면 NVIDIA NVENC GPU 인코더 사용. 이벤트당 ~0.9s → ~0.09s (10×), CPU -1.5 vCPU. T4 NVENC session slot 3개 제한이므로 `POSTPROCESS_WORKERS=2` + `CLIP_SAVE_MAX_CONCURRENT=1` 유지 권장.

### Overlay static-cache

`CandidateClipBuilder._make_context_overlay_applier()`가 skeleton/ROI primitives를 첫 프레임에서 한 번만 렌더하고 boolean mask로 이후 프레임에 copy 적용. skeleton_summary가 static이라 63프레임 × 12 persons × 17 keypoints 재그리기 대신 mask 1회 + memcpy 63회. 이벤트 burst CPU -0.25 vCPU.

### Episode 1-per-emission + quiet period

`EpisodeManager.update()`는 scenario 조건이 **지속되는 한** 같은 episode로 취급해 단 1회 emit합니다. `max_gap_sec=6` 이상 non-detected 프레임이 지나야 episode가 종료되고 새 episode가 형성됩니다. 즉:

- 계산대에 직원이 서 있어서 cash conf=0.70이 계속 찍혀도 → 최초 1회만 emit
- 손님이 떠나고 6초 후 다시 와서 cash trigger → 새 episode 1회 emit
- 이전 구조(cooldown_sec 경과 시 재emit)의 "20초 metronome" 문제 해소

Suppression 이유는 proposal 메타데이터의 `metadata.episode.suppressed`에서 확인:

```
same_episode_already_emitted  : 같은 episode에서 이미 한 번 emit됨
warming_up                    : min_hits / min_duration 미달
cooldown                      : 직전 emission 후 cooldown_sec 안 지남
```

### FP subtype labeling

`/monitor/labeling`에서 decline(FP) 판정 시 7종 error_type 중 **필수** 선택:

| 키 | error_type | 의미 |
|---|---|---|
| `Q` | `phone_or_device` | 스마트폰/기기 오인 |
| `W` | `receipt_or_paper` | 영수증/종이 오인 |
| `E` | `card` | 카드 오인 |
| `R` | `empty_scene` | 빈 장면 |
| `T` | `staff_only` | 직원만 있음 |
| `Y` | `no_transfer` | 전달 동작 없음 |
| `U` | `other` | 기타 |

UI (`errSection.missing` 스타일 경고) + 서버 400 (`vlm_api.py /feedback/`) 양쪽에서 차단합니다. `R`은 FP 선택 시엔 error_type으로, 평시엔 overlay 토글로 동작 (context-aware binding). FP / Unclear에는 note 한 줄 이상도 필수.

### Tier-2 SigLIP 게이트 (suppression)

Tier-1 pose는 recall 우선이라 손목 ROI + 다중 인원이면 거의 다 candidate로 승격됩니다. SigLIP은 본래 +0.20 보너스만 더했지만 그것만으론 게이트 역할을 못 했습니다 (실제 SigLIP cash=0.028 케이스도 Tier-1 conf 0.60으로 통과 → Gemini 호출 → 비용 낭비).

새 동작 (`temporal_engine._cash_result` / `_violence_result`):

```
if semantic_score > 0 and semantic_score < SIGLIP_GATE:
    score = 0      # Tier-1 pose가 뭐라 하든 override
    reasons += [f"siglip_<scenario>_gate<{GATE:.2f}"]
```

`local_prefilter`에 다음 필드 노출:

```json
{
  "passed": false,
  "score": 0.0,
  "pre_gate_score": 0.60,
  "siglip_cash_score": 0.028,
  "siglip_gate_triggered": true,
  "reasons": ["wrist_inside_cashier_zone", "handover_like_pose", "siglip_cash_gate<0.30"]
}
```

`/monitor/gemini-logs`의 "Tier-2 (SigLIP)" 컬럼에 GATE 배지로 시각 표시.

### Skeleton overlay no-ghost policy

이전: `skeleton_summary`가 admission 순간 snapshot이라 63프레임 overlay clip에 같은 위치 좌표를 매번 그림. 사람 움직이면 skeleton만 고정 → 잔상.

해결: 기본 `V3_OVERLAY_SKELETON_FRAMES=0` → context_overlay에 **cashier ROI 빨간 polygon만** 그림. ROI는 진짜 static이라 ghost 없음. Gemini는 raw video + ROI hint로 판단.

opt-in: `V3_OVERLAY_SKELETON_FRAMES=3`으로 설정하면 첫 3프레임만 arms-only skeleton snapshot 표시 (나머지는 ROI only). 단순화된 skeleton draw:

```
ARM_KEYPOINTS = (5, 6, 7, 8, 9, 10)        # shoulder-elbow-wrist 양쪽
ARM_LIMBS     = ((5,7),(7,9),(6,8),(8,10)) # 어깨→팔꿈치→손목 4 라인만
- person bbox 제거
- "SoM #N" / "LW" / "RW" 텍스트 라벨 제거
- 손목 dot만 초록색
```

### Spot interruption flush-now 경로

`tools/spot_interruption_watcher.py`가 AWS IMDS `spot/instance-action` 감지 시:

1. `POST http://127.0.0.1:8000/api/vlm/flush-now/?include_today=true` 호출
   - FlushWorker가 `get_pending_dates(include_today=True)`로 오늘 이벤트까지 DB server로 drain
   - 단, `archive_date()`는 today에 대해 스킵 → 당일 재시작 시 남은 이벤트 계속 flush 가능
2. `systemctl stop vlm-model vlm-db vlm-frontend` — TimeoutStopSec=120 graceful shutdown
3. `FlushWorker.stop(final_flush=True)` — 종료 직전 한 번 더 flush
4. grace 90초 대기 후 watcher exit (systemd가 인스턴스 교체)
5. 새 스팟 인스턴스에서 AUTO_RESTORE_CAMERAS_ON_BOOT=true로 자동 복원

### Overlay는 1개만 저장, fps는 낮춰서

기본 evidence는 원본 raw clip과 하나의 context overlay입니다. overlay는 clip_fps의 1/3 속도로 샘플링해 CPU 렌더 비용을 절감합니다. Gemini는 4~5 fps overlay로도 ROI + skeleton 힌트를 충분히 받습니다.

### Overlay는 증거가 아님

빨간 cashier box는 거래가 일어나는 관심 영역을 표시하는 힌트입니다. Gemini prompt와 parser는 빨간 박스, skeleton line, SoM marker 자체를 사건 증거로 쓰지 않도록 제한합니다.

### NVDEC + Ingest 다운샘플 필수

g4dn.2xlarge 8 vCPU에서 4 camera를 돌리려면 RTSP 디코드가 반드시 NVDEC여야 합니다. 이것 하나가 안 되면 CPU 5~6 vCPU가 추가로 소모되어 8 vCPU를 초과합니다. 이미지는 720p로 다운샘플해서 overlay 렌더와 ring buffer RAM을 절반으로 줄입니다.

### Feedback은 보존

human feedback이 달린 이벤트는 retention cleanup에서 보호합니다. 이 데이터는 나중에 threshold 조정, prompt 회귀 분석, multi-head SigLIP 학습 (`docs/FUTURE_MULTIHEAD_SIGLIP_PLAN.md`)에 쓰입니다.

### Multi-head 공유 backbone 학습은 보류

Fire-Detection-Siglip2와 Human-Action-Recognition은 둘 다 `siglip2-base-patch16-224` 기반이지만 공식 체크포인트는 full fine-tune이라 backbone 가중치가 서로 다릅니다. 따라서 `image_embed` 한 번에 두 head를 돌리는 "shared backbone"은 공개 체크포인트로는 불가능. 호텔 GT 300+건이 쌓인 후 `docs/FUTURE_MULTIHEAD_SIGLIP_PLAN.md` 에 따라 학습 예정입니다.

## 20. Quick Checklist

```text
[ ] .env에 GEMINI_API_KEY 설정
[ ] models/yolo26s-pose.pt 존재
[ ] models/yolov26_fire_detection_best.pt 존재
[ ] YOLO26_DETECT_WEIGHTS는 비워둠 (4cam 배포)
[ ] HF 사전 캐시 완료 (prithivMLmods/Fire-Detection-Siglip2, Human-Action-Recognition, google/siglip2-base-patch16-224)
[ ] V3_SCENARIOS=cash,fire,violence
[ ] V3_FIRE_CLASSIFIER_ENABLED=true
[ ] V3_ACTION_CLASSIFIER_ENABLED=true
[ ] V3_CLIP_ARTIFACT_MODE=minimal
[ ] ALLOW_CPU_FALLBACK=false
[ ] GEMINI_MAX_CONCURRENT=2
[ ] RTSP_HWACCEL_DECODER=h264_cuvid
[ ] INGEST_DOWNSAMPLE_HEIGHT=720
[ ] FFMPEG_PRESET=ultrafast
[ ] V3_OVERLAY_FPS_DIVISOR=1
[ ] BASE_FPS=3.0
[ ] POSTPROCESS_WORKERS=2
[ ] FLUSH_INTERVAL_SEC=120
[ ] Smoke test 통과
[ ] NVDEC 작동 확인 (nvidia-smi decoder util > 0%)
[ ] /monitor/adhoc에서 camera 4대 추가
[ ] cashier zone + exchange_band polygon 저장
[ ] event_cooldown_sec=20 확인
[ ] Gemini logs에서 raw/context_overlay 확인
[ ] labeling에서 TP/FP feedback 저장
[ ] (스팟) systemd units + spot watcher 실행 중
```

## 21. Time / Locale Handling (KST)

운영자가 모두 한국 시간 기준으로 보기 때문에, 타임스탬프와 디렉터리 버킷팅을 KST로 통일합니다.

### 21.1 Why naive ISO was unreliable

- 호스트의 `.env` 에 `TZ=Asia/Seoul` 이 있으나 Windows Python은 IANA tz 이름을 해석하지 못해 `datetime.now()` 가 UTC로 떨어지는 케이스가 있었습니다.
- 그 결과 `at` 필드가 환경마다 KST naive 또는 UTC naive로 섞여 저장됐고, 같은 디렉터리(`data/events/YYYYMMDD/`) 안에 두 포맷이 공존했습니다.
- `event_id` (`ev_<epoch_ms>_<scenario>_<camera>`) 의 epoch ms 부분은 항상 UTC ms 로 기록되므로 시각의 단일 진실 소스(SoT) 입니다.

### 21.2 Backend rules

- `model_server/vlm_api.py` : `KST = timezone(timedelta(hours=9))` 와 `now_kst_iso()` 헬퍼 추가. 모든 이벤트 `at`, `saved_at`, `server_time`, `server_start_time` 에 KST tz-aware ISO (`...+09:00`) 사용.
- `model_server/local_storage.py` : `_now_kst()` 가 모든 디렉터리 strftime 과 `saved_at` 에 사용됨. 신규 이벤트는 KST 일자 디렉터리에 들어갑니다.
- `model_server/event_postprocessor.py` : dead-letter `at` 도 KST.
- 기존 데이터에는 UTC 버킷팅 잔재가 남아있어, 조회시 KST 일자 ↔ UTC 디렉터리 매핑 보정이 필요합니다 (아래 18.4).

### 21.3 Frontend rules

- 표시 / 정렬은 항상 `event_id` epoch ms 기반. 표시는 `toLocaleString("ko-KR", { timeZone: "Asia/Seoul" })` 로 강제.
- 폴백 파서 (`parseAt`) 는 naive ISO 를 KST(`+09:00`) 로 anchor 하고 6 자리 마이크로초를 ms 로 정규화.
- 적용 파일: `frontend_server/templates/vlm_pipeline/gemini_logs.html`, `v3_proposal_logs.html`, `labeling.html`.
- 날짜 드롭다운: KST 기준 "오늘"을 항상 선택지로 노출, `YYYY-MM-DD` 형식.

### 21.4 KST-day query API

```text
GET /api/vlm/events/?kst_date=YYYYMMDD&limit=N
```

- `kst_date` 가 지정되면 백엔드가 다음을 수행합니다:
  1. KST 일자 → epoch ms 윈도우 [start, end) 계산
  2. 해당 KST 일자가 걸치는 두 UTC 버킷 (`D-1`, `D`) 을 모두 읽음
  3. 파일명의 `ev_<ms>_` 토큰으로 윈도우를 벗어나는 파일은 디스크 read 전에 스킵
  4. 남은 결과를 epoch 기준 desc 정렬 후 limit 만큼 반환
- 기존 `date=YYYYMMDD` (UTC 버킷 단위) 도 그대로 동작합니다. 운영 화면은 `kst_date` 를 사용해야 사람이 보는 날짜와 결과 집합이 일치합니다.

### 21.5 Performance notes (gemini-logs)

- 폴링 간격 6s → 12s, 백그라운드 탭에서는 폴링 정지, 포커스 복귀 시 즉시 새로고침.
- 동시 fetch 가드 (`_evLoading`) 로 폴링이 누적되지 않도록 함.
- 최상단 `event_id` 가 동일하면 테이블 재렌더 스킵 (DOM 작업 절약).
- realtime 기본 `limit` 400 → 150 (1.5s 단축). 날짜 선택 시 `limit=3000` 으로 KST 하루 전체.
- 디렉터리당 수천 개 JSON 이 쌓여도 파일명 epoch 사전 필터로 디스크 I/O 가 KST 윈도우에 비례.

