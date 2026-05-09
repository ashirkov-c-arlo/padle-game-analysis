# Overall Plan

## Phase 1 — Court Calibration

| Priority | Approach |
|---|---|
| Main approach | DeepLSD → ориентации → template fit |
| Best speed | Hough Lines + Template Matching |
| Best accuracy | PnLCalib с переобучением |

## Phase 2 — Player Detection, Tracking, Coordinate Analytics

### 1. Detection

| Priority | Approach |
|---|---|
| Main approach | RF-DETR / RT-DETR |
| Best speed | YOLO11 |
| Best accuracy | Co-DETR / Co-DINO |

### 2. Tracking

| Priority | Approach |
|---|---|
| Main approach | BoT-SORT-ReID |
| Best speed | ByteTrack |
| Post-processing | фильтрация коротких треков (< 1 sec), NMS по IoU для дублей, ограничение до 4 активных треков |

### 3. Team / Player Assignment

Кластеризация по доминантному цвету формы:

- KMeans, k=2 на HSV crop торса
- court side → team assignment

### 4. Pixel → Court Transform + Smoothing

Pipeline:

```text
ground_point_pixel → H → court_xy
```

Smoothing:

- Savitzky-Golay filter (`scipy`) или 1D Kalman filter для подавления jitter
- Window ~5–7 frames

### 5. Kinematics Engine

На сглаженных court coordinates:

| Metric | Method |
|---|---|
| Distance | cumulative Euclidean displacement per frame |
| Speed | finite difference + median filter |
| Acceleration | second derivative + clipping outliers |
| Sprint detection | speed > threshold configurable, ~4 m/s for padel |

### 6. Zone / Formation / Pair Analytics

Court zone model:

- Net zone: 0–3.5m от сетки
- Mid-court: 3.5–7m
- Baseline: 7–10m — по половине 10m

Analytics:

| Component | Method |
|---|---|
| Formation classifier | rule-based по zone membership пары: both-net, both-baseline, one-up-one-back, split |
| Partner spacing | Euclidean distance между teammates в court coords |
| Coverage gaps | Voronoi decomposition на позициях 2 игроков → uncovered area относительно court half |

## Phase 3 — Scoreboard OCR

### 1. Region Detection

| Priority | Approach |
|---|---|
| Main approach | YOLO11-n trained на 50–100 ручно размеченных кадрах разных broadcast-ов, один bbox class |
| Fallback | Florence-2 (Microsoft 2024) с region grounding промптом |

### 2. OCR

| Priority | Approach |
|---|---|
| Main approach | PaddleOCR |
| Fallback | Qwen2.5-VL-7B-Instruct или InternVL2.5-8B |

## Phase 4 — Ball Detection & Tracking + Serve & Shot Analytics

### 1. Ball Detection

| Priority | Approach |
|---|---|
| Main approach | WASB-SBDT |
| Fallback | TrackNetV4 |

### 2. Trajectory Smoothing & Gap Fill

Main approach:

- confidence-weighted Kalman filter
- cubic spline interpolation для gaps до ~10 frames
- TrackNetV3 даёт per-frame heatmap → extract peak → track

Alternative:

- custom trajectory fitting, parabolic model для airborne segments
- физически обоснованнее, но сложнее

### 3. Ball Ground-Plane Projection

```text
ball_pixel → H → court_xy
```

Constraints:

- корректно только для мяча на уровне корта, bounce points
- Airborne ball: projection даёт approximate court position с confidence flag `airborne=True`

### 4. Bounce Detection

Method:

- trajectory kink detection
- резкое изменение вертикальной компоненты velocity в image space
- ball proximity к court plane, y-coordinate near baseline в image

Аргументация:

- bounce = trajectory inflection + speed drop + near-ground position

### 5. Wall / Glass / Net Contact

Main approach:

- rule-based
- ball trajectory reversal, dx or dy sign change
- near wall/glass boundary, определён в `CourtModel`
- buffer zone ~0.5m

Аргументация:

- wall contact в padel = мяч меняет направление у стены
- geometric rule достаточен

Alternative:

- audio-based detection, wall hit sound
- не всегда доступен audio

### 6. Touch / Hit Candidate Detection

Signals:

- ball-player proximity, court coords
- sudden trajectory change, angle > threshold
- speed change

Output:

```text
touch_candidate(frame, player_id, confidence)
```

### 7. Net Event Detection

Rule:

- ball trajectory crossing net line, y = 10m in court coords
- termination check: ball stops / no continuation

### 8. Out-of-Bounds Candidate

Rule:

- ball landing court_xy outside court boundaries
- с учётом wall play rules padel
- confidence based on landing position accuracy

### 9. Serve Detection

Rule-based state machine:

1. Score change, new point → expect serve
2. Server player behind baseline in service zone
3. Pre-serve stillness, low velocity > 1 sec
4. Ball first appearance / upward motion, toss
5. First touch candidate → diagonal trajectory to opponent service box

Аргументация:

- serve в padel имеет жёсткую структуру: underhand, bounce before hit, diagonal
- хорошо описывается правилами

### 10. Serve Placement Classification

Landing point в service box → classify:

- wide
- body
- T, center

Simple geometric zones на service box coords.

### 11. Shot Direction / Depth Classification

Direction:

- вектор ball trajectory после touch
- angle relative to court axis
- cross-court / down-the-line / middle

Depth:

- landing/bounce y-coordinate в opponent half
- deep >7m / mid / short <3m

### 12. Player-Ball Association

```text
nearest_player(ball_court_xy, time_window)
```

Matching:

- Hungarian matching по distance + time

## Phase 6 — Pose Estimation & Technique Analysis

### 1. Pose Estimation

| Priority | Approach |
|---|---|
| Main approach | RTMPose-L |
| Fallback | YOLO11‑Pose |
| Max-accuracy | Sapiens-pose-0.6B |

### 2. Pose Feature Extraction

Из 17 COCO keypoints:

| Feature | Method |
|---|---|
| Stance width | distance(left_ankle, right_ankle) normalized by hip width |
| Body orientation | shoulder line angle relative to court axis |
| Hip-shoulder separation | angle between hip line и shoulder line |
| Knee bend | angle(hip, knee, ankle) |

### 3. Split-Step Detection

Inputs:

- foot/ankle keypoints
- window around opponent touch candidate

Pattern:

- simultaneous bilateral ankle lift, both feet leave ground
- ankles move up → land → lateral push-off

### 4. Contact Posture & Preparation Timing

Contact posture:

- pose at frame closest to touch_candidate
- extract arm extension
- weight transfer, hip position relative to feet
- racket arm angle proxy, wrist-elbow-shoulder

Preparation timing:

- first significant upper-body rotation before touch_candidate
- time delta

### 5. Jump Detection

Signals:

- vertical displacement of ankle midpoint + hip keypoint
- threshold-based

Approximate height:

- peak ankle displacement × calibration scale factor

### 6. Reaction Time

Rule:

- opponent touch_candidate timestamp → first significant velocity change of current player
- delta = reaction time

## Phase 7 — Shot Type Classification

### 1. Feature Window Construction

Per touch_candidate, extract temporal window ±15 frames:

- Player pose sequence: joint angles, orientations
- Ball trajectory: direction, speed, arc
- Player court zone
- Ball court zone
- Contact height: ball y in image relative to player bbox
- Pre/post bounce/wall events

### 2. Classifier

| Priority | Approach |
|---|---|
| Skeleton-based | SkateFormer |
| Motion-based | VideoMAE V2 or InternVideo2.5 |
| Overall fallback | Qwen2.5-VL-72B или Gemini 2.0 Flash в few-shot — clip + structured prompt c описанием классов |

## Phase 8 — Advanced Analytics

### 1 — Point Outcome Resolution

Fusion:

- rally_end event
- score_change
- last touch
- ball out/net/wall

Output:

- кто выиграл point и почему

Method:

- rule-based FSM, не ML

### 2 — Winners / Errors Classification

Features per ending shot:

- contact zone
- direction
- depth
- speed proxy
- time-to-touch opponent
- opponent position relative to ball
- post-touch ball state

Models:

| Priority | Approach |
|---|---|
| Primary | LightGBM binary winner/error → multi-class error type: forced/unforced/net/out/wall-fault |
| Data | 200–500 labeled rallies достаточно |
| Alternative | XGBoost, близкая accuracy |

### 3 — Tactical Pattern Mining

Main approach:

- N-gram analysis over shot sequences, type + direction + zone → frequent patterns

Аргументация:

- interpretable
- не требует labeled data

Alternative:

- Sequence-to-sequence model для point outcome prediction
- мощнее, но нужен большой dataset

### 4 — Pressure Index & Critical Points

Score FSM → classify point type:

- break point
- set point
- match point

Analysis:

- Player metrics в critical moments vs regular moments → performance delta

### 5 — Win Probability Model

Main approach:

- XGBoost / LightGBM

Features:

- score state
- server/receiver
- formation
- rally length
- recent momentum

Аргументация:

- tabular data → gradient boosting SOTA
- interpretable feature importance

Alternative:

- logistic regression, baseline
- neural network, если достаточно данных

### 6 — Fatigue Curve

Longitudinal tracking по set/match:

- rolling window средних speed
- recovery time
- split-step frequency
- posture metrics

Detection:

- degradation trend
