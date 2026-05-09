# MVP Implementation Plan for Claude Code Agents

## MVP Scope

1. Automatic 2D court registration from `.mp4` video only
2. DeepLSD-based court line detection
3. Known 2D metric court geometry and floor homography
4. YOLO11 player detection
5. ByteTrack player tracking
6. Player analytics in court coordinates
7. 2D ball tracking in image coordinates
8. Scoreboard OCR
9. JSON/CSV exports
10. Debug visualization overlay and minimap

## Required Input

The MVP accepts only:

```text
1. .mp4 match video
```

No camera calibration file is available.
No manual runtime calibration is allowed.
No camera intrinsics, extrinsics, or 3D pose are estimated in MVP.

Main command:

```bash
python scripts/run_mvp.py \
  --video data/raw/match.mp4 \
  --config configs/default.yaml \
  --out data/outputs/match_001
```

## Known Court Geometry

Assume standard padel court geometry is known from config.

Coordinate system:

```text
x: left to right, meters, 0..10
y: near baseline to far baseline, meters, 0..20
floor plane only: z is not used for MVP analytics
net line: y=10
```

Geometry config:

```yaml
court:
  width_m: 10.0
  length_m: 20.0
  net_y_m: 10.0
  service_line_offset_from_net_m: 6.95
  net_height_center_m: 0.88
  net_height_posts_m: 0.92
  lines:
    - near_baseline
    - far_baseline
    - left_sideline
    - right_sideline
    - net_line
    - near_service_line
    - far_service_line
    - near_center_service_line
    - far_center_service_line
```

`net_height_*` is stored as known metadata only. It is not used for 3D projection or height-aware analytics in MVP.

## Target Pipeline

```text
input .mp4 video
  -> frame sampler
  -> DeepLSD court-line detection
  -> automatic line filtering and clustering
  -> known 2D court template fitting
  -> estimate floor homography image <-> court_xy
  -> validate 2D registration quality
  -> YOLO11 player detection
  -> ByteTrack player tracking
  -> bbox footpoint pixel -> court_xy
  -> player trajectory smoothing
  -> player kinematics and tactical metrics
  -> TrackNetV4-style 2D ball detection
  -> image-space ball tracking, smoothing, short-gap fill
  -> ball-derived proxy metrics
  -> scoreboard OCR
  -> score FSM
  -> JSON/CSV export
  -> annotated video and minimap
```

Ball tracking is image-space first. Use the floor homography only for optional approximate floor projection and court ROI masking.

## Registration Modes

The court registration module must output one of these modes:

```text
floor_homography
  DeepLSD/template fit succeeds.
  image <-> court_xy homography is valid.
  Player metric analytics are enabled.
  Optional ball floor projection is allowed with approximate flag.

pixel_only
  Court registration fails.
  Run player detection/tracking, 2D ball tracking, and scoreboard OCR only.
  Do not export metric distance/speed/spacing analytics.
```

For MVP metric analytics, target `floor_homography`.

## Repository Structure

```text
padel-cv/
  src/
    config/
    video_io/
    court_geometry/
    calibration/
    detection/
    tracking/
    coordinates/
    analytics/
    ball_tracking/
    scoreboard/
    export/
    visualization/
  scripts/
    run_mvp.py
    extract_frames.py
    train_scoreboard_detector.py
    train_ball_tracker.py
    evaluate_calibration.py
    evaluate_tracking.py
    evaluate_ball_tracking.py
  data/
    raw/
    frames/
    labels/
    models/
    outputs/
  tests/
  configs/
    default.yaml
    bytetrack.yaml
  README.md
```

## Shared Interfaces

Create shared schemas first. Do not let agents change these independently.

Required schemas:

```text
CourtGeometry2D
CourtRegistration2D
FrameResult
PlayerDetection
PlayerTrack
PlayerMetricFrame
BallDetection2D
BallTrack2D
BallMetricFrame
BallEventCandidate
RallyTempoMetric
ServePlacementMetric
ShotDirectionProxyMetric
ShotDepthProxyMetric
PlayerPressureMetric
ScoreboardState
MVPOutput
```

Core output files:

```text
court_geometry.json
court_registration.json
frames.jsonl
tracks.csv
metrics.csv
ball_tracks.csv
ball_detections.jsonl
ball_metrics.csv
rally_metrics.csv
ball_event_candidates.jsonl
scoreboard.csv
summary.json
bounce_heatmap.png
annotated.mp4
minimap.mp4
```

## Agent 1 - Project Skeleton and Config

Deliver:

- repository structure
- dependencies
- config loader
- logging
- shared schemas
- main CLI entrypoint
- basic test runner

Files:

```text
src/config/
src/schemas.py
scripts/run_mvp.py
configs/default.yaml
README.md
```

Acceptance:

```text
run_mvp.py accepts --video, --config, --out
output folder is created
empty summary.json can be written
court geometry config loads successfully
```

## Agent 2 - Video-Only 2D Court Registration

Constraint: automatic only. No manual keypoints. No camera calibration input. No 3D camera/court estimation.

Main method:

```text
DeepLSD -> line filtering -> line clustering -> known 2D padel template fit -> floor homography
```

Fallback:

```text
Canny + Hough only when DeepLSD fails to produce a usable template fit
```

Implement:

```text
sample stable frames every N seconds
run DeepLSD on sampled frames
filter court-like line segments
cluster lines by court-axis orientation
match detected lines to known 2D court geometry
fit 10m x 20m padel floor template with RANSAC/robust optimization
estimate image_to_court and court_to_image homographies
aggregate best registration across sampled frames
validate line/template reprojection quality
save court_geometry.json and court_registration.json
```

Registration output:

```json
{
  "mode": "floor_homography",
  "court_size_m": [10.0, 20.0],
  "coordinate_system": "x_left_to_right_y_near_to_far",
  "homography_image_to_court": [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
  "homography_court_to_image": [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
  "quality": {
    "mean_line_reprojection_error_px": 0.0,
    "matched_model_lines": 0,
    "confidence": 0.0,
    "method": "DeepLSD"
  }
}
```

Acceptance:

```text
DeepLSD is the default line detector
Canny/Hough path is used only after DeepLSD failure
2D court overlay aligns on sampled frames
homography maps player footpoints into plausible court_xy coordinates
mean line reprojection error < 10 px when measurable
registration is stable across sampled frames
```

Labeling use:

```text
Use Gemini 3, Qwen2.5-VL-7B-Instruct, or InternVL2.5-8B only for offline labels/evaluation.
Runtime registration remains automatic.
```

## Agent 3 - YOLO11 Player Detection

Implement:

```text
YOLO11 person detector wrapper
batch/video inference
confidence filtering
court ROI filtering when homography is available
NMS cleanup
per-frame detection export
```

Start with COCO-pretrained YOLO11 person class.
Fine-tune only if baseline quality is weak.

Optional labeling:

```text
label 500-1000 frames with player boxes
use VLMs for pre-labeling
review validation frames manually or semi-automatically
train YOLO11 player-only detector
```

Output:

```json
{
  "frame": 123,
  "detections": [
    {
      "bbox_xyxy": [412, 280, 500, 690],
      "class": "player",
      "confidence": 0.91
    }
  ]
}
```

Acceptance:

```text
finds 4 players in normal rally frames
few false positives outside court area
usable speed on FullHD input
```

## Agent 4 - ByteTrack Tracking and Identity Stabilization

Implement:

```text
YOLO11 detections -> ByteTrack
persistent track IDs
short-track removal
lost-track recovery
court ROI filtering when homography is available
```

Use court side and left/right ordering for MVP identity:

```text
near_left
near_right
far_left
far_right
```

Do not use shirt color as the primary team assignment method.

Post-processing:

```text
remove tracks shorter than 1 sec
merge fragments by spatial continuity in court coordinates when available
fallback to image-space continuity in pixel_only mode
reject tracks whose footpoints are outside valid court floor area when homography is available
limit permanent active player identities to 4 where possible
```

Acceptance:

```text
low ID switches on 1-2 minute clip
stable four player tracks during normal rallies
lost players recover without permanent duplicate identities
```

## Agent 5 - Court Coordinates and Player Analytics

Use the floor homography as the MVP source of truth for player coordinates.

Pipeline:

```text
bbox bottom-center pixel
  -> image_to_court homography
  -> court_xy = [x, y]
  -> smooth trajectory
  -> calculate metrics
```

Smoothing:

```text
Savitzky-Golay or Kalman filter
clip impossible jumps
mark low-confidence frames
```

Metrics:

```text
distance covered
speed
acceleration
time in net/mid/baseline zones
partner spacing
team centroid
formation state
court heatmap
```

Zone model per half:

```text
net zone:      0.0-3.5 m from net
mid zone:      3.5-6.95 m from net
baseline zone: 6.95-10.0 m from net
```

Formation states:

```text
both_net
both_mid
both_baseline
one_up_one_back
split_unknown
```

Output:

```json
{
  "frame": 123,
  "registration_mode": "floor_homography",
  "players": [
    {
      "player_id": "near_left",
      "court_xy": [2.4, 17.1],
      "speed_mps": 1.8,
      "zone": "baseline",
      "track_confidence": 0.88,
      "metric_quality": "estimated"
    }
  ],
  "teams": {
    "near": {
      "partner_spacing_m": 4.2,
      "formation": "both_baseline"
    }
  }
}
```

Metric rules:

```text
floor_homography: export all MVP floor metrics with metric_quality=estimated
pixel_only: do not export metric distance/speed/spacing; export tracks, 2D ball tracks, and OCR only
```

Acceptance:

```text
projected player positions mostly stay inside court floor polygon
speed values are physically plausible after smoothing/clipping
heatmaps and zone percentages are generated when registration mode is floor_homography
```

## Agent 6 - 2D Ball Tracking

Goal: produce frame-level 2D ball coordinates in image space. Do not infer bounces, touches, serves, walls, net events, or point outcomes in MVP.

Chosen implementation:

```text
primary: TrackNetV4-style temporal heatmap detector
```

Pipeline:

```text
extract frames at native FPS
optionally crop/mask to court ROI when homography is available
run temporal ball detector on frame windows, e.g. t-2..t+2
extract heatmap peak -> candidate ball image_xy
apply confidence threshold
run image-space Kalman filter
fill short gaps up to 10 frames
reject impossible jumps
write image-space ball track
if floor_homography is available, compute floor_projection_xy_approx
write ball track outputs for downstream metrics
```

Projection rules:

```text
image_xy:
  source of truth for 2D ball tracking

floor_projection_xy_approx:
  available only when registration_mode=floor_homography
  used only for proxy metrics and visualizations
  not true airborne ball position

pixel_only:
  export ball_tracks.csv and tracking quality only
  disable bounce heatmap, serve placement, shot direction/depth, and pressure metrics

Training/labeling plan:

```text
start with 2,000-5,000 labeled frames
label ball center as x_px, y_px plus visibility state
include hard negatives: shoes, racket heads, lights, logos, reflections
use Gemini 3, Qwen2.5-VL-7B-Instruct, or InternVL2.5-8B for pre-labeling only
review validation set separately
fine-tune on the target camera style before relying on output
```

Visibility labels:

```text
visible
blurred
occluded
not_visible
```

Ball detection output:

```json
{
  "frame": 123,
  "time_s": 4.1,
  "ball": {
    "image_xy": [812.4, 366.7],
    "confidence": 0.76,
    "visibility": "visible",
    "source": "tracknetv4_heatmap"
  }
}
```

Ball track output:

```json
{
  "frame": 123,
  "time_s": 4.1,
  "ball_track": {
    "image_xy": [812.4, 366.7],
    "velocity_px_s": [221.0, -84.0],
    "confidence": 0.72,
    "state": "tracked",
    "interpolated": false,
    "gap_len": 0
  }
}
```

CSV columns:

```text
frame,time_s,x_px,y_px,floor_x_m_approx,floor_y_m_approx,vx_px_s,vy_px_s,confidence,visibility,state,interpolated,gap_len,source
```

States:

```text
detected
tracked
interpolated
missing
```

Acceptance:

```text
ball marker follows visible ball on annotated video
short occlusions are bridged without large jumps
long occlusions become missing, not hallucinated tracks
ball_tracks.csv and ball_detections.jsonl are exported
tracker works even when court registration fails
no MVP metric depends on unverified ball court coordinates
```

## Agent 6B - Ball-Derived MVP Proxy Metrics

Use ball tracks, player tracks, scoreboard state, known 2D court geometry, and floor homography when available. Compute exactly these six MVP metrics:

```text
1. bounce heatmap
2. serve placement
3. shot direction proxy
4. shot depth proxy
5. rally tempo
6. player pressure from ball
```

All outputs are proxy/candidate metrics. If registration mode is `pixel_only`, export only image-space tracking quality and rally tempo candidates.

Candidate event types:

```text
bounce_candidate
  trajectory kink + velocity change + plausible court-floor projection

touch_candidate
  ball-player proximity + trajectory direction/speed change

serve_candidate
  scoreboard new-point context + server position + first touch_candidate + diagonal trajectory + first bounce_candidate

net_crossing_candidate
  projected trajectory crosses y=10m net line
```

These are internal candidate events for metrics. Do not expose them as confirmed calls.

### 1. Bounce Heatmap

Use `bounce_candidate` events projected to court coordinates.

Method:

```text
trajectory kink detection
image-space vertical velocity reversal
speed drop or direction change
court projection inside valid floor polygon
confidence threshold
```

Outputs:

```text
bounce_heatmap.png
bounce_count
bounce_zone_distribution
corner_bounce_pct
deep_bounce_pct
service_box_bounce_pct
```

### 2. Serve Placement

Use scoreboard point-start context and early rally ball events.

Method:

```text
scoreboard FSM indicates new point
identify likely server from side/position
find first touch_candidate
find first bounce_candidate in diagonal opponent service box
classify landing zone: wide / body / T_or_center / unknown
```

Rules:

```text
padel serve uses bounce/drop + underhand contact
no tennis-style toss logic
low-confidence serves are classified as unknown
```

Outputs:

```text
serve_count
wide_serve_pct
body_serve_pct
T_or_center_serve_pct
unknown_serve_pct
serve_depth_distribution
```

### 3. Shot Direction Proxy

Use outgoing projected ball vector after `touch_candidate`.

Classes:

```text
cross_court
down_the_line
middle
unknown
```

Outputs:

```text
cross_court_pct
down_the_line_pct
middle_pct
unknown_direction_pct
shot_direction_distribution_per_player
shot_direction_distribution_per_team
```

### 4. Shot Depth Proxy

Use projected bounce candidate or reliable projected endpoint in opponent half.

Classes:

```text
short: landing/bounce in net zone
mid: landing/bounce in mid zone
deep: landing/bounce in baseline zone
unknown: insufficient ball confidence or no reliable projection
```

Outputs:

```text
short_pct
mid_pct
deep_pct
unknown_depth_pct
avg_projected_landing_depth_m
shot_depth_distribution_per_player
shot_depth_distribution_per_team
```

### 5. Rally Tempo

Use scoreboard state, ball activity, and touch candidates.

Outputs:

```text
rally_duration_s
ball_in_play_duration_s
estimated_shots_per_rally
avg_time_between_touch_candidates_s
median_time_between_touch_candidates_s
shots_per_minute
longest_rally_estimated_shots
```

If scoreboard OCR is unavailable or low-confidence, use ball activity gaps as fallback rally segmentation and mark `rally_source=ball_activity`.

### 6. Player Pressure From Ball

Estimate received-ball difficulty using projected ball path/bounce and defender positions.

Method:

```text
for each outgoing shot proxy:
  estimate projected target/bounce location
  find nearest defending player
  compute defender distance to projected bounce/path
  compute time available until bounce or next touch_candidate
  combine distance, time, depth, and corner proximity into pressure score
```

Outputs:

```text
defender_distance_to_projected_bounce_m
time_available_to_receiver_s
high_pressure_shot_count
pressure_score_per_team
pressure_score_per_player_received
```

Simple pressure rule:

```text
high pressure if:
  projected bounce is deep or near corner
  nearest defender is far from projected bounce/path
  available time is short
  receiving team starts from defensive or wrong-side position
```

Ball event output:

```json
{
  "frame": 18420,
  "time_s": 614.0,
  "event_type": "bounce_candidate",
  "image_xy": [812.4, 366.7],
  "court_xy_approx": [3.4, 15.9],
  "confidence": 0.71,
  "projection_quality": "approximate",
  "evidence": {
    "trajectory_kink_deg": 34.0,
    "speed_change_ratio": 0.62
  }
}
```

Metric output:

```json
{
  "frame": 18420,
  "metric_type": "shot_direction_proxy",
  "player_id": "near_left",
  "team_id": "near",
  "court_xy_projected": [3.2, 16.8],
  "value": "cross_court",
  "confidence": 0.68,
  "quality": "proxy"
}
```

Output files:

```text
ball_tracks.csv
ball_detections.jsonl
ball_event_candidates.jsonl
ball_metrics.csv
rally_metrics.csv
bounce_heatmap.png
```

States:

```text
detected
tracked
interpolated
missing
```

Acceptance:

```text
ball marker follows visible ball on annotated video
short occlusions are bridged without large jumps
long occlusions become missing, not hallucinated tracks
ball_tracks.csv and ball_detections.jsonl are exported
bounce_heatmap is generated when floor_homography is available
serve placement is exported for scoreboard-detected points when confidence is sufficient
shot direction/depth proxies are exported with unknown class fallback
rally tempo metrics are exported per rally/point segment
pressure metrics are exported only when player tracks and ball candidates overlap reliably
all ball-derived outputs include confidence and metric_quality=proxy
```

## Agent 7 - Scoreboard OCR

Scoreboard OCR is independent of court registration and ball tracking.

Implement:

```text
scoreboard region detection
scoreboard crop stabilization
OCR
score parser
score FSM validation
score timeline export
```

MVP region detection:

```text
sample frames
find high-contrast text-like regions near frame edges
cluster stable regions over time
select candidate scoreboard ROI
```

If automatic ROI is weak:

```text
train YOLO11-n scoreboard detector
50-100 labeled frames
one class: scoreboard
```

OCR:

```text
primary: PaddleOCR
fallback for hard crops: Gemini 3, Qwen2.5-VL-7B-Instruct, or InternVL2.5-8B
```

Score FSM:

```text
0 -> 15 -> 30 -> 40 -> game
40 -> AD -> game
40 -> 40
tie-break numeric states
set/game counters
reject impossible OCR jumps
```

Output:

```json
{
  "frame": 1200,
  "scoreboard": {
    "raw_text": "6 4 | 30 15",
    "parsed": {
      "sets": [[6, 4]],
      "game_score": ["30", "15"]
    },
    "confidence": 0.82
  }
}
```

Acceptance:

```text
stable scoreboard crop
OCR changes only when score changes
FSM suppresses obvious OCR noise
scoreboard.csv is exported
```

## Agent 8 - Visualization and Exports

Implement:

```text
annotated video overlay
2D court overlay
2D ball track overlay
top-down minimap video
CSV exports
JSONL exports
summary report
```

Overlay must show:

```text
projected 2D court floor lines
player boxes
track IDs
footpoints
2D ball marker, trail, and confidence
court-coordinate minimap when homography is available
speed labels when metric coordinates are available
scoreboard OCR result
formation state when metric coordinates are available
registration mode and confidence
```

Exports:

```text
court_geometry.json
court_registration.json
frames.jsonl
tracks.csv
metrics.csv
ball_tracks.csv
ball_detections.jsonl
scoreboard.csv
summary.json
annotated.mp4
minimap.mp4
```

Summary example:

```json
{
  "duration_s": 642.2,
  "registration_mode": "floor_homography",
  "registration_confidence": 0.82,
  "ball_tracking": {
    "visible_or_tracked_frame_pct": 0.58,
    "interpolated_frame_pct": 0.07,
    "missing_frame_pct": 0.35,
    "model": "tracknetv4_heatmap"
  },
  "players": {
    "near_left": {
      "distance_m": 812.5,
      "avg_speed_mps": 1.26,
      "max_speed_mps": 5.12,
      "net_zone_pct": 0.21,
      "baseline_zone_pct": 0.57,
      "metric_quality": "estimated"
    }
  },
  "teams": {
    "near": {
      "avg_partner_spacing_m": 4.8,
      "both_net_pct": 0.18,
      "both_baseline_pct": 0.46
    }
  }
}
```

Acceptance:

```text
all required files are created
annotated video is readable
2D court overlay aligns with visible court lines when registration succeeds
2D ball overlay is visible when ball is detected/tracked
minimap matches tracked player positions when registration succeeds
summary metrics are non-empty unless registration mode is pixel_only
```

## Agent 9 - Tests and Evaluation

Implement tests for:

```text
court geometry config loading
DeepLSD line detection interface
DeepLSD failure -> Canny/Hough fallback
line clustering
2D template fitting
homography transform roundtrip
court bounds validation
track smoothing
zone assignment
formation classifier
2D ball detection schema
2D ball Kalman tracker
short-gap interpolation
long-gap missing-state behavior
score FSM transitions
export schema validity
```

Evaluation commands:

```bash
python scripts/evaluate_calibration.py \
  --registration data/outputs/match_001/court_registration.json \
  --frames data/frames

python scripts/evaluate_tracking.py \
  --pred data/outputs/match_001/tracks.csv \
  --labels data/labels/tracks.json

python scripts/evaluate_ball_tracking.py \
  --pred data/outputs/match_001/ball_tracks.csv \
  --labels data/labels/ball_centers.jsonl
```

Minimum quality gates:

```text
Registration:
  DeepLSD used by default
  Canny/Hough used only after DeepLSD failure
  2D court overlay acceptable on sampled frames
  line reprojection error < 10 px when measurable
  output mode is floor_homography for MVP metric analytics

Player tracking:
  4 player identities stable on normal rally segments
  no permanent duplicate tracks for the same player

2D ball tracking:
  ball coordinates exported in image pixels
  short gaps <= 10 frames may be interpolated
  long gaps are marked missing
  no event or score logic depends on ball track in MVP

Analytics:
  distance and speed generated only when registration mode permits metric analytics
  impossible speeds removed or flagged

Scoreboard:
  score timeline generated
  impossible score transitions rejected

Runtime:
  full pipeline runs end-to-end from one command using .mp4 only
```

## Parallel Implementation Order

```text
1. Agent 1 creates skeleton, schemas, config, CLI.
2. Agents 2, 3, 6, and 7 work independently:
   - video-only DeepLSD 2D court registration
   - YOLO11 player detection
   - TrackNetV4-style 2D ball tracking
   - scoreboard OCR
3. Agent 4 integrates player detection with ByteTrack.
4. Agent 5 integrates player tracking with court coordinates and metrics.
5. Agent 8 builds visualization and exports.
6. Agent 9 adds tests/evaluation and fixes integration failures.
```

## Branches or Worktrees

```text
agent/skeleton
agent/video-only-2d-registration
agent/player-detection
agent/player-tracking
agent/player-analytics
agent/ball-tracking
agent/scoreboard
agent/export
agent/tests
```

Each agent must report:

```text
implemented files
public functions/classes
expected inputs/outputs
known limitations
unit tests added
manual run command
```

Do not change shared interfaces without updating:

```text
src/schemas.py
configs/default.yaml
README.md
```

## Definition of Done

This command:

```bash
python scripts/run_mvp.py \
  --video input.mp4 \
  --out outputs/match_001
```

must produce:

```text
court_geometry.json
court_registration.json
frames.jsonl
tracks.csv
metrics.csv
ball_tracks.csv
ball_detections.jsonl
scoreboard.csv
summary.json
annotated.mp4
minimap.mp4
```

MVP is complete when one fixed-camera FullHD padel video produces automatic 2D court registration, stable player tracking, estimated court-coordinate analytics, 2D ball tracking in image space, scoreboard OCR timeline, and visual debug output without any camera calibration input or manual runtime calibration.
