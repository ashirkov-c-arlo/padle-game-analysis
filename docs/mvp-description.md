# MVP Description

## Purpose

The MVP analyzes a single-camera padel match video and produces structured match data plus visual outputs. It focuses on a practical 2D baseline:

- Court registration from visible court lines.
- Player detection, tracking, semantic identity assignment, and court-coordinate analytics.
- Ball detection, image-space tracking, and event candidates.
- Scoreboard region detection, OCR or VLM-based score extraction, and score stabilization.
- CSV, JSON, JSONL, annotated video, minimap video, and HTML dashboard outputs.

The MVP does not estimate full 3D camera parameters or true 3D ball position. Court coordinates are based on a floor-plane homography, so player footpoints can be projected reliably when calibration succeeds. Ball projection is only a proxy unless the ball is close to the floor plane, such as at bounce points.

## Runtime Architecture

The main entry point is `scripts/run_mvp.py`. It runs the pipeline as a staged process over one input video:

1. Load configuration with `src.config.loader.load_config()`.
2. Read video metadata with `src.video_io.reader.get_video_info()`.
3. Initialize `CourtGeometry2D` from court dimensions in `configs/default.yaml`.
4. Register the court with `src.calibration.court_registration.register_court()`.
5. Decode the video once with `src.video_io.single_pass.run_single_pass()` and dispatch frames to player, ball, and scoreboard processors.
6. Track player detections with `src.tracking.tracker.track_players()`.
7. Project player tracks to court coordinates and compute player analytics.
8. Post-process ball detections with a Kalman tracker.
9. Detect ball event candidates and rally tempo metrics.
10. Export structured files.
11. Render annotated video, minimap video, and dashboard.

The single-pass stage is the main runtime optimization. It avoids decoding the same video separately for player detection, ball detection, and scoreboard extraction. Each processor implements:

- `should_process(frame_idx)`: decides whether it needs the frame.
- `process_frame(frame, frame_idx)`: consumes the frame and stores internal results.
- `finalize()`: runs processor-level post-processing after decoding.

## Core Data Contracts

The MVP uses Pydantic models from `src/schemas.py` as the main internal contracts:

| Contract | Purpose |
| --- | --- |
| `CourtGeometry2D` | Court size, net line, service lines, and zone boundaries. |
| `CourtRegistration2D` | Registration mode, homographies, reprojection error, inlier count, and confidence. |
| `PlayerDetection` | Per-frame person bbox, class id, and confidence. |
| `PlayerTrack` | Semantic player id, frame list, bboxes, confidences, and team label. |
| `PlayerMetricFrame` | Per-player per-frame court position, speed, zone, confidence, and quality flag. |
| `BallDetection2D` | Per-frame ball detection in image coordinates. |
| `BallTrack2D` | Ball position, velocity, confidence, state, and gap metadata. |
| `BallEventCandidate` | Bounce, touch, serve, or net-crossing candidate with evidence. |
| `RallyTempoMetric` | Rally duration, estimated shot count, and touch intervals. |
| `ScoreboardState` | Detected scoreboard ROI, raw text, parsed set score, parsed game score, and confidence. |

The pipeline is resilient to missing stages. For example, if court registration fails, the registration mode becomes `pixel_only`. Player bbox tracks, image-space ball tracks, scoreboard states, and summary output can still be produced, but court-coordinate analytics are skipped.

## Models Used

### Court Line Model

DeepLSD is the primary court-line detector. It is configured under `calibration.deeplsd` and uses weights from `data/models/deeplsd_md.tar` by default. The calibration module can also use a Hough/LSD-style fallback when `calibration.method` is set to `hough`.

### Player Detector

Players are detected with Ultralytics YOLO. The default model in `configs/default.yaml` is `yolo11l`. The detector:

- Lazy-loads weights on first use.
- Stores downloaded weights under `models.cache_dir`.
- Uses class id `0` for persons.
- Runs on CUDA when available, otherwise CPU.
- Keeps only the top configured detections per frame.

### Player Tracker

Player tracking uses a local ByteTrack-style tracker, not an external neural ReID model. The tracker combines:

- Constant-velocity Kalman filtering over bbox state `[cx, cy, aspect_ratio, height, vx, vy, va, vh]`.
- IoU matching.
- Hungarian assignment.
- Separate high-confidence and low-confidence association passes.
- Lost-track recovery and pruning by `track_buffer`.

### Ball Detector

Ball detection uses WASB-SBDT when configured and available. The default model path is `data/models/wasb_tennis_best.pth.tar`. The local adapter loads the WASB-SBDT repository, builds the configured WASB model, processes a temporal frame clip, predicts heatmaps, extracts the best candidate, and maps it back to original image coordinates.

If WASB-SBDT cannot be loaded and `fallback_to_heuristic` is true, the MVP uses a heuristic detector:

- MOG2 background subtraction for motion.
- HSV masks for yellow/green and white balls.
- Contour area filtering.
- Circularity filtering.
- Confidence from size, shape, and color or motion evidence.

### Scoreboard Models

Scoreboard extraction has two paths:

- Optional Bedrock VLM path through `src.scoreboard.vlm_detector`, configured by `scoreboard.vlm`.
- PaddleOCR fallback through `src.scoreboard.ocr_engine.ScoreboardOCR`.

The VLM path sends sampled full frames and asks for JSON containing the scoreboard bbox and score. The OCR path crops a detected ROI, preprocesses it, reads text, parses the score, and validates score transitions.

## Algorithms

### Court Registration

The court registration stage estimates a 2D floor homography:

1. Sample stable frames at `calibration.frame_sample_interval_s`.
2. Detect line segments using DeepLSD or Hough.
3. Filter lines using image geometry and a floor-color mask.
4. Cluster filtered lines by orientation.
5. Match detected lines to a known 10m x 20m padel-court template.
6. Fit `homography_image_to_court` with RANSAC.
7. Invert it to get `homography_court_to_image`.
8. Validate against projected template lines.
9. Accept the best frame only if the reprojection error is within `max_reprojection_error_px`.

Successful registration returns `mode="floor_homography"`. Failure returns `mode="pixel_only"` with zero confidence. The homography is only valid for the court floor plane.

### Player Detection And ROI Filtering

Player detection runs on every frame during the single-pass decode. After detection, the MVP filters player bboxes by court ROI when a homography exists:

1. Compute the bbox bottom-center point as the player footpoint.
2. Project the footpoint to court coordinates.
3. Keep detections inside the court bounds with a small meter-level margin.

In `pixel_only` mode, no court ROI filter is applied because court coordinates are unavailable.

### Player Tracking And Identity Assignment

Tracking consumes frame-indexed `PlayerDetection` lists:

1. Remove bboxes below `min_box_area`.
2. Split detections into high-confidence and low-confidence groups.
3. Predict active and lost track states.
4. Match active tracks to high-confidence detections by IoU and Hungarian assignment.
5. Match remaining active tracks to low-confidence detections with a looser threshold.
6. Match unmatched high-confidence detections to lost tracks for recovery.
7. Create new tracks from unmatched high-confidence detections.
8. Drop lost tracks that exceed `track_buffer`.
9. Return accumulated track histories.

Semantic player ids are assigned after tracking:

- If a homography exists, median track footpoints are projected to court coordinates.
- If no homography exists, median normalized pixel footpoints are used.
- Team is assigned from median y position relative to the net or image half.
- Side is assigned from median x position.
- Final ids are `near_left`, `near_right`, `far_left`, and `far_right`.

Identity stabilization removes short tracks, resolves duplicate semantic ids, and merges compatible fragments separated by less than about two seconds.

### Player Court Coordinates And Analytics

Player analytics only run when `registration.mode == "floor_homography"`:

1. Project each player bbox bottom-center through `homography_image_to_court`.
2. Clip impossible jumps using `smoothing.max_speed_mps`.
3. Apply Savitzky-Golay smoothing only when `smoothing.method == "savgol"`.
4. Clip coordinates to court bounds `[0, 10] x [0, 20]`.
5. Compute distance, speed, acceleration, average speed, and peak speed by finite differences.
6. Classify each position into `net`, `mid`, or `baseline` by distance from the net line.
7. Compute zone time, team formation, and partner spacing.

Current implementation note: the default config sets `smoothing.method: kalman`, but the player smoother currently applies a Savitzky-Golay filter only for `savgol`. With any other method value it still performs jump clipping and bounds clipping, but does not apply an additional player Kalman filter.

### Ball Detection

Ball processing can use a court ROI mask when registration is reliable enough:

1. Project the four court corners to image coordinates.
2. Fill the projected polygon and expand it by a small margin.
3. Disable the mask if registration confidence or mask area fails configured thresholds.
4. Run ball detection on the masked frame.

With WASB-SBDT, the detector:

1. Maintains the required number of recent frames for the model clip.
2. Warps frames to the model input size.
3. Normalizes RGB values with ImageNet-style mean and standard deviation.
4. Runs the WASB model without gradients.
5. Applies sigmoid to model heatmaps.
6. Extracts candidates using connected components or NMS.
7. Selects the candidate with the strongest rank score.
8. Applies the inverse affine transform to recover original image coordinates.

With the heuristic fallback, the detector combines motion, color, area, and circularity evidence. The heuristic is useful as a baseline, but it is less reliable for small, blurred, occluded, or fast-moving balls.

### Ball Tracking

Ball detections are post-processed by `BallKalmanTracker`:

- State: `[x, y, vx, vy]` in image pixels.
- Measurement: `[x, y]` from ball detection.
- Motion model: constant velocity.
- Measurement noise increases when detection confidence is low.
- Optional Mahalanobis gating rejects detections that are too far from prediction.
- Process noise can increase with gap length and current speed.

The tracker emits:

- `detected` or `tracked` frames when a detection is accepted.
- `interpolated` frames for gaps up to `ball_tracking.max_gap_frames`.
- `missing` frames after the allowed gap is exceeded.

Ball velocity is exported in pixels per second. Court-coordinate ball tracking is not part of the current main pipeline.

### Ball Event Candidates

Ball events are rule-based candidates, not final referee-grade decisions:

- Bounce candidate: image-space vertical velocity changes from downward to upward, optionally with a speed drop.
- Touch candidate: the ball changes direction by at least about 30 degrees and is near a player position.
- Net crossing candidate: with a homography, projected ball y crosses the court net line at `y=10m`.
- Rally tempo: touch candidates are grouped into rallies; gaps longer than four seconds start a new rally.

Event candidates include confidence and evidence fields so downstream tools can inspect why they were emitted.

### Scoreboard Extraction

Scoreboard processing samples every `scoreboard.sample_interval_s` seconds:

1. Collect the first sampled frames for ROI detection.
2. Prefer VLM detection when enabled and available.
3. Fall back to CV ROI detection if VLM is unavailable or fails.
4. Detect CV ROI candidates from stable light panels or text-like edge regions near the frame edges.
5. Crop the ROI and run PaddleOCR when OCR is needed.
6. Parse score text with regex-based patterns for set scores and game scores.
7. Validate updates with a tennis/padel score FSM.
8. Stabilize results so a new score must appear consistently before it is accepted.

This design handles common broadcast overlays without requiring a custom scoreboard training set, but it depends on readable scoreboard frames and stable overlay placement.

## Outputs

For an input video `<name>.mp4`, outputs are written under `<output_dir>/<name>/`.

Structured outputs:

- `court_geometry.json`
- `court_registration.json`
- `tracks.csv`
- `metrics.csv`
- `ball_tracks.csv`
- `ball_detections.jsonl`
- `ball_event_candidates.jsonl`
- `scoreboard.csv`
- `rally_metrics.csv`
- `summary.json`

Visual outputs:

- `annotated.mp4`: original video with court overlay, player boxes, speed labels, ball marker, and scoreboard information.
- `minimap.mp4`: top-down court view with player positions and trails when court-coordinate metrics exist.
- `dashboard.html`: self-contained HTML dashboard built from exported files.

## Visual Gap Filling

Annotated video and minimap can smooth short player-track gaps visually.

- Enabled by `export.video.max_player_gap_fill_frames` in `configs/default.yaml`.
- Default: `15` frames.
- Method: linear interpolation by frame index.
- Bboxes in annotated video are linearly interpolated between neighboring real track observations.
- Player markers in minimap are linearly interpolated between neighboring real `PlayerMetricFrame.court_xy` observations.
- Interpolated visuals use reduced confidence styling.
- Source data stays unchanged: no synthetic rows are added to tracks, metrics, exports, or analytics.

## Degradation Behavior

The MVP is designed to keep producing partial outputs when one subsystem fails:

- Court registration failure: output remains `pixel_only`; player bbox tracking and image-space ball tracking can continue.
- Missing player detections: tracking can interpolate internally for short gaps, but long gaps become missing track segments.
- Missing ball detections: the ball Kalman tracker predicts through short gaps and then marks the ball as missing.
- Scoreboard failure: scoreboard states may be empty or low-confidence, while other analytics continue.
- Export failure: `run_mvp.py` still writes a fallback `summary.json` when possible.

## Known Limitations

- A single monocular camera cannot recover true 3D ball height without additional assumptions or calibration.
- The floor homography is valid for player footpoints and court-floor contacts, not airborne ball positions.
- Court-coordinate analytics depend on successful line detection and template fitting.
- Player identity assignment is geometric only; it does not use jersey color, face, pose, or ReID embeddings.
- ByteTrack can swap identities during severe occlusions or close crossings.
- The ball detector is sensitive to motion blur, occlusions, camera resolution, bitrate, and domain shift from tennis to padel.
- Scoreboard extraction depends on readable overlays and can fail on unusual broadcast styles.
- Minimap ball rendering is not currently available because main-pipeline ball tracks are image-space only.
