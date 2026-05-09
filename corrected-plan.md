# Overall Plan

## Phase 1 - Court Calibration

- Main: manual 4-8 court keypoints + homography from image to court floor.
- Auto refinement: DeepLSD or Hough lines + template fit.
- Add `CourtModel`: floor, net line, service boxes, glass/wall/mesh zones.
- Output: homography `H`, reprojection error, overlay preview.
- Do calibration once per fixed camera; revalidate periodically.

## Phase 2 - Player Tracking and Court Analytics - MVP Core

### Detection

- Main: YOLO11 or RT-DETR person detector.
- Fallback: YOLO11-Pose if ankle keypoints are useful.

### Tracking

- Main: ByteTrack or BoT-SORT.
- Do not force exactly 4 active tracks every frame; allow lost/recovered states.
- Remove short false tracks and fix ID switches manually if needed.

### Ground Point

Priority order:

1. ankle midpoint from pose;
2. bottom of segmentation mask;
3. bbox bottom-center fallback.

```text
ground_point_pixel -> H -> court_xy
```

Smooth `court_xy` with Kalman or Savitzky-Golay filtering.

### Metrics

- distance covered;
- average/peak speed with outlier clipping;
- acceleration with smoothing;
- player heatmaps;
- time in net/mid/baseline zones;
- partner spacing;
- formation: both-net, both-baseline, one-up-one-back, split;
- court dominance: net-control percentage;
- coverage gaps inside each half.

### Team Assignment

- Use court side first.
- Use clothing color only as an identity-continuity cue, not as the main team signal.

## Phase 3 - Optional Scoreboard OCR

Only implement if the video has a visible scoreboard overlay.

- Region: fixed crop first; YOLO11-n only if scoreboard location varies.
- OCR: PaddleOCR.
- Add a scoring FSM to reject impossible score transitions.

## Phase 4 - Ball Tracking and Events - Phase 2, Not MVP

### Ball Detection

- Main: WASB-SBDT or TrackNetV4.
- Expect padel-specific fine-tuning.

### Trajectory

- Use confidence-weighted Kalman filtering.
- Interpolate only short gaps.
- Store `image_xy` and `floor_projection_xy` separately.

Important correction:

```text
ball_pixel -> H -> court_xy
```

This is valid only for bounce/floor events. For airborne ball positions, it is only a floor projection, not true 3D ball position.

### Events

- `bounce_candidate`: trajectory kink + speed change + near-floor evidence.
- `touch_candidate`: ball-player proximity + trajectory/speed change.
- `net_crossing_candidate`: trajectory crosses net line.
- `wall_contact_candidate`: reversal near projected wall/glass/mesh zone.
- `out_candidate`: low confidence unless landing point is reliable.

### Serve Detection

Use a rule-based state machine:

1. pre-point pause or rally end;
2. server behind service line;
3. player stillness;
4. low ball bounce/drop before serve;
5. underhand touch candidate;
6. diagonal trajectory;
7. first bounce in opposite service box.

Do not use a tennis-style toss assumption.

### Shot Analytics

- Direction: cross-court, down-the-line, middle from post-touch trajectory.
- Depth: short, mid, deep from landing/bounce coordinate.
- Serve placement: wide, body, T if first bounce is reliable.

## Phase 5 - Pose and Technique - Defer

- Main: RTMPose-L or YOLO11-Pose.
- Use only coarse features: stance width, body orientation, knee bend, preparation timing.
- Split-step, jump height, and contact posture are proxy metrics only.
- Rename reaction time to movement initiation latency.

## Phase 6 - Shot Type Classification - Defer

Start with coarse classes:

- groundstroke;
- volley;
- overhead;
- lob candidate;
- wall-rebound return candidate.

Avoid fine forehand/backhand/bandeja/vibora classification until labeled data exists.

## Phase 7 - Advanced Analytics - Defer

Keep:

- tactical pattern mining;
- per-rally summaries;
- critical-point metrics if scoreboard OCR works.

Defer:

- winners/errors;
- forced/unforced errors;
- win probability;
- fatigue from pose degradation.

## Output Schema

Per-frame:

```text
frame, time_s
players: id, team, bbox, court_xy, speed, confidence
ball: image_xy, floor_projection_xy, confidence, visible
```

Events:

```text
frame, time_s, event_type, player_id, court_xy, image_xy, confidence, evidence, flags
```

## Evaluation

Track these before adding more features:

- court reprojection error;
- player ID switches;
- player coordinate jitter;
- ball detection recall at 5-10 px;
- bounce/touch/serve precision and recall;
- event timing error in frames.

## MVP Deliverables

- corrected court overlay;
- top-down minimap;
- player trajectories;
- heatmaps;
- distance/speed metrics;
- zone dominance;
- partner spacing;
- formation timeline;
- JSON/CSV export.
