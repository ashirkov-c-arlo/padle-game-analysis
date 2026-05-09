# MVP Implementation Progress

## Agent 1: Project Skeleton

- [x] pyproject.toml with all dependencies
- [x] Directory structure (src/, scripts/, data/, tests/, configs/, tasks/)
- [x] configs/default.yaml - full court geometry and pipeline config
- [x] configs/bytetrack.yaml - ByteTrack parameters
- [x] src/config/loader.py - YAML config loading with merge
- [x] src/schemas.py - All Pydantic models for shared interfaces
- [x] scripts/run_mvp.py - Click CLI entry point
- [x] scripts/ - All placeholder scripts
- [x] tests/test_schemas.py - Schema instantiation tests
- [x] tests/test_config.py - Config loading tests
- [x] .gitignore - Standard Python ignores
- [x] Data directory placeholders

## Agent 2: Court Detection & Calibration

- [ ] src/court_geometry/ implementation
- [ ] src/calibration/ implementation
- [ ] DeepLSD line detection
- [ ] Homography estimation

## Agent 3: Player Detection & Tracking

- [ ] src/detection/ - YOLO player detection
- [ ] src/tracking/ - ByteTrack integration
- [ ] src/coordinates/ - Court coordinate mapping

## Agent 4: Ball Tracking & Events

- [ ] src/ball_tracking/ - TrackNet integration
- [ ] Ball event detection (bounces, touches)
- [ ] Kalman filter smoothing

## Agent 5: Analytics & Export

- [ ] src/analytics/ - Metric computation
- [ ] src/scoreboard/ - OCR integration
- [ ] src/export/ - JSON/CSV export
- [ ] src/visualization/ - Annotated video + minimap

---

## Single-Pass Architecture Refactor

Goal: Eliminate redundant video decode passes. Currently video is decoded 3+ times
(player detection, ball detection, annotated export) + seek for calibration/OCR.
Replace with a single sequential decode where each frame is dispatched to all consumers.

### Plan

- [ ] Create `src/video_io/single_pass.py` with `FrameProcessor` protocol and `run_single_pass` orchestrator
- [ ] Create `src/detection/player_processor.py` — wraps `PlayerDetector.detect_frame` as a FrameProcessor
- [ ] Create `src/ball_tracking/ball_processor.py` — wraps `BallDetector.detect_frame` as a FrameProcessor
- [ ] Create `src/scoreboard/scoreboard_processor.py` — wraps scoreboard OCR as a sampled FrameProcessor
- [ ] Refactor `run_mvp.py` stages 5, 8, 10 to use single-pass instead of separate video reads
- [ ] Keep calibration as pre-pass (needs only ~20 frames via seek, runs before everything else)
- [ ] Keep annotated video export as post-pass (needs computed results, reads video one more time)
- [ ] Verify pipeline produces same results
