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

---

## YOLO11m/l Player Track Count

Goal: Keep all four padel players when switching between YOLO11 model sizes.

### Plan

- [x] Inspect detector, tracker, config, and relevant tests for model-size-sensitive filtering.
- [x] Fix the tracker threshold path so full pipeline config values are honored.
- [x] Align the default new-track threshold with the detector/tracking acceptance threshold.
- [x] Add focused regression tests for nested ByteTrack config and a lower-confidence fourth player.
- [x] Run focused tests and document the result.

### Review

- Root cause: the pipeline passes full config with ByteTrack settings under `tracking.bytetrack`, but `ByteTracker`
  only read root-level `bytetrack`, so real config threshold changes were ignored.
- Lowered default `tracking.bytetrack.new_track_thresh` from `0.6` to `0.5` so a player accepted by detection at
  the default threshold can start a track.
- Verification: `uv run --no-project --with numpy --with scipy --with pydantic --with loguru --with pytest pytest tests/test_tracking.py -q`
  passed (`34 passed`).
- Verification: `uv run --no-project --with ruff ruff check src/tracking/bytetrack.py src/tracking/tracker.py tests/test_tracking.py`
  passed.

### Follow-up Plan

- [x] Inspect why false tracks can displace real player tracks after lowering thresholds.
- [x] Prefer stable, high-quality tracks before assigning the two near/two far player slots.
- [x] Tighten court ROI filtering defaults so out-of-court false detections are actually filtered when homography exists.
- [x] Add focused regressions for false-track displacement and ROI filtering.
- [x] Run focused tests and update this review.

### Follow-up Review

- Set `tracking.bytetrack.new_track_thresh` back to `0.5`; `0.7` still allows low-confidence YOLO boxes at the
  detector stage when `detection.confidence_threshold` is `0.5`, but blocks real players below `0.7` from starting tracks.
- Identity assignment now chooses the strongest tracks per team by duration and confidence before assigning player IDs,
  so a short false track cannot steal a player ID before stabilization.
- Homography ROI filtering now defaults to a 1-meter court-space margin instead of 50 meters, so obvious out-of-court
  false detections are filtered when court registration is available.
- Verification: `uv run --no-project --with numpy --with scipy --with pydantic --with loguru --with pytest --with ultralytics pytest tests/test_tracking.py tests/test_detection.py -q`
  passed (`54 passed`).
- Verification: `uv run --no-project --with ruff ruff check src/tracking/identity.py src/detection/roi_filter.py tests/test_tracking.py tests/test_detection.py`
  passed.

---

## PaddleOCR Installation

Goal: Install PaddleOCR for the existing scoreboard OCR path with the smallest dependency and code changes needed.

### Plan

- [x] Add PaddleOCR runtime dependencies to project metadata using the CPU PaddlePaddle inference engine.
- [x] Refresh `uv.lock` through `uv` so installed versions are recorded by the project.
- [x] Verify `paddle` and `paddleocr` import successfully in this Python 3.12 environment.
- [x] Check the installed PaddleOCR API against `src/scoreboard/ocr_engine.py`; update only that adapter if the current package API requires it.
- [x] Run focused scoreboard OCR tests and ruff on touched files.

### Review

- Added `paddlepaddle==3.3.1` and `paddleocr==3.5.0` to `pyproject.toml`; `uv.lock` now records the matching packages.
- Installed the same PaddleOCR packages into `.venv` with `uv pip install --python .venv/bin/python ...` because full `uv add` sync hit the existing ROCm Torch/Python 3.12 wheel mismatch in the lock.
- Updated `ScoreboardOCR` for PaddleOCR 3.x: removed deprecated constructor arguments, uses `predict()`, converts preprocessed grayscale crops back to BGR, and parses 3.x results.
- Disabled PaddleOCR MKL-DNN in the adapter after a live CPU smoke test hit a Paddle runtime `NotImplementedError` with MKL-DNN enabled.
- Verification: `.venv/bin/python` imports `paddle 3.3.1` and `paddleocr 3.5.0`; `ScoreboardOCR` initializes with engine `paddleocr`.
- Verification: `.venv/bin/python -m pytest tests/test_scoreboard.py -q` passed (`38 passed`).
- Verification: `.venv/bin/python -m ruff check src/scoreboard/ocr_engine.py tests/test_scoreboard.py` passed.
