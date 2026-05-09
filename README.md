# padel-cv

Computer vision pipeline for analyzing padel tennis game videos. Processes video through court calibration, player detection and tracking, ball tracking, scoreboard OCR, and analytics to produce annotated videos and structured data exports.

## Features

- **Court Calibration** - DeepLSD line detection + template fitting for court registration
- **Player Detection & Tracking** - YOLO11 + ByteTrack for up to 4 players
- **Ball Detection & Tracking** - WASB-SBDT model with Kalman filtering and event detection
- **Scoreboard OCR** - PaddleOCR-based score extraction
- **Analytics** - Speed, acceleration, zone occupancy, formations, rally metrics
- **Export** - Annotated video, minimap overlay, JSON, and CSV outputs

## Requirements

- Python 3.10 or 3.11
- NVIDIA GPU with CUDA 12.6+ (or AMD GPU with ROCm)
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

### 1. Clone the repository

```bash
git clone --recurse-submodules https://github.com/<your-org>/padle-game-analysis.git
cd padle-game-analysis
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 2. Install dependencies

```bash
uv sync --dev
```

This installs all Python dependencies including PyTorch with CUDA 12.6 support, ultralytics (YOLO11), PaddleOCR, and DeepLSD.

### 3. Download external models

Create the models directory:

```bash
mkdir -p data/models
```

**DeepLSD weights** (court line detection):

```bash
wget -O data/models/deeplsd_md.tar https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_md.tar
```

**WASB-SBDT tennis model** (ball detection):

Download the WASB Tennis model from [Google Drive](https://drive.google.com/file/d/14AeyIOCQ2UaQmbZLNQJa1H_eSwxUXk7z/view?usp=drive_link) and place it at:

```
data/models/wasb_tennis_best.pth.tar
```

**YOLO11 and PaddleOCR** models are downloaded automatically on first run.

## Usage

### Run the full pipeline

```bash
uv run python scripts/run_mvp.py --video <input.mp4> --config configs/default.yaml --out data/outputs/
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | (required) | Path to input video |
| `--config` | `configs/default.yaml` | Config YAML path |
| `--out` | `data/outputs` | Output directory |
| `--log-level` | `INFO` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |

### Output structure

```
data/outputs/
├── tracks.json          # Player tracks and ball detections
├── metrics.csv          # Per-frame kinematics
├── annotated.mp4        # Video with overlays
└── minimap.mp4          # Top-down court minimap
```

## Configuration

All pipeline parameters are in [configs/default.yaml](configs/default.yaml). Key sections:

- `court` - Court dimensions (standard padel: 10m x 20m)
- `detection` - YOLO model variant and confidence threshold
- `tracking` - ByteTrack hyperparameters
- `calibration` - DeepLSD settings and weights path
- `ball_tracking` - WASB-SBDT model path and Kalman filter tuning
- `scoreboard` - OCR engine and sampling interval
- `export` - Output formats and video options

## Development

### Dev Container (recommended)

The project includes dev container configurations for both NVIDIA (CUDA) and AMD (ROCm) GPUs in `.devcontainer/`. Open in VS Code or GitHub Codespaces for a pre-configured environment.

### Run tests

```bash
uv run pytest
```

### Lint

```bash
uv run ruff check src/ scripts/ tests/
```

## Project Structure

```
src/
├── calibration/       # Court registration (DeepLSD, template fitting)
├── detection/         # Player detection (YOLO11)
├── tracking/          # Player tracking (ByteTrack, identity assignment)
├── ball_tracking/     # Ball detection (WASB-SBDT, Kalman filtering)
├── coordinates/       # Homography projection, smoothing
├── analytics/         # Kinematics, zone analysis, rally metrics
├── scoreboard/        # Scoreboard detection, OCR, state machine
├── export/            # JSON, CSV, video writers
├── visualization/     # Overlay annotations, minimap
├── video_io/          # Video I/O utilities
├── config/            # Configuration loader
└── schemas.py         # Pydantic data models
scripts/
├── run_mvp.py         # Full pipeline entry point
└── evaluate_*.py      # Component evaluation scripts
configs/
└── default.yaml       # Default pipeline configuration
WASB-SBDT/             # Ball detection submodule
```
