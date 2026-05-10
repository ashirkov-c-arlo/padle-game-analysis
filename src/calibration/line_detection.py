from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

_DEEPLSD_DEFAULT_WEIGHTS_URL = "https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_md.tar"
_DEEPLSD_DEFAULT_WEIGHTS_PATH = "data/models/deeplsd_md.tar"
_DEEPLSD_FALLBACK_REASONS_LOGGED: set[str] = set()


def detect_lines_deeplsd(frame: np.ndarray, config: dict | None = None) -> np.ndarray:
    """Run DeepLSD on a frame, return line segments as Nx4 (x1,y1,x2,y2).

    Falls back to Hough transform if DeepLSD is unavailable or fails.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    # Try DeepLSD first
    try:
        lines = _run_deeplsd(gray, config)
        if lines is not None and len(lines) > 0:
            logger.debug("DeepLSD detected {} lines", len(lines))
            return lines
        logger.debug("DeepLSD returned no lines; falling back to Hough transform")
    except ImportError:
        _log_deeplsd_fallback_once("import", "DeepLSD not available; using Hough line detection")
    except Exception as e:
        _log_deeplsd_fallback_once("runtime", "DeepLSD failed; using Hough line detection")
        logger.opt(exception=e).debug("DeepLSD failure details")

    # Fallback: Hough transform
    logger.debug("Using Hough transform for line detection")
    return detect_lines_hough(frame)


def _log_deeplsd_fallback_once(reason: str, message: str) -> None:
    if reason not in _DEEPLSD_FALLBACK_REASONS_LOGGED:
        logger.warning(message)
        _DEEPLSD_FALLBACK_REASONS_LOGGED.add(reason)


def _run_deeplsd(gray: np.ndarray, config: dict | None = None) -> np.ndarray | None:
    """Attempt to run DeepLSD. Returns Nx4 array or None. Raises ImportError if not installed."""
    import torch  # noqa: F401 — let ImportError propagate
    from deeplsd.models.deeplsd_inference import DeepLSD  # type: ignore  # noqa: F401

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cal_cfg = (config or {}).get("calibration", {}).get("deeplsd", {})
    target_w = cal_cfg.get("target_width", 640)
    target_h = cal_cfg.get("target_height", 480)

    h, w = gray.shape[:2]
    scale_x = w / target_w
    scale_y = h / target_h
    resized = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)

    logger.debug(
        "DeepLSD input: original={}x{}, resized={}x{}, scale_x={:.3f}, scale_y={:.3f}, device={}",
        w, h, target_w, target_h, scale_x, scale_y, device,
    )

    # Normalize to [0, 1] float
    img_tensor = torch.tensor(resized[None, None], dtype=torch.float32, device=device) / 255.0

    model = _get_deeplsd_model(device, config)
    with torch.no_grad():
        output = model({"image": img_tensor})

    df_norm = output["df_norm"][0].cpu().numpy() if "df_norm" in output else None
    if df_norm is not None:
        logger.debug(
            "DeepLSD df_norm: shape={}, range=[{:.3f}, {:.3f}], mean={:.3f}",
            df_norm.shape, df_norm.min(), df_norm.max(), df_norm.mean(),
        )

    raw = output["lines"][0]
    lines = raw.cpu().numpy() if hasattr(raw, "cpu") else np.asarray(raw)  # Nx2x2
    logger.debug("DeepLSD raw lines: count={}", len(lines))

    if len(lines) == 0:
        return None

    # Scale line coordinates back to original resolution
    lines = lines.reshape(-1, 4)
    lines[:, [0, 2]] *= scale_x
    lines[:, [1, 3]] *= scale_y

    return lines


_deeplsd_model_cache: dict = {}


def _ensure_weights(weights_path: str, weights_url: str) -> Path:
    """Download DeepLSD weights if not present. Returns resolved path."""
    import urllib.request

    path = Path(weights_path)
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading DeepLSD weights from {} to {}", weights_url, path)
    urllib.request.urlretrieve(weights_url, str(path))
    logger.info("DeepLSD weights downloaded ({:.1f} MB)", path.stat().st_size / 1e6)
    return path


def _get_deeplsd_model(device, config: dict | None = None):
    """Lazy-load and cache DeepLSD model with pretrained weights."""
    import torch
    from deeplsd.models.deeplsd_inference import DeepLSD  # type: ignore

    key = str(device)
    if key not in _deeplsd_model_cache:
        cal_cfg = (config or {}).get("calibration", {}).get("deeplsd", {})
        weights_path = cal_cfg.get("weights_path", _DEEPLSD_DEFAULT_WEIGHTS_PATH)
        weights_url = cal_cfg.get("weights_url", _DEEPLSD_DEFAULT_WEIGHTS_URL)

        weights_file = _ensure_weights(weights_path, weights_url)

        conf = {
            "detect_lines": True,
            "line_detection_params": {
                "merge": False,
                "filtering": True,
                "grad_thresh": 3,
                "grad_nfa": True,
            },
        }

        net = DeepLSD(conf)
        ckpt = torch.load(str(weights_file), map_location="cpu", weights_only=False)
        net.load_state_dict(ckpt["model"])
        net = net.to(device).eval()
        logger.info("DeepLSD model loaded on {} with weights from {}", device, weights_file)
        _deeplsd_model_cache[key] = net

    return _deeplsd_model_cache[key]


def detect_lines_hough(frame: np.ndarray) -> np.ndarray:
    """Fallback: Canny edge + probabilistic Hough transform. Returns Nx4 array."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Probabilistic Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=50,
        maxLineGap=15,
    )

    if lines is None or len(lines) == 0:
        return np.empty((0, 4), dtype=np.float64)

    # lines shape is (N, 1, 4), reshape to (N, 4)
    result = lines.reshape(-1, 4).astype(np.float64)
    logger.debug("Hough detected {} lines", len(result))
    return result
