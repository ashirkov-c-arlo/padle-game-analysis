from __future__ import annotations

import cv2
import numpy as np
from loguru import logger


def detect_lines_deeplsd(frame: np.ndarray) -> np.ndarray:
    """Run DeepLSD on a frame, return line segments as Nx4 (x1,y1,x2,y2).

    Falls back to Hough transform if DeepLSD is unavailable or fails.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    # Try DeepLSD first
    try:
        lines = _run_deeplsd(gray)
        if lines is not None and len(lines) > 0:
            logger.debug("DeepLSD detected {} lines", len(lines))
            return lines
    except ImportError:
        logger.warning("DeepLSD not installed — falling back to Hough transform")
    except Exception as e:
        logger.warning("DeepLSD failed: {} — falling back to Hough transform", e)

    # Fallback: Hough transform
    return detect_lines_hough(frame)


def _run_deeplsd(gray: np.ndarray) -> np.ndarray | None:
    """Attempt to run DeepLSD. Returns Nx4 array or None. Raises ImportError if not installed."""
    import torch  # noqa: F401 — let ImportError propagate
    from deeplsd.models.deeplsd_inference import DeepLSD  # type: ignore  # noqa: F401

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize to [0, 1] float
    img_tensor = torch.tensor(gray[None, None], dtype=torch.float32, device=device) / 255.0

    model = _get_deeplsd_model(device)
    with torch.no_grad():
        output = model({"image": img_tensor})

    lines = output["lines"][0].cpu().numpy()  # Nx2x2
    # Reshape to Nx4 (x1, y1, x2, y2)
    if len(lines) == 0:
        return None
    return lines.reshape(-1, 4)


_deeplsd_model_cache: dict = {}


def _get_deeplsd_model(device):
    """Lazy-load and cache DeepLSD model."""
    from deeplsd.models.deeplsd_inference import DeepLSD  # type: ignore

    key = str(device)
    if key not in _deeplsd_model_cache:
        conf = {
            "detect_lines": True,
            "line_detection_params": {
                "merge": False,
                "filtering": True,
                "grad_thresh": 3,
                "grad_nfa": True,
            },
        }
        model = DeepLSD(conf)
        model = model.to(device)
        model.eval()
        _deeplsd_model_cache[key] = model

    return _deeplsd_model_cache[key]


def _run_opencv_lsd(gray: np.ndarray) -> np.ndarray | None:
    """Run OpenCV's Line Segment Detector. Returns Nx4 array or None."""
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines, _, _, _ = lsd.detect(gray)
    if lines is None or len(lines) == 0:
        return None
    # lines shape is (N, 1, 4), reshape to (N, 4)
    return lines.reshape(-1, 4)


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
    return lines.reshape(-1, 4).astype(np.float64)
