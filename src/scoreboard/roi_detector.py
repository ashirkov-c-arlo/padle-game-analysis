from __future__ import annotations

import cv2
import numpy as np
from loguru import logger


def detect_scoreboard_roi(
    frames: list[tuple[int, np.ndarray]],
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """
    Find scoreboard region in video frames.

    Strategy:
    1. Look near edges (top/bottom, usually top) of frame
    2. Find high-contrast text-like regions using edge density
    3. Look for regions that are stable across multiple frames
    4. Score candidates by:
       - Consistent position across frames
       - High edge density (text)
       - Rectangular shape
       - Appropriate size (not too large, not too small)

    Returns (x1, y1, x2, y2) or None if no scoreboard found.
    """
    if not frames:
        return None

    h, _ = image_shape
    panel_candidates: list[list[tuple[int, int, int, int]]] = []
    all_candidates: list[list[tuple[int, int, int, int]]] = []

    for _, frame in frames:
        panel_candidates.append(_filter_edge_candidates(find_scoreboard_panel_regions(frame), h))
        candidates = find_text_regions(frame)
        all_candidates.append(_filter_edge_candidates(candidates, h))

    best_panel_roi = _find_stable_region(
        panel_candidates,
        image_shape,
        allow_unstable_fallback=False,
    )
    if best_panel_roi is not None:
        best_panel_roi = _pad_panel_roi(best_panel_roi, image_shape)
        logger.debug(
            "Scoreboard panel ROI detected: x1={}, y1={}, x2={}, y2={}",
            *best_panel_roi,
        )
        return best_panel_roi

    if not all_candidates or all(len(c) == 0 for c in all_candidates):
        logger.debug("No scoreboard candidates found in any frame")
        return None

    # Find regions that appear consistently across frames
    best_roi = _find_stable_region(all_candidates, image_shape)
    if best_roi is not None:
        logger.debug(
            "Scoreboard ROI detected: x1={}, y1={}, x2={}, y2={}",
            *best_roi,
        )
    return best_roi


def find_scoreboard_panel_regions(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Find broadcast-style scoreboard panels near the top/bottom edges.

    Many score overlays are not text-only; they are light rectangular panels with
    dark text and adjacent score boxes. The generic edge detector can merge such
    panels with surrounding audience/court edges, so this looks for solid light
    horizontal panels first.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]
    band_height = max(1, int(h * 0.25))
    bands = [
        (0, hsv[:band_height]),
        (h - band_height, hsv[h - band_height :]),
    ]

    candidates: list[tuple[int, int, int, int]] = []
    for y_offset, band in bands:
        candidates.extend(_find_light_panel_regions(band, y_offset, image_shape=(h, w)))

    return candidates


def find_text_regions(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Find candidate text regions using morphological operations.
    - Convert to gray
    - Apply morphological gradient to highlight text edges
    - Find connected components of appropriate size
    - Group nearby components into text blocks
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Morphological gradient to detect edges (text boundaries)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    # Threshold to get binary edge map
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Dilate to connect nearby text characters into blocks
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    connected = cv2.dilate(binary, connect_kernel, iterations=2)

    # Find contours of text blocks
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    min_area = h * w * 0.001  # At least 0.1% of frame
    max_area = h * w * 0.15   # No more than 15% of frame

    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch

        if area < min_area or area > max_area:
            continue

        # Aspect ratio filter: scoreboards are wider than tall
        aspect = cw / max(ch, 1)
        if aspect < 1.5 or aspect > 15.0:
            continue

        # Check edge density within region — text regions have high density
        roi = binary[y : y + ch, x : x + cw]
        density = np.count_nonzero(roi) / max(area, 1)
        if density < 0.05 or density > 0.8:
            continue

        candidates.append((x, y, x + cw, y + ch))

    return candidates


def _find_light_panel_regions(
    hsv_band: np.ndarray,
    y_offset: int,
    image_shape: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    h, w = image_shape
    _, saturation, value = cv2.split(hsv_band)
    mask = np.where((saturation < 85) & (value > 115), 255, 0).astype(np.uint8)

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(15, int(w * 0.018)), max(3, int(h * 0.004))),
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(15, int(w * 0.013)), max(5, int(h * 0.006))),
    )
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(7, int(w * 0.005)), max(3, int(h * 0.004))),
    )
    panel_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    panel_mask = cv2.morphologyEx(panel_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    panel_mask = cv2.dilate(panel_mask, dilate_kernel, iterations=1)

    contours, _ = cv2.findContours(panel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[int, int, int, int]] = []
    min_width = w * 0.08
    max_width = w * 0.45
    min_height = h * 0.018
    max_height = h * 0.14

    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        aspect = cw / max(ch, 1)
        fill_ratio = cv2.countNonZero(panel_mask[y : y + ch, x : x + cw]) / max(area, 1)

        if cw < min_width or cw > max_width:
            continue
        if ch < min_height or ch > max_height:
            continue
        if aspect < 2.5 or aspect > 16.0:
            continue
        if fill_ratio < 0.35:
            continue

        candidates.append((x, y + y_offset, x + cw, y + y_offset + ch))

    return candidates


def _filter_edge_candidates(
    candidates: list[tuple[int, int, int, int]],
    image_height: int,
) -> list[tuple[int, int, int, int]]:
    edge_candidates = []
    for x1, y1, x2, y2 in candidates:
        cy = (y1 + y2) // 2
        # Scoreboard is typically in top 20% or bottom 20%
        if cy < image_height * 0.20 or cy > image_height * 0.80:
            edge_candidates.append((x1, y1, x2, y2))
    return edge_candidates


def _pad_panel_roi(
    roi: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    h, w = image_shape
    x1, y1, x2, y2 = roi
    roi_w = x2 - x1
    roi_h = y2 - y1

    left_pad = max(8, int(roi_w * 0.03))
    right_pad = max(24, int(roi_w * 0.18))
    top_pad = max(6, int(roi_h * 0.15))
    bottom_pad = max(10, int(roi_h * 0.55))

    return (
        max(0, x1 - left_pad),
        max(0, y1 - top_pad),
        min(w, x2 + right_pad),
        min(h, y2 + bottom_pad),
    )


def _find_stable_region(
    all_candidates: list[list[tuple[int, int, int, int]]],
    image_shape: tuple[int, int],
    allow_unstable_fallback: bool = True,
) -> tuple[int, int, int, int] | None:
    """Find the region that appears most consistently across frames."""
    h, w = image_shape
    tolerance = int(min(h, w) * 0.05)  # 5% position tolerance

    if not all_candidates:
        return None

    # Flatten all candidates with frame counts
    flat_candidates = []
    for frame_cands in all_candidates:
        flat_candidates.extend(frame_cands)

    if not flat_candidates:
        return None

    # Score each candidate by how many frames have a similar region
    best_score = 0
    best_roi = None

    for candidate in flat_candidates:
        score = 0
        for frame_cands in all_candidates:
            for other in frame_cands:
                if _regions_overlap(candidate, other, tolerance):
                    score += 1
                    break
        if score > best_score:
            best_score = score
            best_roi = candidate

    # Require the region to appear in at least 50% of frames
    min_frames = max(1, len(all_candidates) // 2)
    if best_score >= min_frames:
        return best_roi

    if not allow_unstable_fallback:
        return None

    # If no stable region, return the largest candidate from the first frame
    if flat_candidates:
        largest = max(
            flat_candidates,
            key=lambda r: (r[2] - r[0]) * (r[3] - r[1]),
        )
        return largest

    return None


def _regions_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    tolerance: int,
) -> bool:
    """Check if two regions overlap within tolerance."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (
        abs(ax1 - bx1) < tolerance
        and abs(ay1 - by1) < tolerance
        and abs(ax2 - bx2) < tolerance
        and abs(ay2 - by2) < tolerance
    )
