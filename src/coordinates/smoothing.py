from __future__ import annotations

import numpy as np
from loguru import logger
from scipy.signal import savgol_filter


def clip_impossible_jumps(
    positions: list[tuple[float, float]],
    max_speed_mps: float,
    fps: float,
) -> list[tuple[float, float]]:
    """Replace positions that imply impossible speed with interpolated values.

    If the displacement between consecutive frames implies a speed exceeding
    max_speed_mps, the offending position is replaced by linear interpolation
    from the last valid position to the next valid position.

    Args:
        positions: List of (x, y) court coordinates.
        max_speed_mps: Maximum plausible speed in meters per second.
        fps: Video frame rate.

    Returns:
        Cleaned list of positions with impossible jumps interpolated.
    """
    if len(positions) < 2:
        return list(positions)

    max_dist_per_frame = max_speed_mps / fps
    result = list(positions)
    valid_mask = [True] * len(positions)

    # Mark invalid positions (jumps that exceed max speed)
    for i in range(1, len(positions)):
        if not valid_mask[i - 1]:
            # Find last valid position before i
            last_valid_idx = None
            for j in range(i - 1, -1, -1):
                if valid_mask[j]:
                    last_valid_idx = j
                    break
            if last_valid_idx is None:
                continue
            ref_pos = result[last_valid_idx]
            frames_gap = i - last_valid_idx
        else:
            ref_pos = result[i - 1]
            last_valid_idx = i - 1
            frames_gap = 1

        dx = positions[i][0] - ref_pos[0]
        dy = positions[i][1] - ref_pos[1]
        dist = np.sqrt(dx * dx + dy * dy)

        if dist > max_dist_per_frame * frames_gap:
            valid_mask[i] = False

    # Interpolate invalid positions
    for i in range(len(positions)):
        if valid_mask[i]:
            continue

        # Find previous valid
        prev_idx = None
        for j in range(i - 1, -1, -1):
            if valid_mask[j]:
                prev_idx = j
                break

        # Find next valid
        next_idx = None
        for j in range(i + 1, len(positions)):
            if valid_mask[j]:
                next_idx = j
                break

        if prev_idx is not None and next_idx is not None:
            # Linear interpolation
            t = (i - prev_idx) / (next_idx - prev_idx)
            x = result[prev_idx][0] + t * (result[next_idx][0] - result[prev_idx][0])
            y = result[prev_idx][1] + t * (result[next_idx][1] - result[prev_idx][1])
            result[i] = (x, y)
        elif prev_idx is not None:
            result[i] = result[prev_idx]
        elif next_idx is not None:
            result[i] = result[next_idx]
        # else: leave as-is (no valid neighbors)

    return result


def smooth_trajectory(
    positions: list[tuple[float, float]],
    method: str = "savgol",
    window_frames: int = 7,
    polyorder: int = 3,
    max_speed_mps: float = 8.0,
    fps: float = 30.0,
) -> list[tuple[float, float]]:
    """Smooth court_xy trajectory.

    Pipeline:
    1. Clip impossible jumps (speed > max_speed_mps between consecutive frames)
    2. Apply Savitzky-Golay filter (or pass-through if too few points)
    3. Clip to court bounds [0, 10] x [0, 20]

    Args:
        positions: List of (x, y) court coordinates.
        method: Smoothing method ("savgol" supported).
        window_frames: Window length for Savitzky-Golay filter.
        polyorder: Polynomial order for Savitzky-Golay filter.
        max_speed_mps: Maximum plausible speed for jump clipping.
        fps: Video frame rate.

    Returns:
        Smoothed list of (x, y) court coordinates.
    """
    if len(positions) < 2:
        return list(positions)

    # Step 1: Clip impossible jumps
    clipped = clip_impossible_jumps(positions, max_speed_mps, fps)

    # Step 2: Apply smoothing filter
    xs = np.array([p[0] for p in clipped], dtype=np.float64)
    ys = np.array([p[1] for p in clipped], dtype=np.float64)

    if method == "savgol" and len(clipped) >= window_frames:
        # Ensure window_frames is odd
        win = window_frames if window_frames % 2 == 1 else window_frames + 1
        # Ensure polyorder < window length
        poly = min(polyorder, win - 1)

        xs = savgol_filter(xs, win, poly)
        ys = savgol_filter(ys, win, poly)
    else:
        if method == "savgol" and len(clipped) < window_frames:
            logger.debug(
                "Not enough points ({}) for savgol window ({}), skipping smoothing",
                len(clipped),
                window_frames,
            )

    # Step 3: Clip to court bounds
    xs = np.clip(xs, 0.0, 10.0)
    ys = np.clip(ys, 0.0, 20.0)

    return [(float(xs[i]), float(ys[i])) for i in range(len(xs))]
