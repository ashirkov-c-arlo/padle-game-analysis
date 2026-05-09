from __future__ import annotations

import numpy as np


def compute_kinematics(
    positions: list[tuple[float, float]],
    fps: float,
) -> dict:
    """Compute kinematic metrics from smoothed court positions.

    Computes:
    - distance_m: cumulative Euclidean displacement
    - speeds: per-frame speed in m/s (finite difference)
    - accelerations: per-frame acceleration in m/s^2
    - avg_speed_mps: mean speed
    - max_speed_mps: peak speed

    Args:
        positions: List of (x, y) smoothed court coordinates.
        fps: Video frame rate.

    Returns:
        Dictionary with keys: distance_m, speeds, accelerations,
        avg_speed_mps, max_speed_mps.
    """
    if len(positions) < 2:
        return {
            "distance_m": 0.0,
            "speeds": [0.0] * len(positions),
            "accelerations": [0.0] * len(positions),
            "avg_speed_mps": 0.0,
            "max_speed_mps": 0.0,
        }

    xs = np.array([p[0] for p in positions], dtype=np.float64)
    ys = np.array([p[1] for p in positions], dtype=np.float64)

    # Displacements between consecutive frames
    dx = np.diff(xs)
    dy = np.diff(ys)
    frame_dists = np.sqrt(dx**2 + dy**2)

    # Cumulative distance
    distance_m = float(np.sum(frame_dists))

    # Per-frame speeds (m/s) via finite difference
    dt = 1.0 / fps
    speeds_array = frame_dists / dt
    # Pad first frame with 0 speed
    speeds = [0.0] + speeds_array.tolist()

    # Per-frame accelerations (m/s^2)
    acc_array = np.diff(speeds_array) / dt
    # Pad first two frames with 0 acceleration
    accelerations = [0.0, 0.0] + acc_array.tolist()

    avg_speed_mps = float(np.mean(speeds_array)) if len(speeds_array) > 0 else 0.0
    max_speed_mps = float(np.max(speeds_array)) if len(speeds_array) > 0 else 0.0

    return {
        "distance_m": distance_m,
        "speeds": speeds,
        "accelerations": accelerations,
        "avg_speed_mps": avg_speed_mps,
        "max_speed_mps": max_speed_mps,
    }
