from __future__ import annotations

import numpy as np


def generate_heatmap(
    positions: list[tuple[float, float]],
    court_width_m: float = 10.0,
    court_length_m: float = 20.0,
    resolution: float = 0.5,
) -> np.ndarray:
    """Generate 2D histogram of court positions.

    Creates a grid over the court and counts how many positions fall into each
    cell. The result is normalized to [0, 1].

    Args:
        positions: List of (x, y) court coordinates.
        court_width_m: Court width in meters (x-axis).
        court_length_m: Court length in meters (y-axis).
        resolution: Meters per cell in the output grid.

    Returns:
        2D numpy array of shape (n_rows, n_cols) normalized to [0, 1].
        Rows correspond to y-axis (length), columns to x-axis (width).
        Returns all-zeros array if positions is empty.
    """
    n_cols = int(np.ceil(court_width_m / resolution))
    n_rows = int(np.ceil(court_length_m / resolution))

    if not positions:
        return np.zeros((n_rows, n_cols), dtype=np.float64)

    xs = np.array([p[0] for p in positions], dtype=np.float64)
    ys = np.array([p[1] for p in positions], dtype=np.float64)

    # Compute 2D histogram
    heatmap, _, _ = np.histogram2d(
        ys,
        xs,
        bins=[n_rows, n_cols],
        range=[[0.0, court_length_m], [0.0, court_width_m]],
    )

    # Normalize to [0, 1]
    max_val = heatmap.max()
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap
