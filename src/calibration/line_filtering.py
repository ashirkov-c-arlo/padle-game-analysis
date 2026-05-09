from __future__ import annotations

import numpy as np
from loguru import logger


def filter_court_lines(lines: np.ndarray, image_shape: tuple) -> np.ndarray:
    """Filter lines by length, angle, and position to keep only court-like lines.

    Args:
        lines: Nx4 array (x1, y1, x2, y2)
        image_shape: (height, width) or (height, width, channels)

    Returns:
        Filtered Nx4 array of court-candidate lines.
    """
    if len(lines) == 0:
        return lines

    h, w = image_shape[:2]

    # Compute line properties
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    lengths = np.sqrt(dx**2 + dy**2)

    # Minimum length: at least 2% of image diagonal
    min_length = 0.02 * np.sqrt(h**2 + w**2)
    length_mask = lengths >= min_length

    # Compute angles in degrees (0=horizontal, 90=vertical)
    angles = np.abs(np.degrees(np.arctan2(dy, dx)))
    # Normalize to 0-90 range
    angles = np.where(angles > 90, 180 - angles, angles)

    # Keep lines that are roughly horizontal (0-25 deg) or vertical (65-90 deg)
    # Court lines in typical camera views are mostly horizontal or vertical
    angle_mask = (angles <= 30) | (angles >= 60)

    # Position filter: keep lines where at least one endpoint is in the central region
    # Court is typically in the central 80% of the frame
    margin_x = w * 0.1
    margin_y = h * 0.1
    x_coords = np.column_stack([lines[:, 0], lines[:, 2]])
    y_coords = np.column_stack([lines[:, 1], lines[:, 3]])

    in_bounds_x = np.any((x_coords >= margin_x) & (x_coords <= w - margin_x), axis=1)
    in_bounds_y = np.any((y_coords >= margin_y) & (y_coords <= h - margin_y), axis=1)
    position_mask = in_bounds_x & in_bounds_y

    combined_mask = length_mask & angle_mask & position_mask
    filtered = lines[combined_mask]

    logger.debug(
        "Line filter: {} -> {} lines (length={}, angle={}, position={})",
        len(lines),
        len(filtered),
        int(length_mask.sum()),
        int(angle_mask.sum()),
        int(position_mask.sum()),
    )
    return filtered


def cluster_lines(lines: np.ndarray) -> dict:
    """Cluster filtered lines into groups by orientation.

    Returns dict with keys 'horizontal' and 'vertical', each containing Nx4 arrays.
    Nearby parallel lines within each group are merged.
    """
    if len(lines) == 0:
        return {"horizontal": np.empty((0, 4)), "vertical": np.empty((0, 4))}

    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    angles = np.abs(np.degrees(np.arctan2(dy, dx)))
    angles = np.where(angles > 90, 180 - angles, angles)

    # Split into horizontal (0-45 deg) and vertical (45-90 deg)
    h_mask = angles < 45
    v_mask = ~h_mask

    horizontal = lines[h_mask]
    vertical = lines[v_mask]

    # Merge nearby parallel fragments within each group.
    horizontal = _merge_nearby_lines(horizontal, distance_threshold=20, angle_threshold=12)
    vertical = _merge_nearby_lines(vertical, distance_threshold=20, angle_threshold=12)

    logger.debug(
        "Line clustering: {} horizontal, {} vertical",
        len(horizontal),
        len(vertical),
    )
    return {"horizontal": horizontal, "vertical": vertical}


def _merge_nearby_lines(
    lines: np.ndarray, distance_threshold: float, angle_threshold: float
) -> np.ndarray:
    """Merge lines that are nearby and roughly parallel.

    For each cluster of parallel nearby fragments, return a single extended
    segment spanning the projected endpoints of the cluster.
    """
    if len(lines) <= 1:
        return lines

    # Compute midpoints and angles
    mid_x = (lines[:, 0] + lines[:, 2]) / 2
    mid_y = (lines[:, 1] + lines[:, 3]) / 2
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    angles = np.degrees(np.arctan2(dy, dx))
    lengths = np.sqrt(dx**2 + dy**2)

    n = len(lines)
    used = np.zeros(n, dtype=bool)
    merged_lines = []

    # Sort by length descending - prefer longer lines
    order = np.argsort(-lengths)

    for i in range(n):
        idx = order[i]
        if used[idx]:
            continue

        group = [idx]
        used[idx] = True

        # Find lines close to this one
        for j in range(i + 1, n):
            jdx = order[j]
            if used[jdx]:
                continue

            # Check angle similarity
            angle_diff = abs(angles[idx] - angles[jdx])
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            if angle_diff > angle_threshold and abs(angle_diff - 180) > angle_threshold:
                continue

            # Check perpendicular distance between midpoints
            dist = np.sqrt((mid_x[idx] - mid_x[jdx]) ** 2 + (mid_y[idx] - mid_y[jdx]) ** 2)

            # Project distance perpendicular to the line direction
            if lengths[idx] > 0:
                # Unit normal to the line
                nx = -dy[idx] / lengths[idx]
                ny = dx[idx] / lengths[idx]
                perp_dist = abs(
                    nx * (mid_x[jdx] - mid_x[idx]) + ny * (mid_y[jdx] - mid_y[idx])
                )
            else:
                perp_dist = dist

            if perp_dist < distance_threshold:
                used[jdx] = True
                group.append(jdx)

        merged_lines.append(_merge_line_group(lines[np.array(group)], lines[idx]))

    return np.array(merged_lines, dtype=np.float64)


def _merge_line_group(group: np.ndarray, reference_line: np.ndarray) -> np.ndarray:
    """Merge collinear line fragments into one line segment."""
    if len(group) == 1:
        return group[0]

    x1, y1, x2, y2 = reference_line
    direction = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return reference_line

    direction /= norm
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)

    endpoints = group.reshape(-1, 2)
    along = endpoints @ direction
    across = endpoints @ normal

    start = direction * along.min() + normal * np.median(across)
    end = direction * along.max() + normal * np.median(across)
    return np.array([start[0], start[1], end[0], end[1]], dtype=np.float64)
