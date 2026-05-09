from __future__ import annotations

import math
from statistics import median

import numpy as np
from loguru import logger

from src.schemas import (
    BallEventCandidate,
    BallTrack2D,
    CourtGeometry2D,
    RallyTempoMetric,
    ShotDepthProxyMetric,
    ShotDirectionProxyMetric,
)


def compute_rally_tempo(
    touch_candidates: list[BallEventCandidate],
    fps: float,
) -> list[RallyTempoMetric]:
    """Compute tempo metrics from touch candidate timing.

    Groups touches into rallies (separated by gaps > 4 seconds)
    and computes timing statistics for each rally.
    """
    if len(touch_candidates) < 2:
        return []

    # Sort by time
    sorted_touches = sorted(touch_candidates, key=lambda t: t.time_s)

    # Group into rallies: a gap > 4s indicates a new rally
    rally_gap_threshold_s = 4.0
    rallies: list[list[BallEventCandidate]] = []
    current_rally: list[BallEventCandidate] = [sorted_touches[0]]

    for i in range(1, len(sorted_touches)):
        gap = sorted_touches[i].time_s - sorted_touches[i - 1].time_s
        if gap > rally_gap_threshold_s:
            if len(current_rally) >= 2:
                rallies.append(current_rally)
            current_rally = [sorted_touches[i]]
        else:
            current_rally.append(sorted_touches[i])

    if len(current_rally) >= 2:
        rallies.append(current_rally)

    # Compute metrics for each rally
    metrics: list[RallyTempoMetric] = []
    for rally_id, rally in enumerate(rallies):
        duration_s = rally[-1].time_s - rally[0].time_s
        estimated_shots = len(rally)

        # Time between consecutive touches
        intervals = [
            rally[i].time_s - rally[i - 1].time_s for i in range(1, len(rally))
        ]

        avg_interval = sum(intervals) / len(intervals) if intervals else None
        median_interval = median(intervals) if intervals else None

        metrics.append(
            RallyTempoMetric(
                rally_id=rally_id,
                duration_s=round(duration_s, 3),
                estimated_shots=estimated_shots,
                avg_time_between_touches_s=round(avg_interval, 3) if avg_interval else None,
                median_time_between_touches_s=round(median_interval, 3) if median_interval else None,
            )
        )

    logger.debug("Computed tempo for {} rallies", len(metrics))
    return metrics


def compute_bounce_heatmap(
    bounce_candidates: list[BallEventCandidate],
    court_width_m: float = 10.0,
    court_length_m: float = 20.0,
) -> np.ndarray | None:
    """Generate 2D bounce distribution on court.

    Returns a 2D array (20x10 grid, 1m resolution) with bounce counts,
    or None if no bounces with court coordinates are available.
    """
    # Filter to bounces with court coordinates
    bounces_with_coords = [
        b for b in bounce_candidates if b.court_xy_approx is not None
    ]

    if not bounces_with_coords:
        return None

    # Create grid (1m resolution)
    grid_w = int(court_width_m)
    grid_l = int(court_length_m)
    heatmap = np.zeros((grid_l, grid_w), dtype=np.float64)

    for bounce in bounces_with_coords:
        x, y = bounce.court_xy_approx
        # Clamp to court bounds
        gx = int(min(max(x, 0), court_width_m - 0.01))
        gy = int(min(max(y, 0), court_length_m - 0.01))

        if 0 <= gx < grid_w and 0 <= gy < grid_l:
            heatmap[gy, gx] += bounce.confidence

    logger.debug(
        "Bounce heatmap: {} bounces mapped to grid",
        len(bounces_with_coords),
    )
    return heatmap


def compute_shot_direction(
    touch_candidates: list[BallEventCandidate],
    ball_tracks: list[BallTrack2D],
    fps: float,
) -> list[ShotDirectionProxyMetric]:
    """Classify outgoing ball direction after each touch.

    Direction classification:
    - cross_court: ball moves laterally across > 60% of court width
    - down_the_line: ball moves mostly along the sideline (< 20% lateral)
    - middle: ball goes toward center of court
    - unknown: insufficient data
    """
    if not touch_candidates or not ball_tracks:
        return []

    # Index tracks by frame for lookup
    track_by_frame = {t.frame: t for t in ball_tracks if t.state != "missing"}

    metrics: list[ShotDirectionProxyMetric] = []

    for touch in touch_candidates:
        # Look at ball trajectory after touch (next 5-15 frames)
        after_positions: list[tuple[float, float]] = []
        for offset in range(3, 16):
            future_frame = touch.frame + offset
            if future_frame in track_by_frame:
                after_positions.append(track_by_frame[future_frame].image_xy)

        if len(after_positions) < 3:
            continue

        # Compute direction from touch point to average of future positions
        start_x, start_y = touch.image_xy
        end_x = sum(p[0] for p in after_positions[-3:]) / 3
        end_y = sum(p[1] for p in after_positions[-3:]) / 3

        dx = end_x - start_x
        dy = end_y - start_y
        total_dist = math.sqrt(dx**2 + dy**2)

        if total_dist < 20:  # too small to classify
            continue

        # Classify based on lateral vs longitudinal movement
        lateral_ratio = abs(dx) / total_dist if total_dist > 0 else 0

        if lateral_ratio > 0.6:
            direction = "cross_court"
        elif lateral_ratio < 0.2:
            direction = "down_the_line"
        else:
            direction = "middle"

        # Determine player from touch evidence
        player_id = "near_left"  # default
        if touch.evidence and "nearest_player" in touch.evidence:
            pid = touch.evidence["nearest_player"]
            if pid in ("near_left", "near_right", "far_left", "far_right"):
                player_id = pid

        confidence = min(1.0, touch.confidence * 0.8)

        metrics.append(
            ShotDirectionProxyMetric(
                frame=touch.frame,
                time_s=touch.time_s,
                player_id=player_id,
                direction=direction,
                confidence=confidence,
            )
        )

    logger.debug("Computed {} shot direction metrics", len(metrics))
    return metrics


def compute_shot_depth(
    bounce_candidates: list[BallEventCandidate],
    geometry: CourtGeometry2D,
) -> list[ShotDepthProxyMetric]:
    """Classify bounce depth based on court zone.

    Depth classification (relative to receiving side):
    - short: lands in service box area (< service_line distance from net)
    - mid: lands between service line and baseline midpoint
    - deep: lands near baseline
    - unknown: no court coordinates available
    """
    if not bounce_candidates:
        return []

    net_y = geometry.net_y_m
    half_length = geometry.length_m / 2.0
    service_offset = geometry.service_line_offset_from_net_m

    metrics: list[ShotDepthProxyMetric] = []

    for bounce in bounce_candidates:
        if bounce.court_xy_approx is None:
            continue

        _, by = bounce.court_xy_approx

        # Determine which half of the court the bounce is on
        if by < net_y:
            # Near side: distance from net
            dist_from_net = net_y - by
        else:
            # Far side: distance from net
            dist_from_net = by - net_y

        # Classify depth
        if dist_from_net < service_offset * 0.5:
            depth = "short"
        elif dist_from_net < service_offset:
            depth = "mid"
        elif dist_from_net < half_length:
            depth = "deep"
        else:
            depth = "unknown"

        # Determine player (approximation: near side bounce = far player hit it)
        if by < net_y:
            player_id = "far_left"  # ball in near half, hit by far player
        else:
            player_id = "near_left"  # ball in far half, hit by near player

        confidence = min(1.0, bounce.confidence * 0.7)

        metrics.append(
            ShotDepthProxyMetric(
                frame=bounce.frame,
                time_s=bounce.time_s,
                player_id=player_id,
                depth=depth,
                confidence=confidence,
            )
        )

    logger.debug("Computed {} shot depth metrics", len(metrics))
    return metrics
