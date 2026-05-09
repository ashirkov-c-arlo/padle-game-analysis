from __future__ import annotations

import math

from loguru import logger

from src.detection.roi_filter import project_to_court
from src.schemas import BallEventCandidate, BallTrack2D, CourtRegistration2D


def detect_bounce_candidates(
    tracks: list[BallTrack2D],
    registration: CourtRegistration2D | None,
    fps: float,
) -> list[BallEventCandidate]:
    """
    Detect bounce candidates from trajectory kinks:
    - Vertical velocity reversal (vy changes sign from positive to negative)
    - Speed drop or sudden direction change
    - Position near expected court floor level
    """
    if len(tracks) < 3:
        return []

    candidates: list[BallEventCandidate] = []

    # Filter to tracks that have actual positions (not missing)
    valid_tracks = [t for t in tracks if t.state != "missing"]
    if len(valid_tracks) < 3:
        return []

    for i in range(1, len(valid_tracks) - 1):
        prev = valid_tracks[i - 1]
        curr = valid_tracks[i]
        nxt = valid_tracks[i + 1]

        # Skip if frames are too far apart
        if curr.frame - prev.frame > 5 or nxt.frame - curr.frame > 5:
            continue

        # Compute vertical velocity before and after
        dt_before = (curr.frame - prev.frame) / fps if fps > 0 else 1.0
        dt_after = (nxt.frame - curr.frame) / fps if fps > 0 else 1.0

        vy_before = (curr.image_xy[1] - prev.image_xy[1]) / dt_before
        vy_after = (nxt.image_xy[1] - curr.image_xy[1]) / dt_after

        # Bounce: ball going down (vy > 0 in image coords) then going up (vy < 0)
        # In image coords, y increases downward, so downward = positive vy
        is_velocity_reversal = vy_before > 20.0 and vy_after < -10.0

        # Speed change
        vx_before = (curr.image_xy[0] - prev.image_xy[0]) / dt_before
        vx_after = (nxt.image_xy[0] - curr.image_xy[0]) / dt_after
        speed_before = math.sqrt(vx_before**2 + vy_before**2)
        speed_after = math.sqrt(vx_after**2 + vy_after**2)
        speed_ratio = speed_after / speed_before if speed_before > 0 else 1.0
        is_speed_drop = speed_ratio < 0.7

        if not is_velocity_reversal:
            continue

        # Compute confidence
        confidence = 0.4
        if is_speed_drop:
            confidence += 0.2
        if curr.state in ("detected", "tracked"):
            confidence += 0.2
        confidence = min(1.0, confidence)

        # Project to court coordinates if possible
        court_xy = None
        proj_quality = "proxy"
        if (
            registration is not None
            and registration.mode == "floor_homography"
            and registration.homography_image_to_court is not None
        ):
            court_xy = project_to_court(
                curr.image_xy, registration.homography_image_to_court
            )
            proj_quality = "estimated"

        candidates.append(
            BallEventCandidate(
                frame=curr.frame,
                time_s=curr.time_s,
                event_type="bounce_candidate",
                image_xy=curr.image_xy,
                court_xy_approx=court_xy,
                confidence=confidence,
                projection_quality=proj_quality,
                evidence={
                    "vy_before": round(vy_before, 1),
                    "vy_after": round(vy_after, 1),
                    "speed_ratio": round(speed_ratio, 3),
                },
            )
        )

    logger.debug("Found {} bounce candidates", len(candidates))
    return candidates


def detect_touch_candidates(
    ball_tracks: list[BallTrack2D],
    player_positions: dict[str, list[tuple[float, float]]],
    fps: float,
    proximity_threshold_px: float = 100.0,
) -> list[BallEventCandidate]:
    """
    Detect touch candidates:
    - Ball near a player (proximity in image space)
    - Sudden trajectory direction/speed change
    """
    if len(ball_tracks) < 3 or not player_positions:
        return []

    candidates: list[BallEventCandidate] = []
    valid_tracks = [t for t in ball_tracks if t.state != "missing"]

    if len(valid_tracks) < 3:
        return []

    for i in range(1, len(valid_tracks) - 1):
        prev = valid_tracks[i - 1]
        curr = valid_tracks[i]
        nxt = valid_tracks[i + 1]

        # Skip if frames too far apart
        if curr.frame - prev.frame > 5 or nxt.frame - curr.frame > 5:
            continue

        # Check direction change
        dt_before = (curr.frame - prev.frame) / fps if fps > 0 else 1.0
        dt_after = (nxt.frame - curr.frame) / fps if fps > 0 else 1.0

        vx_before = (curr.image_xy[0] - prev.image_xy[0]) / dt_before
        vy_before = (curr.image_xy[1] - prev.image_xy[1]) / dt_before
        vx_after = (nxt.image_xy[0] - curr.image_xy[0]) / dt_after
        vy_after = (nxt.image_xy[1] - curr.image_xy[1]) / dt_after

        # Direction change (angle between velocity vectors)
        speed_before = math.sqrt(vx_before**2 + vy_before**2)
        speed_after = math.sqrt(vx_after**2 + vy_after**2)

        if speed_before < 10.0 or speed_after < 10.0:
            continue

        dot = vx_before * vx_after + vy_before * vy_after
        cos_angle = dot / (speed_before * speed_after)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_change = math.acos(cos_angle)

        # Significant direction change (> 30 degrees)
        if angle_change < math.radians(30):
            continue

        # Check proximity to any player
        nearest_player = None
        nearest_distance = float("inf")

        for player_id, positions in player_positions.items():
            # Find the position at or near this frame
            frame_idx = curr.frame
            if frame_idx < len(positions):
                player_pos = positions[frame_idx]
            elif positions:
                player_pos = positions[-1]
            else:
                continue

            dx = curr.image_xy[0] - player_pos[0]
            dy = curr.image_xy[1] - player_pos[1]
            dist = math.sqrt(dx**2 + dy**2)

            if dist < nearest_distance:
                nearest_distance = dist
                nearest_player = player_id

        if nearest_distance > proximity_threshold_px:
            continue

        # Compute confidence
        confidence = 0.3
        if angle_change > math.radians(60):
            confidence += 0.2
        if nearest_distance < proximity_threshold_px * 0.5:
            confidence += 0.2
        if curr.state in ("detected", "tracked"):
            confidence += 0.1
        confidence = min(1.0, confidence)

        candidates.append(
            BallEventCandidate(
                frame=curr.frame,
                time_s=curr.time_s,
                event_type="touch_candidate",
                image_xy=curr.image_xy,
                court_xy_approx=None,
                confidence=confidence,
                projection_quality="proxy",
                evidence={
                    "angle_change_deg": round(math.degrees(angle_change), 1),
                    "nearest_player": nearest_player,
                    "distance_px": round(nearest_distance, 1),
                },
            )
        )

    logger.debug("Found {} touch candidates", len(candidates))
    return candidates


def detect_net_crossings(
    tracks: list[BallTrack2D],
    registration: CourtRegistration2D | None,
) -> list[BallEventCandidate]:
    """
    Detect when ball trajectory crosses net line (y=10 in court coords).
    Only available with floor_homography.
    """
    if (
        registration is None
        or registration.mode == "pixel_only"
        or registration.homography_image_to_court is None
    ):
        return []

    valid_tracks = [t for t in tracks if t.state != "missing"]
    if len(valid_tracks) < 2:
        return []

    candidates: list[BallEventCandidate] = []
    H = registration.homography_image_to_court
    net_y = 10.0  # Court net at y=10m

    prev_court_y = None
    for i, track in enumerate(valid_tracks):
        court_xy = project_to_court(track.image_xy, H)
        court_y = court_xy[1]

        if prev_court_y is not None:
            # Check if crossed net line
            crossed = (prev_court_y < net_y and court_y >= net_y) or (
                prev_court_y >= net_y and court_y < net_y
            )
            if crossed:
                cross_x = court_xy[0]  # approximate
                crossing_court_xy = (cross_x, net_y)

                confidence = 0.5
                if track.state in ("detected", "tracked"):
                    confidence += 0.2
                if track.confidence > 0.6:
                    confidence += 0.1
                confidence = min(1.0, confidence)

                candidates.append(
                    BallEventCandidate(
                        frame=track.frame,
                        time_s=track.time_s,
                        event_type="net_crossing_candidate",
                        image_xy=track.image_xy,
                        court_xy_approx=crossing_court_xy,
                        confidence=confidence,
                        projection_quality="estimated",
                        evidence={
                            "direction": "near_to_far" if prev_court_y < net_y else "far_to_near",
                            "court_y_before": round(prev_court_y, 2),
                            "court_y_after": round(court_y, 2),
                        },
                    )
                )

        prev_court_y = court_y

    logger.debug("Found {} net crossing candidates", len(candidates))
    return candidates
