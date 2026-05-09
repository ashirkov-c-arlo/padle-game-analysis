from __future__ import annotations

from loguru import logger

from src.analytics.kinematics import compute_kinematics
from src.analytics.zones import (
    classify_formation,
    classify_zone,
    compute_partner_spacing,
    compute_zone_time,
)
from src.coordinates.projection import project_tracks_to_court
from src.coordinates.smoothing import smooth_trajectory
from src.schemas import (
    CourtGeometry2D,
    CourtRegistration2D,
    PlayerMetricFrame,
    PlayerTrack,
)


def compute_player_metrics(
    tracks: list[PlayerTrack],
    registration: CourtRegistration2D,
    geometry: CourtGeometry2D,
    config: dict,
    fps: float,
) -> dict:
    """Full analytics pipeline for player metrics.

    Pipeline steps:
    1. Project tracks to court coords
    2. Smooth trajectories
    3. Compute kinematics per player
    4. Compute zone time per player
    5. Compute team formations over time
    6. Compute partner spacing over time
    7. Return structured metrics dict

    If registration.mode == "pixel_only": return empty metrics (graceful degradation).

    Args:
        tracks: List of player tracks.
        registration: Court registration with homography.
        geometry: Court geometry for zone classification.
        config: Configuration dict (smoothing params).
        fps: Video frame rate.

    Returns:
        Dictionary with player metrics, formations, spacing.
        Returns empty dict if registration mode is pixel_only.
    """
    if registration.mode == "pixel_only":
        logger.debug("Skipping court-coordinate metrics: registration mode is pixel_only")
        return {}

    # Extract smoothing config
    smoothing_cfg = config.get("smoothing", {})
    method = smoothing_cfg.get("method", "savgol")
    window_frames = smoothing_cfg.get("window_frames", 7)
    polyorder = smoothing_cfg.get("polyorder", 3)
    max_speed_mps = smoothing_cfg.get("max_speed_mps", 8.0)

    # Step 1: Project tracks to court coordinates
    projected = project_tracks_to_court(tracks, registration, fps)

    if not projected:
        logger.warning("No tracks projected to court coordinates")
        return {}

    # Step 2: Smooth trajectories
    smoothed: dict[str, list[tuple[float, float]]] = {}
    for player_id, positions in projected.items():
        smoothed[player_id] = smooth_trajectory(
            positions,
            method=method,
            window_frames=window_frames,
            polyorder=polyorder,
            max_speed_mps=max_speed_mps,
            fps=fps,
        )

    # Step 3: Compute kinematics per player
    kinematics: dict[str, dict] = {}
    for player_id, positions in smoothed.items():
        kinematics[player_id] = compute_kinematics(positions, fps)

    # Step 4: Compute zone time per player
    zone_times: dict[str, dict[str, float]] = {}
    for player_id, positions in smoothed.items():
        zone_times[player_id] = compute_zone_time(
            positions, player_id, geometry, fps  # type: ignore[arg-type]
        )

    # Step 5: Compute team formations over time
    # Group players by team
    team_players: dict[str, list[str]] = {"near": [], "far": []}
    for track in tracks:
        if track.player_id in smoothed:
            team_players[track.team].append(track.player_id)

    formations: dict[str, list[str]] = {}
    for team, players in team_players.items():
        if len(players) == 2:
            p1_positions = smoothed[players[0]]
            p2_positions = smoothed[players[1]]
            min_len = min(len(p1_positions), len(p2_positions))
            team_formations: list[str] = []
            for i in range(min_len):
                z1 = classify_zone(p1_positions[i], geometry)
                z2 = classify_zone(p2_positions[i], geometry)
                formation = classify_formation(z1, z2)
                team_formations.append(formation)
            formations[team] = team_formations

    # Step 6: Compute partner spacing over time
    spacing: dict[str, list[float]] = {}
    for team, players in team_players.items():
        if len(players) == 2:
            p1_positions = smoothed[players[0]]
            p2_positions = smoothed[players[1]]
            min_len = min(len(p1_positions), len(p2_positions))
            team_spacing: list[float] = []
            for i in range(min_len):
                dist = compute_partner_spacing(p1_positions[i], p2_positions[i])
                team_spacing.append(dist)
            spacing[team] = team_spacing

    return {
        "projected": projected,
        "smoothed": smoothed,
        "kinematics": kinematics,
        "zone_times": zone_times,
        "formations": formations,
        "spacing": spacing,
    }


def build_player_metric_frames(
    tracks: list[PlayerTrack],
    registration: CourtRegistration2D,
    geometry: CourtGeometry2D,
    config: dict,
    fps: float,
) -> list[PlayerMetricFrame]:
    """Build per-frame PlayerMetricFrame objects for export.

    Args:
        tracks: List of player tracks.
        registration: Court registration with homography.
        geometry: Court geometry for zone classification.
        config: Configuration dict (smoothing params).
        fps: Video frame rate.

    Returns:
        List of PlayerMetricFrame objects, one per player per frame.
        Returns empty list if registration mode is pixel_only.
    """
    if registration.mode == "pixel_only":
        return []

    # Get smoothing config
    smoothing_cfg = config.get("smoothing", {})
    method = smoothing_cfg.get("method", "savgol")
    window_frames = smoothing_cfg.get("window_frames", 7)
    polyorder = smoothing_cfg.get("polyorder", 3)
    max_speed_mps = smoothing_cfg.get("max_speed_mps", 8.0)

    # Project and smooth
    projected = project_tracks_to_court(tracks, registration, fps)
    if not projected:
        return []

    frames_list: list[PlayerMetricFrame] = []

    for track in tracks:
        if track.player_id not in projected:
            continue

        positions = projected[track.player_id]
        smoothed_positions = smooth_trajectory(
            positions,
            method=method,
            window_frames=window_frames,
            polyorder=polyorder,
            max_speed_mps=max_speed_mps,
            fps=fps,
        )

        # Compute speeds
        kin = compute_kinematics(smoothed_positions, fps)
        speeds = kin["speeds"]

        for i, frame_num in enumerate(track.frames):
            if i >= len(smoothed_positions):
                break

            court_xy = smoothed_positions[i]
            zone = classify_zone(court_xy, geometry)
            speed = speeds[i] if i < len(speeds) else 0.0
            confidence = track.confidences[i] if i < len(track.confidences) else 0.0

            metric_frame = PlayerMetricFrame(
                frame=frame_num,
                time_s=frame_num / fps,
                player_id=track.player_id,
                court_xy=court_xy,
                speed_mps=speed,
                zone=zone,
                confidence=confidence,
                metric_quality="estimated",
            )
            frames_list.append(metric_frame)

    return frames_list
