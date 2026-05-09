from __future__ import annotations

from loguru import logger

from src.schemas import CourtRegistration2D, PlayerDetection, PlayerTrack
from src.tracking.bytetrack import ByteTracker
from src.tracking.identity import assign_player_identities, stabilize_identities


def track_players(
    video_path: str,
    detections: dict[int, list[PlayerDetection]],
    config: dict,
    registration: CourtRegistration2D | None = None,
    fps: float = 30.0,
    image_shape: tuple[int, int] = (1080, 1920),
) -> list[PlayerTrack]:
    """Full tracking pipeline: detect -> track -> assign identity -> stabilize.

    Steps:
    1. Initialize ByteTracker with config.
    2. Feed frame-by-frame detections in order.
    3. Collect raw tracks.
    4. Assign player identities (near_left, near_right, far_left, far_right).
    5. Stabilize (remove short tracks, merge fragments).
    6. Return final PlayerTrack list (up to 4 players).

    Args:
        video_path: Path to video file (for logging context).
        detections: Dict mapping frame index to list of PlayerDetection.
        config: Full pipeline config dict (must contain 'tracking', including 'tracking.bytetrack').
        registration: Court registration result for identity assignment.
        fps: Video frame rate.
        image_shape: (height, width) of video frames.

    Returns:
        List of PlayerTrack (up to 4 players) with semantic IDs.
    """
    tracking_cfg = config.get("tracking", {})
    min_track_duration_s = tracking_cfg.get("min_track_duration_s", 1.0)
    max_active_players = tracking_cfg.get("max_active_players", 4)

    # Step 1: Initialize tracker
    tracker = ByteTracker(config)
    logger.info("Tracking players in '{}' ({} frames with detections)", video_path, len(detections))

    # Step 2: Feed detections frame by frame (sorted by frame index)
    sorted_frames = sorted(detections.keys())
    for frame_idx in sorted_frames:
        frame_dets = detections[frame_idx]
        tracker.update(frame_dets, frame_idx)

    # Step 3: Collect raw tracks
    raw_tracks = tracker.get_tracks()
    logger.info("ByteTrack produced {} raw tracks", len(raw_tracks))

    if not raw_tracks:
        return []

    # Step 4: Assign player identities
    player_tracks = assign_player_identities(raw_tracks, registration, image_shape)

    # Step 5: Stabilize identities
    stable_tracks = stabilize_identities(player_tracks, min_track_duration_s, fps)

    # Limit to max_active_players (keep most confident/longest)
    if len(stable_tracks) > max_active_players:
        stable_tracks.sort(
            key=lambda t: len(t.frames) * (sum(t.confidences) / max(len(t.confidences), 1)),
            reverse=True,
        )
        stable_tracks = stable_tracks[:max_active_players]

    logger.info(
        "Final tracking result: {} players identified ({})",
        len(stable_tracks),
        ", ".join(t.player_id for t in stable_tracks),
    )
    return stable_tracks
