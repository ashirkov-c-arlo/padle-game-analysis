from __future__ import annotations

from typing import cast

import numpy as np
from loguru import logger

from src.detection.roi_filter import get_footpoint, project_to_court
from src.schemas import CourtRegistration2D, PlayerID, PlayerTrack


def _median_position(
    observations: list[dict],
    registration: CourtRegistration2D | None,
    image_shape: tuple[int, int],
) -> tuple[float, float]:
    """Compute median position for a track in court or pixel coordinates.

    Args:
        observations: List of {frame, bbox_xyxy, confidence} dicts.
        registration: Court registration (may be None).
        image_shape: (height, width) of image.

    Returns:
        (x, y) median position in court coords or normalized pixel coords.
    """
    positions = []
    use_homography = (
        registration is not None
        and registration.mode == "floor_homography"
        and registration.homography_image_to_court is not None
    )

    for obs in observations:
        foot = get_footpoint(obs["bbox_xyxy"])
        if use_homography:
            pos = project_to_court(foot, registration.homography_image_to_court)
        else:
            # Normalize to [0, 1] range for pixel_only mode
            h, w = image_shape
            pos = (foot[0] / w if w > 0 else 0.0, foot[1] / h if h > 0 else 0.0)
        positions.append(pos)

    if not positions:
        return (0.0, 0.0)

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    return (float(np.median(xs)), float(np.median(ys)))


def _assign_team(
    y_pos: float,
    registration: CourtRegistration2D | None,
) -> str:
    """Determine team (near/far) based on y position.

    For floor_homography: y < 10 is near team (near baseline at y=0), y > 10 is far.
    For pixel_only: y > 0.5 (lower in image) is near, y < 0.5 is far.
    """
    use_homography = (
        registration is not None
        and registration.mode == "floor_homography"
        and registration.homography_image_to_court is not None
    )

    if use_homography:
        return "near" if y_pos < 10.0 else "far"
    else:
        # pixel_only: larger y means lower in image (closer to camera = near)
        return "near" if y_pos > 0.5 else "far"


def _assign_side(
    x_pos: float,
    registration: CourtRegistration2D | None,
) -> str:
    """Determine side (left/right) based on x position.

    For floor_homography: x < 5 is left, x >= 5 is right.
    For pixel_only: x < 0.5 is left, x >= 0.5 is right.
    """
    use_homography = (
        registration is not None
        and registration.mode == "floor_homography"
        and registration.homography_image_to_court is not None
    )

    if use_homography:
        return "left" if x_pos < 5.0 else "right"
    else:
        return "left" if x_pos < 0.5 else "right"


def _track_score(observations: list[dict]) -> float:
    """Score tracks by duration and average confidence."""
    if not observations:
        return 0.0
    avg_confidence = sum(obs["confidence"] for obs in observations) / len(observations)
    return len(observations) * avg_confidence


def _select_team_tracks(
    candidates: list[tuple[int, float, str, float]],
) -> list[tuple[int, str]]:
    """Select up to two stable tracks for a team and assign left/right IDs."""
    if not candidates:
        return []
    if len(candidates) == 1:
        track_id, _, side, _ = candidates[0]
        return [(track_id, side)]

    by_side: dict[str, list[tuple[int, float, str, float]]] = {"left": [], "right": []}
    for candidate in candidates:
        by_side[candidate[2]].append(candidate)

    selected: list[tuple[int, float, str, float]]
    if by_side["left"] and by_side["right"]:
        selected = [
            max(by_side["left"], key=lambda item: item[3]),
            max(by_side["right"], key=lambda item: item[3]),
        ]
    else:
        selected = sorted(candidates, key=lambda item: item[3], reverse=True)[:2]

    selected.sort(key=lambda item: item[1])
    if len(selected) == 1:
        track_id, _, side, _ = selected[0]
        return [(track_id, side)]

    return [(selected[0][0], "left"), (selected[1][0], "right")]


def assign_player_identities(
    tracks: dict[int, list[dict]],
    registration: CourtRegistration2D | None,
    image_shape: tuple[int, int],
) -> list[PlayerTrack]:
    """Assign semantic player IDs to raw track IDs based on court position.

    Strategy:
    1. If floor_homography available:
       - Project footpoints to court coordinates
       - Near team: y < 10 (closer to near baseline at y=0)
       - Far team: y > 10 (closer to far baseline at y=20)
       - Left/right: x < 5 vs x >= 5
    2. If pixel_only:
       - Near team: lower in image (larger y pixel, normalized > 0.5)
       - Far team: upper in image (smaller y pixel, normalized < 0.5)
       - Left/right: x position in image
    3. Use median position over track lifetime for stable assignment.

    Args:
        tracks: Dict of {track_id: [{frame, bbox_xyxy, confidence}, ...]}.
        registration: Court registration result (may be None).
        image_shape: (height, width) of the image.

    Returns:
        List of PlayerTrack with assigned semantic IDs.
    """
    if not tracks:
        return []

    # Compute each raw track's team/side and keep the strongest two candidates per team.
    candidates_by_team: dict[str, list[tuple[int, float, str, float]]] = {"near": [], "far": []}
    for track_id, observations in tracks.items():
        if not observations:
            continue
        x_pos, y_pos = _median_position(observations, registration, image_shape)
        team = _assign_team(y_pos, registration)
        side = _assign_side(x_pos, registration)
        candidates_by_team[team].append((track_id, x_pos, side, _track_score(observations)))

    # Assign identities
    id_assignments: dict[int, PlayerID] = {}
    for team, candidates in candidates_by_team.items():
        for track_id, side in _select_team_tracks(candidates):
            id_assignments[track_id] = cast(PlayerID, f"{team}_{side}")

    # Build PlayerTrack objects
    result: list[PlayerTrack] = []
    for track_id, player_id in id_assignments.items():
        observations = tracks[track_id]
        team = "near" if "near" in player_id else "far"
        result.append(PlayerTrack(
            player_id=player_id,
            frames=[obs["frame"] for obs in observations],
            bboxes=[obs["bbox_xyxy"] for obs in observations],
            confidences=[obs["confidence"] for obs in observations],
            team=team,
        ))

    logger.debug("Assigned {} player identities from {} tracks", len(result), len(tracks))
    return result


def stabilize_identities(
    tracks: list[PlayerTrack],
    min_duration_s: float,
    fps: float,
) -> list[PlayerTrack]:
    """Post-process tracks for stability.

    Operations:
    - Remove tracks shorter than min_duration_s.
    - If multiple tracks claim same player_id, keep longest/most confident.
    - Merge fragments that are likely the same player (gap < 2s, similar position).

    Args:
        tracks: List of PlayerTrack with assigned IDs.
        min_duration_s: Minimum track duration in seconds.
        fps: Video frame rate.

    Returns:
        Stabilized list of PlayerTrack (up to 4 players).
    """
    if not tracks:
        return []

    min_frames = int(min_duration_s * fps)

    # Step 1: Remove short tracks
    filtered = [t for t in tracks if len(t.frames) >= min_frames]
    logger.debug(
        "Removed {} short tracks (min {} frames)",
        len(tracks) - len(filtered),
        min_frames,
    )

    # Step 2: Resolve duplicate player_id assignments
    # Group by player_id
    by_id: dict[PlayerID, list[PlayerTrack]] = {}
    for track in filtered:
        if track.player_id not in by_id:
            by_id[track.player_id] = []
        by_id[track.player_id].append(track)

    resolved: list[PlayerTrack] = []
    for player_id, candidates in by_id.items():
        if len(candidates) == 1:
            resolved.append(candidates[0])
            continue

        # Try to merge fragments
        merged = _merge_fragments(candidates, fps)
        if merged:
            resolved.append(merged)
        else:
            # Keep the track with highest total confidence (length * avg_confidence)
            best = max(
                candidates,
                key=lambda t: len(t.frames) * (sum(t.confidences) / max(len(t.confidences), 1)),
            )
            resolved.append(best)

    logger.debug("Stabilized to {} player tracks", len(resolved))
    return resolved


def _merge_fragments(
    candidates: list[PlayerTrack],
    fps: float,
) -> PlayerTrack | None:
    """Attempt to merge track fragments for the same player.

    Merge if:
    - Gap between fragments < 2 seconds
    - Fragments don't overlap in time

    Args:
        candidates: Multiple tracks with the same player_id.
        fps: Frame rate.

    Returns:
        Merged PlayerTrack or None if merge not possible.
    """
    max_gap_frames = int(2.0 * fps)

    # Sort by first frame
    sorted_tracks = sorted(candidates, key=lambda t: t.frames[0] if t.frames else 0)

    # Check if fragments can be merged sequentially
    merged_frames: list[int] = []
    merged_bboxes: list[tuple[float, float, float, float]] = []
    merged_confidences: list[float] = []

    for i, track in enumerate(sorted_tracks):
        if not track.frames:
            continue

        if merged_frames:
            last_frame = merged_frames[-1]
            first_frame = track.frames[0]
            gap = first_frame - last_frame

            # Check for overlap
            if gap < 0:
                return None
            # Check gap is reasonable
            if gap > max_gap_frames:
                return None

        merged_frames.extend(track.frames)
        merged_bboxes.extend(track.bboxes)
        merged_confidences.extend(track.confidences)

    if not merged_frames:
        return None

    return PlayerTrack(
        player_id=sorted_tracks[0].player_id,
        frames=merged_frames,
        bboxes=merged_bboxes,
        confidences=merged_confidences,
        team=sorted_tracks[0].team,
    )
