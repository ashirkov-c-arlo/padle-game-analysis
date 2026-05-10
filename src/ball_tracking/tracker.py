from __future__ import annotations

import cv2
import numpy as np
from loguru import logger

from src.ball_tracking.kalman_tracker import BallKalmanTracker
from src.schemas import BallDetection2D, BallTrack2D, CourtRegistration2D
from src.video_io.reader import get_video_info


def track_ball(
    video_path: str,
    config: dict,
    registration: CourtRegistration2D | None = None,
) -> tuple[list[BallDetection2D], list[BallTrack2D]]:
    """Compatibility wrapper for the single-pass ball detector and Kalman tracker."""
    from src.ball_tracking.ball_processor import BallFrameProcessor
    from src.video_io.single_pass import run_single_pass

    info = get_video_info(video_path)
    fps = info["fps"]
    total_frames = info["total_frames"]

    court_roi = _build_court_roi(registration, info) if registration else None

    processor = BallFrameProcessor(config, fps, court_roi)
    run_single_pass(video_path, [processor])
    detections = processor.detections
    tracks = build_ball_tracks(detections, config, total_frames, fps)

    logger.info(
        "Ball tracking complete: {} detections, {} track frames",
        len(detections),
        len(tracks),
    )
    return detections, tracks


def build_ball_tracks(
    detections: list[BallDetection2D],
    config: dict,
    total_frames: int,
    fps: float,
) -> list[BallTrack2D]:
    """Run Kalman tracking on pre-computed ball detections.

    This is the post-processing step that can be used after single-pass detection.
    """
    bt_config = config.get("ball_tracking", {})
    max_gap = bt_config.get("max_gap_frames", 10)
    process_noise = bt_config.get("kalman_process_noise", 0.1)

    det_by_frame: dict[int, BallDetection2D] = {d.frame: d for d in detections}

    kalman = BallKalmanTracker(process_noise=process_noise)
    tracks: list[BallTrack2D] = []
    gap_count = 0

    for frame_idx in range(total_frames):
        time_s = frame_idx / fps if fps > 0 else 0.0
        det = det_by_frame.get(frame_idx)

        if det is not None:
            kalman.predict()
            kalman.update(det.image_xy, det.confidence)
            pos, vel = kalman.get_state()

            velocity_px_s = (vel[0] * fps, vel[1] * fps) if fps > 0 else None

            state = "tracked" if kalman.initialized else "detected"
            tracks.append(
                BallTrack2D(
                    frame=frame_idx,
                    time_s=time_s,
                    image_xy=det.image_xy,
                    velocity_px_s=velocity_px_s,
                    confidence=det.confidence,
                    state=state,
                    interpolated=False,
                    gap_len=0,
                )
            )
            gap_count = 0
        elif kalman.initialized:
            gap_count += 1
            if gap_count <= max_gap:
                predicted_pos = kalman.predict()
                _, vel = kalman.get_state()
                velocity_px_s = (vel[0] * fps, vel[1] * fps) if fps > 0 else None

                conf = max(0.1, 0.8 - gap_count * 0.07)
                tracks.append(
                    BallTrack2D(
                        frame=frame_idx,
                        time_s=time_s,
                        image_xy=predicted_pos,
                        velocity_px_s=velocity_px_s,
                        confidence=conf,
                        state="interpolated",
                        interpolated=True,
                        gap_len=gap_count,
                    )
                )
            else:
                pos, vel = kalman.get_state()
                tracks.append(
                    BallTrack2D(
                        frame=frame_idx,
                        time_s=time_s,
                        image_xy=pos,
                        velocity_px_s=None,
                        confidence=0.0,
                        state="missing",
                        interpolated=False,
                        gap_len=gap_count,
                    )
                )

    return tracks


def _build_court_roi(
    registration: CourtRegistration2D, video_info: dict
) -> np.ndarray | None:
    """Build a binary mask covering the court area in image space."""
    if registration.mode == "pixel_only" or registration.homography_court_to_image is None:
        return None

    h = video_info["height"]
    w = video_info["width"]

    # Project court corners to image space
    H = np.array(registration.homography_court_to_image, dtype=np.float64)

    # Court corners in court coords (10m x 20m)
    court_corners = np.array(
        [[0, 0], [10, 0], [10, 20], [0, 20]], dtype=np.float64
    )

    # Project to image
    ones = np.ones((4, 1), dtype=np.float64)
    pts_h = np.hstack([court_corners, ones])
    projected = (H @ pts_h.T).T
    projected_xy = projected[:, :2] / projected[:, 2:3]

    # Create mask with some margin
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = projected_xy.astype(np.int32).reshape((-1, 1, 2))

    # Expand polygon slightly for margin
    center = pts.mean(axis=0)
    expanded = ((pts - center) * 1.2 + center).astype(np.int32)

    cv2.fillPoly(mask, [expanded], 255)
    return mask
