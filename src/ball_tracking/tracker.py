from __future__ import annotations

import cv2
import numpy as np
from loguru import logger

from src.ball_tracking.detector import BallDetector
from src.ball_tracking.kalman_tracker import BallKalmanTracker
from src.schemas import BallDetection2D, BallTrack2D, CourtRegistration2D
from src.video_io.reader import get_video_info


def track_ball(
    video_path: str,
    config: dict,
    registration: CourtRegistration2D | None = None,
) -> tuple[list[BallDetection2D], list[BallTrack2D]]:
    """
    Full ball tracking pipeline (legacy, decodes video internally).
    Prefer using BallFrameProcessor + build_ball_tracks for single-pass.
    """
    bt_config = config.get("ball_tracking", {})

    info = get_video_info(video_path)
    fps = info["fps"]
    total_frames = info["total_frames"]

    court_roi = _build_court_roi(registration, info) if registration else None

    detector = BallDetector(config)
    detections = detector.detect_video(video_path, court_roi=court_roi)

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


def interpolate_gaps(
    tracks: list[BallTrack2D],
    max_gap_frames: int,
) -> list[BallTrack2D]:
    """Fill short gaps with linear interpolation.

    Operates on an existing track list and fills in missing frames
    between detected/tracked segments when the gap is within limits.
    """
    if not tracks:
        return tracks

    result: list[BallTrack2D] = []
    # Find segments of detected/tracked frames
    detected_indices: list[int] = []
    for i, t in enumerate(tracks):
        if t.state in ("detected", "tracked"):
            detected_indices.append(i)

    if len(detected_indices) < 2:
        return tracks

    # Process gaps between detected segments
    output_frames: set[int] = set()

    for seg_idx in range(len(detected_indices) - 1):
        start_idx = detected_indices[seg_idx]
        end_idx = detected_indices[seg_idx + 1]
        start_track = tracks[start_idx]
        end_track = tracks[end_idx]

        gap_len = end_track.frame - start_track.frame - 1

        if gap_len <= 0:
            # Adjacent frames, no gap
            if start_track.frame not in output_frames:
                result.append(start_track)
                output_frames.add(start_track.frame)
            continue

        # Add start frame
        if start_track.frame not in output_frames:
            result.append(start_track)
            output_frames.add(start_track.frame)

        if gap_len <= max_gap_frames:
            # Interpolate the gap
            x0, y0 = start_track.image_xy
            x1, y1 = end_track.image_xy

            for g in range(1, gap_len + 1):
                t = g / (gap_len + 1)
                ix = x0 + t * (x1 - x0)
                iy = y0 + t * (y1 - y0)
                interp_frame = start_track.frame + g

                # Compute time
                fps_ratio = 0.0
                if start_track.time_s != end_track.time_s:
                    fps_ratio = (end_track.time_s - start_track.time_s) / (gap_len + 1)
                time_s = start_track.time_s + g * fps_ratio

                # Confidence decays toward middle of gap
                dist_to_edge = min(g, gap_len + 1 - g)
                conf = max(0.2, 0.7 * dist_to_edge / ((gap_len + 1) / 2))

                if interp_frame not in output_frames:
                    result.append(
                        BallTrack2D(
                            frame=interp_frame,
                            time_s=time_s,
                            image_xy=(ix, iy),
                            velocity_px_s=None,
                            confidence=conf,
                            state="interpolated",
                            interpolated=True,
                            gap_len=g,
                        )
                    )
                    output_frames.add(interp_frame)
        else:
            # Gap too long, mark as missing
            for g in range(1, gap_len + 1):
                missing_frame = start_track.frame + g
                if missing_frame not in output_frames:
                    fps_ratio = 0.0
                    if start_track.time_s != end_track.time_s:
                        fps_ratio = (end_track.time_s - start_track.time_s) / (gap_len + 1)
                    time_s = start_track.time_s + g * fps_ratio

                    result.append(
                        BallTrack2D(
                            frame=missing_frame,
                            time_s=time_s,
                            image_xy=start_track.image_xy,
                            velocity_px_s=None,
                            confidence=0.0,
                            state="missing",
                            interpolated=False,
                            gap_len=g,
                        )
                    )
                    output_frames.add(missing_frame)

    # Add the last detected frame
    last_track = tracks[detected_indices[-1]]
    if last_track.frame not in output_frames:
        result.append(last_track)
        output_frames.add(last_track.frame)

    # Sort by frame
    result.sort(key=lambda t: t.frame)
    return result


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
