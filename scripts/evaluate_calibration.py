"""Evaluate court calibration quality."""
from __future__ import annotations

import json
from pathlib import Path

import click
import cv2
import numpy as np
from loguru import logger

from src.calibration.line_detection import detect_lines_deeplsd
from src.calibration.line_filtering import filter_court_lines
from src.calibration.template_fitting import get_court_template_lines
from src.logging_config import LOG_LEVELS, configure_logging
from src.schemas import CourtGeometry2D, CourtRegistration2D
from src.video_io.reader import get_video_info, read_frame

REPROJECTION_THRESHOLD_PX = 10.0


def load_registration(path: Path) -> CourtRegistration2D:
    with open(path) as f:
        data = json.load(f)
    return CourtRegistration2D(**data)


def load_frames_from_video(video_path: Path, interval_s: float = 2.0) -> list[tuple[str, np.ndarray]]:
    info = get_video_info(str(video_path))
    fps = info["fps"]
    total = info["total_frames"]
    step = max(1, int(fps * interval_s))

    frames = []
    for idx in range(0, total, step):
        frame = read_frame(str(video_path), idx)
        frames.append((f"frame_{idx:06d}", frame))
    return frames


def project_template_lines_to_image(
    geometry: CourtGeometry2D, H_court_to_image: np.ndarray
) -> np.ndarray:
    template_lines = get_court_template_lines(geometry)
    projected = []
    for line in template_lines:
        pts = np.array([[line[0], line[1], 1.0], [line[2], line[3], 1.0]])
        proj = (H_court_to_image @ pts.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        projected.append([proj[0, 0], proj[0, 1], proj[1, 0], proj[1, 1]])
    return np.array(projected, dtype=np.float64)


def point_to_segment_distance(px: float, py: float, seg: np.ndarray) -> float:
    x1, y1, x2, y2 = seg
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return np.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)


def compute_reprojection_error(
    projected_lines: np.ndarray, detected_lines: np.ndarray
) -> tuple[float, int]:
    if len(detected_lines) == 0 or len(projected_lines) == 0:
        return float("inf"), 0

    matched_count = 0
    total_error = 0.0

    for proj_line in projected_lines:
        px1, py1, px2, py2 = proj_line
        sample_points = np.linspace(0, 1, 10)
        pts = np.column_stack([
            px1 + sample_points * (px2 - px1),
            py1 + sample_points * (py2 - py1),
        ])

        min_distances = []
        for pt in pts:
            dists = np.array([
                point_to_segment_distance(pt[0], pt[1], det)
                for det in detected_lines
            ])
            min_distances.append(dists.min())

        line_error = np.mean(min_distances)
        if line_error < 50.0:
            matched_count += 1
            total_error += line_error

    if matched_count == 0:
        return float("inf"), 0

    return total_error / matched_count, matched_count


def draw_overlay(frame: np.ndarray, projected_lines: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    for line in projected_lines:
        pt1 = (int(round(line[0])), int(round(line[1])))
        pt2 = (int(round(line[2])), int(round(line[3])))
        cv2.line(overlay, pt1, pt2, (0, 255, 0), 2)
    return overlay


@click.command()
@click.option(
    "--registration", required=True, type=click.Path(exists=True),
    help="Path to court_registration.json",
)
@click.option(
    "--frames", required=True, type=click.Path(exists=True),
    help="Path to .mp4 video",
)
@click.option(
    "--out", default=None, type=click.Path(),
    help="Output directory for overlay images",
)
@click.option(
    "--log-level",
    default=None,
    type=click.Choice(LOG_LEVELS, case_sensitive=False),
    help="Log level. Defaults to PADEL_CV_LOG_LEVEL or INFO.",
)
def main(registration: str, frames: str, out: str | None, log_level: str | None) -> None:
    """Evaluate court calibration quality by reprojecting court lines onto frames."""
    configure_logging(log_level)
    reg_path = Path(registration)
    frames_path = Path(frames)

    reg = load_registration(reg_path)
    logger.info("Loaded registration: mode={}, confidence={:.3f}", reg.mode, reg.confidence)

    if reg.homography_court_to_image is None:
        raise click.ClickException(
            "Registration has no homography_court_to_image. Cannot evaluate."
        )

    H_court_to_image = np.array(reg.homography_court_to_image, dtype=np.float64)
    geometry = CourtGeometry2D()

    if frames_path.suffix.lower() == ".mp4":
        frame_list = load_frames_from_video(frames_path)
    else:
        raise click.ClickException(
            f"--frames must be a .mp4 file, got: {frames_path}"
        )

    logger.info("Evaluating calibration: frames={}", len(frame_list))

    out_dir = Path(out) if out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    projected_lines = project_template_lines_to_image(geometry, H_court_to_image)

    per_frame_errors: list[tuple[str, float, int]] = []

    for name, frame in frame_list:
        detected_raw = detect_lines_deeplsd(frame)
        detected = filter_court_lines(detected_raw, frame.shape)
        error, matched = compute_reprojection_error(projected_lines, detected)
        per_frame_errors.append((name, error, matched))
        logger.debug("{}: error={:.2f}px, matched_lines={}", name, error, matched)

        if out_dir:
            overlay = draw_overlay(frame, projected_lines)
            cv2.imwrite(str(out_dir / f"overlay_{name}.png"), overlay)

    valid_errors = [e for _, e, _ in per_frame_errors if e != float("inf")]
    mean_error = np.mean(valid_errors) if valid_errors else float("inf")
    total_matched = sum(m for _, _, m in per_frame_errors)
    passed = mean_error < REPROJECTION_THRESHOLD_PX

    print("\n" + "=" * 60)
    print("CALIBRATION EVALUATION REPORT")
    print("=" * 60)
    print(f"Registration mode:         {reg.mode}")
    print(f"Registration confidence:   {reg.confidence:.3f}")
    print(f"Frames evaluated:          {len(frame_list)}")
    print(f"Mean reprojection error:   {mean_error:.2f} px")
    print(f"Total matched model lines: {total_matched}")
    print(f"Threshold:                 {REPROJECTION_THRESHOLD_PX:.1f} px")
    print(f"Result:                    {'PASS' if passed else 'FAIL'}")
    print("-" * 60)
    print("Per-frame breakdown:")
    for name, error, matched in per_frame_errors:
        err_str = f"{error:.2f}" if error != float("inf") else "inf"
        print(f"  {name:30s}  error={err_str:>8s} px  matched={matched}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
