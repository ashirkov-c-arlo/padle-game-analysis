"""Evaluate ball tracking quality."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger

from src.logging_config import LOG_LEVELS, configure_logging


def load_predictions(pred_path: Path) -> pd.DataFrame:
    df = pd.read_csv(pred_path)
    required = {"frame", "x_px", "y_px", "confidence", "state", "interpolated"}
    missing = required - set(df.columns)
    if missing:
        logger.error("Predictions file missing columns: {}", sorted(missing))
        sys.exit(1)
    return df


def load_labels(labels_path: Path) -> pd.DataFrame:
    records = []
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    required = {"frame", "x_px", "y_px", "visibility"}
    missing = required - set(df.columns)
    if missing:
        logger.error("Labels file missing columns: {}", sorted(missing))
        sys.exit(1)
    return df


def _find_gaps(missing_frames: list[int]) -> list[list[int]]:
    if not missing_frames:
        return []
    gaps = []
    current_gap = [missing_frames[0]]
    for i in range(1, len(missing_frames)):
        if missing_frames[i] == missing_frames[i - 1] + 1:
            current_gap.append(missing_frames[i])
        else:
            gaps.append(current_gap)
            current_gap = [missing_frames[i]]
    gaps.append(current_gap)
    return gaps


def compute_metrics(pred: pd.DataFrame, labels: pd.DataFrame, distance_threshold: float) -> dict:
    labels_visible = labels[labels["visibility"].isin(["visible", "blurred"])].copy()
    labels_not_visible = labels[labels["visibility"] == "not_visible"].copy()

    pred_by_frame = pred.set_index("frame")

    matched_distances = []
    detected_count = 0
    total_visible = len(labels_visible)

    for _, row in labels_visible.iterrows():
        frame = row["frame"]
        if frame in pred_by_frame.index:
            p = pred_by_frame.loc[frame]
            if isinstance(p, pd.DataFrame):
                p = p.iloc[0]
            dist = np.sqrt((p["x_px"] - row["x_px"]) ** 2 + (p["y_px"] - row["y_px"]) ** 2)
            if dist <= distance_threshold:
                detected_count += 1
                matched_distances.append(dist)

    detection_rate = detected_count / total_visible if total_visible > 0 else 0.0
    mean_error = float(np.mean(matched_distances)) if matched_distances else float("inf")
    median_error = float(np.median(matched_distances)) if matched_distances else float("inf")

    fp_count = 0
    for _, row in labels_not_visible.iterrows():
        frame = row["frame"]
        if frame in pred_by_frame.index:
            p = pred_by_frame.loc[frame]
            if isinstance(p, pd.DataFrame):
                p = p.iloc[0]
            if p["state"] != "missing":
                fp_count += 1
    fp_rate = fp_count / len(labels_not_visible) if len(labels_not_visible) > 0 else 0.0

    pred_interpolated = pred[pred["interpolated"].astype(bool)]
    interp_distances = []
    for _, p in pred_interpolated.iterrows():
        frame = p["frame"]
        gt_row = labels_visible[labels_visible["frame"] == frame]
        if not gt_row.empty:
            gt = gt_row.iloc[0]
            dist = np.sqrt((p["x_px"] - gt["x_px"]) ** 2 + (p["y_px"] - gt["y_px"]) ** 2)
            interp_distances.append(dist)
    interp_mean_error = float(np.mean(interp_distances)) if interp_distances else float("nan")
    interp_count = len(interp_distances)

    pred_sorted = pred.sort_values("frame").reset_index(drop=True)
    all_frames = set(labels["frame"].unique())
    pred_frames_set = set(pred_sorted[pred_sorted["state"] != "missing"]["frame"].unique())

    missing_frames = sorted(all_frames - pred_frames_set)
    gaps = _find_gaps(missing_frames)

    short_gaps = [g for g in gaps if len(g) <= 10]
    long_gaps = [g for g in gaps if len(g) > 10]

    short_gap_frames = {f for g in short_gaps for f in g}
    short_gap_interpolated = pred_sorted[
        (pred_sorted["frame"].isin(short_gap_frames)) & (pred_sorted["interpolated"].astype(bool))
    ]
    short_gap_total = len(short_gap_frames)
    short_gap_filled = len(short_gap_interpolated)

    short_gap_interp_distances = []
    for _, p in short_gap_interpolated.iterrows():
        gt_row = labels_visible[labels_visible["frame"] == p["frame"]]
        if not gt_row.empty:
            gt = gt_row.iloc[0]
            dist = np.sqrt((p["x_px"] - gt["x_px"]) ** 2 + (p["y_px"] - gt["y_px"]) ** 2)
            short_gap_interp_distances.append(dist)
    short_gap_interp_accuracy = (
        float(np.mean(short_gap_interp_distances)) if short_gap_interp_distances else float("nan")
    )

    long_gap_frames = {f for g in long_gaps for f in g}
    long_gap_preds = pred_sorted[pred_sorted["frame"].isin(long_gap_frames)]
    long_gap_hallucinated = long_gap_preds[long_gap_preds["state"] != "missing"]
    long_gap_total = len(long_gap_frames)
    long_gap_correct = long_gap_total - len(long_gap_hallucinated)

    pred_sorted_state = pred_sorted.sort_values("frame")["state"].values
    breaks = 0
    for i in range(1, len(pred_sorted_state)):
        prev = pred_sorted_state[i - 1]
        curr = pred_sorted_state[i]
        if prev in ("detected", "tracked", "interpolated") and curr == "missing":
            breaks += 1
        elif prev == "missing" and curr in ("detected", "tracked", "interpolated"):
            breaks += 1

    visibility_breakdown = {}
    for vis in ["visible", "blurred", "occluded", "not_visible"]:
        subset = labels[labels["visibility"] == vis]
        if len(subset) == 0:
            continue
        if vis in ("visible", "blurred"):
            det = 0
            dists = []
            for _, row in subset.iterrows():
                frame = row["frame"]
                if frame in pred_by_frame.index:
                    p = pred_by_frame.loc[frame]
                    if isinstance(p, pd.DataFrame):
                        p = p.iloc[0]
                    dist = np.sqrt((p["x_px"] - row["x_px"]) ** 2 + (p["y_px"] - row["y_px"]) ** 2)
                    if dist <= distance_threshold:
                        det += 1
                        dists.append(dist)
            visibility_breakdown[vis] = {
                "count": len(subset),
                "detection_rate": det / len(subset),
                "mean_error_px": float(np.mean(dists)) if dists else float("nan"),
            }
        else:
            fp = 0
            for _, row in subset.iterrows():
                frame = row["frame"]
                if frame in pred_by_frame.index:
                    p = pred_by_frame.loc[frame]
                    if isinstance(p, pd.DataFrame):
                        p = p.iloc[0]
                    if p["state"] != "missing":
                        fp += 1
            visibility_breakdown[vis] = {
                "count": len(subset),
                "false_positive_rate": fp / len(subset),
            }

    return {
        "detection_rate": detection_rate,
        "mean_error_px": mean_error,
        "median_error_px": median_error,
        "false_positive_rate": fp_rate,
        "interpolation": {
            "count": interp_count,
            "mean_error_px": interp_mean_error,
        },
        "gap_handling": {
            "short_gaps_count": len(short_gaps),
            "short_gap_frames_total": short_gap_total,
            "short_gap_frames_filled": short_gap_filled,
            "short_gap_interp_mean_error_px": short_gap_interp_accuracy,
            "long_gaps_count": len(long_gaps),
            "long_gap_frames_total": long_gap_total,
            "long_gap_frames_correct": long_gap_correct,
        },
        "track_continuity_breaks": breaks,
        "per_visibility": visibility_breakdown,
    }


def print_report(metrics: dict, distance_threshold: float) -> bool:
    print("\n" + "=" * 60)
    print("  BALL TRACKING EVALUATION REPORT")
    print("=" * 60)

    print(f"\n  Distance threshold: {distance_threshold:.1f} px")

    print(f"\n  Detection rate:        {metrics['detection_rate']:.3f}")
    print(f"  Mean position error:   {metrics['mean_error_px']:.2f} px")
    print(f"  Median position error: {metrics['median_error_px']:.2f} px")
    print(f"  False positive rate:   {metrics['false_positive_rate']:.3f}")

    print("\n  --- Interpolation ---")
    interp = metrics["interpolation"]
    print(f"  Interpolated frames evaluated: {interp['count']}")
    print(f"  Mean interpolation error:      {interp['mean_error_px']:.2f} px")

    print("\n  --- Gap Handling ---")
    gh = metrics["gap_handling"]
    print(f"  Short gaps (<=10 frames): {gh['short_gaps_count']}")
    print(f"    Frames in short gaps:   {gh['short_gap_frames_total']}")
    print(f"    Frames interpolated:    {gh['short_gap_frames_filled']}")
    print(f"    Interpolation error:    {gh['short_gap_interp_mean_error_px']:.2f} px")
    print(f"  Long gaps (>10 frames):   {gh['long_gaps_count']}")
    print(f"    Frames in long gaps:    {gh['long_gap_frames_total']}")
    print(f"    Correctly missing:      {gh['long_gap_frames_correct']}")

    print("\n  --- Track Continuity ---")
    print(f"  Track breaks: {metrics['track_continuity_breaks']}")

    print("\n  --- Per-Visibility Breakdown ---")
    for vis, data in metrics["per_visibility"].items():
        print(f"  [{vis}] (n={data['count']})")
        if "detection_rate" in data:
            print(f"    Detection rate: {data['detection_rate']:.3f}")
            print(f"    Mean error:     {data['mean_error_px']:.2f} px")
        if "false_positive_rate" in data:
            print(f"    FP rate:        {data['false_positive_rate']:.3f}")

    pass_detection = metrics["detection_rate"] > 0.6
    pass_error = metrics["mean_error_px"] < 15.0
    pass_fp = metrics["false_positive_rate"] < 0.1
    overall = pass_detection and pass_error and pass_fp

    print("\n  --- Quality Gates ---")
    print(f"  Detection rate > 0.6:  {'PASS' if pass_detection else 'FAIL'} ({metrics['detection_rate']:.3f})")
    print(f"  Mean error < 15px:     {'PASS' if pass_error else 'FAIL'} ({metrics['mean_error_px']:.2f})")
    print(f"  FP rate < 0.1:         {'PASS' if pass_fp else 'FAIL'} ({metrics['false_positive_rate']:.3f})")
    print(f"\n  OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 60 + "\n")

    return overall


@click.command()
@click.option("--pred", required=True, type=click.Path(), help="Path to ball_tracks.csv")
@click.option("--labels", required=True, type=click.Path(), help="Path to ball_centers.jsonl")
@click.option("--distance-threshold", default=15.0, type=float, help="Max distance (px) to count as match")
@click.option(
    "--log-level",
    default=None,
    type=click.Choice(LOG_LEVELS, case_sensitive=False),
    help="Log level. Defaults to PADEL_CV_LOG_LEVEL or INFO.",
)
def main(pred: str, labels: str, distance_threshold: float, log_level: str | None) -> None:
    """Evaluate ball tracking quality against ground truth labels."""
    configure_logging(log_level)
    pred_path = Path(pred)
    labels_path = Path(labels)

    if not pred_path.exists():
        logger.error("Predictions file not found: {}", pred_path)
        sys.exit(1)
    if not labels_path.exists():
        logger.error("Labels file not found: {}", labels_path)
        sys.exit(1)

    logger.info("Loading predictions: {}", pred_path)
    pred_df = load_predictions(pred_path)
    logger.debug("Predicted frames loaded: {}", len(pred_df))

    logger.info("Loading ground truth: {}", labels_path)
    labels_df = load_labels(labels_path)
    logger.debug("Labeled frames loaded: {}", len(labels_df))

    logger.info("Evaluating ball tracking: distance_threshold={:.1f}px", distance_threshold)
    metrics = compute_metrics(pred_df, labels_df, distance_threshold)
    passed = print_report(metrics, distance_threshold)

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
