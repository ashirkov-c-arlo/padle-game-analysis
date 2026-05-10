"""Evaluate ball tracking quality."""
from __future__ import annotations

import json
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
        raise click.ClickException(f"Predictions file missing columns: {sorted(missing)}")
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
        raise click.ClickException(f"Labels file missing columns: {sorted(missing)}")
    return df


def compute_metrics(pred: pd.DataFrame, labels: pd.DataFrame, distance_threshold: float) -> dict:
    labels_visible = labels[labels["visibility"].isin(["visible", "blurred"])]
    labels_not_visible = labels[labels["visibility"] == "not_visible"]
    pred_by_frame = pred.set_index("frame")

    matched_distances = []
    for _, label in labels_visible.iterrows():
        frame = label["frame"]
        if frame not in pred_by_frame.index:
            continue

        prediction = pred_by_frame.loc[frame]
        if isinstance(prediction, pd.DataFrame):
            prediction = prediction.iloc[0]
        distance = float(np.hypot(prediction["x_px"] - label["x_px"], prediction["y_px"] - label["y_px"]))
        if distance <= distance_threshold:
            matched_distances.append(distance)

    false_positives = 0
    for _, label in labels_not_visible.iterrows():
        frame = label["frame"]
        if frame not in pred_by_frame.index:
            continue

        prediction = pred_by_frame.loc[frame]
        if isinstance(prediction, pd.DataFrame):
            prediction = prediction.iloc[0]
        if prediction["state"] != "missing":
            false_positives += 1

    return {
        "detection_rate": len(matched_distances) / len(labels_visible) if len(labels_visible) > 0 else 0.0,
        "mean_error_px": float(np.mean(matched_distances)) if matched_distances else float("inf"),
        "median_error_px": float(np.median(matched_distances)) if matched_distances else float("inf"),
        "false_positive_rate": false_positives / len(labels_not_visible) if len(labels_not_visible) > 0 else 0.0,
        "visible_labels": len(labels_visible),
        "not_visible_labels": len(labels_not_visible),
    }


def print_report(metrics: dict, distance_threshold: float) -> bool:
    pass_detection = metrics["detection_rate"] > 0.6
    pass_error = metrics["mean_error_px"] < 15.0
    pass_fp = metrics["false_positive_rate"] < 0.1
    overall = pass_detection and pass_error and pass_fp

    print("\n" + "=" * 60)
    print("  BALL TRACKING EVALUATION REPORT")
    print("=" * 60)
    print(f"  Distance threshold: {distance_threshold:.1f} px")
    print(f"  Visible labels:     {metrics['visible_labels']}")
    print(f"  Not-visible labels: {metrics['not_visible_labels']}")
    print(f"  Detection rate:     {metrics['detection_rate']:.3f}")
    print(f"  Mean error:         {metrics['mean_error_px']:.2f} px")
    print(f"  Median error:       {metrics['median_error_px']:.2f} px")
    print(f"  FP rate:            {metrics['false_positive_rate']:.3f}")
    print("-" * 60)
    print(f"  Detection rate > 0.6: {'PASS' if pass_detection else 'FAIL'}")
    print(f"  Mean error < 15px:    {'PASS' if pass_error else 'FAIL'}")
    print(f"  FP rate < 0.1:        {'PASS' if pass_fp else 'FAIL'}")
    print(f"  OVERALL:              {'PASS' if overall else 'FAIL'}")
    print("=" * 60 + "\n")
    return overall


@click.command()
@click.option("--pred", required=True, type=click.Path(exists=True), help="Path to ball_tracks.csv")
@click.option("--labels", required=True, type=click.Path(exists=True), help="Path to ball_centers.jsonl")
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

    logger.info("Loading predictions: {}", pred_path)
    pred_df = load_predictions(pred_path)
    logger.debug("Predicted rows loaded: {}", len(pred_df))

    logger.info("Loading ground truth: {}", labels_path)
    labels_df = load_labels(labels_path)
    logger.debug("Labeled rows loaded: {}", len(labels_df))

    logger.info("Evaluating ball tracking: distance_threshold={:.1f}px", distance_threshold)
    metrics = compute_metrics(pred_df, labels_df, distance_threshold)
    if not print_report(metrics, distance_threshold):
        raise click.ClickException("Ball tracking quality gates failed")


if __name__ == "__main__":
    main()
