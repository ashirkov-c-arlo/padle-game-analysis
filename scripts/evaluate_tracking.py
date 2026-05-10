"""Evaluate player tracking quality against ground truth labels."""
from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd
from loguru import logger

from src.logging_config import LOG_LEVELS, configure_logging


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def load_predictions(pred_path: Path) -> pd.DataFrame:
    df = pd.read_csv(pred_path)
    required = {"frame", "player_id", "x1", "y1", "x2", "y2", "confidence"}
    missing = required - set(df.columns)
    if missing:
        raise click.ClickException(f"Predictions CSV missing columns: {sorted(missing)}")
    return df


def load_labels(labels_path: Path) -> dict:
    with open(labels_path) as f:
        data = json.load(f)
    if "frames" not in data:
        raise click.ClickException("Labels JSON missing 'frames' key")
    return data


def _pred_boxes(preds_in_frame: pd.DataFrame) -> list[dict]:
    return [
        {
            "player_id": row["player_id"],
            "bbox": [row["x1"], row["y1"], row["x2"], row["y2"]],
        }
        for _, row in preds_in_frame.iterrows()
    ]


def evaluate(pred_df: pd.DataFrame, labels: dict, iou_threshold: float) -> dict:
    gt_frames = labels["frames"]
    all_frames = sorted(set(int(f) for f in gt_frames.keys()) | set(pred_df["frame"].unique()))

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_id_switches = 0
    total_gt = 0
    duplicate_frames = 0
    prev_matches: dict[str, str] = {}

    for frame_num in all_frames:
        frame_str = str(frame_num)
        gt_players = gt_frames.get(frame_str, {}).get("players", [])
        pred_boxes = _pred_boxes(pred_df[pred_df["frame"] == frame_num])
        total_gt += len(gt_players)

        for i in range(len(pred_boxes)):
            for j in range(i + 1, len(pred_boxes)):
                if compute_iou(pred_boxes[i]["bbox"], pred_boxes[j]["bbox"]) > 0.7:
                    duplicate_frames += 1

        matched_pred = set()
        matches = []
        for gt in gt_players:
            best_iou = 0.0
            best_pi = -1
            for pi, pred in enumerate(pred_boxes):
                if pi in matched_pred:
                    continue
                iou = compute_iou(gt["bbox_xyxy"], pred["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_pi = pi
            if best_iou >= iou_threshold and best_pi >= 0:
                matches.append((gt["player_id"], pred_boxes[best_pi]["player_id"]))
                matched_pred.add(best_pi)

        total_tp += len(matches)
        total_fp += len(pred_boxes) - len(matches)
        total_fn += len(gt_players) - len(matches)

        for gt_id, pred_id in matches:
            if gt_id in prev_matches and prev_matches[gt_id] != pred_id:
                total_id_switches += 1
            prev_matches[gt_id] = pred_id

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    mota = 1.0 - (total_fp + total_fn + total_id_switches) / total_gt if total_gt > 0 else 0.0
    duration_minutes = len(all_frames) / (30.0 * 60.0)
    id_switches_per_minute = total_id_switches / duration_minutes if duration_minutes > 0 else 0.0
    has_permanent_duplicates = duplicate_frames > (len(all_frames) * 0.1)

    return {
        "precision": precision,
        "recall": recall,
        "mota": mota,
        "id_switches": total_id_switches,
        "id_switches_per_minute": id_switches_per_minute,
        "duplicate_frames": duplicate_frames,
        "has_permanent_duplicates": has_permanent_duplicates,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_gt": total_gt,
        "num_frames": len(all_frames),
    }


def print_report(metrics: dict) -> bool:
    mota_pass = metrics["mota"] > 0.5
    idsw_pass = metrics["id_switches_per_minute"] < 10
    dup_pass = not metrics["has_permanent_duplicates"]
    overall = mota_pass and idsw_pass and dup_pass

    print("\n" + "=" * 60)
    print("  TRACKING EVALUATION REPORT")
    print("=" * 60)
    print(f"  Frames:          {metrics['num_frames']}")
    print(f"  Precision:       {metrics['precision']:.4f}")
    print(f"  Recall:          {metrics['recall']:.4f}")
    print(f"  MOTA:            {metrics['mota']:.4f}")
    print(f"  TP/FP/FN:        {metrics['total_tp']} / {metrics['total_fp']} / {metrics['total_fn']}")
    print(f"  ID switches:     {metrics['id_switches']}")
    print(f"  ID switches/min: {metrics['id_switches_per_minute']:.2f}")
    print(f"  Duplicate frames:{metrics['duplicate_frames']}")
    print("-" * 60)
    print(f"  MOTA > 0.5:              {'PASS' if mota_pass else 'FAIL'}")
    print(f"  ID switches < 10/min:    {'PASS' if idsw_pass else 'FAIL'}")
    print(f"  No permanent duplicates: {'PASS' if dup_pass else 'FAIL'}")
    print(f"  OVERALL:                 {'PASS' if overall else 'FAIL'}")
    print("=" * 60 + "\n")
    return overall


@click.command()
@click.option("--pred", required=True, type=click.Path(exists=True), help="Path to predicted tracks CSV")
@click.option("--labels", required=True, type=click.Path(exists=True), help="Path to ground truth labels JSON")
@click.option("--iou-threshold", default=0.5, type=float, help="IoU threshold for matching")
@click.option(
    "--log-level",
    default=None,
    type=click.Choice(LOG_LEVELS, case_sensitive=False),
    help="Log level. Defaults to PADEL_CV_LOG_LEVEL or INFO.",
)
def main(pred: str, labels: str, iou_threshold: float, log_level: str | None) -> None:
    """Evaluate player tracking quality against ground truth."""
    configure_logging(log_level)
    pred_path = Path(pred)
    labels_path = Path(labels)

    logger.info("Loading predictions: {}", pred_path)
    pred_df = load_predictions(pred_path)
    logger.debug("Prediction rows loaded: {}", len(pred_df))

    logger.info("Loading ground truth: {}", labels_path)
    gt_labels = load_labels(labels_path)
    logger.debug("Ground-truth frames loaded: {}", len(gt_labels["frames"]))

    logger.info("Evaluating tracking: iou_threshold={}", iou_threshold)
    metrics = evaluate(pred_df, gt_labels, iou_threshold)
    if not print_report(metrics):
        raise click.ClickException("Tracking quality gates failed")


if __name__ == "__main__":
    main()
