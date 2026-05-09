"""Evaluate player tracking quality against ground truth labels."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
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

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def load_predictions(pred_path: Path) -> pd.DataFrame:
    df = pd.read_csv(pred_path)
    expected_cols = {"frame", "player_id", "x1", "y1", "x2", "y2", "confidence"}
    missing = expected_cols - set(df.columns)
    if missing:
        logger.error("Predictions CSV missing columns: {}", sorted(missing))
        sys.exit(1)
    return df


def load_labels(labels_path: Path) -> dict:
    with open(labels_path) as f:
        data = json.load(f)
    if "frames" not in data:
        logger.error("Labels JSON missing 'frames' key")
        sys.exit(1)
    return data


def evaluate(pred_df: pd.DataFrame, labels: dict, iou_threshold: float) -> dict:
    gt_frames = labels["frames"]
    all_frames = sorted(set(int(f) for f in gt_frames.keys()) | set(pred_df["frame"].unique()))

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_id_switches = 0
    total_gt_count = 0

    prev_matches: dict[str, str] = {}

    per_player_gt_frames: dict[str, int] = defaultdict(int)
    per_player_matched_frames: dict[str, int] = defaultdict(int)

    duplicate_frames = 0

    for frame_num in all_frames:
        frame_str = str(frame_num)
        gt_players = gt_frames.get(frame_str, {}).get("players", [])
        preds_in_frame = pred_df[pred_df["frame"] == frame_num]

        total_gt_count += len(gt_players)

        for gt in gt_players:
            per_player_gt_frames[gt["player_id"]] += 1

        pred_boxes = []
        for _, row in preds_in_frame.iterrows():
            pred_boxes.append({
                "player_id": row["player_id"],
                "bbox": [row["x1"], row["y1"], row["x2"], row["y2"]],
            })

        for i in range(len(pred_boxes)):
            for j in range(i + 1, len(pred_boxes)):
                iou_ij = compute_iou(pred_boxes[i]["bbox"], pred_boxes[j]["bbox"])
                if iou_ij > 0.7:
                    duplicate_frames += 1

        matched_gt = set()
        matched_pred = set()

        matches = []
        for gi, gt in enumerate(gt_players):
            gt_bbox = gt["bbox_xyxy"]
            best_iou = 0.0
            best_pi = -1
            for pi, pred in enumerate(pred_boxes):
                if pi in matched_pred:
                    continue
                iou = compute_iou(gt_bbox, pred["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_pi = pi
            if best_iou >= iou_threshold and best_pi >= 0:
                matches.append((gi, best_pi, best_iou))
                matched_gt.add(gi)
                matched_pred.add(best_pi)

        tp = len(matches)
        fp = len(pred_boxes) - tp
        fn = len(gt_players) - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

        for gi, pi, _ in matches:
            gt_id = gt_players[gi]["player_id"]
            pred_id = pred_boxes[pi]["player_id"]
            per_player_matched_frames[gt_id] += 1

            if gt_id in prev_matches:
                if prev_matches[gt_id] != pred_id:
                    total_id_switches += 1
            prev_matches[gt_id] = pred_id

    fragmentation = {}
    for player_id in per_player_gt_frames:
        player_preds = pred_df[pred_df["player_id"] == player_id].sort_values("frame")
        if len(player_preds) == 0:
            fragmentation[player_id] = 0
            continue
        frames = player_preds["frame"].values
        gaps = 0
        for i in range(1, len(frames)):
            if frames[i] - frames[i - 1] > 1:
                gaps += 1
        fragmentation[player_id] = gaps

    id_correct = 0
    id_total = 0
    for frame_num in all_frames:
        frame_str = str(frame_num)
        gt_players = gt_frames.get(frame_str, {}).get("players", [])
        preds_in_frame = pred_df[pred_df["frame"] == frame_num]

        pred_boxes = []
        for _, row in preds_in_frame.iterrows():
            pred_boxes.append({
                "player_id": row["player_id"],
                "bbox": [row["x1"], row["y1"], row["x2"], row["y2"]],
            })

        matched_pred_2 = set()
        for gt in gt_players:
            gt_bbox = gt["bbox_xyxy"]
            best_iou = 0.0
            best_pi = -1
            for pi, pred in enumerate(pred_boxes):
                if pi in matched_pred_2:
                    continue
                iou = compute_iou(gt_bbox, pred["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_pi = pi
            if best_iou >= iou_threshold and best_pi >= 0:
                matched_pred_2.add(best_pi)
                id_total += 1
                if gt["player_id"] == pred_boxes[best_pi]["player_id"]:
                    id_correct += 1

    id_consistency = id_correct / id_total if id_total > 0 else 0.0

    track_completeness = {}
    for player_id, gt_count in per_player_gt_frames.items():
        matched = per_player_matched_frames.get(player_id, 0)
        track_completeness[player_id] = matched / gt_count if gt_count > 0 else 0.0

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    mota = 1.0 - (total_fp + total_fn + total_id_switches) / total_gt_count if total_gt_count > 0 else 0.0

    num_frames = len(all_frames)
    duration_minutes = num_frames / (30.0 * 60.0)
    id_switches_per_minute = total_id_switches / duration_minutes if duration_minutes > 0 else 0.0

    has_permanent_duplicates = duplicate_frames > (num_frames * 0.1)

    return {
        "precision": precision,
        "recall": recall,
        "mota": mota,
        "id_switches": total_id_switches,
        "id_switches_per_minute": id_switches_per_minute,
        "id_consistency": id_consistency,
        "fragmentation": fragmentation,
        "track_completeness": track_completeness,
        "duplicate_frames": duplicate_frames,
        "has_permanent_duplicates": has_permanent_duplicates,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_gt": total_gt_count,
        "num_frames": num_frames,
    }


def print_report(metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("  TRACKING EVALUATION REPORT")
    print("=" * 60)

    print("\n--- Detection Metrics (IoU threshold applied) ---")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  TP: {metrics['total_tp']}  FP: {metrics['total_fp']}  FN: {metrics['total_fn']}")

    print("\n--- Identity Metrics ---")
    print(f"  ID Switches:         {metrics['id_switches']}")
    print(f"  ID Switches/min:     {metrics['id_switches_per_minute']:.2f}")
    print(f"  ID Consistency:      {metrics['id_consistency']:.4f}")

    print("\n--- Track Fragmentation ---")
    for player_id, gaps in metrics["fragmentation"].items():
        print(f"  {player_id}: {gaps} fragments")

    print("\n--- Track Completeness ---")
    for player_id, completeness in metrics["track_completeness"].items():
        print(f"  {player_id}: {completeness:.4f}")

    print("\n--- MOTA ---")
    print(f"  MOTA: {metrics['mota']:.4f}")

    print("\n--- Duplicate Detection ---")
    print(f"  Frames with duplicates: {metrics['duplicate_frames']}")
    print(f"  Permanent duplicates:   {'YES' if metrics['has_permanent_duplicates'] else 'NO'}")

    print("\n" + "=" * 60)
    print("  QUALITY GATES")
    print("=" * 60)

    mota_pass = metrics["mota"] > 0.5
    idsw_pass = metrics["id_switches_per_minute"] < 10
    dup_pass = not metrics["has_permanent_duplicates"]

    print(f"  MOTA > 0.5:              {'PASS' if mota_pass else 'FAIL'} ({metrics['mota']:.4f})")
    print(f"  ID switches < 10/min:    {'PASS' if idsw_pass else 'FAIL'} ({metrics['id_switches_per_minute']:.2f})")
    print(f"  No permanent duplicates: {'PASS' if dup_pass else 'FAIL'}")

    overall = mota_pass and idsw_pass and dup_pass
    print(f"\n  OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 60 + "\n")


@click.command()
@click.option("--pred", required=True, type=click.Path(), help="Path to predicted tracks CSV")
@click.option("--labels", required=True, type=click.Path(), help="Path to ground truth labels JSON")
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

    if not pred_path.exists():
        logger.error("Predictions file not found: {}", pred_path)
        sys.exit(1)
    if not labels_path.exists():
        logger.error("Labels file not found: {}", labels_path)
        sys.exit(1)

    logger.info("Loading predictions: {}", pred_path)
    pred_df = load_predictions(pred_path)
    logger.debug("Prediction rows loaded: {}", len(pred_df))

    logger.info("Loading ground truth: {}", labels_path)
    gt_labels = load_labels(labels_path)
    num_gt_frames = len(gt_labels["frames"])
    logger.debug("Ground-truth frames loaded: {}", num_gt_frames)

    logger.info("Evaluating tracking: iou_threshold={}", iou_threshold)
    metrics = evaluate(pred_df, gt_labels, iou_threshold)

    print_report(metrics)

    overall_pass = (
        metrics["mota"] > 0.5
        and metrics["id_switches_per_minute"] < 10
        and not metrics["has_permanent_duplicates"]
    )
    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
