from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from src.schemas import BallDetection2D


class BallDetector:
    """
    Ball detection using a combination of approaches:
    1. If a trained TrackNet model is available in data/models/: use it
    2. Otherwise: use color-based + motion-based detection as baseline

    The baseline detector:
    - Background subtraction (MOG2) to find moving objects
    - Filter by size (ball is small: ~5-20px diameter at FullHD)
    - Filter by circularity
    - Filter by color (yellow/green tennis ball or white ball)
    - Confidence based on how many criteria match
    """

    def __init__(self, config: dict):
        """Load config, try to load trained model, fallback to heuristic."""
        bt_config = config.get("ball_tracking", {})
        self._confidence_threshold = bt_config.get("confidence_threshold", 0.4)
        self._model_path = self._find_model(config)
        self._model = None

        # Heuristic detector parameters
        self._min_area = 20  # min contour area in px^2
        self._max_area = 900  # max contour area in px^2 (~30px diameter)
        self._min_circularity = 0.5

        # Background subtractor
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=40, detectShadows=False
        )

        # Color ranges for ball detection (HSV)
        # Yellow/green ball
        self._yellow_lower = np.array([20, 80, 80], dtype=np.uint8)
        self._yellow_upper = np.array([45, 255, 255], dtype=np.uint8)
        # White ball
        self._white_lower = np.array([0, 0, 200], dtype=np.uint8)
        self._white_upper = np.array([180, 40, 255], dtype=np.uint8)

        if self._model_path:
            logger.info("TrackNet model found at {}", self._model_path)
        else:
            logger.info("No TrackNet model found, using heuristic ball detector")

    def _find_model(self, config: dict) -> str | None:
        """Try to locate a trained TrackNet model file."""
        models_dir = config.get("models", {}).get("cache_dir", "data/models")
        model_dir = Path(models_dir)
        if not model_dir.exists():
            return None

        # Look for common model file patterns
        for pattern in ["tracknet*.pt", "tracknet*.onnx", "ball_detector*.pt"]:
            matches = list(model_dir.glob(pattern))
            if matches:
                return str(matches[0])
        return None

    def detect_frame(
        self, frame: np.ndarray, prev_frames: list[np.ndarray] | None = None
    ) -> BallDetection2D | None:
        """Detect ball in single frame. prev_frames for temporal context."""
        if self._model is not None:
            return self._detect_with_model(frame, prev_frames)
        return self._detect_heuristic(frame)

    def _detect_with_model(
        self, frame: np.ndarray, prev_frames: list[np.ndarray] | None
    ) -> BallDetection2D | None:
        """Placeholder for TrackNet model inference."""
        # Would load and run the trained model here
        # For now, fall back to heuristic
        return self._detect_heuristic(frame)

    def _detect_heuristic(self, frame: np.ndarray) -> BallDetection2D | None:
        """Detect ball using motion + color + shape heuristics."""
        # Step 1: Background subtraction for motion
        fg_mask = self._bg_subtractor.apply(frame)

        # Clean up motion mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Step 2: Color mask for ball
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, self._yellow_lower, self._yellow_upper)
        white_mask = cv2.inRange(hsv, self._white_lower, self._white_upper)
        color_mask = cv2.bitwise_or(yellow_mask, white_mask)

        # Step 3: Combine motion and color
        combined = cv2.bitwise_and(fg_mask, color_mask)

        # Also consider motion-only candidates (ball might not match color perfectly)
        candidates = self._find_candidates(combined, frame, score_boost=0.2)
        motion_candidates = self._find_candidates(fg_mask, frame, score_boost=0.0)

        all_candidates = candidates + motion_candidates

        if not all_candidates:
            return None

        # Pick the best candidate
        best = max(all_candidates, key=lambda c: c["score"])

        if best["score"] < self._confidence_threshold:
            return None

        return BallDetection2D(
            frame=0,  # Will be set by caller
            time_s=0.0,  # Will be set by caller
            image_xy=(best["x"], best["y"]),
            confidence=min(1.0, best["score"]),
            visibility="visible",
            source="heuristic",
        )

    def _find_candidates(
        self, mask: np.ndarray, frame: np.ndarray, score_boost: float
    ) -> list[dict]:
        """Find ball candidates from a binary mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._min_area or area > self._max_area:
                continue

            # Circularity check
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if circularity < self._min_circularity:
                continue

            # Compute center
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            # Score based on circularity, size match, and position
            # Ideal ball area is around 100-300 px^2
            ideal_area = 150.0
            size_score = 1.0 - min(1.0, abs(area - ideal_area) / ideal_area)
            circ_score = circularity

            score = (size_score * 0.3 + circ_score * 0.5 + score_boost) * 0.8 + 0.1

            candidates.append({"x": cx, "y": cy, "score": score, "area": area})

        return candidates

    def detect_video(
        self, video_path: str, court_roi: np.ndarray | None = None
    ) -> list[BallDetection2D]:
        """Run detection on full video, optionally masking to court ROI."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        detections: list[BallDetection2D] = []

        # Reset background subtractor for fresh video
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=40, detectShadows=False
        )

        logger.info("Running ball detection on {} frames", total_frames)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply court ROI mask if available
            if court_roi is not None:
                masked = cv2.bitwise_and(frame, frame, mask=court_roi)
            else:
                masked = frame

            detection = self.detect_frame(masked)
            if detection is not None:
                time_s = frame_idx / fps if fps > 0 else 0.0
                detection = BallDetection2D(
                    frame=frame_idx,
                    time_s=time_s,
                    image_xy=detection.image_xy,
                    confidence=detection.confidence,
                    visibility=detection.visibility,
                    source=detection.source,
                )
                detections.append(detection)

            frame_idx += 1
            if frame_idx % 500 == 0:
                logger.debug("Ball detection progress: {}/{}", frame_idx, total_frames)

        cap.release()
        logger.info("Ball detection complete: {} detections in {} frames", len(detections), frame_idx)
        return detections
