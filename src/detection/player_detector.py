from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger
from ultralytics import YOLO

from src.schemas import PlayerDetection


class PlayerDetector:
    """YOLO11 person detector wrapper."""

    def __init__(self, config: dict):
        """Initialize detector from config.

        Args:
            config: Full pipeline config dict with 'detection' and 'models' sections.
        """
        det_cfg = config.get("detection", {})
        models_cfg = config.get("models", {})

        self._model_name = det_cfg.get("model", "yolo11n")
        self._confidence_threshold = det_cfg.get("confidence_threshold", 0.5)
        self._inference_confidence_threshold = det_cfg.get(
            "inference_confidence_threshold",
            self._confidence_threshold,
        )
        self._person_class_id = det_cfg.get("person_class_id", 0)
        self._max_detections = det_cfg.get("max_detections_per_frame", 10)
        self._cache_dir = Path(models_cfg.get("cache_dir", "data/models"))

        self._model: YOLO | None = None
        logger.debug(
            (
                "PlayerDetector config: model={}, confidence_threshold={}, "
                "inference_confidence_threshold={}, person_class_id={}, max_detections={}"
            ),
            self._model_name,
            self._confidence_threshold,
            self._inference_confidence_threshold,
            self._person_class_id,
            self._max_detections,
        )

    def _ensure_model(self) -> YOLO:
        """Lazy-load the YOLO model on first use."""
        if self._model is not None:
            return self._model

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        model_path = self._cache_dir / f"{self._model_name}.pt"
        if model_path.exists():
            logger.info("Loading YOLO model from {}", model_path)
            self._model = YOLO(str(model_path))
        else:
            logger.info("Downloading YOLO model '{}' to {}", self._model_name, self._cache_dir)
            self._model = YOLO(f"{self._model_name}.pt")
            # Move downloaded weights to cache dir
            default_path = Path(f"{self._model_name}.pt")
            if default_path.exists():
                default_path.rename(model_path)

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug("YOLO model '{}' loaded on device: {}", self._model_name, device)
        return self._model.to(device)

    def detect_frame(self, frame: np.ndarray) -> list[PlayerDetection]:
        """Run YOLO on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of PlayerDetection for persons above inference confidence threshold.
        """
        model = self._ensure_model()
        results = model(frame, verbose=False, conf=self._inference_confidence_threshold)

        detections: list[PlayerDetection] = []
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(classes)):
            if classes[i] != self._person_class_id:
                continue
            if confs[i] < self._inference_confidence_threshold:
                continue
            detections.append(
                PlayerDetection(
                    frame=0,
                    bbox_xyxy=(float(xyxy[i, 0]), float(xyxy[i, 1]),
                               float(xyxy[i, 2]), float(xyxy[i, 3])),
                    cls=int(classes[i]),
                    confidence=float(confs[i]),
                )
            )

        # Keep only top-N by confidence
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[: self._max_detections]

    def detect_video(
        self, video_path: str, frame_indices: list[int] | None = None
    ) -> dict[int, list[PlayerDetection]]:
        """Run detection on video frames.

        Args:
            video_path: Path to video file.
            frame_indices: If provided, detect only these frame indices.
                Otherwise detect all frames.

        Returns:
            Dict mapping frame index to list of PlayerDetection.
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open video: {}", video_path)
            return {}

        target_set = set(frame_indices) if frame_indices is not None else None

        results: dict[int, list[PlayerDetection]] = {}
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if target_set is not None and frame_idx not in target_set:
                frame_idx += 1
                continue

            detections = self.detect_frame(frame)
            # Update frame index on each detection
            for det in detections:
                det.frame = frame_idx
            results[frame_idx] = detections

            frame_idx += 1

            if target_set is not None and frame_idx > max(target_set):
                break

        cap.release()
        logger.debug("Detected persons: frames={}, video={}", len(results), video_path)
        return results

    def detect_batch(
        self, frames: list[np.ndarray], start_frame: int = 0
    ) -> dict[int, list[PlayerDetection]]:
        """Run detection on a batch of frames.

        Args:
            frames: List of BGR images as numpy arrays.
            start_frame: Frame index of the first frame in the batch.

        Returns:
            Dict mapping frame index to list of PlayerDetection.
        """
        model = self._ensure_model()
        results_map: dict[int, list[PlayerDetection]] = {}

        if not frames:
            return results_map

        # YOLO supports batch inference with a list of frames
        all_results = model(frames, verbose=False, conf=self._inference_confidence_threshold)

        for i, results in enumerate(all_results):
            frame_idx = start_frame + i
            detections: list[PlayerDetection] = []
            boxes = results.boxes

            if boxes is None or len(boxes) == 0:
                results_map[frame_idx] = detections
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)

            for j in range(len(classes)):
                if classes[j] != self._person_class_id:
                    continue
                if confs[j] < self._inference_confidence_threshold:
                    continue
                detections.append(
                    PlayerDetection(
                        frame=frame_idx,
                        bbox_xyxy=(float(xyxy[j, 0]), float(xyxy[j, 1]),
                                   float(xyxy[j, 2]), float(xyxy[j, 3])),
                        cls=int(classes[j]),
                        confidence=float(confs[j]),
                    )
                )

            detections.sort(key=lambda d: d.confidence, reverse=True)
            results_map[frame_idx] = detections[: self._max_detections]

        return results_map
