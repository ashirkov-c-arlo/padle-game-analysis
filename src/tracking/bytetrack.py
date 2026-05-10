from __future__ import annotations

import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment

from src.schemas import PlayerDetection


def _bbox_to_xyah(bbox_xyxy: tuple[float, float, float, float]) -> np.ndarray:
    """Convert (x1,y1,x2,y2) to (cx, cy, aspect_ratio, height)."""
    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    a = w / h if h > 0 else 0.0
    return np.array([cx, cy, a, h], dtype=np.float64)


def _xyah_to_xyxy(xyah: np.ndarray) -> tuple[float, float, float, float]:
    """Convert (cx, cy, aspect_ratio, height) to (x1,y1,x2,y2)."""
    cx, cy, a, h = xyah[:4]
    w = a * h
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of bboxes in xyxy format.

    Args:
        bboxes_a: (N, 4) array of bboxes.
        bboxes_b: (M, 4) array of bboxes.

    Returns:
        (N, M) IoU matrix.
    """
    n = bboxes_a.shape[0]
    m = bboxes_b.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)

    # Intersection
    x1 = np.maximum(bboxes_a[:, 0:1], bboxes_b[:, 0:1].T)  # (N, M)
    y1 = np.maximum(bboxes_a[:, 1:2], bboxes_b[:, 1:2].T)
    x2 = np.minimum(bboxes_a[:, 2:3], bboxes_b[:, 2:3].T)
    y2 = np.minimum(bboxes_a[:, 3:4], bboxes_b[:, 3:4].T)

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Areas
    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou


class KalmanBoxTracker:
    """Kalman filter for a single tracked bounding box.

    State: [cx, cy, a, h, vx, vy, va, vh] (center, aspect ratio, height + velocities).
    """

    _count = 0

    def __init__(self, bbox_xyxy: tuple[float, float, float, float]):
        KalmanBoxTracker._count += 1
        self.track_id = KalmanBoxTracker._count

        # State: [cx, cy, a, h, vx, vy, va, vh]
        xyah = _bbox_to_xyah(bbox_xyxy)
        self.x = np.zeros(8, dtype=np.float64)
        self.x[:4] = xyah

        # State covariance
        self.P = np.eye(8, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 100.0  # high uncertainty on velocities

        # Process noise
        self.Q = np.eye(8, dtype=np.float64) * 1.0
        self.Q[4:, 4:] *= 0.01

        # Measurement noise
        self.R = np.eye(4, dtype=np.float64) * 1.0

        # Transition matrix (constant velocity)
        self.F = np.eye(8, dtype=np.float64)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0
        self.F[3, 7] = 1.0

        # Measurement matrix
        self.H = np.zeros((4, 8), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        self.hits = 1
        self.time_since_update = 0
        self.age = 0

    def predict(self) -> tuple[float, float, float, float]:
        """Predict next state and return predicted bbox in xyxy."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        # Ensure height and aspect ratio stay positive
        self.x[2] = max(self.x[2], 0.01)
        self.x[3] = max(self.x[3], 1.0)
        return _xyah_to_xyxy(self.x)

    def update(self, bbox_xyxy: tuple[float, float, float, float]) -> None:
        """Update state with observed measurement."""
        z = _bbox_to_xyah(bbox_xyxy)
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # Update
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.hits += 1
        self.time_since_update = 0

    def get_bbox(self) -> tuple[float, float, float, float]:
        """Get current bbox in xyxy format."""
        return _xyah_to_xyxy(self.x)


class ByteTracker:
    """ByteTrack multi-object tracker for padel players."""

    def __init__(self, config: dict):
        """Initialize ByteTracker from config.

        Args:
            config: Full pipeline config or the 'tracking' section from default.yaml.
                Key params live under 'tracking.bytetrack' in the full config, or 'bytetrack'
                when only the tracking section is passed.
        """
        tracking_cfg = config.get("tracking", {})
        bt_cfg = (
            tracking_cfg.get("bytetrack", {})
            if isinstance(tracking_cfg, dict) and "bytetrack" in tracking_cfg
            else config.get("bytetrack", {})
        )
        self._track_high_thresh = bt_cfg.get("track_thresh", 0.5)
        self._track_low_thresh = bt_cfg.get("track_low_thresh", 0.1)
        self._new_track_thresh = bt_cfg.get("new_track_thresh", self._track_high_thresh)
        self._match_thresh = bt_cfg.get("match_thresh", 0.8)
        self._track_buffer = bt_cfg.get("track_buffer", 30)
        self._min_box_area = bt_cfg.get("min_box_area", 100)
        self._frame_rate = bt_cfg.get("frame_rate", 30)

        self._active_tracks: list[KalmanBoxTracker] = []
        self._lost_tracks: list[KalmanBoxTracker] = []
        self._removed_tracks: list[KalmanBoxTracker] = []

        # Accumulate all track history: {track_id: [{frame, bbox_xyxy, confidence}, ...]}
        self._track_history: dict[int, list[dict]] = {}

        self._frame_count = 0
        logger.debug(
            (
                "ByteTracker initialized: high_thresh={}, low_thresh={}, new_track_thresh={}, "
                "match_thresh={}, buffer={}, min_box_area={}, frame_rate={}"
            ),
            self._track_high_thresh,
            self._track_low_thresh,
            self._new_track_thresh,
            self._match_thresh,
            self._track_buffer,
            self._min_box_area,
            self._frame_rate,
        )

    def _bbox_area(self, bbox: tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _match_tracks(
        self,
        tracks: list[KalmanBoxTracker],
        detections: list[tuple[tuple[float, float, float, float], float]],
        thresh: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Match tracks to detections using IoU + Hungarian algorithm.

        Returns:
            (matches, unmatched_track_indices, unmatched_det_indices)
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Get predicted bboxes for tracks
        track_bboxes = np.array([t.get_bbox() for t in tracks], dtype=np.float64)
        det_bboxes = np.array([d[0] for d in detections], dtype=np.float64)

        iou_matrix = _iou_batch(track_bboxes, det_bboxes)

        # Use 1 - IoU as cost for Hungarian algorithm
        cost_matrix = 1.0 - iou_matrix

        # Apply threshold: set costs above threshold to a large value
        cost_matrix[iou_matrix < (1.0 - thresh)] = 1e5

        if cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        else:
            row_indices, col_indices = np.array([]), np.array([])

        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] >= 1e5:
                continue
            matches.append((int(r), int(c)))
            if r in unmatched_tracks:
                unmatched_tracks.remove(r)
            if c in unmatched_dets:
                unmatched_dets.remove(c)

        return matches, unmatched_tracks, unmatched_dets

    def update(self, detections: list[PlayerDetection], frame_idx: int) -> list[dict]:
        """Feed detections for one frame, return active tracks.

        Args:
            detections: List of PlayerDetection for this frame.
            frame_idx: Current frame index.

        Returns:
            List of active track dicts: {track_id, bbox_xyxy, confidence, frame}.
        """
        self._frame_count += 1

        # Filter by min box area
        valid_dets = [
            d for d in detections if self._bbox_area(d.bbox_xyxy) >= self._min_box_area
        ]

        # Split into high and low confidence detections
        high_dets = [
            (d.bbox_xyxy, d.confidence)
            for d in valid_dets
            if d.confidence >= self._track_high_thresh
        ]
        low_dets = [
            (d.bbox_xyxy, d.confidence)
            for d in valid_dets
            if d.confidence < self._track_high_thresh and d.confidence >= self._track_low_thresh
        ]

        # Predict new locations for all active tracks
        for track in self._active_tracks:
            track.predict()

        # Also predict lost tracks
        for track in self._lost_tracks:
            track.predict()

        # --- First association: high-confidence detections with active tracks ---
        matches_1, unmatched_tracks_1, unmatched_dets_1 = self._match_tracks(
            self._active_tracks, high_dets, self._match_thresh
        )

        # Update matched tracks
        for t_idx, d_idx in matches_1:
            track = self._active_tracks[t_idx]
            bbox, conf = high_dets[d_idx]
            track.update(bbox)
            self._record_track(track.track_id, frame_idx, bbox, conf)

        # --- Second association: low-confidence detections with remaining active tracks ---
        remaining_tracks = [self._active_tracks[i] for i in unmatched_tracks_1]
        matches_2, _, _ = self._match_tracks(
            remaining_tracks, low_dets, 0.5  # lower IoU threshold for low-conf
        )

        matched_low_track_indices = set()
        for t_idx, d_idx in matches_2:
            track = remaining_tracks[t_idx]
            bbox, conf = low_dets[d_idx]
            track.update(bbox)
            self._record_track(track.track_id, frame_idx, bbox, conf)
            matched_low_track_indices.add(unmatched_tracks_1[t_idx])

        # --- Third association: remaining active tracks with lost tracks recovery ---
        # Try to match unmatched high dets with lost tracks
        remaining_high_dets = [high_dets[i] for i in unmatched_dets_1]
        matches_3, _, unmatched_dets_3 = self._match_tracks(
            self._lost_tracks, remaining_high_dets, self._match_thresh
        )

        recovered_tracks = set()
        for t_idx, d_idx in matches_3:
            track = self._lost_tracks[t_idx]
            bbox, conf = remaining_high_dets[d_idx]
            track.update(bbox)
            self._record_track(track.track_id, frame_idx, bbox, conf)
            self._active_tracks.append(track)
            recovered_tracks.add(t_idx)

        self._lost_tracks = [
            t for i, t in enumerate(self._lost_tracks) if i not in recovered_tracks
        ]

        # --- Handle unmatched tracks: move to lost ---
        # Rebuild actual unmatched indices after second association
        still_unmatched = [
            i for i in unmatched_tracks_1
            if i not in matched_low_track_indices
        ]

        new_lost = []
        for i in still_unmatched:
            track = self._active_tracks[i]
            new_lost.append(track)

        self._active_tracks = [
            t for i, t in enumerate(self._active_tracks)
            if i not in still_unmatched
        ]
        self._lost_tracks.extend(new_lost)

        # --- Create new tracks from unmatched high-confidence detections ---
        created_tracks = 0
        for d_idx in unmatched_dets_3:
            bbox, conf = remaining_high_dets[d_idx]
            if conf >= self._new_track_thresh:
                new_track = KalmanBoxTracker(bbox)
                self._active_tracks.append(new_track)
                self._record_track(new_track.track_id, frame_idx, bbox, conf)
                created_tracks += 1

        # --- Remove old lost tracks ---
        lost_before_prune = len(self._lost_tracks)
        self._lost_tracks = [
            t for t in self._lost_tracks
            if t.time_since_update <= self._track_buffer
        ]
        pruned_lost_tracks = lost_before_prune - len(self._lost_tracks)

        if frame_idx % 500 == 0:
            logger.debug(
                (
                    "ByteTrack frame {}: input={}, valid={}, high={}, low={}, "
                    "matched_high={}, matched_low={}, recovered={}, created={}, active={}, lost={}, pruned={}"
                ),
                frame_idx,
                len(detections),
                len(valid_dets),
                len(high_dets),
                len(low_dets),
                len(matches_1),
                len(matches_2),
                len(recovered_tracks),
                created_tracks,
                len(self._active_tracks),
                len(self._lost_tracks),
                pruned_lost_tracks,
            )

        # --- Build output ---
        output = []
        for track in self._active_tracks:
            bbox = track.get_bbox()
            # Get latest confidence from history
            hist = self._track_history.get(track.track_id, [])
            conf = hist[-1]["confidence"] if hist else 0.0
            output.append({
                "track_id": track.track_id,
                "bbox_xyxy": bbox,
                "confidence": conf,
                "frame": frame_idx,
            })

        return output

    def _record_track(
        self,
        track_id: int,
        frame_idx: int,
        bbox: tuple[float, float, float, float],
        confidence: float,
    ) -> None:
        """Record a track observation in history."""
        if track_id not in self._track_history:
            self._track_history[track_id] = []
        self._track_history[track_id].append({
            "frame": frame_idx,
            "bbox_xyxy": bbox,
            "confidence": confidence,
        })

    def get_tracks(self) -> dict[int, list[dict]]:
        """Return all accumulated tracks as {track_id: [{frame, bbox_xyxy, confidence}, ...]}.

        Returns:
            Dict mapping track_id to list of observations.
        """
        return dict(self._track_history)

    def reset(self) -> None:
        """Reset tracker state."""
        self._active_tracks = []
        self._lost_tracks = []
        self._removed_tracks = []
        self._track_history = {}
        self._frame_count = 0
        KalmanBoxTracker._count = 0
