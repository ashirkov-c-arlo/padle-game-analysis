from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detection.player_detector import PlayerDetector
from src.detection.roi_filter import (
    filter_detections_by_court_roi,
    get_footpoint,
    project_to_court,
)
from src.schemas import CourtRegistration2D, PlayerDetection


@pytest.fixture
def default_config():
    return {
        "detection": {
            "model": "yolo11n",
            "confidence_threshold": 0.5,
            "person_class_id": 0,
            "max_detections_per_frame": 10,
        },
        "models": {
            "cache_dir": "data/models",
            "lazy_download": True,
        },
    }


class TestPlayerDetectorInit:
    def test_instantiation_with_config(self, default_config):
        detector = PlayerDetector(default_config)
        assert detector._model_name == "yolo11n"
        assert detector._confidence_threshold == 0.5
        assert detector._person_class_id == 0
        assert detector._max_detections == 10
        assert detector._model is None  # lazy loading

    def test_instantiation_with_defaults(self):
        detector = PlayerDetector({})
        assert detector._model_name == "yolo11n"
        assert detector._confidence_threshold == 0.5
        assert detector._person_class_id == 0
        assert detector._max_detections == 10

    def test_custom_config(self):
        config = {
            "detection": {
                "model": "yolo11s",
                "confidence_threshold": 0.7,
                "person_class_id": 0,
                "max_detections_per_frame": 4,
            },
            "models": {"cache_dir": "/tmp/models"},
        }
        detector = PlayerDetector(config)
        assert detector._model_name == "yolo11s"
        assert detector._confidence_threshold == 0.7
        assert detector._max_detections == 4


class TestDetectFrame:
    @patch("src.detection.player_detector.YOLO")
    def test_detect_frame_returns_player_detections(self, mock_yolo_cls, default_config):
        # Set up mock model
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        # Mock YOLO results
        import torch

        mock_boxes = MagicMock()
        mock_boxes.xyxy = torch.tensor([[100.0, 50.0, 200.0, 300.0],
                                         [300.0, 100.0, 400.0, 350.0]])
        mock_boxes.conf = torch.tensor([0.9, 0.7])
        mock_boxes.cls = torch.tensor([0.0, 0.0])
        mock_boxes.__len__ = lambda self: 2

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]

        detector = PlayerDetector(default_config)
        detector._model = mock_model

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect_frame(frame)

        assert len(detections) == 2
        assert all(isinstance(d, PlayerDetection) for d in detections)
        assert detections[0].confidence == pytest.approx(0.9, abs=1e-6)
        assert detections[1].confidence == pytest.approx(0.7, abs=1e-6)

    @patch("src.detection.player_detector.YOLO")
    def test_detect_frame_filters_non_person_classes(self, mock_yolo_cls, default_config):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        import torch

        mock_boxes = MagicMock()
        mock_boxes.xyxy = torch.tensor([[100.0, 50.0, 200.0, 300.0],
                                         [300.0, 100.0, 400.0, 350.0]])
        mock_boxes.conf = torch.tensor([0.9, 0.8])
        mock_boxes.cls = torch.tensor([0.0, 1.0])  # person, bicycle
        mock_boxes.__len__ = lambda self: 2

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]

        detector = PlayerDetector(default_config)
        detector._model = mock_model

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect_frame(frame)

        assert len(detections) == 1
        assert detections[0].confidence == pytest.approx(0.9, abs=1e-6)

    @patch("src.detection.player_detector.YOLO")
    def test_detect_frame_respects_max_detections(self, mock_yolo_cls):
        config = {
            "detection": {
                "model": "yolo11n",
                "confidence_threshold": 0.3,
                "person_class_id": 0,
                "max_detections_per_frame": 2,
            },
            "models": {"cache_dir": "data/models"},
        }
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        import torch

        mock_boxes = MagicMock()
        mock_boxes.xyxy = torch.tensor([[10.0, 10.0, 50.0, 100.0]] * 5)
        mock_boxes.conf = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        mock_boxes.cls = torch.tensor([0.0] * 5)
        mock_boxes.__len__ = lambda self: 5

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]

        detector = PlayerDetector(config)
        detector._model = mock_model

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect_frame(frame)

        assert len(detections) == 2
        assert detections[0].confidence == pytest.approx(0.9, abs=1e-6)
        assert detections[1].confidence == pytest.approx(0.8, abs=1e-6)

    @patch("src.detection.player_detector.YOLO")
    def test_detect_frame_empty_results(self, mock_yolo_cls, default_config):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        mock_boxes = MagicMock()
        mock_boxes.__len__ = lambda self: 0

        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]

        detector = PlayerDetector(default_config)
        detector._model = mock_model

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect_frame(frame)

        assert detections == []


class TestGetFootpoint:
    def test_basic_footpoint(self):
        bbox = (100.0, 50.0, 200.0, 300.0)
        foot = get_footpoint(bbox)
        assert foot == (150.0, 300.0)

    def test_footpoint_zero_box(self):
        bbox = (0.0, 0.0, 0.0, 0.0)
        foot = get_footpoint(bbox)
        assert foot == (0.0, 0.0)

    def test_footpoint_asymmetric(self):
        bbox = (10.0, 20.0, 30.0, 100.0)
        foot = get_footpoint(bbox)
        assert foot == (20.0, 100.0)


class TestProjectToCourt:
    def test_identity_homography(self):
        H_identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        result = project_to_court((5.0, 10.0), H_identity)
        assert abs(result[0] - 5.0) < 1e-9
        assert abs(result[1] - 10.0) < 1e-9

    def test_translation_homography(self):
        # Translate by (2, 3)
        H = [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]]
        result = project_to_court((5.0, 10.0), H)
        assert abs(result[0] - 7.0) < 1e-9
        assert abs(result[1] - 13.0) < 1e-9

    def test_scaling_homography(self):
        # Scale by 0.5
        H = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]
        result = project_to_court((10.0, 20.0), H)
        assert abs(result[0] - 5.0) < 1e-9
        assert abs(result[1] - 10.0) < 1e-9


class TestFilterDetectionsByCourtROI:
    def test_no_registration_returns_all(self):
        detections = [
            PlayerDetection(frame=0, bbox_xyxy=(0, 0, 100, 200), confidence=0.9),
            PlayerDetection(frame=0, bbox_xyxy=(500, 500, 600, 700), confidence=0.8),
        ]
        result = filter_detections_by_court_roi(detections, None, (720, 1280))
        assert len(result) == 2

    def test_pixel_only_mode_returns_all(self):
        reg = CourtRegistration2D(mode="pixel_only", confidence=0.5)
        detections = [
            PlayerDetection(frame=0, bbox_xyxy=(0, 0, 100, 200), confidence=0.9),
        ]
        result = filter_detections_by_court_roi(detections, reg, (720, 1280))
        assert len(result) == 1

    def test_homography_filters_out_of_court(self):
        # Identity homography: pixel coords = court coords
        H = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        reg = CourtRegistration2D(
            mode="floor_homography",
            homography_image_to_court=H,
            confidence=0.9,
        )

        detections = [
            # Footpoint at (5, 10) - inside court (10x20)
            PlayerDetection(frame=0, bbox_xyxy=(4.0, 0.0, 6.0, 10.0), confidence=0.9),
            # Footpoint at (50, 100) - way outside court
            PlayerDetection(frame=0, bbox_xyxy=(40.0, 0.0, 60.0, 100.0), confidence=0.8),
        ]

        result = filter_detections_by_court_roi(detections, reg, (720, 1280), margin_px=1)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_margin_allows_near_boundary(self):
        H = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        reg = CourtRegistration2D(
            mode="floor_homography",
            homography_image_to_court=H,
            confidence=0.9,
        )

        # Footpoint at (-0.5, 10) - slightly outside left sideline
        detections = [
            PlayerDetection(frame=0, bbox_xyxy=(-1.5, 0.0, 0.5, 10.0), confidence=0.9),
        ]

        # With margin=1, should be included
        result = filter_detections_by_court_roi(detections, reg, (720, 1280), margin_px=1)
        assert len(result) == 1

        # With margin=0, should be excluded
        result = filter_detections_by_court_roi(detections, reg, (720, 1280), margin_px=0)
        assert len(result) == 0
