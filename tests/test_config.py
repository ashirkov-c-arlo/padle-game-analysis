"""Tests for config loading."""
from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from src.config.loader import load_config


def test_load_default_config():
    cfg = load_config()
    assert cfg["court"]["width_m"] == 10.0
    assert cfg["court"]["length_m"] == 20.0
    assert cfg["court"]["net_y_m"] == 10.0


def test_court_geometry_values():
    cfg = load_config()
    assert cfg["court"]["service_line_offset_from_net_m"] == 6.95
    assert cfg["court"]["net_height_center_m"] == 0.88
    assert cfg["court"]["net_height_posts_m"] == 0.92
    assert len(cfg["court"]["lines"]) == 9


def test_detection_config():
    cfg = load_config()
    assert cfg["detection"]["model"] == "yolo11n"
    assert cfg["detection"]["confidence_threshold"] == 0.5
    assert cfg["detection"]["person_class_id"] == 0


def test_tracking_config():
    cfg = load_config()
    assert cfg["tracking"]["method"] == "bytetrack"
    assert cfg["tracking"]["max_active_players"] == 4


def test_calibration_config():
    cfg = load_config()
    assert cfg["calibration"]["method"] == "deeplsd"
    assert cfg["calibration"]["max_reprojection_error_px"] == 10.0


def test_ball_tracking_config():
    cfg = load_config()
    assert cfg["ball_tracking"]["model"] == "tracknetv4"
    assert cfg["ball_tracking"]["confidence_threshold"] == 0.4


def test_export_config():
    cfg = load_config()
    assert "json" in cfg["export"]["formats"]
    assert "csv" in cfg["export"]["formats"]


def test_override_merge():
    override = {"court": {"width_m": 12.0}, "detection": {"confidence_threshold": 0.7}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(override, f)
        tmp_path = f.name

    cfg = load_config(tmp_path)
    # Overridden values
    assert cfg["court"]["width_m"] == 12.0
    assert cfg["detection"]["confidence_threshold"] == 0.7
    # Non-overridden values remain
    assert cfg["court"]["length_m"] == 20.0
    assert cfg["detection"]["model"] == "yolo11n"

    Path(tmp_path).unlink()


def test_zones_config():
    cfg = load_config()
    assert cfg["zones"]["net_distance_from_net_m"] == 3.5
    assert cfg["zones"]["mid_distance_from_net_m"] == 6.95


def test_smoothing_config():
    cfg = load_config()
    assert cfg["smoothing"]["method"] == "savgol"
    assert cfg["smoothing"]["window_frames"] == 7
    assert cfg["smoothing"]["max_speed_mps"] == 8.0
