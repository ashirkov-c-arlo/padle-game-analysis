from __future__ import annotations

import io

import pytest
from loguru import logger

from src.logging_config import configure_logging, normalize_log_level


def test_info_logging_is_concise(monkeypatch):
    monkeypatch.delenv("PADEL_CV_LOG_LEVEL", raising=False)
    sink = io.StringIO()

    configure_logging("INFO", sink=sink)
    logger.info("Pipeline ready: frames={}", 12)
    logger.debug("hidden detail")

    output = sink.getvalue()
    assert "INFO" in output
    assert "Pipeline ready: frames=12" in output
    assert "hidden detail" not in output
    assert "test_logging_config" not in output


def test_debug_logging_includes_source_location(monkeypatch):
    monkeypatch.delenv("PADEL_CV_LOG_LEVEL", raising=False)
    sink = io.StringIO()

    configure_logging("DEBUG", sink=sink)
    logger.debug("debug detail: {}", "roi")

    output = sink.getvalue()
    assert "DEBUG" in output
    assert "debug detail: roi" in output
    assert "test_logging_config" in output


def test_normalize_log_level_uses_environment(monkeypatch):
    monkeypatch.setenv("PADEL_CV_LOG_LEVEL", "debug")

    assert normalize_log_level() == "DEBUG"


def test_normalize_log_level_rejects_unknown_level(monkeypatch):
    monkeypatch.delenv("PADEL_CV_LOG_LEVEL", raising=False)

    with pytest.raises(ValueError, match="Unsupported log level"):
        normalize_log_level("verbose")
