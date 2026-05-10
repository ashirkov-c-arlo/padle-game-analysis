from __future__ import annotations

import logging
import os
import sys
from typing import TextIO

from loguru import logger

LOG_LEVELS = ("TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_VALID_LEVELS = set(LOG_LEVELS)
_DEFAULT_LEVEL = "INFO"

_INFO_FORMAT = "{time:HH:mm:ss} | {level:<7} | {message}"
_DEBUG_FORMAT = "{time:HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}"


def normalize_log_level(level: str | None = None) -> str:
    """Return a supported Loguru level name."""
    normalized = (level or os.getenv("PADEL_CV_LOG_LEVEL") or _DEFAULT_LEVEL).upper()
    if normalized not in _VALID_LEVELS:
        allowed = ", ".join(sorted(_VALID_LEVELS))
        raise ValueError(f"Unsupported log level '{level}'. Expected one of: {allowed}")
    return normalized


def configure_logging(level: str | None = None, *, sink: TextIO | None = None) -> str:
    """Configure project logging and return the active level."""
    active_level = normalize_log_level(level)
    detailed = active_level in {"TRACE", "DEBUG"}
    log_sink = sink or sys.stderr

    logger.remove()
    logger.add(
        log_sink,
        level=active_level,
        format=_DEBUG_FORMAT if detailed else _INFO_FORMAT,
        colorize=False,
        backtrace=detailed,
        diagnose=detailed,
    )
    logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
    logger.debug("Logging configured: level={}", active_level)
    return active_level
