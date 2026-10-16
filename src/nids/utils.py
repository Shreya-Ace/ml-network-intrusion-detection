"""Small shared helpers (logging, filesystem)."""

from __future__ import annotations

import logging
from pathlib import Path

from nids import config

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def ensure_directories(directories: tuple[Path, ...] = config.REQUIRED_DIRS) -> None:
    """Create the project's working directories if they are missing."""
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a console logger with a consistent format.

    Safe to call repeatedly: handlers are only attached once per logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    return logger
