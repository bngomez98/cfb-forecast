"""Logging configuration shared by the CLI, the pipeline and the source collectors.

Console output goes to stderr. A run can also write a UTC-timestamped log file;
the pipeline puts one per run in ``CFB_LOG_DIR`` (default ``<artifacts>/logs``).
The level comes from ``CFB_LOG_LEVEL`` (default INFO).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATEFMT = "%Y-%m-%dT%H:%M:%SZ"
_FILE_HANDLER = "cfb-file"


class _UTCFormatter(logging.Formatter):
    converter = time.gmtime


def _level(level=None):
    value = level or os.environ.get("CFB_LOG_LEVEL", "INFO")
    return value if isinstance(value, int) else logging.getLevelName(str(value).upper())


def configure(level=None):
    """Install a single console handler on the package logger (idempotent)."""
    logger = logging.getLogger("cfb_forecast")
    logger.setLevel(_level(level))
    logger.propagate = False
    if not any(getattr(h, "name", None) == "cfb-console" for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.name = "cfb-console"
        handler.setFormatter(_UTCFormatter(FORMAT, DATEFMT))
        logger.addHandler(handler)
    return logger


def add_file(log_dir, run_name, slot=_FILE_HANDLER):
    """Also write records to ``<log_dir>/<run_name>_<UTC timestamp>.log``.

    Each ``slot`` holds one file at a time: a pipeline run replaces the previous run's
    file, while the scheduler keeps its own session file in a separate slot.
    """
    logger = configure()
    for handler in [h for h in logger.handlers if getattr(h, "name", None) == slot]:
        logger.removeHandler(handler)
        handler.close()
    folder = Path(log_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{run_name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.name = slot
    handler.setFormatter(_UTCFormatter(FORMAT, DATEFMT))
    logger.addHandler(handler)
    return path


def get(name):
    return logging.getLogger(f"cfb_forecast.{name}")


def close_file(slot=_FILE_HANDLER):
    """Detach and close the file handler in ``slot`` (end of a pipeline run)."""
    logger = logging.getLogger("cfb_forecast")
    for handler in [h for h in logger.handlers if getattr(h, "name", None) == slot]:
        logger.removeHandler(handler)
        handler.close()
