"""Logging helpers used across the package.

The library emits records through the standard ``logging`` module. CLI entry
points call :func:`configure_cli_logging` once so that command-line output
remains readable while the underlying modules stay silent when imported from a
test or notebook.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def configure_cli_logging(level: str | int | None = None) -> None:
    """Configure root logging for the CLI.

    The log level comes from, in order: the explicit ``level`` argument, the
    ``CFB_LOG_LEVEL`` environment variable, or ``INFO``. Repeated calls are
    no-ops so sub-commands can configure logging without duplicating handlers.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    resolved = level if level is not None else os.environ.get("CFB_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger. Convenience for consistency across modules."""
    return logging.getLogger(name)
