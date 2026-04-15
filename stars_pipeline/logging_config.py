"""
logging_config.py
=================
Structured stdout logging for the STARS pipeline.

Emits one JSON object per log record to stdout. SageMaker Processing Jobs
capture stdout/stderr to CloudWatch Logs automatically — no additional
logging configuration is needed in the job definition.

Usage
-----
    from stars_pipeline.logging_config import configure_logging, get_logger

    # Call once at process startup (in cli.py main()):
    configure_logging()

    # Then anywhere in the codebase:
    logger = get_logger(__name__)
    logger.info("Processing segment", extra={"strata_id": 84})
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

# Fields that are part of LogRecord internals — exclude them from extras
_INTERNAL_FIELDS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


class _JsonFormatter(logging.Formatter):
    """
    Formats each log record as a single JSON line.

    Required fields in every record:
        timestamp  ISO-8601 UTC string
        level      Log level name (INFO, WARNING, ERROR, ...)
        logger     Logger name (module path)
        message    Formatted log message

    Any extra fields passed via ``extra=`` are merged into the top-level
    JSON object. Non-JSON-serialisable values are coerced to strings.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _INTERNAL_FIELDS or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except TypeError:
                payload[key] = str(value)
        return json.dumps(payload)


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger to emit JSON lines to stdout.

    Call exactly once at process startup — typically inside ``cli.main()``.
    Subsequent calls replace the existing handler configuration.

    Args:
        level: Minimum log level (default: logging.INFO).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Call at module level:
        logger = get_logger(__name__)

    Args:
        name: Logger name — use ``__name__`` for the calling module.

    Returns:
        A standard ``logging.Logger`` instance.
    """
    return logging.getLogger(name)
