# tests/test_logging_config.py
import json
import logging
from io import StringIO

from stars_pipeline.logging_config import configure_logging, get_logger


def test_json_output_contains_required_fields(capsys):
    configure_logging(level=logging.INFO)
    logger = get_logger("test.module")
    logger.info("hello world")
    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert payload["level"] == "INFO"
    assert payload["message"] == "hello world"
    assert payload["logger"] == "test.module"
    assert "timestamp" in payload


def test_extra_fields_are_included(capsys):
    configure_logging(level=logging.INFO)
    logger = get_logger("test.extra")
    logger.info("with extras", extra={"strata_id": 84, "n_segments": 10})
    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert payload["strata_id"] == 84
    assert payload["n_segments"] == 10
