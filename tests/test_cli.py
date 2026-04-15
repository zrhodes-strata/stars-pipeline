# tests/test_cli.py
from datetime import date
from pathlib import Path

import pytest

from stars_pipeline.cli import _build_parser, _build_run_config


def test_required_strata_ids():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # --strata-ids is required


def test_strata_ids_parsed_to_int_list():
    parser = _build_parser()
    args = parser.parse_args(["--strata-ids", "84,14,1318"])
    cfg = _build_run_config(args)
    assert cfg.strata_ids == [84, 14, 1318]


def test_defaults():
    parser = _build_parser()
    args = parser.parse_args(["--strata-ids", "84"])
    cfg = _build_run_config(args)
    assert cfg.date_from == date(2022, 1, 1)
    assert cfg.date_to == date.today()
    assert cfg.recent_days == 90
    assert cfg.train_days is None
    assert cfg.collection_id is None
    assert cfg.run_id is None
    assert cfg.output_path == Path(f"stars_results_{date.today()}.csv")


def test_optional_filters():
    parser = _build_parser()
    args = parser.parse_args([
        "--strata-ids", "84",
        "--collection-id", "COL123",
        "--run-id", "RUN456",
        "--entity-id", "E01",
        "--patient-type", "Inpatient",
        "--service-line", "Cardiology",
        "--output", "/tmp/out.csv",
    ])
    cfg = _build_run_config(args)
    assert cfg.collection_id == "COL123"
    assert cfg.run_id == "RUN456"
    assert cfg.entity_id == "E01"
    assert cfg.patient_type == "Inpatient"
    assert cfg.service_line == "Cardiology"
    assert cfg.output_path == Path("/tmp/out.csv")


def test_custom_date_from_and_to():
    parser = _build_parser()
    args = parser.parse_args([
        "--strata-ids", "84",
        "--date-from", "2023-06-15",
        "--date-to", "2024-12-31",
    ])
    cfg = _build_run_config(args)
    assert cfg.date_from == date(2023, 6, 15)
    assert cfg.date_to == date(2024, 12, 31)
    assert cfg.output_path == Path("stars_results_2024-12-31.csv")


def test_invalid_date_format_raises():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--strata-ids", "84", "--date-from", "2023/06/15"])
