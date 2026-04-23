# tests/test_cli.py
import argparse
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
    assert cfg.output_path == Path(f"stars_results_{date.today()}.csv")
    assert cfg.run_mode == "today"
    assert cfg.run_mode_date_from is None
    assert cfg.run_mode_date_to is None


def test_optional_filters():
    parser = _build_parser()
    args = parser.parse_args([
        "--strata-ids", "84",
        "--collection-id", "COL123",
        "--entity-id", "E01",
        "--patient-type", "Inpatient",
        "--service-line", "Cardiology",
        "--output", "/tmp/out.csv",
    ])
    cfg = _build_run_config(args)
    assert cfg.collection_id == "COL123"
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


def test_run_mode_defaults_to_today():
    parser = _build_parser()
    args = parser.parse_args(["--strata-ids", "84"])
    cfg = _build_run_config(args)
    assert cfg.run_mode == "today"
    assert cfg.run_mode_date_from is None
    assert cfg.run_mode_date_to is None


def test_run_mode_most_recent():
    parser = _build_parser()
    args = parser.parse_args(["--strata-ids", "84", "--run-mode", "most-recent"])
    cfg = _build_run_config(args)
    assert cfg.run_mode == "most-recent"


def test_run_mode_date_single_date_normalizes():
    parser = _build_parser()
    args = parser.parse_args([
        "--strata-ids", "84",
        "--run-mode", "date-range",
        "--run-mode-date", "2025-03-15",
    ])
    cfg = _build_run_config(args)
    assert cfg.run_mode == "date-range"
    assert cfg.run_mode_date_from == date(2025, 3, 15)
    assert cfg.run_mode_date_to == date(2025, 3, 15)


def test_run_mode_date_range_explicit():
    parser = _build_parser()
    args = parser.parse_args([
        "--strata-ids", "84",
        "--run-mode", "date-range",
        "--run-mode-date-from", "2025-01-01",
        "--run-mode-date-to", "2025-01-31",
    ])
    cfg = _build_run_config(args)
    assert cfg.run_mode == "date-range"
    assert cfg.run_mode_date_from == date(2025, 1, 1)
    assert cfg.run_mode_date_to == date(2025, 1, 31)


def test_run_mode_and_collection_id_are_mutually_exclusive():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--strata-ids", "84",
            "--run-mode", "today",
            "--collection-id", "COL123",
        ])


def test_date_range_requires_date_args():
    parser = _build_parser()
    args = parser.parse_args([
        "--strata-ids", "84",
        "--run-mode", "date-range",
    ])
    with pytest.raises(argparse.ArgumentTypeError):
        _build_run_config(args)


def test_run_mode_date_only_valid_with_date_range():
    parser = _build_parser()
    args = parser.parse_args([
        "--strata-ids", "84",
        "--run-mode", "today",
        "--run-mode-date", "2025-03-15",
    ])
    with pytest.raises(argparse.ArgumentTypeError):
        _build_run_config(args)


def test_collection_id_sets_run_mode_none():
    parser = _build_parser()
    args = parser.parse_args([
        "--strata-ids", "84",
        "--collection-id", "COL123",
    ])
    cfg = _build_run_config(args)
    assert cfg.collection_id == "COL123"
    assert cfg.run_mode is None
