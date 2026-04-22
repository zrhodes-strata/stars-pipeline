# tests/stars/test_warnings.py
import pandas as pd
from pathlib import Path

from stars_pipeline.stars.warnings import build_warnings_df, write_warnings_csv


_WARNING_COLS = [
    "warning_type", "strata_id", "entity_id", "run_mode",
    "requested_date", "fallback_date", "message",
]


def test_build_warnings_df_empty():
    df = build_warnings_df([])
    assert list(df.columns) == _WARNING_COLS
    assert len(df) == 0


def test_build_warnings_df_today_fallback():
    warnings = [{
        "warning_type": "today_fallback",
        "strata_id": None,
        "entity_id": None,
        "run_mode": "today",
        "requested_date": "today",
        "fallback_date": None,
        "message": "No runs found for today; fell back to most-recent",
    }]
    df = build_warnings_df(warnings)
    assert len(df) == 1
    assert df.iloc[0]["warning_type"] == "today_fallback"
    assert df.iloc[0]["entity_id"] is None or pd.isna(df.iloc[0]["entity_id"])


def test_build_warnings_df_skipped_pair():
    warnings = [{
        "warning_type": "skipped_pair",
        "strata_id": 84,
        "entity_id": 1001,
        "run_mode": "date-range",
        "requested_date": "2025-01-01 to 2025-01-31",
        "fallback_date": None,
        "message": "No run found for (84, 1001) in date-range",
    }]
    df = build_warnings_df(warnings)
    assert len(df) == 1
    assert df.iloc[0]["strata_id"] == 84
    assert df.iloc[0]["entity_id"] == 1001


def test_write_warnings_csv_creates_file(tmp_path):
    out = tmp_path / "results.csv"
    warnings = [{
        "warning_type": "today_fallback",
        "strata_id": None,
        "entity_id": None,
        "run_mode": "today",
        "requested_date": "today",
        "fallback_date": None,
        "message": "No runs found for today; fell back to most-recent",
    }]
    write_warnings_csv(warnings, out)
    warnings_path = tmp_path / "results_warnings.csv"
    assert warnings_path.exists()
    df = pd.read_csv(warnings_path)
    assert list(df.columns) == _WARNING_COLS
    assert len(df) == 1


def test_write_warnings_csv_empty_creates_file_with_headers(tmp_path):
    out = tmp_path / "stars_pilots2.csv"
    write_warnings_csv([], out)
    warnings_path = tmp_path / "stars_pilots2_warnings.csv"
    assert warnings_path.exists()
    df = pd.read_csv(warnings_path)
    assert list(df.columns) == _WARNING_COLS
    assert len(df) == 0
