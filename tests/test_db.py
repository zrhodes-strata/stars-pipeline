# tests/test_db.py
import pandas as pd
import pytest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

from stars_pipeline.config import RunConfig
from stars_pipeline.db import fetch_actuals, _get_connection


def _make_run_cfg(**kwargs):
    defaults = dict(
        strata_ids=[84],
        collection_id=None,
        run_id=None,
        run_mode="today",
        run_mode_date_from=None,
        run_mode_date_to=None,
        date_from=date(2022, 1, 1),
        date_to=date(2026, 1, 1),
        recent_days=90,
        train_days=None,
        entity_id=None,
        patient_type=None,
        service_line=None,
        output_path=Path("out.csv"),
    )
    defaults.update(kwargs)
    return RunConfig(**defaults)


def test_missing_env_vars_raises(monkeypatch):
    for var in ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD",
                "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA"]:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(EnvironmentError, match="Missing required Snowflake"):
        _get_connection()


def test_fetch_actuals_returns_expected_columns(monkeypatch):
    # Set dummy env vars so _get_connection() doesn't raise
    for var in ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD",
                "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA"]:
        monkeypatch.setenv(var, "dummy")

    fake_df = pd.DataFrame({
        "STRATA_ID": [84, 84],
        "ENTITY_ID": ["E01", "E01"],
        "PATIENT_TYPE_ROLLUP": ["Inpatient", "Inpatient"],
        "SERVICE_LINE": ["Cardiology", "Cardiology"],
        "DATE": ["2025-01-01", "2025-01-02"],
        "ACTUAL": [100.0, 110.0],
        "MESH": [2.5, 2.5],
    })

    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.return_value = fake_df
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch("stars_pipeline.db._get_connection", return_value=mock_conn):
        result = fetch_actuals(_make_run_cfg())

    expected_cols = {"strata_id", "entity_id", "patient_type_rollup",
                     "service_line", "date", "actual", "mesh"}
    assert expected_cols.issubset(set(result.columns))
    assert pd.api.types.is_datetime64_any_dtype(result["date"])
