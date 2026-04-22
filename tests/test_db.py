# tests/test_db.py
import pandas as pd
import pytest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

from stars_pipeline.config import RunConfig
from stars_pipeline.db import fetch_actuals, _get_connection, _resolve_collection_id


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
        df, warnings = fetch_actuals(_make_run_cfg(run_mode=None, collection_id="COL123"))

    expected_cols = {"strata_id", "entity_id", "patient_type_rollup",
                     "service_line", "date", "actual", "mesh"}
    assert expected_cols.issubset(set(df.columns))
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert warnings == []


def _make_dagster_rows(pairs: list[tuple[int, int, str]]) -> pd.DataFrame:
    """Build a fake dagster_run_details result. pairs = [(strata_id, entity_id, pipeline_run_id)]"""
    return pd.DataFrame(
        pairs, columns=["strata_id", "entity_id", "pipeline_run_id"]
    )


def test_resolve_today_all_pairs_found(monkeypatch):
    run_cfg = _make_run_cfg(run_mode="today", strata_ids=[84])

    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.return_value = _make_dagster_rows([
        (84, 1001, "run-abc"),
        (84, 1002, "run-abc"),
    ])
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    resolved, warnings = _resolve_collection_id(run_cfg, mock_conn)

    assert resolved == {(84, 1001): "run-abc", (84, 1002): "run-abc"}
    assert warnings == []


def test_resolve_today_some_pairs_missing_logs_warning(monkeypatch):
    run_cfg = _make_run_cfg(run_mode="today", strata_ids=[84])

    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.return_value = _make_dagster_rows([
        (84, 1001, "run-abc"),
    ])
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    resolved, warnings = _resolve_collection_id(run_cfg, mock_conn)

    assert (84, 1001) in resolved
    assert (84, 1002) not in resolved
    assert warnings == []


def test_resolve_today_zero_results_falls_back_to_most_recent():
    run_cfg = _make_run_cfg(run_mode="today", strata_ids=[84])

    today_df = _make_dagster_rows([])
    most_recent_df = _make_dagster_rows([(84, 1001, "run-old")])

    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.side_effect = [today_df, most_recent_df]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    resolved, warnings = _resolve_collection_id(run_cfg, mock_conn)

    assert resolved == {(84, 1001): "run-old"}
    assert len(warnings) == 1
    assert warnings[0]["warning_type"] == "today_fallback"
    assert warnings[0]["entity_id"] is None


def test_resolve_most_recent_zero_results_raises():
    run_cfg = _make_run_cfg(run_mode="most-recent", strata_ids=[84])

    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.return_value = _make_dagster_rows([])
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with pytest.raises(ValueError, match="most-recent"):
        _resolve_collection_id(run_cfg, mock_conn)


def test_resolve_any_mode_duplicate_pipeline_run_id_raises():
    run_cfg = _make_run_cfg(run_mode="today", strata_ids=[84])

    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.return_value = _make_dagster_rows([
        (84, 1001, "run-aaa"),
        (84, 1001, "run-bbb"),
    ])
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with pytest.raises(ValueError, match="2\\+ results"):
        _resolve_collection_id(run_cfg, mock_conn)


def test_resolve_date_range_missing_pair_returns_warning():
    run_cfg = _make_run_cfg(
        run_mode="date-range",
        run_mode_date_from=date(2025, 1, 1),
        run_mode_date_to=date(2025, 1, 31),
        strata_ids=[84],
    )

    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.return_value = _make_dagster_rows([
        (84, 1001, "run-jan"),
    ])
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    resolved, warnings = _resolve_collection_id(run_cfg, mock_conn)
    assert (84, 1001) in resolved


def test_resolve_date_range_zero_results_raises():
    run_cfg = _make_run_cfg(
        run_mode="date-range",
        run_mode_date_from=date(2025, 1, 1),
        run_mode_date_to=date(2025, 1, 31),
        strata_ids=[84],
    )

    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.return_value = _make_dagster_rows([])
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with pytest.raises(ValueError, match="date-range"):
        _resolve_collection_id(run_cfg, mock_conn)
