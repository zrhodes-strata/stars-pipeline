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


def _make_pipeline_rows(rows: list[dict]) -> pd.DataFrame:
    """Build a fake step-A result (pipeline_run_id rows)."""
    return pd.DataFrame(rows)


def _make_expansion_rows(rows: list[dict]) -> pd.DataFrame:
    """Build a fake step-B result (run_id + collection_id rows)."""
    return pd.DataFrame(rows)


def test_resolve_today_returns_run_ids_and_collection_id():
    run_cfg = _make_run_cfg(run_mode="today", strata_ids=[84])

    step_a = _make_pipeline_rows([{"pipeline_run_id": "pipe-1"}])
    step_b = _make_expansion_rows([
        {"strata_id": 84, "entity_id": 1001, "pipeline_run_id": "pipe-1",
         "run_id": "run-aaa", "collection_id": "col-xyz"},
        {"strata_id": 84, "entity_id": 1001, "pipeline_run_id": "pipe-1",
         "run_id": "run-bbb", "collection_id": "col-xyz"},
    ])

    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.side_effect = [step_a, step_b]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    run_ids, collection_id, warnings = _resolve_collection_id(run_cfg, mock_conn)

    assert set(run_ids) == {"run-aaa", "run-bbb"}
    assert collection_id == "col-xyz"
    assert warnings == []


def test_resolve_today_null_collection_id_returns_none():
    run_cfg = _make_run_cfg(run_mode="today", strata_ids=[84])

    step_a = _make_pipeline_rows([{"pipeline_run_id": "pipe-1"}])
    step_b = _make_expansion_rows([
        {"strata_id": 84, "entity_id": 1001, "pipeline_run_id": "pipe-1",
         "run_id": "run-aaa", "collection_id": None},
    ])

    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.side_effect = [step_a, step_b]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    run_ids, collection_id, warnings = _resolve_collection_id(run_cfg, mock_conn)

    assert run_ids == ["run-aaa"]
    assert collection_id is None
    assert warnings == []


def test_resolve_today_zero_results_falls_back_to_most_recent():
    run_cfg = _make_run_cfg(run_mode="today", strata_ids=[84])

    empty_step_a = _make_pipeline_rows([])
    fallback_step_a = _make_pipeline_rows([{"pipeline_run_id": "pipe-old"}])
    fallback_step_b = _make_expansion_rows([
        {"strata_id": 84, "entity_id": 1001, "pipeline_run_id": "pipe-old",
         "run_id": "run-old", "collection_id": "col-old"},
    ])

    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.side_effect = [empty_step_a, fallback_step_a, fallback_step_b]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    run_ids, collection_id, warnings = _resolve_collection_id(run_cfg, mock_conn)

    assert run_ids == ["run-old"]
    assert collection_id == "col-old"
    assert len(warnings) == 1
    assert warnings[0]["warning_type"] == "today_fallback"


def test_resolve_most_recent_zero_results_raises():
    run_cfg = _make_run_cfg(run_mode="most-recent", strata_ids=[84])

    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.return_value = _make_pipeline_rows([])
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with pytest.raises(ValueError, match="most-recent"):
        _resolve_collection_id(run_cfg, mock_conn)


def test_resolve_duplicate_pipeline_run_id_per_entity_raises():
    """Same (strata_id, entity_id) maps to two different pipeline_run_ids — not allowed."""
    run_cfg = _make_run_cfg(run_mode="today", strata_ids=[84])

    step_a = _make_pipeline_rows([
        {"pipeline_run_id": "pipe-1"},
        {"pipeline_run_id": "pipe-2"},
    ])
    step_b = _make_expansion_rows([
        {"strata_id": 84, "entity_id": 1001, "pipeline_run_id": "pipe-1",
         "run_id": "run-aaa", "collection_id": "col-x"},
        {"strata_id": 84, "entity_id": 1001, "pipeline_run_id": "pipe-2",
         "run_id": "run-bbb", "collection_id": "col-y"},
    ])

    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.side_effect = [step_a, step_b]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with pytest.raises(ValueError, match="2\\+ pipeline_run_ids"):
        _resolve_collection_id(run_cfg, mock_conn)


def test_resolve_date_range_zero_results_raises():
    run_cfg = _make_run_cfg(
        run_mode="date-range",
        run_mode_date_from=date(2025, 1, 1),
        run_mode_date_to=date(2025, 1, 31),
        strata_ids=[84],
    )

    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.return_value = _make_pipeline_rows([])
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with pytest.raises(ValueError, match="date-range"):
        _resolve_collection_id(run_cfg, mock_conn)


def test_fetch_actuals_empty_run_ids_raises(monkeypatch):
    for var in ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD",
                "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA"]:
        monkeypatch.setenv(var, "dummy")

    # collection_id lookup returns no run_ids (all NULL)
    dagster_rows = pd.DataFrame({"RUN_ID": pd.Series([], dtype="object")})
    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.return_value = dagster_rows
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch("stars_pipeline.db._get_connection", return_value=mock_conn):
        with pytest.raises(ValueError, match="returned 0 run_ids from dagster"):
            fetch_actuals(_make_run_cfg(run_mode=None, collection_id="COL123"))


def test_fetch_actuals_returns_expected_columns(monkeypatch):
    for var in ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD",
                "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA"]:
        monkeypatch.setenv(var, "dummy")

    # Fake dagster lookup (collection_id → run_ids)
    dagster_rows = pd.DataFrame({"RUN_ID": ["run-aaa", "run-bbb"]})

    actuals_rows = pd.DataFrame({
        "STRATA_ID": [84, 84],
        "ENTITY_ID": [1, 1],
        "PATIENT_TYPE_ROLLUP_ID": [1, 1],
        "PATIENT_TYPE_ROLLUP_CLEAN": ["IP", "IP"],
        "SERVICE_LINE_ID": ["10", "10"],
        "SERVICE_LINE_CLEAN": ["Cardiology", "Cardiology"],
        "DATE": ["2025-01-01", "2025-01-02"],
        "ACTUAL": [100.0, 110.0],
        "ROW_COUNT": [1, 1],
    })

    cv_rows = pd.DataFrame({
        "STRATA_ID": [84],
        "ENTITY_ID": [1],
        "PATIENT_TYPE_ROLLUP_ID": [1],
        "PATIENT_TYPE_ROLLUP_CLEAN": ["IP"],
        "SERVICE_LINE_ID": ["10"],
        "SERVICE_LINE_CLEAN": ["Cardiology"],
        "MODEL_NAME": ["ModelA"],
        "PREDICTION": [95.0],
        "ACTUAL": [100.0],
    })

    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.side_effect = [dagster_rows, actuals_rows, cv_rows]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch("stars_pipeline.db._get_connection", return_value=mock_conn):
        df, warnings = fetch_actuals(_make_run_cfg(run_mode=None, collection_id="COL123"))

    expected_cols = {
        "strata_id", "entity_id", "patient_type_rollup_id",
        "patient_type_rollup", "service_line_id", "service_line",
        "date", "actual", "row_count", "mesh",
    }
    assert expected_cols.issubset(set(df.columns))
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert warnings == []
