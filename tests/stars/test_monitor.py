# tests/stars/test_monitor.py
import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta
from pathlib import Path

from stars_pipeline.config import MonitorConfig, RunConfig
from stars_pipeline.stars.monitor import apply_thresholds, run_monitoring


def _make_run_cfg(recent_days=90, train_days=None):
    return RunConfig(
        strata_ids=[1],
        collection_id=None,
        run_id=None,
        date_from=date(2022, 1, 1),
        date_to=date(2026, 1, 1),
        recent_days=recent_days,
        train_days=train_days,
        entity_id=None,
        patient_type=None,
        service_line=None,
        output_path=Path("out.csv"),
    )


def _make_df(n_days=400, mean=100.0, std=10.0, seed=42):
    """Create a synthetic single-segment DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    return pd.DataFrame({
        "strata_id": "1",
        "entity_id": "E01",
        "patient_type_rollup": "Inpatient",
        "service_line": "Cardiology",
        "date": dates,
        "actual": rng.normal(mean, std, n_days).clip(0),
        "mesh": 2.5,
    })


def test_run_monitoring_returns_one_row_per_segment():
    df = _make_df()
    result = run_monitoring(df, _make_run_cfg(), MonitorConfig())
    assert len(result) == 1


def test_run_monitoring_output_has_expected_columns():
    df = _make_df()
    result = run_monitoring(df, _make_run_cfg(), MonitorConfig())
    expected = {
        "strata_id", "entity_id", "patient_type_rollup", "service_line",
        "feature_segment",
        "ks_distribution_value", "ks_distribution_flag",
        "level_shift_value", "level_shift_flag",
        "dw_shift_value", "dw_shift_flag",
        "slope_change_ratio_value", "slope_change_ratio_flag",
        "stationarity_value", "stationarity_flag",
        "trend_significance_value", "trend_significance_flag",
        "coverage_shift_value", "coverage_shift_flag",
        "sparsity_change_value", "sparsity_change_flag",
        "low_volume_value", "low_volume_flag",
        "volatility_shift_value", "volatility_shift_flag",
        "outlier_rate_value", "outlier_rate_flag",
        "acf_divergence_value", "acf_divergence_flag",
        "dow_pattern_shift_value", "dow_pattern_shift_flag",
    }
    assert expected.issubset(set(result.columns))


def test_apply_thresholds_adds_summary_columns():
    df = _make_df()
    stats = run_monitoring(df, _make_run_cfg(), MonitorConfig())
    result = apply_thresholds(stats)
    assert "is_normal" in result.columns
    assert "stars_family_violated" in result.columns


def test_normal_segment_is_classified_normal():
    # Use relaxed thresholds to avoid near-zero slope ratio instability and KPSS
    # p-value boundary effects that can spuriously flag iid normal data.
    # slope_change_ratio_threshold=50 sits above the typical noise ratio (~36x)
    # for near-flat series; kpss_alpha=0.04 avoids the 0.10 table boundary.
    cfg = MonitorConfig(slope_change_ratio_threshold=50.0, kpss_alpha=0.04)
    df = _make_df(n_days=400, mean=100.0, std=5.0)  # stable, no shift
    stats = run_monitoring(df, _make_run_cfg(), cfg)
    result = apply_thresholds(stats)
    assert bool(result["is_normal"].iloc[0]) is True
    assert result["stars_family_violated"].iloc[0] is None


def test_multiple_segments_produce_multiple_rows():
    df1 = _make_df(seed=1)
    df2 = _make_df(seed=2)
    df2["service_line"] = "Orthopedics"
    df = pd.concat([df1, df2], ignore_index=True)
    result = run_monitoring(df, _make_run_cfg(), MonitorConfig())
    assert len(result) == 2
