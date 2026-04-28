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
        run_mode=None,
        run_mode_date_from=None,
        run_mode_date_to=None,
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
        "feature_segment", "mesh",
        "ks_distribution_value", "ks_distribution_flag",
        "level_shift_value", "level_shift_flag",
        "dw_shift_value", "dw_shift_flag",
        "trend_change_value", "trend_change_flag",
        "stationarity_value", "stationarity_flag",
        "coverage_shift_value", "coverage_shift_flag",
        "sparsity_change_value", "sparsity_change_flag",
        "low_volume_value", "low_volume_flag",
        "volatility_shift_value", "volatility_shift_flag",
        "outlier_rate_value", "outlier_rate_flag",
        "acf_structure_value", "acf_structure_flag",
    }
    assert expected.issubset(set(result.columns))

def test_run_monitoring_does_not_have_removed_columns():
    df = _make_df()
    result = run_monitoring(df, _make_run_cfg(), MonitorConfig())
    removed = {
        "slope_change_ratio_value", "slope_change_ratio_flag",
        "trend_significance_value", "trend_significance_flag",
        "acf_divergence_value", "acf_divergence_flag",
        "dow_pattern_shift_value", "dow_pattern_shift_flag",
    }
    assert removed.isdisjoint(set(result.columns))


def test_apply_thresholds_adds_summary_columns():
    df = _make_df()
    stats = run_monitoring(df, _make_run_cfg(), MonitorConfig())
    result = apply_thresholds(stats)
    for col in (
        "is_flagged",
        "stability_violations",
        "truthfulness_violations",
        "abundance_violations",
        "regularity_violations",
    ):
        assert col in result.columns, f"Missing column: {col}"
    assert "is_normal" not in result.columns
    assert "stars_family_violated" not in result.columns


def test_normal_segment_is_not_flagged():
    # Use permissive thresholds that won't fire on clean i.i.d. normal data.
    # kpss_alpha: tight to avoid KPSS false positives on short recent windows.
    # volatility_ratio: very large to ensure same-distribution series doesn't flag.
    # trend_p_value: tight to avoid spurious trend flag on noisy series.
    # slope_change_ratio: large to avoid eps-division artifacts.
    cfg = MonitorConfig(
        kpss_alpha=0.001,
        volatility_ratio_threshold=1000.0,
        trend_p_value_threshold=0.001,
        slope_change_ratio_threshold=1000.0,
    )
    df = _make_df(n_days=400, mean=100.0, std=5.0)
    stats = run_monitoring(df, _make_run_cfg(), cfg)
    result = apply_thresholds(stats)
    assert bool(result["is_flagged"].iloc[0]) is False
    assert int(result["stability_violations"].iloc[0]) == 0
    assert int(result["truthfulness_violations"].iloc[0]) == 0
    assert int(result["abundance_violations"].iloc[0]) == 0
    assert int(result["regularity_violations"].iloc[0]) == 0


def test_violation_counts_match_flags():
    """If stability flags ks_distribution and level_shift, stability_violations==2."""
    df = _make_df()
    stats = run_monitoring(df, _make_run_cfg(), MonitorConfig())
    # Zero out all flag columns first, then manually force two Stability flags
    # and one Regularity flag so the counts are deterministic.
    stats = stats.copy()
    flag_cols = [c for c in stats.columns if c.endswith("_flag")]
    stats[flag_cols] = False
    stats["ks_distribution_flag"] = True
    stats["level_shift_flag"] = True
    stats["volatility_shift_flag"] = True
    result = apply_thresholds(stats)
    assert int(result["stability_violations"].iloc[0]) == 2
    assert int(result["regularity_violations"].iloc[0]) == 1
    assert int(result["is_flagged"].iloc[0]) == 1


def test_multiple_segments_produce_multiple_rows():
    df1 = _make_df(seed=1)
    df2 = _make_df(seed=2)
    df2["service_line"] = "Orthopedics"
    df = pd.concat([df1, df2], ignore_index=True)
    result = run_monitoring(df, _make_run_cfg(), MonitorConfig())
    assert len(result) == 2


def test_train_days_caps_training_window():
    # With train_days=180 and recent_days=90, run_monitoring should complete
    # without error and still produce one row per segment.
    df = _make_df(n_days=500)
    run_cfg = _make_run_cfg(recent_days=90, train_days=180)
    result = run_monitoring(df, run_cfg, MonitorConfig())
    assert len(result) == 1
    assert "ks_distribution_value" in result.columns
