# tests/stars/test_output.py
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from stars_pipeline.stars.output import to_long_format, write_long_csv


def _make_stats_row(**overrides):
    """Minimal stats DataFrame row — one Normal segment."""
    row = {
        "strata_id": "84",
        "entity_id": "E01",
        "patient_type_rollup": "Inpatient",
        "service_line": "Cardiology",
        "feature_segment": "84|E01|Inpatient|Cardiology",
        "mesh": 2.5,
        "ks_distribution_value": 0.10,  "ks_distribution_flag": False,
        "level_shift_value": 0.50,      "level_shift_flag": False,
        "dw_shift_value": 0.40,         "dw_shift_flag": False,
        "slope_change_ratio_value": 0.8,"slope_change_ratio_flag": False,
        "stationarity_value": 0.15,     "stationarity_flag": False,
        "trend_significance_value": 0.3,"trend_significance_flag": False,
        "coverage_shift_value": 0.02,   "coverage_shift_flag": False,
        "sparsity_change_value": 0.01,  "sparsity_change_flag": False,
        "low_volume_value": 50.0,       "low_volume_flag": False,
        "volatility_shift_value": 1.1,  "volatility_shift_flag": False,
        "outlier_rate_value": 0.02,     "outlier_rate_flag": False,
        "acf_divergence_value": 0.3,    "acf_divergence_flag": False,
        "dow_pattern_shift_value": 0.2, "dow_pattern_shift_flag": False,
        "is_normal": True,
        "stars_family_violated": None,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_long_format_has_correct_columns():
    stats = _make_stats_row()
    result = to_long_format(stats)
    assert set(result.columns) == {
        "strata_id", "entity_id", "patient_type_rollup", "service_line",
        "feature_segment", "stars_family", "metric_name",
        "metric_value", "metric_flag",
    }


def test_long_format_has_one_row_per_indicator_plus_summary():
    stats = _make_stats_row()
    result = to_long_format(stats)
    # 13 indicators + 2 summary rows (is_normal, stars_family_violated)
    assert len(result) == 15


def test_long_format_metric_names_match_indicators():
    stats = _make_stats_row()
    result = to_long_format(stats)
    names = set(result["metric_name"])
    for name in ["ks_distribution", "level_shift", "low_volume",
                 "is_normal", "stars_family_violated"]:
        assert name in names


def test_is_normal_row_flag_is_1_for_normal_segment():
    stats = _make_stats_row(is_normal=True)
    result = to_long_format(stats)
    row = result[result["metric_name"] == "is_normal"].iloc[0]
    assert int(row["metric_flag"]) == 1
    assert row["metric_value"] is None or pd.isna(row["metric_value"])


def test_stars_family_violated_carries_family_name():
    stats = _make_stats_row(is_normal=False, stars_family_violated="Stability")
    result = to_long_format(stats)
    row = result[result["metric_name"] == "stars_family_violated"].iloc[0]
    assert row["metric_value"] == "Stability"
    assert row["metric_flag"] is None or pd.isna(row["metric_flag"])


def test_flagged_indicator_produces_metric_flag_1():
    stats = _make_stats_row(ks_distribution_flag=True, ks_distribution_value=0.42)
    result = to_long_format(stats)
    row = result[result["metric_name"] == "ks_distribution"].iloc[0]
    assert int(row["metric_flag"]) == 1
    assert row["metric_value"] == "0.42"


def test_write_long_csv_creates_file(tmp_path):
    stats = _make_stats_row()
    out = tmp_path / "results.csv"
    write_long_csv(stats, out)
    assert out.exists()
    df = pd.read_csv(out)
    assert len(df) == 15
