# tests/stars/test_output.py
import pandas as pd
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
        "trend_change_value": 0.15,     "trend_change_flag": False,
        "stationarity_value": 0.15,     "stationarity_flag": False,
        "coverage_shift_value": 0.02,   "coverage_shift_flag": False,
        "sparsity_change_value": 0.01,  "sparsity_change_flag": False,
        "low_volume_value": 50.0,       "low_volume_flag": False,
        "volatility_shift_value": 1.1,  "volatility_shift_flag": False,
        "outlier_rate_value": 0.02,     "outlier_rate_flag": False,
        "acf_structure_value": 0.3,     "acf_structure_flag": False,
        # New summary columns from apply_thresholds()
        "is_flagged": False,
        "stability_violations": 0,
        "truthfulness_violations": 0,
        "abundance_violations": 0,
        "regularity_violations": 0,
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


def test_long_format_has_60_rows_per_segment():
    stats = _make_stats_row()
    result = to_long_format(stats)
    # 11 primary + 40 intermediates + 5 original summary + 4 mesh/band summary rows
    assert len(result) == 60


def test_long_format_metric_names_include_new_summaries():
    stats = _make_stats_row()
    result = to_long_format(stats)
    names = set(result["metric_name"])
    for name in [
        "ks_distribution", "level_shift", "low_volume",
        "is_flagged",
        "stability_violations",
        "truthfulness_violations",
        "abundance_violations",
        "regularity_violations",
    ]:
        assert name in names, f"Missing metric_name: {name}"
    assert "is_normal" not in names
    assert "stars_family_violated" not in names


def test_is_flagged_row_no_violations():
    stats = _make_stats_row(is_flagged=False, stability_violations=0,
                            truthfulness_violations=0, abundance_violations=0,
                            regularity_violations=0)
    result = to_long_format(stats)
    row = result[result["metric_name"] == "is_flagged"].iloc[0]
    assert int(row["metric_flag"]) == 0
    assert row["metric_value"] == "0"


def test_is_flagged_row_with_violations():
    stats = _make_stats_row(
        is_flagged=True,
        stability_violations=2,
        truthfulness_violations=1,
        abundance_violations=0,
        regularity_violations=0,
        ks_distribution_flag=True,
        level_shift_flag=True,
        coverage_shift_flag=True,
    )
    result = to_long_format(stats)
    row = result[result["metric_name"] == "is_flagged"].iloc[0]
    assert int(row["metric_flag"]) == 1
    assert row["metric_value"] == "3"  # 2 + 1 + 0 + 0


def test_stability_violations_row():
    stats = _make_stats_row(
        stability_violations=3, is_flagged=True,
        ks_distribution_flag=True, level_shift_flag=True, dw_shift_flag=True,
    )
    result = to_long_format(stats)
    row = result[result["metric_name"] == "stability_violations"].iloc[0]
    assert int(row["metric_flag"]) == 1
    assert row["metric_value"] == "3"


def test_zero_violations_family_row_flag_is_0():
    stats = _make_stats_row(regularity_violations=0)
    result = to_long_format(stats)
    row = result[result["metric_name"] == "regularity_violations"].iloc[0]
    assert int(row["metric_flag"]) == 0
    assert row["metric_value"] == "0"


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
    assert len(df) == 60


def test_mesh_summary_row():
    stats = _make_stats_row(mesh=2.5)
    result = to_long_format(stats)
    row = result[result["metric_name"] == "mesh"].iloc[0]
    assert row["stars_family"] == "Summary"
    assert row["metric_value"] == "2.5"
    assert int(row["metric_flag"]) == 0  # 2.5 <= 10, so NOT outside band → flag=0


def test_mesh_flag_outside_band():
    stats = _make_stats_row(mesh=15.0)
    result = to_long_format(stats)
    row = result[result["metric_name"] == "mesh"].iloc[0]
    assert int(row["metric_flag"]) == 1  # 15.0 > 10 → outside broadest band → flag=1


def test_within_3_row():
    stats = _make_stats_row(mesh=2.5)
    result = to_long_format(stats)
    row = result[result["metric_name"] == "within_3"].iloc[0]
    assert row["stars_family"] == "Summary"
    assert row["metric_value"] == "2.5"
    assert int(row["metric_flag"]) == 1  # 2.5 <= 3


def test_within_5_row():
    stats = _make_stats_row(mesh=4.0)
    result = to_long_format(stats)
    row = result[result["metric_name"] == "within_5"].iloc[0]
    assert row["stars_family"] == "Summary"
    assert row["metric_value"] == "4.0"
    assert int(row["metric_flag"]) == 1  # 4.0 <= 5


def test_within_10_row():
    stats = _make_stats_row(mesh=8.0)
    result = to_long_format(stats)
    row = result[result["metric_name"] == "within_10"].iloc[0]
    assert row["stars_family"] == "Summary"
    assert row["metric_value"] == "8.0"
    assert int(row["metric_flag"]) == 1  # 8.0 <= 10


def test_within_bands_outside():
    stats = _make_stats_row(mesh=7.0)
    result = to_long_format(stats)
    assert int(result[result["metric_name"] == "within_3"]["metric_flag"].iloc[0]) == 0
    assert int(result[result["metric_name"] == "within_5"]["metric_flag"].iloc[0]) == 0
    assert int(result[result["metric_name"] == "within_10"]["metric_flag"].iloc[0]) == 1


def test_mesh_boundary_at_10():
    """At exactly 10.0: mesh flag=0 (not outside band), within_10 flag=1 (inside band)."""
    stats = _make_stats_row(mesh=10.0)
    result = to_long_format(stats)
    assert int(result[result["metric_name"] == "mesh"]["metric_flag"].iloc[0]) == 0
    assert int(result[result["metric_name"] == "within_10"]["metric_flag"].iloc[0]) == 1
    assert int(result[result["metric_name"] == "within_5"]["metric_flag"].iloc[0]) == 0
    assert int(result[result["metric_name"] == "within_3"]["metric_flag"].iloc[0]) == 0


def test_mesh_none_produces_null_rows():
    stats = _make_stats_row(mesh=None)
    result = to_long_format(stats)
    for name in ["mesh", "within_3", "within_5", "within_10"]:
        row = result[result["metric_name"] == name].iloc[0]
        assert row["metric_value"] is None or pd.isna(row["metric_value"]), \
            f"{name} metric_value should be None when mesh is None"
        assert row["metric_flag"] is None or pd.isna(row["metric_flag"]), \
            f"{name} metric_flag should be None when mesh is None"
