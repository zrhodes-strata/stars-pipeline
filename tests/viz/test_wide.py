import pandas as pd
import pytest
from stars_pipeline.viz._wide import long_to_wide
from tests.viz.conftest import _make_long_df


def test_long_to_wide_returns_one_row_per_segment():
    df = long_to_wide(_make_long_df())
    assert len(df) == 1


def test_long_to_wide_has_primary_value_and_flag_columns():
    df = long_to_wide(_make_long_df())
    for metric in ["ks_distribution", "level_shift", "dw_shift", "trend_change",
                   "stationarity", "coverage_shift", "sparsity_change",
                   "low_volume", "volatility_shift", "outlier_rate", "acf_structure"]:
        assert f"{metric}_value" in df.columns, f"missing {metric}_value"
        assert f"{metric}_flag" in df.columns, f"missing {metric}_flag"


def test_long_to_wide_has_summary_columns():
    df = long_to_wide(_make_long_df())
    for col in ["is_flagged", "stability_violations", "truthfulness_violations",
                "abundance_violations", "regularity_violations"]:
        assert col in df.columns, f"missing {col}"


def test_long_to_wide_preserves_flag_values():
    df = long_to_wide(_make_long_df())
    assert int(df["trend_change_flag"].iloc[0]) == 1
    assert int(df["ks_distribution_flag"].iloc[0]) == 0
    assert int(df["is_flagged"].iloc[0]) == 1
    assert int(df["stability_violations"].iloc[0]) == 1


def test_long_to_wide_drops_intermediate_rows():
    long_df = _make_long_df()
    extra = long_df.iloc[0].to_dict()
    extra["stars_family"] = "Intermediate"
    extra["metric_name"] = "ks_distribution__ks_p_value"
    extra["metric_value"] = "0.999"
    extra["metric_flag"] = 0
    augmented = pd.concat([long_df, pd.DataFrame([extra])], ignore_index=True)
    df = long_to_wide(augmented)
    assert "ks_distribution__ks_p_value_value" not in df.columns
    assert "ks_distribution__ks_p_value_flag" not in df.columns


def test_long_to_wide_multi_segment():
    seg1 = _make_long_df()
    seg2 = _make_long_df()
    seg2["feature_segment"] = "84|E02|Outpatient|Neurology"
    seg2["entity_id"] = "E02"
    combined = pd.concat([seg1, seg2], ignore_index=True)
    df = long_to_wide(combined)
    assert len(df) == 2


def test_long_to_wide_has_mesh_and_band_columns():
    df = long_to_wide(_make_long_df())
    for col in ["mesh", "within_3", "within_5", "within_10"]:
        assert col in df.columns, f"missing column: {col}"


def test_long_to_wide_mesh_is_numeric():
    df = long_to_wide(_make_long_df())
    assert float(df["mesh"].iloc[0]) == pytest.approx(2.5)


def test_long_to_wide_within_bands_are_flags():
    df = long_to_wide(_make_long_df())
    assert int(df["within_3"].iloc[0]) == 1
    assert int(df["within_5"].iloc[0]) == 1
    assert int(df["within_10"].iloc[0]) == 1
