import pandas as pd
import pytest
from stars_pipeline.viz._wide import long_to_wide


def _make_long_df():
    rows = []
    seg = {
        "strata_id": "84", "entity_id": "E01",
        "patient_type_rollup": "Inpatient", "service_line": "Cardiology",
        "feature_segment": "84|E01|Inpatient|Cardiology",
    }
    # Primary rows
    for metric, family, value, flag in [
        ("ks_distribution",  "Stability",    "0.15", 0),
        ("level_shift",      "Stability",    "0.80", 0),
        ("dw_shift",         "Stability",    "0.30", 0),
        ("trend_change",     "Stability",    "0.18", 1),
        ("stationarity",     "Stability",    "0.05", 0),
        ("coverage_shift",   "Truthfulness", "0.02", 0),
        ("sparsity_change",  "Truthfulness", "0.01", 0),
        ("low_volume",       "Abundance",    "5.00", 0),
        ("volatility_shift", "Regularity",   "1.10", 0),
        ("outlier_rate",     "Regularity",   "0.05", 0),
        ("acf_structure",    "Regularity",   "0.20", 0),
    ]:
        rows.append({**seg, "stars_family": family, "metric_name": metric,
                     "metric_value": value, "metric_flag": flag})
    # Summary rows
    for metric, value, flag in [
        ("is_flagged", "1", 1),
        ("stability_violations", "1", 1),
        ("truthfulness_violations", "0", 0),
        ("abundance_violations", "0", 0),
        ("regularity_violations", "0", 0),
    ]:
        rows.append({**seg, "stars_family": "Summary", "metric_name": metric,
                     "metric_value": value, "metric_flag": flag})
    return pd.DataFrame(rows)


def test_long_to_wide_returns_one_row_per_segment():
    df = long_to_wide(_make_long_df())
    assert len(df) == 1


def test_long_to_wide_has_primary_value_and_flag_columns():
    df = long_to_wide(_make_long_df())
    for metric in ["ks_distribution", "level_shift", "trend_change",
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
