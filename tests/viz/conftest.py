import pandas as pd
import pytest


def _make_long_df() -> pd.DataFrame:
    rows = []
    seg = {
        "strata_id": "84", "entity_id": "E01",
        "patient_type_rollup": "Inpatient", "service_line": "Cardiology",
        "feature_segment": "84|E01|Inpatient|Cardiology",
    }
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
