from __future__ import annotations
import pandas as pd

_ID_COLS = ["strata_id", "entity_id", "patient_type_rollup", "service_line", "feature_segment"]

# Summary metrics read from metric_value (numeric: counts and MESH score)
_VALUE_SUMMARY_METRICS = {
    "stability_violations", "truthfulness_violations",
    "abundance_violations", "regularity_violations",
    "mesh",
}

# Summary metrics read from metric_flag (binary 0/1: overall flag + band membership)
_FLAG_SUMMARY_METRICS = {
    "is_flagged", "within_3", "within_5", "within_10",
}

_SUMMARY_METRICS = _VALUE_SUMMARY_METRICS | _FLAG_SUMMARY_METRICS


def long_to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot a long-format STARS results DataFrame back to wide stats format.

    Accepts the output of to_long_format() (or a CSV written by write_long_csv()).
    Returns one row per segment with {metric}_value, {metric}_flag, and summary
    columns — the same shape as the DataFrame produced by apply_thresholds().

    Intermediate rows (stars_family="Intermediate") are silently dropped; they
    are not needed by the plot functions.
    """
    id_cols = [c for c in _ID_COLS if c in long_df.columns]

    # Primary and summary rows only (drop intermediates)
    filtered = long_df[long_df["stars_family"] != "Intermediate"].copy()
    filtered["metric_value"] = pd.to_numeric(filtered["metric_value"], errors="coerce")
    filtered["metric_flag"] = pd.to_numeric(filtered["metric_flag"], errors="coerce")

    rows: list[dict] = []
    for seg_key, grp in filtered.groupby("feature_segment"):
        row: dict = {col: grp[col].iloc[0] for col in id_cols if col in grp.columns}
        for _, r in grp.iterrows():
            metric = r["metric_name"]
            sentinel = metric if metric in _SUMMARY_METRICS else f"{metric}_value"
            if sentinel in row:
                raise ValueError(
                    f"Duplicate metric '{metric}' for segment '{seg_key}'"
                )
            if metric in _SUMMARY_METRICS:
                if metric in _FLAG_SUMMARY_METRICS:
                    row[metric] = r["metric_flag"]
                else:
                    row[metric] = r["metric_value"]
            else:
                row[f"{metric}_value"] = r["metric_value"]
                row[f"{metric}_flag"] = r["metric_flag"]
        rows.append(row)

    return pd.DataFrame(rows)
