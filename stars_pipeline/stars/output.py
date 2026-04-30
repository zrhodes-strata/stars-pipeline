"""
output.py
=========
Long-format CSV writer for STARS pipeline results.

Output schema — one row per segment per metric:

    strata_id           str     Strata identifier
    entity_id           str     Entity identifier
    patient_type_rollup str     Patient type rollup
    service_line        str     Service line name
    feature_segment     str     Concatenated key: strata_id|entity_id|patient_type|service_line
    stars_family        str     STARS family (Stability / Truthfulness / Abundance / Regularity / Summary / Intermediate)
    metric_name         str     Indicator name, intermediate stat name, or summary metric name
    metric_value        str     Raw statistic as string; NULL for rows with no numeric value
    metric_flag         int     1 = flagged/abnormal, 0 = pass/normal; NULL for intermediates

Primary indicator rows (one per segment per indicator, metric_flag set):
    ks_distribution, level_shift, dw_shift, trend_change, stationarity,
    coverage_shift, sparsity_change, low_volume, volatility_shift,
    outlier_rate, acf_structure

Intermediate rows (stars_family="Intermediate", metric_flag=NULL):
    Per-indicator intermediate statistics named {metric}__{key}, e.g.:
    ks_distribution__ks_p_value, stationarity__kpss_p_train, acf_structure__acf_train_lag1

Five summary rows per segment (stars_family="Summary"):
    is_flagged                metric_value=total violations (int), metric_flag=1/0
    stability_violations      metric_value=count (0-5),            metric_flag=1/0
    truthfulness_violations   metric_value=count (0-2),            metric_flag=1/0
    abundance_violations      metric_value=count (0-1),            metric_flag=1/0
    regularity_violations     metric_value=count (0-3),            metric_flag=1/0
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from stars_pipeline.logging_config import get_logger

logger = get_logger(__name__)

# Maps metric name → STARS family for primary indicator rows
_METRIC_FAMILY: dict[str, str] = {
    "ks_distribution":  "Stability",
    "level_shift":      "Stability",
    "dw_shift":         "Stability",
    "trend_change":     "Stability",
    "stationarity":     "Stability",
    "coverage_shift":   "Truthfulness",
    "sparsity_change":  "Truthfulness",
    "low_volume":       "Abundance",
    "volatility_shift": "Regularity",
    "outlier_rate":     "Regularity",
    "acf_structure":    "Regularity",
}

# Intermediate columns emitted per metric: {metric_name: [extra_key, ...]}
# These map to stats_df columns named {metric_name}_{extra_key}.
_METRIC_INTERMEDIATES: dict[str, list[str]] = {
    "ks_distribution":  ["ks_p_value", "mean_train", "mean_recent"],
    "level_shift":      ["p_value", "mean_train", "mean_recent"],
    "dw_shift":         ["dw_train", "dw_recent"],
    "trend_change":     ["slope_train", "slope_recent", "slope_delta", "slope_change_ratio"],
    "stationarity":     ["kpss_p_train", "kpss_p_recent", "train_stationary"],
    "coverage_shift":   ["p_value", "coverage_train", "coverage_recent"],
    "sparsity_change":  ["p_value", "sparsity_train", "sparsity_recent"],
    "low_volume":       ["total_volume_train", "n_months_train"],
    "volatility_shift": ["cv_train", "cv_recent", "mean_train", "mean_recent"],
    "outlier_rate":     ["outlier_count", "train_median", "train_mad"],
    "acf_structure":    [
        "bartlett_bound",
        "acf_train_lag1",  "acf_recent_lag1",  "acf_p_lag1",
        "acf_train_lag7",  "acf_recent_lag7",  "acf_p_lag7",
        "acf_train_lag30", "acf_recent_lag30", "acf_p_lag30",
    ],
}

_ID_COLS = ("strata_id", "entity_id", "patient_type_rollup", "service_line", "feature_segment")

_SUMMARY_METRICS = (
    "stability_violations",
    "truthfulness_violations",
    "abundance_violations",
    "regularity_violations",
)


def to_long_format(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt a wide stats DataFrame (one row per segment) to long format.

    Emits three categories of rows per segment:
      1. One primary row per STARS indicator (metric_flag set).
      2. One intermediate row per extra statistic per indicator
         (stars_family="Intermediate", metric_flag=NULL).
      3. Five summary rows (stars_family="Summary").

    Args:
        stats_df: DataFrame produced by monitor.apply_thresholds().

    Returns:
        Long-format DataFrame with columns:
            strata_id, entity_id, patient_type_rollup, service_line,
            feature_segment, stars_family, metric_name, metric_value, metric_flag
    """
    rows: list[dict] = []

    for _, stat_row in stats_df.iterrows():
        segment = {col: stat_row[col] for col in _ID_COLS if col in stat_row.index}

        # ── Primary indicator rows ────────────────────────────────────────────
        for metric, family in _METRIC_FAMILY.items():
            val_col  = f"{metric}_value"
            flag_col = f"{metric}_flag"
            raw_val  = stat_row.get(val_col)
            raw_flag = stat_row.get(flag_col)

            rows.append({
                **segment,
                "stars_family": family,
                "metric_name":  metric,
                "metric_value": str(raw_val) if pd.notna(raw_val) else None,
                "metric_flag":  int(raw_flag) if pd.notna(raw_flag) else None,
            })

            # ── Intermediate rows for this metric ─────────────────────────────
            for extra_key in _METRIC_INTERMEDIATES.get(metric, []):
                col = f"{metric}_{extra_key}"
                raw = stat_row.get(col)
                rows.append({
                    **segment,
                    "stars_family": "Intermediate",
                    "metric_name":  f"{metric}__{extra_key}",
                    "metric_value": str(raw) if pd.notna(raw) else None,
                    "metric_flag":  None,
                })

        # ── Summary rows ─────────────────────────────────────────────────────
        family_counts = [int(stat_row.get(m, 0) or 0) for m in _SUMMARY_METRICS]
        total_violations = sum(family_counts)
        is_flagged = stat_row.get("is_flagged")
        rows.append({
            **segment,
            "stars_family": "Summary",
            "metric_name":  "is_flagged",
            "metric_value": str(total_violations),
            "metric_flag":  int(bool(is_flagged)) if pd.notna(is_flagged) else None,
        })
        for metric_name, count in zip(_SUMMARY_METRICS, family_counts):
            rows.append({
                **segment,
                "stars_family": "Summary",
                "metric_name":  metric_name,
                "metric_value": str(count),
                "metric_flag":  1 if count > 0 else 0,
            })

        # ── MESH + accuracy band rows ─────────────────────────────────────────
        mesh_val = stat_row.get("mesh")
        if pd.notna(mesh_val):
            mesh_float = float(mesh_val)
            rows.append({
                **segment,
                "stars_family": "Summary",
                "metric_name":  "mesh",
                "metric_value": str(mesh_float),
                "metric_flag":  1 if mesh_float > 10.0 else 0,
            })
            for band, threshold in [("within_3", 3.0), ("within_5", 5.0), ("within_10", 10.0)]:
                rows.append({
                    **segment,
                    "stars_family": "Summary",
                    "metric_name":  band,
                    "metric_value": str(mesh_float),
                    "metric_flag":  1 if mesh_float <= threshold else 0,
                })
        else:
            for name in ["mesh", "within_3", "within_5", "within_10"]:
                rows.append({
                    **segment,
                    "stars_family": "Summary",
                    "metric_name":  name,
                    "metric_value": None,
                    "metric_flag":  None,
                })

    _OUTPUT_COLS = [
        "strata_id", "entity_id", "patient_type_rollup", "service_line",
        "feature_segment", "stars_family", "metric_name", "metric_value", "metric_flag",
    ]
    if not rows:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.DataFrame(rows)


def write_long_csv(stats_df: pd.DataFrame, output_path: Path) -> None:
    """
    Convert stats_df to long format and write to a CSV file.

    Creates parent directories if they do not exist.

    Args:
        stats_df:    DataFrame produced by monitor.apply_thresholds().
        output_path: Destination path for the CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    long_df = to_long_format(stats_df)
    long_df.to_csv(output_path, index=False)
    logger.info(
        "Long-format CSV written",
        extra={"rows": len(long_df), "path": str(output_path)},
    )
