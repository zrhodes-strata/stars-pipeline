"""
output.py
=========
Long-format CSV writer for STARS pipeline results.

Output schema — one row per segment per STARS indicator:

    strata_id           str     Strata identifier
    entity_id           str     Entity identifier
    patient_type_rollup str     Patient type rollup
    service_line        str     Service line name
    feature_segment     str     Concatenated key: strata_id|entity_id|patient_type|service_line
    stars_family        str     STARS family (Stability / Truthfulness / Abundance / Regularity / Summary)
    metric_name         str     Indicator name or summary metric name
    metric_value        str     Raw statistic as string; NULL for binary-only rows
    metric_flag         int     1 = flagged/abnormal, 0 = pass/normal

Five summary rows are appended for each segment (stars_family="Summary"):
    is_flagged                metric_value=total violations (int), metric_flag=1/0
    stability_violations      metric_value=count (0-6),            metric_flag=1/0
    truthfulness_violations   metric_value=count (0-2),            metric_flag=1/0
    abundance_violations      metric_value=count (0-1),            metric_flag=1/0
    regularity_violations     metric_value=count (0-4),            metric_flag=1/0
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from stars_pipeline.logging_config import get_logger

logger = get_logger(__name__)

# Maps metric name → STARS family for the output rows
_METRIC_FAMILY: dict[str, str] = {
    "ks_distribution": "Stability",
    "level_shift":     "Stability",
    "dw_shift":        "Stability",
    "trend_change":    "Stability",
    "stationarity":    "Stability",
    "coverage_shift":  "Truthfulness",
    "sparsity_change": "Truthfulness",
    "low_volume":      "Abundance",
    "volatility_shift": "Regularity",
    "outlier_rate":    "Regularity",
    "acf_structure":   "Regularity",
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
    Melt a wide stats DataFrame (one row per segment) to long format
    (one row per segment per indicator).

    Args:
        stats_df: DataFrame produced by monitor.apply_thresholds(). Must contain
                  columns of the form ``{metric_name}_value`` and ``{metric_name}_flag``
                  for each indicator in _METRIC_FAMILY, plus ``is_flagged``,
                  ``stability_violations``, ``truthfulness_violations``,
                  ``abundance_violations``, and ``regularity_violations``.

    Returns:
        Long-format DataFrame with columns:
            strata_id, entity_id, patient_type_rollup, service_line,
            feature_segment, stars_family, metric_name, metric_value, metric_flag
    """
    rows: list[dict] = []

    for _, stat_row in stats_df.iterrows():
        segment = {col: stat_row[col] for col in _ID_COLS if col in stat_row.index}

        # One row per STARS indicator
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

        # Summary: is_flagged — value is total violation count, flag is 1/0
        family_counts = [
            int(stat_row.get(m, 0) or 0) for m in _SUMMARY_METRICS
        ]
        total_violations = sum(family_counts)
        is_flagged = stat_row.get("is_flagged")
        rows.append({
            **segment,
            "stars_family": "Summary",
            "metric_name":  "is_flagged",
            "metric_value": str(total_violations),
            "metric_flag":  int(bool(is_flagged)) if pd.notna(is_flagged) else None,
        })

        # Summary: one row per family violation count
        for metric_name, count in zip(_SUMMARY_METRICS, family_counts):
            rows.append({
                **segment,
                "stars_family": "Summary",
                "metric_name":  metric_name,
                "metric_value": str(count),
                "metric_flag":  1 if count > 0 else 0,
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
