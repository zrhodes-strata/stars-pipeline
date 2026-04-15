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
    metric_name         str     Indicator name (e.g. ks_distribution, level_shift, is_normal)
    metric_value        str     Raw statistic as string; family name for stars_family_violated; NULL for is_normal
    metric_flag         int     1 = flagged/abnormal, 0 = pass/normal; NULL for stars_family_violated

Two summary rows are appended for each segment:
    is_normal             metric_value=NULL, metric_flag=1/0
    stars_family_violated metric_value=<family name or NULL>, metric_flag=NULL
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from stars_pipeline.logging_config import get_logger

logger = get_logger(__name__)

# Maps metric name → STARS family for the output rows
_METRIC_FAMILY: dict[str, str] = {
    "ks_distribution":     "Stability",
    "level_shift":         "Stability",
    "dw_shift":            "Stability",
    "slope_change_ratio":  "Stability",
    "stationarity":        "Stability",
    "trend_significance":  "Stability",
    "coverage_shift":      "Truthfulness",
    "sparsity_change":     "Truthfulness",
    "low_volume":          "Abundance",
    "volatility_shift":    "Regularity",
    "outlier_rate":        "Regularity",
    "acf_divergence":      "Regularity",
    "dow_pattern_shift":   "Regularity",
}

_ID_COLS = ("strata_id", "entity_id", "patient_type_rollup", "service_line", "feature_segment")


def to_long_format(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt a wide stats DataFrame (one row per segment) to long format
    (one row per segment per indicator).

    Args:
        stats_df: DataFrame produced by monitor.apply_thresholds(). Must contain
                  columns of the form ``{metric_name}_value`` and ``{metric_name}_flag``
                  for each indicator in _METRIC_FAMILY, plus ``is_normal`` and
                  ``stars_family_violated``.

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

        # Summary: is_normal
        is_normal = stat_row.get("is_normal")
        rows.append({
            **segment,
            "stars_family": "Summary",
            "metric_name":  "is_normal",
            "metric_value": None,
            "metric_flag":  int(bool(is_normal)) if pd.notna(is_normal) else None,
        })

        # Summary: stars_family_violated
        family_violated = stat_row.get("stars_family_violated")
        rows.append({
            **segment,
            "stars_family": "Summary",
            "metric_name":  "stars_family_violated",
            "metric_value": family_violated if pd.notna(family_violated) else None,
            "metric_flag":  None,
        })

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
