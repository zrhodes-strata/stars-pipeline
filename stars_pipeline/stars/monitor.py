"""
monitor.py
==========
STARS monitoring orchestrator.

Two public functions:

run_monitoring(df, run_cfg, monitor_cfg)
    Groups the input DataFrame by segment, splits each series into training
    and recent windows, calls all 13 test functions from tests.py, and
    returns one wide row per segment with all raw statistics.

apply_thresholds(stats_df)
    Takes the wide stats DataFrame produced by run_monitoring() and derives
    two summary columns:
      is_normal              True if no flag in any family
      stars_family_violated  Name of the first violated family, or None

Compute / classify decoupling
------------------------------
run_monitoring() is the expensive step (per-series loops, statistical tests).
apply_thresholds() is vectorised pandas — O(n_segments). This separation
allows downstream re-classification without re-running the expensive tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stars_pipeline.config import MonitorConfig, RunConfig
from stars_pipeline.logging_config import get_logger
from stars_pipeline.stars.tests import (
    test_acf_divergence,
    test_coverage_shift,
    test_dow_pattern_shift,
    test_dw_shift,
    test_ks_distribution,
    test_level_shift,
    test_low_volume,
    test_outlier_rate,
    test_slope_change,
    test_sparsity_change,
    test_stationarity,
    test_trend_significance,
    test_volatility_shift,
)

logger = get_logger(__name__)

_GROUP_COLS = ("strata_id", "entity_id", "patient_type_rollup", "service_line")


def _prepare_series(
    group: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "actual",
) -> tuple[pd.DatetimeIndex, pd.Series, pd.Series, pd.Series]:
    """
    Reindex a single segment's data to a complete daily date range.

    Aggregates duplicate dates by sum, then reindexes to fill gaps.
    Tracks which days had observations (present) and which had zero
    values (zero) before filling missing days with 0.

    Args:
        group:     DataFrame rows for one segment.
        date_col:  Name of the date column.
        value_col: Name of the volume/actual column.

    Returns:
        (dates, values, present, zero) where:
            dates:   DatetimeIndex of every calendar day in the series span
            values:  float Series, missing days filled with 0
            present: bool Series, True = day had an observed value
            zero:    bool Series, True = observed or filled value is 0
    """
    g = group[[date_col, value_col]].copy()
    g[date_col] = pd.to_datetime(g[date_col])
    g = g.groupby(date_col, as_index=False)[value_col].sum()
    g = g.set_index(date_col).sort_index()

    full_idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
    g = g.reindex(full_idx)

    present = g[value_col].notna()
    g[value_col] = g[value_col].fillna(0.0)
    zero = g[value_col] == 0.0

    return full_idx, g[value_col], present, zero


def run_monitoring(
    df: pd.DataFrame,
    run_cfg: RunConfig,
    monitor_cfg: MonitorConfig,
) -> pd.DataFrame:
    """
    Run all 13 STARS diagnostic tests for every segment in df.

    Groups the input by (strata_id, entity_id, patient_type_rollup, service_line),
    splits each series into training and recent windows according to run_cfg,
    and calls all test functions from stars/tests.py.

    Args:
        df:          Daily DataFrame with columns: strata_id, entity_id,
                     patient_type_rollup, service_line, date, actual, mesh.
        run_cfg:     RunConfig built from CLI args (defines window sizes).
        monitor_cfg: MonitorConfig with hard-coded STARS thresholds.

    Returns:
        DataFrame with one row per segment. Columns include the four segment
        keys, feature_segment, mesh, and for each of the 13 indicators:
        ``{name}_value`` (float) and ``{name}_flag`` (bool).
    """
    results: list[dict] = []
    group_cols = [c for c in _GROUP_COLS if c in df.columns]

    for keys, group in df.groupby(group_cols, dropna=False):
        keys_dict = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))

        dates, values, present, zero = _prepare_series(group)

        # Split into train and recent windows
        most_recent = dates.max()
        recent_start = most_recent - pd.Timedelta(days=run_cfg.recent_days - 1)

        recent_mask = dates >= recent_start
        train_mask = ~recent_mask

        if run_cfg.train_days is not None:
            train_start = recent_start - pd.Timedelta(days=run_cfg.train_days)
            train_mask = train_mask & (dates >= train_start)

        train_vals   = values[train_mask].values
        recent_vals  = values[recent_mask].values
        train_pres   = present[train_mask].values
        recent_pres  = present[recent_mask].values
        train_zero   = zero[train_mask].values
        recent_zero  = zero[recent_mask].values
        all_vals     = values.values  # full series for outlier test

        row: dict = {
            **keys_dict,
            "feature_segment": "|".join(str(keys_dict.get(c, "")) for c in _GROUP_COLS),
            "mesh": group["mesh"].iloc[0] if "mesh" in group.columns else None,
        }

        # ── Stability ────────────────────────────────────────────────────────
        row["ks_distribution_value"],    row["ks_distribution_flag"]    = test_ks_distribution(train_vals, recent_vals, monitor_cfg)
        row["level_shift_value"],        row["level_shift_flag"]        = test_level_shift(train_vals, recent_vals, monitor_cfg)
        row["dw_shift_value"],           row["dw_shift_flag"]           = test_dw_shift(train_vals, recent_vals, monitor_cfg)
        row["slope_change_ratio_value"], row["slope_change_ratio_flag"] = test_slope_change(train_vals, recent_vals, monitor_cfg)
        row["stationarity_value"],       row["stationarity_flag"]       = test_stationarity(train_vals, recent_vals, monitor_cfg)
        row["trend_significance_value"], row["trend_significance_flag"] = test_trend_significance(train_vals, recent_vals, monitor_cfg)

        # ── Truthfulness ─────────────────────────────────────────────────────
        row["coverage_shift_value"],  row["coverage_shift_flag"]  = test_coverage_shift(train_pres, recent_pres, monitor_cfg)
        row["sparsity_change_value"], row["sparsity_change_flag"] = test_sparsity_change(train_zero, recent_zero, monitor_cfg)

        # ── Abundance ────────────────────────────────────────────────────────
        row["low_volume_value"], row["low_volume_flag"] = test_low_volume(train_vals, monitor_cfg)

        # ── Regularity ───────────────────────────────────────────────────────
        row["volatility_shift_value"],  row["volatility_shift_flag"]  = test_volatility_shift(train_vals, recent_vals, monitor_cfg)
        row["outlier_rate_value"],      row["outlier_rate_flag"]      = test_outlier_rate(all_vals, monitor_cfg)
        row["acf_divergence_value"],    row["acf_divergence_flag"]    = test_acf_divergence(train_vals, recent_vals, monitor_cfg)
        row["dow_pattern_shift_value"], row["dow_pattern_shift_flag"] = test_dow_pattern_shift(train_vals, recent_vals, monitor_cfg)

        results.append(row)
        logger.debug("Segment processed", extra={"feature_segment": row["feature_segment"]})

    logger.info("run_monitoring complete", extra={"n_segments": len(results)})
    return pd.DataFrame(results)


_STABILITY_FLAGS = [
    "ks_distribution_flag", "level_shift_flag", "dw_shift_flag",
    "slope_change_ratio_flag", "stationarity_flag", "trend_significance_flag",
]
_TRUTHFULNESS_FLAGS = ["coverage_shift_flag", "sparsity_change_flag"]
_ABUNDANCE_FLAGS    = ["low_volume_flag"]
_REGULARITY_FLAGS   = [
    "volatility_shift_flag", "outlier_rate_flag",
    "acf_divergence_flag", "dow_pattern_shift_flag",
]


def apply_thresholds(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive violation counts and is_flagged from pre-computed flag columns.

    Vectorised — no per-series loops. Designed to be called after run_monitoring().
    Can also be called independently for re-classification (e.g., threshold tuning)
    without re-running the expensive statistical tests.

    Args:
        stats_df: DataFrame produced by run_monitoring(), with one row per
                  segment and boolean flag columns for each indicator.

    Returns:
        The same DataFrame with five additional columns:
            stability_violations (int)
                Number of Stability flags set (0–6).
            truthfulness_violations (int)
                Number of Truthfulness flags set (0–2).
            abundance_violations (int)
                Number of Abundance flags set (0–1).
            regularity_violations (int)
                Number of Regularity flags set (0–4).
            is_flagged (bool)
                True if any family has at least one violation.
    """
    df = stats_df.copy()

    def _count_flags(cols: list[str]) -> pd.Series:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            logger.warning(
                "Expected flag columns missing from stats_df",
                extra={"missing_columns": missing},
            )
        present = [c for c in cols if c in df.columns]
        if not present:
            return pd.Series(0, index=df.index, dtype=int)
        return df[present].fillna(False).astype(bool).sum(axis=1).astype(int)

    df["stability_violations"]    = _count_flags(_STABILITY_FLAGS)
    df["truthfulness_violations"] = _count_flags(_TRUTHFULNESS_FLAGS)
    df["abundance_violations"]    = _count_flags(_ABUNDANCE_FLAGS)
    df["regularity_violations"]   = _count_flags(_REGULARITY_FLAGS)

    df["is_flagged"] = (
        (df["stability_violations"] > 0)
        | (df["truthfulness_violations"] > 0)
        | (df["abundance_violations"] > 0)
        | (df["regularity_violations"] > 0)
    )

    return df
