"""
config.py
=========
Configuration dataclasses for the STARS pipeline.

Two dataclasses with clearly separated concerns:

RunConfig
    Data-selection and execution parameters built from CLI arguments.
    Mutable — constructed fresh each run.

MonitorConfig
    Statistical thresholds for the STARS diagnostic framework.
    FROZEN — no threshold may be changed at runtime.
    All values are sourced from the canonical STARS indicator table.
    To change a threshold, update this file AND the design specification:
        docs/superpowers/specs/2026-04-14-stars-pipeline-design.md
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass
class RunConfig:
    """
    Data-selection and execution parameters derived from CLI arguments.

    Attributes
    ----------
    strata_ids:
        List of integer strata IDs to include in the evaluation.
    collection_id:
        Collection identifier. Passed to the SQL layer as a bind parameter.
        TODO: wire into actuals.sql WHERE clause once schema is confirmed.
    run_mode:
        Run selection mode. One of "today", "most-recent", "date-range", or None.
        None means collection_id is used directly (explicit override).
        Default "today" when collection_id is not provided.
    run_mode_date_from:
        Start date for date-range mode. Also set for --run-mode-date shorthand.
    run_mode_date_to:
        End date for date-range mode. Also set for --run-mode-date shorthand.
    date_from:
        Start of the data pull window (inclusive).
    date_to:
        End of the data pull window (inclusive).
    recent_days:
        Number of calendar days treated as the "recent" window for shift
        detection. The most recent ``recent_days`` days of each series are
        compared against the training window.
    train_days:
        Number of calendar days in the training window. If None, all
        available data preceding the recent window is used as training data.
    entity_id:
        Optional filter to narrow the pull to a single entity within the
        selected strata.
    patient_type:
        Optional filter to narrow the pull to a single patient type rollup.
    service_line:
        Optional filter to narrow the pull to a single service line.
    output_path:
        Filesystem path where the long-format results CSV will be written.
        For SageMaker, use ``/opt/ml/processing/output/stars_results.csv``.
    """

    strata_ids: list[int]
    collection_id: str | None
    run_mode: str | None          # "today" | "most-recent" | "date-range" | None
    run_mode_date_from: date | None
    run_mode_date_to: date | None
    date_from: date
    date_to: date
    recent_days: int
    train_days: int | None
    entity_id: str | None
    patient_type: str | None
    service_line: str | None
    output_path: Path


@dataclass(frozen=True)
class MonitorConfig:
    """
    Statistical thresholds for the STARS diagnostic framework.

    This dataclass is FROZEN — no field is settable after construction.
    Instantiate with no arguments to get the canonical threshold set:

        cfg = MonitorConfig()

    Threshold reference
    -------------------
    Each field documents its STARS family, the indicator it governs,
    the statistical test used, and the canonical threshold value.

    Stability
    ~~~~~~~~~
    ks_d_threshold
        KS two-sample test statistic threshold for distributional shift.
        Flag when KS statistic >= 0.30.

    level_shift_min_cohen_d
        Cohen's d practical significance gate for the Welch's t-test level
        shift indicator. The Welch's t-test p-value must also pass ``alpha``
        before Cohen's d is evaluated.
        Flag when Cohen's d >= 1.15 (large effect size).

    dw_delta_threshold
        Absolute change in the OLS Durbin-Watson statistic between the
        training and recent windows. Measures residual autocorrelation shift.
        Flag when |DW_recent - DW_train| >= 1.50.

    slope_change_ratio_threshold
        Ratio of the slope change to the training slope magnitude:
        |slope_recent - slope_train| / (|slope_train| + eps).
        Flag when ratio >= 1.50 (one of three gates in test_trend_change).

    slope_threshold
        Minimum absolute slope magnitude gate in test_trend_change.
        Prevents flagging near-flat series where a ratio is meaningless.
        Flag requires max(|slope_train|, |slope_recent|) >= 0.05.

    kpss_alpha
        Significance level for the KPSS stationarity test.
        Flag when training window is stationary (KPSS p > kpss_alpha) but
        the recent window is non-stationary (KPSS p <= kpss_alpha).
        Uses 0.10 (more sensitive than the global alpha = 0.05).

    trend_p_value_threshold
        Linear regression p-value threshold for trend change significance.
        Flag when the recent window's trend p-value < 0.20.

    Truthfulness
    ~~~~~~~~~~~~
    coverage_delta_threshold
        Minimum absolute shift in coverage rate (proportion of non-missing
        days) required to flag, after the two-proportion z-test passes alpha.
        Flag when |coverage_recent - coverage_train| >= 0.40.

    sparsity_delta_threshold
        Minimum absolute shift in sparsity rate (proportion of zero-value
        days) required to flag, after the two-proportion z-test passes alpha.
        Flag when |sparsity_recent - sparsity_train| >= 0.30.

    Abundance
    ~~~~~~~~~
    low_volume_monthly_threshold
        Minimum acceptable average monthly volume in the training window.
        Flag when avg_monthly_volume_train < 3.0.

    Regularity
    ~~~~~~~~~~
    volatility_ratio_threshold
        Coefficient of variation ratio: (sigma/mu)_recent / (sigma/mu)_train.
        Bidirectional: flag when cv_ratio >= 1.50 (volatility increased)
        OR cv_ratio <= 1/1.50 (volatility collapsed).

    outlier_z_threshold
        MAD multiplier defining an outlier: a point is an outlier when
        |value - median| > outlier_z_threshold * MAD.

    outlier_rate_threshold
        Maximum acceptable fraction of outlier points in the recent window.
        Flag when outlier_count / len(recent) >= 0.40.

    acf_divergence_p_threshold
        Fisher Z-transform p-value threshold for ACF lag-1 divergence.
        Also reused for the day-of-week pattern shift (lag-7 ACF) test.
        Flag when p-value < 0.05.

    Global
    ~~~~~~
    alpha
        Statistical significance level applied to all hypothesis tests
        as a p-value gate before evaluating magnitude thresholds.
    """

    # ── Stability ─────────────────────────────────────────────────────────────
    ks_d_threshold: float = 0.30
    level_shift_min_cohen_d: float = 1.15
    dw_delta_threshold: float = 1.50
    slope_change_ratio_threshold: float = 1.50
    slope_threshold: float = 0.05
    kpss_alpha: float = 0.10
    trend_p_value_threshold: float = 0.20

    # ── Truthfulness ──────────────────────────────────────────────────────────
    coverage_delta_threshold: float = 0.40
    sparsity_delta_threshold: float = 0.30

    # ── Abundance ─────────────────────────────────────────────────────────────
    low_volume_monthly_threshold: float = 3.00

    # ── Regularity ────────────────────────────────────────────────────────────
    volatility_ratio_threshold: float = 1.50
    outlier_z_threshold: float = 3.50
    outlier_rate_threshold: float = 0.40
    acf_divergence_p_threshold: float = 0.05

    # ── Global ────────────────────────────────────────────────────────────────
    alpha: float = 0.05
