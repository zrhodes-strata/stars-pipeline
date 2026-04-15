"""
tests.py
========
STARS diagnostic test functions — one function per indicator.

Each function:
- Accepts window arrays and relevant thresholds from MonitorConfig
- Returns a (metric_value: float, flag: bool) tuple
- Is fully documented with: broken assumption, STARS family, statistical
  method, inputs, outputs, threshold, and interpretation notes

Window convention
-----------------
``train``   Daily volume values from the training window (earlier period).
``recent``  Daily volume values from the recent window (most recent N days).

For coverage/sparsity tests, the arrays are boolean:
``train_present``   True where the training window had an observation.
``recent_present``  True where the recent window had an observation.
``train_zero``      True where the training window value was zero.
``recent_zero``     True where the recent window value was zero.

Guard clauses
-------------
All functions return (float("nan"), False) when input arrays are too short
to compute the statistic reliably. Callers should treat NaN metric_value as
"insufficient data" rather than "no problem".
"""
from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from scipy.stats import ks_2samp, linregress, norm, ttest_ind
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.tsa.stattools import acf, kpss

from stars_pipeline.config import MonitorConfig


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fisher_z_test(r1: float, n1: int, r2: float, n2: int) -> float:
    """
    Two-sample Fisher Z-transform test for equality of two correlations.

    Tests H0: rho1 == rho2 using the Fisher Z transformation.
    Used by test_acf_divergence and test_dow_pattern_shift.

    Args:
        r1: Pearson correlation from sample 1.
        n1: Sample size of sample 1.
        r2: Pearson correlation from sample 2.
        n2: Sample size of sample 2.

    Returns:
        Two-sided p-value. Small p-value (< alpha) indicates the
        correlations are significantly different.
    """
    z1 = np.arctanh(np.clip(r1, -0.9999, 0.9999))
    z2 = np.arctanh(np.clip(r2, -0.9999, 0.9999))
    se = np.sqrt(1.0 / max(n1 - 3, 1) + 1.0 / max(n2 - 3, 1))
    if se == 0:
        return 1.0
    z_stat = (z1 - z2) / se
    return float(2.0 * (1.0 - norm.cdf(abs(z_stat))))


# ── Stability ─────────────────────────────────────────────────────────────────


def test_ks_distribution(
    train: np.ndarray,
    recent: np.ndarray,
    cfg: MonitorConfig,
) -> tuple[float, bool]:
    """
    Detect distributional shift between the training and recent windows.

    STARS Family:       Stability
    Broken Assumption:  Not Stable — the volume distribution has changed
    Method:             Kolmogorov-Smirnov two-sample test
    Threshold:          cfg.ks_d_threshold (default 0.30)

    The KS statistic measures the maximum absolute difference between the
    empirical CDFs of the two samples. A large KS statistic indicates the
    two windows are drawn from different distributions.

    Args:
        train:  Daily volume values from the training window.
        recent: Daily volume values from the recent window.
        cfg:    MonitorConfig with hard-coded thresholds.

    Returns:
        (ks_statistic, flag) where flag=True if ks_statistic >= cfg.ks_d_threshold.
        Returns (nan, False) if either array has fewer than 2 observations.
    """
    if len(train) < 2 or len(recent) < 2:
        return (float("nan"), False)
    stat, _ = ks_2samp(train, recent)
    return (float(stat), bool(stat >= cfg.ks_d_threshold))


def test_level_shift(
    train: np.ndarray,
    recent: np.ndarray,
    cfg: MonitorConfig,
) -> tuple[float, bool]:
    """
    Detect a statistically significant and practically large shift in the mean.

    STARS Family:       Stability
    Broken Assumption:  Not Stable — the mean level of the series has changed
    Method:             Welch's t-test (unequal variances) with Cohen's d gate
    Threshold:          cfg.level_shift_min_cohen_d (default 1.00 — large effect)

    Two-stage test:
    1. Welch's t-test must be significant (p < cfg.alpha) — rules out noise.
    2. Cohen's d must be >= cfg.level_shift_min_cohen_d — ensures the shift
       is large enough to be practically meaningful, not just statistically so.

    Args:
        train:  Daily volume values from the training window.
        recent: Daily volume values from the recent window.
        cfg:    MonitorConfig with hard-coded thresholds.

    Returns:
        (cohens_d, flag) where flag=True if both gates pass.
        Returns (nan, False) if either array has fewer than 2 observations.
    """
    if len(train) < 2 or len(recent) < 2:
        return (float("nan"), False)
    _, p_value = ttest_ind(train, recent, equal_var=False)
    if p_value >= cfg.alpha:
        return (0.0, False)
    pooled_std = np.sqrt(
        (train.std(ddof=1) ** 2 + recent.std(ddof=1) ** 2) / 2.0
    )
    if pooled_std == 0:
        return (0.0, False)
    cohens_d = abs(recent.mean() - train.mean()) / pooled_std
    return (float(cohens_d), bool(cohens_d >= cfg.level_shift_min_cohen_d))


def test_dw_shift(
    train: np.ndarray,
    recent: np.ndarray,
    cfg: MonitorConfig,
) -> tuple[float, bool]:
    """
    Detect a shift in residual autocorrelation structure (Durbin-Watson).

    STARS Family:       Stability
    Broken Assumption:  Not Stable — the autocorrelation of residuals has changed
    Method:             OLS linear fit on each window; Durbin-Watson on residuals
    Threshold:          cfg.dw_delta_threshold (default 1.15)

    The Durbin-Watson statistic ranges from 0 to 4:
      ~2 = no autocorrelation, ~0 = strong positive, ~4 = strong negative.
    A large delta between train and recent indicates a structural change in
    how successive residuals relate to each other.

    Args:
        train:  Daily volume values from the training window.
        recent: Daily volume values from the recent window.
        cfg:    MonitorConfig with hard-coded thresholds.

    Returns:
        (dw_delta, flag) where dw_delta = |DW_recent - DW_train|.
        Returns (nan, False) if either array has fewer than 3 observations.
    """
    def _dw(arr: np.ndarray) -> float:
        if len(arr) < 3:
            return float("nan")
        x = sm.add_constant(np.arange(len(arr), dtype=float))
        resid = sm.OLS(arr, x).fit().resid
        denom = float(np.sum(resid ** 2))
        if denom == 0:
            return float("nan")
        return float(np.sum(np.diff(resid) ** 2) / denom)

    dw_train = _dw(train)
    dw_recent = _dw(recent)
    if np.isnan(dw_train) or np.isnan(dw_recent):
        return (float("nan"), False)
    delta = abs(dw_recent - dw_train)
    return (float(delta), bool(delta >= cfg.dw_delta_threshold))


def test_slope_change(
    train: np.ndarray,
    recent: np.ndarray,
    cfg: MonitorConfig,
) -> tuple[float, bool]:
    """
    Detect a disproportionate change in trend slope between windows.

    STARS Family:       Stability
    Broken Assumption:  Not Stable — the trend direction or magnitude has changed
    Method:             Linear regression slope ratio
    Threshold:          cfg.slope_change_ratio_threshold (default 1.50)

    Computes |slope_recent| / |slope_train|. A ratio > 1.50 indicates the
    recent trend is at least 50% steeper than the historical trend.
    Returns (nan, False) when the training slope is zero (no baseline trend).

    Args:
        train:  Daily volume values from the training window.
        recent: Daily volume values from the recent window.
        cfg:    MonitorConfig with hard-coded thresholds.

    Returns:
        (slope_ratio, flag) where flag=True if ratio >= cfg.slope_change_ratio_threshold.
        Returns (nan, False) if either array has fewer than 2 observations or
        if the training slope is zero.
    """
    if len(train) < 2 or len(recent) < 2:
        return (float("nan"), False)
    slope_train = linregress(np.arange(len(train)), train).slope
    slope_recent = linregress(np.arange(len(recent)), recent).slope
    if slope_train == 0:
        return (float("nan"), False)
    ratio = abs(slope_recent) / abs(slope_train)
    return (float(ratio), bool(ratio >= cfg.slope_change_ratio_threshold))


def test_stationarity(
    train: np.ndarray,
    recent: np.ndarray,
    cfg: MonitorConfig,
) -> tuple[float, bool]:
    """
    Detect a transition from stationary to non-stationary behaviour (KPSS).

    STARS Family:       Stability
    Broken Assumption:  Not Stable — the series has lost its stationary structure
    Method:             KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test
    Threshold:          cfg.kpss_alpha (default 0.10)

    Flag fires when:
      - Training window is stationary  (KPSS p-value >  cfg.kpss_alpha)
      - Recent window is non-stationary (KPSS p-value <= cfg.kpss_alpha)

    This one-directional test specifically catches the case where a previously
    stable series has become non-stationary — i.e., the model's stationarity
    assumption has been violated in the recent window.

    Args:
        train:  Daily volume values from the training window.
        recent: Daily volume values from the recent window.
        cfg:    MonitorConfig with hard-coded thresholds.

    Returns:
        (kpss_statistic_recent, flag).
        Returns (nan, False) if either array has fewer than 10 observations
        (KPSS requires sufficient data to be reliable).
    """
    if len(train) < 10 or len(recent) < 10:
        return (float("nan"), False)
    try:
        _, p_train, _, _ = kpss(train, regression="c", nlags="auto")
        stat_recent, p_recent, _, _ = kpss(recent, regression="c", nlags="auto")
    except Exception:
        return (float("nan"), False)
    train_stationary = p_train >= cfg.kpss_alpha
    recent_nonstationary = p_recent <= cfg.kpss_alpha
    return (float(stat_recent), bool(train_stationary and recent_nonstationary))


def test_trend_significance(
    train: np.ndarray,
    recent: np.ndarray,
    cfg: MonitorConfig,
) -> tuple[float, bool]:
    """
    Detect a statistically significant linear trend in the recent window.

    STARS Family:       Stability
    Broken Assumption:  Not Stable — a significant new trend has emerged recently
    Method:             Linear regression OLS p-value on the recent window
    Threshold:          cfg.trend_p_value_threshold (default 0.05)

    Unlike test_slope_change (which compares train vs recent magnitude),
    this test asks whether the recent window has a trend that is statistically
    distinguishable from a flat line — regardless of what the training trend was.

    Args:
        train:  Daily volume values from the training window (unused in computation;
                included for interface consistency with other test functions).
        recent: Daily volume values from the recent window.
        cfg:    MonitorConfig with hard-coded thresholds.

    Returns:
        (p_value, flag) where flag=True if p_value < cfg.trend_p_value_threshold.
        Returns (nan, False) if recent has fewer than 3 observations.
    """
    if len(recent) < 3:
        return (float("nan"), False)
    result = linregress(np.arange(len(recent), dtype=float), recent)
    p_value = float(result.pvalue)
    return (p_value, bool(p_value < cfg.trend_p_value_threshold))
