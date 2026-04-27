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

import warnings

import numpy as np
import statsmodels.api as sm
from scipy.stats import ks_2samp, linregress, norm, ttest_ind
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.tools.sm_exceptions import InterpolationWarning
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
    Method:             |slope_delta| / (|slope_train| + eps)
                        where slope_delta = slope_recent - slope_train
    Threshold:          cfg.slope_change_ratio_threshold (default 1.50)

    Measures how large the slope *change* is relative to the training baseline.
    A small eps prevents division by zero while still producing large ratios
    when slope_train ≈ 0 but slope_recent is substantial.

    Args:
        train:  Daily volume values from the training window.
        recent: Daily volume values from the recent window.
        cfg:    MonitorConfig with hard-coded thresholds.

    Returns:
        (slope_ratio, flag) where flag=True if ratio >= cfg.slope_change_ratio_threshold.
        Returns (nan, False) if either array has fewer than 2 observations.
    """
    if len(train) < 2 or len(recent) < 2:
        return (float("nan"), False)
    slope_train = linregress(np.arange(len(train)), train).slope
    slope_recent = linregress(np.arange(len(recent)), recent).slope
    eps = 1e-8
    slope_delta = slope_recent - slope_train
    ratio = abs(slope_delta) / (abs(slope_train) + eps)
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
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


# ── Truthfulness ──────────────────────────────────────────────────────────────


def test_coverage_shift(
    train_present: np.ndarray,
    recent_present: np.ndarray,
    cfg: MonitorConfig,
) -> tuple[float, bool]:
    """
    Detect a significant shift in data coverage (proportion of observed days).

    STARS Family:       Truthfulness
    Broken Assumption:  Not Truthful — missing data pattern has changed,
                        suggesting the data no longer accurately represents
                        the underlying process
    Method:             Two-proportion z-test on non-missing day rates
    Threshold:          cfg.coverage_delta_threshold (default 0.30)
                        and cfg.alpha (default 0.05) for statistical gate

    Coverage rate = non-missing_days / total_days in the window.
    Both statistical significance (z-test p < alpha) and practical magnitude
    (|delta| >= coverage_delta_threshold) must be met to flag.

    Args:
        train_present:  Boolean array; True = day had an observed value in train.
        recent_present: Boolean array; True = day had an observed value in recent.
        cfg:            MonitorConfig with hard-coded thresholds.

    Returns:
        (coverage_delta, flag) where coverage_delta = |rate_recent - rate_train|.
        Returns (nan, False) if either array is empty.
    """
    if len(train_present) == 0 or len(recent_present) == 0:
        return (float("nan"), False)
    cov_train = float(train_present.sum()) / len(train_present)
    cov_recent = float(recent_present.sum()) / len(recent_present)
    delta = abs(cov_recent - cov_train)
    count = np.array([int(train_present.sum()), int(recent_present.sum())])
    nobs = np.array([len(train_present), len(recent_present)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, p_value = proportions_ztest(count, nobs)
    flag = bool((float(p_value) < cfg.alpha) and (delta >= cfg.coverage_delta_threshold))
    return (float(delta), flag)


def test_sparsity_change(
    train_zero: np.ndarray,
    recent_zero: np.ndarray,
    cfg: MonitorConfig,
) -> tuple[float, bool]:
    """
    Detect a significant shift in sparsity rate (proportion of zero-value days).

    STARS Family:       Truthfulness
    Broken Assumption:  Not Truthful — the zero-value pattern has changed,
                        which may indicate data suppression, reporting changes,
                        or structural operational changes
    Method:             Two-proportion z-test on zero-value day rates
    Threshold:          cfg.sparsity_delta_threshold (default 0.30)
                        and cfg.alpha (default 0.05) for statistical gate

    Sparsity rate = zero_value_days / total_days in the window.
    Both statistical significance (z-test p < alpha) and practical magnitude
    (|delta| >= sparsity_delta_threshold) must be met to flag.

    Args:
        train_zero:  Boolean array; True = day had a zero value in train window.
        recent_zero: Boolean array; True = day had a zero value in recent window.
        cfg:         MonitorConfig with hard-coded thresholds.

    Returns:
        (sparsity_delta, flag) where sparsity_delta = |rate_recent - rate_train|.
        Returns (nan, False) if either array is empty.
    """
    if len(train_zero) == 0 or len(recent_zero) == 0:
        return (float("nan"), False)
    sparsity_train = float(train_zero.sum()) / len(train_zero)
    sparsity_recent = float(recent_zero.sum()) / len(recent_zero)
    delta = abs(sparsity_recent - sparsity_train)
    count = np.array([int(train_zero.sum()), int(recent_zero.sum())])
    nobs = np.array([len(train_zero), len(recent_zero)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, p_value = proportions_ztest(count, nobs)
    flag = bool((float(p_value) < cfg.alpha) and (delta >= cfg.sparsity_delta_threshold))
    return (float(delta), flag)


# ── Abundance ─────────────────────────────────────────────────────────────────


def test_low_volume(
    train: np.ndarray,
    cfg: MonitorConfig,
) -> tuple[float, bool]:
    """
    Detect insufficient signal volume in the training window.

    STARS Family:       Abundance
    Broken Assumption:  Not Abundant — too few observations to infer patterns
    Method:             Average monthly volume (assumes 30.44 days/month)
    Threshold:          cfg.low_volume_monthly_threshold (default 3.0 per month)

    Converts the total training volume to an average monthly rate and flags
    when this rate is below the threshold. Note: ``train`` should contain
    the raw daily volume values (not just presence indicators).

    Args:
        train: Daily volume values from the training window.
        cfg:   MonitorConfig with hard-coded thresholds.

    Returns:
        (avg_monthly_volume, flag) where flag=True if avg_monthly < threshold.
        Returns (nan, True) if train is empty (no data = flag by convention).
    """
    if len(train) == 0:
        return (float("nan"), True)
    avg_monthly = float(train.mean()) * 30.44
    return (float(avg_monthly), bool(avg_monthly < cfg.low_volume_monthly_threshold))


# ── Regularity ────────────────────────────────────────────────────────────────


def test_volatility_shift(
    train: np.ndarray,
    recent: np.ndarray,
    cfg: MonitorConfig,
) -> tuple[float, bool]:
    """
    Detect a disproportionate increase in relative variability (CV ratio).

    STARS Family:       Regularity
    Broken Assumption:  Not Regular — the series has become much more volatile
    Method:             Coefficient of Variation ratio:
                        (sigma/mu)_recent / (sigma/mu)_train
    Threshold:          cfg.volatility_ratio_threshold (default 3.50)

    The CV (sigma/mu) normalises variability by the mean, making the ratio
    scale-independent. A ratio of 3.5 means the recent window is 3.5x more
    variable relative to its mean than the training window was.

    Args:
        train:  Daily volume values from the training window.
        recent: Daily volume values from the recent window.
        cfg:    MonitorConfig with hard-coded thresholds.

    Returns:
        (cv_ratio, flag) where flag=True if cv_ratio >= cfg.volatility_ratio_threshold.
        Returns (nan, False) if either array has fewer than 2 observations,
        or if either mean is zero (CV undefined).
    """
    if len(train) < 2 or len(recent) < 2:
        return (float("nan"), False)
    mean_train, mean_recent = train.mean(), recent.mean()
    if mean_train == 0 or mean_recent == 0:
        return (float("nan"), False)
    cv_train = train.std(ddof=1) / abs(mean_train)
    cv_recent = recent.std(ddof=1) / abs(mean_recent)
    if cv_train == 0:
        return (float("nan"), False)
    ratio = cv_recent / cv_train
    return (float(ratio), bool(ratio >= cfg.volatility_ratio_threshold))


def test_outlier_rate(
    train: np.ndarray,
    recent: np.ndarray,
    cfg: MonitorConfig,
) -> tuple[float, bool]:
    """
    Detect an excessive proportion of outlier points in the recent window.

    STARS Family:       Regularity
    Broken Assumption:  Not Regular — extreme values dominate the recent window
    Method:             Train-window MAD baseline; modified z-score on recent values.
                        rate = outlier_count / len(recent)
    Threshold:          cfg.outlier_z_threshold (default 3.50) for outlier
                        definition; cfg.outlier_rate_threshold (default 0.40)
                        for the rate gate

    The modified z-score uses the training window to establish the MAD baseline
    (median and MAD), then applies it to the recent window. This is more robust
    than a full-series approach because recent outliers do not inflate the baseline.

    Modified z-score: 0.6745 * |value - train_median| / train_MAD
    If train_MAD is zero, falls back to train_std as the scale.

    Args:
        train:  Daily volume values from the training window.
        recent: Daily volume values from the recent window.
        cfg:    MonitorConfig with hard-coded thresholds.

    Returns:
        (outlier_rate, flag) where flag=True if rate >= cfg.outlier_rate_threshold.
        Returns (nan, False) if train has fewer than 3 observations or recent is empty.
    """
    if len(train) < 3 or len(recent) == 0:
        return (float("nan"), False)
    train_median = float(np.median(train))
    train_mad = float(np.median(np.abs(train - train_median)))
    if train_mad == 0:
        train_std = float(train.std(ddof=1))
        if train_std == 0:
            # Completely flat training window: any deviation in recent is an outlier.
            outlier_mask = recent != train_median
            rate = float(outlier_mask.sum()) / len(recent)
            return (float(rate), bool(rate >= cfg.outlier_rate_threshold))
        scale = train_std / 0.6745
    else:
        scale = train_mad / 0.6745
    modified_z = np.abs(recent - train_median) / scale
    outlier_mask = modified_z > cfg.outlier_z_threshold
    rate = float(outlier_mask.sum()) / len(recent)
    return (float(rate), bool(rate >= cfg.outlier_rate_threshold))


def test_acf_divergence(
    train: np.ndarray,
    recent: np.ndarray,
    cfg: MonitorConfig,
) -> tuple[float, bool]:
    """
    Detect a significant change in lag-1 autocorrelation structure.

    STARS Family:       Regularity
    Broken Assumption:  Not Regular — the short-term autocorrelation pattern
                        has changed (predictable regularities have been lost
                        or new ones introduced)
    Method:             Fisher Z-transform test comparing lag-1 ACF between
                        training and recent windows
    Threshold:          cfg.acf_divergence_p_threshold (default 0.05)

    The lag-1 ACF captures how much each day predicts the next. A significant
    shift suggests the series' short-term dependence structure has changed.

    Args:
        train:  Daily volume values from the training window.
        recent: Daily volume values from the recent window.
        cfg:    MonitorConfig with hard-coded thresholds.

    Returns:
        (p_value, flag) where flag=True if p_value < cfg.acf_divergence_p_threshold.
        Returns (nan, False) if either array has fewer than 5 observations.
    """
    if len(train) < 5 or len(recent) < 5:
        return (float("nan"), False)
    acf_train = float(acf(train, nlags=1, fft=False)[1])
    acf_recent = float(acf(recent, nlags=1, fft=False)[1])
    p_value = _fisher_z_test(acf_train, len(train), acf_recent, len(recent))
    return (float(p_value), bool(p_value < cfg.acf_divergence_p_threshold))


def test_dow_pattern_shift(
    train: np.ndarray,
    recent: np.ndarray,
    cfg: MonitorConfig,
) -> tuple[float, bool]:
    """
    Detect a significant change in day-of-week autocorrelation (lag-7 ACF).

    STARS Family:       Regularity
    Broken Assumption:  Not Regular — the weekly seasonal pattern has changed
    Method:             Fisher Z-transform test comparing lag-7 ACF between
                        training and recent windows
    Threshold:          cfg.acf_divergence_p_threshold (default 0.05)
                        (reused — no separate threshold needed)

    Lag-7 ACF captures weekly periodicity. A significant shift in lag-7 ACF
    between train and recent windows indicates the day-of-week pattern has
    changed — e.g., weekday/weekend volume ratios have shifted.

    Args:
        train:  Daily volume values from the training window.
        recent: Daily volume values from the recent window.
        cfg:    MonitorConfig with hard-coded thresholds.

    Returns:
        (p_value, flag) where flag=True if p_value < cfg.acf_divergence_p_threshold.
        Returns (nan, False) if either array has fewer than 14 observations
        (need at least 2 full weeks for lag-7 to be meaningful).
    """
    if len(train) < 14 or len(recent) < 14:
        return (float("nan"), False)
    acf_train_7 = float(acf(train, nlags=7, fft=False)[7])
    acf_recent_7 = float(acf(recent, nlags=7, fft=False)[7])
    p_value = _fisher_z_test(acf_train_7, len(train), acf_recent_7, len(recent))
    return (float(p_value), bool(p_value < cfg.acf_divergence_p_threshold))
