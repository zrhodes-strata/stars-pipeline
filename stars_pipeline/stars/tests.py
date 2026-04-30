"""
tests.py
========
STARS diagnostic test functions — one function per indicator.

Each function:
- Accepts window arrays and relevant thresholds from MonitorConfig
- Returns a dict with at minimum ``value`` (float) and ``flag`` (bool), plus
  intermediate statistics for diagnostics and downstream analysis
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
All functions return a dict with ``value=nan`` and ``flag=False`` when input
arrays are too short to compute the statistic reliably. Callers should treat
NaN ``value`` as "insufficient data" rather than "no problem".

Return dict convention
----------------------
Every dict contains:
    value   float   Primary statistic (the one stored as metric_value)
    flag    bool    Whether the threshold was exceeded

Additional keys vary by test and are documented in each function's Returns
section. monitor.py writes them as ``{metric_name}_{key}`` columns.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ks_2samp, norm, ttest_ind
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import acf, kpss

from stars_pipeline.config import MonitorConfig


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fisher_z_test(r1: float, n1: int, r2: float, n2: int) -> float:
    """
    Two-sample Fisher Z-transform test for equality of two correlations.

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


def _nan_dict(value: float = float("nan"), flag: bool = False, **extras) -> dict:
    return {"value": value, "flag": flag, **extras}


# ── Stability ─────────────────────────────────────────────────────────────────


def test_ks_distribution(
    train: np.ndarray,
    recent: np.ndarray,
    cfg: MonitorConfig,
) -> dict:
    """
    Detect distributional shift between the training and recent windows.

    STARS Family:       Stability
    Broken Assumption:  Not Stable — the volume distribution has changed
    Method:             Kolmogorov-Smirnov two-sample test
    Threshold:          cfg.ks_d_threshold (default 0.30)

    Both the KS statistic gate (stat >= threshold) and the p-value gate
    (p < alpha) must pass.

    Args:
        train:  Daily volume values from the training window.
        recent: Daily volume values from the recent window.
        cfg:    MonitorConfig with hard-coded thresholds.

    Returns:
        value       KS statistic
        flag        True if stat >= cfg.ks_d_threshold AND p < cfg.alpha
        ks_p_value  p-value from the two-sample KS test
        mean_train  Mean of training window
        mean_recent Mean of recent window
    """
    if len(train) < 10 or len(recent) < 10:
        return _nan_dict(mean_train=float("nan"), mean_recent=float("nan"),
                         ks_p_value=float("nan"))
    stat, p = ks_2samp(train, recent)
    return {
        "value":       float(stat),
        "flag":        bool(stat >= cfg.ks_d_threshold and p < cfg.alpha),
        "ks_p_value":  float(p),
        "mean_train":  float(train.mean()),
        "mean_recent": float(recent.mean()),
    }


def test_level_shift(
    train: np.ndarray,
    recent: np.ndarray,
    cfg: MonitorConfig,
) -> dict:
    """
    Detect a statistically significant and practically large shift in the mean.

    STARS Family:       Stability
    Broken Assumption:  Not Stable — the mean level of the series has changed
    Method:             Welch's t-test (unequal variances) with Cohen's d gate
    Threshold:          cfg.level_shift_min_cohen_d (default 1.15)

    Args:
        train:  Daily volume values from the training window.
        recent: Daily volume values from the recent window.
        cfg:    MonitorConfig with hard-coded thresholds.

    Returns:
        value        Cohen's d (0.0 if t-test p >= alpha)
        flag         True if t-test p < alpha AND cohens_d >= threshold
        p_value      Welch's t-test p-value
        mean_train   Mean of training window
        mean_recent  Mean of recent window
    """
    if len(train) < 2 or len(recent) < 2:
        return _nan_dict(p_value=float("nan"), mean_train=float("nan"),
                         mean_recent=float("nan"))
    _, p_value = ttest_ind(train, recent, equal_var=False)
    mean_train = float(train.mean())
    mean_recent = float(recent.mean())
    if p_value >= cfg.alpha:
        return {"value": 0.0, "flag": False, "p_value": float(p_value),
                "mean_train": mean_train, "mean_recent": mean_recent}
    pooled_std = np.sqrt((train.std(ddof=1) ** 2 + recent.std(ddof=1) ** 2) / 2.0)
    if pooled_std == 0:
        return {"value": 0.0, "flag": False, "p_value": float(p_value),
                "mean_train": mean_train, "mean_recent": mean_recent}
    cohens_d = abs(mean_recent - mean_train) / pooled_std
    return {
        "value":       float(cohens_d),
        "flag":        bool(cohens_d >= cfg.level_shift_min_cohen_d),
        "p_value":     float(p_value),
        "mean_train":  mean_train,
        "mean_recent": mean_recent,
    }


def test_dw_shift(
    train: np.ndarray,
    recent: np.ndarray,
    cfg: MonitorConfig,
) -> dict:
    """
    Detect a shift in residual autocorrelation structure (Durbin-Watson).

    STARS Family:       Stability
    Broken Assumption:  Not Stable — the autocorrelation of residuals has changed
    Method:             OLS linear fit on each window; Durbin-Watson on residuals
    Threshold:          cfg.dw_delta_threshold (default 1.50)

    Args:
        train:  Daily volume values from the training window.
        recent: Daily volume values from the recent window.
        cfg:    MonitorConfig with hard-coded thresholds.

    Returns:
        value      |DW_recent - DW_train|
        flag       True if delta >= cfg.dw_delta_threshold
        dw_train   Durbin-Watson statistic for the training window
        dw_recent  Durbin-Watson statistic for the recent window
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
        return _nan_dict(dw_train=float("nan"), dw_recent=float("nan"))
    delta = abs(dw_recent - dw_train)
    return {
        "value":     float(delta),
        "flag":      bool(delta >= cfg.dw_delta_threshold),
        "dw_train":  dw_train,
        "dw_recent": dw_recent,
    }


def test_trend_change(
    train: np.ndarray,
    recent: np.ndarray,
    cfg: MonitorConfig,
) -> dict:
    """
    Detect a statistically significant change in trend slope between windows.

    STARS Family:       Stability
    Broken Assumption:  Not Stable — the trend has shifted between training and recent
    Method:             Interaction-term OLS: y ~ 1 + t + recent_ind + (t × recent_ind)
    Threshold:          Three gates: p < trend_p_value_threshold, max_slope >= slope_threshold,
                        slope_change_ratio >= slope_change_ratio_threshold

    Args:
        train:  Daily volume values from the training window.
        recent: Daily volume values from the recent window.
        cfg:    MonitorConfig with hard-coded thresholds.

    Returns:
        value               p-value of the interaction term
        flag                True if all three gates pass
        slope_train         OLS slope for the training window
        slope_recent        OLS slope for the recent window
        slope_delta         slope_recent - slope_train
        slope_change_ratio  |slope_delta| / (max(|slope_train|, |slope_recent|) + eps)
    """
    if len(train) < 10 or len(recent) < 10:
        return _nan_dict(slope_train=float("nan"), slope_recent=float("nan"),
                         slope_delta=float("nan"), slope_change_ratio=float("nan"))

    t_train = np.arange(len(train), dtype=float)
    t_recent = np.arange(len(recent), dtype=float) + len(train)
    t_all = np.concatenate([t_train, t_recent])
    y_all = np.concatenate([train, recent]).astype(float)
    ind = np.concatenate([np.zeros(len(train)), np.ones(len(recent))])
    t_x_ind = t_all * ind

    X = sm.add_constant(np.column_stack([t_all, ind, t_x_ind]), has_constant="add")
    try:
        result = sm.OLS(y_all, X).fit()
    except Exception:
        return _nan_dict(slope_train=float("nan"), slope_recent=float("nan"),
                         slope_delta=float("nan"), slope_change_ratio=float("nan"))

    slope_train = float(result.params[1])
    slope_delta = float(result.params[3])
    slope_recent = slope_train + slope_delta
    p_val = float(result.pvalues[3])
    eps = 1e-9
    max_slope = max(abs(slope_train), abs(slope_recent))
    slope_change_ratio = abs(slope_delta) / (max_slope + eps)

    return {
        "value":              float(p_val),
        "flag":               bool(
                                  p_val < cfg.trend_p_value_threshold
                                  and max_slope >= cfg.slope_threshold
                                  and slope_change_ratio >= cfg.slope_change_ratio_threshold
                              ),
        "slope_train":        slope_train,
        "slope_recent":       slope_recent,
        "slope_delta":        slope_delta,
        "slope_change_ratio": slope_change_ratio,
    }


def test_stationarity(
    train: np.ndarray,
    recent: np.ndarray,
    cfg: MonitorConfig,
) -> dict:
    """
    Detect a transition from stationary to non-stationary behaviour (KPSS).

    STARS Family:       Stability
    Broken Assumption:  Not Stable — the series has lost its stationary structure
    Method:             KPSS test; flag when train stationary (p > kpss_alpha)
                        but recent non-stationary (p <= kpss_alpha)
    Threshold:          cfg.kpss_alpha (default 0.10)

    Args:
        train:  Daily volume values from the training window.
        recent: Daily volume values from the recent window.
        cfg:    MonitorConfig with hard-coded thresholds.

    Returns:
        value            KPSS statistic for the recent window
        flag             True if train stationary AND recent non-stationary
        kpss_p_train     KPSS p-value for the training window
        kpss_p_recent    KPSS p-value for the recent window
        train_stationary True if kpss_p_train > cfg.kpss_alpha
    """
    if len(train) < 30 or len(recent) < 30:
        return _nan_dict(kpss_p_train=float("nan"), kpss_p_recent=float("nan"),
                         train_stationary=None)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            _, p_train, _, _ = kpss(train, regression="c", nlags="auto")
            stat_recent, p_recent, _, _ = kpss(recent, regression="c", nlags="auto")
    except Exception:
        return _nan_dict(kpss_p_train=float("nan"), kpss_p_recent=float("nan"),
                         train_stationary=None)
    train_stationary = bool(p_train > cfg.kpss_alpha)
    recent_nonstationary = bool(p_recent <= cfg.kpss_alpha)
    return {
        "value":           float(stat_recent),
        "flag":            bool(train_stationary and recent_nonstationary),
        "kpss_p_train":    float(p_train),
        "kpss_p_recent":   float(p_recent),
        "train_stationary": train_stationary,
    }


# ── Truthfulness ──────────────────────────────────────────────────────────────


def test_coverage_shift(
    train_present: np.ndarray,
    recent_present: np.ndarray,
    cfg: MonitorConfig,
) -> dict:
    """
    Detect a significant shift in data coverage (proportion of observed days).

    STARS Family:       Truthfulness
    Broken Assumption:  Not Truthful — missing data pattern has changed
    Method:             Two-proportion z-test on non-missing day rates
    Threshold:          cfg.coverage_delta_threshold (default 0.40)

    Args:
        train_present:  Boolean array; True = day had an observed value in train.
        recent_present: Boolean array; True = day had an observed value in recent.
        cfg:            MonitorConfig with hard-coded thresholds.

    Returns:
        value            |coverage_recent - coverage_train|
        flag             True if p < alpha AND delta >= threshold
        p_value          Two-proportion z-test p-value
        coverage_train   Coverage rate in the training window
        coverage_recent  Coverage rate in the recent window
    """
    if len(train_present) == 0 or len(recent_present) == 0:
        return _nan_dict(p_value=float("nan"), coverage_train=float("nan"),
                         coverage_recent=float("nan"))
    cov_train = float(train_present.sum()) / len(train_present)
    cov_recent = float(recent_present.sum()) / len(recent_present)
    delta = abs(cov_recent - cov_train)
    count = np.array([int(train_present.sum()), int(recent_present.sum())])
    nobs = np.array([len(train_present), len(recent_present)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, p_value = proportions_ztest(count, nobs)
    return {
        "value":            float(delta),
        "flag":             bool(float(p_value) < cfg.alpha and delta >= cfg.coverage_delta_threshold),
        "p_value":          float(p_value),
        "coverage_train":   cov_train,
        "coverage_recent":  cov_recent,
    }


def test_sparsity_change(
    train_zero: np.ndarray,
    recent_zero: np.ndarray,
    cfg: MonitorConfig,
) -> dict:
    """
    Detect a significant shift in sparsity rate (proportion of zero-value days).

    STARS Family:       Truthfulness
    Broken Assumption:  Not Truthful — the zero-value pattern has changed
    Method:             Two-proportion z-test on zero-value day rates
    Threshold:          cfg.sparsity_delta_threshold (default 0.30)

    Args:
        train_zero:  Boolean array; True = day had a zero value in train window.
        recent_zero: Boolean array; True = day had a zero value in recent window.
        cfg:         MonitorConfig with hard-coded thresholds.

    Returns:
        value            |sparsity_recent - sparsity_train|
        flag             True if p < alpha AND delta >= threshold
        p_value          Two-proportion z-test p-value
        sparsity_train   Zero-rate in the training window
        sparsity_recent  Zero-rate in the recent window
    """
    if len(train_zero) == 0 or len(recent_zero) == 0:
        return _nan_dict(p_value=float("nan"), sparsity_train=float("nan"),
                         sparsity_recent=float("nan"))
    sparsity_train = float(train_zero.sum()) / len(train_zero)
    sparsity_recent = float(recent_zero.sum()) / len(recent_zero)
    delta = abs(sparsity_recent - sparsity_train)
    count = np.array([int(train_zero.sum()), int(recent_zero.sum())])
    nobs = np.array([len(train_zero), len(recent_zero)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, p_value = proportions_ztest(count, nobs)
    return {
        "value":            float(delta),
        "flag":             bool(float(p_value) < cfg.alpha and delta >= cfg.sparsity_delta_threshold),
        "p_value":          float(p_value),
        "sparsity_train":   sparsity_train,
        "sparsity_recent":  sparsity_recent,
    }


# ── Abundance ─────────────────────────────────────────────────────────────────


def test_low_volume(
    train: np.ndarray,
    train_dates: pd.DatetimeIndex,
    cfg: MonitorConfig,
) -> dict:
    """
    Detect insufficient signal volume in the training window.

    STARS Family:       Abundance
    Broken Assumption:  Not Abundant — too few observations to infer patterns
    Method:             Mean of per-calendar-month totals in the training window
    Threshold:          cfg.low_volume_monthly_threshold (default 3.0 per month)

    Args:
        train:       Daily volume values from the training window.
        train_dates: DatetimeIndex aligned with ``train`` (same length).
        cfg:         MonitorConfig with hard-coded thresholds.

    Returns:
        value              Average monthly volume
        flag               True if avg_monthly < threshold
        total_volume_train Total volume in the training window
        n_months_train     Number of calendar months in the training window
    """
    if len(train) == 0:
        return _nan_dict(True, total_volume_train=float("nan"), n_months_train=0)
    monthly = (
        pd.Series(train, index=train_dates)
        .groupby(train_dates.to_period("M"))
        .sum()
    )
    avg_monthly = float(monthly.mean())
    return {
        "value":             float(avg_monthly),
        "flag":              bool(avg_monthly < cfg.low_volume_monthly_threshold),
        "total_volume_train": float(train.sum()),
        "n_months_train":    int(len(monthly)),
    }


# ── Regularity ────────────────────────────────────────────────────────────────


def test_volatility_shift(
    train: np.ndarray,
    recent: np.ndarray,
    cfg: MonitorConfig,
) -> dict:
    """
    Detect a disproportionate shift in relative variability (CV ratio).

    STARS Family:       Regularity
    Broken Assumption:  Not Regular — the series variability has shifted materially
    Method:             cv_ratio = (sigma/mu)_recent / (sigma/mu)_train
    Threshold:          cfg.volatility_ratio_threshold (default 1.50); bidirectional

    Args:
        train:  Daily volume values from the training window.
        recent: Daily volume values from the recent window.
        cfg:    MonitorConfig with hard-coded thresholds.

    Returns:
        value       cv_ratio
        flag        True if cv_ratio >= threshold OR cv_ratio <= 1/threshold
        cv_train    Coefficient of variation for the training window
        cv_recent   Coefficient of variation for the recent window
        mean_train  Mean of training window
        mean_recent Mean of recent window
    """
    if len(train) < 10 or len(recent) < 10:
        return _nan_dict(cv_train=float("nan"), cv_recent=float("nan"),
                         mean_train=float("nan"), mean_recent=float("nan"))
    mean_train = float(train.mean())
    mean_recent = float(recent.mean())
    if abs(mean_train) < 1e-6 or abs(mean_recent) < 1e-6:
        return _nan_dict(cv_train=float("nan"), cv_recent=float("nan"),
                         mean_train=mean_train, mean_recent=mean_recent)
    cv_train = float(train.std(ddof=1) / abs(mean_train))
    cv_recent = float(recent.std(ddof=1) / abs(mean_recent))
    eps = 1e-9
    ratio = cv_recent / (cv_train + eps)
    thr = cfg.volatility_ratio_threshold
    return {
        "value":       float(ratio),
        "flag":        bool(ratio >= thr or ratio <= 1.0 / thr),
        "cv_train":    cv_train,
        "cv_recent":   cv_recent,
        "mean_train":  mean_train,
        "mean_recent": mean_recent,
    }


def test_outlier_rate(
    train: np.ndarray,
    recent: np.ndarray,
    cfg: MonitorConfig,
) -> dict:
    """
    Detect an excessive proportion of outlier points in the recent window.

    STARS Family:       Regularity
    Broken Assumption:  Not Regular — extreme values dominate the recent window
    Method:             Train-window MAD baseline; modified z-score on recent values
    Threshold:          cfg.outlier_z_threshold (default 3.50) for definition;
                        cfg.outlier_rate_threshold (default 0.40) for rate gate

    Args:
        train:  Daily volume values from the training window.
        recent: Daily volume values from the recent window.
        cfg:    MonitorConfig with hard-coded thresholds.

    Returns:
        value          outlier_count / len(recent)
        flag           True if rate >= cfg.outlier_rate_threshold
        outlier_count  Number of outlier points in the recent window
        train_median   Median of the training window (baseline)
        train_mad      MAD of the training window (baseline)
    """
    if len(train) < 3 or len(recent) == 0:
        return _nan_dict(outlier_count=0, train_median=float("nan"),
                         train_mad=float("nan"))
    train_median = float(np.median(train))
    train_mad = float(np.median(np.abs(train - train_median)))
    if train_mad == 0:
        train_std = float(train.std(ddof=1))
        if train_std == 0:
            outlier_mask = recent != train_median
            rate = float(outlier_mask.sum()) / len(recent)
            return {"value": float(rate), "flag": bool(rate >= cfg.outlier_rate_threshold),
                    "outlier_count": int(outlier_mask.sum()),
                    "train_median": train_median, "train_mad": train_mad}
        scale = train_std / 0.6745
    else:
        scale = train_mad / 0.6745
    modified_z = np.abs(recent - train_median) / scale
    outlier_mask = modified_z > cfg.outlier_z_threshold
    outlier_count = int(outlier_mask.sum())
    rate = float(outlier_count) / len(recent)
    return {
        "value":         float(rate),
        "flag":          bool(rate >= cfg.outlier_rate_threshold),
        "outlier_count": outlier_count,
        "train_median":  train_median,
        "train_mad":     train_mad,
    }


_ACF_STRUCTURE_LAGS: tuple[int, ...] = (1, 7, 30)


def test_acf_structure(
    train: np.ndarray,
    recent: np.ndarray,
    cfg: MonitorConfig,
) -> dict:
    """
    Detect divergence in autocorrelation structure across lags 1, 7, and 30.

    STARS Family:       Regularity
    Broken Assumption:  Not Regular — temporal patterns the model relied on are gone
    Method:             For each lag in (1, 7, 30): if training ACF exceeds the
                        Bartlett bound (1.96/sqrt(n_train)), apply Fisher Z-transform
                        test to check divergence (p < cfg.acf_divergence_p_threshold).
    Threshold:          cfg.acf_divergence_p_threshold (default 0.05)

    Args:
        train:  Daily volume values from the training window.
        recent: Daily volume values from the recent window.
        cfg:    MonitorConfig with hard-coded thresholds.

    Returns:
        value               Minimum divergence p-value across tested lags (nan if none tested)
        flag                True if any lag diverged
        bartlett_bound      1.96 / sqrt(n_train)
        acf_train_lag1      ACF at lag 1 for training window
        acf_recent_lag1     ACF at lag 1 for recent window
        acf_p_lag1          Fisher Z p-value at lag 1 (nan if lag not significant in train)
        acf_train_lag7      ACF at lag 7 for training window
        acf_recent_lag7     ACF at lag 7 for recent window
        acf_p_lag7          Fisher Z p-value at lag 7 (nan if lag not significant in train)
        acf_train_lag30     ACF at lag 30 for training window
        acf_recent_lag30    ACF at lag 30 for recent window
        acf_p_lag30         Fisher Z p-value at lag 30 (nan if lag not significant in train)
    """
    nan_acf = {
        "bartlett_bound":  float("nan"),
        "acf_train_lag1":  float("nan"), "acf_recent_lag1":  float("nan"), "acf_p_lag1":  float("nan"),
        "acf_train_lag7":  float("nan"), "acf_recent_lag7":  float("nan"), "acf_p_lag7":  float("nan"),
        "acf_train_lag30": float("nan"), "acf_recent_lag30": float("nan"), "acf_p_lag30": float("nan"),
    }
    if len(train) < 10 or len(recent) < 10:
        return _nan_dict(**nan_acf)

    n_train, n_recent = len(train), len(recent)
    max_lag = max(_ACF_STRUCTURE_LAGS)
    train_nlags = min(max_lag, n_train - 1)
    recent_nlags = min(max_lag, n_recent - 1)

    try:
        acf_train_vals = acf(train, nlags=train_nlags, fft=True)
    except Exception:
        acf_train_vals = np.full(train_nlags + 1, np.nan)
    try:
        acf_recent_vals = acf(recent, nlags=recent_nlags, fft=True)
    except Exception:
        acf_recent_vals = np.full(recent_nlags + 1, np.nan)

    bartlett_bound = 1.96 / np.sqrt(n_train)
    min_p = float("nan")
    any_diverged = False

    result = {
        "value":           float("nan"),
        "flag":            False,
        "bartlett_bound":  float(bartlett_bound),
        "acf_train_lag1":  float("nan"), "acf_recent_lag1":  float("nan"), "acf_p_lag1":  float("nan"),
        "acf_train_lag7":  float("nan"), "acf_recent_lag7":  float("nan"), "acf_p_lag7":  float("nan"),
        "acf_train_lag30": float("nan"), "acf_recent_lag30": float("nan"), "acf_p_lag30": float("nan"),
    }

    for lag in _ACF_STRUCTURE_LAGS:
        r_train  = float(acf_train_vals[lag])  if lag <= train_nlags  and np.isfinite(acf_train_vals[lag])  else float("nan")
        r_recent = float(acf_recent_vals[lag]) if lag <= recent_nlags and np.isfinite(acf_recent_vals[lag]) else float("nan")

        result[f"acf_train_lag{lag}"]  = r_train
        result[f"acf_recent_lag{lag}"] = r_recent

        if np.isnan(r_train) or np.isnan(r_recent):
            continue
        if abs(r_train) <= bartlett_bound:
            continue

        p = _fisher_z_test(r_train, n_train, r_recent, n_recent)
        result[f"acf_p_lag{lag}"] = float(p)
        if np.isnan(min_p) or p < min_p:
            min_p = p
        if p < cfg.acf_divergence_p_threshold:
            any_diverged = True

    result["value"] = float(min_p)
    result["flag"]  = any_diverged
    return result
