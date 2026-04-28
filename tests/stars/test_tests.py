# tests/stars/test_tests.py
import numpy as np
import pandas as pd
import pytest
from stars_pipeline.config import MonitorConfig
import stars_pipeline.stars.tests as _st

CFG = MonitorConfig()
RNG = np.random.default_rng(42)

# ── Shared fixtures ────────────────────────────────────────────────────────────
TRAIN_STABLE   = RNG.normal(100, 10, 365).clip(0)   # mean=100, std=10, 1 year
RECENT_STABLE  = RNG.normal(100, 10, 90).clip(0)    # same distribution
RECENT_SHIFTED = RNG.normal(250, 10, 90).clip(0)    # mean shifted to 250
RECENT_VOLATILE = RNG.normal(100, 80, 90).clip(0)   # much higher variance


# ── test_ks_distribution ───────────────────────────────────────────────────────
def test_ks_flags_when_distributions_clearly_differ():
    stat, flag = _st.test_ks_distribution(TRAIN_STABLE, RECENT_SHIFTED, CFG)
    assert flag is True
    assert stat >= CFG.ks_d_threshold

def test_ks_no_flag_for_same_distribution():
    stat, flag = _st.test_ks_distribution(TRAIN_STABLE, RECENT_STABLE, CFG)
    assert flag is False

def test_ks_returns_nan_flag_false_for_short_input():
    # min window is 10 — single-element arrays are too short
    stat, flag = _st.test_ks_distribution(np.array([1.0]), np.array([2.0]), CFG)
    assert np.isnan(stat)
    assert flag is False

def test_ks_returns_nan_for_nine_element_arrays():
    stat, flag = _st.test_ks_distribution(np.arange(9, dtype=float), np.arange(9, dtype=float), CFG)
    assert np.isnan(stat)
    assert flag is False


# ── test_level_shift ───────────────────────────────────────────────────────────
def test_level_shift_flags_large_mean_change():
    _, flag = _st.test_level_shift(TRAIN_STABLE, RECENT_SHIFTED, CFG)
    assert flag is True

def test_level_shift_no_flag_for_same_mean():
    _, flag = _st.test_level_shift(TRAIN_STABLE, RECENT_STABLE, CFG)
    assert flag is False

def test_level_shift_returns_cohens_d():
    value, _ = _st.test_level_shift(TRAIN_STABLE, RECENT_SHIFTED, CFG)
    assert value > 0


# ── test_dw_shift ──────────────────────────────────────────────────────────────
def test_dw_shift_flags_autocorrelation_change():
    # Highly autocorrelated recent vs. random train
    autocorr_recent = np.cumsum(RNG.normal(0, 1, 90))
    _, flag = _st.test_dw_shift(TRAIN_STABLE, autocorr_recent, CFG)
    assert flag is True

def test_dw_shift_no_flag_for_stable_series():
    _, flag = _st.test_dw_shift(TRAIN_STABLE, RECENT_STABLE, CFG)
    assert flag is False


# ── test_trend_change ──────────────────────────────────────────────────────────
def test_trend_change_flags_large_slope_shift():
    # flat train, strongly upward recent → slope_delta is large relative to train slope
    flat_train   = np.linspace(100.0, 100.5, 365)
    steep_recent = np.arange(90, dtype=float) * 5.0 + 100.0
    _, flag = _st.test_trend_change(flat_train, steep_recent, CFG)
    assert flag is True

def test_trend_change_no_flag_for_similar_slopes():
    # Both windows on same linear trajectory → interaction coefficient ≈ 0
    rng = np.random.default_rng(7)
    t = np.arange(455, dtype=float)
    series = t * 0.5 + rng.normal(0, 0.1, 455)
    _, flag = _st.test_trend_change(series[:365], series[365:], CFG)
    assert flag is False

def test_trend_change_returns_p_value():
    flat_train   = np.linspace(100.0, 100.5, 365)
    steep_recent = np.arange(90, dtype=float) * 5.0 + 100.0
    p_val, _ = _st.test_trend_change(flat_train, steep_recent, CFG)
    assert 0.0 <= p_val <= 1.0

def test_trend_change_returns_nan_for_short_input():
    stat, flag = _st.test_trend_change(np.arange(5, dtype=float), np.arange(5, dtype=float), CFG)
    assert np.isnan(stat)
    assert flag is False


# ── test_stationarity ──────────────────────────────────────────────────────────
def test_stationarity_flags_transition_to_nonstationary():
    # min window is 30 — use arrays large enough to pass the guard
    stationary_train  = RNG.normal(0, 1, 90)
    nonstationary_recent = np.cumsum(RNG.normal(0, 1, 90))
    _, flag = _st.test_stationarity(stationary_train, nonstationary_recent, CFG)
    assert flag is True

def test_stationarity_no_flag_when_both_stationary():
    # White noise around a fixed mean — KPSS will not flag either window, so
    # the condition (train_stationary AND recent_non_stationary) cannot be True.
    # Use a tight kpss_alpha to reduce the chance of a false positive.
    cfg_tight = MonitorConfig(kpss_alpha=0.01)
    rng2 = np.random.default_rng(1234)
    train_wn  = rng2.normal(0, 1, 90)
    recent_wn = rng2.normal(0, 1, 90)
    _, flag = _st.test_stationarity(train_wn, recent_wn, cfg_tight)
    assert flag is False

def test_stationarity_returns_nan_for_short_input():
    # arrays of length 3 are below the 30-observation minimum
    stat, flag = _st.test_stationarity(np.ones(3), np.ones(3), CFG)
    assert np.isnan(stat)
    assert flag is False

def test_stationarity_returns_nan_for_under_30():
    stat, flag = _st.test_stationarity(np.ones(29), np.ones(29), CFG)
    assert np.isnan(stat)
    assert flag is False




# ── test_coverage_shift ────────────────────────────────────────────────────────
def test_coverage_flags_large_drop_in_coverage():
    train_present  = np.ones(365, dtype=bool)        # 100% coverage in train
    recent_present = np.array([True] * 40 + [False] * 50)  # ~44% coverage
    _, flag = _st.test_coverage_shift(train_present, recent_present, CFG)
    assert flag is True

def test_coverage_no_flag_for_stable_coverage():
    train_present  = np.ones(365, dtype=bool)
    recent_present = np.ones(90, dtype=bool)
    _, flag = _st.test_coverage_shift(train_present, recent_present, CFG)
    assert flag is False

def test_coverage_returns_delta():
    train_present  = np.ones(365, dtype=bool)
    recent_present = np.zeros(90, dtype=bool)
    value, _ = _st.test_coverage_shift(train_present, recent_present, CFG)
    assert abs(value - 1.0) < 1e-6


# ── test_sparsity_change ───────────────────────────────────────────────────────
def test_sparsity_flags_large_increase_in_zeros():
    train_zero  = np.zeros(365, dtype=bool)           # no zeros in train
    recent_zero = np.array([True] * 60 + [False] * 30)  # 67% zeros recently
    _, flag = _st.test_sparsity_change(train_zero, recent_zero, CFG)
    assert flag is True

def test_sparsity_no_flag_for_stable_sparsity():
    train_zero  = np.zeros(365, dtype=bool)
    recent_zero = np.zeros(90, dtype=bool)
    _, flag = _st.test_sparsity_change(train_zero, recent_zero, CFG)
    assert flag is False


# ── test_low_volume ────────────────────────────────────────────────────────────
_LOW_VOL_DATES = pd.date_range("2023-01-01", periods=365, freq="D")

def test_low_volume_flags_below_threshold():
    # avg monthly ≈ 0.05 * 30 ≈ 1.5 → below threshold of 3
    low = np.full(365, 0.05)
    _, flag = _st.test_low_volume(low, _LOW_VOL_DATES, CFG)
    assert flag is True

def test_low_volume_no_flag_above_threshold():
    high = np.full(365, 10.0)
    _, flag = _st.test_low_volume(high, _LOW_VOL_DATES, CFG)
    assert flag is False

def test_low_volume_returns_avg_monthly():
    arr = np.full(365, 5.0)
    value, _ = _st.test_low_volume(arr, _LOW_VOL_DATES, CFG)
    assert value > 0


# ── test_volatility_shift ──────────────────────────────────────────────────────
def test_volatility_flags_increased_variability():
    # recent CV >> train CV → cv_ratio >= 1.50 → flag
    stable_train   = RNG.normal(100, 5, 365)        # CV ~0.05
    volatile_recent = RNG.normal(100, 50, 90)       # CV ~0.50 → ratio ~10 >> 1.50
    _, flag = _st.test_volatility_shift(stable_train, volatile_recent, CFG)
    assert flag is True

def test_volatility_flags_collapsed_variability():
    # recent CV << train CV → cv_ratio <= 1/1.50 ≈ 0.67 → flag
    volatile_train  = RNG.normal(100, 40, 365)       # CV ~0.40
    flat_recent     = np.full(90, 100.0) + RNG.normal(0, 0.1, 90)  # CV ≈ 0.001
    _, flag = _st.test_volatility_shift(volatile_train, flat_recent, CFG)
    assert flag is True

def test_volatility_no_flag_for_same_distribution():
    # same distribution → cv_ratio ≈ 1.0, well within (1/1.50, 1.50) → no flag
    same_train  = RNG.normal(100, 10, 365)
    same_recent = RNG.normal(100, 10, 90)
    _, flag = _st.test_volatility_shift(same_train, same_recent, CFG)
    assert flag is False

def test_volatility_returns_nan_for_zero_mean():
    zero_train = np.zeros(365)
    _, flag = _st.test_volatility_shift(zero_train, RECENT_STABLE, CFG)
    assert flag is False


# ── test_outlier_rate ──────────────────────────────────────────────────────────
def test_outlier_rate_flags_high_outlier_proportion():
    # Train is flat at 10; recent has 50% extreme values at 1000 → outliers
    train  = np.full(200, 10.0)
    recent = np.concatenate([np.full(50, 10.0), np.full(50, 1000.0)])
    _, flag = _st.test_outlier_rate(train, recent, CFG)
    assert flag is True

def test_outlier_rate_no_flag_for_clean_recent():
    # Recent is drawn from same distribution as train — no outliers
    train  = RNG.normal(100, 5, 365)
    recent = RNG.normal(100, 5, 90)
    _, flag = _st.test_outlier_rate(train, recent, CFG)
    assert flag is False

def test_outlier_rate_returns_rate():
    train  = np.full(200, 10.0)
    recent = np.concatenate([np.full(50, 10.0), np.full(50, 1000.0)])
    value, _ = _st.test_outlier_rate(train, recent, CFG)
    assert 0.0 <= value <= 1.0

def test_outlier_rate_returns_nan_for_short_train():
    _, flag = _st.test_outlier_rate(np.array([1.0, 2.0]), np.array([1.0, 2.0]), CFG)
    assert flag is False


# ── test_acf_structure ────────────────────────────────────────────────────────
def test_acf_structure_flags_when_acf_pattern_disappears():
    # Strongly autocorrelated train; white-noise recent — lag-1 ACF will diverge
    rng2 = np.random.default_rng(99)
    autocorr_train = np.cumsum(rng2.normal(0, 1, 200))
    white_recent   = rng2.normal(0, 1, 90)
    _, flag = _st.test_acf_structure(autocorr_train, white_recent, CFG)
    assert flag is True

def test_acf_structure_no_flag_for_similar_acf():
    # i.i.d. series — training ACF is near zero at all lags → no Bartlett-significant lags
    _, flag = _st.test_acf_structure(TRAIN_STABLE, RECENT_STABLE, CFG)
    assert flag is False

def test_acf_structure_returns_nan_for_short_input():
    stat, flag = _st.test_acf_structure(np.ones(3), np.ones(3), CFG)
    assert np.isnan(stat)
    assert flag is False
