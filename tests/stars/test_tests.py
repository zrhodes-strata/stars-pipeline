# tests/stars/test_tests.py
import numpy as np
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
    stat, flag = _st.test_ks_distribution(np.array([1.0]), np.array([2.0]), CFG)
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


# ── test_slope_change ──────────────────────────────────────────────────────────
def test_slope_change_flags_large_delta():
    # train has near-zero slope; recent has slope=10 → large |delta|/|baseline|
    flat_train   = np.linspace(1.0, 1.001, 365)       # slope ~2.7e-6 per step
    steep_recent = np.arange(90, dtype=float) * 10    # slope = 10
    _, flag = _st.test_slope_change(flat_train, steep_recent, CFG)
    assert flag is True

def test_slope_change_no_flag_for_similar_slopes():
    # Both windows have slope ≈ 1: delta ≈ 0, ratio ≈ 0
    linear        = np.arange(365, dtype=float)
    linear_recent = np.arange(90, dtype=float)
    _, flag = _st.test_slope_change(linear, linear_recent, CFG)
    assert flag is False

def test_slope_change_returns_ratio():
    flat_train   = np.linspace(1.0, 1.001, 365)
    steep_recent = np.arange(90, dtype=float) * 10
    ratio, _ = _st.test_slope_change(flat_train, steep_recent, CFG)
    assert ratio > 0


# ── test_stationarity ──────────────────────────────────────────────────────────
def test_stationarity_flags_transition_to_nonstationary():
    stationary_train  = RNG.normal(0, 1, 60)
    nonstationary_recent = np.cumsum(RNG.normal(0, 1, 40))
    _, flag = _st.test_stationarity(stationary_train, nonstationary_recent, CFG)
    assert flag is True

def test_stationarity_no_flag_when_both_stationary():
    stationary_train  = RNG.normal(0, 1, 60)
    stationary_recent = RNG.normal(0, 1, 40)
    _, flag = _st.test_stationarity(stationary_train, stationary_recent, CFG)
    assert flag is False

def test_stationarity_returns_nan_for_short_input():
    stat, flag = _st.test_stationarity(np.ones(3), np.ones(3), CFG)
    assert np.isnan(stat)
    assert flag is False


# ── test_trend_significance ────────────────────────────────────────────────────
def test_trend_significance_flags_strong_trend():
    # Deterministic ramp — p-value is effectively 0, well below 0.20
    strong_trend = np.arange(90, dtype=float) + RNG.normal(0, 0.1, 90)
    _, flag = _st.test_trend_significance(TRAIN_STABLE, strong_trend, CFG)
    assert flag is True

def test_trend_significance_no_flag_for_exactly_flat_series():
    # Constant series has no trend — p-value = 1.0, well above 0.20
    perfectly_flat = np.full(90, 100.0)
    _, flag = _st.test_trend_significance(TRAIN_STABLE, perfectly_flat, CFG)
    assert flag is False


# ── Task 7 additional imports ──────────────────────────────────────────────────
# (import is already done via `import stars_pipeline.stars.tests as _st`)


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
def test_low_volume_flags_below_threshold():
    # avg monthly = 0.05/day * 30.44 ≈ 1.52 → below threshold of 3
    low = np.full(365, 0.05)
    _, flag = _st.test_low_volume(low, CFG)
    assert flag is True

def test_low_volume_no_flag_above_threshold():
    high = np.full(365, 10.0)
    _, flag = _st.test_low_volume(high, CFG)
    assert flag is False

def test_low_volume_returns_avg_monthly():
    arr = np.full(365, 5.0)
    value, _ = _st.test_low_volume(arr, CFG)
    assert value > 0


# ── test_volatility_shift ──────────────────────────────────────────────────────
def test_volatility_flags_cv_above_threshold():
    # With threshold=0.10, any recent CV >= 10% of train CV flags.
    # A ratio of ~1.0 (same distribution) is well above 0.10.
    stable_train   = RNG.normal(100, 5, 365)         # CV ~0.05
    similar_recent = RNG.normal(100, 5, 90)          # CV ~0.05 → ratio ~1.0 >= 0.10
    _, flag = _st.test_volatility_shift(stable_train, similar_recent, CFG)
    assert flag is True

def test_volatility_no_flag_for_nearly_constant_recent():
    # Recent with tiny std → CV_recent << threshold * CV_train → no flag
    stable_train       = RNG.normal(100, 30, 365)    # CV ~0.30
    nearly_flat_recent = np.full(90, 100.0) + RNG.normal(0, 0.0001, 90)  # CV ≈ 0
    _, flag = _st.test_volatility_shift(stable_train, nearly_flat_recent, CFG)
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


# ── test_acf_divergence ────────────────────────────────────────────────────────
def test_acf_divergence_flags_when_acf_changes_significantly():
    # train: autocorrelated; recent: white noise
    autocorr = np.cumsum(RNG.normal(0, 1, 200))
    white_noise = RNG.normal(0, 1, 90)
    _, flag = _st.test_acf_divergence(autocorr, white_noise, CFG)
    assert flag is True

def test_acf_divergence_no_flag_for_similar_acf():
    _, flag = _st.test_acf_divergence(TRAIN_STABLE, RECENT_STABLE, CFG)
    assert flag is False

def test_acf_divergence_returns_nan_for_short_input():
    stat, flag = _st.test_acf_divergence(np.ones(3), np.ones(3), CFG)
    assert np.isnan(stat)
    assert flag is False


# ── test_dow_pattern_shift ─────────────────────────────────────────────────────
def test_dow_pattern_no_flag_for_stable_series():
    # Both windows have no strong DOW pattern — ACF lag-7 ~ 0
    _, flag = _st.test_dow_pattern_shift(TRAIN_STABLE, RECENT_STABLE, CFG)
    assert flag is False

def test_dow_pattern_returns_nan_for_short_input():
    stat, flag = _st.test_dow_pattern_shift(np.ones(5), np.ones(5), CFG)
    assert np.isnan(stat)
    assert flag is False
