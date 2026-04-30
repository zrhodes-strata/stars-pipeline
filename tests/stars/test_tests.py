# tests/stars/test_tests.py
import numpy as np
import pandas as pd
import pytest
from stars_pipeline.config import MonitorConfig
import stars_pipeline.stars.tests as _st

CFG = MonitorConfig()
RNG = np.random.default_rng(42)

# ── Shared fixtures ────────────────────────────────────────────────────────────
TRAIN_STABLE    = RNG.normal(100, 10, 365).clip(0)
RECENT_STABLE   = RNG.normal(100, 10, 90).clip(0)
RECENT_SHIFTED  = RNG.normal(250, 10, 90).clip(0)
RECENT_VOLATILE = RNG.normal(100, 80, 90).clip(0)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _val(r): return r["value"]
def _flag(r): return r["flag"]


# ── test_ks_distribution ───────────────────────────────────────────────────────
def test_ks_flags_when_distributions_clearly_differ():
    r = _st.test_ks_distribution(TRAIN_STABLE, RECENT_SHIFTED, CFG)
    assert _flag(r) is True
    assert _val(r) >= CFG.ks_d_threshold
    assert "ks_p_value" in r
    assert "mean_train" in r
    assert "mean_recent" in r

def test_ks_no_flag_for_same_distribution():
    assert _flag(_st.test_ks_distribution(TRAIN_STABLE, RECENT_STABLE, CFG)) is False

def test_ks_returns_nan_flag_false_for_short_input():
    r = _st.test_ks_distribution(np.array([1.0]), np.array([2.0]), CFG)
    assert np.isnan(_val(r))
    assert _flag(r) is False

def test_ks_returns_nan_for_nine_element_arrays():
    r = _st.test_ks_distribution(np.arange(9, dtype=float), np.arange(9, dtype=float), CFG)
    assert np.isnan(_val(r))
    assert _flag(r) is False


# ── test_level_shift ───────────────────────────────────────────────────────────
def test_level_shift_flags_large_mean_change():
    r = _st.test_level_shift(TRAIN_STABLE, RECENT_SHIFTED, CFG)
    assert _flag(r) is True
    assert "p_value" in r
    assert "mean_train" in r
    assert "mean_recent" in r

def test_level_shift_no_flag_for_same_mean():
    assert _flag(_st.test_level_shift(TRAIN_STABLE, RECENT_STABLE, CFG)) is False

def test_level_shift_returns_cohens_d():
    assert _val(_st.test_level_shift(TRAIN_STABLE, RECENT_SHIFTED, CFG)) > 0


# ── test_dw_shift ──────────────────────────────────────────────────────────────
def test_dw_shift_flags_autocorrelation_change():
    autocorr_recent = np.cumsum(RNG.normal(0, 1, 90))
    r = _st.test_dw_shift(TRAIN_STABLE, autocorr_recent, CFG)
    assert _flag(r) is True
    assert "dw_train" in r
    assert "dw_recent" in r

def test_dw_shift_no_flag_for_stable_series():
    assert _flag(_st.test_dw_shift(TRAIN_STABLE, RECENT_STABLE, CFG)) is False


# ── test_trend_change ──────────────────────────────────────────────────────────
def test_trend_change_flags_large_slope_shift():
    flat_train   = np.linspace(100.0, 100.5, 365)
    steep_recent = np.arange(90, dtype=float) * 5.0 + 100.0
    r = _st.test_trend_change(flat_train, steep_recent, CFG)
    assert _flag(r) is True
    assert "slope_train" in r
    assert "slope_recent" in r
    assert "slope_delta" in r
    assert "slope_change_ratio" in r

def test_trend_change_no_flag_for_similar_slopes():
    rng = np.random.default_rng(7)
    t = np.arange(455, dtype=float)
    series = t * 0.5 + rng.normal(0, 0.1, 455)
    assert _flag(_st.test_trend_change(series[:365], series[365:], CFG)) is False

def test_trend_change_returns_p_value():
    flat_train   = np.linspace(100.0, 100.5, 365)
    steep_recent = np.arange(90, dtype=float) * 5.0 + 100.0
    p_val = _val(_st.test_trend_change(flat_train, steep_recent, CFG))
    assert 0.0 <= p_val <= 1.0

def test_trend_change_returns_nan_for_short_input():
    r = _st.test_trend_change(np.arange(5, dtype=float), np.arange(5, dtype=float), CFG)
    assert np.isnan(_val(r))
    assert _flag(r) is False


# ── test_stationarity ──────────────────────────────────────────────────────────
def test_stationarity_flags_transition_to_nonstationary():
    cfg_flag = MonitorConfig(kpss_alpha=0.05)
    t = np.arange(90, dtype=float)
    stationary_train = 100.0 + np.sin(2 * np.pi * t / 7)
    nonstationary_recent = np.cumsum(np.ones(90) * 2.0)
    r = _st.test_stationarity(stationary_train, nonstationary_recent, cfg_flag)
    assert _flag(r) is True
    assert "kpss_p_train" in r
    assert "kpss_p_recent" in r
    assert "train_stationary" in r

def test_stationarity_no_flag_when_both_stationary():
    cfg_tight = MonitorConfig(kpss_alpha=0.01)
    rng2 = np.random.default_rng(1234)
    assert _flag(_st.test_stationarity(rng2.normal(0, 1, 90), rng2.normal(0, 1, 90), cfg_tight)) is False

def test_stationarity_returns_nan_for_short_input():
    r = _st.test_stationarity(np.ones(3), np.ones(3), CFG)
    assert np.isnan(_val(r))
    assert _flag(r) is False

def test_stationarity_returns_nan_for_under_30():
    r = _st.test_stationarity(np.ones(29), np.ones(29), CFG)
    assert np.isnan(_val(r))
    assert _flag(r) is False


# ── test_coverage_shift ────────────────────────────────────────────────────────
def test_coverage_flags_large_drop_in_coverage():
    train_present  = np.ones(365, dtype=bool)
    recent_present = np.array([True] * 40 + [False] * 50)
    r = _st.test_coverage_shift(train_present, recent_present, CFG)
    assert _flag(r) is True
    assert "coverage_train" in r
    assert "coverage_recent" in r
    assert "p_value" in r

def test_coverage_no_flag_for_stable_coverage():
    assert _flag(_st.test_coverage_shift(np.ones(365, dtype=bool), np.ones(90, dtype=bool), CFG)) is False

def test_coverage_returns_delta():
    r = _st.test_coverage_shift(np.ones(365, dtype=bool), np.zeros(90, dtype=bool), CFG)
    assert abs(_val(r) - 1.0) < 1e-6


# ── test_sparsity_change ───────────────────────────────────────────────────────
def test_sparsity_flags_large_increase_in_zeros():
    train_zero  = np.zeros(365, dtype=bool)
    recent_zero = np.array([True] * 60 + [False] * 30)
    r = _st.test_sparsity_change(train_zero, recent_zero, CFG)
    assert _flag(r) is True
    assert "sparsity_train" in r
    assert "sparsity_recent" in r

def test_sparsity_no_flag_for_stable_sparsity():
    assert _flag(_st.test_sparsity_change(np.zeros(365, dtype=bool), np.zeros(90, dtype=bool), CFG)) is False


# ── test_low_volume ────────────────────────────────────────────────────────────
_LOW_VOL_DATES = pd.date_range("2023-01-01", periods=365, freq="D")

def test_low_volume_flags_below_threshold():
    assert _flag(_st.test_low_volume(np.full(365, 0.05), _LOW_VOL_DATES, CFG)) is True

def test_low_volume_no_flag_above_threshold():
    assert _flag(_st.test_low_volume(np.full(365, 10.0), _LOW_VOL_DATES, CFG)) is False

def test_low_volume_returns_avg_monthly():
    r = _st.test_low_volume(np.full(365, 5.0), _LOW_VOL_DATES, CFG)
    assert _val(r) > 0
    assert "total_volume_train" in r
    assert "n_months_train" in r


# ── test_volatility_shift ──────────────────────────────────────────────────────
def test_volatility_flags_increased_variability():
    r = _st.test_volatility_shift(RNG.normal(100, 5, 365), RNG.normal(100, 50, 90), CFG)
    assert _flag(r) is True
    assert "cv_train" in r
    assert "cv_recent" in r

def test_volatility_flags_collapsed_variability():
    volatile_train = RNG.normal(100, 40, 365)
    flat_recent    = np.full(90, 100.0) + RNG.normal(0, 0.1, 90)
    assert _flag(_st.test_volatility_shift(volatile_train, flat_recent, CFG)) is True

def test_volatility_no_flag_for_same_distribution():
    assert _flag(_st.test_volatility_shift(RNG.normal(100, 10, 365), RNG.normal(100, 10, 90), CFG)) is False

def test_volatility_returns_nan_for_zero_mean():
    assert _flag(_st.test_volatility_shift(np.zeros(365), RECENT_STABLE, CFG)) is False


# ── test_outlier_rate ──────────────────────────────────────────────────────────
def test_outlier_rate_flags_high_outlier_proportion():
    train  = np.full(200, 10.0)
    recent = np.concatenate([np.full(50, 10.0), np.full(50, 1000.0)])
    r = _st.test_outlier_rate(train, recent, CFG)
    assert _flag(r) is True
    assert "outlier_count" in r
    assert "train_median" in r
    assert "train_mad" in r

def test_outlier_rate_no_flag_for_clean_recent():
    assert _flag(_st.test_outlier_rate(RNG.normal(100, 5, 365), RNG.normal(100, 5, 90), CFG)) is False

def test_outlier_rate_returns_rate():
    train  = np.full(200, 10.0)
    recent = np.concatenate([np.full(50, 10.0), np.full(50, 1000.0)])
    assert 0.0 <= _val(_st.test_outlier_rate(train, recent, CFG)) <= 1.0

def test_outlier_rate_returns_nan_for_short_train():
    assert _flag(_st.test_outlier_rate(np.array([1.0, 2.0]), np.array([1.0, 2.0]), CFG)) is False


# ── test_acf_structure ────────────────────────────────────────────────────────
def test_acf_structure_flags_when_acf_pattern_disappears():
    rng2 = np.random.default_rng(99)
    autocorr_train = np.cumsum(rng2.normal(0, 1, 200))
    white_recent   = rng2.normal(0, 1, 90)
    r = _st.test_acf_structure(autocorr_train, white_recent, CFG)
    assert _flag(r) is True
    assert "bartlett_bound" in r
    assert "acf_train_lag1" in r
    assert "acf_recent_lag1" in r
    assert "acf_p_lag1" in r
    assert "acf_train_lag7" in r
    assert "acf_train_lag30" in r

def test_acf_structure_no_flag_for_similar_acf():
    assert _flag(_st.test_acf_structure(TRAIN_STABLE, RECENT_STABLE, CFG)) is False

def test_acf_structure_returns_nan_for_short_input():
    r = _st.test_acf_structure(np.ones(3), np.ones(3), CFG)
    assert np.isnan(_val(r))
    assert _flag(r) is False
