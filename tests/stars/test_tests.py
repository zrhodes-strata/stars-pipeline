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
def test_slope_change_flags_large_ratio():
    # linspace gives a non-zero baseline slope; steep_recent has slope=10
    flat_train   = np.linspace(1.0, 1.001, 365)       # slope ~2.7e-6 per step
    steep_recent = np.arange(90, dtype=float) * 10    # slope = 10
    _, flag = _st.test_slope_change(flat_train, steep_recent, CFG)
    assert flag is True

def test_slope_change_no_flag_for_similar_slopes():
    linear = np.arange(365, dtype=float)
    linear_recent = np.arange(90, dtype=float)
    _, flag = _st.test_slope_change(linear, linear_recent, CFG)
    assert flag is False


# ── test_stationarity ──────────────────────────────────────────────────────────
def test_stationarity_flags_transition_to_nonstationary():
    stationary_train  = RNG.normal(0, 1, 60)
    nonstationary_recent = np.cumsum(RNG.normal(0, 1, 40))
    _, flag = _st.test_stationarity(stationary_train, nonstationary_recent, CFG)
    assert flag is True

def test_stationarity_returns_nan_for_short_input():
    stat, flag = _st.test_stationarity(np.ones(3), np.ones(3), CFG)
    assert np.isnan(stat)
    assert flag is False


# ── test_trend_significance ────────────────────────────────────────────────────
def test_trend_significance_flags_strong_trend():
    strong_trend = np.arange(90, dtype=float) + RNG.normal(0, 0.1, 90)
    _, flag = _st.test_trend_significance(TRAIN_STABLE, strong_trend, CFG)
    assert flag is True

def test_trend_significance_no_flag_for_flat_series():
    flat = RNG.normal(100, 5, 90)
    _, flag = _st.test_trend_significance(TRAIN_STABLE, flat, CFG)
    assert flag is False
