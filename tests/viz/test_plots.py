import math
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.pyplot as plt
from pathlib import Path
from stars_pipeline.viz.plots import (
    plot_metric_distributions,
    plot_normal_breakdowns,
    plot_flag_correlation_grid,
    plot_flag_rates_by_dim,
    plot_severity_and_families,
    plot_threshold_proximity,
    plot_segment_series,
)


def _make_stats_df(n: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_flagged = n // 5
    rows = []
    for i in range(n):
        flagged = i < n_flagged
        rows.append({
            "strata_id": "84",
            "entity_id": f"E{i % 10:02d}",
            "patient_type_rollup": rng.choice(["Inpatient", "Outpatient", "Observation"]),
            "service_line": rng.choice(["Cardiology", "Orthopedics", "Neurology"]),
            "feature_segment": f"84|E{i % 10:02d}|pt|sl",
            "mesh": float(rng.uniform(0.02, 0.25)),
            "ks_distribution_value":  float(rng.uniform(0, 0.6)),
            "ks_distribution_flag":   int(flagged and rng.random() > 0.5),
            "level_shift_value":      float(rng.uniform(0, 3)),
            "level_shift_flag":       int(flagged and rng.random() > 0.5),
            "dw_shift_value":         float(rng.uniform(0, 2)),
            "dw_shift_flag":          0,
            "trend_change_value":     float(rng.uniform(0, 0.4)),
            "trend_change_flag":      0,
            "trend_change__slope_change_ratio": float(rng.uniform(0, 3)),
            "stationarity_value":     float(rng.uniform(0, 1)),
            "stationarity_flag":      0,
            "coverage_shift_value":   float(rng.uniform(0, 0.5)),
            "coverage_shift_flag":    0,
            "sparsity_change_value":  float(rng.uniform(0, 0.4)),
            "sparsity_change_flag":   0,
            "low_volume_value":       float(rng.uniform(0, 20)),
            "low_volume_flag":        0,
            "volatility_shift_value": float(rng.uniform(0.5, 5)),
            "volatility_shift_flag":  0,
            "outlier_rate_value":     float(rng.uniform(0, 0.5)),
            "outlier_rate_flag":      0,
            "acf_structure_value":    float(rng.uniform(0, 0.3)),
            "acf_structure_flag":     0,
            "is_flagged":             int(flagged),
            "stability_violations":   int(flagged) * rng.integers(1, 3),
            "truthfulness_violations": 0,
            "abundance_violations":   0,
            "regularity_violations":  0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def stats_df():
    return _make_stats_df()


def test_plot_metric_distributions_returns_figure(stats_df):
    fig = plot_metric_distributions(stats_df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_normal_breakdowns_returns_figure(stats_df):
    fig = plot_normal_breakdowns(stats_df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_flag_correlation_grid_returns_figure(stats_df):
    fig = plot_flag_correlation_grid(stats_df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_flag_rates_by_dim_returns_figure(stats_df):
    fig = plot_flag_rates_by_dim(stats_df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_severity_and_families_returns_figure(stats_df):
    fig = plot_severity_and_families(stats_df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_threshold_proximity_returns_figure(stats_df):
    fig = plot_threshold_proximity(stats_df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_segment_series_returns_figure():
    dates = pd.date_range("2024-01-01", periods=400, freq="D")
    series_df = pd.DataFrame({
        "date": dates,
        "actual": np.random.default_rng(0).normal(100, 10, 400).clip(0),
        "strata_id": "84", "entity_id": "E01",
        "patient_type_rollup": "Inpatient", "service_line": "Cardiology",
        "feature_segment": "84|E01|Inpatient|Cardiology",
    })
    fig = plot_segment_series(series_df, "84|E01|Inpatient|Cardiology")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
