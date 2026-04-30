from __future__ import annotations

import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from stars_pipeline.config import MonitorConfig

# Maps new wide column name → (display label, upper clip for tail display, flag column)
_METRIC_DISPLAY: dict[str, tuple[str, float | None, str | None]] = {
    "ks_distribution_value":            ("KS Statistic",                None,  "ks_distribution_flag"),
    "level_shift_value":                ("Level Shift |Cohen's d|",     4.0,   "level_shift_flag"),
    "dw_shift_value":                   ("DW Shift |Δ|",                None,  "dw_shift_flag"),
    "trend_change__slope_change_ratio": ("Slope Change Ratio",          6.0,   "trend_change_flag"),
    "coverage_shift_value":             ("Coverage Δ",                  None,  "coverage_shift_flag"),
    "sparsity_change_value":            ("Sparsity Δ",                  None,  "sparsity_change_flag"),
    "low_volume_value":                 ("Avg Monthly Volume (train)",   300.0, "low_volume_flag"),
    "volatility_shift_value":           ("CV Ratio",                    12.0,  "volatility_shift_flag"),
    "outlier_rate_value":               ("Outlier Rate",                0.5,   "outlier_rate_flag"),
}


def plot_metric_distributions(
    stats_df: pd.DataFrame,
    *,
    thresholds: dict[str, float] | None = None,
    ncols: int = 3,
    bins: int = 40,
    figsize_per_panel: tuple[float, float] = (4.2, 3.0),
) -> plt.Figure:
    if thresholds is None:
        thresholds = {}

    normal_mask = ~stats_df["is_flagged"].astype(bool)
    metrics = [m for m in _METRIC_DISPLAY if m in stats_df.columns]
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
    )
    axes_flat = np.array(axes).flatten()

    for i, col in enumerate(metrics):
        ax = axes_flat[i]
        label, clip_hi, flag_col = _METRIC_DISPLAY[col]
        for mask, name, color in [
            (normal_mask,  "Normal",   "#2ca02c"),
            (~normal_mask, "Atypical", "#d62728"),
        ]:
            vals = stats_df.loc[mask, col].dropna().astype(float)
            if clip_hi is not None:
                vals = vals.clip(upper=clip_hi)
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=bins, color=color, alpha=0.45, density=True, label=name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    sns.kdeplot(vals, ax=ax, color=color, linewidth=1.4)
                except Exception:
                    pass
        if col in thresholds:
            ax.axvline(thresholds[col], color="black", linestyle="--", linewidth=1.0,
                       label=f"thr={thresholds[col]:.3g}")
        if flag_col and flag_col in stats_df.columns:
            flag_rate = stats_df[flag_col].astype(float).mean()
        else:
            flag_rate = (~normal_mask).mean()
        ax.set_title(f"{label}\n(flag rate {flag_rate:.1%})", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(len(metrics), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Metric Distributions — Normal vs Atypical", fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


def plot_normal_breakdowns(stats_df: pd.DataFrame, **kwargs) -> plt.Figure:
    raise NotImplementedError

def plot_flag_correlation_grid(stats_df: pd.DataFrame, **kwargs) -> plt.Figure:
    raise NotImplementedError

def plot_flag_rates_by_dim(stats_df: pd.DataFrame, **kwargs) -> plt.Figure:
    raise NotImplementedError

def plot_severity_and_families(stats_df: pd.DataFrame, **kwargs) -> plt.Figure:
    raise NotImplementedError

def plot_threshold_proximity(stats_df: pd.DataFrame, cfg=None, **kwargs) -> plt.Figure:
    raise NotImplementedError

def plot_segment_series(series_df: pd.DataFrame, feature_segment: str, **kwargs) -> plt.Figure:
    raise NotImplementedError
