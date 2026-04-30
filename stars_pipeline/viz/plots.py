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


def plot_normal_breakdowns(
    stats_df: pd.DataFrame,
    *,
    top_n_entities: int = 20,
    figsize: tuple[float, float] = (15, 13),
) -> plt.Figure:
    """
    Three-panel breakdown of Normal (not-flagged) rates.

    Panel 1 — by patient_type_rollup
    Panel 2 — by entity_id (top N by segment count)
    Panel 3 — heatmap: entity × patient_type (top N entities)
    """
    stats_df = stats_df.copy()
    stats_df["is_normal"] = ~stats_df["is_flagged"].astype(bool)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.4)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    def _color(v: float) -> str:
        return "#d62728" if v < 0.5 else "#2ca02c" if v >= 0.75 else "#ff7f0e"

    # Panel 1: by patient_type_rollup
    if "patient_type_rollup" in stats_df.columns:
        pt = (
            stats_df.groupby("patient_type_rollup")
            .agg(pct_normal=("is_normal", "mean"), n=("is_normal", "count"))
            .reset_index()
            .sort_values("pct_normal")
        )
        colors = [_color(v) for v in pt["pct_normal"]]
        ax1.barh(pt["patient_type_rollup"], pt["pct_normal"], color=colors,
                 edgecolor="white", linewidth=0.5)
        for _, row in pt.iterrows():
            ax1.text(row["pct_normal"] + 0.01, row["patient_type_rollup"],
                     f"{row['n']}", va="center", fontsize=7)
        ax1.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)
        ax1.set_xlim(0, 1.15)
        ax1.set_xlabel("% Normal")
        ax1.set_title("Normal Rate by Patient Type", fontsize=9)
        ax1.tick_params(labelsize=7)

    # Panel 2: by entity_id
    if "entity_id" in stats_df.columns:
        ent = (
            stats_df.groupby("entity_id")
            .agg(pct_normal=("is_normal", "mean"), n=("is_normal", "count"))
            .reset_index()
            .nlargest(top_n_entities, "n")
            .sort_values("pct_normal")
        )
        colors2 = [_color(v) for v in ent["pct_normal"]]
        ax2.barh(ent["entity_id"].astype(str), ent["pct_normal"], color=colors2,
                 edgecolor="white", linewidth=0.5)
        for _, row in ent.iterrows():
            ax2.text(row["pct_normal"] + 0.01, str(row["entity_id"]),
                     f"{row['n']}", va="center", fontsize=7)
        ax2.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)
        ax2.set_xlim(0, 1.15)
        ax2.set_xlabel("% Normal")
        ax2.set_title(f"Normal Rate by Entity (top {top_n_entities})", fontsize=9)
        ax2.tick_params(labelsize=7)

    # Panel 3: entity × patient_type heatmap
    if "entity_id" in stats_df.columns and "patient_type_rollup" in stats_df.columns:
        top_entities = stats_df.groupby("entity_id").size().nlargest(top_n_entities).index
        heat_df = (
            stats_df[stats_df["entity_id"].isin(top_entities)]
            .groupby(["entity_id", "patient_type_rollup"])["is_normal"]
            .mean()
            .unstack("patient_type_rollup")
        )
        if not heat_df.empty:
            nan_mask = heat_df.isna()
            cmap = sns.diverging_palette(10, 130, as_cmap=True)
            sns.heatmap(
                heat_df, ax=ax3, cmap=cmap, vmin=0, vmax=1,
                mask=nan_mask,
                annot=True, fmt=".0%", linewidths=0.4,
                annot_kws={"size": 7},
                cbar_kws={"label": "% Normal"},
            )
        ax3.set_title("Normal Rate — Entity × Patient Type", fontsize=9)
        ax3.set_xlabel("")
        ax3.tick_params(labelsize=7)

    fig.suptitle("STARS Normal Classification Breakdowns", fontsize=12, y=1.01)
    return fig

def plot_flag_correlation_grid(
    stats_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (13, 11),
) -> plt.Figure:
    """Lower-triangle Pearson/phi correlation heatmap of all STARS flag columns."""
    flag_cols = [c for c in stats_df.columns if c.endswith("_flag") and c != "is_flagged"]
    flag_cols += ["is_flagged"]
    present = [c for c in flag_cols if c in stats_df.columns]
    if len(present) < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No flag columns found", ha="center", va="center")
        return fig

    label_map = {
        "ks_distribution_flag":  "KS Dist.",
        "level_shift_flag":      "Level Shift",
        "dw_shift_flag":         "DW Shift",
        "trend_change_flag":     "Trend",
        "stationarity_flag":     "Stationarity",
        "coverage_shift_flag":   "Coverage",
        "sparsity_change_flag":  "Sparsity",
        "low_volume_flag":       "Low Volume",
        "volatility_shift_flag": "Volatility",
        "outlier_rate_flag":     "Outlier Rate",
        "acf_structure_flag":    "ACF Structure",
        "is_flagged":            "Is Flagged",
    }

    mat = stats_df[present].astype(float).corr()
    mat.index   = [label_map.get(c, c) for c in mat.index]
    mat.columns = [label_map.get(c, c) for c in mat.columns]

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
    sns.heatmap(
        mat, ax=ax, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", linewidths=0.3, annot_kws={"size": 7},
        cbar_kws={"label": "Pearson / phi correlation"}, square=True,
    )
    ax.set_title("Flag Co-occurrence Correlation Grid", fontsize=11)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def plot_flag_rates_by_dim(
    stats_df: pd.DataFrame,
    *,
    dims: list[str] | None = None,
    top_n: int = 15,
    figsize: tuple[float, float] = (15, 5),
) -> plt.Figure:
    """Stacked horizontal bar showing STARS family violation rates per grouping dimension."""
    if dims is None:
        dims = [d for d in ["strata_id", "patient_type_rollup", "service_line"]
                if d in stats_df.columns]

    df = stats_df.copy()
    def _viol_flag(col: str) -> pd.Series:
        if col in df.columns:
            return (df[col] > 0).astype(float)
        return pd.Series(0.0, index=df.index)

    df["family_stable"]   = _viol_flag("stability_violations")
    df["family_truthful"] = _viol_flag("truthfulness_violations")
    df["family_abundant"] = _viol_flag("abundance_violations")
    df["family_regular"]  = _viol_flag("regularity_violations")
    if "is_flagged" not in df.columns:
        df["is_flagged"] = False
    df["is_normal"] = ~df["is_flagged"].astype(bool)

    family_cols = ["family_stable", "family_truthful", "family_abundant", "family_regular"]
    family_labels = {
        "family_stable":   "Stable",
        "family_truthful": "Truthful",
        "family_abundant": "Abundant",
        "family_regular":  "Regular",
    }
    family_colors = {
        "family_stable":   "#1f77b4",
        "family_truthful": "#ff7f0e",
        "family_abundant": "#9467bd",
        "family_regular":  "#d62728",
    }

    n_dims = len(dims)
    if n_dims == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No grouping dimensions found", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(1, n_dims, figsize=figsize, squeeze=False)

    for ax, dim in zip(axes[0], dims):
        grp_cols = [c for c in family_cols + ["is_normal"] if c in df.columns]
        grp = df.groupby(dim)[grp_cols].mean().reset_index()
        grp = grp.sort_values("is_normal").head(top_n)
        y_labels = grp[dim].astype(str).tolist()
        lefts = np.zeros(len(grp))

        for fcol in family_cols:
            if fcol not in grp.columns:
                continue
            vals = grp[fcol].fillna(0).values
            ax.barh(y_labels, vals, left=lefts,
                    color=family_colors.get(fcol, "grey"),
                    label=family_labels.get(fcol, fcol), alpha=0.85)
            lefts += vals

        ax.set_xlabel("Violation Rate")
        ax.set_title(f"STARS Family Rates by {dim}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, max(lefts.max() * 1.05, 0.05))
        if dim == dims[0]:
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("STARS Family Violation Rates by Dimension", fontsize=11, y=1.01)
    fig.tight_layout()
    return fig

def plot_severity_and_families(stats_df: pd.DataFrame, **kwargs) -> plt.Figure:
    raise NotImplementedError

def plot_threshold_proximity(stats_df: pd.DataFrame, cfg=None, **kwargs) -> plt.Figure:
    raise NotImplementedError

def plot_segment_series(series_df: pd.DataFrame, feature_segment: str, **kwargs) -> plt.Figure:
    raise NotImplementedError
