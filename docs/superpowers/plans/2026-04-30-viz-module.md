# Visualization Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the seven STARS diagnostic plot functions from `Scratch and Tools/Predictions/performance_investigation.py` into a clean `stars_pipeline/viz/` module that accepts the long-format CSV or wide stats DataFrame as input and writes static PNG files to a caller-specified output directory.

**Architecture:** A single `stars_pipeline/viz/plots.py` module exposes one public function per chart type, all accepting a wide `stats_df` (output of `apply_thresholds()`) as their primary argument. A `stars_pipeline/viz/__init__.py` re-exports the public API. A separate `stars_pipeline/viz/cli.py` adds a `stars-viz` CLI entry point that reads the long-format CSV, pivots it to wide format, and runs all plots. No Streamlit — static PNG output only for now.

**Tech Stack:** matplotlib, seaborn, pandas, numpy (all already in scope or easily added); `matplotlib` and `seaborn` added as optional `viz` dependencies in `pyproject.toml`.

---

## Context for the implementer

The new `stars_pipeline` package already has:
- `stars_pipeline/stars/monitor.py` → `run_monitoring()` returns a wide DataFrame (one row per segment), `apply_thresholds()` adds `is_flagged`, `*_violations` columns
- `stars_pipeline/stars/output.py` → `to_long_format()` / `write_long_csv()` — long format has columns: `strata_id, entity_id, patient_type_rollup, service_line, feature_segment, stars_family, metric_name, metric_value, metric_flag`
- `stars_pipeline/config.py` → `MonitorConfig` (frozen dataclass with all canonical thresholds)

The plots are ported from `performance_investigation.py`. Key differences:
- Old code uses `is_normal` (True = good). New pipeline uses `is_flagged` (True = bad). All plots must invert: `normal_mask = ~stats_df["is_flagged"].astype(bool)`.
- Old metric column names differ from new ones. Mapping (old → new wide column):
  - `ks_stat` → `ks_distribution_value`
  - `coverage_delta` → `coverage_shift_value`
  - `outlier_rate` → `outlier_rate_value`
  - `trend_slope_change_ratio` → `trend_change__slope_change_ratio` (intermediate)
  - `volatility_ratio` → `volatility_shift_value`
  - `level_shift_cohen_d` → `level_shift_value`
  - `cv_ratio` → `volatility_shift_value` (same metric, different old name)
- `within_10` / `within_5` / `within_3` are prediction accuracy labels from the MESH column (`mesh <= threshold`). They must be passed into the plots that need them (MC plots). The core diagnostic plots only need `stats_df`.
- The `family_stable/truthful/abundant/regular` columns in the old code map to `stability_violations > 0`, etc. in new code.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `stars_pipeline/viz/__init__.py` | Create | Re-export public plot functions |
| `stars_pipeline/viz/plots.py` | Create | All 7 plot functions |
| `stars_pipeline/viz/cli.py` | Create | `stars-viz` CLI entry point |
| `stars_pipeline/viz/_wide.py` | Create | `long_to_wide()` helper: pivot long CSV back to wide stats format |
| `tests/viz/test_plots.py` | Create | Smoke tests: each function returns a `plt.Figure` without crashing |
| `tests/viz/test_wide.py` | Create | Unit tests for `long_to_wide()` |
| `pyproject.toml` | Modify | Add `matplotlib`, `seaborn` to `[project.optional-dependencies]` viz group; add `stars-viz` script entry point |

---

## Task 1: Add viz dependencies and package scaffold

**Files:**
- Modify: `pyproject.toml`
- Create: `stars_pipeline/viz/__init__.py`
- Create: `stars_pipeline/viz/plots.py` (stub)
- Create: `stars_pipeline/viz/cli.py` (stub)
- Create: `stars_pipeline/viz/_wide.py` (stub)
- Create: `tests/viz/__init__.py`

- [ ] **Step 1: Add optional `[viz]` dependencies to pyproject.toml**

In `pyproject.toml`, update `[project.optional-dependencies]` to add a `viz` group, and add the `stars-viz` entry point:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "boto3>=1.28.0",
]
viz = [
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
]

[project.scripts]
stars-pipeline = "stars_pipeline.cli:main"
stars-viz = "stars_pipeline.viz.cli:main"
```

- [ ] **Step 2: Install viz extras**

```bash
uv add --optional viz matplotlib seaborn
```

Expected: no errors, `matplotlib` and `seaborn` appear in `pyproject.toml` under `[project.optional-dependencies] viz`.

- [ ] **Step 3: Create package files**

Create `stars_pipeline/viz/__init__.py`:
```python
from stars_pipeline.viz.plots import (
    plot_metric_distributions,
    plot_normal_breakdowns,
    plot_flag_correlation_grid,
    plot_flag_rates_by_dim,
    plot_severity_and_families,
    plot_threshold_proximity,
    plot_segment_series,
)

__all__ = [
    "plot_metric_distributions",
    "plot_normal_breakdowns",
    "plot_flag_correlation_grid",
    "plot_flag_rates_by_dim",
    "plot_severity_and_families",
    "plot_threshold_proximity",
    "plot_segment_series",
]
```

Create `stars_pipeline/viz/plots.py` stub:
```python
from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd

def plot_metric_distributions(stats_df: pd.DataFrame, **kwargs) -> plt.Figure:
    raise NotImplementedError

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
```

Create `stars_pipeline/viz/_wide.py` stub:
```python
from __future__ import annotations
import pandas as pd

def long_to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError
```

Create `stars_pipeline/viz/cli.py` stub:
```python
from __future__ import annotations
import sys

def main(argv=None) -> int:
    raise NotImplementedError

if __name__ == "__main__":
    sys.exit(main())
```

Create `tests/viz/__init__.py` (empty file).

- [ ] **Step 4: Verify import works**

```bash
uv run python -c "from stars_pipeline.viz import plot_metric_distributions; print('ok')"
```

Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml stars_pipeline/viz/ tests/viz/__init__.py
git commit -m "feat: scaffold stars_pipeline/viz package with stub functions"
```

---

## Task 2: Implement `long_to_wide()` helper

The CLI needs to pivot the long-format CSV back into the wide stats format that all plot functions expect. This is distinct from `to_long_format()` — it goes the other direction.

**Files:**
- Modify: `stars_pipeline/viz/_wide.py`
- Create: `tests/viz/test_wide.py`

- [ ] **Step 1: Write failing tests**

Create `tests/viz/test_wide.py`:
```python
import pandas as pd
import pytest
from stars_pipeline.viz._wide import long_to_wide


def _make_long_df():
    rows = []
    seg = {
        "strata_id": "84", "entity_id": "E01",
        "patient_type_rollup": "Inpatient", "service_line": "Cardiology",
        "feature_segment": "84|E01|Inpatient|Cardiology",
    }
    # Primary rows
    for metric, family, value, flag in [
        ("ks_distribution",  "Stability",    "0.15", 0),
        ("level_shift",      "Stability",    "0.80", 0),
        ("dw_shift",         "Stability",    "0.30", 0),
        ("trend_change",     "Stability",    "0.18", 1),
        ("stationarity",     "Stability",    "0.05", 0),
        ("coverage_shift",   "Truthfulness", "0.02", 0),
        ("sparsity_change",  "Truthfulness", "0.01", 0),
        ("low_volume",       "Abundance",    "5.00", 0),
        ("volatility_shift", "Regularity",   "1.10", 0),
        ("outlier_rate",     "Regularity",   "0.05", 0),
        ("acf_structure",    "Regularity",   "0.20", 0),
    ]:
        rows.append({**seg, "stars_family": family, "metric_name": metric,
                     "metric_value": value, "metric_flag": flag})
    # Summary rows
    for metric, value, flag in [
        ("is_flagged", "1", 1),
        ("stability_violations", "1", 1),
        ("truthfulness_violations", "0", 0),
        ("abundance_violations", "0", 0),
        ("regularity_violations", "0", 0),
    ]:
        rows.append({**seg, "stars_family": "Summary", "metric_name": metric,
                     "metric_value": value, "metric_flag": flag})
    return pd.DataFrame(rows)


def test_long_to_wide_returns_one_row_per_segment():
    df = long_to_wide(_make_long_df())
    assert len(df) == 1


def test_long_to_wide_has_primary_value_and_flag_columns():
    df = long_to_wide(_make_long_df())
    for metric in ["ks_distribution", "level_shift", "trend_change",
                   "stationarity", "coverage_shift", "sparsity_change",
                   "low_volume", "volatility_shift", "outlier_rate", "acf_structure"]:
        assert f"{metric}_value" in df.columns, f"missing {metric}_value"
        assert f"{metric}_flag" in df.columns, f"missing {metric}_flag"


def test_long_to_wide_has_summary_columns():
    df = long_to_wide(_make_long_df())
    for col in ["is_flagged", "stability_violations", "truthfulness_violations",
                "abundance_violations", "regularity_violations"]:
        assert col in df.columns, f"missing {col}"


def test_long_to_wide_preserves_flag_values():
    df = long_to_wide(_make_long_df())
    assert int(df["trend_change_flag"].iloc[0]) == 1
    assert int(df["ks_distribution_flag"].iloc[0]) == 0
    assert int(df["is_flagged"].iloc[0]) == 1
    assert int(df["stability_violations"].iloc[0]) == 1
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/viz/test_wide.py -v
```

Expected: 4 failures with `NotImplementedError`.

- [ ] **Step 3: Implement `long_to_wide()`**

Replace the stub in `stars_pipeline/viz/_wide.py`:
```python
from __future__ import annotations
import pandas as pd

_ID_COLS = ["strata_id", "entity_id", "patient_type_rollup", "service_line", "feature_segment"]

_SUMMARY_METRICS = {
    "is_flagged", "stability_violations", "truthfulness_violations",
    "abundance_violations", "regularity_violations",
}


def long_to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot a long-format STARS results DataFrame back to wide stats format.

    Accepts the output of to_long_format() (or a CSV written by write_long_csv()).
    Returns one row per segment with {metric}_value, {metric}_flag, and summary
    columns — the same shape as the DataFrame produced by apply_thresholds().

    Intermediate rows (stars_family="Intermediate") are silently dropped; they
    are not needed by the plot functions.
    """
    id_cols = [c for c in _ID_COLS if c in long_df.columns]

    # Primary and summary rows only (drop intermediates)
    filtered = long_df[long_df["stars_family"] != "Intermediate"].copy()
    filtered["metric_value"] = pd.to_numeric(filtered["metric_value"], errors="coerce")
    filtered["metric_flag"] = pd.to_numeric(filtered["metric_flag"], errors="coerce")

    rows: list[dict] = []
    for seg_key, grp in filtered.groupby("feature_segment"):
        row: dict = {col: grp[col].iloc[0] for col in id_cols if col in grp.columns}
        for _, r in grp.iterrows():
            metric = r["metric_name"]
            if metric in _SUMMARY_METRICS:
                row[metric] = r["metric_flag"] if metric == "is_flagged" else r["metric_value"]
            else:
                row[f"{metric}_value"] = r["metric_value"]
                row[f"{metric}_flag"] = r["metric_flag"]
        rows.append(row)

    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/viz/test_wide.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add stars_pipeline/viz/_wide.py tests/viz/test_wide.py
git commit -m "feat: implement long_to_wide() pivot helper for viz module"
```

---

## Task 3: Implement `plot_metric_distributions`

Grid of histograms + KDE per metric, split by Normal vs Atypical (i.e., not-flagged vs flagged).

**Files:**
- Modify: `stars_pipeline/viz/plots.py`
- Modify: `tests/viz/test_plots.py`

- [ ] **Step 1: Write failing smoke test**

Create `tests/viz/test_plots.py`:
```python
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
```

- [ ] **Step 2: Run to verify failures**

```bash
uv run pytest tests/viz/test_plots.py -v
```

Expected: 7 failures, all `NotImplementedError`.

- [ ] **Step 3: Implement `plot_metric_distributions`**

Replace the stub for `plot_metric_distributions` in `stars_pipeline/viz/plots.py` with:

```python
from __future__ import annotations

import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from stars_pipeline.config import MonitorConfig

# Maps new wide column name → (display label, upper clip for tail display)
_METRIC_DISPLAY: dict[str, tuple[str, float | None]] = {
    "ks_distribution_value":          ("KS Statistic", None),
    "level_shift_value":              ("Level Shift |Cohen's d|", 4.0),
    "dw_shift_value":                 ("DW Shift |Δ|", None),
    "trend_change__slope_change_ratio": ("Slope Change Ratio", 6.0),
    "coverage_shift_value":           ("Coverage Δ", None),
    "sparsity_change_value":          ("Sparsity Δ", None),
    "low_volume_value":               ("Avg Monthly Volume (train)", 300.0),
    "volatility_shift_value":         ("CV Ratio", 12.0),
    "outlier_rate_value":             ("Outlier Rate", 0.5),
}


def plot_metric_distributions(
    stats_df: pd.DataFrame,
    *,
    thresholds: dict[str, float] | None = None,
    ncols: int = 3,
    bins: int = 40,
    figsize_per_panel: tuple[float, float] = (4.2, 3.0),
) -> plt.Figure:
    """
    Grid of metric distributions split by Normal (not flagged) vs Atypical (flagged).

    Args:
        stats_df:   Wide stats DataFrame from apply_thresholds() — one row per segment.
        thresholds: Optional {column_name: value} for vertical threshold lines.
        ncols:      Number of columns in the subplot grid.
        bins:       Histogram bin count.
        figsize_per_panel: (width, height) per subplot panel.

    Returns:
        matplotlib Figure.
    """
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
        label, clip_hi = _METRIC_DISPLAY[col]
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
```

- [ ] **Step 4: Run test for this function only**

```bash
uv run pytest tests/viz/test_plots.py::test_plot_metric_distributions_returns_figure -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add stars_pipeline/viz/plots.py tests/viz/test_plots.py
git commit -m "feat: implement plot_metric_distributions"
```

---

## Task 4: Implement `plot_normal_breakdowns`

Three-panel breakdown: by patient_type, by entity (top N), and entity × patient_type heatmap.

**Files:**
- Modify: `stars_pipeline/viz/plots.py`

- [ ] **Step 1: Replace stub for `plot_normal_breakdowns`**

```python
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

    Args:
        stats_df:        Wide stats DataFrame with is_flagged and dimension columns.
        top_n_entities:  How many entities to show in panel 2 and 3.
        figsize:         Overall figure size.

    Returns:
        matplotlib Figure.
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
            stats_df.groupby("patient_type_rollup")["is_normal"]
            .agg(pct_normal="mean", n="count")
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
            stats_df.groupby("entity_id")["is_normal"]
            .agg(pct_normal="mean", n="count")
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
        cmap = sns.diverging_palette(10, 130, as_cmap=True)
        sns.heatmap(
            heat_df, ax=ax3, cmap=cmap, vmin=0, vmax=1,
            annot=True, fmt=".0%", linewidths=0.4,
            annot_kws={"size": 7},
            cbar_kws={"label": "% Normal"},
        )
        ax3.set_title("Normal Rate — Entity × Patient Type", fontsize=9)
        ax3.set_xlabel("")
        ax3.tick_params(labelsize=7)

    fig.suptitle("STARS Normal Classification Breakdowns", fontsize=12, y=1.01)
    return fig
```

- [ ] **Step 2: Run test**

```bash
uv run pytest tests/viz/test_plots.py::test_plot_normal_breakdowns_returns_figure -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add stars_pipeline/viz/plots.py
git commit -m "feat: implement plot_normal_breakdowns"
```

---

## Task 5: Implement `plot_flag_correlation_grid` and `plot_flag_rates_by_dim`

**Files:**
- Modify: `stars_pipeline/viz/plots.py`

- [ ] **Step 1: Implement `plot_flag_correlation_grid`**

```python
def plot_flag_correlation_grid(
    stats_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (13, 11),
) -> plt.Figure:
    """
    Lower-triangle Pearson/phi correlation heatmap of all STARS flag columns.

    Args:
        stats_df: Wide stats DataFrame with *_flag columns.
        figsize:  Overall figure size.

    Returns:
        matplotlib Figure.
    """
    flag_cols = [c for c in stats_df.columns if c.endswith("_flag") and c != "is_flagged"]
    flag_cols += ["is_flagged"]
    present = [c for c in flag_cols if c in stats_df.columns]

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
        "is_flagged":            "Is Flagged ✓",
    }

    mat = stats_df[present].astype(float).corr()
    mat.index   = [label_map.get(c, c) for c in mat.index]
    mat.columns = [label_map.get(c, c) for c in mat.columns]

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
    sns.heatmap(
        mat, ax=ax, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", linewidths=0.3, annot_kws={"size": 7},
        cbar_kws={"label": "Pearson / ϕ correlation"}, square=True,
    )
    ax.set_title("Flag Co-occurrence Correlation Grid", fontsize=11)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig
```

- [ ] **Step 2: Implement `plot_flag_rates_by_dim`**

```python
def plot_flag_rates_by_dim(
    stats_df: pd.DataFrame,
    *,
    dims: list[str] | None = None,
    top_n: int = 15,
    figsize: tuple[float, float] = (15, 5),
) -> plt.Figure:
    """
    Stacked horizontal bar showing STARS family violation rates per grouping dimension.

    Args:
        stats_df: Wide stats DataFrame with *_violations columns and dimension columns.
        dims:     List of column names to group by. Defaults to strata_id,
                  patient_type_rollup, service_line.
        top_n:    Maximum values to show per dimension (by highest abnormal rate).
        figsize:  Overall figure size.

    Returns:
        matplotlib Figure.
    """
    if dims is None:
        dims = [d for d in ["strata_id", "patient_type_rollup", "service_line"]
                if d in stats_df.columns]

    df = stats_df.copy()
    df["family_stable"]    = (df.get("stability_violations",    0) > 0).astype(float)
    df["family_truthful"]  = (df.get("truthfulness_violations", 0) > 0).astype(float)
    df["family_abundant"]  = (df.get("abundance_violations",    0) > 0).astype(float)
    df["family_regular"]   = (df.get("regularity_violations",   0) > 0).astype(float)
    df["is_normal"]        = ~df["is_flagged"].astype(bool)

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
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/viz/test_plots.py::test_plot_flag_correlation_grid_returns_figure tests/viz/test_plots.py::test_plot_flag_rates_by_dim_returns_figure -v
```

Expected: 2 tests pass.

- [ ] **Step 4: Commit**

```bash
git add stars_pipeline/viz/plots.py
git commit -m "feat: implement plot_flag_correlation_grid and plot_flag_rates_by_dim"
```

---

## Task 6: Implement `plot_severity_and_families`, `plot_threshold_proximity`, `plot_segment_series`

**Files:**
- Modify: `stars_pipeline/viz/plots.py`

- [ ] **Step 1: Implement `plot_severity_and_families`**

```python
def plot_severity_and_families(
    stats_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (14, 5),
) -> plt.Figure:
    """
    Two-panel plot: total violation count distribution and family violation rates.

    Panel 1 — histogram of total violations per segment (0 through max).
    Panel 2 — stacked bar: fraction of segments with each family violated.

    Args:
        stats_df: Wide stats DataFrame with *_violations columns.
        figsize:  Overall figure size.

    Returns:
        matplotlib Figure.
    """
    df = stats_df.copy()
    viol_cols = [c for c in ["stability_violations", "truthfulness_violations",
                              "abundance_violations", "regularity_violations"]
                 if c in df.columns]
    df["total_violations"] = df[viol_cols].fillna(0).sum(axis=1)

    family_colors = {
        "stability_violations":    "#1f77b4",
        "truthfulness_violations": "#ff7f0e",
        "abundance_violations":    "#9467bd",
        "regularity_violations":   "#d62728",
    }
    family_labels = {
        "stability_violations":    "Stability",
        "truthfulness_violations": "Truthfulness",
        "abundance_violations":    "Abundance",
        "regularity_violations":   "Regularity",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: total violation distribution
    max_v = int(df["total_violations"].max()) if len(df) > 0 else 5
    bins = range(0, max_v + 2)
    ax1.hist(df["total_violations"], bins=bins, align="left", color="#5555cc", edgecolor="white")
    ax1.set_xlabel("Total Violations per Segment")
    ax1.set_ylabel("Count of Segments")
    ax1.set_title("Violation Count Distribution")
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Panel 2: family violation rates (stacked horizontal bar, one bar total)
    rates = {vc: float((df[vc].fillna(0) > 0).mean()) for vc in viol_cols if vc in df.columns}
    y = ["Segments"]
    left = 0.0
    for vc, rate in rates.items():
        ax2.barh(y, [rate], left=[left],
                 color=family_colors.get(vc, "grey"),
                 label=family_labels.get(vc, vc), alpha=0.85)
        if rate > 0.03:
            ax2.text(left + rate / 2, 0, f"{rate:.0%}", ha="center", va="center",
                     fontsize=8, color="white", fontweight="bold")
        left += rate
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Fraction of Segments")
    ax2.set_title("Fraction of Segments with Each Family Violated")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.tick_params(left=False, labelleft=False)

    n_flagged = int(df["is_flagged"].astype(bool).sum()) if "is_flagged" in df.columns else "?"
    fig.suptitle(
        f"STARS Severity Overview  |  {len(df):,} segments, {n_flagged} flagged",
        fontsize=11,
    )
    fig.tight_layout()
    return fig
```

- [ ] **Step 2: Implement `plot_threshold_proximity`**

```python
def plot_threshold_proximity(
    stats_df: pd.DataFrame,
    cfg: MonitorConfig | None = None,
    *,
    ncols: int = 3,
    bins: int = 40,
    figsize_per_panel: tuple[float, float] = (4.2, 3.0),
) -> plt.Figure:
    """
    Show where the canonical threshold sits within each metric's value distribution.

    Each panel: histogram of the metric value with a vertical line at the threshold.
    Red = above threshold (flagged side), blue = below.

    Args:
        stats_df: Wide stats DataFrame.
        cfg:      MonitorConfig (defaults to canonical thresholds).
        ncols:    Subplot grid columns.
        bins:     Histogram bin count.
        figsize_per_panel: (width, height) per panel.

    Returns:
        matplotlib Figure.
    """
    if cfg is None:
        cfg = MonitorConfig()

    # Map: wide column name -> (display label, threshold value, flag direction)
    # direction: ">=" means flagged when value >= threshold
    threshold_map: dict[str, tuple[str, float, str]] = {
        "ks_distribution_value":   ("KS Statistic",          cfg.ks_d_threshold,                  ">="),
        "level_shift_value":       ("Cohen's d",              cfg.level_shift_min_cohen_d,          ">="),
        "dw_shift_value":          ("|DW Delta|",             cfg.dw_delta_threshold,               ">="),
        "coverage_shift_value":    ("Coverage Δ",             cfg.coverage_delta_threshold,          ">="),
        "sparsity_change_value":   ("Sparsity Δ",             cfg.sparsity_delta_threshold,          ">="),
        "low_volume_value":        ("Avg Monthly Volume",     cfg.low_volume_monthly_threshold,     "<"),
        "volatility_shift_value":  ("CV Ratio",               cfg.volatility_ratio_threshold,        ">="),
        "outlier_rate_value":      ("Outlier Rate",           cfg.outlier_rate_threshold,            ">="),
    }

    metrics = [(col, label, thr, direction)
               for col, (label, thr, direction) in threshold_map.items()
               if col in stats_df.columns]

    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
    )
    axes_flat = np.array(axes).flatten()

    for i, (col, label, thr, direction) in enumerate(metrics):
        ax = axes_flat[i]
        vals = stats_df[col].dropna().astype(float)
        if direction == ">=":
            flagged_vals   = vals[vals >= thr]
            unflagged_vals = vals[vals < thr]
        else:  # "<"
            flagged_vals   = vals[vals < thr]
            unflagged_vals = vals[vals >= thr]

        all_bins = np.histogram_bin_edges(vals, bins=bins)
        ax.hist(unflagged_vals, bins=all_bins, color="#2ca02c", alpha=0.6, label="Pass")
        ax.hist(flagged_vals,   bins=all_bins, color="#d62728", alpha=0.6, label="Flag")
        ax.axvline(thr, color="black", linestyle="--", linewidth=1.2,
                   label=f"thr={thr:.3g}")
        pct_flagged = len(flagged_vals) / len(vals) if len(vals) > 0 else 0
        ax.set_title(f"{label}\n({pct_flagged:.1%} flagged)", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(len(metrics), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Threshold Proximity — Metric Value Distributions", fontsize=11, y=1.01)
    fig.tight_layout()
    return fig
```

- [ ] **Step 3: Implement `plot_segment_series`**

```python
def plot_segment_series(
    series_df: pd.DataFrame,
    feature_segment: str,
    *,
    recent_days: int = 90,
    figsize: tuple[float, float] = (13, 4),
) -> plt.Figure:
    """
    Time-series plot for a single segment showing train vs recent window split.

    Args:
        series_df:       Daily DataFrame with columns: date, actual, feature_segment.
        feature_segment: The segment key to filter and plot.
        recent_days:     Size of the recent window (shaded differently).
        figsize:         Figure size.

    Returns:
        matplotlib Figure with one axes.
    """
    seg = series_df[series_df["feature_segment"] == feature_segment].copy()
    if seg.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No data for segment: {feature_segment}",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    seg["date"] = pd.to_datetime(seg["date"])
    seg = seg.sort_values("date")

    cutoff = seg["date"].max() - pd.Timedelta(days=recent_days - 1)
    train  = seg[seg["date"] < cutoff]
    recent = seg[seg["date"] >= cutoff]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(train["date"],  train["actual"],  color="#1f77b4", lw=1.2, label="Training window")
    ax.plot(recent["date"], recent["actual"], color="#d62728", lw=1.5, label=f"Recent {recent_days}d")
    ax.axvline(cutoff, color="grey", linestyle="--", linewidth=0.9)
    ax.fill_between(recent["date"], recent["actual"], alpha=0.12, color="#d62728")
    ax.set_title(f"Segment: {feature_segment}", fontsize=9)
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run all tests**

```bash
uv run pytest tests/viz/test_plots.py -v
```

Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add stars_pipeline/viz/plots.py
git commit -m "feat: implement plot_severity_and_families, plot_threshold_proximity, plot_segment_series"
```

---

## Task 7: Implement `stars-viz` CLI entry point

**Files:**
- Modify: `stars_pipeline/viz/cli.py`
- Create: `tests/viz/test_cli.py`

- [ ] **Step 1: Write failing test**

Create `tests/viz/test_cli.py`:
```python
import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from stars_pipeline.viz.cli import main


def _make_long_csv(tmp_path: Path) -> Path:
    """Write a minimal long-format CSV for CLI testing."""
    from tests.viz.test_wide import _make_long_df
    df = _make_long_df()
    p = tmp_path / "stars_results.csv"
    df.to_csv(p, index=False)
    return p


def test_cli_produces_png_files(tmp_path):
    csv_path = _make_long_csv(tmp_path)
    out_dir = tmp_path / "plots"
    rc = main(["--input", str(csv_path), "--output-dir", str(out_dir)])
    assert rc == 0
    pngs = list(out_dir.glob("*.png"))
    assert len(pngs) >= 5, f"Expected at least 5 PNG files, got {len(pngs)}: {pngs}"


def test_cli_missing_input_returns_nonzero(tmp_path):
    rc = main(["--input", str(tmp_path / "nonexistent.csv"),
               "--output-dir", str(tmp_path / "out")])
    assert rc != 0
```

- [ ] **Step 2: Run to verify failures**

```bash
uv run pytest tests/viz/test_cli.py -v
```

Expected: failures (NotImplementedError or missing arg).

- [ ] **Step 3: Implement `cli.py`**

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stars-viz",
        description="Generate diagnostic visualizations from a STARS long-format CSV.",
    )
    p.add_argument("--input",      required=True,  help="Path to long-format CSV from stars-pipeline.")
    p.add_argument("--output-dir", required=True,  help="Directory to write PNG files.")
    p.add_argument("--recent-days", type=int, default=90, help="Recent window size (for series plot).")
    p.add_argument("--dpi", type=int, default=150, help="PNG resolution in DPI.")
    return p


def main(argv: list[str] | None = None) -> int:
    import pandas as pd
    from stars_pipeline.viz._wide import long_to_wide
    from stars_pipeline.viz.plots import (
        plot_metric_distributions,
        plot_normal_breakdowns,
        plot_flag_correlation_grid,
        plot_flag_rates_by_dim,
        plot_severity_and_families,
        plot_threshold_proximity,
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df = pd.read_csv(input_path)
    stats_df = long_to_wide(long_df)

    plots = [
        ("metric_distributions",  lambda: plot_metric_distributions(stats_df)),
        ("normal_breakdowns",     lambda: plot_normal_breakdowns(stats_df)),
        ("flag_correlation_grid", lambda: plot_flag_correlation_grid(stats_df)),
        ("flag_rates_by_dim",     lambda: plot_flag_rates_by_dim(stats_df)),
        ("severity_and_families", lambda: plot_severity_and_families(stats_df)),
        ("threshold_proximity",   lambda: plot_threshold_proximity(stats_df)),
    ]

    for name, fn in plots:
        try:
            fig = fn()
            out_path = out_dir / f"stars_{name}.png"
            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"Written: {out_path}")
        except Exception as exc:
            print(f"WARNING: {name} failed: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/viz/test_cli.py -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
uv run pytest -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add stars_pipeline/viz/cli.py tests/viz/test_cli.py
git commit -m "feat: implement stars-viz CLI entry point with PNG output"
```

---

## Self-Review

**Spec coverage:**
- ✅ 7 plot functions ported from `performance_investigation.py`
- ✅ `long_to_wide()` pivot helper for CLI use
- ✅ `stars-viz` CLI entry point
- ✅ `is_normal` → `is_flagged` inversion applied throughout
- ✅ Old column name → new column name mapping in `_METRIC_DISPLAY` and `threshold_map`
- ✅ `matplotlib`/`seaborn` added as optional `viz` dependencies
- ✅ Static PNG output (no Streamlit)
- ✅ Smoke tests for every plot function (returns `plt.Figure`)

**Placeholder scan:** No placeholders found.

**Type consistency:** All functions take `stats_df: pd.DataFrame` and return `plt.Figure`. `long_to_wide` takes/returns `pd.DataFrame`. CLI `main()` signature is `(argv: list[str] | None = None) -> int` throughout.
