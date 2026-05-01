# MESH + Accuracy Bands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface champion MESH and accuracy bands (`within_3`, `within_5`, `within_10`) in the long-format output CSV, update the viz module to handle them, and add three MESH × STARS analysis plots.

**Architecture:** MESH is already computed in `stats_df` by `run_monitoring()`. We extend `to_long_format()` to emit four new Summary rows per segment, update the wide pivot helper and its fixtures, then add three plot functions that show how MESH and accuracy bands relate to STARS flags.

**Tech Stack:** Python 3.11, pandas 2.x, matplotlib 3.8, seaborn 0.13, pytest

---

## File Map

| File | Change |
|------|--------|
| `stars_pipeline/stars/output.py` | Add 4 new Summary rows: `mesh`, `within_3`, `within_5`, `within_10` |
| `stars_pipeline/viz/_wide.py` | Add 4 new names to `_SUMMARY_METRICS` with special handling for `mesh` (from `metric_value`) |
| `tests/viz/conftest.py` | Add 4 new summary rows to `_make_long_df()` |
| `stars_pipeline/viz/plots.py` | Add 3 new plot functions |
| `stars_pipeline/viz/__init__.py` | Re-export 3 new plot functions |
| `stars_pipeline/viz/cli.py` | Add 3 new plots to the run list |
| `tests/stars/test_output.py` | Update row count (56→60) + add assertions for new rows |
| `tests/viz/test_wide.py` | Add assertions for new summary columns |
| `tests/viz/test_plots.py` | Add smoke tests for 3 new plot functions; add mesh/band cols to `_make_stats_df` |
| `tests/viz/test_cli.py` | Update min PNG count (5→8) |

---

### Task 1: Add MESH + bands to `to_long_format()`

**Files:**
- Modify: `stars_pipeline/stars/output.py`
- Modify: `tests/stars/test_output.py`

**Background:** `to_long_format()` already emits 5 Summary rows per segment ending at line 158. MESH is already a column in `stats_df` (put there by `run_monitoring()` reading `group["mesh"].iloc[0]`). We need to emit 4 more Summary rows per segment.

**MESH row encoding:**
- `mesh`: `metric_value = str(mesh_value)`, `metric_flag = 1 if mesh > 10 else 0` (within_10 boundary; `within_10=1` means MESH ≤ 10, so `mesh_flag=1` means *outside* the broadest band, consistent with all other flags meaning "bad")
- `within_3`: `metric_value = str(mesh_value)`, `metric_flag = 1 if mesh <= 3 else 0`
- `within_5`: `metric_value = str(mesh_value)`, `metric_flag = 1 if mesh <= 5 else 0`
- `within_10`: `metric_value = str(mesh_value)`, `metric_flag = 1 if mesh <= 10 else 0`

Note: for `within_N` rows, `metric_flag=1` means "within band" (good). For `mesh`, `metric_flag=1` means "outside 10% band" (bad), consistent with all other flags where 1=problem.

- [ ] **Step 1: Write failing tests**

In `tests/stars/test_output.py`, after `test_write_long_csv_creates_file`, add:

```python
def test_long_format_has_60_rows_per_segment():
    stats = _make_stats_row()
    result = to_long_format(stats)
    # 11 primary + 40 intermediates + 5 original summary + 4 mesh/band summary rows
    assert len(result) == 60


def test_mesh_summary_row():
    stats = _make_stats_row(mesh=2.5)
    result = to_long_format(stats)
    row = result[result["metric_name"] == "mesh"].iloc[0]
    assert row["stars_family"] == "Summary"
    assert row["metric_value"] == "2.5"
    assert int(row["metric_flag"]) == 0  # 2.5 <= 10, so NOT outside band → flag=0


def test_mesh_flag_outside_band():
    stats = _make_stats_row(mesh=15.0)
    result = to_long_format(stats)
    row = result[result["metric_name"] == "mesh"].iloc[0]
    assert int(row["metric_flag"]) == 1  # 15.0 > 10 → outside broadest band → flag=1


def test_within_3_row():
    stats = _make_stats_row(mesh=2.5)
    result = to_long_format(stats)
    row = result[result["metric_name"] == "within_3"].iloc[0]
    assert row["stars_family"] == "Summary"
    assert row["metric_value"] == "2.5"
    assert int(row["metric_flag"]) == 1  # 2.5 <= 3


def test_within_5_row():
    stats = _make_stats_row(mesh=4.0)
    result = to_long_format(stats)
    row = result[result["metric_name"] == "within_5"].iloc[0]
    assert row["stars_family"] == "Summary"
    assert row["metric_value"] == "4.0"
    assert int(row["metric_flag"]) == 1  # 4.0 <= 5


def test_within_10_row():
    stats = _make_stats_row(mesh=8.0)
    result = to_long_format(stats)
    row = result[result["metric_name"] == "within_10"].iloc[0]
    assert row["stars_family"] == "Summary"
    assert row["metric_value"] == "8.0"
    assert int(row["metric_flag"]) == 1  # 8.0 <= 10


def test_within_bands_outside():
    stats = _make_stats_row(mesh=7.0)
    result = to_long_format(stats)
    assert int(result[result["metric_name"] == "within_3"]["metric_flag"].iloc[0]) == 0
    assert int(result[result["metric_name"] == "within_5"]["metric_flag"].iloc[0]) == 0
    assert int(result[result["metric_name"] == "within_10"]["metric_flag"].iloc[0]) == 1


def test_mesh_none_produces_null_rows():
    stats = _make_stats_row(mesh=None)
    result = to_long_format(stats)
    for name in ["mesh", "within_3", "within_5", "within_10"]:
        row = result[result["metric_name"] == name].iloc[0]
        assert row["metric_value"] is None or pd.isna(row["metric_value"]), \
            f"{name} metric_value should be None when mesh is None"
        assert row["metric_flag"] is None or pd.isna(row["metric_flag"]), \
            f"{name} metric_flag should be None when mesh is None"
```

Also update `test_long_format_has_56_rows_per_segment` to expect 60 rows, and update `test_write_long_csv_creates_file` to check 60 rows:

```python
def test_long_format_has_56_rows_per_segment():
    # rename/update to 60
    stats = _make_stats_row()
    result = to_long_format(stats)
    # 11 primary + 40 intermediates + 5 original summary + 4 mesh/band summary rows
    assert len(result) == 60


# In test_write_long_csv_creates_file, change:
    assert len(df) == 60
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/stars/test_output.py -v`

Expected: `test_long_format_has_56_rows_per_segment` FAILS (expected 60, got 56); all new tests FAIL with "0 rows" or similar.

- [ ] **Step 3: Update `to_long_format()` in `output.py`**

Add the 4 new Summary rows at the end of the per-segment summary block (after the existing 5 summary rows, around line 158). The mesh bands are computed from the raw `mesh` value. Replace the summary block (lines 141–158) with:

```python
        # ── Summary rows ─────────────────────────────────────────────────────
        family_counts = [int(stat_row.get(m, 0) or 0) for m in _SUMMARY_METRICS]
        total_violations = sum(family_counts)
        is_flagged = stat_row.get("is_flagged")
        rows.append({
            **segment,
            "stars_family": "Summary",
            "metric_name":  "is_flagged",
            "metric_value": str(total_violations),
            "metric_flag":  int(bool(is_flagged)) if pd.notna(is_flagged) else None,
        })
        for metric_name, count in zip(_SUMMARY_METRICS, family_counts):
            rows.append({
                **segment,
                "stars_family": "Summary",
                "metric_name":  metric_name,
                "metric_value": str(count),
                "metric_flag":  1 if count > 0 else 0,
            })

        # ── MESH + accuracy band rows ─────────────────────────────────────────
        mesh_val = stat_row.get("mesh")
        if pd.notna(mesh_val):
            mesh_float = float(mesh_val)
            rows.append({
                **segment,
                "stars_family": "Summary",
                "metric_name":  "mesh",
                "metric_value": str(mesh_float),
                "metric_flag":  1 if mesh_float > 10.0 else 0,
            })
            for band, threshold in [("within_3", 3.0), ("within_5", 5.0), ("within_10", 10.0)]:
                rows.append({
                    **segment,
                    "stars_family": "Summary",
                    "metric_name":  band,
                    "metric_value": str(mesh_float),
                    "metric_flag":  1 if mesh_float <= threshold else 0,
                })
        else:
            for name in ["mesh", "within_3", "within_5", "within_10"]:
                rows.append({
                    **segment,
                    "stars_family": "Summary",
                    "metric_name":  name,
                    "metric_value": None,
                    "metric_flag":  None,
                })
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/stars/test_output.py -v`

Expected: All tests PASS (including the new 8 + updated 2).

- [ ] **Step 5: Commit**

```bash
git add stars_pipeline/stars/output.py tests/stars/test_output.py
git commit -m "feat: add mesh and accuracy band rows to long-format output"
```

---

### Task 2: Update `long_to_wide()` for new summary metrics

**Files:**
- Modify: `stars_pipeline/viz/_wide.py`
- Modify: `tests/viz/conftest.py`
- Modify: `tests/viz/test_wide.py`

**Background:** `long_to_wide()` uses `_SUMMARY_METRICS` to decide which rows to read from `metric_flag` vs `metric_value`. The new metrics need different treatment:
- `mesh`: read from `metric_value` (it's a float, like violation counts). Like `stability_violations`.
- `within_3`, `within_5`, `within_10`: read from `metric_flag` (they are binary 0/1 flags). Like `is_flagged`.

The current `long_to_wide()` branch for summary metrics (line 40–41):
```python
if metric in _SUMMARY_METRICS:
    row[metric] = r["metric_flag"] if metric == "is_flagged" else r["metric_value"]
```
We need to extend this with a second "flag-valued" set for `within_N` metrics.

- [ ] **Step 1: Add 4 rows to `_make_long_df()` in `conftest.py`**

After the last summary row (`("regularity_violations", "0", 0)`), add:

```python
    for metric, value, flag in [
        ("mesh",     "2.5", 0),
        ("within_3", "2.5", 1),
        ("within_5", "2.5", 1),
        ("within_10","2.5", 1),
    ]:
        rows.append({**seg, "stars_family": "Summary", "metric_name": metric,
                     "metric_value": value, "metric_flag": flag})
```

- [ ] **Step 2: Write failing tests in `test_wide.py`**

After `test_long_to_wide_multi_segment`, add:

```python
def test_long_to_wide_has_mesh_and_band_columns():
    df = long_to_wide(_make_long_df())
    for col in ["mesh", "within_3", "within_5", "within_10"]:
        assert col in df.columns, f"missing column: {col}"


def test_long_to_wide_mesh_is_numeric():
    df = long_to_wide(_make_long_df())
    assert float(df["mesh"].iloc[0]) == pytest.approx(2.5)


def test_long_to_wide_within_bands_are_flags():
    df = long_to_wide(_make_long_df())
    assert int(df["within_3"].iloc[0]) == 1
    assert int(df["within_5"].iloc[0]) == 1
    assert int(df["within_10"].iloc[0]) == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/viz/test_wide.py -v`

Expected: 3 new tests FAIL (KeyError or assertion on missing column).

- [ ] **Step 4: Update `_wide.py`**

Replace the current `_SUMMARY_METRICS` set and summary handling in `long_to_wide()`:

```python
_ID_COLS = ["strata_id", "entity_id", "patient_type_rollup", "service_line", "feature_segment"]

# Summary metrics read from metric_value (numeric)
_VALUE_SUMMARY_METRICS = {
    "stability_violations", "truthfulness_violations",
    "abundance_violations", "regularity_violations",
    "mesh",
}

# Summary metrics read from metric_flag (binary 0/1)
_FLAG_SUMMARY_METRICS = {
    "is_flagged", "within_3", "within_5", "within_10",
}

_SUMMARY_METRICS = _VALUE_SUMMARY_METRICS | _FLAG_SUMMARY_METRICS
```

And replace the summary branch inside the `for _, r in grp.iterrows():` loop:

```python
            if metric in _SUMMARY_METRICS:
                if metric in _FLAG_SUMMARY_METRICS:
                    row[metric] = r["metric_flag"]
                else:
                    row[metric] = r["metric_value"]
```

The duplicate-detection sentinel logic also needs updating. Replace the sentinel line:

```python
            sentinel = metric if metric in _SUMMARY_METRICS else f"{metric}_value"
```

This line stays the same — the sentinel is still `metric` for summary rows and `{metric}_value` for primary rows. No change needed there.

Full replacement of `_wide.py`:

```python
from __future__ import annotations
import pandas as pd

_ID_COLS = ["strata_id", "entity_id", "patient_type_rollup", "service_line", "feature_segment"]

# Summary metrics read from metric_value (numeric)
_VALUE_SUMMARY_METRICS = {
    "stability_violations", "truthfulness_violations",
    "abundance_violations", "regularity_violations",
    "mesh",
}

# Summary metrics read from metric_flag (binary 0/1)
_FLAG_SUMMARY_METRICS = {
    "is_flagged", "within_3", "within_5", "within_10",
}

_SUMMARY_METRICS = _VALUE_SUMMARY_METRICS | _FLAG_SUMMARY_METRICS


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
            sentinel = metric if metric in _SUMMARY_METRICS else f"{metric}_value"
            if sentinel in row:
                raise ValueError(
                    f"Duplicate metric '{metric}' for segment '{seg_key}'"
                )
            if metric in _SUMMARY_METRICS:
                if metric in _FLAG_SUMMARY_METRICS:
                    row[metric] = r["metric_flag"]
                else:
                    row[metric] = r["metric_value"]
            else:
                row[f"{metric}_value"] = r["metric_value"]
                row[f"{metric}_flag"] = r["metric_flag"]
        rows.append(row)

    return pd.DataFrame(rows)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/viz/test_wide.py tests/viz/test_cli.py -v`

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add stars_pipeline/viz/_wide.py tests/viz/conftest.py tests/viz/test_wide.py
git commit -m "feat: add mesh and accuracy band columns to long_to_wide"
```

---

### Task 3: Add MESH × STARS analysis plot functions

**Files:**
- Modify: `stars_pipeline/viz/plots.py`
- Modify: `stars_pipeline/viz/__init__.py`
- Modify: `tests/viz/test_plots.py`

**Background:** Three new plots that relate MESH accuracy to STARS flag status:

1. **`plot_mesh_distribution`** — MESH distribution split by `is_flagged` (Normal vs Atypical), with vertical lines at 3%, 5%, 10% thresholds. Shows whether flagged segments tend to have higher prediction error.

2. **`plot_mesh_by_flag`** — Box/violin plot: MESH by each STARS flag column (one box per flag=0, flag=1 per indicator). Compact multi-panel. Shows which specific STARS flags correlate with high MESH.

3. **`plot_accuracy_band_by_flag`** — Stacked bar: for each accuracy band (`within_3`, `within_5`, `within_10`), show the fraction of Normal vs Atypical segments that fall within the band. Shows accuracy target attainment by STARS status.

These functions accept wide `stats_df` (output of `long_to_wide()`).

- [ ] **Step 1: Add mesh/band columns to `_make_stats_df()` in `test_plots.py`**

In the existing `_make_stats_df()` function, inside the `rows.append({...})` dict (after the `"mesh"` line if present, or after `"is_flagged"`), add:

```python
            "mesh": float(rng.uniform(1.0, 20.0)),
            "within_3":  int(rng.random() < 0.4),
            "within_5":  int(rng.random() < 0.6),
            "within_10": int(rng.random() < 0.8),
```

If `"mesh"` is already in `_make_stats_df()` (the summary shows it was added in the viz module session), update its range to `(1.0, 20.0)` to ensure a mix above and below all three thresholds.

- [ ] **Step 2: Write failing smoke tests in `test_plots.py`**

After `test_plot_segment_series_returns_figure`, add:

```python
from stars_pipeline.viz.plots import (
    plot_mesh_distribution,
    plot_mesh_by_flag,
    plot_accuracy_band_by_flag,
)


def test_plot_mesh_distribution_returns_figure(stats_df):
    fig = plot_mesh_distribution(stats_df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_mesh_by_flag_returns_figure(stats_df):
    fig = plot_mesh_by_flag(stats_df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_accuracy_band_by_flag_returns_figure(stats_df):
    fig = plot_accuracy_band_by_flag(stats_df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
```

Note: move the extra import block to the top of the file with the other imports.

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/viz/test_plots.py::test_plot_mesh_distribution_returns_figure tests/viz/test_plots.py::test_plot_mesh_by_flag_returns_figure tests/viz/test_plots.py::test_plot_accuracy_band_by_flag_returns_figure -v`

Expected: ImportError — functions not yet defined.

- [ ] **Step 4: Implement the 3 new plot functions in `plots.py`**

Add after `plot_segment_series` (at the end of the file):

```python
def plot_mesh_distribution(
    stats_df: pd.DataFrame,
    *,
    bins: int = 40,
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """MESH distribution split by STARS flag status, with accuracy band thresholds."""
    if "mesh" not in stats_df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No 'mesh' column found", ha="center", va="center")
        return fig

    df = stats_df.copy()
    df["mesh"] = pd.to_numeric(df["mesh"], errors="coerce")
    normal_mask = ~df["is_flagged"].astype(bool) if "is_flagged" in df.columns else pd.Series(True, index=df.index)

    fig, ax = plt.subplots(figsize=figsize)

    for mask, label, color in [
        (normal_mask,  "Normal (not flagged)",   "#2ca02c"),
        (~normal_mask, "Atypical (flagged)",      "#d62728"),
    ]:
        vals = df.loc[mask, "mesh"].dropna()
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, color=color, alpha=0.45, density=True, label=label)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                sns.kdeplot(vals, ax=ax, color=color, linewidth=1.4)
            except Exception:
                pass

    for thr, label, ls in [(3.0, "3%", "--"), (5.0, "5%", "-."), (10.0, "10%", ":")]:
        ax.axvline(thr, color="black", linestyle=ls, linewidth=0.9, label=f"Band {label}")

    ax.set_xlabel("Champion MESH (%)")
    ax.set_ylabel("Density")
    ax.set_title("Champion MESH Distribution — Normal vs Atypical Segments", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_mesh_by_flag(
    stats_df: pd.DataFrame,
    *,
    ncols: int = 3,
    figsize_per_panel: tuple[float, float] = (3.5, 3.0),
) -> plt.Figure:
    """Box plots of MESH by flag value (0/1) for each STARS indicator flag column."""
    if "mesh" not in stats_df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No 'mesh' column found", ha="center", va="center")
        return fig

    df = stats_df.copy()
    df["mesh"] = pd.to_numeric(df["mesh"], errors="coerce")

    flag_cols = [c for c in stats_df.columns
                 if c.endswith("_flag") and c != "is_flagged" and c in stats_df.columns]
    flag_label_map = {
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
    }

    if not flag_cols:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No flag columns found", ha="center", va="center")
        return fig

    nrows = math.ceil(len(flag_cols) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, fcol in enumerate(flag_cols):
        ax = axes_flat[i]
        label = flag_label_map.get(fcol, fcol.replace("_flag", ""))
        groups = [
            df.loc[df[fcol].astype(float) == v, "mesh"].dropna()
            for v in [0, 1]
        ]
        ax.boxplot(
            [g for g in groups if len(g) > 0],
            labels=[f"Pass (n={len(groups[0])})", f"Flag (n={len(groups[1])})"],
            patch_artist=True,
            boxprops=dict(facecolor="#c6dbef"),
            medianprops=dict(color="black", linewidth=1.5),
        )
        ax.set_title(label, fontsize=8)
        ax.set_ylabel("MESH (%)" if i % ncols == 0 else "")
        ax.tick_params(labelsize=7)

    for j in range(len(flag_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Champion MESH by STARS Indicator Flag", fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


def plot_accuracy_band_by_flag(
    stats_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure:
    """
    For each accuracy band (within_3/5/10), show attainment rate split by
    Normal vs Atypical STARS classification.
    """
    band_cols = [c for c in ["within_3", "within_5", "within_10"] if c in stats_df.columns]

    if not band_cols:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No accuracy band columns found", ha="center", va="center")
        return fig

    df = stats_df.copy()
    if "is_flagged" in df.columns:
        df["stars_status"] = df["is_flagged"].astype(bool).map({False: "Normal", True: "Atypical"})
    else:
        df["stars_status"] = "Unknown"

    band_labels = {"within_3": "Within 3%", "within_5": "Within 5%", "within_10": "Within 10%"}
    statuses = ["Normal", "Atypical"]
    status_colors = {"Normal": "#2ca02c", "Atypical": "#d62728"}

    x = np.arange(len(band_cols))
    width = 0.35
    fig, ax = plt.subplots(figsize=figsize)

    for i, status in enumerate(statuses):
        subset = df[df["stars_status"] == status]
        rates = [
            subset[bc].astype(float).mean() if bc in subset.columns and len(subset) > 0 else 0.0
            for bc in band_cols
        ]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, rates, width,
                      label=f"{status} (n={len(subset)})",
                      color=status_colors[status], alpha=0.8)
        for bar, rate in zip(bars, rates):
            if rate > 0.04:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{rate:.0%}",
                    ha="center", va="bottom", fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([band_labels.get(bc, bc) for bc in band_cols])
    ax.set_ylabel("Fraction of Segments Within Band")
    ax.set_ylim(0, 1.15)
    ax.set_title("Accuracy Band Attainment — Normal vs Atypical Segments", fontsize=10)
    ax.legend(fontsize=9)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.7)
    fig.tight_layout()
    return fig
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/viz/test_plots.py -v`

Expected: All tests PASS including 3 new ones.

- [ ] **Step 6: Commit**

```bash
git add stars_pipeline/viz/plots.py tests/viz/test_plots.py
git commit -m "feat: add mesh_distribution, mesh_by_flag, accuracy_band_by_flag plots"
```

---

### Task 4: Wire new plots into `__init__.py` and CLI

**Files:**
- Modify: `stars_pipeline/viz/__init__.py`
- Modify: `stars_pipeline/viz/cli.py`
- Modify: `tests/viz/test_cli.py`

**Background:** The CLI runs every plot in a list and saves to PNG. We need to add the 3 new functions to the imports and the run list. The CLI test asserts `len(pngs) >= 5`; we update to `>= 8`.

- [ ] **Step 1: Write failing test**

In `test_cli.py`, update `test_cli_produces_png_files`:

```python
def test_cli_produces_png_files(tmp_path):
    csv_path = _make_long_csv(tmp_path)
    out_dir = tmp_path / "plots"
    rc = main(["--input", str(csv_path), "--output-dir", str(out_dir)])
    assert rc == 0
    pngs = list(out_dir.glob("*.png"))
    assert len(pngs) >= 8, f"Expected at least 8 PNG files, got {len(pngs)}: {pngs}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/viz/test_cli.py::test_cli_produces_png_files -v`

Expected: FAIL — only 6 PNGs produced, expected >= 8.

- [ ] **Step 3: Update `__init__.py`**

Replace `stars_pipeline/viz/__init__.py` with:

```python
from stars_pipeline.viz.plots import (
    plot_metric_distributions,
    plot_normal_breakdowns,
    plot_flag_correlation_grid,
    plot_flag_rates_by_dim,
    plot_severity_and_families,
    plot_threshold_proximity,
    plot_segment_series,
    plot_mesh_distribution,
    plot_mesh_by_flag,
    plot_accuracy_band_by_flag,
)

__all__ = [
    "plot_metric_distributions",
    "plot_normal_breakdowns",
    "plot_flag_correlation_grid",
    "plot_flag_rates_by_dim",
    "plot_severity_and_families",
    "plot_threshold_proximity",
    "plot_segment_series",
    "plot_mesh_distribution",
    "plot_mesh_by_flag",
    "plot_accuracy_band_by_flag",
]
```

- [ ] **Step 4: Update `cli.py`**

Add imports and 3 new plot entries. In the `main()` function, add imports:

```python
    from stars_pipeline.viz.plots import (
        plot_metric_distributions,
        plot_normal_breakdowns,
        plot_flag_correlation_grid,
        plot_flag_rates_by_dim,
        plot_severity_and_families,
        plot_threshold_proximity,
        plot_mesh_distribution,
        plot_mesh_by_flag,
        plot_accuracy_band_by_flag,
    )
```

And extend the `plots` list:

```python
    plots = [
        ("metric_distributions",    lambda: plot_metric_distributions(stats_df)),
        ("normal_breakdowns",       lambda: plot_normal_breakdowns(stats_df)),
        ("flag_correlation_grid",   lambda: plot_flag_correlation_grid(stats_df)),
        ("flag_rates_by_dim",       lambda: plot_flag_rates_by_dim(stats_df)),
        ("severity_and_families",   lambda: plot_severity_and_families(stats_df)),
        ("threshold_proximity",     lambda: plot_threshold_proximity(stats_df)),
        ("mesh_distribution",       lambda: plot_mesh_distribution(stats_df)),
        ("mesh_by_flag",            lambda: plot_mesh_by_flag(stats_df)),
        ("accuracy_band_by_flag",   lambda: plot_accuracy_band_by_flag(stats_df)),
    ]
```

- [ ] **Step 5: Run full test suite to verify everything passes**

Run: `pytest tests/viz/ tests/stars/test_output.py -v`

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add stars_pipeline/viz/__init__.py stars_pipeline/viz/cli.py tests/viz/test_cli.py
git commit -m "feat: wire mesh/accuracy-band plots into viz CLI and public API"
```

---

## Self-Review

**Spec coverage:**

| Requirement | Task |
|-------------|------|
| Add `mesh` to output CSV | Task 1 |
| Add `within_3`, `within_5`, `within_10` to output CSV | Task 1 |
| Update `long_to_wide()` for new cols | Task 2 |
| Update conftest fixture | Task 2 |
| MESH distribution plot | Task 3 |
| MESH by STARS flag plot | Task 3 |
| Accuracy band × flag attainment plot | Task 3 |
| Wire into CLI | Task 4 |

**Placeholder scan:** No TBDs, TODOs, or "similar to Task N" references. All code blocks are complete.

**Type consistency:**
- `plot_mesh_distribution`, `plot_mesh_by_flag`, `plot_accuracy_band_by_flag` all accept `pd.DataFrame` → `plt.Figure`. Consistent with all other plot functions.
- `_FLAG_SUMMARY_METRICS` and `_VALUE_SUMMARY_METRICS` used consistently in Task 2 throughout the replacement.
- `mesh`, `within_3`, `within_5`, `within_10` column names consistent across `output.py`, `_wide.py`, and `plots.py`.

**Row count arithmetic:**
- Current: 11 primary + 40 intermediates + 5 summary = 56 rows
- After Task 1: 56 + 4 = 60 rows
- All test assertions updated to 60.

**Flag encoding note:** For `within_N` bands, flag=1 means *within* band (good). This is intentional — these columns are accuracy targets, not STARS diagnostic flags. The `mesh` column uses flag=1 to mean *outside* the 10% band (bad), consistent with all other STARS flags where 1=problem. Tests explicitly verify both behaviors.
