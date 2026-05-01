# Training / Threshold Calibration Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the Monte Carlo threshold calibration system from `performance_investigation.py` into a clean `stars_pipeline/training/` module that accepts a pre-computed wide stats DataFrame and a labeled segment DataFrame, searches the threshold parameter space, and returns Pareto-optimal configurations plus an accuracy dial.

**Architecture:** Three focused files: `mc_search.py` (the search loop and Pareto helpers), `pareto.py` (neighborhood analysis and accuracy dial), and a `cli.py` entry point (`stars-train`). The labeled segment DataFrame with `within_10` / `within_5` / `within_3` columns is derived by the caller from the MESH column (`mesh <= threshold`) — this module never touches Snowflake. All functions are pure (stats in → results out) with no global state.

**Tech Stack:** numpy, pandas, scipy (already in pyproject.toml); no new dependencies needed.

---

## Context for the implementer

### What the old pipeline did (performance_investigation.py)

1. Loaded segment data with columns including `mesh` (model error score per segment).
2. Derived binary labels: `within_10 = mesh <= 10%`, `within_5 = mesh <= 5%`, `within_3 = mesh <= 3%`. A segment is "good" if `within_10 = True`.
3. Ran `run_monitoring()` to get a wide stats DataFrame (one row per segment, raw statistics).
4. Called `apply_thresholds_to_stats(stats_df, cfg)` to classify each segment as normal/abnormal given a config.
5. Ran `run_mc_search(stats_df, segment_df, base_cfg, n_samples=48000)` — random sampling over the threshold parameter space. Each sample: draw thresholds → classify → measure accuracy/retention.
6. Tagged Pareto-optimal samples (maximize both retention AND accuracy simultaneously).
7. Called `analyze_pareto_neighborhood()` to get elasticity / recommended values per parameter.
8. Called `build_accuracy_dial()` to produce a lookup: target_accuracy → (expected_retention_range, example configs).

### Key adaptation for the new pipeline

- Old: `apply_thresholds_to_stats(stats_df, cfg)` returned a column named `is_normal` (True = good).
- New: `apply_thresholds(stats_df, cfg=cfg)` returns `is_flagged` (True = bad). All accuracy computations invert this: `normal_mask = ~stats_df["is_flagged"].astype(bool)`.
- Old `MonitorConfig` had test toggle fields inside an `optional_tests` dict. New `MonitorConfig` has `low_volume_enabled: bool` and `volatility_shift_enabled: bool` as direct fields.
- The MC search currently varies these toggles randomly. In the new pipeline, the only two toggleable tests are `low_volume_enabled` and `volatility_shift_enabled`.

### Float threshold parameters to search (from `MonitorConfig`)

```python
MC_PARAM_BOUNDS = {
    "ks_d_threshold":               (0.10, 0.60),
    "level_shift_min_cohen_d":      (0.20, 2.50),
    "dw_delta_threshold":           (0.20, 2.50),
    "slope_change_ratio_threshold": (0.10, 3.00),
    "slope_threshold":              (0.001, 0.10),
    "kpss_alpha":                   (0.001, 0.20),
    "trend_p_value_threshold":      (0.001, 0.30),
    "coverage_delta_threshold":     (0.005, 0.60),
    "sparsity_delta_threshold":     (0.01, 0.60),
    "low_volume_monthly_threshold": (0.0,  15.0),
    "volatility_ratio_threshold":   (1.10, 8.00),
    "outlier_z_threshold":          (1.50, 6.00),
    "outlier_rate_threshold":       (0.01, 0.60),
    "acf_divergence_p_threshold":   (0.001, 0.20),
    "alpha":                        (0.001, 0.20),
}
```

Toggle parameters: `low_volume_enabled` (bool), `volatility_shift_enabled` (bool).

### Accuracy metric columns the search produces

- `normal_accuracy_10`: mean `within_10` among segments classified as normal. This is the primary metric — a segment the model classifies as "normal" should actually be a good predictor (`within_10=True`).
- `normal_accuracy_5` and `normal_accuracy_3`: same for tighter thresholds.
- `abnormal_accuracy_10`: mean `within_10` among segments classified as abnormal. Expected to be low.
- `overall_accuracy_10`: mean `within_10` across all segments.
- `pct_normal`: fraction of segments classified as normal.

### Pareto front definition

A sample is Pareto-optimal if no other sample has **both** higher `pct_normal` AND higher `normal_accuracy_10`. Pareto neighborhood: within 2% tolerance on both axes.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `stars_pipeline/training/__init__.py` | Create | Re-export public API |
| `stars_pipeline/training/mc_search.py` | Create | `run_mc_search()`, `_pareto_front()`, `_pareto_neighborhood()`, `MC_PARAM_BOUNDS` |
| `stars_pipeline/training/pareto.py` | Create | `analyze_pareto_neighborhood()`, `build_accuracy_dial()`, `dial_to_config()` |
| `stars_pipeline/training/cli.py` | Create | `stars-train` CLI entry point |
| `tests/training/__init__.py` | Create | Empty |
| `tests/training/test_mc_search.py` | Create | Unit tests for MC search and Pareto helpers |
| `tests/training/test_pareto.py` | Create | Unit tests for neighborhood analysis and accuracy dial |
| `tests/training/test_cli.py` | Create | CLI smoke test |
| `pyproject.toml` | Modify | Add `stars-train` entry point |

---

## Task 1: Scaffold training package

**Files:**
- Create: `stars_pipeline/training/__init__.py`
- Create: `stars_pipeline/training/mc_search.py` (stub)
- Create: `stars_pipeline/training/pareto.py` (stub)
- Create: `stars_pipeline/training/cli.py` (stub)
- Create: `tests/training/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `stars-train` entry point to pyproject.toml**

In `[project.scripts]`, add:
```toml
stars-train = "stars_pipeline.training.cli:main"
```

- [ ] **Step 2: Create package files**

Create `stars_pipeline/training/__init__.py`:
```python
from stars_pipeline.training.mc_search import run_mc_search, MC_PARAM_BOUNDS
from stars_pipeline.training.pareto import (
    analyze_pareto_neighborhood,
    build_accuracy_dial,
    dial_to_config,
)

__all__ = [
    "run_mc_search",
    "MC_PARAM_BOUNDS",
    "analyze_pareto_neighborhood",
    "build_accuracy_dial",
    "dial_to_config",
]
```

Create `stars_pipeline/training/mc_search.py` stub:
```python
from __future__ import annotations
import pandas as pd
from stars_pipeline.config import MonitorConfig

MC_PARAM_BOUNDS: dict[str, tuple[float, float]] = {}

def run_mc_search(
    stats_df: pd.DataFrame,
    segment_df: pd.DataFrame,
    base_cfg: MonitorConfig,
    *,
    n_samples: int = 500,
    param_bounds: dict[str, tuple[float, float]] | None = None,
    rng_seed: int = 42,
    fixed_toggles: dict[str, bool] | None = None,
) -> pd.DataFrame:
    raise NotImplementedError
```

Create `stars_pipeline/training/pareto.py` stub:
```python
from __future__ import annotations
import pandas as pd
from stars_pipeline.config import MonitorConfig

def analyze_pareto_neighborhood(
    mc_results: pd.DataFrame,
    *,
    tol: float = 0.02,
    accuracy_col: str = "normal_accuracy_10",
    param_bounds: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    raise NotImplementedError

def build_accuracy_dial(
    mc_results: pd.DataFrame,
    *,
    accuracy_col: str = "normal_accuracy_10",
    band_width: float = 0.02,
    step: float = 0.01,
    min_configs: int = 10,
    param_bounds: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    raise NotImplementedError

def dial_to_config(
    dial_df: pd.DataFrame,
    target_accuracy: float,
    *,
    prefer: str = "balanced",
    accuracy_col: str = "normal_accuracy_10",
    param_bounds: dict[str, tuple[float, float]] | None = None,
) -> MonitorConfig:
    raise NotImplementedError
```

Create `stars_pipeline/training/cli.py` stub:
```python
from __future__ import annotations
import sys

def main(argv=None) -> int:
    raise NotImplementedError

if __name__ == "__main__":
    sys.exit(main())
```

Create `tests/training/__init__.py` (empty file).

- [ ] **Step 3: Verify import works**

```bash
uv run python -c "from stars_pipeline.training import run_mc_search; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml stars_pipeline/training/ tests/training/__init__.py
git commit -m "feat: scaffold stars_pipeline/training package"
```

---

## Task 2: Implement `run_mc_search` and Pareto helpers

**Files:**
- Modify: `stars_pipeline/training/mc_search.py`
- Create: `tests/training/test_mc_search.py`

- [ ] **Step 1: Write failing tests**

Create `tests/training/test_mc_search.py`:
```python
import numpy as np
import pandas as pd
import pytest
from stars_pipeline.config import MonitorConfig
from stars_pipeline.training.mc_search import run_mc_search, MC_PARAM_BOUNDS


def _make_stats_df(n: int = 30, seed: int = 0) -> pd.DataFrame:
    """Minimal wide stats DataFrame (just the threshold columns mc_search applies)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feature_segment":          [f"seg_{i}" for i in range(n)],
        "strata_id":                ["84"] * n,
        "entity_id":                [f"E{i}" for i in range(n)],
        "patient_type_rollup":      ["Inpatient"] * n,
        "service_line":             ["Cardiology"] * n,
        "ks_distribution_value":    rng.uniform(0, 0.6, n),
        "ks_distribution_flag":     rng.integers(0, 2, n),
        "level_shift_value":        rng.uniform(0, 3, n),
        "level_shift_flag":         rng.integers(0, 2, n),
        "dw_shift_value":           rng.uniform(0, 2, n),
        "dw_shift_flag":            rng.integers(0, 2, n),
        "trend_change_value":       rng.uniform(0, 0.4, n),
        "trend_change_flag":        rng.integers(0, 2, n),
        "trend_change__slope_change_ratio": rng.uniform(0, 3, n),
        "stationarity_value":       rng.uniform(0, 1, n),
        "stationarity_flag":        rng.integers(0, 2, n),
        "coverage_shift_value":     rng.uniform(0, 0.5, n),
        "coverage_shift_flag":      rng.integers(0, 2, n),
        "sparsity_change_value":    rng.uniform(0, 0.4, n),
        "sparsity_change_flag":     rng.integers(0, 2, n),
        "low_volume_value":         rng.uniform(0, 20, n),
        "low_volume_flag":          rng.integers(0, 2, n),
        "volatility_shift_value":   rng.uniform(0.5, 5, n),
        "volatility_shift_flag":    rng.integers(0, 2, n),
        "outlier_rate_value":       rng.uniform(0, 0.5, n),
        "outlier_rate_flag":        rng.integers(0, 2, n),
        "acf_structure_value":      rng.uniform(0, 0.3, n),
        "acf_structure_flag":       rng.integers(0, 2, n),
        "is_flagged":               rng.integers(0, 2, n),
        "stability_violations":     rng.integers(0, 5, n),
        "truthfulness_violations":  rng.integers(0, 2, n),
        "abundance_violations":     rng.integers(0, 1, n),
        "regularity_violations":    rng.integers(0, 3, n),
    })


def _make_segment_df(n: int = 30, seed: int = 0) -> pd.DataFrame:
    """Segment-level DataFrame with MESH labels."""
    rng = np.random.default_rng(seed)
    mesh = rng.uniform(0.02, 0.25, n)
    return pd.DataFrame({
        "feature_segment": [f"seg_{i}" for i in range(n)],
        "mesh": mesh,
        "within_10": (mesh <= 0.10).astype(int),
        "within_5":  (mesh <= 0.05).astype(int),
        "within_3":  (mesh <= 0.03).astype(int),
    })


def test_run_mc_search_returns_dataframe():
    stats_df   = _make_stats_df()
    segment_df = _make_segment_df()
    result = run_mc_search(stats_df, segment_df, MonitorConfig(), n_samples=20, rng_seed=0)
    assert isinstance(result, pd.DataFrame)


def test_run_mc_search_returns_n_rows():
    stats_df   = _make_stats_df()
    segment_df = _make_segment_df()
    result = run_mc_search(stats_df, segment_df, MonitorConfig(), n_samples=20, rng_seed=0)
    assert len(result) == 20


def test_run_mc_search_has_required_columns():
    stats_df   = _make_stats_df()
    segment_df = _make_segment_df()
    result = run_mc_search(stats_df, segment_df, MonitorConfig(), n_samples=20, rng_seed=0)
    for col in ["pct_normal", "normal_accuracy_10", "overall_accuracy_10",
                "pareto_front", "pareto_near"]:
        assert col in result.columns, f"missing {col}"


def test_run_mc_search_pct_normal_in_range():
    stats_df   = _make_stats_df()
    segment_df = _make_segment_df()
    result = run_mc_search(stats_df, segment_df, MonitorConfig(), n_samples=20, rng_seed=0)
    assert (result["pct_normal"].dropna().between(0, 1)).all()


def test_run_mc_search_reproducible():
    stats_df   = _make_stats_df()
    segment_df = _make_segment_df()
    r1 = run_mc_search(stats_df, segment_df, MonitorConfig(), n_samples=10, rng_seed=7)
    r2 = run_mc_search(stats_df, segment_df, MonitorConfig(), n_samples=10, rng_seed=7)
    pd.testing.assert_frame_equal(r1, r2)


def test_run_mc_search_fixed_toggles_are_pinned():
    stats_df   = _make_stats_df()
    segment_df = _make_segment_df()
    result = run_mc_search(
        stats_df, segment_df, MonitorConfig(), n_samples=20, rng_seed=0,
        fixed_toggles={"low_volume_enabled": False},
    )
    assert "use_low_volume_enabled" in result.columns
    assert (result["use_low_volume_enabled"] == False).all()


def test_mc_param_bounds_keys_match_monitor_config():
    from dataclasses import fields
    cfg_fields = {f.name for f in fields(MonitorConfig) if isinstance(f.default, float)}
    for key in MC_PARAM_BOUNDS:
        assert key in cfg_fields, f"MC_PARAM_BOUNDS key '{key}' not in MonitorConfig"
```

- [ ] **Step 2: Run to verify failures**

```bash
uv run pytest tests/training/test_mc_search.py -v
```

Expected: 7 failures with `NotImplementedError`.

- [ ] **Step 3: Implement `mc_search.py`**

Replace `stars_pipeline/training/mc_search.py` with:

```python
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from stars_pipeline.config import MonitorConfig
from stars_pipeline.stars.monitor import apply_thresholds

MC_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "ks_d_threshold":               (0.10, 0.60),
    "level_shift_min_cohen_d":      (0.20, 2.50),
    "dw_delta_threshold":           (0.20, 2.50),
    "slope_change_ratio_threshold": (0.10, 3.00),
    "slope_threshold":              (0.001, 0.10),
    "kpss_alpha":                   (0.001, 0.20),
    "trend_p_value_threshold":      (0.001, 0.30),
    "coverage_delta_threshold":     (0.005, 0.60),
    "sparsity_delta_threshold":     (0.01, 0.60),
    "low_volume_monthly_threshold": (0.0,  15.0),
    "volatility_ratio_threshold":   (1.10, 8.00),
    "outlier_z_threshold":          (1.50, 6.00),
    "outlier_rate_threshold":       (0.01, 0.60),
    "acf_divergence_p_threshold":   (0.001, 0.20),
    "alpha":                        (0.001, 0.20),
}

_TOGGLEABLE: tuple[str, ...] = ("low_volume_enabled", "volatility_shift_enabled")


def _pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return boolean mask of Pareto-optimal points (maximize both x and y)."""
    n = len(x)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if x[j] >= x[i] and y[j] >= y[i] and (x[j] > x[i] or y[j] > y[i]):
                dominated[i] = True
                break
    return ~dominated


def _pareto_neighborhood(
    x: np.ndarray, y: np.ndarray, tol: float = 0.02
) -> np.ndarray:
    """Return boolean mask of points within ``tol`` of the Pareto front."""
    front_mask = _pareto_front(x, y)
    front_x = x[front_mask]
    front_y = y[front_mask]
    near = np.zeros(len(x), dtype=bool)
    for i in range(len(x)):
        if any(
            abs(x[i] - fx) <= tol and abs(y[i] - fy) <= tol
            for fx, fy in zip(front_x, front_y)
        ):
            near[i] = True
    return near


def run_mc_search(
    stats_df: pd.DataFrame,
    segment_df: pd.DataFrame,
    base_cfg: MonitorConfig,
    *,
    n_samples: int = 500,
    param_bounds: dict[str, tuple[float, float]] | None = None,
    rng_seed: int = 42,
    fixed_toggles: dict[str, bool] | None = None,
) -> pd.DataFrame:
    """
    Monte Carlo search over STARS threshold parameter space.

    For each random sample: draw threshold values → re-classify all segments via
    apply_thresholds() (vectorised, no per-series re-computation) → measure the
    accuracy/retention tradeoff.

    Args:
        stats_df:      Wide stats DataFrame from run_monitoring() — one row per segment.
                       Must contain all *_flag and *_violations columns.
        segment_df:    One row per segment with columns: feature_segment, within_10,
                       within_5, within_3 (binary labels: mesh <= threshold).
        base_cfg:      MonitorConfig used as the base (non-varied settings preserved).
        n_samples:     Number of random configurations to evaluate.
        param_bounds:  Dict of param_name -> (lo, hi). Defaults to MC_PARAM_BOUNDS.
        rng_seed:      Reproducibility seed.
        fixed_toggles: Dict of toggle_name -> bool. Keys pinned; others randomly sampled.
                       E.g. {"low_volume_enabled": False} always disables low_volume.

    Returns:
        DataFrame with one row per sample. Columns include all sampled parameters,
        use_* toggle columns, and result metrics:
            pct_normal          Fraction of segments classified as normal
            n_normal            Count of normal segments
            n_abnormal          Count of abnormal segments
            normal_accuracy_10  Mean within_10 among normal segments
            normal_accuracy_5   Mean within_5 among normal segments
            normal_accuracy_3   Mean within_3 among normal segments
            abnormal_accuracy_10 Mean within_10 among abnormal segments
            overall_accuracy_10  Mean within_10 across all segments
            pareto_front        True if Pareto-optimal (maximize pct_normal & accuracy)
            pareto_near         True if within 2% tolerance of the Pareto front
    """
    if param_bounds is None:
        param_bounds = MC_PARAM_BOUNDS
    if fixed_toggles is None:
        fixed_toggles = {}

    free_toggles = [t for t in _TOGGLEABLE if t not in fixed_toggles]
    base_fields = {f.name for f in dataclasses.fields(MonitorConfig)}

    rng = np.random.default_rng(rng_seed)
    n_total = len(segment_df)
    rows: list[dict] = []

    for i in range(n_samples):
        sampled = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in param_bounds.items()}
        toggles = {**fixed_toggles, **{t: bool(rng.integers(0, 2)) for t in free_toggles}}

        # Build a fresh MonitorConfig with sampled float params + toggle values
        cfg_kwargs = {k: v for k, v in sampled.items() if k in base_fields}
        cfg_kwargs.update({k: v for k, v in toggles.items() if k in base_fields})
        cfg_i = dataclasses.replace(base_cfg, **cfg_kwargs)

        classified = apply_thresholds(stats_df, cfg=cfg_i)
        normal_mask = ~classified["is_flagged"].astype(bool)

        seg_i = segment_df.merge(
            classified[["feature_segment", "is_flagged"]],
            on="feature_segment",
            how="left",
        )
        seg_i["is_flagged"] = seg_i["is_flagged"].fillna(True)
        seg_normal   = ~seg_i["is_flagged"].astype(bool)
        seg_abnormal =  seg_i["is_flagged"].astype(bool)

        row: dict = {
            "sample_id": i,
            **sampled,
            **{f"use_{t}": v for t, v in toggles.items()},
        }
        row["pct_normal"]   = float(seg_normal.sum()) / n_total
        row["n_normal"]     = int(seg_normal.sum())
        row["n_abnormal"]   = int(seg_abnormal.sum())
        row["normal_accuracy_10"]   = float(seg_i.loc[seg_normal,   "within_10"].mean()) if seg_normal.any()   else float("nan")
        row["normal_accuracy_5"]    = float(seg_i.loc[seg_normal,   "within_5"].mean())  if seg_normal.any()   else float("nan")
        row["normal_accuracy_3"]    = float(seg_i.loc[seg_normal,   "within_3"].mean())  if seg_normal.any()   else float("nan")
        row["abnormal_accuracy_10"] = float(seg_i.loc[seg_abnormal, "within_10"].mean()) if seg_abnormal.any() else float("nan")
        row["overall_accuracy_10"]  = float(seg_i["within_10"].mean())
        rows.append(row)

    result = pd.DataFrame(rows)

    valid_mask = result["pct_normal"].notna() & result["normal_accuracy_10"].notna()
    result["pareto_front"] = False
    result["pareto_near"]  = False

    if valid_mask.any():
        x_vals   = result.loc[valid_mask, "pct_normal"].values
        y_vals   = result.loc[valid_mask, "normal_accuracy_10"].values
        valid_idx = result[valid_mask].index
        result.loc[valid_idx[_pareto_front(x_vals, y_vals)],        "pareto_front"] = True
        result.loc[valid_idx[_pareto_neighborhood(x_vals, y_vals)], "pareto_near"]  = True

    return result
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/training/test_mc_search.py -v
```

Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add stars_pipeline/training/mc_search.py tests/training/test_mc_search.py
git commit -m "feat: implement run_mc_search with Pareto front and neighborhood tagging"
```

---

## Task 3: Implement `analyze_pareto_neighborhood` and `build_accuracy_dial`

**Files:**
- Modify: `stars_pipeline/training/pareto.py`
- Create: `tests/training/test_pareto.py`

- [ ] **Step 1: Write failing tests**

Create `tests/training/test_pareto.py`:
```python
import numpy as np
import pandas as pd
import pytest
from stars_pipeline.config import MonitorConfig
from stars_pipeline.training.mc_search import run_mc_search, MC_PARAM_BOUNDS
from stars_pipeline.training.pareto import (
    analyze_pareto_neighborhood,
    build_accuracy_dial,
    dial_to_config,
)
from tests.training.test_mc_search import _make_stats_df, _make_segment_df


@pytest.fixture(scope="module")
def mc_results():
    stats_df   = _make_stats_df(n=60, seed=1)
    segment_df = _make_segment_df(n=60, seed=1)
    return run_mc_search(stats_df, segment_df, MonitorConfig(), n_samples=200, rng_seed=1)


def test_analyze_pareto_neighborhood_returns_dataframe(mc_results):
    result = analyze_pareto_neighborhood(mc_results)
    assert isinstance(result, pd.DataFrame)


def test_analyze_pareto_neighborhood_has_required_columns(mc_results):
    result = analyze_pareto_neighborhood(mc_results)
    for col in ["parameter", "param_type", "elasticity"]:
        assert col in result.columns, f"missing {col}"


def test_analyze_pareto_neighborhood_covers_all_params(mc_results):
    result = analyze_pareto_neighborhood(mc_results, param_bounds=MC_PARAM_BOUNDS)
    float_params = set(result.loc[result["param_type"] == "float", "parameter"])
    for p in MC_PARAM_BOUNDS:
        assert p in float_params, f"MC param '{p}' missing from analysis"


def test_analyze_pareto_neighborhood_sorted_by_elasticity(mc_results):
    result = analyze_pareto_neighborhood(mc_results)
    if len(result) > 1:
        abs_elast = result["elasticity"].abs().dropna()
        assert (abs_elast.values[:-1] >= abs_elast.values[1:]).all()


def test_build_accuracy_dial_returns_dataframe(mc_results):
    dial = build_accuracy_dial(mc_results)
    assert isinstance(dial, pd.DataFrame)


def test_build_accuracy_dial_has_required_columns(mc_results):
    dial = build_accuracy_dial(mc_results)
    for col in ["target_accuracy", "n_configs", "median_pct_normal"]:
        assert col in dial.columns, f"missing {col}"


def test_dial_to_config_returns_monitor_config(mc_results):
    dial = build_accuracy_dial(mc_results, min_configs=1)
    if len(dial) == 0:
        pytest.skip("No dial entries produced with this data size")
    target = float(dial["target_accuracy"].iloc[0])
    cfg = dial_to_config(dial, target_accuracy=target, param_bounds=MC_PARAM_BOUNDS)
    assert isinstance(cfg, MonitorConfig)


def test_dial_to_config_prefer_retention(mc_results):
    dial = build_accuracy_dial(mc_results, min_configs=1)
    if len(dial) == 0:
        pytest.skip("No dial entries produced with this data size")
    target = float(dial["target_accuracy"].iloc[0])
    cfg_ret = dial_to_config(dial, target_accuracy=target, prefer="retention",
                             param_bounds=MC_PARAM_BOUNDS)
    cfg_acc = dial_to_config(dial, target_accuracy=target, prefer="accuracy",
                             param_bounds=MC_PARAM_BOUNDS)
    assert isinstance(cfg_ret, MonitorConfig)
    assert isinstance(cfg_acc, MonitorConfig)
```

- [ ] **Step 2: Run to verify failures**

```bash
uv run pytest tests/training/test_pareto.py -v
```

Expected: failures with `NotImplementedError`.

- [ ] **Step 3: Implement `pareto.py`**

Replace `stars_pipeline/training/pareto.py` with:

```python
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from stars_pipeline.config import MonitorConfig
from stars_pipeline.training.mc_search import MC_PARAM_BOUNDS, _pareto_neighborhood

_TOGGLEABLE_COLS = ("use_low_volume_enabled", "use_volatility_shift_enabled")


def analyze_pareto_neighborhood(
    mc_results: pd.DataFrame,
    *,
    tol: float = 0.02,
    accuracy_col: str = "normal_accuracy_10",
    param_bounds: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """
    Elasticity and optimization guidance for the Pareto neighborhood.

    For float thresholds: reports the quantile range, Pearson r, and OLS
    elasticity (fractional accuracy change when sweeping end-to-end) within the
    Pareto neighborhood.

    For toggle parameters (use_*): reports fraction ON vs OFF in the neighborhood,
    accuracy difference, and a recommended setting.

    Results are sorted by |elasticity| descending so the biggest levers appear first.

    Args:
        mc_results:   Output of run_mc_search().
        tol:          Pareto neighborhood tolerance (2% on both axes).
        accuracy_col: Accuracy column to analyse.
        param_bounds: Float threshold ranges for elasticity normalisation.

    Returns:
        DataFrame with columns: parameter, param_type, q25_near, median_near,
        q75_near, std_near, pct_on_near, pct_on_all, acc_when_on, acc_when_off,
        acc_delta, recommended, pearson_r, elasticity.
    """
    if param_bounds is None:
        param_bounds = MC_PARAM_BOUNDS

    valid = mc_results.dropna(subset=["pct_normal", accuracy_col]).copy()

    if "pareto_near" in valid.columns:
        near = valid[valid["pareto_near"]]
    else:
        mask = _pareto_neighborhood(
            valid["pct_normal"].values, valid[accuracy_col].values, tol=tol
        )
        near = valid[mask]

    if len(near) < 5:
        return pd.DataFrame(columns=[
            "parameter", "param_type", "q25_near", "median_near", "q75_near",
            "std_near", "pct_on_near", "pct_on_all", "acc_when_on", "acc_when_off",
            "acc_delta", "recommended", "pearson_r", "elasticity",
        ])

    mean_acc = float(near[accuracy_col].mean())
    toggle_cols = [c for c in valid.columns if c.startswith("use_")]
    float_params = [p for p in param_bounds if p in valid.columns]

    rows: list[dict] = []

    for p in float_params:
        col = near[p].astype(float)
        p_range = param_bounds[p][1] - param_bounds[p][0]
        x = col.values
        acc = near[accuracy_col].values

        r = float(np.corrcoef(x, acc)[0, 1]) if len(x) > 2 else float("nan")
        slope = float(np.polyfit(x, acc, 1)[0]) if p_range > 0 and len(x) >= 3 else float("nan")
        elasticity = slope * p_range / mean_acc if np.isfinite(slope) and mean_acc > 0 else float("nan")

        rows.append({
            "parameter":    p,
            "param_type":   "float",
            "q25_near":     round(float(col.quantile(0.25)), 4),
            "median_near":  round(float(col.median()), 4),
            "q75_near":     round(float(col.quantile(0.75)), 4),
            "std_near":     round(float(col.std()), 4),
            "pct_on_near":  float("nan"),
            "pct_on_all":   float("nan"),
            "acc_when_on":  float("nan"),
            "acc_when_off": float("nan"),
            "acc_delta":    float("nan"),
            "recommended":  None,
            "pearson_r":    round(r, 4) if np.isfinite(r) else float("nan"),
            "elasticity":   round(elasticity, 4) if np.isfinite(elasticity) else float("nan"),
        })

    for p in toggle_cols:
        if p not in near.columns:
            continue
        col_near = near[p].astype(float)
        col_all  = valid[p].astype(float)

        pct_on_near = float(col_near.mean())
        pct_on_all  = float(col_all.mean())
        on_mask  = col_near == 1
        off_mask = col_near == 0
        acc_on  = float(near.loc[on_mask,  accuracy_col].mean()) if on_mask.any()  else float("nan")
        acc_off = float(near.loc[off_mask, accuracy_col].mean()) if off_mask.any() else float("nan")
        acc_delta = (acc_on - acc_off) if (np.isfinite(acc_on) and np.isfinite(acc_off)) else float("nan")

        if np.isfinite(acc_delta):
            rec = "ON" if (pct_on_near > 0.5 and acc_delta >= 0) or (acc_delta > 0 and pct_on_near > 0.3) else "OFF"
        else:
            rec = "ON" if pct_on_near >= 0.5 else "OFF"

        x_vals  = col_near.values
        acc_vals = near[accuracy_col].values
        r = float(np.corrcoef(x_vals, acc_vals)[0, 1]) if len(x_vals) > 2 and col_near.nunique() > 1 else float("nan")
        elasticity = acc_delta / mean_acc if np.isfinite(acc_delta) and mean_acc > 0 else float("nan")

        rows.append({
            "parameter":    p,
            "param_type":   "toggle",
            "q25_near":     float("nan"),
            "median_near":  float("nan"),
            "q75_near":     float("nan"),
            "std_near":     float("nan"),
            "pct_on_near":  round(pct_on_near, 3),
            "pct_on_all":   round(pct_on_all, 3),
            "acc_when_on":  round(acc_on, 4)  if np.isfinite(acc_on)  else float("nan"),
            "acc_when_off": round(acc_off, 4) if np.isfinite(acc_off) else float("nan"),
            "acc_delta":    round(acc_delta, 4) if np.isfinite(acc_delta) else float("nan"),
            "recommended":  rec,
            "pearson_r":    round(r, 4) if np.isfinite(r) else float("nan"),
            "elasticity":   round(elasticity, 4) if np.isfinite(elasticity) else float("nan"),
        })

    result = (
        pd.DataFrame(rows)
        .sort_values("elasticity", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )
    return result


def build_accuracy_dial(
    mc_results: pd.DataFrame,
    *,
    accuracy_col: str = "normal_accuracy_10",
    band_width: float = 0.02,
    step: float = 0.01,
    min_configs: int = 10,
    param_bounds: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """
    Distil MC results into a lookup table: target accuracy → expected retention range.

    For each accuracy level (spaced by ``step``), selects all MC samples within
    ±band_width of that level and summarises their retention statistics.

    Args:
        mc_results:  Output of run_mc_search().
        accuracy_col: Accuracy metric to target.
        band_width:   Half-width of the accuracy band (±).
        step:         Step size between target accuracy levels.
        min_configs:  Minimum MC samples required to emit a row.
        param_bounds: Float threshold ranges for parameter median computation.

    Returns:
        DataFrame with one row per target accuracy level. Columns include:
            target_accuracy     The target accuracy level
            n_configs           Number of MC samples in the accuracy band
            median_pct_normal   Median retention in the band
            p10_pct_normal      10th percentile retention
            p90_pct_normal      90th percentile retention
            Plus one column per float threshold with the median value in the band.
    """
    if param_bounds is None:
        param_bounds = MC_PARAM_BOUNDS

    valid = mc_results.dropna(subset=["pct_normal", accuracy_col]).copy()
    float_params = [p for p in param_bounds if p in valid.columns]

    lo = float(valid[accuracy_col].quantile(0.05))
    hi = float(valid[accuracy_col].quantile(0.95))

    levels = np.arange(lo, hi + step / 2, step)
    rows: list[dict] = []

    for target in levels:
        band = valid[(valid[accuracy_col] >= target - band_width) &
                     (valid[accuracy_col] <= target + band_width)]
        if len(band) < min_configs:
            continue
        row: dict = {
            "target_accuracy":   round(float(target), 4),
            "n_configs":         len(band),
            "median_pct_normal": round(float(band["pct_normal"].median()), 4),
            "p10_pct_normal":    round(float(band["pct_normal"].quantile(0.10)), 4),
            "p90_pct_normal":    round(float(band["pct_normal"].quantile(0.90)), 4),
        }
        for p in float_params:
            row[f"median_{p}"] = round(float(band[p].median()), 4)
        rows.append(row)

    return pd.DataFrame(rows)


def dial_to_config(
    dial_df: pd.DataFrame,
    target_accuracy: float,
    *,
    prefer: str = "balanced",
    accuracy_col: str = "normal_accuracy_10",
    param_bounds: dict[str, tuple[float, float]] | None = None,
) -> MonitorConfig:
    """
    Retrieve a MonitorConfig for a target accuracy from the accuracy dial.

    Args:
        dial_df:         Output of build_accuracy_dial().
        target_accuracy: Desired normal_accuracy_10 (or other accuracy_col) level.
        prefer:          "balanced" → row nearest to target; "retention" → row with
                         highest median_pct_normal; "accuracy" → row with highest
                         target_accuracy.
        accuracy_col:    Column name used to match target (informational only).
        param_bounds:    Used to extract float param column names.

    Returns:
        MonitorConfig with threshold values taken from the matched dial row.
    """
    if param_bounds is None:
        param_bounds = MC_PARAM_BOUNDS

    if dial_df.empty:
        return MonitorConfig()

    if prefer == "retention":
        row = dial_df.loc[dial_df["median_pct_normal"].idxmax()]
    elif prefer == "accuracy":
        row = dial_df.loc[dial_df["target_accuracy"].idxmax()]
    else:
        idx = (dial_df["target_accuracy"] - target_accuracy).abs().idxmin()
        row = dial_df.loc[idx]

    cfg_fields = {f.name for f in dataclasses.fields(MonitorConfig)}
    kwargs: dict = {}
    for p in param_bounds:
        col = f"median_{p}"
        if col in row.index and p in cfg_fields:
            val = row[col]
            if pd.notna(val):
                kwargs[p] = float(val)

    return dataclasses.replace(MonitorConfig(), **kwargs)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/training/test_pareto.py -v
```

Expected: all tests pass (some may skip if dial is empty with small dataset — that's correct behavior).

- [ ] **Step 5: Commit**

```bash
git add stars_pipeline/training/pareto.py tests/training/test_pareto.py
git commit -m "feat: implement analyze_pareto_neighborhood, build_accuracy_dial, dial_to_config"
```

---

## Task 4: Implement `stars-train` CLI

**Files:**
- Modify: `stars_pipeline/training/cli.py`
- Create: `tests/training/test_cli.py`

The CLI reads:
- `--stats-csv`: wide stats CSV (output of `stars-pipeline`... pivoted back to wide via `long_to_wide`, OR saved separately). The simplest path: caller supplies the wide format directly. We support reading a long-format CSV and converting internally.
- `--segment-csv`: CSV with columns `feature_segment, mesh` (or `feature_segment, within_10, within_5, within_3` directly).
- `--n-samples`: number of MC samples (default 10000).
- `--output-dir`: where to write `mc_results.csv`, `pareto_analysis.csv`, `accuracy_dial.csv`.

- [ ] **Step 1: Write failing test**

Create `tests/training/test_cli.py`:
```python
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from tests.training.test_mc_search import _make_stats_df, _make_segment_df
from stars_pipeline.training.cli import main


def _write_wide_csv(tmp_path: Path) -> Path:
    df = _make_stats_df(n=40, seed=5)
    p = tmp_path / "stats.csv"
    df.to_csv(p, index=False)
    return p


def _write_segment_csv(tmp_path: Path) -> Path:
    df = _make_segment_df(n=40, seed=5)
    p = tmp_path / "segments.csv"
    df.to_csv(p, index=False)
    return p


def test_cli_produces_output_files(tmp_path):
    stats_csv   = _write_wide_csv(tmp_path)
    segment_csv = _write_segment_csv(tmp_path)
    out_dir = tmp_path / "training_out"
    rc = main([
        "--stats-csv",   str(stats_csv),
        "--segment-csv", str(segment_csv),
        "--n-samples",   "50",
        "--output-dir",  str(out_dir),
    ])
    assert rc == 0
    assert (out_dir / "mc_results.csv").exists()
    assert (out_dir / "pareto_analysis.csv").exists()
    assert (out_dir / "accuracy_dial.csv").exists()


def test_cli_missing_stats_file_returns_nonzero(tmp_path):
    segment_csv = _write_segment_csv(tmp_path)
    rc = main([
        "--stats-csv",   str(tmp_path / "missing.csv"),
        "--segment-csv", str(segment_csv),
        "--output-dir",  str(tmp_path / "out"),
    ])
    assert rc != 0
```

- [ ] **Step 2: Run to verify failures**

```bash
uv run pytest tests/training/test_cli.py -v
```

Expected: 2 failures (NotImplementedError).

- [ ] **Step 3: Implement `cli.py`**

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stars-train",
        description=(
            "Monte Carlo threshold calibration for the STARS pipeline. "
            "Reads pre-computed wide stats and segment-level MESH labels, "
            "searches the threshold parameter space, and writes Pareto analysis "
            "and accuracy dial CSVs."
        ),
    )
    p.add_argument("--stats-csv",   required=True,
                   help="Wide stats CSV (output of run_monitoring via stars-pipeline or stars_pipeline.training).")
    p.add_argument("--segment-csv", required=True,
                   help="Segment CSV with columns: feature_segment, within_10, within_5, within_3 (or mesh).")
    p.add_argument("--n-samples",   type=int, default=10_000,
                   help="Number of MC samples (default: 10000).")
    p.add_argument("--rng-seed",    type=int, default=42,
                   help="Random seed for reproducibility (default: 42).")
    p.add_argument("--output-dir",  required=True,
                   help="Directory to write mc_results.csv, pareto_analysis.csv, accuracy_dial.csv.")
    return p


def main(argv: list[str] | None = None) -> int:
    import pandas as pd
    from stars_pipeline.config import MonitorConfig
    from stars_pipeline.training.mc_search import run_mc_search
    from stars_pipeline.training.pareto import analyze_pareto_neighborhood, build_accuracy_dial

    parser = _build_parser()
    args = parser.parse_args(argv)

    stats_path   = Path(args.stats_csv)
    segment_path = Path(args.segment_csv)

    if not stats_path.exists():
        print(f"ERROR: stats CSV not found: {stats_path}", file=sys.stderr)
        return 1
    if not segment_path.exists():
        print(f"ERROR: segment CSV not found: {segment_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_df   = pd.read_csv(stats_path)
    segment_df = pd.read_csv(segment_path)

    # Derive within_* labels from mesh if not already present
    if "within_10" not in segment_df.columns and "mesh" in segment_df.columns:
        segment_df["within_10"] = (segment_df["mesh"] <= 0.10).astype(int)
        segment_df["within_5"]  = (segment_df["mesh"] <= 0.05).astype(int)
        segment_df["within_3"]  = (segment_df["mesh"] <= 0.03).astype(int)

    # Ensure is_flagged column exists (if stats CSV came from wide format)
    if "is_flagged" not in stats_df.columns:
        from stars_pipeline.stars.monitor import apply_thresholds
        stats_df = apply_thresholds(stats_df)

    print(f"[info] Running MC search: {args.n_samples} samples, {len(stats_df)} segments...")
    mc_results = run_mc_search(
        stats_df, segment_df, MonitorConfig(),
        n_samples=args.n_samples,
        rng_seed=args.rng_seed,
    )
    mc_path = out_dir / "mc_results.csv"
    mc_results.to_csv(mc_path, index=False)
    print(f"Written: {mc_path}")

    pareto_df = analyze_pareto_neighborhood(mc_results)
    pareto_path = out_dir / "pareto_analysis.csv"
    pareto_df.to_csv(pareto_path, index=False)
    print(f"Written: {pareto_path}")

    dial_df = build_accuracy_dial(mc_results)
    dial_path = out_dir / "accuracy_dial.csv"
    dial_df.to_csv(dial_path, index=False)
    print(f"Written: {dial_path}")

    n_front = int(mc_results["pareto_front"].sum())
    print(f"[info] Pareto front: {n_front} configs")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/training/test_cli.py -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add stars_pipeline/training/cli.py tests/training/test_cli.py
git commit -m "feat: implement stars-train CLI for Monte Carlo threshold calibration"
```

---

## Self-Review

**Spec coverage:**
- ✅ `run_mc_search()` — random threshold search with vectorised `apply_thresholds()`, Pareto tagging
- ✅ `_pareto_front()` and `_pareto_neighborhood()` helpers
- ✅ `MC_PARAM_BOUNDS` with all 15 float threshold parameters
- ✅ Toggle parameters: `low_volume_enabled`, `volatility_shift_enabled` (the only two in new MonitorConfig)
- ✅ `fixed_toggles` to pin selected toggles while searching others
- ✅ `analyze_pareto_neighborhood()` — elasticity, Pearson r, float quantiles, toggle ON/OFF analysis
- ✅ `build_accuracy_dial()` — target accuracy → retention range lookup
- ✅ `dial_to_config()` — dial row → MonitorConfig with prefer=balanced/retention/accuracy
- ✅ `stars-train` CLI entry point
- ✅ `is_flagged` used throughout (not `is_normal`) — correctly inverted from old pipeline
- ✅ `dataclasses.replace()` used to create per-sample MonitorConfig (frozen dataclass)
- ✅ No new dependencies required

**Placeholder scan:** No placeholders found.

**Type consistency:** `run_mc_search` → `pd.DataFrame`; `analyze_pareto_neighborhood` → `pd.DataFrame`; `build_accuracy_dial` → `pd.DataFrame`; `dial_to_config` → `MonitorConfig`. All consistent across tasks and cross-references.
