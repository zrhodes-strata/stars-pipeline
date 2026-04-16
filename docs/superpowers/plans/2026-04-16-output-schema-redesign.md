# Output Schema Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `is_normal`/`stars_family_violated` with `is_flagged` (with total violation count) and four per-family violation count rows; suppress noisy warnings from KPSS and proportions_ztest.

**Architecture:** Three files change: `monitor.py` replaces two boolean summary columns with five integer/bool columns; `output.py` replaces two summary row blocks with five; `tests.py` wraps two call sites in warning suppressors. Tests for each file are updated in lockstep with the production changes.

**Tech Stack:** Python 3.11+, pandas, numpy, statsmodels, pytest

---

## File Map

| File | Change |
|---|---|
| `stars_pipeline/stars/monitor.py` | Replace `is_normal` + `stars_family_violated` with `is_flagged`, `stability_violations`, `truthfulness_violations`, `abundance_violations`, `regularity_violations` |
| `stars_pipeline/stars/output.py` | Replace 2 summary row blocks with 5; update docstring |
| `stars_pipeline/stars/tests.py` | Suppress `InterpolationWarning` (lines 261–262) and `RuntimeWarning` (lines 343, 384) |
| `tests/stars/test_monitor.py` | Update assertions for new column names/types |
| `tests/stars/test_output.py` | Update row count (15→18), fixture, and summary row assertions |

---

## Task 1: Update `apply_thresholds()` in `monitor.py`

**Files:**
- Modify: `stars_pipeline/stars/monitor.py:188-243`
- Test: `tests/stars/test_monitor.py`

- [ ] **Step 1: Write failing tests**

Replace the two existing tests that reference `is_normal` and `stars_family_violated` and add new ones. Open `tests/stars/test_monitor.py` and replace lines 72–90 with:

```python
def test_apply_thresholds_adds_summary_columns():
    df = _make_df()
    stats = run_monitoring(df, _make_run_cfg(), MonitorConfig())
    result = apply_thresholds(stats)
    for col in (
        "is_flagged",
        "stability_violations",
        "truthfulness_violations",
        "abundance_violations",
        "regularity_violations",
    ):
        assert col in result.columns, f"Missing column: {col}"
    assert "is_normal" not in result.columns
    assert "stars_family_violated" not in result.columns


def test_normal_segment_is_not_flagged():
    cfg = MonitorConfig(slope_change_ratio_threshold=50.0, kpss_alpha=0.04)
    df = _make_df(n_days=400, mean=100.0, std=5.0)
    stats = run_monitoring(df, _make_run_cfg(), cfg)
    result = apply_thresholds(stats)
    assert bool(result["is_flagged"].iloc[0]) is False
    assert int(result["stability_violations"].iloc[0]) == 0
    assert int(result["truthfulness_violations"].iloc[0]) == 0
    assert int(result["abundance_violations"].iloc[0]) == 0
    assert int(result["regularity_violations"].iloc[0]) == 0


def test_violation_counts_match_flags():
    """If stability flags ks_distribution and level_shift, stability_violations==2."""
    df = _make_df()
    stats = run_monitoring(df, _make_run_cfg(), MonitorConfig())
    # Manually force two Stability flags and one Regularity flag
    stats = stats.copy()
    stats["ks_distribution_flag"] = True
    stats["level_shift_flag"] = True
    stats["volatility_shift_flag"] = True
    result = apply_thresholds(stats)
    assert int(result["stability_violations"].iloc[0]) == 2
    assert int(result["regularity_violations"].iloc[0]) == 1
    assert int(result["is_flagged"].iloc[0]) == 1
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/stars/test_monitor.py::test_apply_thresholds_adds_summary_columns tests/stars/test_monitor.py::test_normal_segment_is_not_flagged tests/stars/test_monitor.py::test_violation_counts_match_flags -v
```

Expected: FAIL — `is_flagged` not in columns, `is_normal` still present.

- [ ] **Step 3: Implement new `apply_thresholds()` in `monitor.py`**

Replace lines 188–243 (the full `apply_thresholds` function) with:

```python
def apply_thresholds(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive violation counts and is_flagged from pre-computed flag columns.

    Vectorised — no per-series loops. Designed to be called after run_monitoring().
    Can also be called independently for re-classification (e.g., threshold tuning)
    without re-running the expensive statistical tests.

    Args:
        stats_df: DataFrame produced by run_monitoring(), with one row per
                  segment and boolean flag columns for each indicator.

    Returns:
        The same DataFrame with five additional columns:
            stability_violations (int)
                Number of Stability flags set (0–6).
            truthfulness_violations (int)
                Number of Truthfulness flags set (0–2).
            abundance_violations (int)
                Number of Abundance flags set (0–1).
            regularity_violations (int)
                Number of Regularity flags set (0–4).
            is_flagged (bool)
                True if any family has at least one violation.
    """
    df = stats_df.copy()

    def _count_flags(cols: list[str]) -> pd.Series:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            logger.warning(
                "Expected flag columns missing from stats_df",
                extra={"missing_columns": missing},
            )
        present = [c for c in cols if c in df.columns]
        if not present:
            return pd.Series(0, index=df.index, dtype=int)
        return df[present].fillna(False).astype(bool).sum(axis=1).astype(int)

    df["stability_violations"]    = _count_flags(_STABILITY_FLAGS)
    df["truthfulness_violations"] = _count_flags(_TRUTHFULNESS_FLAGS)
    df["abundance_violations"]    = _count_flags(_ABUNDANCE_FLAGS)
    df["regularity_violations"]   = _count_flags(_REGULARITY_FLAGS)

    df["is_flagged"] = (
        (df["stability_violations"] > 0)
        | (df["truthfulness_violations"] > 0)
        | (df["abundance_violations"] > 0)
        | (df["regularity_violations"] > 0)
    )

    return df
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/stars/test_monitor.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add stars_pipeline/stars/monitor.py tests/stars/test_monitor.py
git commit -m "feat: replace is_normal/stars_family_violated with violation counts"
```

---

## Task 2: Update `to_long_format()` in `output.py`

**Files:**
- Modify: `stars_pipeline/stars/output.py:1-114`
- Test: `tests/stars/test_output.py`

- [ ] **Step 1: Write failing tests**

Replace the full contents of `tests/stars/test_output.py` with:

```python
# tests/stars/test_output.py
import pandas as pd
import pytest
from pathlib import Path

from stars_pipeline.stars.output import to_long_format, write_long_csv


def _make_stats_row(**overrides):
    """Minimal stats DataFrame row — one Normal segment."""
    row = {
        "strata_id": "84",
        "entity_id": "E01",
        "patient_type_rollup": "Inpatient",
        "service_line": "Cardiology",
        "feature_segment": "84|E01|Inpatient|Cardiology",
        "mesh": 2.5,
        "ks_distribution_value": 0.10,   "ks_distribution_flag": False,
        "level_shift_value": 0.50,       "level_shift_flag": False,
        "dw_shift_value": 0.40,          "dw_shift_flag": False,
        "slope_change_ratio_value": 0.8, "slope_change_ratio_flag": False,
        "stationarity_value": 0.15,      "stationarity_flag": False,
        "trend_significance_value": 0.3, "trend_significance_flag": False,
        "coverage_shift_value": 0.02,    "coverage_shift_flag": False,
        "sparsity_change_value": 0.01,   "sparsity_change_flag": False,
        "low_volume_value": 50.0,        "low_volume_flag": False,
        "volatility_shift_value": 1.1,   "volatility_shift_flag": False,
        "outlier_rate_value": 0.02,      "outlier_rate_flag": False,
        "acf_divergence_value": 0.3,     "acf_divergence_flag": False,
        "dow_pattern_shift_value": 0.2,  "dow_pattern_shift_flag": False,
        # New summary columns from apply_thresholds()
        "is_flagged": False,
        "stability_violations": 0,
        "truthfulness_violations": 0,
        "abundance_violations": 0,
        "regularity_violations": 0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_long_format_has_correct_columns():
    stats = _make_stats_row()
    result = to_long_format(stats)
    assert set(result.columns) == {
        "strata_id", "entity_id", "patient_type_rollup", "service_line",
        "feature_segment", "stars_family", "metric_name",
        "metric_value", "metric_flag",
    }


def test_long_format_has_18_rows_per_segment():
    stats = _make_stats_row()
    result = to_long_format(stats)
    # 13 indicators + 5 summary rows
    assert len(result) == 18


def test_long_format_metric_names_include_new_summaries():
    stats = _make_stats_row()
    result = to_long_format(stats)
    names = set(result["metric_name"])
    for name in [
        "ks_distribution", "level_shift", "low_volume",
        "is_flagged",
        "stability_violations",
        "truthfulness_violations",
        "abundance_violations",
        "regularity_violations",
    ]:
        assert name in names, f"Missing metric_name: {name}"
    assert "is_normal" not in names
    assert "stars_family_violated" not in names


def test_is_flagged_row_no_violations():
    stats = _make_stats_row(is_flagged=False, stability_violations=0,
                            truthfulness_violations=0, abundance_violations=0,
                            regularity_violations=0)
    result = to_long_format(stats)
    row = result[result["metric_name"] == "is_flagged"].iloc[0]
    assert int(row["metric_flag"]) == 0
    assert row["metric_value"] == "0"


def test_is_flagged_row_with_violations():
    stats = _make_stats_row(
        is_flagged=True,
        stability_violations=2,
        truthfulness_violations=1,
        abundance_violations=0,
        regularity_violations=0,
        ks_distribution_flag=True,
        level_shift_flag=True,
        coverage_shift_flag=True,
    )
    result = to_long_format(stats)
    row = result[result["metric_name"] == "is_flagged"].iloc[0]
    assert int(row["metric_flag"]) == 1
    assert row["metric_value"] == "3"  # 2 + 1 + 0 + 0


def test_stability_violations_row():
    stats = _make_stats_row(
        stability_violations=3, is_flagged=True,
        ks_distribution_flag=True, level_shift_flag=True, dw_shift_flag=True,
    )
    result = to_long_format(stats)
    row = result[result["metric_name"] == "stability_violations"].iloc[0]
    assert int(row["metric_flag"]) == 1
    assert row["metric_value"] == "3"


def test_zero_violations_family_row_flag_is_0():
    stats = _make_stats_row(regularity_violations=0)
    result = to_long_format(stats)
    row = result[result["metric_name"] == "regularity_violations"].iloc[0]
    assert int(row["metric_flag"]) == 0
    assert row["metric_value"] == "0"


def test_flagged_indicator_produces_metric_flag_1():
    stats = _make_stats_row(ks_distribution_flag=True, ks_distribution_value=0.42)
    result = to_long_format(stats)
    row = result[result["metric_name"] == "ks_distribution"].iloc[0]
    assert int(row["metric_flag"]) == 1
    assert row["metric_value"] == "0.42"


def test_write_long_csv_creates_file(tmp_path):
    stats = _make_stats_row()
    out = tmp_path / "results.csv"
    write_long_csv(stats, out)
    assert out.exists()
    df = pd.read_csv(out)
    assert len(df) == 18
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/stars/test_output.py -v
```

Expected: multiple FAILs — `is_flagged` not in metric names, row count is 15 not 18, etc.

- [ ] **Step 3: Update `output.py`**

Replace the full contents of `stars_pipeline/stars/output.py` with:

```python
"""
output.py
=========
Long-format CSV writer for STARS pipeline results.

Output schema — one row per segment per STARS indicator:

    strata_id           str     Strata identifier
    entity_id           str     Entity identifier
    patient_type_rollup str     Patient type rollup
    service_line        str     Service line name
    feature_segment     str     Concatenated key: strata_id|entity_id|patient_type|service_line
    stars_family        str     STARS family (Stability / Truthfulness / Abundance / Regularity / Summary)
    metric_name         str     Indicator name or summary metric name
    metric_value        str     Raw statistic as string; NULL for binary-only rows
    metric_flag         int     1 = flagged/abnormal, 0 = pass/normal

Five summary rows are appended for each segment (stars_family="Summary"):
    is_flagged                metric_value=total violations (int), metric_flag=1/0
    stability_violations      metric_value=count (0-6),            metric_flag=1/0
    truthfulness_violations   metric_value=count (0-2),            metric_flag=1/0
    abundance_violations      metric_value=count (0-1),            metric_flag=1/0
    regularity_violations     metric_value=count (0-4),            metric_flag=1/0
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from stars_pipeline.logging_config import get_logger

logger = get_logger(__name__)

# Maps metric name → STARS family for the output rows
_METRIC_FAMILY: dict[str, str] = {
    "ks_distribution":     "Stability",
    "level_shift":         "Stability",
    "dw_shift":            "Stability",
    "slope_change_ratio":  "Stability",
    "stationarity":        "Stability",
    "trend_significance":  "Stability",
    "coverage_shift":      "Truthfulness",
    "sparsity_change":     "Truthfulness",
    "low_volume":          "Abundance",
    "volatility_shift":    "Regularity",
    "outlier_rate":        "Regularity",
    "acf_divergence":      "Regularity",
    "dow_pattern_shift":   "Regularity",
}

_ID_COLS = ("strata_id", "entity_id", "patient_type_rollup", "service_line", "feature_segment")

_SUMMARY_METRICS = (
    "stability_violations",
    "truthfulness_violations",
    "abundance_violations",
    "regularity_violations",
)


def to_long_format(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt a wide stats DataFrame (one row per segment) to long format
    (one row per segment per indicator).

    Args:
        stats_df: DataFrame produced by monitor.apply_thresholds(). Must contain
                  columns of the form ``{metric_name}_value`` and ``{metric_name}_flag``
                  for each indicator in _METRIC_FAMILY, plus ``is_flagged``,
                  ``stability_violations``, ``truthfulness_violations``,
                  ``abundance_violations``, and ``regularity_violations``.

    Returns:
        Long-format DataFrame with columns:
            strata_id, entity_id, patient_type_rollup, service_line,
            feature_segment, stars_family, metric_name, metric_value, metric_flag
    """
    rows: list[dict] = []

    for _, stat_row in stats_df.iterrows():
        segment = {col: stat_row[col] for col in _ID_COLS if col in stat_row.index}

        # One row per STARS indicator
        for metric, family in _METRIC_FAMILY.items():
            val_col  = f"{metric}_value"
            flag_col = f"{metric}_flag"
            raw_val  = stat_row.get(val_col)
            raw_flag = stat_row.get(flag_col)

            rows.append({
                **segment,
                "stars_family": family,
                "metric_name":  metric,
                "metric_value": str(raw_val) if pd.notna(raw_val) else None,
                "metric_flag":  int(raw_flag) if pd.notna(raw_flag) else None,
            })

        # Summary: is_flagged — value is total violation count, flag is 1/0
        family_counts = [
            int(stat_row.get(m, 0) or 0) for m in _SUMMARY_METRICS
        ]
        total_violations = sum(family_counts)
        is_flagged = stat_row.get("is_flagged")
        rows.append({
            **segment,
            "stars_family": "Summary",
            "metric_name":  "is_flagged",
            "metric_value": str(total_violations),
            "metric_flag":  int(bool(is_flagged)) if pd.notna(is_flagged) else None,
        })

        # Summary: one row per family violation count
        for metric_name, count in zip(_SUMMARY_METRICS, family_counts):
            rows.append({
                **segment,
                "stars_family": "Summary",
                "metric_name":  metric_name,
                "metric_value": str(count),
                "metric_flag":  1 if count > 0 else 0,
            })

    _OUTPUT_COLS = [
        "strata_id", "entity_id", "patient_type_rollup", "service_line",
        "feature_segment", "stars_family", "metric_name", "metric_value", "metric_flag",
    ]
    if not rows:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.DataFrame(rows)


def write_long_csv(stats_df: pd.DataFrame, output_path: Path) -> None:
    """
    Convert stats_df to long format and write to a CSV file.

    Creates parent directories if they do not exist.

    Args:
        stats_df:    DataFrame produced by monitor.apply_thresholds().
        output_path: Destination path for the CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    long_df = to_long_format(stats_df)
    long_df.to_csv(output_path, index=False)
    logger.info(
        "Long-format CSV written",
        extra={"rows": len(long_df), "path": str(output_path)},
    )
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/stars/test_output.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add stars_pipeline/stars/output.py tests/stars/test_output.py
git commit -m "feat: expand summary rows to per-family violation counts"
```

---

## Task 3: Suppress warnings in `tests.py`

**Files:**
- Modify: `stars_pipeline/stars/tests.py:258-267` (KPSS call site)
- Modify: `stars_pipeline/stars/tests.py:336-344` (coverage_shift call site)
- Modify: `stars_pipeline/stars/tests.py:377-385` (sparsity_change call site)

No new tests needed — the existing test suite will confirm no regressions.

- [ ] **Step 1: Suppress KPSS `InterpolationWarning` at `tests.py:260-263`**

The call site is inside `test_stationarity()`. Find the `try` block that calls `kpss` (around line 260) and wrap those two calls:

```python
    try:
        import warnings
        from statsmodels.tools.sm_exceptions import InterpolationWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            _, p_train, _, _ = kpss(train, regression="c", nlags="auto")
            stat_recent, p_recent, _, _ = kpss(recent, regression="c", nlags="auto")
    except Exception:
        return (float("nan"), False)
```

- [ ] **Step 2: Suppress `RuntimeWarning` in `test_coverage_shift()` around line 343**

Wrap the `proportions_ztest` call:

```python
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, p_value = proportions_ztest(count, nobs)
```

- [ ] **Step 3: Suppress `RuntimeWarning` in `test_sparsity_change()` around line 384**

Wrap the `proportions_ztest` call:

```python
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, p_value = proportions_ztest(count, nobs)
```

- [ ] **Step 4: Run full test suite to confirm no regressions**

```
pytest -v
```

Expected: all tests PASS, no `InterpolationWarning` or `RuntimeWarning` in output.

- [ ] **Step 5: Commit**

```bash
git add stars_pipeline/stars/tests.py
git commit -m "fix: suppress expected KPSS and proportions_ztest warnings"
```

---

## Task 4: Update `cli.py` if it references old column names

**Files:**
- Check and modify if needed: `stars_pipeline/cli.py`

- [ ] **Step 1: Search for references to old column names**

```bash
grep -n "is_normal\|stars_family_violated" stars_pipeline/cli.py
```

If output is empty, skip to Step 3.

- [ ] **Step 2: Replace any references found**

If `is_normal` appears, replace with `is_flagged` (noting the inversion — `is_normal=True` ↔ `is_flagged=False`).
If `stars_family_violated` appears, remove or replace with appropriate per-family column references.

- [ ] **Step 3: Run full test suite**

```
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit if changes were made**

```bash
git add stars_pipeline/cli.py
git commit -m "fix: update cli references to renamed summary columns"
```

---

## Task 5: Smoke test end-to-end

- [ ] **Step 1: Run full test suite one final time**

```
pytest -v
```

Expected: all tests PASS, no warnings about `is_normal` or `stars_family_violated`.

- [ ] **Step 2: Verify CSV output schema**

```bash
export SNOWFLAKE_CONNECTION_NAME=my_example_connection
.venv/Scripts/stars-pipeline --strata-ids 1921 --date-from 2025-01-01 --output ./smoke_test.csv
python -c "
import pandas as pd
df = pd.read_csv('smoke_test.csv')
summary = df[df['stars_family'] == 'Summary']
print(summary[['metric_name','metric_value','metric_flag']].drop_duplicates('metric_name').to_string())
assert set(summary['metric_name'].unique()) == {'is_flagged','stability_violations','truthfulness_violations','abundance_violations','regularity_violations'}, 'Unexpected metric names'
print('Schema OK')
"
```

Expected output: table showing 5 summary metric names with integer `metric_value` and 0/1 `metric_flag`. Final line: `Schema OK`.

- [ ] **Step 3: Remove smoke test CSV**

```bash
rm smoke_test.csv
```
