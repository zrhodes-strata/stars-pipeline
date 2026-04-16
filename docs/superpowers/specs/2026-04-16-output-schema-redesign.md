# Output Schema Redesign — STARS Summary Metrics

**Date:** 2026-04-16  
**Status:** Approved

---

## Problem

The current long-format output includes two summary rows per segment:

| `metric_name` | `metric_value` | `metric_flag` |
|---|---|---|
| `is_normal` | NULL | 1 (normal) or 0 (flagged) |
| `stars_family_violated` | first violated family name or NULL | NULL |

These have two weaknesses:
1. `is_normal` carries only a binary flag — no count of how many tests failed.
2. `stars_family_violated` reports only the *first* violated family (priority order: Stability > Truthfulness > Abundance > Regularity), hiding violations in other families.

## Goal

Replace those 2 rows with 5 rows that expose violation counts and per-family flags while keeping the long-format structure and `stars_family=Summary` grouping.

---

## New Summary Rows (per segment)

All 5 rows have `stars_family = "Summary"`.

| `metric_name` | `metric_value` | `metric_flag` |
|---|---|---|
| `is_flagged` | total # of failed tests across all families (int) | `1` if ≥1 violation, else `0` |
| `stability_violations` | # failed Stability tests (0–6) | `1` if ≥1, else `0` |
| `truthfulness_violations` | # failed Truthfulness tests (0–2) | `1` if ≥1, else `0` |
| `abundance_violations` | # failed Abundance tests (0–1) | `1` if ≥1, else `0` |
| `regularity_violations` | # failed Regularity tests (0–4) | `1` if ≥1, else `0` |

`metric_value` for `is_flagged` is the sum of all four family violation counts.  
Total rows per segment: 13 indicators + 5 summary = **18 rows**.

---

## Removed Fields

- `is_normal` — replaced by `is_flagged` (inverse) with a violation count
- `stars_family_violated` — replaced by four per-family rows that expose all violations

---

## Affected Files

### `stars_pipeline/stars/monitor.py` — `apply_thresholds()`

Replace the computed columns `is_normal` and `stars_family_violated` with:

- `stability_violations` (int): sum of `_STABILITY_FLAGS`
- `truthfulness_violations` (int): sum of `_TRUTHFULNESS_FLAGS`
- `abundance_violations` (int): sum of `_ABUNDANCE_FLAGS`
- `regularity_violations` (int): sum of `_REGULARITY_FLAGS`
- `is_flagged` (bool): True if any family violation count ≥ 1

### `stars_pipeline/stars/output.py` — `to_long_format()`

Replace the 2 summary row blocks with 5 blocks, one per metric above.

### `stars_pipeline/stars/tests.py` — Warning suppression

Suppress known benign warnings at their call sites using `warnings.catch_warnings`:

1. `InterpolationWarning` from `statsmodels` at lines 261–262 (KPSS p-value table boundary)
2. `RuntimeWarning: invalid value encountered in scalar divide` at lines 343, 384 (`proportions_ztest` with zero variance)

### Tests

- `tests/stars/test_monitor.py`: update assertions for new column names and types
- `tests/stars/test_output.py`: update row count (15 → 18), column assertions, summary row checks

---

## Non-Goals

- No changes to the 13 indicator rows
- No changes to identity columns or CSV structure
- No changes to threshold logic or test statistics
