# Run Selection Modes Design

**Date:** 2026-04-20  
**Status:** Approved

---

## Problem

The pipeline currently accepts `--collection-id` as a direct pass-through to filter the volume query, but the WHERE clause is commented out (pending schema confirmation). There is no way to say "give me today's runs" or "give me the most recent runs" without knowing the `pipeline_run_id` in advance.

The goal is to add three run-selection modes that automatically resolve the correct `pipeline_run_id` from `dagster_run_details` before executing the main query — while keeping `--collection-id` as a direct override for known IDs.

---

## New CLI Arguments

Added alongside the existing `--collection-id` and `--run-id`:

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--run-mode` | `today` \| `most-recent` \| `date-range` | `today` | Mutually exclusive with `--collection-id` |
| `--run-mode-date` | `YYYY-MM-DD` | None | Single-day shorthand for `date-range`; normalized to `date-from == date-to` |
| `--run-mode-date-from` | `YYYY-MM-DD` | None | Start of range for `date-range` mode |
| `--run-mode-date-to` | `YYYY-MM-DD` | None | End of range for `date-range` mode |

**Validation at parse time:**
- `--run-mode` and `--collection-id` are mutually exclusive
- `date-range` requires either `--run-mode-date` or both `--run-mode-date-from` and `--run-mode-date-to`
- `--run-mode-date`, `--run-mode-date-from`, `--run-mode-date-to` are only valid when `--run-mode date-range`

---

## RunConfig Changes

Three new fields:

```python
run_mode: str | None          # "today" | "most-recent" | "date-range" | None
run_mode_date_from: date | None   # date-range start (also set for single --run-mode-date)
run_mode_date_to: date | None     # date-range end   (also set for single --run-mode-date)
```

`--run-mode-date <D>` normalizes to `run_mode_date_from = run_mode_date_to = D` at parse time. No separate `run_mode_date` field on RunConfig.

---

## Resolution Logic (`db.py`)

A new `_resolve_collection_id(run_cfg, conn)` function is called inside `fetch_actuals()` when `run_cfg.collection_id is None`.

### Source table

```sql
datalake_sandbox.public_volume_predictions.dagster_run_details
```

### Resolution query (all modes)

```sql
SELECT
    strata_id,
    PARSE_JSON(metadata):tags:cliententityid::INT AS entity_id,
    pipeline_run_id
FROM dagster_run_details
WHERE <mode_filter>
```

Mode filters:
- **`today`**: `DATE(created_at) = CURRENT_DATE`
- **`most-recent`**: `DATE(created_at) = (SELECT MAX(DATE(created_at)) FROM dagster_run_details)`
- **`date-range`**: `DATE(created_at) BETWEEN %(date_from)s AND %(date_to)s`

The query also restricts to `strata_id IN ({strata_ids})` to limit scope to the requested strata.

### "One result" definition

For each unique `(strata_id, entity_id)` pair: exactly one distinct `pipeline_run_id` is expected.

### Behavior by mode

| Scenario | `today` | `most-recent` | `date-range` |
|---|---|---|---|
| All pairs → exactly 1 result | ✅ proceed | ✅ proceed | ✅ proceed |
| Some pairs → 0, rest → 1 | ✅ skip missing, log warning | ❌ hard error | ✅ skip missing, log warning |
| All pairs → 0 results | ⚠️ fall back to most-recent, log warning | ❌ hard error | ❌ hard error |
| Any pair → 2+ results | ❌ hard error | ❌ hard error | ❌ hard error |

**Today fallback:** If `today` resolves 0 results, re-run resolution using `most-recent` logic. Log: `"No runs found for today ({date}), falling back to most-recent ({fallback_date})"`.

**Hard errors** raise `ValueError` with mode, offending `(strata_id, entity_id)` pairs, and counts found.

### Output

Returns a dict mapping `(strata_id, entity_id)` → `pipeline_run_id` for all successfully resolved pairs. Pairs that were skipped are excluded. The main query is scoped to only the resolved pairs.

---

## SQL Changes (`actuals.sql`)

Uncomment the `collection_id` filter:

```sql
AND v.collection_id = %(collection_id)s
```

Since the query may now run per-`pipeline_run_id` (or with a single resolved ID), the binding is unchanged. The `run_id` filter remains commented out — it is not part of this feature.

---

## Warnings CSV

A second output file is written alongside the main CSV:

**Path:** `<output_stem>_warnings.csv` (e.g., `stars_pilots2_warnings.csv`)

**Columns:**

| Column | Description |
|---|---|
| `warning_type` | `skipped_pair` \| `today_fallback` |
| `strata_id` | Strata identifier |
| `entity_id` | Entity identifier (NULL for `today_fallback` rows) |
| `run_mode` | The mode that was active (`today`, `most-recent`, `date-range`) |
| `requested_date` | The date/range that was requested |
| `fallback_date` | The date used after fallback (NULL if no fallback) |
| `message` | Human-readable description |

The warnings CSV is always written (even if empty, with headers only). This makes it easy to detect clean runs programmatically.

---

## Error and Logging

- Hard errors: `ValueError` with structured message including mode, offending pairs, and counts
- Warnings: `logger.warning(...)` with structured `extra={}` fields (mode, date, pairs) — in addition to the warnings CSV row
- Fallback: `logger.warning(...)` at WARNING level before re-running resolution

---

## Non-Goals

- `--run-id` filter is not wired in this feature (remains a TODO)
- No changes to the statistical test logic or output schema
- No retry logic on Snowflake failures
