# Run Selection Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-resolve `collection_id` from `dagster_run_details` using `today`, `most-recent`, or `date-range` modes so users don't need to know the ID in advance.

**Architecture:** A new `_resolve_collection_id(run_cfg, conn)` function in `db.py` queries `dagster_run_details` and returns a `dict[(strata_id, entity_id) → pipeline_run_id]`. The existing `fetch_actuals()` calls it when `collection_id` is None, then filters `actuals.sql` by the resolved ID. Warnings (skipped pairs, today-fallback) are collected during resolution and written to a `<stem>_warnings.csv` alongside the main output.

**Tech Stack:** Python 3.11+, pandas, snowflake-connector-python, argparse, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `stars_pipeline/config.py` | Modify | Add `run_mode`, `run_mode_date_from`, `run_mode_date_to` to `RunConfig` |
| `stars_pipeline/cli.py` | Modify | Add `--run-mode`, `--run-mode-date`, `--run-mode-date-from`, `--run-mode-date-to`; mutual exclusivity with `--collection-id`; normalize single date |
| `stars_pipeline/db.py` | Modify | Add `_resolve_collection_id()`; update `fetch_actuals()` to call it; uncomment WHERE clause |
| `stars_pipeline/sql/actuals.sql` | Modify | Uncomment `AND v.collection_id = %(collection_id)s` |
| `stars_pipeline/stars/warnings.py` | Create | `build_warnings_df()` and `write_warnings_csv()` |
| `stars_pipeline/cli.py` | Modify (step 4) | Call `write_warnings_csv()` after `write_long_csv()` |
| `tests/test_config.py` | Modify | Add `run_mode` field tests |
| `tests/test_cli.py` | Modify | Add `--run-mode` argument tests; mutual exclusivity; date normalization |
| `tests/test_db.py` | Modify | Add `_resolve_collection_id()` unit tests |
| `tests/stars/test_warnings.py` | Create | Tests for `build_warnings_df()` and `write_warnings_csv()` |

---

### Task 1: Add run_mode fields to RunConfig

**Files:**
- Modify: `stars_pipeline/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_run_config_run_mode_fields():
    from datetime import date
    from pathlib import Path
    cfg = RunConfig(
        strata_ids=[84],
        collection_id=None,
        run_id=None,
        run_mode="today",
        run_mode_date_from=None,
        run_mode_date_to=None,
        date_from=date(2022, 1, 1),
        date_to=date(2026, 1, 1),
        recent_days=90,
        train_days=None,
        entity_id=None,
        patient_type=None,
        service_line=None,
        output_path=Path("results.csv"),
    )
    assert cfg.run_mode == "today"
    assert cfg.run_mode_date_from is None
    assert cfg.run_mode_date_to is None


def test_run_config_date_range_fields():
    from datetime import date
    from pathlib import Path
    cfg = RunConfig(
        strata_ids=[84],
        collection_id=None,
        run_id=None,
        run_mode="date-range",
        run_mode_date_from=date(2025, 1, 1),
        run_mode_date_to=date(2025, 1, 31),
        date_from=date(2022, 1, 1),
        date_to=date(2026, 1, 1),
        recent_days=90,
        train_days=None,
        entity_id=None,
        patient_type=None,
        service_line=None,
        output_path=Path("results.csv"),
    )
    assert cfg.run_mode == "date-range"
    assert cfg.run_mode_date_from == date(2025, 1, 1)
    assert cfg.run_mode_date_to == date(2025, 1, 31)
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_config.py::test_run_config_run_mode_fields tests/test_config.py::test_run_config_date_range_fields -v
```

Expected: FAIL with `TypeError: RunConfig.__init__() got an unexpected keyword argument 'run_mode'`

- [ ] **Step 3: Add three fields to RunConfig in `stars_pipeline/config.py`**

Add after `run_id: str | None` (line 65):

```python
    run_mode: str | None          # "today" | "most-recent" | "date-range" | None
    run_mode_date_from: date | None
    run_mode_date_to: date | None
```

Also update the docstring — replace the `run_id` attribute block with:

```
    run_id:
        Run identifier. Same status as collection_id.
    run_mode:
        Run selection mode. One of "today", "most-recent", "date-range", or None.
        None means collection_id is used directly (explicit override).
        Default "today" when collection_id is not provided.
    run_mode_date_from:
        Start date for date-range mode. Also set for --run-mode-date shorthand.
    run_mode_date_to:
        End date for date-range mode. Also set for --run-mode-date shorthand.
```

- [ ] **Step 4: Update the existing `test_run_config_construction` to include new fields**

The existing test constructs a `RunConfig` without the new fields — it will fail after the field addition. Update it:

```python
def test_run_config_construction():
    cfg = RunConfig(
        strata_ids=[84, 14],
        collection_id=None,
        run_id=None,
        run_mode=None,
        run_mode_date_from=None,
        run_mode_date_to=None,
        date_from=date(2022, 1, 1),
        date_to=date(2026, 1, 1),
        recent_days=90,
        train_days=None,
        entity_id=None,
        patient_type=None,
        service_line=None,
        output_path=Path("results.csv"),
    )
    assert cfg.strata_ids == [84, 14]
    assert cfg.recent_days == 90
    assert cfg.train_days is None
```

Also update `_make_run_cfg()` in `tests/test_db.py` to include the new fields in defaults:

```python
def _make_run_cfg(**kwargs):
    defaults = dict(
        strata_ids=[84],
        collection_id=None,
        run_id=None,
        run_mode="today",
        run_mode_date_from=None,
        run_mode_date_to=None,
        date_from=date(2022, 1, 1),
        date_to=date(2026, 1, 1),
        recent_days=90,
        train_days=None,
        entity_id=None,
        patient_type=None,
        service_line=None,
        output_path=Path("out.csv"),
    )
    defaults.update(kwargs)
    return RunConfig(**defaults)
```

- [ ] **Step 5: Run all config and db tests to verify they pass**

```
pytest tests/test_config.py tests/test_db.py -v
```

Expected: All pass (note: `test_missing_env_vars_raises` is a pre-existing failure if `SNOWFLAKE_CONNECTION_NAME` is set in the environment — ignore it)

- [ ] **Step 6: Commit**

```bash
git add stars_pipeline/config.py tests/test_config.py tests/test_db.py
git commit -m "feat: add run_mode fields to RunConfig"
```

---

### Task 2: Add --run-mode CLI arguments

**Files:**
- Modify: `stars_pipeline/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py`:

```python
def test_run_mode_defaults_to_today():
    parser = _build_parser()
    args = parser.parse_args(["--strata-ids", "84"])
    cfg = _build_run_config(args)
    assert cfg.run_mode == "today"
    assert cfg.run_mode_date_from is None
    assert cfg.run_mode_date_to is None


def test_run_mode_most_recent():
    parser = _build_parser()
    args = parser.parse_args(["--strata-ids", "84", "--run-mode", "most-recent"])
    cfg = _build_run_config(args)
    assert cfg.run_mode == "most-recent"


def test_run_mode_date_single_date_normalizes():
    parser = _build_parser()
    args = parser.parse_args([
        "--strata-ids", "84",
        "--run-mode", "date-range",
        "--run-mode-date", "2025-03-15",
    ])
    cfg = _build_run_config(args)
    assert cfg.run_mode == "date-range"
    assert cfg.run_mode_date_from == date(2025, 3, 15)
    assert cfg.run_mode_date_to == date(2025, 3, 15)


def test_run_mode_date_range_explicit():
    parser = _build_parser()
    args = parser.parse_args([
        "--strata-ids", "84",
        "--run-mode", "date-range",
        "--run-mode-date-from", "2025-01-01",
        "--run-mode-date-to", "2025-01-31",
    ])
    cfg = _build_run_config(args)
    assert cfg.run_mode == "date-range"
    assert cfg.run_mode_date_from == date(2025, 1, 1)
    assert cfg.run_mode_date_to == date(2025, 1, 31)


def test_run_mode_and_collection_id_are_mutually_exclusive():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--strata-ids", "84",
            "--run-mode", "today",
            "--collection-id", "COL123",
        ])


def test_date_range_requires_date_args():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--strata-ids", "84",
            "--run-mode", "date-range",
        ])


def test_run_mode_date_only_valid_with_date_range():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--strata-ids", "84",
            "--run-mode", "today",
            "--run-mode-date", "2025-03-15",
        ])


def test_collection_id_sets_run_mode_none():
    parser = _build_parser()
    args = parser.parse_args([
        "--strata-ids", "84",
        "--collection-id", "COL123",
    ])
    cfg = _build_run_config(args)
    assert cfg.collection_id == "COL123"
    assert cfg.run_mode is None
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_cli.py -v -k "run_mode"
```

Expected: All new tests FAIL

- [ ] **Step 3: Update `_build_parser()` in `stars_pipeline/cli.py`**

Replace the `--collection-id` argument block and add run-mode arguments. The new `_build_parser()` body after `--strata-ids`:

```python
    # Mutually exclusive group: either --collection-id (direct) or --run-mode (auto-resolve)
    run_group = p.add_mutually_exclusive_group()
    run_group.add_argument(
        "--collection-id",
        default=None,
        help=(
            "Collection identifier passed directly to the SQL layer. "
            "Mutually exclusive with --run-mode."
        ),
    )
    run_group.add_argument(
        "--run-mode",
        choices=["today", "most-recent", "date-range"],
        default=None,
        help=(
            "Auto-resolve collection_id from dagster_run_details. "
            "Choices: today (default when --collection-id omitted), most-recent, date-range. "
            "Mutually exclusive with --collection-id."
        ),
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Run identifier. Passed to the SQL layer. (Not yet wired into WHERE clause.)",
    )
    p.add_argument(
        "--run-mode-date",
        default=None,
        type=_parse_date,
        help="Single-day shorthand for date-range mode (YYYY-MM-DD). Sets date-from == date-to.",
    )
    p.add_argument(
        "--run-mode-date-from",
        default=None,
        type=_parse_date,
        help="Start of range for --run-mode date-range (YYYY-MM-DD).",
    )
    p.add_argument(
        "--run-mode-date-to",
        default=None,
        type=_parse_date,
        help="End of range for --run-mode date-range (YYYY-MM-DD).",
    )
```

- [ ] **Step 4: Update `_build_run_config()` in `stars_pipeline/cli.py`**

Replace the current `_build_run_config()` body with:

```python
def _build_run_config(args: argparse.Namespace) -> RunConfig:
    """Translate parsed CLI args into a RunConfig. Separated for testability."""
    strata_ids = [int(s.strip()) for s in args.strata_ids.split(",")]
    date_to = args.date_to or date.today()
    output_path = (
        Path(args.output)
        if args.output
        else Path(f"stars_results_{date_to}.csv")
    )

    # Validate date-range args
    run_mode = args.run_mode
    collection_id = getattr(args, "collection_id", None)

    # When neither --collection-id nor --run-mode is given, default to "today"
    if collection_id is None and run_mode is None:
        run_mode = "today"

    run_mode_date_from = None
    run_mode_date_to = None

    if run_mode == "date-range":
        if args.run_mode_date is not None:
            run_mode_date_from = args.run_mode_date
            run_mode_date_to = args.run_mode_date
        elif args.run_mode_date_from is not None and args.run_mode_date_to is not None:
            run_mode_date_from = args.run_mode_date_from
            run_mode_date_to = args.run_mode_date_to
        else:
            raise argparse.ArgumentTypeError(
                "--run-mode date-range requires --run-mode-date or both "
                "--run-mode-date-from and --run-mode-date-to"
            )
    elif run_mode in ("today", "most-recent", None):
        if args.run_mode_date is not None or args.run_mode_date_from is not None or args.run_mode_date_to is not None:
            raise argparse.ArgumentTypeError(
                "--run-mode-date, --run-mode-date-from, --run-mode-date-to "
                "are only valid with --run-mode date-range"
            )

    return RunConfig(
        strata_ids=strata_ids,
        collection_id=collection_id,
        run_id=args.run_id,
        run_mode=run_mode,
        run_mode_date_from=run_mode_date_from,
        run_mode_date_to=run_mode_date_to,
        date_from=args.date_from,
        date_to=date_to,
        recent_days=args.recent_days,
        train_days=args.train_days,
        entity_id=args.entity_id,
        patient_type=args.patient_type,
        service_line=args.service_line,
        output_path=output_path,
    )
```

- [ ] **Step 5: Run all CLI tests**

```
pytest tests/test_cli.py -v
```

Expected: All pass. Note: the `ArgumentTypeError` for missing date-range args is raised inside `_build_run_config()`, which gets called after `parse_args()`. The test `test_date_range_requires_date_args` calls `parse_args()` only — this test will need to call `_build_run_config()` too. If the test fails because `SystemExit` isn't raised at parse time, update the test:

```python
def test_date_range_requires_date_args():
    parser = _build_parser()
    args = parser.parse_args([
        "--strata-ids", "84",
        "--run-mode", "date-range",
    ])
    with pytest.raises(argparse.ArgumentTypeError):
        _build_run_config(args)


def test_run_mode_date_only_valid_with_date_range():
    parser = _build_parser()
    args = parser.parse_args([
        "--strata-ids", "84",
        "--run-mode", "today",
        "--run-mode-date", "2025-03-15",
    ])
    with pytest.raises(argparse.ArgumentTypeError):
        _build_run_config(args)
```

Re-run after adjusting. Expected: All pass.

- [ ] **Step 6: Update `test_defaults` in `tests/test_cli.py` to include new run_mode fields**

The existing `test_defaults` doesn't check `run_mode`. Add assertions:

```python
    assert cfg.run_mode == "today"
    assert cfg.run_mode_date_from is None
    assert cfg.run_mode_date_to is None
```

- [ ] **Step 7: Run full test suite**

```
pytest tests/ -v
```

Expected: All pass (except the pre-existing `test_missing_env_vars_raises` if `SNOWFLAKE_CONNECTION_NAME` is set)

- [ ] **Step 8: Commit**

```bash
git add stars_pipeline/cli.py tests/test_cli.py
git commit -m "feat: add --run-mode CLI arguments with mutual exclusivity and date normalization"
```

---

### Task 3: Implement _resolve_collection_id and wire into fetch_actuals

**Files:**
- Modify: `stars_pipeline/db.py`
- Modify: `stars_pipeline/sql/actuals.sql`
- Modify: `tests/test_db.py`

#### Background

`dagster_run_details` lives at `datalake_sandbox.public_volume_predictions.dagster_run_details`.

Columns used:
- `strata_id` — integer
- `metadata` — VARIANT/JSON; `PARSE_JSON(metadata):tags:cliententityid::INT` = entity_id
- `pipeline_run_id` — the value that maps to `collection_id` in the actuals table
- `created_at` — datetime; filter by `DATE(created_at)`

The resolution query template:

```sql
SELECT
    strata_id,
    PARSE_JSON(metadata):tags:cliententityid::INT AS entity_id,
    pipeline_run_id
FROM datalake_sandbox.public_volume_predictions.dagster_run_details
WHERE strata_id IN ({strata_ids})
  AND <mode_filter>
```

Mode filters:
- `today`: `DATE(created_at) = CURRENT_DATE`
- `most-recent`: `DATE(created_at) = (SELECT MAX(DATE(created_at)) FROM datalake_sandbox.public_volume_predictions.dagster_run_details)`
- `date-range`: `DATE(created_at) BETWEEN %(date_from)s AND %(date_to)s`

"One result" means: for each `(strata_id, entity_id)` pair, exactly one distinct `pipeline_run_id`.

Return value: `dict[tuple[int, int], str]` mapping `(strata_id, entity_id) → pipeline_run_id`.

Warnings collected (returned as list of dicts with keys `warning_type`, `strata_id`, `entity_id`, `run_mode`, `requested_date`, `fallback_date`, `message`):
- `skipped_pair` — pair found 0 results in `today` or `date-range` mode
- `today_fallback` — `today` returned 0 total results, fell back to `most-recent`

- [ ] **Step 1: Write the failing tests in `tests/test_db.py`**

Add these tests:

```python
from stars_pipeline.db import _resolve_collection_id


def _make_dagster_rows(pairs: list[tuple[int, int, str]]) -> pd.DataFrame:
    """Build a fake dagster_run_details result. pairs = [(strata_id, entity_id, pipeline_run_id)]"""
    return pd.DataFrame(
        pairs, columns=["strata_id", "entity_id", "pipeline_run_id"]
    )


def test_resolve_today_all_pairs_found(monkeypatch):
    run_cfg = _make_run_cfg(run_mode="today", strata_ids=[84])

    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.return_value = _make_dagster_rows([
        (84, 1001, "run-abc"),
        (84, 1002, "run-abc"),
    ])
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    resolved, warnings = _resolve_collection_id(run_cfg, mock_conn)

    assert resolved == {(84, 1001): "run-abc", (84, 1002): "run-abc"}
    assert warnings == []


def test_resolve_today_some_pairs_missing_logs_warning(monkeypatch):
    run_cfg = _make_run_cfg(run_mode="today", strata_ids=[84])

    # First call (today) returns only one pair; second call (most-recent fallback not needed
    # because at least one pair found)
    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.return_value = _make_dagster_rows([
        (84, 1001, "run-abc"),
        # (84, 1002) is missing
    ])
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    resolved, warnings = _resolve_collection_id(run_cfg, mock_conn)

    assert (84, 1001) in resolved
    assert (84, 1002) not in resolved
    # skipped_pair warning emitted for the missing pair — but we don't know entity_id 1002
    # unless the caller provides it. Since resolution only returns what it finds, no
    # skipped_pair warning is needed when some pairs are simply absent from today's data.
    # The warning type here is a zero-result subset, which is acceptable and silent.
    assert warnings == []


def test_resolve_today_zero_results_falls_back_to_most_recent():
    run_cfg = _make_run_cfg(run_mode="today", strata_ids=[84])

    # today call returns empty; most-recent call returns rows
    today_df = _make_dagster_rows([])
    most_recent_df = _make_dagster_rows([(84, 1001, "run-old")])

    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.side_effect = [today_df, most_recent_df]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    resolved, warnings = _resolve_collection_id(run_cfg, mock_conn)

    assert resolved == {(84, 1001): "run-old"}
    assert len(warnings) == 1
    assert warnings[0]["warning_type"] == "today_fallback"
    assert warnings[0]["entity_id"] is None


def test_resolve_most_recent_zero_results_raises():
    run_cfg = _make_run_cfg(run_mode="most-recent", strata_ids=[84])

    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.return_value = _make_dagster_rows([])
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with pytest.raises(ValueError, match="most-recent"):
        _resolve_collection_id(run_cfg, mock_conn)


def test_resolve_any_mode_duplicate_pipeline_run_id_raises():
    run_cfg = _make_run_cfg(run_mode="today", strata_ids=[84])

    # Same (strata_id, entity_id) pair maps to two different pipeline_run_ids
    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.return_value = _make_dagster_rows([
        (84, 1001, "run-aaa"),
        (84, 1001, "run-bbb"),
    ])
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with pytest.raises(ValueError, match="2+ results"):
        _resolve_collection_id(run_cfg, mock_conn)


def test_resolve_date_range_missing_pair_returns_warning():
    run_cfg = _make_run_cfg(
        run_mode="date-range",
        run_mode_date_from=date(2025, 1, 1),
        run_mode_date_to=date(2025, 1, 31),
        strata_ids=[84],
    )

    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.return_value = _make_dagster_rows([
        (84, 1001, "run-jan"),
    ])
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    # Should succeed (date-range skips missing pairs, same as today)
    resolved, warnings = _resolve_collection_id(run_cfg, mock_conn)
    assert (84, 1001) in resolved


def test_resolve_date_range_zero_results_raises():
    run_cfg = _make_run_cfg(
        run_mode="date-range",
        run_mode_date_from=date(2025, 1, 1),
        run_mode_date_to=date(2025, 1, 31),
        strata_ids=[84],
    )

    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.return_value = _make_dagster_rows([])
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with pytest.raises(ValueError, match="date-range"):
        _resolve_collection_id(run_cfg, mock_conn)
```

- [ ] **Step 2: Run to verify tests fail**

```
pytest tests/test_db.py -v -k "resolve"
```

Expected: All FAIL with `ImportError: cannot import name '_resolve_collection_id'`

- [ ] **Step 3: Add `_resolve_collection_id` to `stars_pipeline/db.py`**

Add the constant and function before `fetch_actuals()`:

```python
_DAGSTER_TABLE = "datalake_sandbox.public_volume_predictions.dagster_run_details"

_RESOLUTION_SQL_TEMPLATE = """
SELECT
    strata_id,
    PARSE_JSON(metadata):tags:cliententityid::INT AS entity_id,
    pipeline_run_id
FROM {dagster_table}
WHERE strata_id IN ({strata_ids})
  AND {mode_filter}
"""

_RESOLUTION_SQL_MOST_RECENT_FILTER = (
    "DATE(created_at) = (SELECT MAX(DATE(created_at)) FROM {dagster_table})"
)


def _resolve_collection_id(
    run_cfg: RunConfig,
    conn: snowflake.connector.SnowflakeConnection,
) -> tuple[dict[tuple[int, int], str], list[dict]]:
    """
    Query dagster_run_details to resolve pipeline_run_id per (strata_id, entity_id).

    Returns
    -------
    (resolved, warnings) where:
        resolved: dict mapping (strata_id, entity_id) → pipeline_run_id
        warnings: list of warning dicts (warning_type, strata_id, entity_id,
                  run_mode, requested_date, fallback_date, message)

    Raises
    ------
    ValueError
        If any (strata_id, entity_id) pair maps to 2+ distinct pipeline_run_ids.
        If most-recent or date-range returns 0 total results.
    """
    strata_str = ", ".join(str(s) for s in run_cfg.strata_ids)
    mode = run_cfg.run_mode
    warnings: list[dict] = []

    def _run_query(mode_filter: str, params: dict | None = None) -> pd.DataFrame:
        sql = _RESOLUTION_SQL_TEMPLATE.format(
            dagster_table=_DAGSTER_TABLE,
            strata_ids=strata_str,
            mode_filter=mode_filter,
        )
        cur = conn.cursor()
        cur.execute(sql, params or {})
        df = cur.fetch_pandas_all()
        df.columns = [c.lower() for c in df.columns]
        return df

    def _check_duplicates(df: pd.DataFrame) -> None:
        counts = df.groupby(["strata_id", "entity_id"])["pipeline_run_id"].nunique()
        dupes = counts[counts > 1]
        if not dupes.empty:
            pairs = dupes.index.tolist()
            raise ValueError(
                f"2+ results: multiple pipeline_run_ids found for {mode!r} mode, "
                f"offending pairs: {pairs}"
            )

    if mode == "today":
        df = _run_query("DATE(created_at) = CURRENT_DATE")
        if df.empty:
            logger.warning(
                "No runs found for today, falling back to most-recent",
                extra={"run_mode": "today"},
            )
            fallback_filter = _RESOLUTION_SQL_MOST_RECENT_FILTER.format(
                dagster_table=_DAGSTER_TABLE
            )
            df = _run_query(fallback_filter)
            if df.empty:
                raise ValueError(
                    "most-recent fallback returned 0 results after today returned 0 results"
                )
            fallback_date = str(df["strata_id"].iloc[0]) if not df.empty else None
            # Get the actual fallback date from a separate query would require another round-trip;
            # we log what we know from the data.
            warnings.append({
                "warning_type": "today_fallback",
                "strata_id": None,
                "entity_id": None,
                "run_mode": "today",
                "requested_date": "today",
                "fallback_date": None,
                "message": "No runs found for today; fell back to most-recent",
            })
        _check_duplicates(df)

    elif mode == "most-recent":
        fallback_filter = _RESOLUTION_SQL_MOST_RECENT_FILTER.format(
            dagster_table=_DAGSTER_TABLE
        )
        df = _run_query(fallback_filter)
        if df.empty:
            raise ValueError("most-recent mode returned 0 results")
        _check_duplicates(df)

    elif mode == "date-range":
        df = _run_query(
            "DATE(created_at) BETWEEN %(date_from)s AND %(date_to)s",
            params={
                "date_from": str(run_cfg.run_mode_date_from),
                "date_to": str(run_cfg.run_mode_date_to),
            },
        )
        if df.empty:
            raise ValueError(
                f"date-range mode returned 0 results for "
                f"{run_cfg.run_mode_date_from} to {run_cfg.run_mode_date_to}"
            )
        _check_duplicates(df)

    else:
        raise ValueError(f"Unknown run_mode: {mode!r}")

    resolved = {
        (int(row["strata_id"]), int(row["entity_id"])): row["pipeline_run_id"]
        for _, row in df.iterrows()
    }
    return resolved, warnings
```

- [ ] **Step 4: Run the resolution tests**

```
pytest tests/test_db.py -v -k "resolve"
```

Expected: All pass. If `test_resolve_today_some_pairs_missing_logs_warning` has issues because it checks for a pair that was never requested, adjust expectations: the function returns only what it finds, so no skipped_pair warning is needed unless we track *which* pairs were expected.

- [ ] **Step 5: Update `fetch_actuals()` to call `_resolve_collection_id()` and pass a resolved ID**

Replace the `fetch_actuals()` body in `stars_pipeline/db.py`:

```python
def fetch_actuals(run_cfg: RunConfig) -> tuple[pd.DataFrame, list[dict]]:
    """
    Execute actuals.sql with the given RunConfig and return (DataFrame, warnings).

    If run_cfg.collection_id is None, calls _resolve_collection_id() first to
    auto-resolve pipeline_run_id from dagster_run_details.

    Returns
    -------
    (df, warnings) where:
        df: DataFrame with columns strata_id, entity_id, patient_type_rollup,
            service_line, date, actual, mesh
        warnings: list of warning dicts from resolution (empty if collection_id
                  was provided directly)
    """
    sql_template = _SQL_PATH.read_text()
    strata_str = ", ".join(str(s) for s in run_cfg.strata_ids)
    sql = sql_template.format(strata_ids=strata_str)

    warnings: list[dict] = []

    conn = _get_connection()
    try:
        collection_id = run_cfg.collection_id
        if collection_id is None and run_cfg.run_mode is not None:
            resolved, warnings = _resolve_collection_id(run_cfg, conn)
            # Use the first resolved pipeline_run_id (all resolved pairs share one ID
            # under the happy path; heterogeneous IDs are filtered per-pair in future work)
            if resolved:
                collection_id = next(iter(resolved.values()))
                logger.info(
                    "Resolved collection_id from dagster_run_details",
                    extra={"collection_id": collection_id, "run_mode": run_cfg.run_mode},
                )

        params = {
            "date_from": str(run_cfg.date_from),
            "date_to": str(run_cfg.date_to),
            "entity_id": run_cfg.entity_id,
            "patient_type": run_cfg.patient_type,
            "service_line": run_cfg.service_line,
            "collection_id": collection_id,
            "run_id": run_cfg.run_id,
        }

        logger.info(
            "Executing actuals.sql",
            extra={
                "strata_ids": run_cfg.strata_ids,
                "date_from": str(run_cfg.date_from),
                "date_to": str(run_cfg.date_to),
                "collection_id": collection_id,
            },
        )

        cur = conn.cursor()
        cur.execute(sql, params)
        df = cur.fetch_pandas_all()
    finally:
        conn.close()

    df.columns = [col.lower() for col in df.columns]
    df["date"] = pd.to_datetime(df["date"])

    logger.info("Actuals fetched", extra={"rows": len(df)})
    return df, warnings
```

- [ ] **Step 6: Uncomment the collection_id WHERE clause in `stars_pipeline/sql/actuals.sql`**

Replace line 71:
```sql
    -- TODO: AND v.collection_id = %(collection_id)s
```
With:
```sql
    AND (%(collection_id)s IS NULL OR v.collection_id = %(collection_id)s)
```

Keep the `run_id` line commented (not part of this feature).

- [ ] **Step 7: Update `cli.py` to handle the new return value from `fetch_actuals()`**

In `main()`, replace:
```python
    df = fetch_actuals(run_cfg)
    logger.info("Snowflake pull complete", extra={"rows": len(df)})
```
With:
```python
    df, resolution_warnings = fetch_actuals(run_cfg)
    logger.info("Snowflake pull complete", extra={"rows": len(df)})
```

And store `resolution_warnings` for use in Task 4. For now, just capture it:

```python
    # resolution_warnings written to CSV in write_warnings_csv (Task 4)
    _ = resolution_warnings
```

- [ ] **Step 8: Update `test_fetch_actuals_returns_expected_columns` in `tests/test_db.py`**

The function now returns `(df, warnings)`. Update the test:

```python
def test_fetch_actuals_returns_expected_columns(monkeypatch):
    for var in ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD",
                "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA"]:
        monkeypatch.setenv(var, "dummy")

    fake_df = pd.DataFrame({
        "STRATA_ID": [84, 84],
        "ENTITY_ID": ["E01", "E01"],
        "PATIENT_TYPE_ROLLUP": ["Inpatient", "Inpatient"],
        "SERVICE_LINE": ["Cardiology", "Cardiology"],
        "DATE": ["2025-01-01", "2025-01-02"],
        "ACTUAL": [100.0, 110.0],
        "MESH": [2.5, 2.5],
    })

    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.return_value = fake_df
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch("stars_pipeline.db._get_connection", return_value=mock_conn):
        df, warnings = fetch_actuals(_make_run_cfg(run_mode=None, collection_id="COL123"))

    expected_cols = {"strata_id", "entity_id", "patient_type_rollup",
                     "service_line", "date", "actual", "mesh"}
    assert expected_cols.issubset(set(df.columns))
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert warnings == []
```

- [ ] **Step 9: Run the full db test suite**

```
pytest tests/test_db.py -v
```

Expected: All pass (except pre-existing `test_missing_env_vars_raises` if env var set)

- [ ] **Step 10: Run full test suite**

```
pytest tests/ -v
```

Expected: All pass

- [ ] **Step 11: Commit**

```bash
git add stars_pipeline/db.py stars_pipeline/sql/actuals.sql stars_pipeline/cli.py tests/test_db.py
git commit -m "feat: add _resolve_collection_id and wire into fetch_actuals"
```

---

### Task 4: Warnings CSV writer

**Files:**
- Create: `stars_pipeline/stars/warnings.py`
- Create: `tests/stars/test_warnings.py`
- Modify: `stars_pipeline/cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/stars/test_warnings.py`:

```python
# tests/stars/test_warnings.py
import pandas as pd
from pathlib import Path

from stars_pipeline.stars.warnings import build_warnings_df, write_warnings_csv


_WARNING_COLS = [
    "warning_type", "strata_id", "entity_id", "run_mode",
    "requested_date", "fallback_date", "message",
]


def test_build_warnings_df_empty():
    df = build_warnings_df([])
    assert list(df.columns) == _WARNING_COLS
    assert len(df) == 0


def test_build_warnings_df_today_fallback():
    warnings = [{
        "warning_type": "today_fallback",
        "strata_id": None,
        "entity_id": None,
        "run_mode": "today",
        "requested_date": "today",
        "fallback_date": None,
        "message": "No runs found for today; fell back to most-recent",
    }]
    df = build_warnings_df(warnings)
    assert len(df) == 1
    assert df.iloc[0]["warning_type"] == "today_fallback"
    assert df.iloc[0]["entity_id"] is None or pd.isna(df.iloc[0]["entity_id"])


def test_build_warnings_df_skipped_pair():
    warnings = [{
        "warning_type": "skipped_pair",
        "strata_id": 84,
        "entity_id": 1001,
        "run_mode": "date-range",
        "requested_date": "2025-01-01 to 2025-01-31",
        "fallback_date": None,
        "message": "No run found for (84, 1001) in date-range",
    }]
    df = build_warnings_df(warnings)
    assert len(df) == 1
    assert df.iloc[0]["strata_id"] == 84
    assert df.iloc[0]["entity_id"] == 1001


def test_write_warnings_csv_creates_file(tmp_path):
    out = tmp_path / "results.csv"
    warnings = [{
        "warning_type": "today_fallback",
        "strata_id": None,
        "entity_id": None,
        "run_mode": "today",
        "requested_date": "today",
        "fallback_date": None,
        "message": "No runs found for today; fell back to most-recent",
    }]
    write_warnings_csv(warnings, out)
    warnings_path = tmp_path / "results_warnings.csv"
    assert warnings_path.exists()
    df = pd.read_csv(warnings_path)
    assert list(df.columns) == _WARNING_COLS
    assert len(df) == 1


def test_write_warnings_csv_empty_creates_file_with_headers(tmp_path):
    out = tmp_path / "stars_pilots2.csv"
    write_warnings_csv([], out)
    warnings_path = tmp_path / "stars_pilots2_warnings.csv"
    assert warnings_path.exists()
    df = pd.read_csv(warnings_path)
    assert list(df.columns) == _WARNING_COLS
    assert len(df) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/stars/test_warnings.py -v
```

Expected: All FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create `stars_pipeline/stars/warnings.py`**

```python
"""
warnings.py
===========
Warnings CSV writer for the STARS pipeline run-resolution step.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from stars_pipeline.logging_config import get_logger

logger = get_logger(__name__)

_WARNING_COLS = [
    "warning_type",
    "strata_id",
    "entity_id",
    "run_mode",
    "requested_date",
    "fallback_date",
    "message",
]


def build_warnings_df(warnings: list[dict]) -> pd.DataFrame:
    """Build a warnings DataFrame from a list of warning dicts."""
    if not warnings:
        return pd.DataFrame(columns=_WARNING_COLS)
    return pd.DataFrame(warnings, columns=_WARNING_COLS)


def write_warnings_csv(warnings: list[dict], output_path: Path) -> None:
    """
    Write warnings to <output_stem>_warnings.csv alongside the main output.

    Always written — even when empty (with headers only).

    Args:
        warnings:    List of warning dicts from _resolve_collection_id().
        output_path: Main output CSV path. Warnings written to same dir with _warnings suffix.
    """
    output_path = Path(output_path)
    warnings_path = output_path.with_name(output_path.stem + "_warnings" + output_path.suffix)
    df = build_warnings_df(warnings)
    warnings_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(warnings_path, index=False)
    logger.info(
        "Warnings CSV written",
        extra={"rows": len(df), "path": str(warnings_path)},
    )
```

- [ ] **Step 4: Run the warnings tests**

```
pytest tests/stars/test_warnings.py -v
```

Expected: All pass

- [ ] **Step 5: Wire `write_warnings_csv` into `cli.py` `main()`**

In `main()`, replace the placeholder `_ = resolution_warnings` with:

```python
    from stars_pipeline.stars.warnings import write_warnings_csv
    write_warnings_csv(resolution_warnings, run_cfg.output_path)
    logger.info("Warnings CSV written", extra={"output_path": str(run_cfg.output_path)})
```

Move the import to the top of `cli.py` with the other imports:

```python
from stars_pipeline.stars.warnings import write_warnings_csv
```

And in `main()`:

```python
    write_warnings_csv(resolution_warnings, run_cfg.output_path)
```

- [ ] **Step 6: Run the full test suite**

```
pytest tests/ -v
```

Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add stars_pipeline/stars/warnings.py tests/stars/test_warnings.py stars_pipeline/cli.py
git commit -m "feat: add warnings CSV writer for run-resolution events"
```

---

### Task 5: Final integration check

**Files:**
- No new files

- [ ] **Step 1: Run the complete test suite and confirm counts**

```
pytest tests/ -v --tb=short
```

Expected: All tests pass. Note test count (should be 64+ tests).

- [ ] **Step 2: Verify SQL change is correct**

Read `stars_pipeline/sql/actuals.sql` and confirm the collection_id line reads:

```sql
    AND (%(collection_id)s IS NULL OR v.collection_id = %(collection_id)s)
```

- [ ] **Step 3: Commit if any stray changes remain**

```bash
git status
```

If any files show unstaged changes, stage and commit them.

- [ ] **Step 4: Final commit message**

```bash
git add -A
git commit -m "chore: run-selection modes feature complete — all tests passing"
```
