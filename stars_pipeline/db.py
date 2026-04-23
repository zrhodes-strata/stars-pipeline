"""
db.py
=====
Snowflake connection and query execution for the STARS pipeline.

Credentials are read exclusively from environment variables.
Never pass credentials as CLI arguments or store them in config files.

Authentication modes (mutually exclusive, checked in order)
------------------------------------------------------------
1. Named connection (local / developer use — simplest):
       SNOWFLAKE_CONNECTION_NAME=my_example_connection
   Reads account/user/authenticator/role from ~/.snowflake/connections.toml.
   Still requires SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA
   as overrides (set them as env vars or add them to the connections.toml entry).

2. Password auth (SageMaker / service accounts):
       SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD,
       SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA
   The connector uses username + password directly.

3. SSO auth (local / developer use):
       SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER,
       SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA
   Leave SNOWFLAKE_PASSWORD unset. The connector opens a browser tab for SSO.
   Override the authenticator with SNOWFLAKE_AUTHENTICATOR (default:
   ``externalbrowser``).

SageMaker
---------
Pass credentials as the ``Environment`` dict in the SageMaker
``CreateProcessingJob`` API call. See sagemaker/job_definition.py
for a documented example, including an AWS Secrets Manager alternative.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import snowflake.connector

from stars_pipeline.config import RunConfig
from stars_pipeline.logging_config import get_logger
from stars_pipeline.stars.mesh import compute_mesh

logger = get_logger(__name__)

_SQL_PATH = Path(__file__).parent / "sql" / "actuals.sql"

_REQUIRED_ENV_VARS = (
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA",
)

_REQUIRED_WITH_CONNECTION = (
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA",
)


def _get_connection() -> snowflake.connector.SnowflakeConnection:
    """
    Build a Snowflake connection from environment variables.

    Checks for SNOWFLAKE_CONNECTION_NAME first (named connection from
    ~/.snowflake/connections.toml). Falls back to password or SSO auth.

    Raises
    ------
    EnvironmentError
        If required environment variables are absent or empty.
    """
    connection_name = os.environ.get("SNOWFLAKE_CONNECTION_NAME")

    if connection_name:
        logger.info("Using named connection", extra={"connection_name": connection_name})
        # Warehouse/database/schema are read from connections.toml; env vars
        # can optionally override them if needed.
        overrides = {
            k: os.environ[v]
            for k, v in [
                ("warehouse", "SNOWFLAKE_WAREHOUSE"),
                ("database",  "SNOWFLAKE_DATABASE"),
                ("schema",    "SNOWFLAKE_SCHEMA"),
            ]
            if os.environ.get(v)
        }
        return snowflake.connector.connect(connection_name=connection_name, **overrides)

    missing = [v for v in _REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        raise EnvironmentError(
            f"Missing required Snowflake environment variables: {', '.join(missing)}"
        )

    password = os.environ.get("SNOWFLAKE_PASSWORD")
    if password:
        return snowflake.connector.connect(
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            user=os.environ["SNOWFLAKE_USER"],
            password=password,
            warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
            database=os.environ["SNOWFLAKE_DATABASE"],
            schema=os.environ["SNOWFLAKE_SCHEMA"],
        )

    authenticator = os.environ.get("SNOWFLAKE_AUTHENTICATOR", "externalbrowser")
    logger.info("Using SSO authentication", extra={"authenticator": authenticator})
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        authenticator=authenticator,
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
    )


_DAGSTER_TABLE = "DATALAKE_SANDBOX.PUBLIC_VOLUME_PREDICTIONS.DAGSTER_RUN_DETAILS"

_STEP_A_SQL = """
SELECT DISTINCT pipeline_run_id
FROM {dagster_table}
WHERE strata_id IN ({strata_ids})
  AND launch_type = 'daily_research'
  AND {date_filter}
"""

_STEP_B_SQL = """
SELECT DISTINCT
    strata_id,
    PARSE_JSON(metadata):tags:cliententityid::INT AS entity_id,
    pipeline_run_id,
    run_id,
    collection_id
FROM {dagster_table}
WHERE pipeline_run_id IN ({pipeline_run_ids})
  AND strata_id IN ({strata_ids})
"""

_MOST_RECENT_DATE_FILTER = (
    "DATE(created_at) = ("
    "SELECT MAX(DATE(created_at)) FROM {dagster_table} "
    "WHERE launch_type = 'daily_research'"
    ")"
)

_CV_SQL_PATH = Path(__file__).parent / "sql" / "cv.sql"

_COLLECTION_ID_TO_RUN_IDS_SQL = """
SELECT DISTINCT run_id
FROM DATALAKE_SANDBOX.PUBLIC_VOLUME_PREDICTIONS.DAGSTER_RUN_DETAILS
WHERE collection_id = %(collection_id)s
"""


def _resolve_collection_id(
    run_cfg: RunConfig,
    conn: snowflake.connector.SnowflakeConnection,
) -> tuple[list[str], str | None, list[dict]]:
    """
    Resolve run_ids and collection_id from dagster_run_details.

    Two-step process:
      Step A: find pipeline_run_ids matching the date/mode filter.
      Step B: expand pipeline_run_ids to run_ids and collection_id.

    Returns
    -------
    (run_ids, collection_id, warnings) where:
        run_ids:       distinct run_ids for use in actuals/cv SQL
        collection_id: first non-null collection_id found; None if all null
        warnings:      list of warning dicts

    Raises
    ------
    ValueError
        If any (strata_id, entity_id) maps to 2+ distinct pipeline_run_ids.
        If most-recent or date-range returns 0 pipeline_run_ids.
    """
    strata_str = ", ".join(str(s) for s in run_cfg.strata_ids)
    mode = run_cfg.run_mode
    warnings: list[dict] = []

    def _run_step_a(date_filter: str, params: dict | None = None) -> pd.DataFrame:
        sql = _STEP_A_SQL.format(
            dagster_table=_DAGSTER_TABLE,
            strata_ids=strata_str,
            date_filter=date_filter,
        )
        cur = conn.cursor()
        cur.execute(sql, params or {})
        df = cur.fetch_pandas_all()
        df.columns = [c.lower() for c in df.columns]
        return df

    def _run_step_b(pipeline_run_ids: list[str]) -> pd.DataFrame:
        ids_str = ", ".join(f"'{pid}'" for pid in pipeline_run_ids)
        sql = _STEP_B_SQL.format(
            dagster_table=_DAGSTER_TABLE,
            pipeline_run_ids=ids_str,
            strata_ids=strata_str,
        )
        cur = conn.cursor()
        cur.execute(sql)
        df = cur.fetch_pandas_all()
        df.columns = [c.lower() for c in df.columns]
        return df

    def _check_duplicate_pipelines(expansion: pd.DataFrame) -> None:
        counts = (
            expansion.groupby(["strata_id", "entity_id"])["pipeline_run_id"]
            .nunique()
        )
        dupes = counts[counts > 1]
        if not dupes.empty:
            pairs = dupes.index.tolist()
            raise ValueError(
                f"2+ pipeline_run_ids found for mode {mode!r}, "
                f"offending (strata_id, entity_id) pairs: {pairs}"
            )

    most_recent_filter = _MOST_RECENT_DATE_FILTER.format(dagster_table=_DAGSTER_TABLE)

    if mode == "today":
        step_a = _run_step_a("DATE(created_at) = CURRENT_DATE")
        if step_a.empty:
            logger.warning("No runs found for today, falling back to most-recent")
            step_a = _run_step_a(most_recent_filter)
            if step_a.empty:
                raise ValueError(
                    "most-recent fallback returned 0 results after today returned 0 results"
                )
            warnings.append({
                "warning_type": "today_fallback",
                "strata_id": None,
                "entity_id": None,
                "run_mode": "today",
                "requested_date": "today",
                "fallback_date": None,
                "message": "No runs found for today; fell back to most-recent",
            })

    elif mode == "most-recent":
        step_a = _run_step_a(most_recent_filter)
        if step_a.empty:
            raise ValueError("most-recent mode returned 0 pipeline_run_ids")

    elif mode == "date-range":
        step_a = _run_step_a(
            "DATE(created_at) BETWEEN %(date_from)s AND %(date_to)s",
            params={
                "date_from": str(run_cfg.run_mode_date_from),
                "date_to": str(run_cfg.run_mode_date_to),
            },
        )
        if step_a.empty:
            raise ValueError(
                f"date-range mode returned 0 pipeline_run_ids for "
                f"{run_cfg.run_mode_date_from} to {run_cfg.run_mode_date_to}"
            )

    else:
        raise ValueError(f"Unknown run_mode: {mode!r}")

    pipeline_run_ids = step_a["pipeline_run_id"].tolist()
    expansion = _run_step_b(pipeline_run_ids)
    _check_duplicate_pipelines(expansion)

    run_ids = expansion["run_id"].dropna().unique().tolist()
    non_null_collection_ids = expansion["collection_id"].dropna()
    collection_id = non_null_collection_ids.iloc[0] if not non_null_collection_ids.empty else None

    logger.info(
        "Resolved run_ids from dagster_run_details",
        extra={
            "run_ids": run_ids,
            "collection_id": collection_id,
            "run_mode": mode,
        },
    )
    return run_ids, collection_id, warnings


def fetch_actuals(run_cfg: RunConfig) -> tuple[pd.DataFrame, list[dict]]:
    """
    Fetch daily actuals and compute segment MESH scores.

    Steps:
      1. Resolve run_ids and collection_id (via dagster or direct collection_id lookup).
      2. Execute actuals.sql → daily volume per segment-day.
      3. Execute cv.sql → raw short-term CV rows.
      4. Compute champion MESH per segment from CV rows.
      5. Join MESH onto actuals.

    Returns
    -------
    (df, warnings) where df has columns:
        strata_id, entity_id, patient_type_rollup_id, patient_type_rollup_clean,
        service_line_id, service_line_clean, date, actual, row_count, mesh
    """
    strata_str = ", ".join(str(s) for s in run_cfg.strata_ids)
    warnings: list[dict] = []

    conn = _get_connection()
    try:
        # --- Step 1: resolve run_ids and collection_id ---
        if run_cfg.collection_id is not None:
            cur = conn.cursor()
            cur.execute(_COLLECTION_ID_TO_RUN_IDS_SQL, {"collection_id": run_cfg.collection_id})
            lookup = cur.fetch_pandas_all()
            lookup.columns = [c.lower() for c in lookup.columns]
            run_ids = lookup["run_id"].dropna().unique().tolist()
            collection_id = run_cfg.collection_id
            logger.info(
                "Using explicit collection_id",
                extra={"collection_id": collection_id, "run_ids": run_ids},
            )
        else:
            run_ids, collection_id, warnings = _resolve_collection_id(run_cfg, conn)

        run_ids_str = ", ".join(f"'{rid}'" for rid in run_ids)

        if not run_ids:
            if run_cfg.collection_id is not None:
                raise ValueError(
                    f"collection_id {run_cfg.collection_id!r} returned 0 run_ids from dagster. "
                    "Cannot execute actuals or CV queries."
                )
            raise ValueError(
                f"Resolution returned 0 run_ids for mode {run_cfg.run_mode!r}. "
                "Cannot execute actuals or CV queries."
            )

        # --- Step 2: actuals ---
        actuals_sql = _SQL_PATH.read_text().format(
            strata_ids=strata_str,
            run_ids=run_ids_str,
        )
        logger.info(
            "Executing actuals.sql",
            extra={"strata_ids": run_cfg.strata_ids, "run_ids": run_ids},
        )
        cur = conn.cursor()
        cur.execute(actuals_sql)
        actuals_df = cur.fetch_pandas_all()
        actuals_df.columns = [c.lower() for c in actuals_df.columns]

        # --- Step 3: CV ---
        cv_sql = _CV_SQL_PATH.read_text().format(
            strata_ids=strata_str,
            run_ids=run_ids_str,
        )
        logger.info("Executing cv.sql", extra={"collection_id": collection_id})
        cur = conn.cursor()
        # collection_id=None is passed as SQL NULL; the connector translates Python None → NULL.
        # cv.sql uses %(collection_id)s IS NOT NULL to switch between collection_id and run_id filtering.
        cur.execute(cv_sql, {"collection_id": collection_id})
        cv_df = cur.fetch_pandas_all()
        cv_df.columns = [c.lower() for c in cv_df.columns]

    finally:
        conn.close()

    actuals_df["date"] = pd.to_datetime(actuals_df["date"])

    # --- Step 4: MESH ---
    mesh_df = compute_mesh(cv_df)

    # --- Step 5: join ---
    _SEGMENT_COLS = [
        "strata_id", "entity_id", "patient_type_rollup_id",
        "patient_type_rollup_clean", "service_line_id", "service_line_clean",
    ]
    df = actuals_df.merge(mesh_df, on=_SEGMENT_COLS, how="left")

    df = df.rename(columns={
        "patient_type_rollup_clean": "patient_type_rollup",
        "service_line_clean": "service_line",
    })

    logger.info("Actuals fetched", extra={"rows": len(df)})
    return df, warnings
