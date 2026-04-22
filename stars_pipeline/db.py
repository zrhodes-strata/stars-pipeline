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
