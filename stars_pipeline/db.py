"""
db.py
=====
Snowflake connection and query execution for the STARS pipeline.

Credentials are read exclusively from environment variables.
Never pass credentials as CLI arguments or store them in config files.

Required Environment Variables
-------------------------------
SNOWFLAKE_ACCOUNT
    Account identifier, e.g. ``xy12345.us-east-1``.
SNOWFLAKE_USER
    Service account username.
SNOWFLAKE_PASSWORD
    Service account password.
SNOWFLAKE_WAREHOUSE
    Compute warehouse name.
SNOWFLAKE_DATABASE
    Target database name.
SNOWFLAKE_SCHEMA
    Target schema name.

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
    "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA",
)


def _get_connection() -> snowflake.connector.SnowflakeConnection:
    """
    Build a Snowflake connection from environment variables.

    Raises
    ------
    EnvironmentError
        If any required environment variable is absent or empty.
    """
    missing = [v for v in _REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        raise EnvironmentError(
            f"Missing required Snowflake environment variables: {', '.join(missing)}"
        )
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
    )


def fetch_actuals(run_cfg: RunConfig) -> pd.DataFrame:
    """
    Execute actuals.sql with the given RunConfig and return a DataFrame.

    The SQL template is read from stars_pipeline/sql/actuals.sql.
    ``strata_ids`` are rendered as integer literals into the IN clause
    before execution (safe — they are integers, not user strings).
    All other parameters use named snowflake-connector bind parameters.

    Args:
        run_cfg: RunConfig built from CLI arguments.

    Returns:
        DataFrame with columns:
            strata_id (int), entity_id (str), patient_type_rollup (str),
            service_line (str), date (datetime64), actual (float), mesh (float)

    Notes:
        collection_id and run_id are passed as bind parameters but are not
        yet wired into the WHERE clause. See TODO comments in actuals.sql.
    """
    sql_template = _SQL_PATH.read_text()

    # Render strata_ids as comma-separated integer literals for the IN clause.
    strata_str = ", ".join(str(s) for s in run_cfg.strata_ids)
    sql = sql_template.format(strata_ids=strata_str)

    params = {
        "date_from": str(run_cfg.date_from),
        "date_to": str(run_cfg.date_to),
        "entity_id": run_cfg.entity_id,
        "patient_type": run_cfg.patient_type,
        "service_line": run_cfg.service_line,
        "collection_id": run_cfg.collection_id,  # TODO: wire into WHERE clause once schema confirmed
        "run_id": run_cfg.run_id,                # TODO: wire into WHERE clause once schema confirmed
    }

    logger.info(
        "Executing actuals.sql",
        extra={
            "strata_ids": run_cfg.strata_ids,
            "date_from": str(run_cfg.date_from),
            "date_to": str(run_cfg.date_to),
        },
    )

    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        df = cur.fetch_pandas_all()
    finally:
        conn.close()

    # Normalise column names to lowercase
    df.columns = [col.lower() for col in df.columns]
    df["date"] = pd.to_datetime(df["date"])

    logger.info("Actuals fetched", extra={"rows": len(df)})
    return df
