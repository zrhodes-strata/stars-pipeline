"""
cli.py
======
Command-line entry point for the STARS pipeline.

Usage
-----
Local::

    python -m stars_pipeline.cli \\
        --strata-ids 84,14,1318 \\
        --date-from 2022-01-01 \\
        --output ./stars_results.csv

SageMaker (output path points to the mounted volume; SageMaker uploads to S3)::

    python -m stars_pipeline.cli \\
        --strata-ids 84,14,1318 \\
        --date-from 2022-01-01 \\
        --output /opt/ml/processing/output/stars_results.csv

Snowflake credentials must be set as environment variables before running:
    SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD,
    SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA

See stars_pipeline/db.py for full credential documentation.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

from stars_pipeline.config import MonitorConfig, RunConfig
from stars_pipeline.db import fetch_actuals
from stars_pipeline.logging_config import configure_logging, get_logger
from stars_pipeline.stars.monitor import apply_thresholds, run_monitoring
from stars_pipeline.stars.output import write_long_csv

logger = get_logger(__name__)


def _parse_date(value: str) -> date:
    """Parse a YYYY-MM-DD string to a date object (used as argparse type)."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}' — expected YYYY-MM-DD")


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser. Separated for testability."""
    p = argparse.ArgumentParser(
        prog="stars-pipeline",
        description=(
            "Run the STARS diagnostic framework against Snowflake segment data "
            "and write a long-format CSV of metric values and flags."
        ),
    )
    p.add_argument(
        "--strata-ids",
        required=True,
        help="Comma-separated integer strata IDs to evaluate (e.g. 84,14,1318).",
    )
    p.add_argument(
        "--collection-id",
        default=None,
        help=(
            "Collection identifier. Passed to the SQL layer as a bind parameter. "
            "TODO: wire into actuals.sql WHERE clause once schema is confirmed."
        ),
    )
    p.add_argument(
        "--run-id",
        default=None,
        help=(
            "Run identifier. Passed to the SQL layer as a bind parameter. "
            "TODO: wire into actuals.sql WHERE clause once schema is confirmed."
        ),
    )
    p.add_argument(
        "--date-from",
        default="2022-01-01",
        type=_parse_date,
        help="Start of the data pull window, inclusive (YYYY-MM-DD, default: 2022-01-01).",
    )
    p.add_argument(
        "--date-to",
        default=None,
        type=_parse_date,
        help="End of the data pull window, inclusive (YYYY-MM-DD, default: today).",
    )
    p.add_argument(
        "--recent-days",
        type=int,
        default=90,
        help="Size of the recent window in days for shift detection (default: 90).",
    )
    p.add_argument(
        "--train-days",
        type=int,
        default=None,
        help="Size of the training window in days. Omit to use all available data.",
    )
    p.add_argument(
        "--entity-id",
        default=None,
        help="Narrow the pull to a single entity within the selected strata.",
    )
    p.add_argument(
        "--patient-type",
        default=None,
        help="Narrow the pull to a single patient type rollup.",
    )
    p.add_argument(
        "--service-line",
        default=None,
        help="Narrow the pull to a single service line.",
    )
    p.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV path. "
            "Default: stars_results_YYYY-MM-DD.csv in the current directory. "
            "For SageMaker use /opt/ml/processing/output/stars_results.csv."
        ),
    )
    return p


def _build_run_config(args: argparse.Namespace) -> RunConfig:
    """Translate parsed CLI args into a RunConfig. Separated for testability."""
    strata_ids = [int(s.strip()) for s in args.strata_ids.split(",")]
    date_to = args.date_to or date.today()
    output_path = (
        Path(args.output)
        if args.output
        else Path(f"stars_results_{date.today()}.csv")
    )
    return RunConfig(
        strata_ids=strata_ids,
        collection_id=args.collection_id,
        run_id=args.run_id,
        date_from=args.date_from,
        date_to=date_to,
        recent_days=args.recent_days,
        train_days=args.train_days,
        entity_id=args.entity_id,
        patient_type=args.patient_type,
        service_line=args.service_line,
        output_path=output_path,
    )


def main(argv: list[str] | None = None) -> int:
    """
    Pipeline entry point.

    Args:
        argv: Argument list (defaults to sys.argv[1:]). Override in tests.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    configure_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)

    run_cfg = _build_run_config(args)
    monitor_cfg = MonitorConfig()

    logger.info(
        "Starting STARS pipeline",
        extra={
            "strata_ids": run_cfg.strata_ids,
            "date_from": str(run_cfg.date_from),
            "date_to": str(run_cfg.date_to),
            "recent_days": run_cfg.recent_days,
            "train_days": run_cfg.train_days,
        },
    )

    df = fetch_actuals(run_cfg)
    logger.info("Snowflake pull complete", extra={"rows": len(df)})

    stats_df = run_monitoring(df, run_cfg, monitor_cfg)
    logger.info("STARS monitoring complete", extra={"segments": len(stats_df)})

    stats_df = apply_thresholds(stats_df)

    write_long_csv(stats_df, run_cfg.output_path)
    logger.info("Results written", extra={"output_path": str(run_cfg.output_path)})

    return 0


if __name__ == "__main__":
    sys.exit(main())
