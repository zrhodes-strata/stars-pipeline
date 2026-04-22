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
    """Write warnings to <output_stem>_warnings.csv alongside the main output.

    Always written — even when empty (with headers only).
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
