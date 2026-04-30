import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from stars_pipeline.viz.cli import main


def _make_long_csv(tmp_path: Path) -> Path:
    """Write a minimal long-format CSV for CLI testing."""
    from tests.viz.test_wide import _make_long_df
    df = _make_long_df()
    p = tmp_path / "stars_results.csv"
    df.to_csv(p, index=False)
    return p


def test_cli_produces_png_files(tmp_path):
    csv_path = _make_long_csv(tmp_path)
    out_dir = tmp_path / "plots"
    rc = main(["--input", str(csv_path), "--output-dir", str(out_dir)])
    assert rc == 0
    pngs = list(out_dir.glob("*.png"))
    assert len(pngs) >= 5, f"Expected at least 5 PNG files, got {len(pngs)}: {pngs}"


def test_cli_missing_input_returns_nonzero(tmp_path):
    rc = main(["--input", str(tmp_path / "nonexistent.csv"),
               "--output-dir", str(tmp_path / "out")])
    assert rc != 0
