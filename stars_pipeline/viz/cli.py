from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stars-viz",
        description="Generate diagnostic visualizations from a STARS long-format CSV.",
    )
    p.add_argument("--input",       required=True, help="Path to long-format CSV from stars-pipeline.")
    p.add_argument("--output-dir",  required=True, help="Directory to write PNG files.")
    p.add_argument("--recent-days", type=int, default=90, help="Recent window size (for series plot).")
    p.add_argument("--dpi",         type=int, default=150, help="PNG resolution in DPI.")
    return p


def main(argv: list[str] | None = None) -> int:
    matplotlib.use("Agg")
    import pandas as pd
    from stars_pipeline.viz._wide import long_to_wide
    from stars_pipeline.viz.plots import (
        plot_metric_distributions,
        plot_normal_breakdowns,
        plot_flag_correlation_grid,
        plot_flag_rates_by_dim,
        plot_severity_and_families,
        plot_threshold_proximity,
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        long_df = pd.read_csv(input_path)
        stats_df = long_to_wide(long_df)
    except Exception as exc:
        print(f"ERROR: Failed to read/pivot input: {exc}", file=sys.stderr)
        return 1

    plots = [
        ("metric_distributions",  lambda: plot_metric_distributions(stats_df)),
        ("normal_breakdowns",     lambda: plot_normal_breakdowns(stats_df)),
        ("flag_correlation_grid", lambda: plot_flag_correlation_grid(stats_df)),
        ("flag_rates_by_dim",     lambda: plot_flag_rates_by_dim(stats_df)),
        ("severity_and_families", lambda: plot_severity_and_families(stats_df)),
        ("threshold_proximity",   lambda: plot_threshold_proximity(stats_df)),
    ]

    for name, fn in plots:
        try:
            fig = fn()
            out_path = out_dir / f"stars_{name}.png"
            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"Written: {out_path}")
        except Exception as exc:
            print(f"WARNING: {name} failed: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
