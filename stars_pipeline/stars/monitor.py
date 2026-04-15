"""Stub — full implementation in Task 9."""
from __future__ import annotations
import pandas as pd
from stars_pipeline.config import MonitorConfig, RunConfig

def run_monitoring(df: pd.DataFrame, run_cfg: RunConfig, monitor_cfg: MonitorConfig) -> pd.DataFrame:
    raise NotImplementedError("monitor.py not yet implemented")

def apply_thresholds(stats_df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("monitor.py not yet implemented")
