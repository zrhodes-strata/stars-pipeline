from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd

def plot_metric_distributions(stats_df: pd.DataFrame, **kwargs) -> plt.Figure:
    raise NotImplementedError

def plot_normal_breakdowns(stats_df: pd.DataFrame, **kwargs) -> plt.Figure:
    raise NotImplementedError

def plot_flag_correlation_grid(stats_df: pd.DataFrame, **kwargs) -> plt.Figure:
    raise NotImplementedError

def plot_flag_rates_by_dim(stats_df: pd.DataFrame, **kwargs) -> plt.Figure:
    raise NotImplementedError

def plot_severity_and_families(stats_df: pd.DataFrame, **kwargs) -> plt.Figure:
    raise NotImplementedError

def plot_threshold_proximity(stats_df: pd.DataFrame, cfg=None, **kwargs) -> plt.Figure:
    raise NotImplementedError

def plot_segment_series(series_df: pd.DataFrame, feature_segment: str, **kwargs) -> plt.Figure:
    raise NotImplementedError
