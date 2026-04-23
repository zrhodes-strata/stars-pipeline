# tests/test_config.py
import pytest
from datetime import date
from pathlib import Path
from dataclasses import FrozenInstanceError

from stars_pipeline.config import MonitorConfig, RunConfig


def test_monitor_config_is_frozen():
    cfg = MonitorConfig()
    with pytest.raises((FrozenInstanceError, TypeError)):
        cfg.ks_d_threshold = 0.99  # type: ignore


def test_monitor_config_canonical_thresholds():
    cfg = MonitorConfig()
    assert cfg.ks_d_threshold == 0.30
    assert cfg.level_shift_min_cohen_d == 1.00
    assert cfg.dw_delta_threshold == 1.15
    assert cfg.slope_change_ratio_threshold == 1.50
    assert cfg.kpss_alpha == 0.10
    assert cfg.trend_p_value_threshold == 0.05
    assert cfg.coverage_delta_threshold == 0.30
    assert cfg.sparsity_delta_threshold == 0.30
    assert cfg.low_volume_monthly_threshold == 3.00
    assert cfg.volatility_ratio_threshold == 3.50
    assert cfg.outlier_z_threshold == 3.50
    assert cfg.outlier_rate_threshold == 0.30
    assert cfg.acf_divergence_p_threshold == 0.05
    assert cfg.alpha == 0.05


def test_run_config_construction():
    cfg = RunConfig(
        strata_ids=[84, 14],
        collection_id=None,

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


def test_run_config_run_mode_fields():
    cfg = RunConfig(
        strata_ids=[84],
        collection_id=None,

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
    cfg = RunConfig(
        strata_ids=[84],
        collection_id=None,

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
