# tests/stars/test_mesh.py
import pandas as pd
import pytest
from stars_pipeline.stars.mesh import compute_mesh

_SEG_COLS = [
    "strata_id", "entity_id", "patient_type_rollup_id", "patient_type_rollup_clean",
    "service_line_id", "service_line_clean",
]


def _make_cv_df(rows):
    return pd.DataFrame(rows)


def test_compute_mesh_single_segment_single_model():
    df = _make_cv_df([
        dict(strata_id=84, entity_id=1, patient_type_rollup_id=1,
             patient_type_rollup_clean="IP", service_line_id="10",
             service_line_clean="Cardiology", model_name="ModelA",
             prediction=90.0, actual=100.0),
        dict(strata_id=84, entity_id=1, patient_type_rollup_id=1,
             patient_type_rollup_clean="IP", service_line_id="10",
             service_line_clean="Cardiology", model_name="ModelA",
             prediction=110.0, actual=100.0),
    ])
    result = compute_mesh(df)
    assert len(result) == 1
    # ESH row1 = 100*abs(100-90)/max(100,100) = 10.0
    # ESH row2 = 100*abs(100-110)/max(100,100) = 10.0
    # MESH = mean([10.0, 10.0]) = 10.0
    assert abs(result.iloc[0]["mesh"] - 10.0) < 1e-9
    assert set(_SEG_COLS + ["mesh"]).issubset(result.columns)


def test_compute_mesh_selects_champion_model():
    """Champion is the model with the lowest MESH."""
    df = _make_cv_df([
        dict(strata_id=84, entity_id=1, patient_type_rollup_id=1,
             patient_type_rollup_clean="IP", service_line_id="10",
             service_line_clean="Cardiology", model_name="ModelA",
             prediction=80.0, actual=100.0),   # ESH = 20
        dict(strata_id=84, entity_id=1, patient_type_rollup_id=1,
             patient_type_rollup_clean="IP", service_line_id="10",
             service_line_clean="Cardiology", model_name="ModelB",
             prediction=95.0, actual=100.0),   # ESH = 5
    ])
    result = compute_mesh(df)
    assert len(result) == 1
    assert abs(result.iloc[0]["mesh"] - 5.0) < 1e-9


def test_compute_mesh_low_actual_uses_floor_of_100():
    """actual < 100 uses 100 as denominator."""
    df = _make_cv_df([
        dict(strata_id=84, entity_id=1, patient_type_rollup_id=1,
             patient_type_rollup_clean="IP", service_line_id="10",
             service_line_clean="Cardiology", model_name="ModelA",
             prediction=40.0, actual=50.0),   # ESH = 100*10/100 = 10.0, not 100*10/50=20
    ])
    result = compute_mesh(df)
    assert abs(result.iloc[0]["mesh"] - 10.0) < 1e-9


def test_compute_mesh_multiple_segments():
    rows = []
    for entity_id in [1, 2]:
        rows.append(dict(
            strata_id=84, entity_id=entity_id, patient_type_rollup_id=1,
            patient_type_rollup_clean="IP", service_line_id="10",
            service_line_clean="Cardiology", model_name="ModelA",
            prediction=90.0, actual=100.0,
        ))
    result = compute_mesh(pd.DataFrame(rows))
    assert len(result) == 2


def test_compute_mesh_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        compute_mesh(pd.DataFrame(columns=[
            "strata_id", "entity_id", "patient_type_rollup_id",
            "patient_type_rollup_clean", "service_line_id", "service_line_clean",
            "model_name", "prediction", "actual",
        ]))
