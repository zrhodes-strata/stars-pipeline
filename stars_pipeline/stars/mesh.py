"""
mesh.py
=======
Computes champion model MESH scores from raw short-term CV output rows.

Public API
----------
compute_mesh(cv_df) -> pd.DataFrame
    ESH per row → MESH per (segment, model) → champion per segment.
"""
from __future__ import annotations

import pandas as pd

_SEGMENT_COLS = [
    "strata_id",
    "entity_id",
    "patient_type_rollup_id",
    "patient_type_rollup_clean",
    "service_line_id",
    "service_line_clean",
]


def compute_mesh(cv_df: pd.DataFrame) -> pd.DataFrame:
    if cv_df.empty:
        raise ValueError("compute_mesh: cv_df is empty — no CV data to compute MESH from")

    df = cv_df.copy()
    df["esh"] = 100.0 * (df["actual"] - df["prediction"]).abs() / df["actual"].clip(lower=100.0)

    group_cols = _SEGMENT_COLS + ["model_name"]
    mesh_by_model = df.groupby(group_cols, sort=False)["esh"].mean().reset_index()
    mesh_by_model = mesh_by_model.rename(columns={"esh": "mesh"})

    champion_idx = mesh_by_model.groupby(_SEGMENT_COLS, sort=False)["mesh"].idxmin()
    champion = mesh_by_model.loc[champion_idx, _SEGMENT_COLS + ["mesh"]].reset_index(drop=True)
    return champion
