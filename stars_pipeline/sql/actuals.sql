-- =============================================================================
-- actuals.sql
-- =============================================================================
-- Pulls daily segment-level volume (actuals) and MESH scores for the STARS
-- evaluation pipeline.
--
-- This file is read and rendered by stars_pipeline/db.py. The {strata_ids}
-- placeholder is a Python .format() token (filled with integer literals before
-- execution). All other parameters use named snowflake-connector binding.
--
-- Parameters
-- ----------
-- {strata_ids}      Comma-separated integer literals, e.g. "84, 14, 1318"
--                   Rendered by db.py — safe because strata_ids are integers.
-- %(date_from)s     Start date, inclusive (YYYY-MM-DD string)
-- %(date_to)s       End date, inclusive (YYYY-MM-DD string)
-- %(entity_id)s     Optional entity filter  (NULL = no filter)
-- %(patient_type)s  Optional patient type filter (NULL = no filter)
-- %(service_line)s  Optional service line filter (NULL = no filter)
-- %(collection_id)s Collection identifier — passed but not yet used in WHERE.
--                   TODO: uncomment the collection_id filter once schema confirmed.
-- %(run_id)s        Run identifier — passed but not yet used in WHERE.
--                   TODO: uncomment the run_id filter once schema confirmed.
--
-- Output Columns
-- --------------
-- strata_id             INTEGER
-- entity_id             VARCHAR
-- patient_type_rollup   VARCHAR
-- service_line          VARCHAR
-- date                  DATE        daily observation date
-- actual                FLOAT       observed daily volume (summed if duplicates)
-- mesh                  FLOAT       segment-level MESH error score (broadcast to all days)
--
-- Open Items
-- ----------
-- TODO (#1): Confirm table names and join key column names against live schema.
--            Current assumption: prod_cv_validation_daily_volume has
--            entity_id, patient_type_rollup_id, service_line_id columns
--            that match prod_champion_cv_results.
-- TODO (#2): Wire collection_id and run_id into WHERE clause once schema confirmed.
-- =============================================================================

SELECT
    v.strata_id,
    v.entity_id,
    v.patient_type_rollup_clean   AS patient_type_rollup,
    v.service_line_clean          AS service_line,
    v.admitdate                   AS date,
    SUM(v.actual)                 AS actual,
    -- MESH is a scalar per segment; MAX() broadcasts it to every daily row.
    MAX(champ.mesh)               AS mesh

FROM datalake_sandbox.res.prod_cv_validation_daily_volume v

-- Join champion model MESH score at the segment level.
-- TODO: confirm join key column names (patient_type_rollup_id, service_line_id)
LEFT JOIN datalake_sandbox.res.prod_champion_cv_results champ
    ON  champ.strata_id             = v.strata_id
    AND champ.entity_id             = v.entity_id
    AND champ.patient_type_rollup_id = v.patient_type_rollup_id
    AND champ.service_line_id        = v.service_line_id
    AND champ.cv_type               != 'Short Term'

WHERE
    v.strata_id IN ({strata_ids})
    AND v.admitdate BETWEEN %(date_from)s AND %(date_to)s
    AND (%(entity_id)s   IS NULL OR v.entity_id                = %(entity_id)s)
    AND (%(patient_type)s IS NULL OR v.patient_type_rollup_clean = %(patient_type)s)
    AND (%(service_line)s IS NULL OR v.service_line_clean       = %(service_line)s)
    AND (%(collection_id)s IS NULL OR v.collection_id = %(collection_id)s)
    -- TODO: AND v.run_id        = %(run_id)s

GROUP BY
    v.strata_id,
    v.entity_id,
    v.patient_type_rollup_clean,
    v.service_line_clean,
    v.admitdate

ORDER BY
    v.strata_id,
    v.entity_id,
    v.patient_type_rollup_clean,
    v.service_line_clean,
    v.admitdate
