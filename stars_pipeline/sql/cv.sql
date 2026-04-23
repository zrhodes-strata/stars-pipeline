-- =============================================================================
-- cv.sql
-- =============================================================================
-- Pulls raw short-term CV rows for MESH computation.
--
-- Rendered by db.py:
--   {strata_ids}   Comma-separated integer literals
--   {run_ids}      Comma-separated single-quoted UUIDs (used when collection_id is NULL)
--
-- Bind parameters:
--   %(collection_id)s   Collection ID — if not NULL, filter by collection_id.
--                       If NULL, filter by run_ids instead.
--
-- Output columns:
--   strata_id, entity_id, patient_type_rollup_id, patient_type_rollup_clean,
--   service_line_id, service_line_clean, model_name,
--   prediction (monthly sum), actual (monthly sum)
-- =============================================================================

SELECT
    strata_id,
    entity_id,
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean,
    model_name,
    SUM(prediction)  AS prediction,
    SUM(actual)      AS actual
FROM DATALAKE_SANDBOX.PUBLIC_VOLUME_PREDICTIONS.SHORT_TERM_CROSS_VALIDATION_OUTPUTS
WHERE strata_id IN ({strata_ids})
  AND (
      (%(collection_id)s IS NOT NULL AND collection_id = %(collection_id)s)
      OR
      (%(collection_id)s IS NULL AND run_id IN ({run_ids}))
  )
GROUP BY
    strata_id,
    entity_id,
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean,
    model_name
