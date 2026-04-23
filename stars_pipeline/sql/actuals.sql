-- =============================================================================
-- actuals.sql
-- =============================================================================
-- Pulls daily segment-level volume from CLIENT_DATA_SERVICE_LINE_AGGREGATIONS.
--
-- Rendered by db.py:
--   {strata_ids}   Comma-separated integer literals, e.g. "84, 14"
--   {run_ids}      Comma-separated single-quoted UUIDs, e.g. "'uuid1','uuid2'"
--                  Rendered as string literals (safe — UUIDs from dagster resolution).
--
-- Output columns:
--   strata_id, entity_id, patient_type_rollup_id, patient_type_rollup_clean,
--   service_line_id, service_line_clean, date, actual, row_count
--
-- row_count should be 1 for every row. Values > 1 indicate unexpected duplicates.
-- =============================================================================

SELECT
    strata_id,
    cliententityid                AS entity_id,
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean,
    admitdate                     AS date,
    SUM(volume)                   AS actual,
    COUNT(*)                      AS row_count
FROM DATALAKE_SANDBOX.PUBLIC_VOLUME_PREDICTIONS.CLIENT_DATA_SERVICE_LINE_AGGREGATIONS
WHERE strata_id IN ({strata_ids})
  AND run_id IN ({run_ids})
  AND snapshot_date = (
      SELECT MAX(snapshot_date)
        FROM DATALAKE_SANDBOX.PUBLIC_VOLUME_PREDICTIONS.CLIENT_DATA_SERVICE_LINE_AGGREGATIONS
       WHERE strata_id IN ({strata_ids})
         AND run_id IN ({run_ids})
         AND snapshot_date >= DATEADD('month', -1, CURRENT_DATE)
  )
GROUP BY
    strata_id,
    cliententityid,
    patient_type_rollup_id,
    patient_type_rollup_clean,
    service_line_id,
    service_line_clean,
    admitdate
ORDER BY
    strata_id,
    cliententityid,
    patient_type_rollup_id,
    service_line_id,
    admitdate
