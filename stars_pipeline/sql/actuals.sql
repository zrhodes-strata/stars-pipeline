-- =============================================================================
-- actuals.sql
-- =============================================================================
-- Pulls daily segment-level volume from CLIENT_DATA_SERVICE_LINE_AGGREGATIONS.
--
-- Rendered by db.py:
--   strata_ids     Comma-separated integer literals, e.g. "84, 14"
--   run_ids        Comma-separated single-quoted UUIDs, e.g. "'uuid1','uuid2'"
--                  Rendered as string literals (safe — UUIDs from dagster resolution).
--
-- Deduplication: a single run_id may produce multiple aggregation rows (reruns).
-- We keep only the row with the latest executed_at per run_id, per engineering guidance.
--
-- Output columns:
--   strata_id, entity_id, patient_type_rollup_id, patient_type_rollup_clean,
--   service_line_id, service_line_clean, date, actual, row_count
--
-- row_count should be 1 for every row. Values > 1 indicate unexpected duplicates.
-- =============================================================================

WITH latest_run AS (
    -- For each run_id, find the maximum executed_at (latest aggregation).
    SELECT
        run_id,
        MAX(executed_at) AS max_executed_at
    FROM DATALAKE_SANDBOX.PUBLIC_VOLUME_PREDICTIONS.CLIENT_DATA_SERVICE_LINE_AGGREGATIONS
    WHERE strata_id IN ({strata_ids})
      AND run_id IN ({run_ids})
    GROUP BY run_id
)

SELECT
    a.strata_id,
    a.cliententityid               AS entity_id,
    a.patient_type_rollup_id,
    a.patient_type_rollup_clean,
    a.service_line_id,
    a.service_line_clean,
    a.admitdate                    AS date,
    SUM(a.volume)                  AS actual,
    COUNT(*)                       AS row_count
FROM DATALAKE_SANDBOX.PUBLIC_VOLUME_PREDICTIONS.CLIENT_DATA_SERVICE_LINE_AGGREGATIONS a
JOIN latest_run lr
  ON lr.run_id = a.run_id
 AND lr.max_executed_at = a.executed_at
WHERE a.strata_id IN ({strata_ids})
  AND a.run_id IN ({run_ids})
GROUP BY
    a.strata_id,
    a.cliententityid,
    a.patient_type_rollup_id,
    a.patient_type_rollup_clean,
    a.service_line_id,
    a.service_line_clean,
    a.admitdate
ORDER BY
    a.strata_id,
    a.cliententityid,
    a.patient_type_rollup_id,
    a.service_line_id,
    a.admitdate
