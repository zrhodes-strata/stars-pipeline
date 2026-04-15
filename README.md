# STARS Pipeline

A self-contained Python pipeline that evaluates forecast segments against the
STARS diagnostic framework and writes a long-format CSV of metric values and flags.

## STARS Families

| Family | Broken Assumption | Indicators |
|---|---|---|
| Stability | Not Stable | KS shift, Level shift, DW shift, Slope change, Stationarity, Trend significance |
| Truthfulness | Not Truthful | Coverage shift, Sparsity change |
| Abundance | Not Abundant | Low volume |
| Regularity | Not Regular | Volatility shift, Outlier rate, ACF divergence, DOW pattern shift |

## Installation

```bash
pip install -e ".[dev]"
```

## Snowflake Credentials

Set the following environment variables before running:

```bash
export SNOWFLAKE_ACCOUNT=xy12345.us-east-1
export SNOWFLAKE_USER=svc_stars_pipeline
export SNOWFLAKE_PASSWORD=...
export SNOWFLAKE_WAREHOUSE=STARS_WH
export SNOWFLAKE_DATABASE=DATALAKE_SANDBOX
export SNOWFLAKE_SCHEMA=RES
```

## Local Usage

```bash
# Evaluate strata 84, 14, and 1318 from 2022-01-01 to today
stars-pipeline \
  --strata-ids 84,14,1318 \
  --date-from 2022-01-01 \
  --output ./stars_results.csv

# Narrow to a single entity and patient type
stars-pipeline \
  --strata-ids 84 \
  --entity-id E01 \
  --patient-type Inpatient \
  --output ./stars_results_84.csv
```

## SageMaker Usage

1. Build and push the Docker image to ECR:

```bash
docker build -t stars-pipeline .
docker tag stars-pipeline <account>.dkr.ecr.<region>.amazonaws.com/stars-pipeline:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/stars-pipeline:latest
```

2. Update `ECR_IMAGE_URI`, `IAM_ROLE_ARN`, and `OUTPUT_S3_URI` in
   `sagemaker/job_definition.py`.

3. Launch:

```bash
python -m sagemaker.job_definition
```

Output is written to `/opt/ml/processing/output/` and uploaded to S3 automatically.
For credentials, use AWS Secrets Manager (see `sagemaker/job_definition.py`).

## Output Format

One CSV row per segment per STARS indicator:

| Column | Description |
|---|---|
| `strata_id` | Strata identifier |
| `entity_id` | Entity identifier |
| `patient_type_rollup` | Patient type |
| `service_line` | Service line |
| `feature_segment` | Concatenated key |
| `stars_family` | Stability / Truthfulness / Abundance / Regularity / Summary |
| `metric_name` | Indicator name |
| `metric_value` | Raw statistic (string); family name for `stars_family_violated` |
| `metric_flag` | 1 = flagged, 0 = pass, NULL for `stars_family_violated` |

## Running Tests

```bash
pytest -v
```

## Open Items

- [ ] Wire `--collection-id` and `--run-id` into `actuals.sql` WHERE clause
- [ ] Confirm Snowflake table names and join keys in `actuals.sql`
