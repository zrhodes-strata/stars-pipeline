"""
job_definition.py
=================
Example SageMaker Processing Job definition for the STARS pipeline.

This script demonstrates how to launch the pipeline as a SageMaker Processing
Job using the boto3 SDK. It is a reference and documentation artifact —
not a production deployment script. Adapt it to your CI/CD tooling.

Prerequisites
-------------
1. Build and push the Docker image to Amazon ECR::

       docker build -t stars-pipeline .
       docker tag stars-pipeline <account>.dkr.ecr.<region>.amazonaws.com/stars-pipeline:latest
       docker push <account>.dkr.ecr.<region>.amazonaws.com/stars-pipeline:latest

2. Store Snowflake credentials (recommended: AWS Secrets Manager).
   The secret should be a JSON object with these exact keys::

       {
           "SNOWFLAKE_ACCOUNT":   "xy12345.us-east-1",
           "SNOWFLAKE_USER":      "svc_stars_pipeline",
           "SNOWFLAKE_PASSWORD":  "...",
           "SNOWFLAKE_WAREHOUSE": "STARS_WH",
           "SNOWFLAKE_DATABASE":  "DATALAKE_SANDBOX",
           "SNOWFLAKE_SCHEMA":    "RES"
       }

3. Create an IAM role for the Processing Job with:
   - ``AmazonSageMakerFullAccess`` (or a scoped equivalent)
   - ``s3:PutObject`` on the output bucket
   - ``secretsmanager:GetSecretValue`` if using Secrets Manager

Execution
---------
The pipeline CLI runs inside the container::

    python -m stars_pipeline.cli \\
        --strata-ids 84,14,1318 \\
        --date-from 2022-01-01 \\
        --output /opt/ml/processing/output/stars_results.csv

SageMaker mounts ``/opt/ml/processing/output/`` and uploads its contents
to the S3 URI you specify in ``ProcessingOutputConfig`` after the job ends.
"""
from __future__ import annotations

import json
from datetime import date

import boto3

# ── Configuration — update these for your environment ─────────────────────────

ECR_IMAGE_URI = "<account>.dkr.ecr.<region>.amazonaws.com/stars-pipeline:latest"
IAM_ROLE_ARN  = "arn:aws:iam::<account>:role/SageMakerProcessingRole"
OUTPUT_S3_URI = "s3://<bucket>/stars-pipeline/output/"
INSTANCE_TYPE = "ml.m5.large"  # adjust for workload size


# ── Credential helpers ────────────────────────────────────────────────────────


def get_snowflake_env_from_secrets_manager(
    secret_name: str,
    region: str,
) -> dict[str, str]:
    """
    Retrieve Snowflake credentials from AWS Secrets Manager.

    The secret must be a JSON object whose keys match the environment variable
    names expected by stars_pipeline/db.py:
        SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD,
        SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA

    Args:
        secret_name: Name or ARN of the secret in Secrets Manager.
        region:      AWS region where the secret is stored (e.g. "us-east-1").

    Returns:
        Dict mapping environment variable names to their string values.
        Pass the returned dict as the ``snowflake_env`` argument to
        ``launch_processing_job()``.
    """
    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])


# ── Job launcher ──────────────────────────────────────────────────────────────


def launch_processing_job(
    strata_ids: str,
    date_from: str = "2022-01-01",
    date_to: str | None = None,
    recent_days: int = 90,
    snowflake_env: dict[str, str] | None = None,
    job_name_prefix: str = "stars-pipeline",
) -> str:
    """
    Launch the STARS pipeline as a SageMaker Processing Job.

    The pipeline writes its output CSV to ``/opt/ml/processing/output/``
    inside the container. SageMaker automatically uploads that directory
    to ``OUTPUT_S3_URI`` after the job completes.

    Args:
        strata_ids:      Comma-separated strata IDs (e.g. ``"84,14,1318"``).
        date_from:       Start of the data pull window (YYYY-MM-DD).
        date_to:         End of the data pull window. Defaults to today.
        recent_days:     Size of the recent window for shift detection.
        snowflake_env:   Dict of Snowflake environment variables. If None,
                         credentials must be available in the container by
                         another means (e.g. instance profile, pre-baked image).
                         In production always supply credentials explicitly.
        job_name_prefix: Prefix for the SageMaker job name. The run date is
                         appended automatically.

    Returns:
        The SageMaker Processing Job name.
    """
    date_to = date_to or str(date.today())
    job_name = f"{job_name_prefix}-{date_to}"

    client = boto3.client("sagemaker")
    client.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": INSTANCE_TYPE,
                "VolumeSizeInGB": 20,
            }
        },
        AppSpecification={
            "ImageUri": ECR_IMAGE_URI,
            "ContainerArguments": [
                "--strata-ids",  strata_ids,
                "--date-from",   date_from,
                "--date-to",     date_to,
                "--recent-days", str(recent_days),
                "--output",      "/opt/ml/processing/output/stars_results.csv",
            ],
        },
        ProcessingOutputConfig={
            "Outputs": [
                {
                    "OutputName": "stars-results",
                    "S3Output": {
                        "S3Uri": OUTPUT_S3_URI,
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        },
        RoleArn=IAM_ROLE_ARN,
        Environment=snowflake_env or {},
    )
    print(f"Launched SageMaker Processing Job: {job_name}")
    return job_name


# ── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Fetch credentials from Secrets Manager and launch a job
    creds = get_snowflake_env_from_secrets_manager(
        secret_name="prod/snowflake/stars-pipeline",
        region="us-east-1",
    )
    launch_processing_job(
        strata_ids="84,14,1318",
        date_from="2022-01-01",
        snowflake_env=creds,
    )
