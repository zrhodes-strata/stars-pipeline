FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached across code-only rebuilds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the package in editable mode
COPY . .
RUN pip install --no-cache-dir -e .

# Disable Python output buffering so logs reach CloudWatch in real time
ENV PYTHONUNBUFFERED=1

# Default entry point — override container arguments in the SageMaker job definition
ENTRYPOINT ["python", "-m", "stars_pipeline.cli"]
