# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    AIRFLOW_HOME=/opt/airflow

# System dependencies with apt cache mount
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      libgomp1 \
      curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Upgrade pip using pip cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip

# Install Python requirements using pip cache mount
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt \
 && pip check

# Airflow runtime directory permissions
RUN mkdir -p /opt/airflow/logs /opt/airflow/dags \
 && chmod -R 777 /opt/airflow

EXPOSE 8888 8501 8080

CMD ["python", "-c", "print('CampusWell image is ready => Run: docker compose up -d')"]
