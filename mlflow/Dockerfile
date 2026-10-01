FROM ghcr.io/mlflow/mlflow:v3.16.1

RUN pip install --no-cache-dir \
    psycopg2-binary==2.9.13 \
    boto3

RUN python - <<'PY'
import mlflow
import psycopg2
import boto3
import botocore

print("MLflow:", mlflow.__version__)
print("psycopg2:", psycopg2.__version__)
print("boto3:", boto3.__version__)
print("botocore:", botocore.__version__)
PY
