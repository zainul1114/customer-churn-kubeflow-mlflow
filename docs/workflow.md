# Customer Churn Workflow

## End-to-End Flow

```text
                    +----------------------+
                    | Customer Churn Data  |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Dataset Validation   |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Preprocessing        |
                    | Encoding             |
                    | Train/Test Split     |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | RandomForest Training|
                    +----------+-----------+
                               |
                     +---------+---------+
                     |                   |
                     v                   v
              +-------------+     +-------------+
              | Evaluation  |     | MLflow      |
              | Metrics     |     | Tracking    |
              +-------------+     +------+------+
                                         |
                                  +------+------+
                                  |             |
                                  v             v
                             PostgreSQL       MinIO
                             Metadata        Artifacts
```

## Current Platform

Kubeflow Community Distribution 26.03.1 is used for Kubernetes-native pipeline orchestration.

MLflow 3.16.1 is deployed separately in the `ml-registry` namespace.

The MLflow server uses:

- PostgreSQL 15 for backend metadata
- MinIO for artifact storage
- boto3/botocore for S3-compatible artifact operations
- psycopg2-binary for PostgreSQL connectivity

## Next Architecture

```text
Kubeflow Pipeline
       |
       v
    Katib HPO
       |
       v
Best Model
       |
       v
MLflow
       |
       v
Model Registry
       |
       v
KServe
       |
       v
Prediction API
```
