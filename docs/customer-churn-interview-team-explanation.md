# Customer Churn Prediction — Interview / Team Explanation Guide

## Purpose

This document explains how to present the **Customer Churn Prediction — End-to-End MLOps on Kubernetes** project to an interviewer, architect, manager, or team member.

The goal is not only to explain what tools were installed, but to explain:

1. **Why each stage exists**
2. **What we implemented**
3. **How the components communicate**
4. **What problems we encountered**
5. **How we solved them**
6. **What is completed**
7. **What is planned next**

---

# 1. One-Minute Project Introduction

### Interview answer

> "I built an end-to-end Customer Churn Prediction MLOps platform on Kubernetes using Kubeflow Community Distribution, Kubeflow Pipelines, Katib, MLflow, Kubeflow Model Registry, KServe, FastAPI, Prometheus, Grafana, GitHub Actions, GHCR, Kustomize, and Argo CD.
>
> I started with a reproducible Kubeflow pipeline that creates and validates the customer churn dataset, preprocesses the data, trains a Random Forest model, and evaluates it.
>
> Then I integrated Katib for hyperparameter optimization and MLflow for experiment tracking and model artifact management using PostgreSQL and MinIO.
>
> After selecting the optimized model, I registered it in Kubeflow Model Registry and served it using KServe. I then created a FastAPI prediction layer in front of KServe.
>
> For observability, I exposed application metrics to Prometheus and created Grafana dashboards for prediction counts, API traffic, latency, errors, and KServe model readiness.
>
> Finally, I implemented CI/CD and GitOps using GitHub Actions, GHCR, Kustomize, and Argo CD. A code change can go through testing and image build/publishing, followed by GitOps reconciliation and Kubernetes deployment.
>
> So the project covers the lifecycle from data and model development all the way to model serving, monitoring, and deployment automation."

---

# 2. Complete Architecture Explanation

The easiest way to explain the architecture is from **left to right** and then explain the **CI/CD path** separately.

```text
                         MACHINE LEARNING FLOW

 Customer Churn Dataset
          |
          v
 +-----------------------+
 | Kubeflow Pipelines    |
 |                       |
 | 1. Dataset Creation   |
 | 2. Validation         |
 | 3. Preprocessing      |
 | 4. Training           |
 | 5. Evaluation         |
 +-----------+-----------+
             |
             +----------------------+
             |                      |
             v                      v
        +---------+          +--------------+
        | Katib   |          | MLflow       |
        | HPO     |--------->| Tracking     |
        +---------+          +------+-------+
                                    |
                       +------------+------------+
                       |                         |
                       v                         v
                +-------------+           +-------------+
                | PostgreSQL  |           | MinIO       |
                | Metadata    |           | Artifacts   |
                +-------------+           +------+------+
                                                |
                                                v
                                    +----------------------+
                                    | Kubeflow Model       |
                                    | Registry             |
                                    +----------+-----------+
                                               |
                                               v
                                    +----------------------+
                                    | KServe               |
                                    | Model Serving        |
                                    +----------+-----------+
                                               |
                                               v
                                    +----------------------+
                                    | FastAPI              |
                                    | Prediction API       |
                                    +----------+-----------+
                                               |
                                               v
                                    +----------------------+
                                    | Prometheus + Grafana |
                                    | Monitoring           |
                                    +----------------------+


                         CI/CD + GITOPS FLOW

 Developer
     |
     v
 GitHub
     |
     v
 GitHub Actions
     |
     +---- pytest
     |
     +---- Docker Build
     |
     +---- Push Image
              |
              v
            GHCR
              |
              v
       GitOps Repository
              |
              v
           Argo CD
              |
              v
          Kubernetes
```

### Simple explanation

> "There are two major flows. The first is the ML lifecycle: Kubeflow orchestrates the workflow, Katib optimizes the model, MLflow tracks experiments and artifacts, Model Registry manages the model version, KServe serves it, and FastAPI exposes it to consumers. Prometheus and Grafana provide observability.
>
> The second is the application delivery flow: GitHub Actions tests and builds the API image, GHCR stores the image, the GitOps repository contains the desired Kubernetes configuration, and Argo CD reconciles that configuration into the cluster."

---

# 3. Stage 1 — Kubernetes and Kubeflow Foundation

## What was the objective?

The first objective was to create a Kubernetes-native ML platform instead of running the complete workflow manually from a notebook.

## What we implemented

Installed:

- Kubernetes cluster
- Kubeflow Community Distribution 26.03.1
- Kustomize 5.0.3
- Kubeflow Pipelines 2.16.0
- Katib 0.19.0
- KServe 0.18.0
- Kubeflow Model Registry 0.3.7

## Installation approach

```bash
git clone https://github.com/kubeflow/manifests.git

cd manifests

git checkout 26.03.1
```

Installation:

```bash
while ! kustomize build example | \
  kubectl apply --server-side --force-conflicts -f -; do
    echo "Retrying to apply resources..."
    sleep 15
done
```

## How to explain it

> "I selected the Kubeflow Community Distribution because I wanted the ML lifecycle to run natively on Kubernetes. Instead of treating Kubernetes only as a place to run containers, Kubeflow provides ML-specific orchestration such as Pipelines, Katib, Model Registry, and KServe."

## Interview question

### Why Kubernetes for ML?

> "Kubernetes provides scheduling, resource management, service discovery, workload isolation, scaling, persistent storage integration, and a common operational platform. Kubeflow builds ML lifecycle capabilities on top of that infrastructure."

---

# 4. Stage 2 — Customer Churn Dataset

## Objective

Create a reproducible dataset for the project.

The synthetic dataset contains:

```text
customer_id
age
tenure
monthly_charges
total_charges
contract_type
support_calls
churn
```

Example:

```text
age              -> customer age
tenure           -> duration with service
monthly_charges  -> monthly service charge
total_charges    -> accumulated charge
contract_type    -> Month-to-month / One year / Two year
support_calls    -> number of support calls
churn            -> target label
```

## Why synthetic data?

> "The purpose of this project is to demonstrate the complete MLOps engineering lifecycle. Synthetic data lets me reproduce the workflow without exposing customer information."

## Interview question

### What would change with real data?

> "The infrastructure workflow would remain similar, but I would add stronger data governance, access control, schema management, data quality checks, PII protection, data versioning, and production-grade drift and performance monitoring."

---

# 5. Stage 3 — Kubeflow Pipeline

## Objective

Make the ML workflow reproducible and executable as a pipeline.

The pipeline contains:

```text
create_dataset
      |
      v
validate_dataset
      |
      v
preprocess_data
      |
      v
train_model
      |
      v
evaluate_model
```

## Component 1 — Dataset Creation

Creates the customer churn dataset.

## Component 2 — Dataset Validation

Checks that the generated data has the expected structure and values.

## Component 3 — Preprocessing

The categorical field:

```text
contract_type
```

is one-hot encoded.

The model ultimately receives eight features:

```text
age
tenure
monthly_charges
total_charges
support_calls
contract_type_Month-to-month
contract_type_One year
contract_type_Two year
```

## Component 4 — Training

The project uses:

```text
RandomForestClassifier
```

## Component 5 — Evaluation

The pipeline calculates:

- Accuracy
- Precision
- Recall
- F1
- Confusion matrix

## How to explain it

> "The important point is that I did not keep the ML process as a sequence of manual notebook steps. I converted the process into reusable KFP components so the workflow can be executed repeatedly with consistent inputs and outputs."

---

# 6. Stage 4 — MLflow Integration

## Objective

Kubeflow orchestrates the workflow, but we also need experiment tracking.

This is where MLflow is used.

```text
Kubeflow Pipeline
       |
       v
MLflow Tracking Server
       |
       +---- PostgreSQL
       |
       +---- MinIO
```

## MLflow records

Training:

- model type
- `n_estimators`
- `max_depth`
- `random_state`
- training rows
- feature count
- training accuracy

Evaluation:

- accuracy
- precision
- recall
- F1
- confusion matrix artifact

## Important distinction

### PostgreSQL

Stores MLflow metadata such as:

```text
experiments
runs
parameters
metrics
metadata
```

### MinIO

Stores objects such as:

```text
model files
JSON files
CSV files
reports
evaluation artifacts
```

## Interview question

### Why not store the model directly in PostgreSQL?

> "PostgreSQL is appropriate for tracking metadata, but model binaries and large artifacts are better suited to object storage. Therefore PostgreSQL handles MLflow metadata while MinIO handles the actual artifacts."

---

# 7. Stage 5 — Katib Hyperparameter Optimization

## Objective

Instead of manually selecting Random Forest parameters, use Katib to search for better configurations.

Search parameters:

```text
n_estimators       50–200
max_depth           5–20
min_samples_split   2–10
```

Configuration:

```text
Algorithm: Random Search
maxTrialCount: 8
parallelTrialCount: 2
Objective: maximize accuracy
```

## Result

Best recorded Katib trial:

```text
Trial:
customer-churn-randomforest-n4zqb8cq

accuracy:
0.635

n_estimators:
117

max_depth:
7

min_samples_split:
9
```

## How to explain it

> "Katib separates hyperparameter search from the main training workflow. It creates multiple trials with different parameter combinations, evaluates them, and identifies the best configuration according to the objective metric."

## Important troubleshooting experience

Trial pods initially remained Pending because of CPU scheduling:

```text
0/2 nodes are available:
2 Insufficient cpu
```

The actual node utilization was not the issue. Kubernetes scheduling was based on the requested CPU.

We reduced the training request to:

```yaml
requests:
  cpu: "100m"
  memory: "256Mi"

limits:
  cpu: "1"
  memory: "1Gi"
```

The existing Katib Experiment also had immutable fields, so it had to be recreated rather than modifying restricted fields.

### Interview value

> "One practical Kubernetes lesson from this stage was that low actual CPU utilization does not necessarily mean a pod will schedule. Kubernetes schedules based on resource requests, not simply current utilization."

---

# 8. Stage 6 — Katib + MLflow Integration

After obtaining the optimized parameters, they were integrated into the MLflow training pipeline.

```text
Katib
  |
  | best parameters
  v
Kubeflow Pipeline
  |
  v
Random Forest
  |
  v
MLflow
  |
  +--> PostgreSQL
  |
  +--> MinIO
```

Parameters:

```text
n_estimators = 117
max_depth = 7
min_samples_split = 9
```

Successful MLflow run:

```text
Run:
de79dc64abe8450790bff92694df78d3
```

Metrics:

```text
training_accuracy = 0.8300
test_accuracy     = 0.6250
precision         = 0.6125
recall            = 0.5269
F1                = 0.5665
```

## Important troubleshooting

The first integrated run failed because the KFP training pod could create an MLflow run but could not upload artifacts to MinIO:

```text
botocore.exceptions.NoCredentialsError:
Unable to locate credentials
```

The solution was to create a Kubernetes Secret:

```text
mlflow-s3-credentials
```

and inject:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```

into the KFP training component.

## Interview explanation

> "Tracking metadata and artifact upload are two different operations. The KFP pod could reach the MLflow tracking server, but MLflow artifact upload went from the training pod to the S3-compatible MinIO endpoint, so the pod itself needed the S3 credentials."

---

# 9. Stage 7 — Distributed Training

## Current status

**Planned — not completed.**

This distinction is important in an interview.

Say:

> "Distributed training is the next ML-platform capability I identified, but I have not marked it as completed. The current Random Forest training is not a distributed training implementation."

Do not claim distributed training was implemented.

---

# 10. Stage 8 — Kubeflow Model Registry

## Objective

After training and tracking, we need a model identity and version that can be promoted toward serving.

Registered model:

```text
Name:
customer-churn-randomforest

Version:
v1.0.0
```

Metadata included:

```text
framework
model_type
optimization
mlflow_run_id
Katib accuracy
training accuracy
test accuracy
precision
recall
F1
hyperparameters
```

## Architecture

```text
MLflow
   |
   | model artifact URI
   v
Kubeflow Model Registry
   |
   | model version + metadata
   v
KServe
```

## Interview question

### Why do we need Model Registry if MLflow already has the model?

> "MLflow records experiments and model artifacts. The Model Registry provides a separate model lifecycle and identity layer. It gives the model a registered name and version and provides metadata that can be used as the model moves toward deployment."

---

# 11. Stage 9 — KServe Model Serving

## Objective

Serve the registered trained model inside Kubernetes.

Architecture:

```text
Model Registry
      |
      v
MinIO model artifact
      |
      v
KServe InferenceService
      |
      v
sklearn runtime
      |
      v
Prediction endpoint
```

The KServe service loaded:

```text
RandomForestClassifier
```

with:

```text
n_estimators = 117
max_depth = 7
min_samples_split = 9
```

## Important real-world troubleshooting

The first model artifact was:

```text
model.skops
```

The KServe sklearn runtime used in the environment expected a joblib/pickle-compatible model file and did not load the `.skops` artifact directly.

We created a conversion job:

```text
model.skops
     |
     v
skops.io.load()
     |
     v
RandomForestClassifier
     |
     v
joblib.dump()
     |
     v
model.joblib
```

The converted model was uploaded to MinIO and KServe was pointed to the new artifact location.

## Validation

KServe readiness:

```json
{
  "name": "customer-churn-randomforest",
  "ready": true
}
```

Prediction test:

```json
{
  "predictions": [0]
}
```

## Interview explanation

> "The important lesson here was that model serialization format and serving runtime compatibility matter. A model can be valid from the training framework's perspective but still require conversion for a particular serving runtime."

---

# 12. Stage 10 — FastAPI Prediction API

## Objective

KServe provides model serving, but I wanted an application-facing API layer with request validation and a stable application contract.

Architecture:

```text
Client
  |
  v
FastAPI
  |
  +--> Input validation
  |
  +--> Feature transformation
  |
  +--> KServe readiness check
  |
  v
KServe
  |
  v
Random Forest
```

Endpoints:

```text
GET  /health
GET  /model/ready
POST /predict
GET  /metrics
```

## Feature transformation

FastAPI converts:

```text
Month-to-month
```

to:

```text
[1, 0, 0]
```

```text
One year
```

to:

```text
[0, 1, 0]
```

```text
Two year
```

to:

```text
[0, 0, 1]
```

The final feature vector is:

```text
[
  age,
  tenure,
  monthly_charges,
  total_charges,
  support_calls,
  contract_type_Month-to-month,
  contract_type_One year,
  contract_type_Two year
]
```

## Example response

```json
{
  "prediction": 0,
  "churn": false,
  "prediction_label": "No Churn",
  "model": "customer-churn-randomforest",
  "features": [35, 12, 75.0, 900.0, 3, 1, 0, 0]
}
```

## Interview explanation

> "I intentionally kept FastAPI separate from KServe. KServe is responsible for model serving, while FastAPI provides application-level request validation, feature transformation, error handling, readiness checking, and an API contract for consumers."

---

# 13. Stage 11 — Prometheus and Grafana

## Objective

Monitor the prediction API and model-serving dependency.

Architecture:

```text
FastAPI
   |
   | /metrics
   v
ServiceMonitor
   |
   v
Prometheus
   |
   v
Grafana
```

Custom metrics:

```text
customer_churn_api_requests_total
customer_churn_predictions_total
customer_churn_api_errors_total
customer_churn_prediction_latency_seconds
customer_churn_kserve_ready
```

## Dashboard information

The dashboard tracks:

- total predictions
- churn predictions
- no-churn predictions
- KServe model readiness
- API request rate
- prediction rate
- prediction latency

## Example PromQL

Prediction count:

```promql
sum(customer_churn_predictions_total)
```

Prediction distribution:

```promql
sum by (label) (
  customer_churn_predictions_total
)
```

API traffic:

```promql
sum(rate(customer_churn_api_requests_total[5m]))
```

## Interview explanation

> "I did not monitor only Kubernetes infrastructure metrics. I also added business/application-level metrics such as prediction counts and prediction labels, because for an ML application it is useful to observe what the application is doing, not just whether the pod is running."

---

# 14. Stage 12 — GitHub Actions CI

## Objective

Automate testing and image creation.

Flow:

```text
Git Push
   |
   v
GitHub Actions
   |
   +--> Checkout
   |
   +--> Python 3.11
   |
   +--> Install dependencies
   |
   +--> pytest
   |
   +--> Docker build
   |
   +--> Push image to GHCR
```

The API test suite passed:

```text
8 passed
```

The workflow uses:

```text
.github/workflows/ci.yaml
```

## Interview explanation

> "The CI pipeline prevents me from manually building and testing the application every time I make a change. The code is checked automatically before the container image is published."

---

# 15. Stage 12 — GitHub Container Registry

The API image is published to GHCR.

Image:

```text
ghcr.io/zainul1114/customer-churn-kubeflow-mlflow/customer-churn-api:d2f416d
```

The image was successfully pulled from the Kubernetes environment.

## Why GHCR?

> "GHCR integrates naturally with the GitHub repository and GitHub Actions workflow. It gives the deployment platform a versioned container image instead of relying on a local Docker image."

---

# 16. Stage 12 — Kustomize and GitOps Repository

Application source and Kubernetes desired state were kept separate.

```text
customer_churn_project/
    api/
        main.py
        Dockerfile
        requirements.txt

gitops/
    customer-churn-api/
        base/
        overlays/
            dev/
```

## Why?

> "The application source code describes how the application is built. The GitOps repository structure describes how the application should be deployed. Keeping those concerns separated makes deployment changes auditable and reproducible."

The development overlay updates the image:

```yaml
images:
  - name: customer-churn-api
    newName: ghcr.io/zainul1114/customer-churn-kubeflow-mlflow/customer-churn-api
    newTag: "d2f416d"
```

---

# 17. Stage 12 — Argo CD

## Objective

Make Kubernetes deployment Git-driven.

Architecture:

```text
GitHub
   |
   v
GitOps manifests
   |
   v
Argo CD
   |
   v
Kubernetes
```

Application:

```text
customer-churn-dev
```

Final validation:

```text
Sync Status: Synced
Health Status: Healthy
```

Deployment rollout completed successfully.

## Interview explanation

> "Argo CD continuously compares the desired state stored in Git with the actual Kubernetes state. When the GitOps configuration changes, Argo CD reconciles the cluster toward that declared state."

---

# 18. Complete CI/CD + GitOps Explanation

This is a useful interview sequence to memorize:

```text
1. Developer changes FastAPI code
          |
2. Pushes code to GitHub
          |
3. GitHub Actions starts
          |
4. Python dependencies installed
          |
5. pytest executes
          |
6. Docker image is built
          |
7. Image is pushed to GHCR
          |
8. GitOps configuration references the image tag
          |
9. Git commit is pushed
          |
10. Argo CD detects Git change
          |
11. Kustomize renders manifests
          |
12. Kubernetes Deployment is updated
          |
13. New API pod starts
          |
14. Readiness probe passes
          |
15. Prometheus discovers the ServiceMonitor
          |
16. Grafana displays application metrics
```

### Short interview version

> "GitHub is the source of truth for code, GHCR is the container registry, the GitOps repository is the desired deployment state, and Argo CD is the reconciler that applies that state to Kubernetes."

---

# 19. Problems We Actually Solved

These are particularly valuable to discuss in an interview because they demonstrate troubleshooting rather than only installation.

## Problem 1 — Katib Trial Pods Pending

Error:

```text
Insufficient cpu
```

### Root cause

CPU requests were too high for the available node capacity.

### Solution

Reduced requests:

```yaml
requests:
  cpu: "100m"
  memory: "256Mi"
```

### Lesson

> Kubernetes scheduling considers resource requests.

---

## Problem 2 — Katib Experiment Immutable Fields

Attempting to modify certain Experiment fields produced:

```text
spec: Forbidden
```

### Root cause

Some Katib Experiment fields are immutable after creation.

### Solution

Delete and recreate the Experiment with the corrected resource configuration.

---

## Problem 3 — MLflow Artifact Upload Failure

Error:

```text
NoCredentialsError:
Unable to locate credentials
```

### Root cause

The KFP pod could reach MLflow but did not have credentials to upload artifacts directly to MinIO.

### Solution

Create:

```text
mlflow-s3-credentials
```

and inject the AWS-compatible credentials into the training component.

---

## Problem 4 — KServe Could Not Load `model.skops`

Error indicated the model file could not be located/loaded.

### Root cause

The KServe sklearn runtime in use expected joblib/pickle-compatible model files, while the MLflow model artifact was:

```text
model.skops
```

### Solution

Convert:

```text
model.skops
       |
       v
model.joblib
```

and serve the converted artifact.

---

## Problem 5 — KServe MinIO Access

Initial storage initialization returned:

```text
403 HeadBucket
```

### Root cause

The KServe storage initializer did not have the required S3 configuration propagated correctly.

### Solution

Configure the MinIO endpoint and credentials through the KServe-compatible Secret/ServiceAccount configuration and recreate the serving revision.

---

## Problem 6 — Kubeflow KServe UI Visibility

The working KServe InferenceService was initially deployed in:

```text
customer-churn
```

while the Kubeflow profile UI was looking at:

```text
kubeflow-user-example-com
```

### Solution

Deploy the working InferenceService configuration into the profile namespace so it appears in the Kubeflow KServe Endpoints UI.

---

## Problem 7 — KFP and MLflow Use Different Storage Paths

An important architecture discovery was that:

```text
KFP artifact storage
```

and:

```text
MLflow MinIO artifact storage
```

are separate.

KFP artifacts used the KCD storage integration, while MLflow used the separately deployed MinIO service.

### Lesson

> Similar-looking S3/MinIO configuration does not automatically mean two systems are using the same artifact store.

---

# 20. What Is Completed Today?

Use this answer when someone asks:

### "Where are you currently?"

> "The project is currently completed through Phase 12. The ML workflow, hyperparameter optimization, experiment tracking, model registry, model serving, prediction API, monitoring, CI, container registry, GitOps, and Argo CD deployment are all implemented and validated.
>
> The next stage is production hardening. Distributed training is also a separate planned milestone."

Completed:

```text
Phase 1   Dataset + Validation                 DONE
Phase 2   Preprocessing + Training             DONE
Phase 3   Evaluation                           DONE
Phase 4   MLflow Tracking                      DONE
Phase 5   PostgreSQL + MinIO                   DONE
Phase 6   Katib HPO                            DONE
Phase 6.1 Katib + MLflow                      DONE
Phase 7   Distributed Training                 PLANNED
Phase 8   Model Registry                       DONE
Phase 9   KServe                               DONE
Phase 10  FastAPI Prediction API               DONE
Phase 11  Prometheus + Grafana                 DONE
Phase 12  CI/CD + GitOps + Argo CD             DONE
Phase 13  Production Hardening                 NEXT
```

---

# 21. What Is Next?

## Phase 13 — Production Hardening

The next logical tasks are:

### 1. Stable KServe endpoint

The current FastAPI configuration uses a revision-specific private KServe service.

The goal is to remove that dependency and use a stable serving endpoint.

### 2. Secret management

Move development credentials toward a secure GitOps-compatible solution.

Possible direction:

```text
External Secrets / Sealed Secrets
```

### 3. Container security

Add:

- non-root execution
- security contexts
- image scanning
- SBOM
- dependency scanning
- least-privilege permissions

### 4. CI/CD hardening

Add:

```text
pytest
   |
security scan
   |
Docker build
   |
image scan
   |
GHCR
   |
GitOps promotion
```

### 5. Argo CD hardening

Review:

- RBAC
- automated sync
- self-healing
- pruning
- project boundaries
- deployment health checks

### 6. ML monitoring

Potential future monitoring:

```text
Data drift
Prediction drift
Feature distribution
Model performance
Business KPI correlation
```

### 7. Distributed training

Evaluate distributed training using an appropriate Kubernetes/Kubeflow training operator if required by the project scope.

---

# 22. Five-Minute Interview Walkthrough

If an interviewer asks:

### "Walk me through the project."

Use this structure:

### Step 1 — Business problem

> "The use case is customer churn prediction. We want to classify whether a customer is likely to churn based on customer attributes."

### Step 2 — ML workflow

> "I created a reproducible Kubeflow Pipeline with dataset creation, validation, preprocessing, Random Forest training, and evaluation."

### Step 3 — Optimization

> "I integrated Katib to automatically search Random Forest hyperparameters. The best recorded configuration was 117 estimators, depth 7, and minimum split 9."

### Step 4 — Experiment tracking

> "MLflow tracks the experiments, parameters, metrics, and models. PostgreSQL stores the tracking metadata and MinIO stores the artifacts."

### Step 5 — Model lifecycle

> "The optimized model was registered in Kubeflow Model Registry as `customer-churn-randomforest:v1.0.0`."

### Step 6 — Serving

> "KServe loads the model and exposes inference. Because the deployed sklearn runtime did not directly consume the `model.skops` artifact, I converted it to `model.joblib` and served that artifact."

### Step 7 — API

> "FastAPI sits in front of KServe. It validates customer input, performs feature transformation, checks model readiness, and forwards prediction requests."

### Step 8 — Monitoring

> "The API exposes Prometheus metrics for requests, predictions, errors, latency, and model readiness. Grafana visualizes those metrics."

### Step 9 — CI/CD

> "GitHub Actions runs the API tests, builds the container, and publishes it to GHCR."

### Step 10 — GitOps

> "The deployment manifests are maintained with Kustomize. Argo CD watches the GitOps state and reconciles the Kubernetes deployment."

### Final statement

> "So the project demonstrates an end-to-end ML lifecycle rather than only model training: orchestration, optimization, experiment tracking, model lifecycle, serving, API integration, observability, CI/CD, and GitOps."

---

# 23. Common Interview Questions and Answers

## Q1. Why did you use Kubeflow?

> Kubeflow provides Kubernetes-native ML workflow capabilities. In this project it provides Pipelines, Katib, Model Registry, and KServe integration points.

## Q2. What is the difference between Kubeflow Pipelines and MLflow?

> Kubeflow Pipelines orchestrates the workflow. MLflow tracks experiments, parameters, metrics, and model artifacts.

```text
Kubeflow = workflow orchestration

MLflow = experiment/model tracking
```

## Q3. What is Katib?

> Katib is Kubeflow's hyperparameter optimization component. It creates trials using different parameter combinations and optimizes an objective metric.

## Q4. Why PostgreSQL and MinIO together?

> PostgreSQL stores MLflow metadata. MinIO stores large binary and file artifacts.

## Q5. Why Model Registry?

> It gives the trained model a registered identity, version, and metadata and provides a lifecycle boundary between experimentation and serving.

## Q6. What is KServe?

> KServe is a Kubernetes-native model serving platform. It manages model-serving workloads and provides inference endpoints.

## Q7. Why put FastAPI in front of KServe?

> FastAPI provides application-level validation, feature transformation, API contracts, readiness checks, and custom application metrics. KServe remains responsible for model serving.

## Q8. What does Prometheus do?

> Prometheus collects time-series metrics from monitored endpoints.

## Q9. What does Grafana do?

> Grafana visualizes the collected metrics through dashboards and can support alerting workflows.

## Q10. What does Argo CD do?

> Argo CD implements GitOps continuous delivery by comparing Git-declared desired state with the actual Kubernetes state and reconciling differences.

## Q11. What is GitOps?

> Git is treated as the source of truth for the desired infrastructure/application configuration. Changes are reviewed and committed to Git, and a controller such as Argo CD reconciles the cluster.

## Q12. Why GHCR?

> GHCR provides versioned container image storage integrated with GitHub Actions and the GitHub repository.

## Q13. What was the most important troubleshooting issue?

> One significant issue was the MLflow artifact upload failure. The KFP pod could reach the MLflow server and create a run, but it could not upload artifacts to MinIO because it lacked S3 credentials. I fixed this by injecting the required credentials into the KFP training component.

## Q14. What Kubernetes troubleshooting did you perform?

> I investigated Pending pods, scheduling events, CPU requests, container logs, service endpoints, Secrets, namespaces, and rollout status. One important case was a Katib trial stuck because requested CPU exceeded what the scheduler could allocate.

## Q15. How do you distinguish an application problem from a Kubernetes problem?

> I check progressively: pod status, events, container logs, readiness/liveness probes, service endpoints, DNS/connectivity, application logs, and finally the application dependency itself. This separates scheduling, networking, configuration, and application-level failures.

## Q16. What would you improve for production?

> I would stabilize the KServe endpoint, implement secure secret management, add image scanning and SBOM generation, harden container security, improve CI/CD promotion, strengthen Argo CD RBAC and policies, and introduce model/data drift monitoring.

---

# 24. Strong Technical Summary

The project can be summarized as:

```text
                  MODEL DEVELOPMENT

Dataset
   |
   v
Kubeflow Pipeline
   |
   +--> Validation
   |
   +--> Preprocessing
   |
   +--> Training
   |
   +--> Evaluation
   |
   v
Katib
   |
   v
Optimized Parameters
   |
   v
MLflow
   |
   +--> PostgreSQL
   |
   +--> MinIO
   |
   v
Model Registry
   |
   v
KServe
   |
   v
FastAPI
   |
   v
Prediction


                  OPERATIONS

FastAPI
   |
   v
Prometheus
   |
   v
Grafana


                  DELIVERY

Developer
   |
   v
GitHub
   |
   v
GitHub Actions
   |
   +--> Test
   +--> Build
   +--> GHCR
   |
   v
GitOps Repository
   |
   v
Argo CD
   |
   v
Kubernetes
```

---

# 25. Final Interview Statement

A strong closing statement is:

> "The main value of this project is that I did not stop at training a machine-learning model. I built the surrounding engineering platform needed to operate the model: reproducible Kubeflow pipelines, automated hyperparameter optimization with Katib, experiment and artifact tracking with MLflow, model registration, Kubernetes-native serving with KServe, an application API with FastAPI, observability with Prometheus and Grafana, and automated delivery using GitHub Actions, GHCR, Kustomize, and Argo CD.
>
> I also worked through real integration issues involving Kubernetes scheduling, immutable Katib configuration, MLflow S3 credentials, model serialization compatibility, KServe storage initialization, namespace visibility, and GitOps deployment. Those troubleshooting steps were an important part of making the platform actually work end to end."

---

## Current Project Status

```text
+------------------------------------------------------+
|       CUSTOMER CHURN MLOps PROJECT STATUS            |
+------------------------------------------------------+
| Kubeflow Pipeline                  COMPLETED         |
| MLflow Tracking                    COMPLETED         |
| PostgreSQL + MinIO                 COMPLETED         |
| Katib HPO                          COMPLETED         |
| Katib + MLflow                     COMPLETED         |
| Distributed Training               PLANNED          |
| Model Registry                     COMPLETED         |
| KServe                             COMPLETED         |
| FastAPI                            COMPLETED         |
| Prometheus + Grafana               COMPLETED         |
| GitHub Actions                     COMPLETED         |
| GHCR                               COMPLETED         |
| Kustomize GitOps                   COMPLETED         |
| Argo CD                            COMPLETED         |
| Kubernetes Deployment              COMPLETED         |
| Production Hardening               NEXT             |
+------------------------------------------------------+
```

**Key message to remember:**  
**"I built and validated the complete path from data → ML workflow → optimization → tracking → registry → serving → API → monitoring → CI/CD → GitOps → Kubernetes."**
