# Customer Churn Prediction — Kubeflow + MLflow

An end-to-end Machine Learning workflow built on **Kubeflow Community Distribution (KCD) 26.03.1** and **MLflow 3.16.1**.

The project demonstrates how Kubernetes-native ML orchestration can be combined with experiment tracking, model artifacts, PostgreSQL metadata storage, and MinIO object storage.

![CCP MLOps Archtecture](../images/ccp_mlops_arch.png)

## Project Objective

The objective of this project is to build a reproducible, Kubernetes-native Customer Churn Prediction workflow that covers the complete ML lifecycle from dataset creation and validation through preprocessing, model training, evaluation, experiment tracking, and artifact management.

The project currently focuses on establishing a reliable foundation with:

- Kubeflow Pipelines for workflow orchestration
- MLflow for experiment and run tracking
- PostgreSQL for MLflow metadata
- MinIO for MLflow artifacts and model files
- Scikit-learn RandomForest for the churn model
- KFP Dataset, Model, and Metrics artifacts

The next planned stages are Katib hyperparameter optimization, distributed training, Kubeflow Model Registry, and KServe model serving.

## Current Workflow

```text
Customer Churn Dataset
        |
        v
+-------------------+
| Data Validation   |
+-------------------+
        |
        v
+-------------------+
| Preprocessing     |
| - Encode category |
| - Train/Test split|
+-------------------+
        |
        v
+-------------------+
| Model Training    |
| RandomForest      |
+-------------------+
        |
        +----------------------+
        |                      |
        v                      v
+-------------------+   +-------------------+
| Model Evaluation  |   | MLflow Tracking   |
| Accuracy          |   | Parameters        |
| Precision         |   | Metrics           |
| Recall            |   | Model             |
| F1                |   | Artifacts         |
+-------------------+   +---------+---------+
                                  |
                         +--------+--------+
                         |                 |
                         v                 v
                  +------------+    +-------------+
                  | PostgreSQL |    |    MinIO    |
                  | Metadata   |    | Artifacts   |
                  +------------+    +-------------+
```

## Tools and Technologies

| Area | Technology |
|---|---|
| Container orchestration | Kubernetes |
| ML platform | Kubeflow Community Distribution 26.03.1 |
| Pipeline orchestration | Kubeflow Pipelines 2.16.0 |
| Experiment tracking | MLflow 3.16.1 |
| ML framework | Scikit-learn 1.5.2 |
| Programming | Python 3.11 |
| Metadata database | PostgreSQL 15 |
| Object storage | MinIO |
| ML model | RandomForestClassifier |
| Model serialization | MLflow / joblib |
| Artifact SDK | boto3 / botocore |
| Pipeline compiler | KFP SDK |


## Kubeflow Installation

This project uses **Kubeflow Community Distribution (KCD) 26.03.1** installed from the Kubeflow manifests repository.

The official Kubeflow Community Distribution uses Kustomize-based installation and recommends checking the Kubernetes version supported by the specific release before installation. The manifests installation can be applied with `kustomize build ... | kubectl apply ...`; the upstream documentation also notes that CRD/resource readiness can require re-applying the command. citeturn0search1turn0search0

### Prerequisites

Before installing Kubeflow, prepare a Kubernetes cluster with:

- Kubernetes cluster with a working `kubectl` context
- A default `StorageClass`
- Linux-based Kubernetes worker nodes
- Sufficient CPU and memory for the Kubeflow components
- Internet access to pull the required container images
- `kubectl`
- `kustomize`
- Git
- A user with permission to create cluster-wide Kubernetes resources

For this project, the installation was performed with **KCD 26.03.1** and the project environment was validated on a Kubernetes 1.34+ cluster.

Check the cluster:

```bash
kubectl cluster-info
kubectl get nodes
kubectl get storageclass
kubectl version
```

Verify that a default StorageClass exists:

```bash
kubectl get storageclass
```

Example:

```text
NAME                 PROVISIONER
local-path (default) rancher.io/local-path
```

> **Note:** Resource requirements depend on which Kubeflow components are enabled. The upstream Kubeflow documentation recommends at least 16 GB RAM and 8 CPU cores for the full single-command installation on Kind; smaller installations can be configured by excluding components. citeturn0search1

### Install Kustomize

This project used Kustomize **v5.0.3**:

```bash
wget https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2Fv5.0.3/kustomize_v5.0.3_linux_amd64.tar.gz

tar -xzf kustomize_v5.0.3_linux_amd64.tar.gz

sudo mv kustomize /usr/local/bin/

kustomize version
```

Verify:

```bash
which kustomize
kustomize version
kubectl version
```

### Clone Kubeflow Community Distribution

Clone the Kubeflow manifests repository:

```bash
git clone https://github.com/kubeflow/manifests.git
cd manifests
```

Checkout the project-tested KCD release:

```bash
git checkout 26.03.1
```

Verify:

```bash
git branch --show-current
git describe --tags --always
```

### Install Kubeflow

The complete Kubeflow platform can be installed from the `example` Kustomization.

For this project, the installation command was:

```bash
while ! kustomize build example | kubectl apply --server-side --force-conflicts -f -; do
    echo "Retrying to apply resources..."
    sleep 15
done
```

The retry loop is intentional. Kubeflow contains many CRDs, webhooks, controllers, and dependent resources, so some resources may not be ready during the first application. The upstream manifests documentation also recommends retrying when resources are not yet ready. citeturn0search1

### Verify Kubeflow Installation

Check the main Kubeflow namespace:

```bash
kubectl get pods -n kubeflow
```

Check the supporting namespaces:

```bash
kubectl get pods -n cert-manager
kubectl get pods -n istio-system
kubectl get pods -n auth
kubectl get pods -n oauth2-proxy
kubectl get pods -n knative-serving
kubectl get pods -n kubeflow
kubectl get pods -n kubeflow-user-example-com
```

Check all namespaces:

```bash
kubectl get pods -A
```

Check Kubeflow services:

```bash
kubectl get svc -n kubeflow
```

### Access Kubeflow Dashboard

For local access, port-forward the dashboard service:

```bash
kubectl port-forward -n kubeflow svc/dashboard 8080:80
```

Open:

```text
http://localhost:8080
```

### Kubeflow Component Port Forwards Used in This Project

The following services were used while developing and testing the project:

```bash
# Kubeflow Dashboard
kubectl port-forward -n kubeflow svc/dashboard 8080:80

# Jupyter Web App
kubectl port-forward -n kubeflow svc/jupyter-web-app-service 8081:80

# Kubeflow Pipelines UI
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8082:80

# Katib UI
kubectl port-forward -n kubeflow svc/katib-ui 8083:80

# KServe Models UI
kubectl port-forward -n kubeflow svc/kserve-models-web-application 8084:80

# TensorBoard
kubectl port-forward -n kubeflow svc/tensorboards-web-app-service 8085:80

# Volumes Web App
kubectl port-forward -n kubeflow svc/volumes-web-app-service 8086:80

# Model Catalog
kubectl port-forward -n kubeflow svc/model-catalog 8087:8080
```

### Kubeflow Components Used by This Project

The KCD installation provides the platform components used by the Customer Churn project, including:

```text
Kubeflow Dashboard
       |
       +-- Kubeflow Pipelines
       |
       +-- Jupyter
       |
       +-- Katib
       |
       +-- KServe
       |
       +-- TensorBoard
       |
       +-- Model Registry
       |
       +-- Model Catalog
```

The current project has primarily used **Kubeflow Pipelines** so far. Katib, Model Registry, and KServe are part of the planned next stages.

### Verify Kubeflow Pipeline Service

Check:

```bash
kubectl get pods -n kubeflow | grep -E 'ml-pipeline|ml-pipeline-ui'
```

Check services:

```bash
kubectl get svc -n kubeflow | grep ml-pipeline
```

The pipeline UI can be accessed through:

```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8082:80
```

Then open:

```text
http://localhost:8082
```

### Installation Troubleshooting

If the installation reports an error such as:

```text
resource mapping not found
```

or:

```text
no matches for kind
```

the required CRD may not have become ready before a dependent resource was created.

Re-run the installation command:

```bash
while ! kustomize build example | kubectl apply --server-side --force-conflicts -f -; do
    echo "Retrying to apply resources..."
    sleep 15
done
```

Then check:

```bash
kubectl get pods -A
kubectl get crd
```

The upstream Kubeflow documentation specifically notes that initial `kubectl apply` failures can occur because CRDs and dependent resources become ready at different times. citeturn0search1


## Kubeflow Pipeline

The pipeline contains:

1. `create_dataset`
2. `validate_dataset`
3. `preprocess_data`
4. `train_model`
5. `evaluate_model`

The workflow uses KFP artifact inputs and outputs rather than relying on a shared notebook filesystem.

## MLflow Integration

The pipeline sends experiment information to:

```text
http://mlflow.ml-registry.svc.cluster.local:5000
```

The MLflow experiment is:

```text
customer-churn
```

Training records include:

- model type
- number of estimators
- maximum depth
- random state
- training rows
- feature count
- training accuracy

Evaluation records include:

- accuracy
- precision
- recall
- F1 score
- confusion matrix artifact

## MLflow Storage Architecture

```text
MLflow
  |
  +---- PostgreSQL
  |       |
  |       +-- experiments
  |       +-- runs
  |       +-- parameters
  |       +-- metrics
  |       +-- model metadata
  |
  +---- MinIO
          |
          +-- ML artifacts
          +-- trained models
          +-- datasets
          +-- reports
          +-- JSON/CSV/TXT files
```

## Verified Implementation

The following have been successfully validated:

- Kubeflow Pipeline execution
- Customer churn dataset generation
- Dataset validation
- Preprocessing
- RandomForest training
- Model evaluation
- MLflow experiment creation
- MLflow run creation
- Parameter logging
- Metric logging
- Model logging
- MinIO artifact upload
- MLflow artifact listing
- PostgreSQL MLflow metadata storage

A successful integrated Kubeflow + MLflow pipeline run was completed on October 2, 2026.

## Repository Structure

```text
customer-churn-kubeflow-mlflow/
├── README.md
├── pipelines/
│   └── customer_churn_mlflow_pipeline.py
├── tests/
│   └── mlflow_artifact_test.py
├── mlflow/
│   ├── Dockerfile
│   └── mlflow-registry.yaml
├── docs/
│   └── workflow.md
└── .gitignore
```

## Run the Pipeline

Compile:

```bash
python pipelines/customer_churn_mlflow_pipeline.py
```

This generates:

```text
customer_churn_mlflow_pipeline.yaml
```

Upload the generated YAML to Kubeflow Pipelines and create a run.

## MLflow Artifact Test

The repository also includes a standalone artifact validation script:

```bash
python tests/mlflow_artifact_test.py
```

It creates and logs:

- CSV dataset
- JSON metrics
- JSON model configuration
- CSV predictions
- text report
- trained RandomForest model

## Planned Roadmap

```text
Phase 1  Dataset + Validation                 [Completed]
Phase 2  Preprocessing + Training             [Completed]
Phase 3  Evaluation                           [Completed]
Phase 4  MLflow Tracking                      [Completed]
Phase 5  PostgreSQL + MinIO Artifacts         [Completed]
Phase 6  Katib Hyperparameter Optimization    [Next]
Phase 7  Distributed Training                 [Planned]
Phase 8  Kubeflow Model Registry              [Planned]
Phase 9  KServe Model Serving                 [Planned]
Phase 10 Prediction API                       [Planned]
Phase 11 Monitoring + Observability            [Planned]
Phase 12 CI/CD + GitOps                       [Planned]
```

## Author

Azilehub Academy

Focused on DevOps, MLOps, Kubernetes, AI Infrastructure, HPC, and cloud-native engineering.
