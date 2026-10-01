import json
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


MLFLOW_TRACKING_URI = (
    "http://mlflow.ml-registry.svc.cluster.local:5000"
)

EXPERIMENT_NAME = "mlflow-artifact-test"
RUN_NAME = "artifact-upload-test"

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)

work_dir = Path(
    tempfile.mkdtemp(
        prefix="mlflow-artifact-test-"
    )
)

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    random_state=42,
)

feature_names = [
    f"feature_{i}"
    for i in range(1, 11)
]

df = pd.DataFrame(
    X,
    columns=feature_names,
)

df["target"] = y

dataset_file = (
    work_dir / "training_dataset.csv"
)

df.to_csv(
    dataset_file,
    index=False,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
)

model.fit(
    X_train,
    y_train,
)

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions,
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0,
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0,
)

metrics_file = work_dir / "metrics.json"

with open(metrics_file, "w") as f:
    json.dump(
        {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        },
        f,
        indent=4,
    )

config_file = (
    work_dir / "model_config.json"
)

with open(config_file, "w") as f:
    json.dump(
        {
            "model_type": "RandomForestClassifier",
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
        },
        f,
        indent=4,
    )

prediction_file = (
    work_dir / "predictions.csv"
)

pd.DataFrame(
    {
        "actual": y_test,
        "prediction": predictions,
    }
).to_csv(
    prediction_file,
    index=False,
)

report_file = (
    work_dir / "model_report.txt"
)

with open(report_file, "w") as f:
    f.write(
        "MLflow Model Training Report\n"
    )
    f.write(
        "============================\n"
    )
    f.write(
        f"Accuracy : {accuracy:.4f}\n"
    )
    f.write(
        f"Precision: {precision:.4f}\n"
    )
    f.write(
        f"Recall   : {recall:.4f}\n"
    )
    f.write(
        f"F1 Score : {f1:.4f}\n"
    )

with mlflow.start_run(
    run_name=RUN_NAME
) as run:

    mlflow.log_params(
        {
            "model_type": "RandomForestClassifier",
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
        }
    )

    mlflow.log_metrics(
        {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
    )

    mlflow.log_artifact(
        str(dataset_file),
        artifact_path="dataset",
    )

    mlflow.log_artifact(
        str(metrics_file),
        artifact_path="metrics",
    )

    mlflow.log_artifact(
        str(config_file),
        artifact_path="configuration",
    )

    mlflow.log_artifact(
        str(prediction_file),
        artifact_path="predictions",
    )

    mlflow.log_artifact(
        str(report_file),
        artifact_path="reports",
    )

    mlflow.sklearn.log_model(
        model,
        name="random_forest_model",
    )

    mlflow.set_tag(
        "project",
        "customer-churn",
    )

    mlflow.set_tag(
        "test_type",
        "artifact-upload-test",
    )

    print("=" * 70)
    print("MLflow artifact test completed")
    print("=" * 70)
    print(f"Run ID       : {run.info.run_id}")
    print(
        f"Artifact URI : {mlflow.get_artifact_uri()}"
    )
    print("=" * 70)
