from kfp import dsl, compiler
from kfp.dsl import Dataset, Model, Metrics, Input, Output

MLFLOW_TRACKING_URI = "http://mlflow.ml-registry.svc.cluster.local:5000"


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas==2.2.3",
        "numpy==1.26.4",
    ],
)
def create_dataset(
    dataset: Output[Dataset],
):
    import numpy as np
    import pandas as pd

    np.random.seed(42)

    n = 1000

    df = pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "age": np.random.randint(18, 75, n),
            "tenure": np.random.randint(1, 72, n),
            "monthly_charges": np.round(
                np.random.uniform(20, 150, n), 2
            ),
            "total_charges": np.round(
                np.random.uniform(100, 10000, n), 2
            ),
            "contract_type": np.random.choice(
                ["Month-to-month", "One year", "Two year"],
                n,
                p=[0.55, 0.25, 0.20],
            ),
            "support_calls": np.random.randint(0, 10, n),
        }
    )

    churn_probability = (
        0.15
        + 0.25 * (df["contract_type"] == "Month-to-month")
        + 0.03 * df["support_calls"]
        - 0.002 * df["tenure"]
    )

    churn_probability = np.clip(
        churn_probability,
        0.05,
        0.90,
    )

    df["churn"] = (
        np.random.random(n) < churn_probability
    ).astype(int)

    output_path = dataset.path + ".csv"
    df.to_csv(output_path, index=False)

    dataset.path = output_path


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas==2.2.3",
    ],
)
def validate_dataset(
    dataset: Input[Dataset],
    validated_dataset: Output[Dataset],
):
    import pandas as pd

    df = pd.read_csv(dataset.path)

    required_columns = [
        "customer_id",
        "age",
        "tenure",
        "monthly_charges",
        "total_charges",
        "contract_type",
        "support_calls",
        "churn",
    ]

    missing_columns = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values")

    if df.duplicated().sum() > 0:
        raise ValueError("Dataset contains duplicate rows")

    if not set(df["churn"].unique()).issubset({0, 1}):
        raise ValueError(
            "Churn must contain only 0 and 1"
        )

    output_path = (
        validated_dataset.path
        + ".csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    validated_dataset.path = output_path


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas==2.2.3",
        "numpy==1.26.4",
        "scikit-learn==1.5.2",
    ],
)
def preprocess_data(
    validated_dataset: Input[Dataset],
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(validated_dataset.path)

    df = df.drop(
        columns=["customer_id"]
    )

    df = pd.get_dummies(
        df,
        columns=["contract_type"],
        dtype=int,
    )

    X = df.drop(
        columns=["churn"]
    )

    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    train = X_train.copy()
    train["churn"] = y_train.values

    test = X_test.copy()
    test["churn"] = y_test.values

    train_path = train_dataset.path + ".csv"
    test_path = test_dataset.path + ".csv"

    train.to_csv(
        train_path,
        index=False,
    )

    test.to_csv(
        test_path,
        index=False,
    )

    train_dataset.path = train_path
    test_dataset.path = test_path


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas==2.2.3",
        "numpy==1.26.4",
        "scikit-learn==1.5.2",
        "joblib==1.4.2",
        "mlflow==3.16.1",
        "boto3",
    ],
)
def train_model(
    train_dataset: Input[Dataset],
    model: Output[Model],
):
    import os
    import joblib
    import mlflow
    import pandas as pd

    from sklearn.ensemble import RandomForestClassifier

    mlflow.set_tracking_uri(
        "http://mlflow.ml-registry.svc.cluster.local:5000"
    )

    mlflow.set_experiment(
        "customer-churn"
    )

    train_df = pd.read_csv(
        train_dataset.path
    )

    X_train = train_df.drop(
        columns=["churn"]
    )

    y_train = train_df["churn"]

    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
    )

    with mlflow.start_run(
        run_name="random-forest-training"
    ):

        classifier.fit(
            X_train,
            y_train,
        )

        training_accuracy = classifier.score(
            X_train,
            y_train,
        )

        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
                "training_rows": len(X_train),
                "feature_count": X_train.shape[1],
            }
        )

        mlflow.log_metric(
            "training_accuracy",
            training_accuracy,
        )

        mlflow.sklearn.log_model(
            classifier,
            artifact_path="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ],
        )

    model_path = model.path + ".joblib"

    joblib.dump(
        classifier,
        model_path,
    )

    model.path = model_path


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "pandas==2.2.3",
        "numpy==1.26.4",
        "scikit-learn==1.5.2",
        "joblib==1.4.2",
        "mlflow==3.16.1",
        "boto3",
    ],
)
def evaluate_model(
    test_dataset: Input[Dataset],
    model: Input[Model],
    metrics: Output[Metrics],
):
    import json
    import joblib
    import mlflow
    import pandas as pd

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )

    mlflow.set_tracking_uri(
        "http://mlflow.ml-registry.svc.cluster.local:5000"
    )

    test_df = pd.read_csv(
        test_dataset.path
    )

    classifier = joblib.load(
        model.path
    )

    X_test = test_df.drop(
        columns=["churn"]
    )

    y_test = test_df["churn"]

    predictions = classifier.predict(
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

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    metrics.log_metric(
        "accuracy",
        float(accuracy),
    )

    metrics.log_metric(
        "precision",
        float(precision),
    )

    metrics.log_metric(
        "recall",
        float(recall),
    )

    metrics.log_metric(
        "f1_score",
        float(f1),
    )

    with mlflow.start_run(
        run_name="random-forest-evaluation"
    ):

        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )

        confusion_file = "/tmp/confusion_matrix.json"

        with open(
            confusion_file,
            "w",
        ) as f:
            json.dump(
                matrix.tolist(),
                f,
                indent=2,
            )

        mlflow.log_artifact(
            confusion_file,
            artifact_path="evaluation",
        )


@dsl.pipeline(
    name="customer_churn_mlflow_pipeline",
    description=(
        "End-to-end Customer Churn "
        "Prediction using Kubeflow and MLflow"
    ),
)
def customer_churn_mlflow_pipeline():

    dataset_task = create_dataset()

    validation_task = validate_dataset(
        dataset=dataset_task.outputs["dataset"]
    )

    preprocess_task = preprocess_data(
        validated_dataset=(
            validation_task.outputs["validated_dataset"]
        )
    )

    train_task = train_model(
        train_dataset=(
            preprocess_task.outputs["train_dataset"]
        )
    )

    evaluate_model(
        test_dataset=(
            preprocess_task.outputs["test_dataset"]
        ),
        model=train_task.outputs["model"],
    )


if __name__ == "__main__":

    compiler.Compiler().compile(
        pipeline_func=customer_churn_mlflow_pipeline,
        package_path="customer_churn_mlflow_pipeline.yaml",
    )

    print(
        "Pipeline compiled successfully:"
        " customer_churn_mlflow_pipeline.yaml"
    )
