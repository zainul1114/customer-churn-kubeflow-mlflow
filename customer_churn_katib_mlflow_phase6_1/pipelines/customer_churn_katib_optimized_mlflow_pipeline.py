from kfp import dsl, compiler, kubernetes
from kfp.dsl import component, Dataset, Input, Output, Model, Metrics


# ============================================================
# 1. Create Dataset
# ============================================================

@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas==2.2.3",
        "numpy==1.26.4",
        "scikit-learn==1.5.2",
    ],
)
def create_dataset(
    output_dataset: Output[Dataset],
):
    import os
    import pandas as pd
    import numpy as np

    np.random.seed(42)

    n = 1000

    age = np.random.randint(18, 75, n)

    tenure = np.random.randint(1, 72, n)

    monthly_charges = np.round(
        np.random.uniform(20, 150, n),
        2,
    )

    total_charges = np.maximum(
        np.round(
            monthly_charges * tenure
            + np.random.normal(0, 100, n),
            2,
        ),
        monthly_charges,
    )

    contract_type = np.random.choice(
        [
            "Month-to-month",
            "One year",
            "Two year",
        ],
        n,
        p=[
            0.55,
            0.25,
            0.20,
        ],
    )

    support_calls = np.random.randint(
        0,
        12,
        n,
    )

    churn_probability = (
        0.15
        + (contract_type == "Month-to-month") * 0.20
        + (monthly_charges > 90) * 0.15
        + (support_calls >= 6) * 0.20
        + (tenure < 12) * 0.15
    )

    churn = (
        np.random.random(n)
        < churn_probability
    ).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [
                f"CUST-{i:05d}"
                for i in range(1, n + 1)
            ],
            "age": age,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "support_calls": support_calls,
            "churn": churn,
        }
    )

    os.makedirs(
        os.path.dirname(output_dataset.path),
        exist_ok=True,
    )

    df.to_csv(
        output_dataset.path,
        index=False,
    )


# ============================================================
# 2. Preprocess Dataset
# ============================================================

@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas==2.2.3",
        "numpy==1.26.4",
        "scikit-learn==1.5.2",
    ],
)
def preprocess_data(
    input_dataset: Input[Dataset],
    output_dataset: Output[Dataset],
):
    import os
    import pandas as pd

    from sklearn.model_selection import (
        train_test_split,
    )

    df = pd.read_csv(
        input_dataset.path,
    )

    df["contract_type"] = (
        df["contract_type"].astype(str)
    )

    df = pd.get_dummies(
        df,
        columns=["contract_type"],
        dtype=int,
    )

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["churn"],
    )

    result = pd.concat(
        [
            train_df.assign(
                _split="train"
            ),
            test_df.assign(
                _split="test"
            ),
        ],
        ignore_index=True,
    )

    os.makedirs(
        os.path.dirname(output_dataset.path),
        exist_ok=True,
    )

    result.to_csv(
        output_dataset.path,
        index=False,
    )


# ============================================================
# 3. Katib Optimized Training + MLflow
# ============================================================

@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas==2.2.3",
        "numpy==1.26.4",
        "scikit-learn==1.5.2",
        "mlflow==3.16.1",
        "psycopg2-binary==2.9.13",
        "boto3",
        "botocore",
    ],
)
def train_optimized_model(
    input_dataset: Input[Dataset],
    n_estimators: int,
    max_depth: int,
    min_samples_split: int,
    model: Output[Model],
    metrics: Output[Metrics],
):
    import os
    import json
    import pandas as pd

    import mlflow
    import mlflow.sklearn

    from sklearn.ensemble import (
        RandomForestClassifier,
    )

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    # --------------------------------------------------------
    # MLflow configuration
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        "http://mlflow.ml-registry.svc.cluster.local:5000"
    )

    # MLflow artifact store = separate MinIO
    #
    # This is NOT KFP/SeaweedFS.
    #
    os.environ[
        "MLFLOW_S3_ENDPOINT_URL"
    ] = (
        "http://minio.ml-registry.svc.cluster.local:9000"
    )

    os.environ[
        "AWS_DEFAULT_REGION"
    ] = "us-east-1"

    mlflow.set_experiment(
        "customer-churn"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(
        input_dataset.path,
    )

    train_df = df[
        df["_split"] == "train"
    ].copy()

    test_df = df[
        df["_split"] == "test"
    ].copy()

    drop_columns = [
        "customer_id",
        "churn",
        "_split",
    ]

    X_train = train_df.drop(
        columns=drop_columns
    )

    y_train = train_df["churn"]

    X_test = test_df.drop(
        columns=drop_columns
    )

    y_test = test_df["churn"]

    # --------------------------------------------------------
    # MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="katib-optimized-randomforest"
    ) as run:

        classifier = RandomForestClassifier(
            n_estimators=int(
                n_estimators
            ),
            max_depth=int(
                max_depth
            ),
            min_samples_split=int(
                min_samples_split
            ),
            random_state=42,
        )

        classifier.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        train_predictions = (
            classifier.predict(X_train)
        )

        test_predictions = (
            classifier.predict(X_test)
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        train_accuracy = accuracy_score(
            y_train,
            train_predictions,
        )

        test_accuracy = accuracy_score(
            y_test,
            test_predictions,
        )

        test_precision = precision_score(
            y_test,
            test_predictions,
            zero_division=0,
        )

        test_recall = recall_score(
            y_test,
            test_predictions,
            zero_division=0,
        )

        test_f1 = f1_score(
            y_test,
            test_predictions,
            zero_division=0,
        )

        # ----------------------------------------------------
        # MLflow parameters
        # ----------------------------------------------------

        mlflow.log_params(
            {
                "model_type":
                    "RandomForestClassifier",

                "optimization":
                    "Katib Random Search",

                "n_estimators":
                    int(n_estimators),

                "max_depth":
                    int(max_depth),

                "min_samples_split":
                    int(min_samples_split),

                "random_state":
                    42,

                "training_rows":
                    len(X_train),

                "test_rows":
                    len(X_test),

                "feature_count":
                    X_train.shape[1],
            }
        )

        # ----------------------------------------------------
        # MLflow metrics
        # ----------------------------------------------------

        mlflow.log_metrics(
            {
                "training_accuracy":
                    float(train_accuracy),

                "test_accuracy":
                    float(test_accuracy),

                "test_precision":
                    float(test_precision),

                "test_recall":
                    float(test_recall),

                "test_f1":
                    float(test_f1),
            }
        )

        # ----------------------------------------------------
        # Log model to MLflow MinIO
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            classifier,
            artifact_path="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ],
        )

        # ----------------------------------------------------
        # Save result metadata
        # ----------------------------------------------------

        result = {
            "run_id":
                run.info.run_id,

            "optimization":
                "Katib Random Search",

            "n_estimators":
                int(n_estimators),

            "max_depth":
                int(max_depth),

            "min_samples_split":
                int(min_samples_split),

            "training_accuracy":
                float(train_accuracy),

            "test_accuracy":
                float(test_accuracy),

            "test_precision":
                float(test_precision),

            "test_recall":
                float(test_recall),

            "test_f1":
                float(test_f1),
        }

        result_path = os.path.join(
            os.path.dirname(model.path),
            "optimized_model_result.json",
        )

        with open(
            result_path,
            "w",
        ) as f:
            json.dump(
                result,
                f,
                indent=2,
            )

        mlflow.log_artifact(
            result_path,
            artifact_path="evaluation",
        )

        # ----------------------------------------------------
        # KFP metrics
        # ----------------------------------------------------

        metrics.log_metric(
            "training_accuracy",
            float(train_accuracy),
        )

        metrics.log_metric(
            "test_accuracy",
            float(test_accuracy),
        )

        metrics.log_metric(
            "test_precision",
            float(test_precision),
        )

        metrics.log_metric(
            "test_recall",
            float(test_recall),
        )

        metrics.log_metric(
            "test_f1",
            float(test_f1),
        )

        # ----------------------------------------------------
        # KFP model artifact
        # ----------------------------------------------------

        os.makedirs(
            os.path.dirname(model.path),
            exist_ok=True,
        )

        with open(
            model.path,
            "w",
        ) as f:
            json.dump(
                result,
                f,
                indent=2,
            )


# ============================================================
# 4. Pipeline
# ============================================================

@dsl.pipeline(
    name="customer-churn-katib-optimized-mlflow",

    description=(
        "Final RandomForest training using "
        "Katib-selected hyperparameters "
        "and MLflow tracking."
    ),
)
def customer_churn_katib_optimized_mlflow_pipeline(
    n_estimators: int = 117,
    max_depth: int = 7,
    min_samples_split: int = 9,
):

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = create_dataset()

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    preprocessed = preprocess_data(
        input_dataset=(
            dataset.outputs["output_dataset"]
        )
    )

    # --------------------------------------------------------
    # Optimized training
    # --------------------------------------------------------

    train_task = train_optimized_model(
        input_dataset=(
            preprocessed.outputs[
                "output_dataset"
            ]
        ),

        n_estimators=n_estimators,

        max_depth=max_depth,

        min_samples_split=min_samples_split,
    )

    # --------------------------------------------------------
    # Kubernetes Secret
    # --------------------------------------------------------
    #
    # Secret:
    #
    # kubeflow-user-example-com/
    #   mlflow-s3-credentials
    #
    # Keys:
    #
    # AWS_ACCESS_KEY_ID
    # AWS_SECRET_ACCESS_KEY
    #
    # --------------------------------------------------------

    kubernetes.use_secret_as_env(
        train_task,

        secret_name=(
            "mlflow-s3-credentials"
        ),

        secret_key_to_env={
            "AWS_ACCESS_KEY_ID":
                "AWS_ACCESS_KEY_ID",

            "AWS_SECRET_ACCESS_KEY":
                "AWS_SECRET_ACCESS_KEY",
        },
    )


# ============================================================
# 5. Compile Pipeline
# ============================================================

if __name__ == "__main__":

    compiler.Compiler().compile(
        pipeline_func=(
            customer_churn_katib_optimized_mlflow_pipeline
        ),

        package_path=(
            "customer_churn_katib_optimized_mlflow_pipeline.yaml"
        ),
    )
