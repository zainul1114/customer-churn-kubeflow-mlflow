from model_registry import ModelRegistry


# ---------------------------------------------------------
# Kubeflow Model Registry
# ---------------------------------------------------------
REGISTRY_HOST = "http://127.0.0.1"
REGISTRY_PORT = 8081

registry = ModelRegistry(
    server_address=REGISTRY_HOST,
    port=REGISTRY_PORT,
    author="Azilehub Academy",
    is_secure=False,
)


# ---------------------------------------------------------
# Model information
# ---------------------------------------------------------
MODEL_NAME = "customer-churn-randomforest"
MODEL_VERSION = "v1.0.0"

MLFLOW_RUN_ID = "de79dc64abe8450790bbf92694df78d3"

MODEL_URI = (
    "s3://mlflow-artifacts/"
    "3/"
    f"{MLFLOW_RUN_ID}/"
    "artifacts/model"
)


# ---------------------------------------------------------
# Register model
# ---------------------------------------------------------
print("Registering model...")
print()
print("Model name :", MODEL_NAME)
print("Version    :", MODEL_VERSION)
print("MLflow run :", MLFLOW_RUN_ID)
print("Model URI  :", MODEL_URI)
print()

registered_model = registry.register_model(
    name=MODEL_NAME,
    uri=MODEL_URI,
    version=MODEL_VERSION,
    version_description=(
        "Customer Churn RandomForest model optimized "
        "using Kubeflow Katib Random Search and tracked "
        "using MLflow."
    ),
    model_format_name="sklearn",
    model_format_version="1.5.2",
    metadata={
        "framework": "scikit-learn",
        "model_type": "RandomForestClassifier",
        "optimization": "Katib Random Search",

        # MLflow
        "mlflow_run_id": MLFLOW_RUN_ID,

        # Katib
        "katib_best_accuracy": 0.635,

        # Model metrics
        "training_accuracy": 0.83,
        "test_accuracy": 0.625,
        "test_precision": 0.6125,
        "test_recall": 0.5268817204301075,
        "test_f1": 0.5664739884393064,

        # Optimized hyperparameters
        "n_estimators": 117,
        "max_depth": 7,
        "min_samples_split": 9,
    },
)

print("==============================================")
print("MODEL REGISTERED")
print("==============================================")
print("Name    :", registered_model.name)
print("ID      :", registered_model.id)
print("Owner   :", registered_model.owner)
print("State   :", registered_model.state)
