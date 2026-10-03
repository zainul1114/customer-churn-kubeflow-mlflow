from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import requests
import os
import time


app = FastAPI(
    title="Customer Churn Prediction API",
    description="FastAPI gateway for the Kubeflow/KServe Customer Churn model",
    version="1.1.0",
)


# =========================================================
# Configuration
# =========================================================

KSERVE_URL = os.getenv(
    "KSERVE_URL",
    "http://customer-churn-randomforest-predictor-00002-private:8012",
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "customer-churn-randomforest",
)


# =========================================================
# Prometheus Metrics
# =========================================================

API_REQUESTS = Counter(
    "customer_churn_api_requests_total",
    "Total HTTP requests received by Customer Churn API",
    ["method", "endpoint", "status"],
)


PREDICTIONS_TOTAL = Counter(
    "customer_churn_predictions_total",
    "Total customer churn predictions",
    ["prediction", "label"],
)


API_ERRORS = Counter(
    "customer_churn_api_errors_total",
    "Total Customer Churn API errors",
    ["endpoint", "error_type"],
)


PREDICTION_LATENCY = Histogram(
    "customer_churn_prediction_latency_seconds",
    "Prediction request latency in seconds",
)


KSERVE_READY = Gauge(
    "customer_churn_kserve_ready",
    "KServe model readiness: 1=ready, 0=not ready",
    ["model"],
)


# =========================================================
# Request model
# =========================================================

class CustomerData(BaseModel):

    age: int = Field(..., ge=18)

    tenure: int = Field(..., ge=0)

    monthly_charges: float = Field(..., ge=0)

    total_charges: float = Field(..., ge=0)

    support_calls: int = Field(..., ge=0)

    contract_type: str


# =========================================================
# HTTP Request Metrics Middleware
# =========================================================

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):

    start_time = time.time()

    try:

        response = await call_next(request)

        status = str(response.status_code)

        API_REQUESTS.labels(
            method=request.method,
            endpoint=request.url.path,
            status=status,
        ).inc()

        return response

    except Exception:

        API_REQUESTS.labels(
            method=request.method,
            endpoint=request.url.path,
            status="500",
        ).inc()

        raise

    finally:

        elapsed = time.time() - start_time

        if request.url.path == "/predict":

            PREDICTION_LATENCY.observe(elapsed)


# =========================================================
# Health
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "customer-churn-api",
    }


# =========================================================
# Prometheus Metrics
# =========================================================

@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# =========================================================
# KServe readiness
# =========================================================

@app.get("/model/ready")
def model_ready():

    url = (
        f"{KSERVE_URL}"
        f"/v2/models/{MODEL_NAME}/ready"
    )

    try:

        response = requests.get(
            url,
            timeout=5,
        )

        if response.status_code == 200:

            KSERVE_READY.labels(
                model=MODEL_NAME
            ).set(1)

            return {
                "model": MODEL_NAME,
                "ready": True,
                "kserve_response": response.json(),
            }

        KSERVE_READY.labels(
            model=MODEL_NAME
        ).set(0)

        raise HTTPException(
            status_code=503,
            detail={
                "model": MODEL_NAME,
                "ready": False,
            },
        )

    except requests.RequestException as exc:

        KSERVE_READY.labels(
            model=MODEL_NAME
        ).set(0)

        API_ERRORS.labels(
            endpoint="/model/ready",
            error_type="kserve_unavailable",
        ).inc()

        raise HTTPException(
            status_code=503,
            detail=f"KServe unavailable: {exc}",
        )


# =========================================================
# Feature Transformation
# =========================================================

def transform_features(customer: CustomerData):

    contract = customer.contract_type.strip()

    contract_encoding = {

        "Month-to-month": [1, 0, 0],

        "One year": [0, 1, 0],

        "Two year": [0, 0, 1],
    }

    if contract not in contract_encoding:

        API_ERRORS.labels(
            endpoint="/predict",
            error_type="invalid_contract_type",
        ).inc()

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid contract_type. "
                "Allowed values: Month-to-month, "
                "One year, Two year"
            ),
        )

    return [
        customer.age,
        customer.tenure,
        customer.monthly_charges,
        customer.total_charges,
        customer.support_calls,
        *contract_encoding[contract],
    ]


# =========================================================
# Prediction
# =========================================================

@app.post("/predict")
def predict(customer: CustomerData):

    features = transform_features(customer)

    payload = {
        "instances": [
            features
        ]
    }

    url = (
        f"{KSERVE_URL}"
        f"/v1/models/{MODEL_NAME}:predict"
    )

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=30,
        )

    except requests.RequestException as exc:

        API_ERRORS.labels(
            endpoint="/predict",
            error_type="kserve_unavailable",
        ).inc()

        raise HTTPException(
            status_code=503,
            detail=f"KServe unavailable: {exc}",
        )

    if response.status_code != 200:

        API_ERRORS.labels(
            endpoint="/predict",
            error_type="kserve_prediction_error",
        ).inc()

        raise HTTPException(
            status_code=502,
            detail={
                "message": "KServe prediction failed",
                "status_code": response.status_code,
                "response": response.text,
            },
        )

    result = response.json()

    prediction = result["predictions"][0]

    churn = bool(prediction)

    label = (
        "Churn"
        if churn
        else "No Churn"
    )

    PREDICTIONS_TOTAL.labels(
        prediction=str(prediction),
        label=label,
    ).inc()

    return {
        "prediction": int(prediction),
        "churn": churn,
        "prediction_label": label,
        "model": MODEL_NAME,
        "features": features,
    }
