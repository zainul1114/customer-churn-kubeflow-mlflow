import os
import time

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from starlette.middleware.base import BaseHTTPMiddleware


# ============================================================
# Configuration
# ============================================================

KSERVE_URL = os.getenv(
    "KSERVE_URL",
    "http://knative-local-gateway.istio-system.svc.cluster.local:80",
)

KSERVE_HOST = os.getenv(
    "KSERVE_HOST",
    "customer-churn-randomforest-predictor.kubeflow-user-example-com.example.com",
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "customer-churn-randomforest",
)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    version="2.0.0",
    description="FastAPI prediction service backed by KServe",
)


# ============================================================
# Prometheus metrics
# ============================================================

REQUEST_COUNT = Counter(
    "customer_churn_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)

PREDICTION_COUNT = Counter(
    "customer_churn_predictions_total",
    "Total predictions",
    ["prediction", "label"],
)

ERROR_COUNT = Counter(
    "customer_churn_api_errors_total",
    "Total API errors",
    ["endpoint"],
)

PREDICTION_LATENCY = Histogram(
    "customer_churn_prediction_latency_seconds",
    "Prediction request latency",
)

KSERVE_READY = Gauge(
    "customer_churn_kserve_ready",
    "KServe model readiness",
)


# ============================================================
# Request model
# ============================================================

class CustomerRequest(BaseModel):
    age: int = Field(..., ge=18, le=100)
    tenure: int = Field(..., ge=0)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    contract_type: str
    support_calls: int = Field(..., ge=0)


# ============================================================
# Feature transformation
# IMPORTANT:
# Preserve the existing model feature order:
#
# age
# tenure
# monthly_charges
# total_charges
# support_calls
# contract_type_month_to_month
# contract_type_one_year
# contract_type_two_year
# ============================================================

def transform_features(request: CustomerRequest):

    contract = request.contract_type.strip().lower()

    if contract == "month-to-month":
        contract_features = [1, 0, 0]

    elif contract == "one year":
        contract_features = [0, 1, 0]

    elif contract == "two year":
        contract_features = [0, 0, 1]

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid contract_type. "
                "Must be one of: Month-to-month, One year, Two year"
            ),
        )

    return [
        request.age,
        request.tenure,
        request.monthly_charges,
        request.total_charges,
        request.support_calls,
        *contract_features,
    ]


# ============================================================
# KServe headers
# ============================================================

def kserve_headers():

    return {
        "Content-Type": "application/json",

        # Stable Knative/KServe routing hostname.
        "Host": KSERVE_HOST,

        # Required for the cluster-local Knative route.
        "K-Network-Hash": "override",
    }


# ============================================================
# KServe readiness
# ============================================================

def kserve_ready():

    url = f"{KSERVE_URL}/v2/models/{MODEL_NAME}/ready"

    try:

        response = requests.get(
            url,
            headers=kserve_headers(),
            timeout=10,
        )

        if response.status_code == 200:

            KSERVE_READY.set(1)

            return response.json()

        KSERVE_READY.set(0)

        return {
            "ready": False,
            "status_code": response.status_code,
            "response": response.text,
        }

    except requests.RequestException as exc:

        KSERVE_READY.set(0)

        return {
            "ready": False,
            "error": str(exc),
        }


# ============================================================
# Metrics middleware
# ============================================================

class MetricsMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request, call_next):

        start_time = time.time()

        try:

            response = await call_next(request)

            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=str(response.status_code),
            ).inc()

            return response

        except Exception:

            ERROR_COUNT.labels(
                endpoint=request.url.path
            ).inc()

            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status="500",
            ).inc()

            raise

        finally:

            if request.url.path == "/predict":

                PREDICTION_LATENCY.observe(
                    time.time() - start_time
                )


app.add_middleware(MetricsMiddleware)

app.mount(
    "/metrics",
    make_asgi_app(),
)


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "customer-churn-api",
    }


# ============================================================
# Model readiness
# ============================================================

@app.get("/model/ready")
def model_ready():

    result = kserve_ready()

    return {
        "model": MODEL_NAME,
        "ready": result.get("ready", False),
        "kserve_response": result,
    }


# ============================================================
# Prediction
# ============================================================

@app.post("/predict")
def predict(request: CustomerRequest):

    features = transform_features(request)

    payload = {
        "instances": [
            features
        ]
    }

    url = (
        f"{KSERVE_URL}/v1/models/"
        f"{MODEL_NAME}:predict"
    )

    start_time = time.time()

    try:

        response = requests.post(
            url,
            json=payload,
            headers=kserve_headers(),
            timeout=30,
        )

        PREDICTION_LATENCY.observe(
            time.time() - start_time
        )

        if response.status_code != 200:

            ERROR_COUNT.labels(
                endpoint="/predict"
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

        predictions = result.get("predictions")

        if not predictions:

            ERROR_COUNT.labels(
                endpoint="/predict"
            ).inc()

            raise HTTPException(
                status_code=502,
                detail="KServe returned no predictions",
            )

        prediction = int(predictions[0])

        if prediction == 1:
            label = "Churn"
        else:
            label = "No Churn"

        PREDICTION_COUNT.labels(
            prediction=str(prediction),
            label=label,
        ).inc()

        return {
            "prediction": prediction,
            "churn": prediction == 1,
            "prediction_label": label,
            "model": MODEL_NAME,
            "features": features,
        }

    except HTTPException:
        raise

    except requests.RequestException as exc:

        ERROR_COUNT.labels(
            endpoint="/predict"
        ).inc()

        # Preserve existing API contract:
        # KServe unavailable -> HTTP 503
        raise HTTPException(
            status_code=503,
            detail=f"KServe unavailable: {exc}",
        )


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():

    return {
        "service": "customer-churn-api",
        "version": "2.0.0",
        "model": MODEL_NAME,
        "kserve_host": KSERVE_HOST,
    }
