import sys
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from fastapi.testclient import TestClient

API_DIR = (
    Path(__file__).resolve().parents[1]
    / "customer_churn_project"
    / "api"
)

sys.path.insert(0, str(API_DIR))

from main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "customer-churn-api"


def test_metrics():
    response = client.get("/metrics")

    assert response.status_code == 200

    body = response.text

    assert "customer_churn_api_requests_total" in body
    assert "customer_churn_predictions_total" in body
    assert "customer_churn_prediction_latency_seconds" in body


@patch("main.requests.get")
def test_model_ready(mock_get):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "customer-churn-randomforest",
        "ready": True,
    }

    mock_get.return_value = mock_response

    response = client.get("/model/ready")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "customer-churn-randomforest"
    assert data["ready"] is True
    assert data["kserve_response"]["ready"] is True


@patch("main.requests.post")
def test_predict_no_churn(mock_post):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "predictions": [0]
    }

    mock_post.return_value = mock_response

    payload = {
        "age": 35,
        "tenure": 12,
        "monthly_charges": 75.0,
        "total_charges": 900.0,
        "support_calls": 3,
        "contract_type": "Month-to-month",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 0
    assert data["churn"] is False
    assert data["prediction_label"] == "No Churn"
    assert data["model"] == "customer-churn-randomforest"

    assert data["features"] == [
        35,
        12,
        75.0,
        900.0,
        3,
        1,
        0,
        0,
    ]


@patch("main.requests.post")
def test_predict_churn(mock_post):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "predictions": [1]
    }

    mock_post.return_value = mock_response

    payload = {
        "age": 45,
        "tenure": 3,
        "monthly_charges": 120.0,
        "total_charges": 360.0,
        "support_calls": 8,
        "contract_type": "One year",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 1
    assert data["churn"] is True
    assert data["prediction_label"] == "Churn"

    assert data["features"] == [
        45,
        3,
        120.0,
        360.0,
        8,
        0,
        1,
        0,
    ]


def test_invalid_contract_type():
    payload = {
        "age": 35,
        "tenure": 12,
        "monthly_charges": 75.0,
        "total_charges": 900.0,
        "support_calls": 3,
        "contract_type": "Invalid Contract",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 400

    data = response.json()

    assert "Invalid contract_type" in data["detail"]


@patch("main.requests.post")
def test_kserve_prediction_error(mock_post):
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.text = "KServe internal error"

    mock_post.return_value = mock_response

    payload = {
        "age": 35,
        "tenure": 12,
        "monthly_charges": 75.0,
        "total_charges": 900.0,
        "support_calls": 3,
        "contract_type": "Month-to-month",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 502

    data = response.json()

    assert data["detail"]["message"] == "KServe prediction failed"
    assert data["detail"]["status_code"] == 500


@patch("main.requests.post")
def test_kserve_unavailable(mock_post):
    mock_post.side_effect = requests.RequestException(
        "Connection refused"
    )

    payload = {
        "age": 35,
        "tenure": 12,
        "monthly_charges": 75.0,
        "total_charges": 900.0,
        "support_calls": 3,
        "contract_type": "Month-to-month",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 503

    data = response.json()

    assert "KServe unavailable" in data["detail"]
