import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pickle
from unittest.mock import MagicMock, patch
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ---- /health tests ----

def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200

def test_health_returns_ok(client):
    data = response = client.get("/health").get_json()
    assert data["status"] == "ok"


# ---- /predict tests ----

def test_predict_missing_text_field(client):
    response = client.post("/predict", json={})
    assert response.status_code == 400

def test_predict_empty_text(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 400

def test_predict_no_json(client):
    response = client.post("/predict")
    assert response.status_code in [400, 415]

def test_predict_returns_label(client):
    response = client.post("/predict", json={"text": "I hate you"})
    assert response.status_code == 200
    data = response.get_json()
    assert "label" in data
    assert data["label"] in ["Hate Speech", "Not Hate Speech"]

def test_predict_returns_confidence(client):
    response = client.post("/predict", json={"text": "I love you"})
    data = response.get_json()
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0

def test_predict_returns_cleaned_text(client):
    response = client.post("/predict", json={"text": "I hate you"})
    data = response.get_json()
    assert "cleaned" in data
    assert isinstance(data["cleaned"], str)

def test_predict_returns_input(client):
    response = client.post("/predict", json={"text": "I hate you"})
    data = response.get_json()
    assert "input" in data
    assert data["input"] == "I hate you"
