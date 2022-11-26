from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
import json 

from api.main import app

@pytest.fixture()
def client():
    with TestClient(app) as c:
        response = c.get('/')
        yield c


def test_predict_sick(client):
    test = {
        "age": 0,
        "sex": 0,
        "cp": 0,
        "chol": 0,
        "fbs": 0,
        "restecg": 0,
        "thalach": 0,
        "exang": 0,
        "oldpeak": 0,
        "slope": 0,
        "ca": 0,
        "thal": 0
    }
    response = client.post("/predict",  json.dumps(test))
    assert response.status_code == 200
    assert response.json() == [ 0 ]

def test_predict_healty(client):
    test = {
        "age": 61,
        "sex": 1,
        "cp": 0,
        "chol": 234,
        "fbs": 0,
        "restecg": 0,
        "thalach": 145,
        "exang": 0,
        "oldpeak": 2.6,
        "slope": 1,
        "ca": 2,
        "thal": 0
    }
    response = client.post("/predict", json.dumps(test))
    assert response.status_code == 200
    assert response.json() == [ 1 ]

def test_predict_wrong_input(client):
    test = {
        "age": 61,
        "sex": 1,
        "cp": 0,
        "chol": 234,
        "fbs": 0,
        "restecg": 0,
        "thalach": 145,
        "exang": 0,
        "oldpeak": 2.6,
        "slope": 1,
        "thal": 0
    }
    response = client.post("/predict", json.dumps(test))
    assert response.status_code == 422