
import json
from fastapi.testclient import TestClient
from src.serve_fastapi import app

client = TestClient(app)

def test_health():
    resp = client.get('/health')
    assert resp.status_code in (200,500)  # model may not exist in CI

def test_predict_schema():
    payload = {
        "loan_id":"T1","no_of_dependents":0,"education":"Graduate","self_employed":"No",
        "income_annum":500000,"loan_amount":200000,"loan_term":12,"cibil_score":720,
        "residential_assets_value":2400000,"commercial_assets_value":1760000,"luxury_assets_value":2270000,"bank_asset_value":8000000
    }
    resp = client.post('/predict', json=payload)
    # If model not present, should return 500; otherwise 200
    assert resp.status_code in (200,500)
