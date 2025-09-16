
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import traceback
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # ensure src importable
from preprocess import normalize_columns, load_data
from feature_engineering import add_engineered_features

app = FastAPI(title="Loan Approval Inference API")

MODEL_BUNDLE = None
MODEL_PATH = os.environ.get('MODEL_PATH', 'artifacts/model.joblib')

def load_model():
    global MODEL_BUNDLE
    if MODEL_BUNDLE is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        MODEL_BUNDLE = joblib.load(MODEL_PATH)
    return MODEL_BUNDLE

class Applicant(BaseModel):
    loan_id: str | None = None
    no_of_dependents: int | None = 0
    education: str | None = None
    self_employed: str | None = None
    income_annum: float | None = None
    loan_amount: float | None = None
    loan_term: float | None = None
    cibil_score: float | None = None
    residential_assets_value: float | None = None
    commercial_assets_value: float | None = None
    luxury_assets_value: float | None = None
    bank_asset_value: float | None = None

@app.get('/health')
def health():
    try:
        load_model()
        return {'status':'ok'}
    except Exception as e:
        return {'status':'error', 'error': str(e)}

@app.post('/predict')
def predict_single(applicant: Applicant):
    try:
        bundle = load_model()
        pipeline = bundle.get('pipeline') if isinstance(bundle, dict) else bundle
        # convert to dataframe and normalize
        df = pd.DataFrame([applicant.dict()])
        df = normalize_columns(df)
        df = add_engineered_features(df)
        # drop possible target
        for t in ['Loan_Status','loan_status','LoanStatus']:
            if t in df.columns:
                df = df.drop(columns=[t], errors='ignore')
        preds = pipeline.predict(df)
        proba = None
        try:
            proba = pipeline.predict_proba(df)[:,1].tolist()
        except Exception:
            proba = None
        return {'predictions': preds.tolist(), 'proba': proba, 'input': applicant.dict()}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
