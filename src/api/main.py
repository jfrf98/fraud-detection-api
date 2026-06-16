from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
import logging
from pydantic import BaseModel

from src.api.utils import (
    get_prediction_stats,
    get_risk_decision,
    log_prediction,
    get_prediction_metrics,
    get_recent_predictions
)

from src.api.config import MODEL_VERSION, FEATURE_COUNT, FRAUD_THRESHOLD

app = FastAPI()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = BASE_DIR.parent / "logs" / "predictions.csv"

model = joblib.load(BASE_DIR / "models" / "fraud_model.pkl")

MODEL_TYPE = type(model).__name__

class Transaction(BaseModel):
    features: List[float]

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.get("/health")
def health_check():
     return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION
    }

@app.get("/model-info")
def model_info():
    return {
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "features_used": FEATURE_COUNT,
        "threshold": FRAUD_THRESHOLD
    }

@app.get("/stats")
def stats():
    return get_prediction_stats(LOG_FILE)

@app.get("/metrics")
def metrics():
    return get_prediction_metrics(LOG_FILE)

@app.get("/logs/recent")
def recent_predictions(limit: int = 10):
    return get_recent_predictions(LOG_FILE, limit=limit)

@app.post("/predict")
def predict(transaction: Transaction):
    if len(transaction.features) != FEATURE_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"The transaction must contain exactly {FEATURE_COUNT} features."
        )

    input_data = np.array(transaction.features).reshape(1, -1)

    if not np.isfinite(input_data).all():
        raise HTTPException(
            status_code=400,
            detail="All features must be finite numeric values."
        )
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 0:
        label = "normal"
    else:
        label = "fraud"
    
    risk_level, action = get_risk_decision(probability)

    logger.info(
        f"Prediction={prediction} | "
        f"Probability={probability:.4f} | "
        f"Risk={risk_level} | "
        f"Action={action}"
    )

    log_prediction(
        probability=probability,
        prediction=prediction,
        risk_level=risk_level,
        action=action,
        model_version=MODEL_VERSION,
        log_file=LOG_FILE
    )


    return {
        "prediction": int(prediction),
        "label": label,
        "fraud_probability": float(probability),
        "risk_level": risk_level,
        "recommended_action": action,
        "model_version": MODEL_VERSION,
        "threshold_used": FRAUD_THRESHOLD
    }

