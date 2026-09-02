from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("GRAPHGUARD_MODEL", ROOT / "artifacts" / "baseline" / "xgboost.json"))
THRESHOLD = float(os.getenv("GRAPHGUARD_THRESHOLD", "0.36"))
EXPECTED_FEATURES = 165

app = FastAPI(
    title="GraphGuard API",
    version="0.1.0",
    description="Inference API for illicit Bitcoin transaction risk scoring.",
)

_model: XGBClassifier | None = None


class PredictionRequest(BaseModel):
    features: Annotated[list[float], Field(min_length=165, max_length=165)]


class PredictionResponse(BaseModel):
    risk_score: float
    illicit: bool
    threshold: float


def get_model() -> XGBClassifier:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
        model = XGBClassifier()
        model.load_model(MODEL_PATH)
        _model = model
    return _model


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_available": MODEL_PATH.exists(),
    }


@app.get("/model")
def model_info() -> dict[str, str | float | int]:
    return {
        "model": "XGBoost",
        "feature_count": EXPECTED_FEATURES,
        "threshold": THRESHOLD,
        "protocol": "temporal_train_1_29_validation_30_34_test_35_49",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    features = np.asarray(request.features, dtype=np.float32).reshape(1, -1)
    probability = float(model.predict_proba(features)[0, 1])
    return PredictionResponse(
        risk_score=probability,
        illicit=probability >= THRESHOLD,
        threshold=THRESHOLD,
    )
