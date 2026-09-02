from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

from graphguard.explainability import explain_xgboost
from graphguard.feature_store import TransactionFeatureStore
from graphguard.neo4j_store import Neo4jStore

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("GRAPHGUARD_MODEL", ROOT / "artifacts" / "baseline" / "xgboost.json"))
FEATURE_STORE_PATH = Path(os.getenv("GRAPHGUARD_FEATURE_STORE", ROOT / "artifacts" / "features" / "transaction_features.npz"))
DASHBOARD_PATH = ROOT / "web" / "index.html"
THRESHOLD = float(os.getenv("GRAPHGUARD_THRESHOLD", "0.36"))
EXPECTED_FEATURES = 165

app = FastAPI(
    title="GraphGuard API",
    version="0.1.0",
    description="Inference and graph-investigation API for illicit Bitcoin transaction risk scoring.",
)

_model: XGBClassifier | None = None
_store: Neo4jStore | None = None
_feature_store: TransactionFeatureStore | None = None


class PredictionRequest(BaseModel):
    features: Annotated[list[float], Field(min_length=165, max_length=165)]


class ExplainRequest(BaseModel):
    features: Annotated[list[float], Field(min_length=165, max_length=165)]
    top_k: int = Field(default=10, ge=1, le=50)


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


def get_store() -> Neo4jStore:
    global _store
    if _store is None:
        _store = Neo4jStore()
    return _store


def get_feature_store() -> TransactionFeatureStore:
    global _feature_store
    if _feature_store is None:
        _feature_store = TransactionFeatureStore(FEATURE_STORE_PATH)
    return _feature_store


def _features(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(1, EXPECTED_FEATURES)


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_available": MODEL_PATH.exists(),
        "neo4j_configured": bool(os.getenv("NEO4J_PASSWORD")),
        "feature_store_available": FEATURE_STORE_PATH.exists(),
    }


@app.get("/model")
def model_info() -> dict[str, str | float | int]:
    return {
        "model": "XGBoost",
        "feature_count": EXPECTED_FEATURES,
        "threshold": THRESHOLD,
        "protocol": "temporal_train_1_29_validation_30_34_test_35_49",
    }


@app.get("/dashboard", include_in_schema=False)
def dashboard() -> FileResponse:
    if not DASHBOARD_PATH.exists():
        raise HTTPException(status_code=503, detail="Dashboard asset not found")
    return FileResponse(DASHBOARD_PATH, media_type="text/html")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    features = _features(request.features)
    probability = float(model.predict_proba(features)[0, 1])
    return PredictionResponse(risk_score=probability, illicit=probability >= THRESHOLD, threshold=THRESHOLD)


@app.post("/explain")
def explain(request: ExplainRequest) -> dict[str, object]:
    try:
        model = get_model()
        return explain_xgboost(model, _features(request.features), top_k=request.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/transactions/{tx_id}")
def transaction(tx_id: int) -> dict:
    try:
        result = get_store().get_transaction(tx_id)
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return result


@app.get("/transactions/{tx_id}/neighbors")
def suspicious_neighbors(tx_id: int, limit: int = 20) -> dict[str, object]:
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    try:
        neighbors = get_store().get_suspicious_neighbors(tx_id, limit)
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"tx_id": tx_id, "neighbors": neighbors}


@app.get("/transactions/{tx_id}/explain")
def explain_transaction(tx_id: int, top_k: int = 10) -> dict[str, object]:
    if not 1 <= top_k <= 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")
    try:
        model = get_model()
        features = get_feature_store().get_features(tx_id)
        transaction_data = get_store().get_transaction(tx_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if transaction_data is None or features is None:
        raise HTTPException(status_code=404, detail="Transaction features not found")
    explanation = explain_xgboost(model, features.reshape(1, EXPECTED_FEATURES), top_k=top_k)
    return {"transaction": transaction_data, "explanation": explanation}


@app.get("/cases/{tx_id}")
def investigator_case(tx_id: int, top_k: int = 10, neighbor_limit: int = 10) -> dict[str, object]:
    """Return one investigator-ready case: risk, explanation, and graph context."""
    if not 1 <= top_k <= 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")
    if not 1 <= neighbor_limit <= 100:
        raise HTTPException(status_code=400, detail="neighbor_limit must be between 1 and 100")

    try:
        model = get_model()
        store = get_store()
        transaction_data = store.get_transaction(tx_id)
        features = get_feature_store().get_features(tx_id)
        neighbors = store.get_suspicious_neighbors(tx_id, neighbor_limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if transaction_data is None or features is None:
        raise HTTPException(status_code=404, detail="Transaction not found")

    explanation = explain_xgboost(model, features.reshape(1, EXPECTED_FEATURES), top_k=top_k)
    risk_score = float(explanation["risk_score"])
    return {
        "case": {
            "tx_id": tx_id,
            "time_step": transaction_data.get("time_step"),
            "risk_score": risk_score,
            "decision": "high_risk" if risk_score >= THRESHOLD else "below_threshold",
            "threshold": THRESHOLD,
        },
        "explanation": explanation,
        "graph_context": {
            "neighbor_count": len(neighbors),
            "highest_risk_neighbors": neighbors,
        },
    }
