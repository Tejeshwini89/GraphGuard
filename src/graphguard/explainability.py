from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from xgboost import DMatrix, XGBClassifier


@dataclass(frozen=True)
class FeatureContribution:
    """One feature's additive contribution to the XGBoost margin."""

    feature_index: int
    contribution: float


def explain_xgboost(
    model: XGBClassifier,
    features: np.ndarray,
    *,
    top_k: int = 10,
) -> dict[str, object]:
    """Return local XGBoost feature contributions for one transaction.

    XGBoost's pred_contribs output is additive in the model margin (log-odds
    for the binary logistic objective), not in probability space. The final
    probability is therefore reported separately and contributions are kept
    as signed margin effects for honest interpretation.
    """
    if top_k < 1 or top_k > features.shape[1]:
        raise ValueError("top_k must be between 1 and the feature count")

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 1:
        raise ValueError("features must have shape (1, feature_count)")

    contribution_matrix = model.get_booster().predict(
        DMatrix(values),
        pred_contribs=True,
    )
    contributions = contribution_matrix[0, :-1]
    base_value = float(contribution_matrix[0, -1])
    probability = float(model.predict_proba(values)[0, 1])

    ranked = sorted(
        (
            FeatureContribution(int(index), float(value))
            for index, value in enumerate(contributions)
        ),
        key=lambda item: abs(item.contribution),
        reverse=True,
    )[:top_k]

    return {
        "risk_score": probability,
        "base_margin": base_value,
        "contribution_space": "xgboost_margin",
        "top_features": [
            {
                "feature_index": item.feature_index,
                "contribution": item.contribution,
                "direction": "increases_risk" if item.contribution > 0 else "decreases_risk",
            }
            for item in ranked
        ],
    }
