from __future__ import annotations

import numpy as np
import pytest

from graphguard.explainability import explain_xgboost


class FakeBooster:
    def __init__(self, contributions: np.ndarray) -> None:
        self.contributions = contributions

    def predict(self, matrix, pred_contribs: bool = False) -> np.ndarray:
        assert pred_contribs is True
        return self.contributions


class FakeModel:
    def __init__(self) -> None:
        self.booster = FakeBooster(
            np.array([[0.2, -0.9, 0.4, 0.1]], dtype=np.float32)
        )

    def get_booster(self) -> FakeBooster:
        return self.booster

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        assert features.shape == (1, 3)
        return np.array([[0.25, 0.75]], dtype=np.float32)


def test_explain_ranks_by_absolute_contribution_and_preserves_direction() -> None:
    result = explain_xgboost(FakeModel(), np.zeros((1, 3), dtype=np.float32), top_k=2)

    assert result["risk_score"] == pytest.approx(0.75)
    assert result["base_margin"] == pytest.approx(0.1)
    assert result["contribution_space"] == "xgboost_margin"
    assert result["top_features"] == [
        {"feature_index": 1, "contribution": pytest.approx(-0.9), "direction": "decreases_risk"},
        {"feature_index": 2, "contribution": pytest.approx(0.4), "direction": "increases_risk"},
    ]


def test_explain_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError):
        explain_xgboost(FakeModel(), np.zeros((1, 3), dtype=np.float32), top_k=0)
