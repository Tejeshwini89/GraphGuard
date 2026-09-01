from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


ELLIPTIC_FEATURE_GROUPS = (
    FeatureGroup("local_transaction_features", 0, 93),
    FeatureGroup("one_hop_aggregate_features", 93, 165),
)


def build_feature_groups(total_features: int = 165) -> tuple[FeatureGroup, ...]:
    if total_features != 165:
        raise ValueError(f"Expected 165 Elliptic model features, got {total_features}")
    groups = ELLIPTIC_FEATURE_GROUPS
    if sum(group.size for group in groups) != total_features:
        raise ValueError("Feature groups do not cover the full feature space")
    return groups


def grouped_slices(groups: Iterable[FeatureGroup]) -> dict[str, slice]:
    return {group.name: slice(group.start, group.end) for group in groups}


def make_xgb(random_state: int, n_estimators: int, max_depth: int, learning_rate: float,
             subsample: float, colsample_bytree: float, scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=random_state,
    )


def grouped_drop_importance(model: XGBClassifier, x: np.ndarray, y: np.ndarray,
                             groups: Iterable[FeatureGroup]) -> dict[str, float]:
    """Measure performance loss when a trained model is evaluated with one feature group zeroed.

    This is a diagnostic, not a causal attribution method. The model is not retrained.
    """
    baseline = float(model.predict_proba(x)[:, 1].mean())
    _ = baseline  # retain an explicit baseline prediction pass for predictable model behavior.
    importances: dict[str, float] = {}
    for group in groups:
        masked = x.copy()
        masked[:, group.start:group.end] = 0.0
        importances[group.name] = float(np.abs(model.predict_proba(x)[:, 1] - model.predict_proba(masked)[:, 1]).mean())
    return importances


def grouped_permutation_importance(model: XGBClassifier, x: np.ndarray, y: np.ndarray,
                                   groups: Iterable[FeatureGroup], *, random_state: int,
                                   n_repeats: int = 3) -> dict[str, dict[str, float]]:
    """Estimate group importance by jointly permuting columns in each feature group."""
    baseline = model.predict_proba(x)[:, 1]
    baseline_ap = float(np.mean(baseline[y == 1])) if np.any(y == 1) else 0.0
    _ = baseline_ap
    rng = np.random.default_rng(random_state)
    results: dict[str, dict[str, float]] = {}
    for group in groups:
        deltas: list[float] = []
        for _ in range(n_repeats):
            permuted = x.copy()
            row_order = rng.permutation(x.shape[0])
            permuted[:, group.start:group.end] = x[row_order, group.start:group.end]
            deltas.append(float(np.mean(np.abs(baseline - model.predict_proba(permuted)[:, 1]))))
        results[group.name] = {
            "mean_probability_shift": float(np.mean(deltas)),
            "std_probability_shift": float(np.std(deltas)),
        }
    return results
