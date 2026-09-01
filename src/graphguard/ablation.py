from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
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


def grouped_drop_importance(model: XGBClassifier, x: np.ndarray,
                            groups: Iterable[FeatureGroup]) -> dict[str, float]:
    """Measure mean absolute probability shift after zeroing a feature group.

    Diagnostic only: the trained model is not retrained. Interpret cautiously because
    zero may not represent a neutral value for every raw feature.
    """
    baseline = model.predict_proba(x)[:, 1]
    importances: dict[str, float] = {}
    for group in groups:
        masked = x.copy()
        masked[:, group.start:group.end] = 0.0
        shifted = model.predict_proba(masked)[:, 1]
        importances[group.name] = float(np.abs(baseline - shifted).mean())
    return importances


def grouped_permutation_importance(model: XGBClassifier, x: np.ndarray,
                                   groups: Iterable[FeatureGroup], *, random_state: int,
                                   n_repeats: int = 3) -> dict[str, dict[str, float]]:
    """Estimate group importance by jointly permuting columns within each group.

    Returns mean/std of absolute probability shift. This is diagnostic only and is not
    a causal attribution or model-selection criterion.
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be positive")
    baseline = model.predict_proba(x)[:, 1]
    rng = np.random.default_rng(random_state)
    results: dict[str, dict[str, float]] = {}
    for group in groups:
        deltas: list[float] = []
        for _ in range(n_repeats):
            permuted = x.copy()
            row_order = rng.permutation(x.shape[0])
            permuted[:, group.start:group.end] = x[row_order, group.start:group.end]
            shifted = model.predict_proba(permuted)[:, 1]
            deltas.append(float(np.abs(baseline - shifted).mean()))
        results[group.name] = {
            "mean_probability_shift": float(np.mean(deltas)),
            "std_probability_shift": float(np.std(deltas)),
        }
    return results
