from __future__ import annotations

import numpy as np

from scripts.feature_ablation import _select_threshold


def test_select_threshold_uses_validation_probabilities() -> None:
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = _select_threshold(y, probability)
    assert 0.05 <= threshold <= 0.95
