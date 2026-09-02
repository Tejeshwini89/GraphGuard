from __future__ import annotations

import numpy as np

from graphguard.ablation import build_feature_groups
from scripts.feature_ablation import _select_threshold


def test_elliptic_feature_groups_cover_all_165_features() -> None:
    groups = build_feature_groups(165)
    assert [(group.name, group.start, group.end) for group in groups] == [
        ("local_transaction_features", 0, 93),
        ("one_hop_aggregate_features", 93, 165),
    ]
    assert sum(group.size for group in groups) == 165


def test_select_threshold_uses_validation_probabilities() -> None:
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = _select_threshold(y, probability)
    assert 0.05 <= threshold <= 0.95
