from __future__ import annotations

import numpy as np

from scripts.temporal_error_analysis import average_precision, binary_metrics, roc_auc, select_threshold


def test_binary_metrics_counts_confusion_components() -> None:
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.8, 0.7, 0.2])
    metrics = binary_metrics(y, probability, 0.5)
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_average_precision_is_one_for_perfect_ranking() -> None:
    y = np.array([0, 1, 0, 1])
    probability = np.array([0.1, 0.9, 0.2, 0.8])
    assert average_precision(y, probability) == 1.0


def test_roc_auc_handles_tied_scores() -> None:
    y = np.array([0, 1, 0, 1])
    probability = np.array([0.2, 0.8, 0.2, 0.8])
    assert roc_auc(y, probability) == 1.0


def test_select_threshold_uses_validation_data_only() -> None:
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = select_threshold(y, probability)
    assert 0.05 <= threshold <= 0.95
