from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data
from xgboost import XGBClassifier

from graphguard.splits import TemporalSplit


@dataclass(frozen=True)
class BaselineMetrics:
    """Evaluation metrics for the tabular fraud baseline."""

    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float


def _binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> BaselineMetrics:
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    prediction = (probability >= threshold).astype(np.int64)
    return BaselineMetrics(
        pr_auc=float(average_precision_score(y_true, probability)),
        roc_auc=float(roc_auc_score(y_true, probability)),
        precision=float(precision_score(y_true, prediction, zero_division=0)),
        recall=float(recall_score(y_true, prediction, zero_division=0)),
        f1=float(f1_score(y_true, prediction, zero_division=0)),
        threshold=float(threshold),
    )


def select_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Select the validation threshold maximizing F1 without using test labels."""
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        score = _binary_metrics(y_true, probability, float(threshold)).f1
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def train_xgboost(data: Data, split: TemporalSplit, *, seed: int = 42, **params: Any) -> XGBClassifier:
    """Train XGBoost using only labeled training nodes."""
    x = data.x.detach().cpu().numpy()
    y = data.y.detach().cpu().numpy().astype(np.int64)
    train_mask = split.train_mask.detach().cpu().numpy()
    y_train = y[train_mask]
    positives = max(int((y_train == 1).sum()), 1)
    negatives = max(int((y_train == 0).sum()), 1)
    defaults = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": -1,
        "scale_pos_weight": negatives / positives,
    }
    defaults.update(params)
    model = XGBClassifier(**defaults)
    model.fit(x[train_mask], y_train)
    return model


def evaluate_xgboost(model: XGBClassifier, data: Data, mask: torch.Tensor, threshold: float = 0.5) -> BaselineMetrics:
    """Evaluate a fitted XGBoost model on a labeled mask."""
    indices = mask.detach().cpu().numpy()
    x = data.x.detach().cpu().numpy()
    y = data.y.detach().cpu().numpy().astype(np.int64)
    probability = model.predict_proba(x[indices])[:, 1]
    return _binary_metrics(y[indices], probability, threshold)


def save_baseline_artifact(model: XGBClassifier, output: str | Path) -> Path:
    """Save the trained baseline model."""
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(destination)
    return destination
