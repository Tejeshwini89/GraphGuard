from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from graphguard.elliptic import load_elliptic
from graphguard.splits import make_temporal_masks


def average_precision(y_true: np.ndarray, probability: np.ndarray) -> float:
    order = np.argsort(-probability, kind="mergesort")
    y = y_true[order]
    positives = float(y.sum())
    if positives == 0:
        return float("nan")
    cumulative = np.cumsum(y)
    ranks = np.arange(1, len(y) + 1)
    precision = cumulative / ranks
    return float((precision * y).sum() / positives)


def roc_auc(y_true: np.ndarray, probability: np.ndarray) -> float:
    positives = y_true == 1
    negatives = y_true == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(probability, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = probability >= threshold
    tp = int(np.sum(prediction & (y_true == 1)))
    fp = int(np.sum(prediction & (y_true == 0)))
    fn = int(np.sum(~prediction & (y_true == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def load_xgb_probability(model_path: Path, x: np.ndarray) -> np.ndarray:
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(model_path)
    return model.predict_proba(x)[:, 1]


def select_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 91)
    best = (0.05, -1.0)
    for threshold in thresholds:
        f1 = binary_metrics(y_true, probability, float(threshold))["f1"]
        if f1 > best[1]:
            best = (float(threshold), f1)
    return best[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze XGBoost errors by test time step.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default="artifacts/baseline/xgboost.json")
    parser.add_argument("--output", default="artifacts/error_analysis/temporal_error_analysis.json")
    args = parser.parse_args()

    config = json.loads(json.dumps(__import__("yaml").safe_load(Path(args.config).read_text())))
    dataset = load_elliptic(config["dataset"])
    masks = make_temporal_masks(dataset.time_step, dataset.y, config["splits"])

    validation = masks["validation"]
    test = masks["test"]
    validation_probability = load_xgb_probability(Path(args.model), dataset.x[validation])
    test_probability = load_xgb_probability(Path(args.model), dataset.x[test])
    threshold = select_threshold(dataset.y[validation], validation_probability)

    test_times = dataset.time_step[test]
    test_labels = dataset.y[test]
    rows = []
    for timestep in sorted(np.unique(test_times).tolist()):
        mask = test_times == timestep
        y = test_labels[mask]
        probability = test_probability[mask]
        metrics = binary_metrics(y, probability, threshold)
        rows.append({
            "time_step": int(timestep),
            "labeled_transactions": int(len(y)),
            "licit": int(np.sum(y == 0)),
            "illicit": int(np.sum(y == 1)),
            "illicit_rate": float(np.mean(y == 1)),
            "pr_auc": average_precision(y, probability),
            "roc_auc": roc_auc(y, probability),
            **metrics,
        })

    report = {
        "purpose": "post-hoc temporal error analysis for the frozen XGBoost baseline",
        "warning": "Test labels are used only for post-hoc diagnosis. They must not be used for model selection, tuning, or threshold selection.",
        "threshold_selected_on": "validation",
        "threshold": threshold,
        "split": {
            "train": "1-29",
            "validation": "30-34",
            "test": "35-49",
        },
        "test_by_time_step": rows,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"report: {output}")


if __name__ == "__main__":
    main()
