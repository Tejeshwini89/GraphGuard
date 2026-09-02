from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from graphguard.elliptic import load_elliptic_graph
from graphguard.splits import make_temporal_split


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
    sorted_probability = probability[order]
    ranks = np.empty(len(order), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_probability[end] == sorted_probability[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
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

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset = load_elliptic_graph(config["dataset"]["root"])
    split_cfg = config["splits"]
    split = make_temporal_split(
        dataset.time_step,
        dataset.y,
        train_end=int(split_cfg["train_end"]),
        validation_start=int(split_cfg["validation_start"]),
        validation_end=int(split_cfg["validation_end"]),
        test_start=int(split_cfg["test_start"]),
        unknown_label=int(config["dataset"]["unknown_label"]),
    )

    validation = split.validation_mask.numpy()
    test = split.test_mask.numpy()
    x = dataset.x.numpy()
    y = dataset.y.numpy()
    validation_probability = load_xgb_probability(Path(args.model), x[validation])
    test_probability = load_xgb_probability(Path(args.model), x[test])
    threshold = select_threshold(y[validation], validation_probability)

    test_times = dataset.time_step.numpy()[test]
    test_labels = y[test]
    rows = []
    for timestep in sorted(np.unique(test_times).tolist()):
        mask = test_times == timestep
        labels = test_labels[mask]
        probability = test_probability[mask]
        metrics = binary_metrics(labels, probability, threshold)
        rows.append({
            "time_step": int(timestep),
            "labeled_transactions": int(len(labels)),
            "licit": int(np.sum(labels == 0)),
            "illicit": int(np.sum(labels == 1)),
            "illicit_rate": float(np.mean(labels == 1)),
            "pr_auc": average_precision(labels, probability),
            "roc_auc": roc_auc(labels, probability),
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
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report: {output}")


if __name__ == "__main__":
    main()
