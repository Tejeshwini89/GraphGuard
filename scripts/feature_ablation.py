from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from graphguard.elliptic import load_elliptic_graph
from graphguard.splits import make_temporal_split


def _metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(np.int64)
    return {
        "pr_auc": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "threshold": float(threshold),
    }


def _select_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        score = f1_score(y_true, probability >= threshold, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold


def _train_and_evaluate(
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    test_mask: np.ndarray,
    config: dict,
) -> dict:
    y_train = y[train_mask]
    positives = max(int((y_train == 1).sum()), 1)
    negatives = max(int((y_train == 0).sum()), 1)
    model = XGBClassifier(
        n_estimators=int(config["n_estimators"]),
        max_depth=int(config["max_depth"]),
        learning_rate=float(config["learning_rate"]),
        subsample=float(config["subsample"]),
        colsample_bytree=float(config["colsample_bytree"]),
        scale_pos_weight=negatives / positives,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=int(config["random_state"]),
    )
    model.fit(x[train_mask], y_train)
    validation_probability = model.predict_proba(x[validation_mask])[:, 1]
    test_probability = model.predict_proba(x[test_mask])[:, 1]
    threshold = _select_threshold(y[validation_mask], validation_probability)
    return {
        "validation": _metrics(y[validation_mask], validation_probability, threshold),
        "test": _metrics(y[test_mask], test_probability, threshold),
    }


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    dataset_cfg = config["dataset"]
    split_cfg = config["splits"]
    baseline_cfg = config["baseline"]
    data = load_elliptic_graph(dataset_cfg["root"])
    split = make_temporal_split(
        data.time_step,
        data.y,
        train_end=int(split_cfg["train_end"]),
        validation_start=int(split_cfg["validation_start"]),
        validation_end=int(split_cfg["validation_end"]),
        test_start=int(split_cfg["test_start"]),
        unknown_label=int(dataset_cfg["unknown_label"]),
    )

    x = data.x.numpy()
    y = data.y.numpy()
    # The Elliptic feature layout is 93 local features + 72 one-hop aggregate features.
    feature_sets = {
        "local_transaction_features": slice(0, 93),
        "one_hop_aggregate_features": slice(93, 165),
        "all_165_features": slice(0, 165),
    }
    results: dict[str, dict] = {}
    for name, feature_slice in feature_sets.items():
        results[name] = _train_and_evaluate(
            x[:, feature_slice],
            y,
            split.train_mask.numpy(),
            split.validation_mask.numpy(),
            split.test_mask.numpy(),
            baseline_cfg,
        )
        results[name]["feature_count"] = int(x[:, feature_slice].shape[1])

    artifact = ROOT / "artifacts" / "forensics" / "feature_ablation.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "purpose": "measure how much predictive signal comes from local versus one-hop aggregate features",
        "protocol": "temporal_train_validation_test",
        "split": split_cfg,
        "feature_layout_assumption": "93 local transaction features + 72 one-hop aggregate features",
        "threshold_selected_on": "validation",
        "test_used_only_for_final_evaluation": True,
        "results": results,
    }
    artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report: {artifact}")


if __name__ == "__main__":
    main()
