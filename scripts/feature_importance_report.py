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
from sklearn.inspection import permutation_importance

from graphguard.ablation import build_feature_groups, make_xgb
from graphguard.elliptic import load_elliptic_graph
from graphguard.splits import make_temporal_split


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
    train = split.train_mask.numpy()
    validation = split.validation_mask.numpy()
    test = split.test_mask.numpy()
    y_train = y[train]
    positives = max(int((y_train == 1).sum()), 1)
    negatives = max(int((y_train == 0).sum()), 1)

    model = make_xgb(
        random_state=int(baseline_cfg["random_state"]),
        n_estimators=int(baseline_cfg["n_estimators"]),
        max_depth=int(baseline_cfg["max_depth"]),
        learning_rate=float(baseline_cfg["learning_rate"]),
        subsample=float(baseline_cfg["subsample"]),
        colsample_bytree=float(baseline_cfg["colsample_bytree"]),
        scale_pos_weight=negatives / positives,
    )
    model.fit(x[train], y_train)

    # Permutation importance is computed on the untouched test features strictly as a
    # post-hoc diagnostic. It does not alter model selection or the operating threshold.
    permutation = permutation_importance(
        model,
        x[test],
        y[test],
        scoring="average_precision",
        n_repeats=3,
        random_state=int(config["project"]["seed"]),
        n_jobs=-1,
    )
    ranking = sorted(
        [
            {
                "feature_index": int(index),
                "mean_ap_decrease": float(mean),
                "std_ap_decrease": float(std),
            }
            for index, (mean, std) in enumerate(zip(permutation.importances_mean, permutation.importances_std, strict=True))
        ],
        key=lambda row: row["mean_ap_decrease"],
        reverse=True,
    )

    groups = build_feature_groups(x.shape[1])
    group_ranges = [{"name": group.name, "start": group.start, "end": group.end, "size": group.size} for group in groups]
    report = {
        "purpose": "post-hoc feature importance for the full XGBoost model",
        "warning": "Permutation importance is diagnostic only and test labels were not used for model selection.",
        "test_rows": int(test.sum()),
        "feature_groups": group_ranges,
        "top_20_features_by_test_ap_decrease": ranking[:20],
    }
    artifact = ROOT / "artifacts" / "forensics" / "feature_importance_report.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report: {artifact}")


if __name__ == "__main__":
    main()
