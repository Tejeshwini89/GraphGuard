from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graphguard.baseline import evaluate_xgboost, save_baseline_artifact, select_threshold, train_xgboost
from graphguard.elliptic import load_elliptic_graph
from graphguard.splits import make_temporal_split


def main() -> None:
    root = ROOT / "data" / "raw"
    graph = load_elliptic_graph(root)
    split = make_temporal_split(
        graph.time_step,
        graph.y,
        train_end=29,
        validation_start=30,
        validation_end=34,
        test_start=35,
    )
    model = train_xgboost(graph, split, seed=42)
    validation_probability = model.predict_proba(graph.x[split.validation_mask].cpu().numpy())[:, 1]
    validation_labels = graph.y[split.validation_mask].cpu().numpy()
    threshold = select_threshold(validation_labels, validation_probability)
    validation_metrics = evaluate_xgboost(model, graph, split.validation_mask, threshold)
    test_metrics = evaluate_xgboost(model, graph, split.test_mask, threshold)

    artifact_dir = ROOT / "artifacts" / "baseline"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_baseline_artifact(model, artifact_dir / "xgboost.json")
    report = {
        "model": "xgboost",
        "split": {"train_end": 29, "validation_start": 30, "validation_end": 34, "test_start": 35},
        "threshold_selected_on": "validation",
        "threshold": threshold,
        "validation": validation_metrics.__dict__,
        "test": test_metrics.__dict__,
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"artifact: {artifact_dir / 'xgboost.json'}")


if __name__ == "__main__":
    main()
