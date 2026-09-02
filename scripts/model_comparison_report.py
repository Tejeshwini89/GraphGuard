from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_FILES = {
    "XGBoost": "artifacts/baseline/metrics.json",
    "GraphSAGE": "artifacts/gnn/graphsage_metrics.json",
    "Feature + GraphSAGE hybrid": "artifacts/gnn/hybrid_metrics.json",
    "GAT": "artifacts/gnn/gat_metrics.json",
}

METRICS = ("pr_auc", "roc_auc", "precision", "recall", "f1")


def _load_metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _extract_test(payload: dict) -> dict:
    # XGBoost and GNN reports both expose a `test` mapping.
    test = payload.get("test", {})
    return {metric: test.get(metric) for metric in METRICS}


def build_report(root: Path) -> dict:
    models = []
    for name, relative_path in MODEL_FILES.items():
        path = root / relative_path
        if not path.exists():
            models.append({
                "model": name,
                "status": "pending",
                "artifact": relative_path,
                "test": {metric: None for metric in METRICS},
            })
            continue
        payload = _load_metrics(path)
        models.append({
            "model": name,
            "status": "completed",
            "artifact": relative_path,
            "test": _extract_test(payload),
        })

    completed = [row for row in models if row["status"] == "completed"]
    ranked = sorted(
        completed,
        key=lambda row: row["test"]["pr_auc"] if row["test"]["pr_auc"] is not None else float("-inf"),
        reverse=True,
    )
    return {
        "purpose": "single-source summary of completed forward-test model benchmarks",
        "primary_metric": "pr_auc",
        "ranking_rule": "completed models ranked by test PR-AUC; pending experiments are excluded",
        "models": models,
        "ranking_by_test_pr_auc": [row["model"] for row in ranked],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a model benchmark summary from saved metric artifacts.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="artifacts/model_comparison.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    report = build_report(root)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report: {output}")


if __name__ == "__main__":
    main()
