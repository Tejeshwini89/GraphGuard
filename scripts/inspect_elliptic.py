from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.datasets import EllipticBitcoinDataset


def summarize(root: Path, output: Path | None = None) -> dict:
    """Download/load the Elliptic graph and print/save dataset forensics."""
    dataset = EllipticBitcoinDataset(root=str(root))
    graph = dataset[0]

    # PyG uses the raw feature file to construct x from columns 2 onward;
    # time_step is therefore audited from the original source column instead
    # of being assumed to be part of graph.x.
    feature_path = Path(dataset.raw_paths[0])
    features = pd.read_csv(feature_path, header=None)
    features = features.rename(columns={0: "txId", 1: "time_step"})

    class_path = Path(dataset.raw_paths[2])
    classes = pd.read_csv(class_path)
    class_counts_raw = Counter(classes["class"].astype(str))

    time_counts = features["time_step"].astype(int).value_counts().sort_index().to_dict()

    labels = graph.y.to(torch.long)
    known = labels[labels >= 0]
    unknown = int((labels < 0).sum().item())
    illicit = int((known == 1).sum().item())
    licit = int((known == 0).sum().item())
    known_count = int(known.numel())

    report = {
        "root": str(root.resolve()),
        "nodes": int(graph.num_nodes),
        "edges": int(graph.num_edges),
        "node_features": int(graph.num_node_features),
        "classes": int(dataset.num_classes),
        "known_labels": known_count,
        "unknown_labels": unknown,
        "licit_labels": licit,
        "illicit_labels": illicit,
        "illicit_rate_among_known": (illicit / known_count) if known_count else 0.0,
        "time_step_min": int(features["time_step"].min()),
        "time_step_max": int(features["time_step"].max()),
        "time_step_count": len(time_counts),
        "time_step_counts": {str(k): int(v) for k, v in time_counts.items()},
        "raw_class_counts": dict(class_counts_raw),
        "raw_feature_columns": int(features.shape[1]),
        "model_feature_columns": int(graph.num_node_features),
        "notes": [
            "time_step is metadata used for temporal evaluation and is not included in graph.x.",
            "Unknown labels are excluded from supervised training/evaluation.",
            "Temporal split boundaries must be selected before model training.",
            "No random split is used as the primary benchmark.",
        ],
    }

    print("=== GraphGuard: Elliptic Dataset Forensics ===")
    for key in (
        "root", "nodes", "edges", "node_features", "classes", "known_labels",
        "unknown_labels", "licit_labels", "illicit_labels", "illicit_rate_among_known",
        "time_step_min", "time_step_max", "time_step_count",
    ):
        print(f"{key}: {report[key]}")
    print("time_step_counts:")
    for timestep, count in report["time_step_counts"].items():
        print(f"  {timestep}: {count}")
    print(f"raw_class_counts: {report['raw_class_counts']}")
    print(f"report: {output.resolve() if output is not None else '(not saved)'}")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw", help="Dataset root directory")
    parser.add_argument(
        "--output",
        default="artifacts/forensics/dataset_report.json",
        help="JSON report path",
    )
    args = parser.parse_args()
    summarize(Path(args.root), Path(args.output))


if __name__ == "__main__":
    main()
