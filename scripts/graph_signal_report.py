from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import yaml

from graphguard.elliptic import load_elliptic_graph
from graphguard.splits import make_temporal_split


def edge_homophily(data, mask: torch.Tensor | None = None) -> dict[str, float | int]:
    src, dst = data.edge_index
    edge_mask = torch.ones(src.numel(), dtype=torch.bool)
    if mask is not None:
        edge_mask = mask[src] & mask[dst]
    src = src[edge_mask]
    dst = dst[edge_mask]
    labeled = (data.y[src] >= 0) & (data.y[dst] >= 0)
    src = src[labeled]
    dst = dst[labeled]
    if src.numel() == 0:
        return {"edges": 0, "same_label_rate": float("nan"), "illicit_illicit_rate": float("nan")}
    same = data.y[src] == data.y[dst]
    illicit_illicit = (data.y[src] == 1) & (data.y[dst] == 1)
    return {
        "edges": int(src.numel()),
        "same_label_rate": float(same.float().mean()),
        "illicit_illicit_rate": float(illicit_illicit.float().mean()),
    }


def class_neighbor_purity(data, mask: torch.Tensor) -> dict[str, float | int]:
    src, dst = data.edge_index
    edge_mask = mask[src] & mask[dst]
    src, dst = src[edge_mask], dst[edge_mask]
    labeled = (data.y[src] >= 0) & (data.y[dst] >= 0)
    src, dst = src[labeled], dst[labeled]
    result: dict[str, float | int] = {"edges": int(src.numel())}
    for label, name in [(0, "licit"), (1, "illicit")]:
        rows = data.y[src] == label
        if bool(rows.any()):
            result[f"{name}_neighbor_same_label_rate"] = float((data.y[dst[rows]] == label).float().mean())
        else:
            result[f"{name}_neighbor_same_label_rate"] = float("nan")
    return result


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    data = load_elliptic_graph(config["dataset"]["root"])
    split_cfg = config["splits"]
    split = make_temporal_split(
        data.time_step, data.y,
        train_end=int(split_cfg["train_end"]),
        validation_start=int(split_cfg["validation_start"]),
        validation_end=int(split_cfg["validation_end"]),
        test_start=int(split_cfg["test_start"]),
    )

    global_signal = edge_homophily(data)
    report = {
        "purpose": "quantify whether labeled graph topology contains class signal",
        "global_labeled_edge_signal": global_signal,
        "train_labeled_edge_signal": edge_homophily(data, split.train_mask),
        "validation_labeled_edge_signal": edge_homophily(data, split.validation_mask),
        "test_labeled_edge_signal": edge_homophily(data, split.test_mask),
        "train_class_neighbor_purity": class_neighbor_purity(data, split.train_mask),
        "validation_class_neighbor_purity": class_neighbor_purity(data, split.validation_mask),
        "test_class_neighbor_purity": class_neighbor_purity(data, split.test_mask),
        "interpretation_note": (
            "These statistics are forensic only. They use labels to characterize graph signal and must not be used as model features."
        ),
    }

    output = ROOT / "artifacts" / "forensics" / "graph_signal_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report: {output}")


if __name__ == "__main__":
    main()
