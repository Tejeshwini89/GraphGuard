from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from graphguard.elliptic import load_elliptic_graph
from graphguard.splits import make_temporal_split


def _class_counts(labels: torch.Tensor) -> dict[str, int]:
    return {
        "unknown": int((labels == -1).sum()),
        "licit": int((labels == 0).sum()),
        "illicit": int((labels == 1).sum()),
    }


def main() -> None:
    config_path = ROOT / "config.yaml"
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = load_elliptic_graph(config["dataset"]["root"])
    split_cfg = config["splits"]
    split = make_temporal_split(
        data.time_step,
        data.y,
        train_end=int(split_cfg["train_end"]),
        validation_start=int(split_cfg["validation_start"]),
        validation_end=int(split_cfg["validation_end"]),
        test_start=int(split_cfg["test_start"]),
    )

    src, dst = data.edge_index
    n = data.num_nodes
    in_degree = torch.bincount(dst, minlength=n)
    out_degree = torch.bincount(src, minlength=n)
    degree = in_degree + out_degree

    labeled = data.y >= 0
    labeled_edges = labeled[src] & labeled[dst]
    illicit_endpoint_edges = (data.y[src] == 1) | (data.y[dst] == 1)

    def neighborhood_rate(node_mask: torch.Tensor, target_label: int = 1) -> float:
        edge_src_mask = node_mask[src]
        if not bool(edge_src_mask.any()):
            return float("nan")
        neighbors = data.y[dst[edge_src_mask]]
        known = neighbors >= 0
        if not bool(known.any()):
            return float("nan")
        return float((neighbors[known] == target_label).float().mean())

    timestep_rows = []
    for t in range(int(data.time_step.min()), int(data.time_step.max()) + 1):
        nodes = data.time_step == t
        edges = nodes[src]
        known_nodes = nodes & labeled
        timestep_rows.append(
            {
                "time_step": t,
                "nodes": int(nodes.sum()),
                "known": int(known_nodes.sum()),
                "illicit": int((nodes & (data.y == 1)).sum()),
                "licit": int((nodes & (data.y == 0)).sum()),
                "directed_edges_from_period": int(edges.sum()),
                "mean_total_degree": float(degree[nodes].float().mean()),
                "illicit_neighbor_rate_from_period": neighborhood_rate(nodes),
            }
        )

    report = {
        "nodes": n,
        "edges": int(data.edge_index.shape[1]),
        "directed_edges_with_both_labeled_endpoints": int(labeled_edges.sum()),
        "edges_touching_illicit_endpoint": int(illicit_endpoint_edges.sum()),
        "isolated_nodes": int((degree == 0).sum()),
        "mean_in_degree": float(in_degree.float().mean()),
        "mean_out_degree": float(out_degree.float().mean()),
        "mean_total_degree": float(degree.float().mean()),
        "max_total_degree": int(degree.max()),
        "labels": _class_counts(data.y),
        "split_labeled_counts": {
            "train": _class_counts(data.y[split.train_mask]),
            "validation": _class_counts(data.y[split.validation_mask]),
            "test": _class_counts(data.y[split.test_mask]),
        },
        "same_timestep_edge_fraction": float(
            (data.time_step[src] == data.time_step[dst]).float().mean()
        ),
        "same_timestep_edge_count": int((data.time_step[src] == data.time_step[dst]).sum()),
        "timestep": timestep_rows,
    }

    output = ROOT / "artifacts" / "forensics" / "graph_diagnostics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report: {output}")


if __name__ == "__main__":
    main()
