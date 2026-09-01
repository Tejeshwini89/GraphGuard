from __future__ import annotations

import argparse
from pathlib import Path

from torch_geometric.datasets import EllipticBitcoinDataset


def summarize(root: Path) -> None:
    """Download/load the Elliptic graph and print dataset forensics."""
    dataset = EllipticBitcoinDataset(root=str(root))
    graph = dataset[0]

    print("=== GraphGuard: Elliptic Dataset Forensics ===")
    print(f"root: {root.resolve()}")
    print(f"nodes: {graph.num_nodes}")
    print(f"edges: {graph.num_edges}")
    print(f"node_features: {graph.num_node_features}")
    print(f"classes: {dataset.num_classes}")
    print(f"x_shape: {tuple(graph.x.shape)}")
    print(f"edge_index_shape: {tuple(graph.edge_index.shape)}")
    print(f"label_shape: {tuple(graph.y.shape)}")

    labels = graph.y
    known = labels[labels >= 0]
    unknown = int((labels < 0).sum().item())
    illicit = int((known == 1).sum().item())
    licit = int((known == 0).sum().item())

    print(f"known_labels: {int(known.numel())}")
    print(f"unknown_labels: {unknown}")
    print(f"licit_labels: {licit}")
    print(f"illicit_labels: {illicit}")
    if known.numel():
        print(f"illicit_rate_among_known: {illicit / int(known.numel()):.6f}")

    if hasattr(graph, "time_step"):
        values = graph.time_step.unique(sorted=True)
        print(f"time_steps: {values.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw", help="Dataset root directory")
    args = parser.parse_args()
    summarize(Path(args.root))


if __name__ == "__main__":
    main()
