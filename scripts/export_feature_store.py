from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from graphguard.elliptic import load_elliptic_graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export transaction IDs and model-ready Elliptic features for API explainability."
    )
    parser.add_argument("--root", default="data/raw")
    parser.add_argument(
        "--output",
        default="artifacts/features/transaction_features.npz",
    )
    args = parser.parse_args()

    graph = load_elliptic_graph(args.root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    tx_ids = graph.tx_id.detach().cpu().numpy().astype(np.int64, copy=True)
    features = graph.x.detach().cpu().numpy().astype(np.float32, copy=True)

    if features.ndim != 2 or features.shape[1] != 165:
        raise ValueError(f"Expected 165 model features, got {features.shape}")
    if len(tx_ids) != len(features):
        raise ValueError("Transaction IDs and features are misaligned")

    np.savez_compressed(output, tx_id=tx_ids, features=features)
    print(f"exported_transactions: {len(tx_ids)}")
    print(f"exported_features: {features.shape[1]}")
    print(f"feature_store: {output}")


if __name__ == "__main__":
    main()
