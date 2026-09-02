from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graphguard.elliptic import load_elliptic_graph
from graphguard.neo4j_store import Neo4jStore


def iter_transaction_rows(data, probability: np.ndarray):
    for index in range(data.num_nodes):
        yield {
            "tx_id": int(data.tx_id[index]),
            "time_step": int(data.time_step[index]),
            "risk_score": float(probability[index]),
        }


def iter_edge_rows(data):
    source = data.edge_index[0].numpy()
    target = data.edge_index[1].numpy()
    tx_ids = data.tx_id.numpy()
    for src, dst in zip(source, target, strict=True):
        yield {
            "source_tx_id": int(tx_ids[src]),
            "target_tx_id": int(tx_ids[dst]),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the GraphGuard investigation graph in Neo4j.")
    parser.add_argument("--dataset-root", default="data/raw")
    parser.add_argument("--model", default="artifacts/baseline/xgboost.json")
    args = parser.parse_args()

    data = load_elliptic_graph(args.dataset_root)
    model = XGBClassifier()
    model.load_model(ROOT / args.model)
    probability = model.predict_proba(data.x.numpy())[:, 1]

    store = Neo4jStore()
    try:
        store.create_schema()
        store.upsert_transactions(iter_transaction_rows(data, probability))
        store.upsert_edges(iter_edge_rows(data))
    finally:
        store.close()

    print(f"loaded_nodes: {data.num_nodes}")
    print(f"loaded_edges: {data.edge_index.shape[1]}")
    print("neo4j_graph: ready")


if __name__ == "__main__":
    main()
