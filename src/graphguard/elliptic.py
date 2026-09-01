from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import EllipticBitcoinDataset


RAW_CLASS_MAP = {"unknown": -1, "1": 1, "2": 0}


def _writable_int64(values: pd.Series) -> Tensor:
    """Convert a pandas column to an owned, writable int64 tensor."""
    array = values.to_numpy(dtype="int64", copy=True)
    return torch.from_numpy(array)


def load_elliptic_graph(root: str | Path = "data/raw") -> Data:
    """Load the Elliptic graph with labels/time normalized from raw CSVs.

    PyTorch Geometric's processed dataset uses its own class encoding. GraphGuard
    deliberately does not rely on that encoding because the original dataset
    semantics are: class 1 = illicit, class 2 = licit, unknown = unlabeled.
    """
    dataset = EllipticBitcoinDataset(root=str(root))
    if len(dataset) != 1:
        raise ValueError(f"Expected one Elliptic graph, found {len(dataset)}")

    graph = dataset[0]
    feature_path = Path(dataset.raw_paths[0])
    class_path = Path(dataset.raw_paths[2])

    features = pd.read_csv(feature_path, header=None)
    if features.shape[1] != graph.num_node_features + 2:
        raise ValueError(
            "Unexpected feature layout: expected txId + time_step + model features, "
            f"got {features.shape[1]} columns for {graph.num_node_features} model features"
        )

    tx_ids = _writable_int64(features.iloc[:, 0])
    time_step = _writable_int64(features.iloc[:, 1])

    classes = pd.read_csv(class_path)
    if list(classes.columns) != ["txId", "class"]:
        raise ValueError(f"Unexpected class columns: {list(classes.columns)}")

    classes["txId"] = classes["txId"].astype("int64")
    classes["class"] = classes["class"].astype(str).str.strip()
    unknown_classes = sorted(set(classes["class"]) - RAW_CLASS_MAP.keys())
    if unknown_classes:
        raise ValueError(f"Unexpected raw class values: {unknown_classes}")

    label_by_tx = dict(zip(classes["txId"], classes["class"], strict=True))
    missing_labels = [tx_id.item() for tx_id in tx_ids if tx_id.item() not in label_by_tx]
    if missing_labels:
        raise ValueError(f"Missing labels for {len(missing_labels)} transactions")

    labels = torch.tensor(
        [RAW_CLASS_MAP[label_by_tx[tx_id.item()]] for tx_id in tx_ids],
        dtype=torch.long,
    )

    if len(tx_ids) != graph.num_nodes:
        raise ValueError("Raw feature rows and graph nodes are misaligned")

    return Data(
        x=graph.x.float(),
        edge_index=graph.edge_index.long(),
        y=labels,
        time_step=time_step,
        tx_id=tx_ids,
    )
