from __future__ import annotations

from pathlib import Path

import numpy as np


class TransactionFeatureStore:
    """Read-only lookup store for model-ready Elliptic transaction features."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Feature artifact not found: {self.path}")

        artifact = np.load(self.path, allow_pickle=False)
        self.tx_ids = np.asarray(artifact["tx_id"], dtype=np.int64)
        self.features = np.asarray(artifact["features"], dtype=np.float32)
        artifact.close()

        if self.tx_ids.ndim != 1:
            raise ValueError("tx_id must be a one-dimensional array")
        if self.features.ndim != 2 or self.features.shape[1] != 165:
            raise ValueError("features must have shape (node_count, 165)")
        if len(self.tx_ids) != len(self.features):
            raise ValueError("tx_id and features must contain the same number of rows")
        if len(np.unique(self.tx_ids)) != len(self.tx_ids):
            raise ValueError("tx_id values must be unique")

        self._row_by_tx_id = {int(tx_id): index for index, tx_id in enumerate(self.tx_ids)}

    def get_features(self, tx_id: int) -> np.ndarray | None:
        """Return one transaction's 165 model features, or None if absent."""
        index = self._row_by_tx_id.get(int(tx_id))
        if index is None:
            return None
        return self.features[index].copy()
