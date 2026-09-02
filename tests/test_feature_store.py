import numpy as np
import pytest

from graphguard.feature_store import TransactionFeatureStore


def test_feature_store_lookup(tmp_path):
    path = tmp_path / "features.npz"
    tx_ids = np.array([101, 202], dtype=np.int64)
    features = np.arange(330, dtype=np.float32).reshape(2, 165)
    np.savez_compressed(path, tx_id=tx_ids, features=features)

    store = TransactionFeatureStore(path)

    result = store.get_features(202)
    assert result is not None
    assert result.shape == (165,)
    assert np.array_equal(result, features[1])
    assert store.get_features(999) is None


def test_feature_store_rejects_misaligned_artifact(tmp_path):
    path = tmp_path / "features.npz"
    np.savez_compressed(
        path,
        tx_id=np.array([101], dtype=np.int64),
        features=np.zeros((2, 165), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="same number of rows"):
        TransactionFeatureStore(path)
