import numpy as np
import torch

from graphguard.gat import GAT, select_threshold


def test_gat_output_shape() -> None:
    model = GAT(in_channels=4, hidden_channels=8, heads=2, layers=2, dropout=0.1)
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    logits = model(x, edge_index)
    assert logits.shape == (5,)


def test_gat_threshold_stays_in_expected_range() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probability = np.array([0.01, 0.05, 0.10, 0.80, 0.90, 0.95])
    threshold = select_threshold(y_true, probability)
    assert 0.05 <= threshold <= 0.95


def test_gat_rejects_single_layer() -> None:
    try:
        GAT(in_channels=4, layers=1)
    except ValueError as exc:
        assert "at least two" in str(exc)
    else:
        raise AssertionError("GAT should reject layers < 2")
