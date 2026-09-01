import torch
from torch_geometric.data import Data

from graphguard.gnn import GraphSAGE, fit_feature_scaler, select_threshold


def test_graphsage_output_shape():
    model = GraphSAGE(in_channels=4, hidden_channels=8, layers=2, dropout=0.1)
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    logits = model(x, edge_index)
    assert logits.shape == (5,)


def test_feature_scaler_uses_training_nodes_only():
    data = Data(
        x=torch.tensor(
            [
                [1.0, 10.0],
                [3.0, 20.0],
                [100.0, 200.0],
                [300.0, 400.0],
            ]
        )
    )
    train_mask = torch.tensor([True, True, False, False])
    scaler = fit_feature_scaler(data, train_mask)
    assert torch.allclose(scaler.mean, torch.tensor([2.0, 15.0]))
    assert torch.all(scaler.std > 0)


def test_threshold_selection_stays_in_expected_range():
    y_true = torch.tensor([0, 0, 0, 1, 1, 1]).numpy()
    probability = torch.tensor([0.01, 0.05, 0.10, 0.80, 0.90, 0.95]).numpy()
    threshold = select_threshold(y_true, probability)
    assert 0.05 <= threshold <= 0.95
