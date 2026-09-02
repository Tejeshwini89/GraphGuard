import torch
from torch_geometric.data import Data

from graphguard.forensics import audit_labels, summarize_graph


def test_summarize_graph_counts_known_and_unknown_labels():
    data = Data(
        x=torch.tensor(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
            ]
        ),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        y=torch.tensor([0, 1, -1, 1], dtype=torch.long),
        time_step=torch.tensor([1, 1, 2, 2], dtype=torch.long),
    )

    summary = summarize_graph(data)

    assert summary.nodes == 4
    assert summary.edges == 2
    assert summary.node_features == 2
    assert summary.known_labels == 3
    assert summary.unknown_labels == 1
    assert summary.licit_labels == 1
    assert summary.illicit_labels == 2
    assert summary.illicit_rate_among_known == 2 / 3
    assert summary.min_time_step == 1
    assert summary.max_time_step == 2


def test_summarize_graph_records_time_step_counts_when_available():
    data = Data(
        x=torch.tensor([[1.0], [2.0], [3.0]]),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        y=torch.tensor([0, 1, -1], dtype=torch.long),
        time_step=torch.tensor([1, 1, 2], dtype=torch.long),
    )

    summary = summarize_graph(data)

    assert summary.min_time_step == 1
    assert summary.max_time_step == 2
    assert summary.time_step_counts == {"1": 2, "2": 1}
    assert summary.time_step_label_counts == {
        "1": {"unknown": 0, "licit": 1, "illicit": 1},
        "2": {"unknown": 1, "licit": 0, "illicit": 0},
    }


def test_audit_labels_preserves_unknown_label_semantics():
    data = Data(y=torch.tensor([0, 0, 1, -1, -1], dtype=torch.long))

    report = audit_labels(data)

    assert report["label_counts"] == {"-1": 2, "0": 2, "1": 1}
    assert report["supported_supervised_labels"] == [0, 1]
    assert report["has_unknown_label"] is True
