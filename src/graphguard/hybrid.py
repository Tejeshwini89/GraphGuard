from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from graphguard.gnn import FeatureScaler, fit_feature_scaler
from graphguard.splits import TemporalSplit


@dataclass(frozen=True)
class HybridMetrics:
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float


def _metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> HybridMetrics:
    prediction = (probability >= threshold).astype(np.int64)
    return HybridMetrics(
        pr_auc=float(average_precision_score(y_true, probability)),
        roc_auc=float(roc_auc_score(y_true, probability)),
        precision=float(precision_score(y_true, prediction, zero_division=0)),
        recall=float(recall_score(y_true, prediction, zero_division=0)),
        f1=float(f1_score(y_true, prediction, zero_division=0)),
        threshold=float(threshold),
    )


def select_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        score = _metrics(y_true, probability, float(threshold)).f1
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def evaluate_logits(logits: Tensor, labels: Tensor, mask: Tensor, threshold: float = 0.5) -> HybridMetrics:
    indices = mask.detach().cpu()
    y_true = labels[indices].detach().cpu().numpy().astype(np.int64)
    probability = torch.sigmoid(logits[indices]).detach().cpu().numpy()
    if np.unique(y_true).size < 2:
        raise ValueError("Evaluation mask must contain both classes")
    return _metrics(y_true, probability, threshold)


class FeatureGraphHybrid(nn.Module):
    """Fuse raw transaction features with a learned GraphSAGE embedding."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        graph_layers: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if graph_layers < 1:
            raise ValueError("graph_layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.feature_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.graph_convs = nn.ModuleList([SAGEConv(in_channels, hidden_channels)])
        for _ in range(graph_layers - 1):
            self.graph_convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        feature_embedding = self.feature_encoder(x)
        graph_embedding = x
        for conv in self.graph_convs:
            graph_embedding = torch.relu(conv(graph_embedding, edge_index))
            graph_embedding = self.dropout(graph_embedding)
        fused = torch.cat([feature_embedding, graph_embedding], dim=-1)
        return self.classifier(fused).squeeze(-1)


def train_hybrid(
    data: Data,
    split: TemporalSplit,
    *,
    hidden_channels: int = 128,
    graph_layers: int = 2,
    dropout: float = 0.25,
    epochs: int = 100,
    learning_rate: float = 0.003,
    weight_decay: float = 1e-4,
    patience: int = 12,
    seed: int = 42,
    device: str | None = None,
) -> tuple[FeatureGraphHybrid, FeatureScaler, dict[str, float | int]]:
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data = data.to(target_device)
    train_mask = split.train_mask.to(target_device)
    validation_mask = split.validation_mask.to(target_device)
    scaler = fit_feature_scaler(data, train_mask)
    x = scaler.transform(data.x)
    y = data.y

    model = FeatureGraphHybrid(data.num_node_features, hidden_channels, graph_layers, dropout).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_labels = y[train_mask]
    positives = (train_labels == 1).sum().float().clamp_min(1)
    negatives = (train_labels == 0).sum().float().clamp_min(1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=negatives / positives)

    best_state: dict[str, Tensor] | None = None
    best_val_pr_auc = -float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, data.edge_index)
        loss = criterion(logits[train_mask], train_labels.float())
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(x, data.edge_index)
            validation = evaluate_logits(validation_logits, y, validation_mask)
        if validation.pr_auc > best_val_pr_auc + 1e-6:
            best_val_pr_auc = validation.pr_auc
            best_epoch = epoch
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("Hybrid training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(target_device)
    return model, scaler, {"best_epoch": best_epoch, "best_validation_pr_auc": best_val_pr_auc}
