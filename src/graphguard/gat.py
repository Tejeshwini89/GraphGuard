from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from graphguard.gnn import FeatureScaler, fit_feature_scaler
from graphguard.splits import TemporalSplit


@dataclass(frozen=True)
class GATMetrics:
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float


class GAT(nn.Module):
    """Attention-based node classifier for the controlled graph hypothesis test."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, heads: int = 4, layers: int = 2, dropout: float = 0.25) -> None:
        super().__init__()
        if layers < 2:
            raise ValueError("GAT requires at least two layers")
        if heads < 1 or hidden_channels < 1:
            raise ValueError("heads and hidden_channels must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.convs = nn.ModuleList([GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)])
        for _ in range(layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv in self.convs[:-1]:
            x = torch.relu(conv(x, edge_index))
            x = self.dropout(x)
        x = torch.relu(self.convs[-1](x, edge_index))
        return self.classifier(x).squeeze(-1)


def _metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> GATMetrics:
    prediction = (probability >= threshold).astype(np.int64)
    return GATMetrics(
        pr_auc=float(average_precision_score(y_true, probability)),
        roc_auc=float(roc_auc_score(y_true, probability)),
        precision=float(precision_score(y_true, prediction, zero_division=0)),
        recall=float(recall_score(y_true, prediction, zero_division=0)),
        f1=float(f1_score(y_true, prediction, zero_division=0)),
        threshold=float(threshold),
    )


def select_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        score = _metrics(y_true, probability, float(threshold)).f1
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), score
    return best_threshold


def train_gat(
    data: Data,
    split: TemporalSplit,
    *,
    hidden_channels: int = 64,
    heads: int = 4,
    layers: int = 2,
    dropout: float = 0.25,
    epochs: int = 100,
    learning_rate: float = 0.003,
    weight_decay: float = 1e-4,
    patience: int = 12,
    seed: int = 42,
    device: str | None = None,
) -> tuple[GAT, FeatureScaler, dict[str, float | int]]:
    """Train GAT with exactly the GraphSAGE temporal protocol."""
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

    model = GAT(data.num_node_features, hidden_channels, heads, layers, dropout).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_labels = y[train_mask]
    positives = (train_labels == 1).sum().float().clamp_min(1)
    negatives = (train_labels == 0).sum().float().clamp_min(1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=negatives / positives)

    best_state = None
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
            probability = torch.sigmoid(validation_logits[validation_mask]).cpu().numpy()
        y_val = y[validation_mask].cpu().numpy().astype(np.int64)
        val_pr_auc = float(average_precision_score(y_val, probability))
        if val_pr_auc > best_val_pr_auc + 1e-6:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("GAT training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(target_device)
    return model, scaler, {"best_epoch": best_epoch, "best_validation_pr_auc": best_val_pr_auc}


def save_gat_checkpoint(model: GAT, scaler: FeatureScaler, output: str | Path, *, config: dict[str, int | float | str]) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "feature_mean": scaler.mean.cpu(), "feature_std": scaler.std.cpu(), "config": config}, destination)
    return destination
