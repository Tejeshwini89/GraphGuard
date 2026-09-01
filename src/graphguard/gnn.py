from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from graphguard.splits import TemporalSplit


@dataclass(frozen=True)
class GNNMetrics:
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float


@dataclass(frozen=True)
class FeatureScaler:
    mean: Tensor
    std: Tensor

    def transform(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std


def fit_feature_scaler(data: Data, train_mask: Tensor) -> FeatureScaler:
    """Fit normalization on training nodes only to prevent feature leakage."""
    if train_mask.ndim != 1 or train_mask.numel() != data.num_nodes:
        raise ValueError("train_mask must match the number of graph nodes")
    x_train = data.x[train_mask]
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return FeatureScaler(mean=mean, std=std)


class GraphSAGE(nn.Module):
    """GraphSAGE node classifier for illicit-transaction detection."""

    def __init__(self, in_channels: int, hidden_channels: int = 128, layers: int = 2, dropout: float = 0.25) -> None:
        super().__init__()
        if layers < 2:
            raise ValueError("GraphSAGE requires at least two layers")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.convs = nn.ModuleList([SAGEConv(in_channels, hidden_channels)])
        for _ in range(layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
            x = self.dropout(x)
        return self.classifier(x).squeeze(-1)


def _metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> GNNMetrics:
    prediction = (probability >= threshold).astype(np.int64)
    return GNNMetrics(
        pr_auc=float(average_precision_score(y_true, probability)),
        roc_auc=float(roc_auc_score(y_true, probability)),
        precision=float(precision_score(y_true, prediction, zero_division=0)),
        recall=float(recall_score(y_true, prediction, zero_division=0)),
        f1=float(f1_score(y_true, prediction, zero_division=0)),
        threshold=float(threshold),
    )


def select_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Select classification threshold on validation data only."""
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        score = _metrics(y_true, probability, float(threshold)).f1
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def evaluate_logits(logits: Tensor, labels: Tensor, mask: Tensor, threshold: float = 0.5) -> GNNMetrics:
    """Evaluate only the labeled nodes selected by mask."""
    indices = mask.detach().cpu()
    y_true = labels[indices].detach().cpu().numpy().astype(np.int64)
    probability = torch.sigmoid(logits[indices]).detach().cpu().numpy()
    if np.unique(y_true).size < 2:
        raise ValueError("Evaluation mask must contain both classes")
    return _metrics(y_true, probability, threshold)


def train_graphsage(
    data: Data,
    split: TemporalSplit,
    *,
    hidden_channels: int = 128,
    layers: int = 2,
    dropout: float = 0.25,
    epochs: int = 100,
    learning_rate: float = 0.003,
    weight_decay: float = 1e-4,
    patience: int = 12,
    seed: int = 42,
    device: str | None = None,
) -> tuple[GraphSAGE, FeatureScaler, dict[str, float | int]]:
    """Train with masked temporal supervision and validation early stopping.

    This is a transductive temporal protocol: later graph structure/features can
    participate in message passing, but later labels are never used for fitting,
    threshold selection, or checkpoint selection.
    """
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

    model = GraphSAGE(data.num_node_features, hidden_channels, layers, dropout).to(target_device)
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
        raise RuntimeError("GraphSAGE training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(target_device)
    return model, scaler, {"best_epoch": best_epoch, "best_validation_pr_auc": best_val_pr_auc}


def save_graphsage_checkpoint(
    model: GraphSAGE,
    scaler: FeatureScaler,
    output: str | Path,
    *,
    config: dict[str, int | float | str],
) -> Path:
    """Save weights, training-only scaler parameters, and configuration."""
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": scaler.mean.cpu(),
            "feature_std": scaler.std.cpu(),
            "config": config,
        },
        destination,
    )
    return destination
