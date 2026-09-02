from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graphguard.gat import save_gat_checkpoint, select_threshold, train_gat
from graphguard.elliptic import load_elliptic_graph
from graphguard.splits import make_temporal_split


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    dataset_cfg = config["dataset"]
    split_cfg = config["splits"]
    gat_cfg = config["models"]["gat"]

    data = load_elliptic_graph(dataset_cfg["root"])
    split = make_temporal_split(
        data.time_step,
        data.y,
        train_end=int(split_cfg["train_end"]),
        validation_start=int(split_cfg["validation_start"]),
        validation_end=int(split_cfg["validation_end"]),
        test_start=int(split_cfg["test_start"]),
        unknown_label=int(dataset_cfg["unknown_label"]),
    )
    model, scaler, training = train_gat(
        data,
        split,
        hidden_channels=int(gat_cfg["hidden_channels"]),
        heads=int(gat_cfg["heads"]),
        layers=int(gat_cfg["layers"]),
        dropout=float(gat_cfg["dropout"]),
        epochs=int(gat_cfg.get("epochs", 100)),
        learning_rate=float(gat_cfg.get("learning_rate", 0.003)),
        weight_decay=float(gat_cfg.get("weight_decay", 1e-4)),
        patience=int(gat_cfg.get("patience", 12)),
        seed=int(config["project"]["seed"]),
    )

    model.eval()
    device = next(model.parameters()).device
    x = scaler.transform(data.x.to(device))
    with torch.no_grad():
        logits = model(x, data.edge_index.to(device))
        probability = torch.sigmoid(logits).cpu().numpy()
    y = data.y.numpy()
    validation = split.validation_mask.numpy()
    test = split.test_mask.numpy()
    threshold = select_threshold(y[validation], probability[validation])

    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
    test_probability = probability[test]
    test_y = y[test]
    prediction = test_probability >= threshold
    metrics = {
        "pr_auc": float(average_precision_score(test_y, test_probability)),
        "roc_auc": float(roc_auc_score(test_y, test_probability)),
        "precision": float(precision_score(test_y, prediction, zero_division=0)),
        "recall": float(recall_score(test_y, prediction, zero_division=0)),
        "f1": float(f1_score(test_y, prediction, zero_division=0)),
        "threshold": float(threshold),
    }
    report = {
        "model": "gat",
        "protocol": "transductive_temporal",
        "split": {"train": "1-29", "validation": "30-34", "test": "35-49"},
        "threshold_selected_on": "validation",
        "training": training,
        "validation": {"pr_auc": float(average_precision_score(y[validation], probability[validation])), "threshold": float(threshold)},
        "test": metrics,
    }

    save_gat_checkpoint(model, scaler, ROOT / "artifacts" / "gnn" / "gat.pt", config=gat_cfg)
    output = ROOT / "artifacts" / "gnn" / "gat_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"checkpoint: {ROOT / 'artifacts' / 'gnn' / 'gat.pt'}")
    print(f"metrics: {output}")


if __name__ == "__main__":
    main()
