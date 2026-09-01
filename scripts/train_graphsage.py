from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import yaml

from graphguard.elliptic import load_elliptic_graph
from graphguard.gnn import evaluate_logits, save_graphsage_checkpoint, select_threshold, train_graphsage
from graphguard.splits import make_temporal_split


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    seed = int(config["project"]["seed"])
    dataset_cfg = config["dataset"]
    split_cfg = config["splits"]
    model_cfg = config["models"]["graphsage"]

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

    model, scaler, training = train_graphsage(
        data,
        split,
        hidden_channels=int(model_cfg["hidden_channels"]),
        layers=int(model_cfg["layers"]),
        dropout=float(model_cfg["dropout"]),
        epochs=int(model_cfg.get("epochs", 100)),
        learning_rate=float(model_cfg.get("learning_rate", 0.003)),
        weight_decay=float(model_cfg.get("weight_decay", 1e-4)),
        patience=int(model_cfg.get("patience", 12)),
        seed=seed,
    )

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(scaler.transform(data.x.to(device)), data.edge_index.to(device))

    validation_indices = split.validation_mask.to(device)
    y_validation = data.y.to(device)[validation_indices].detach().cpu().numpy()
    p_validation = torch.sigmoid(logits[validation_indices]).detach().cpu().numpy()
    threshold = select_threshold(y_validation, p_validation)

    validation = evaluate_logits(logits, data.y.to(device), validation_indices, threshold)
    test = evaluate_logits(logits, data.y.to(device), split.test_mask.to(device), threshold)

    artifact = ROOT / "artifacts" / "gnn" / "graphsage.pt"
    save_graphsage_checkpoint(
        model,
        scaler,
        artifact,
        config={
            "model": "graphsage",
            "hidden_channels": int(model_cfg["hidden_channels"]),
            "layers": int(model_cfg["layers"]),
            "dropout": float(model_cfg["dropout"]),
            "seed": seed,
        },
    )

    metrics = {
        "model": "graphsage",
        "protocol": "transductive_temporal",
        "split": split_cfg,
        "threshold_selected_on": "validation",
        "threshold": threshold,
        "training": training,
        "validation": validation.__dict__,
        "test": test.__dict__,
    }
    metrics_path = ROOT / "artifacts" / "gnn" / "graphsage_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"artifact: {artifact}")


if __name__ == "__main__":
    main()
