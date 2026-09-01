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
from graphguard.hybrid import evaluate_logits, select_threshold, train_hybrid
from graphguard.splits import make_temporal_split


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    seed = int(config["project"]["seed"])
    dataset_cfg = config["dataset"]
    split_cfg = config["splits"]
    base_cfg = config["models"]["graphsage"]
    hybrid_cfg = config["models"].get("hybrid", {})

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

    model, scaler, training = train_hybrid(
        data,
        split,
        hidden_channels=int(hybrid_cfg.get("hidden_channels", base_cfg["hidden_channels"])),
        graph_layers=int(hybrid_cfg.get("graph_layers", base_cfg["layers"])),
        dropout=float(hybrid_cfg.get("dropout", base_cfg["dropout"])),
        epochs=int(hybrid_cfg.get("epochs", base_cfg.get("epochs", 100))),
        learning_rate=float(hybrid_cfg.get("learning_rate", base_cfg.get("learning_rate", 0.003))),
        weight_decay=float(hybrid_cfg.get("weight_decay", base_cfg.get("weight_decay", 1e-4))),
        patience=int(hybrid_cfg.get("patience", base_cfg.get("patience", 12))),
        seed=seed,
    )

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(scaler.transform(data.x.to(device)), data.edge_index.to(device))

    validation_mask = split.validation_mask.to(device)
    y_validation = data.y.to(device)[validation_mask].detach().cpu().numpy()
    p_validation = torch.sigmoid(logits[validation_mask]).detach().cpu().numpy()
    threshold = select_threshold(y_validation, p_validation)

    validation = evaluate_logits(logits, data.y.to(device), validation_mask, threshold)
    test = evaluate_logits(logits, data.y.to(device), split.test_mask.to(device), threshold)

    artifact = ROOT / "artifacts" / "gnn" / "hybrid.pt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": scaler.mean.cpu(),
            "feature_std": scaler.std.cpu(),
            "config": {
                "model": "feature_graph_hybrid",
                "hidden_channels": int(hybrid_cfg.get("hidden_channels", base_cfg["hidden_channels"])),
                "graph_layers": int(hybrid_cfg.get("graph_layers", base_cfg["layers"])),
                "dropout": float(hybrid_cfg.get("dropout", base_cfg["dropout"])),
                "seed": seed,
            },
        },
        artifact,
    )

    metrics = {
        "model": "feature_graph_hybrid",
        "protocol": "transductive_temporal",
        "split": split_cfg,
        "threshold_selected_on": "validation",
        "threshold": threshold,
        "training": training,
        "validation": validation.__dict__,
        "test": test.__dict__,
    }
    metrics_path = ROOT / "artifacts" / "gnn" / "hybrid_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"artifact: {artifact}")


if __name__ == "__main__":
    main()
