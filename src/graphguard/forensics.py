from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

from graphguard.elliptic import load_elliptic_graph


@dataclass(frozen=True)
class DatasetSummary:
    """Reproducible summary of the normalized Elliptic graph."""

    nodes: int
    edges: int
    node_features: int
    known_labels: int
    unknown_labels: int
    licit_labels: int
    illicit_labels: int
    illicit_rate_among_known: float
    min_time_step: int | None
    max_time_step: int | None
    time_step_counts: dict[str, int]
    feature_constant_count: int
    feature_missing_value_count: int


def summarize_graph(data: Data) -> DatasetSummary:
    """Compute deterministic dataset statistics from a normalized graph."""
    labels = data.y.to(torch.long)
    known = labels[labels >= 0]
    unknown = int((labels < 0).sum().item())
    illicit = int((known == 1).sum().item())
    licit = int((known == 0).sum().item())
    known_count = int(known.numel())

    time_steps = data.time_step.to(torch.long)
    if time_steps.numel() == 0:
        min_time = max_time = None
        time_counts: dict[str, int] = {}
    else:
        values, counts = torch.unique(time_steps, sorted=True, return_counts=True)
        min_time = int(values.min().item())
        max_time = int(values.max().item())
        time_counts = {str(int(v.item())): int(c.item()) for v, c in zip(values, counts)}

    x = data.x
    finite = torch.isfinite(x)
    feature_missing_value_count = int((~finite).sum().item())
    feature_constant_count = int((x.nan_to_num(nan=0.0).std(dim=0) == 0).sum().item())

    return DatasetSummary(
        nodes=int(data.num_nodes),
        edges=int(data.num_edges),
        node_features=int(data.num_node_features),
        known_labels=known_count,
        unknown_labels=unknown,
        licit_labels=licit,
        illicit_labels=illicit,
        illicit_rate_among_known=(illicit / known_count) if known_count else 0.0,
        min_time_step=min_time,
        max_time_step=max_time,
        time_step_counts=time_counts,
        feature_constant_count=feature_constant_count,
        feature_missing_value_count=feature_missing_value_count,
    )


def audit_labels(data: Data) -> dict[str, Any]:
    """Return normalized label counts and invariants."""
    counts = Counter(int(v) for v in data.y.tolist())
    return {
        "label_counts": {str(k): v for k, v in sorted(counts.items())},
        "supported_supervised_labels": sorted(k for k in counts if k in (0, 1)),
        "has_unknown_label": -1 in counts,
    }


def run_forensics(root: str | Path) -> dict[str, Any]:
    """Load the Elliptic graph and return a machine-readable normalized audit."""
    graph = load_elliptic_graph(root)
    summary = summarize_graph(graph)
    return {
        "dataset": "elliptic-bitcoin",
        "summary": asdict(summary),
        "labels": audit_labels(graph),
        "edges_are_directed_in_source": True,
        "supervised_labels": [0, 1],
        "unknown_label": -1,
        "notes": [
            "Raw labels are normalized as class 1 -> illicit, class 2 -> licit, unknown -> -1.",
            "Unknown labels are excluded from supervised training/evaluation.",
            "Primary experiments must use chronological rather than random splitting.",
            "Feature semantics must be audited before selecting model inputs.",
        ],
    }


def save_forensics(root: str | Path, output: str | Path) -> Path:
    """Run the audit and save JSON output outside the tracked dataset."""
    import json

    report = run_forensics(root)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return destination
