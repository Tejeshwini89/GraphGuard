from __future__ import annotations

from typing import Any


def build_case_view(
    *,
    transaction: dict[str, Any],
    explanation: dict[str, Any],
    neighbors: list[dict[str, Any]],
    threshold: float,
    neighbor_limit: int,
) -> dict[str, Any]:
    risk_score = float(explanation["risk_score"])
    return {
        "case": {
            "tx_id": transaction.get("tx_id"),
            "time_step": transaction.get("time_step"),
            "risk_score": risk_score,
            "threshold": threshold,
            "flagged": risk_score >= threshold,
        },
        "explanation": explanation,
        "graph_context": {
            "neighbor_count": len(neighbors),
            "neighbor_limit": neighbor_limit,
            "highest_risk_neighbors": neighbors,
        },
        "workflow": ["risk", "explanation", "graph_context"],
    }
