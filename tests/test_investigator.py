from graphguard.investigator import build_case_view


def test_build_case_view() -> None:
    result = build_case_view(
        transaction={"tx_id": 42, "time_step": 7, "risk_score": 0.9},
        explanation={"risk_score": 0.9, "top_features": []},
        neighbors=[{"tx_id": 9, "risk_score": 0.8}],
        threshold=0.36,
        neighbor_limit=10,
    )
    assert result["case"]["tx_id"] == 42
    assert result["case"]["flagged"] is True
    assert result["graph_context"]["neighbor_count"] == 1
