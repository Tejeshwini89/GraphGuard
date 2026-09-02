from __future__ import annotations

import json

from scripts.model_comparison_report import build_report


def test_model_comparison_marks_missing_artifacts_pending(tmp_path) -> None:
    baseline = tmp_path / "artifacts" / "baseline"
    baseline.mkdir(parents=True)
    (baseline / "metrics.json").write_text(
        json.dumps({
            "test": {
                "pr_auc": 0.79,
                "roc_auc": 0.92,
                "precision": 0.68,
                "recall": 0.74,
                "f1": 0.71,
            }
        }),
        encoding="utf-8",
    )

    report = build_report(tmp_path)
    by_name = {row["model"]: row for row in report["models"]}
    assert by_name["XGBoost"]["status"] == "completed"
    assert by_name["GAT"]["status"] == "pending"
    assert report["ranking_by_test_pr_auc"] == ["XGBoost"]
