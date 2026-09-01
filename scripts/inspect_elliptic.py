from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphguard.forensics import run_forensics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw", help="Dataset root directory")
    parser.add_argument(
        "--output",
        default="artifacts/forensics/dataset_report.json",
        help="JSON report path",
    )
    args = parser.parse_args()

    report = run_forensics(Path(args.root))
    print("=== GraphGuard: Elliptic Dataset Forensics ===")
    summary = report["summary"]
    for key in (
        "nodes",
        "edges",
        "node_features",
        "known_labels",
        "unknown_labels",
        "licit_labels",
        "illicit_labels",
        "illicit_rate_among_known",
        "min_time_step",
        "max_time_step",
        "feature_constant_count",
        "feature_missing_value_count",
    ):
        print(f"{key}: {summary[key]}")
    print("time_step_counts:")
    for timestep, count in summary["time_step_counts"].items():
        print(f"  {timestep}: {count}")
    print("time_step_label_counts:")
    for timestep, counts in summary["time_step_label_counts"].items():
        print(
            f"  {timestep}: unknown={counts['unknown']}, "
            f"licit={counts['licit']}, illicit={counts['illicit']}"
        )
    print(f"labels: {report['labels']['label_counts']}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report: {output.resolve()}")


if __name__ == "__main__":
    main()
