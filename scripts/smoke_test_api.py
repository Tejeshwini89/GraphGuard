from __future__ import annotations

import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = os.getenv("GRAPHGUARD_API", "http://127.0.0.1:8000").rstrip("/")


def request(method: str, path: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if body else {}
    req = Request(f"{BASE_URL}{path}", data=body, headers=headers, method=method)
    with urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    try:
        health = request("GET", "/health")
        model = request("GET", "/model")
        features = [0.0] * 165
        prediction = request("POST", "/predict", {"features": features})
        explanation = request("POST", "/explain", {"features": features, "top_k": 5})
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        print(f"SMOKE TEST FAILED: {exc}")
        return 1

    checks = {
        "health": health.get("status") == "ok",
        "model": model.get("model") == "XGBoost" and model.get("feature_count") == 165,
        "prediction": isinstance(prediction.get("risk_score"), (int, float)),
        "explanation": (
            isinstance(explanation.get("risk_score"), (int, float))
            and len(explanation.get("top_features", [])) == 5
        ),
    }

    print(json.dumps({"checks": checks, "health": health, "model": model}, indent=2))
    if not all(checks.values()):
        print("SMOKE TEST FAILED: one or more API checks failed")
        return 1

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
