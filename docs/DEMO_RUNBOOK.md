# GraphGuard Demo Runbook

This runbook validates the final investigator workflow after the ML experiments are complete.

## 1. Update the local checkout

```powershell
git pull origin main
```

## 2. Verify the test suite

```powershell
python -m pytest -q
```

Expected baseline: all tests pass. PyTorch deprecation warnings do not count as test failures.

## 3. Prepare the API model

The frozen XGBoost artifact is expected at:

```text
artifacts/baseline/xgboost.json
```

Model artifacts are intentionally not committed to GitHub.

## 4. Start the stack

Create `.env` from `.env.example` and set a strong local Neo4j password, then run:

```powershell
docker compose up -d --build
```

The API is exposed on port `8000`; Neo4j Browser is exposed on port `7474`.

## 5. Check the API

```powershell
python scripts\smoke_test_api.py
```

The smoke test verifies `/health`, `/model`, `/predict`, and `/explain` without requiring an additional HTTP client package.

## 6. Populate Neo4j

After the Neo4j service is healthy and the model artifact is mounted:

```powershell
python scripts\build_neo4j.py
```

The loader stores transaction IDs, time steps, frozen XGBoost risk scores, and payment-flow relationships. Raw model features and labels are not copied into Neo4j.

## 7. Investigator workflow

Use the API in this order:

1. Score a transaction with `/predict`.
2. Explain the score with `/explain`.
3. Look up the transaction with `/transactions/{tx_id}`.
4. Inspect connected high-risk transactions with `/transactions/{tx_id}/neighbors`.

The intended product flow is:

```text
risk score → model evidence → transaction context → suspicious neighbors
```

## 8. Final validation checklist

- [ ] `python -m pytest -q` passes locally
- [ ] Docker image builds successfully
- [ ] API container starts
- [ ] Neo4j container starts with a configured password
- [ ] `python scripts\smoke_test_api.py` passes
- [ ] `python scripts\build_neo4j.py` loads the graph successfully
- [ ] Transaction lookup returns a stored transaction
- [ ] Suspicious-neighbor query returns graph context
- [ ] `/explain` returns signed local XGBoost contributions

## Limitations to state in the demo

- The Elliptic features are anonymized; feature indices are not business-semantic feature names.
- XGBoost is the selected risk engine because it had the strongest measured forward-test PR-AUC.
- GNN experiments were retained as model-investigation evidence rather than forced into production.
- Temporal error analysis shows substantial distribution shift in later test periods.
- Model contributions are explanations of model behavior, not causal claims.
