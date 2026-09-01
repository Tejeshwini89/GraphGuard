# Phase 0 — Dataset Forensics

## Objective

Establish exactly what is in the Elliptic Bitcoin dataset before selecting model inputs, graph construction rules, or temporal split boundaries.

## Questions We Must Answer

1. How many transactions are represented?
2. How many directed transaction-flow edges are present?
3. How many numeric features are available?
4. What are the label values and their counts?
5. How many labels are unknown?
6. How are transactions distributed across time steps 1–49?
7. Are any columns identifiers, timestamps, or target-derived information that could leak labels?
8. Which features can legitimately be available at prediction time?
9. Should the main experiment be node classification on the full graph with temporal masks, or a sequence of timestamped graph snapshots?
10. What chronological train/validation/test boundaries preserve the intended forecasting setting?

## Ground Rules

- Unknown labels are excluded from supervised loss and evaluation.
- No random split will be used as the primary experiment.
- Features will be audited before modeling.
- Target leakage must be documented, not silently removed.
- Every model will use the same main evaluation protocol.
- Any graph transformation must preserve the real transaction-flow semantics.

## Expected Benchmark Facts

PyTorch Geometric documents 203,769 transaction nodes, 234,355 directed payment-flow edges, 165 node features, approximately 4,545 illicit labels and 42,019 licit labels, with the remaining transactions unknown. The temporal variant represents time steps 1 through 49.

These figures are sanity checks, not substitutes for running our own forensic script.

## First Command

```powershell
python scripts\inspect_elliptic.py
```

The command downloads/loads the dataset through PyTorch Geometric and prints the dimensions, label distribution, unknown-label count, illicit rate among known labels, and available time-step information.

## Output to Preserve

Save the terminal output as the initial Phase 0 evidence. Do not commit the downloaded dataset itself unless licensing and repository-size constraints explicitly permit it.

## Next Gate

We do not begin XGBoost or GNN training until the dataset audit answers the questions above and the temporal split is written down in code/configuration.
