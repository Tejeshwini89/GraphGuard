# GraphGuard

GraphGuard is a graph-based financial-forensics system for detecting illicit Bitcoin transactions. The project is designed as a rigorous ML engineering study: establish a strong XGBoost tabular baseline, then test whether graph-aware learning with GraphSAGE and GAT adds predictive value under leakage-safe temporal evaluation.

## Research Question

> Does graph-aware learning improve illicit transaction detection compared with a strong tabular XGBoost baseline?

## Primary Dataset

GraphGuard uses the Elliptic Bitcoin transaction dataset. The source contains 203,769 transaction nodes, 234,355 directed payment-flow edges, 166 columns in the raw feature file, and 165 model features after excluding `txId` and `time_step`. The raw labels are `1` = illicit, `2` = licit, and `unknown` = unlabeled. GraphGuard normalizes these to `1`, `0`, and `-1` respectively. The dataset covers 49 time steps.

## Architecture

```text
                 Elliptic Transactions
                         |
                 Data Validation
                         |
                Temporal Split / Leakage Audit
                         |
            +------------+-------------+
            |                          |
      Tabular Features            Graph Structure
            |                          |
         XGBoost                PyTorch Geometric
            |                    |             |
            |               GraphSAGE          GAT
            |                    |             |
            +------------+-------+-------------+
                         |
                    Model Comparison
                         |
                  Fraud Probability
                         |
                  Threshold Analysis
                         |
             +-----------+-----------+
             |                       |
          Evaluation            Investigation
                                      |
                                    Neo4j
                                      |
                                  FastAPI
                                      |
                                   Docker
```

## Why This Dataset?

The dataset is a natural graph-learning benchmark because transactions are connected through directed Bitcoin payment flows, and labels are attached to transaction nodes. Its temporal organization lets GraphGuard test forward-looking generalization rather than relying on a random split.

## Phase 0 — Completed Forensics Gate

The local dataset audit confirmed:

- 203,769 transaction nodes
- 234,355 directed edges
- 165 model features
- 42,019 licit transactions
- 4,545 illicit transactions
- 157,205 unknown transactions
- 49 time steps
- 2.23% illicit among labeled transactions

A critical implementation detail was discovered and fixed: PyTorch Geometric's processed label encoding does not match the raw Elliptic label semantics. GraphGuard therefore normalizes labels directly from the raw class file rather than trusting the processed `graph.y` encoding. This prevents the 157,205 unknown transactions from being accidentally treated as labeled examples.

The forensic report also records class counts per time step so that the chronological split is frozen only after checking that each evaluation window contains enough labeled examples of both classes.

## Planned Experimental Protocol

1. Normalize and validate raw labels and temporal metadata.
2. Inspect class counts by time step and freeze the chronological split.
3. Audit features and identifiers for target leakage.
4. Exclude unknown labels from supervised loss/evaluation while retaining their graph nodes where appropriate for transductive message passing.
5. Train an XGBoost baseline on tabular transaction features.
6. Report PR-AUC, ROC-AUC, precision, recall, F1, confusion matrix, and threshold behavior.
7. Train GraphSAGE on the transaction graph.
8. Train GAT on the same leakage-safe protocol.
9. Compare all models and perform error analysis.
10. Materialize useful graph relationships in Neo4j for investigation.
11. Expose prediction/investigation functions through FastAPI.
12. Package the system with Docker.
13. Add tests, CI, reproducible evaluation and documentation.

## Important Modeling Rules

- Accuracy is not the primary fraud metric because the labeled classes are highly imbalanced.
- Randomly mixing future transactions into training is not acceptable for the main experiment.
- Unknown labels are not silently converted into legitimate transactions.
- GraphGuard will not claim that a GNN is better until the measured evaluation supports that claim.
- Neo4j is an investigation layer, not a substitute for the GNN training graph.
- The downloaded dataset is kept out of GitHub; only code, configuration, tests, reports, and documentation are versioned.

## Current Project State

**Phase 0 — Forensics implemented and dataset semantics corrected.** The repository now contains a normalized Elliptic loader, deterministic forensic reporting, per-timestep label auditing, temporal split utilities, tests, CI, and pinned dependencies. No model result is claimed yet.

## License

MIT
