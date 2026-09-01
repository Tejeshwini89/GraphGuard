# GraphGuard

GraphGuard is a graph-based financial-forensics system for detecting illicit Bitcoin transactions. The project is designed as a rigorous ML engineering study: establish a strong XGBoost tabular baseline, then test whether graph-aware learning with GraphSAGE and GAT adds predictive value under leakage-safe temporal evaluation.

## Research Question

> Does graph-aware learning improve illicit transaction detection compared with a strong tabular XGBoost baseline?

## Primary Dataset

GraphGuard uses the Elliptic Bitcoin transaction dataset. The PyTorch Geometric representation contains 203,769 transaction nodes, 234,355 directed payment-flow edges, 165 node features, and two labeled classes. Approximately 2% of nodes are labeled illicit, 21% licit, and the remaining transactions are unknown. The dataset is temporal, with 49 time steps. Unknown labels are not treated as negative fraud examples.

Source: Elliptic's public dataset description and PyTorch Geometric's `EllipticBitcoinDataset` / `EllipticBitcoinTemporalDataset` documentation.

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

## Planned Experimental Protocol

1. Download/load the Elliptic dataset through PyTorch Geometric.
2. Inspect dimensions, labels, missing values, and time-step distribution.
3. Audit features and identifiers for target leakage.
4. Exclude unknown labels from supervised training/evaluation.
5. Define a chronological train/validation/test split after inspecting the actual time distribution.
6. Train an XGBoost baseline on tabular transaction features.
7. Report PR-AUC, ROC-AUC, precision, recall, F1, confusion matrix, and threshold behavior.
8. Train GraphSAGE on the transaction graph.
9. Train GAT on the same leakage-safe protocol.
10. Compare all models and perform error analysis.
11. Materialize useful graph relationships in Neo4j for investigation.
12. Expose prediction/investigation functions through FastAPI.
13. Package the system with Docker.
14. Add tests, CI, reproducible evaluation and documentation.

## Important Modeling Rules

- Accuracy is not the primary fraud metric because the labeled classes are highly imbalanced.
- Randomly mixing future transactions into training is not acceptable for the main experiment.
- Unknown labels are not silently converted into legitimate transactions.
- GraphGuard will not claim that a GNN is better until the measured evaluation supports that claim.
- Neo4j is an investigation layer, not a substitute for the GNN training graph.

## Current Project State

**Phase 0 — Foundation initialized.** The repository contains the project specification, configuration, dependency plan, and dataset-forensics entry point. No model result is claimed yet.

## License

MIT
