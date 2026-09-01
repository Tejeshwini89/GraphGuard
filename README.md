# GraphGuard

GraphGuard is a graph-based financial-forensics system for detecting illicit Bitcoin transactions. The project is being developed as a rigorous ML engineering study: establish a strong tabular baseline, test graph-aware learning under chronological evaluation, investigate why graph models generalize differently, and only then select the architecture for the production-style investigation layer.

## Research Question

> Does graph-aware learning improve illicit transaction detection compared with a strong tabular XGBoost baseline?

The project deliberately allows the answer to be **no**. A graph dataset does not automatically imply that a GNN will be the best predictive model.

## Primary Dataset

GraphGuard uses the Elliptic Bitcoin transaction dataset:

- 203,769 transaction nodes
- 234,355 directed payment-flow edges
- 166 raw columns
- 165 model features after excluding `txId` and `time_step`
- 49 time steps
- Raw labels: `1` = illicit, `2` = licit, `unknown` = unlabeled
- Normalized labels: `1` = illicit, `0` = licit, `-1` = unknown

A critical loader issue was identified during forensic validation: the processed PyTorch Geometric label encoding must not be assumed to match the raw Elliptic class semantics. GraphGuard therefore normalizes labels from the raw class information and explicitly validates the resulting counts.

## Experimental Protocol

The main evaluation is chronological:

- **Train:** time steps 1–29
- **Validation:** time steps 30–34
- **Test:** time steps 35–49

Unknown labels are excluded from supervised loss and evaluation. For the current GNN experiments, future-period graph structure and features may participate in message passing, but future labels are never used for optimization, checkpoint selection, or threshold selection. This is documented as a **transductive temporal** protocol.

Thresholds are selected on validation data only. The test window is reserved for final reporting.

## Architecture

```text
                    Elliptic Transactions
                            |
                    Data Validation
                            |
                 Temporal Split / Audit
                            |
             +--------------+--------------+
             |                             |
       Tabular Features              Graph Structure
             |                             |
          XGBoost                  PyTorch Geometric
             |                    /                  \
             |              GraphSAGE                GAT*
             |                    |                    |
             +------------ Hybrid --------------------+
                          |
                    Model Comparison
                          |
                   Fraud Probability
                          |
                 Threshold / Errors
                          |
                Investigation Layer
                          |
             Neo4j + FastAPI + Docker*

* planned / controlled next-stage work
```

## Current Benchmark

The first complete model comparison produced the following forward-test results:

| Model | Validation PR-AUC | Test PR-AUC | Test ROC-AUC | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost, all 165 features | 0.9826 | **0.7909** | 0.9240 | 0.6785 | 0.7405 | 0.7082 |
| GraphSAGE | 0.7846 | 0.4193 | 0.8482 | 0.3763 | 0.5282 | 0.4395 |
| Feature + GraphSAGE hybrid | 0.8174 | 0.4333 | 0.8690 | 0.3769 | 0.5826 | 0.4577 |

### Interpretation

The current evidence does **not** support replacing XGBoost with the tested GraphSAGE models. The hybrid improves GraphSAGE slightly, but both GNN variants experience a large temporal generalization gap relative to the tabular baseline.

This is treated as a research result rather than a project failure. The next experiments investigate whether engineered neighborhood features already capture much of the useful graph information and whether learned aggregation is sensitive to temporal graph shift.

## Graph Forensics

The transaction graph is sparse:

- Mean total degree: **2.30**
- Maximum total degree: **473**
- **100%** of observed edges connect transactions within the same time step
- **36,624** directed edges have both endpoints labeled

Among labeled endpoints, **95.37%** of observed edges connect nodes with the same class globally. However, illicit-neighbor purity is much less stable:

- Train: **52.92%**
- Validation: **71.43%**
- Test: **37.15%**

That instability is important: the graph contains class structure, but the local illicit-neighborhood pattern does not remain stable across the forward test period.

## Feature Ablation

XGBoost feature-group experiments show that local transaction features carry most of the predictive signal:

| Feature set | Features | Validation PR-AUC | Test PR-AUC |
|---|---:|---:|---:|
| Local transaction | 93 | 0.9889 | 0.7710 |
| One-hop aggregate | 72 | 0.8178 | 0.6201 |
| All features | 165 | 0.9826 | **0.7909** |

The one-hop aggregate features are independently useful, while combining them with local features produces the strongest completed test result. This suggests that engineered neighborhood information is already predictive and may overlap with what simple message-passing layers learn.

The weak GNN result therefore **does not prove that graph information is useless**. It shows that the current learned aggregation approach is not transferring well over time on this dataset.

## Completed Work

- Dataset semantics and label-encoding forensic audit
- Deterministic dataset report and per-timestep label auditing
- Chronological train/validation/test split utilities
- Leakage-aware XGBoost baseline
- GraphSAGE with train-only feature scaling, class weighting, early stopping, and validation-only threshold selection
- Feature + GraphSAGE hybrid model
- Graph diagnostics and graph-signal analysis
- Local-vs-one-hop-vs-full feature ablation
- Post-hoc feature-importance tooling
- Unit tests and pinned dependencies
- Investigation log documenting model evidence and hypotheses

## Next Work

1. Run post-hoc feature importance on the untouched test set for interpretation only.
2. Perform temporal error analysis to identify periods and transaction patterns where performance degrades.
3. Directly test redundancy between engineered one-hop features and learned neighborhood aggregation.
4. Introduce GAT only as a controlled hypothesis test: attention may help down-weight misleading neighbors that hurt GraphSAGE under temporal shift.
5. Select the final modeling direction from evidence rather than from architecture preference.
6. Build the Neo4j investigation layer around the selected model and graph relationships.
7. Add FastAPI inference/investigation endpoints, Docker packaging, explainability, CI, and reproducible evaluation artifacts.

## Modeling Rules

- PR-AUC is the primary fraud metric because the labeled classes are imbalanced.
- Random temporal mixing is not acceptable for the main experiment.
- Unknown labels are never silently converted into legitimate transactions.
- Test labels are never used for model selection or threshold tuning.
- A GNN is not considered superior unless measured forward-test performance supports that conclusion.
- Neo4j is an investigation layer, not a substitute for the GNN training graph.
- The downloaded dataset remains outside GitHub; code, configuration, tests, reports, and documentation are versioned.

## License

MIT
