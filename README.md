# GraphGuard

GraphGuard is a graph-based financial-forensics system for detecting illicit Bitcoin transactions. It combines a rigorously evaluated XGBoost risk engine with a Neo4j transaction graph and FastAPI investigation layer. The project was deliberately developed as an evidence-driven ML study: establish a strong tabular baseline, test graph-aware learning under chronological evaluation, investigate generalization, and only then select the architecture for the operational layer.

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

Unknown labels are excluded from supervised loss and evaluation. GNN experiments use a **transductive temporal** protocol: future-period graph structure and features may participate in message passing, but future labels are never used for optimization, checkpoint selection, or threshold selection.

Thresholds are selected on validation data only. The test window is reserved for final reporting and post-hoc diagnostics.

## Architecture Decision

The completed benchmark selects **XGBoost as the primary illicit-transaction risk engine**. GraphSAGE, a feature + GraphSAGE hybrid, and GAT were evaluated under the same forward temporal protocol and did not match the tabular baseline on the test window.

The production-style architecture therefore separates **prediction** from **investigation**:

```text
                    Elliptic Transactions
                            |
                    Data Validation
                            |
                 Temporal Split / Audit
                            |
             +--------------+--------------+
             |                             |
       165 Model Features              Graph Structure
             |                             |
          XGBoost                  PyTorch Geometric
        Risk Engine                Research Track
             |                    /        |        \
             |              GraphSAGE    Hybrid      GAT
             |                    \        |       /
             +----------------------+-------+------+
                            Model Evidence
                                  |
                         Selected Risk Engine
                              XGBoost
                                  |
                    +-------------+-------------+
                    |                           |
                 FastAPI                     Neo4j
              Risk + Explain             Relationships
                    |                           |
                    +-------------+-------------+
                                  |
                       Investigator Workflow
              case → risk → explanation → neighbors
```

This is intentional: the GNN experiments remain valuable as a documented model investigation, while Neo4j provides graph-native context for analysts instead of being forced into the predictive role.

## Final Model Benchmark

| Rank | Model | Validation PR-AUC | Test PR-AUC | Test ROC-AUC | Test Precision | Test Recall | Test F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **XGBoost** | 0.9826 | **0.7909** | 0.9240 | 0.6785 | 0.7405 | 0.7082 |
| 2 | Feature + GraphSAGE hybrid | 0.8174 | 0.4333 | 0.8690 | 0.3769 | 0.5826 | 0.4577 |
| 3 | GraphSAGE | 0.7846 | 0.4193 | 0.8482 | 0.3763 | 0.5282 | 0.4395 |
| 4 | GAT | 0.8466 | 0.3624 | 0.8584 | 0.5095 | 0.3721 | 0.4301 |

**Primary metric:** PR-AUC, because the labeled classes are imbalanced.

### What the benchmark tells us

- XGBoost is substantially stronger on the forward test window than all tested GNN variants.
- GAT achieved the strongest GNN validation PR-AUC (**0.8466**) but fell to **0.3624** on the forward test window, showing that validation strength did not translate into temporal robustness.
- The feature + GraphSAGE hybrid slightly improves GraphSAGE but remains far behind XGBoost.
- The correct engineering decision is therefore to deploy the strongest measured detector and use graph technology for investigation rather than architecture theater.

## Temporal Generalization Finding

The aggregate XGBoost test PR-AUC of **0.7909** hides a major temporal shift.

Performance is strong through much of test time steps 35–42, for example:

- t35: PR-AUC **0.9900**
- t38: **0.9520**
- t41: **0.9666**
- t42: **0.8854**

It then collapses from t43 onward:

- t43: **0.0332**
- t44: **0.0279**
- t45: **0.0068**
- t47: **0.0401**
- t48: **0.1874**
- t49: **0.2059**

This demonstrates that a single aggregate test score is insufficient for a fraud detector expected to operate over changing transaction behavior. GraphGuard therefore retains temporal error analysis as a first-class diagnostic rather than treating the overall benchmark as the whole story.

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

The graph therefore contains class structure, but the local illicit-neighborhood pattern does not remain stable over the forward test period.

## Feature Ablation

XGBoost feature-group experiments show that local transaction features carry most of the predictive signal:

| Feature set | Features | Validation PR-AUC | Test PR-AUC |
|---|---:|---:|---:|
| Local transaction | 93 | 0.9889 | 0.7710 |
| One-hop aggregate | 72 | 0.8178 | 0.6201 |
| All features | 165 | 0.9826 | **0.7909** |

The one-hop aggregate features are independently useful, while combining them with local features produces the strongest completed test result. This supports the hypothesis that engineered neighborhood information may overlap with information learned by simple message-passing layers, although the experiments do not prove complete redundancy.

## Explainability

GraphGuard exposes local XGBoost feature contributions through `POST /explain`. For investigation, the feature store preserves the original 165-feature vector for every transaction, allowing an analyst to request an explanation by transaction ID without manually supplying features.

The explanation uses XGBoost's additive prediction contributions in **margin/log-odds space**, not probability space. Each returned feature has a signed contribution:

- positive → increases the model's illicit-risk margin
- negative → decreases the model's illicit-risk margin

The API returns the risk score separately and can return the top contributing features for an investigator. Feature indices remain anonymized because the Elliptic dataset does not provide semantic names for the model features.

## Investigation Layer

Neo4j stores the transaction graph with:

- `Transaction` nodes
- `tx_id` uniqueness constraint
- `time_step`
- XGBoost `risk_score`
- `PAYS_TO` relationships
- risk-score index

The API supports:

- `GET /health` — service/model/Neo4j/feature-store health
- `GET /model` — selected model and evaluation protocol metadata
- `POST /predict` — 165-feature illicit-risk prediction
- `POST /explain` — local model explanation for supplied features
- `GET /transactions/{tx_id}` — transaction risk lookup
- `GET /transactions/{tx_id}/neighbors` — highest-risk connected neighbors
- `GET /transactions/{tx_id}/explain` — transaction-ID-driven explanation using the feature store
- `GET /cases/{tx_id}` — investigator-ready case combining risk, explanation, and graph context

The intended analyst workflow is:

```text
Transaction ID
     ↓
Case view
     ↓
Risk score + decision
     ↓
Why was it flagged?  →  top feature contributions
     ↓
What is connected?    →  highest-risk neighbors
     ↓
Investigate the surrounding transaction cluster
```

The case endpoint is intentionally a thin orchestration layer: XGBoost remains responsible for risk scoring, while Neo4j supplies relationship context and the feature store supplies the exact model input needed for local explanation.

## Completed Work

- Dataset semantics and label-encoding forensic audit
- Deterministic dataset report and per-timestep label auditing
- Chronological train/validation/test split utilities
- Leakage-aware XGBoost baseline
- GraphSAGE with train-only feature scaling, class weighting, early stopping, and validation-only threshold selection
- Feature + GraphSAGE hybrid model
- GAT benchmark under the same controlled protocol
- Graph diagnostics and graph-signal analysis
- Local-vs-one-hop-vs-full feature ablation
- Post-hoc feature-importance analysis
- Temporal error analysis
- Evidence-based final architecture selection
- Neo4j persistence/query layer
- FastAPI inference and investigation endpoints
- Local XGBoost explainability endpoint
- Persistent transaction feature store for ID-based explanations
- Investigator case aggregation endpoint
- Unit tests and pinned dependencies
- Docker/Compose packaging
- Investigation log documenting model evidence and hypotheses

## Reproducibility

The downloaded dataset is intentionally kept outside GitHub. Code, configuration, tests, and documentation are versioned. Model checkpoints and generated reports are produced locally by the documented scripts.

Typical validation commands:

```powershell
python -m pytest -q
python scripts\train_xgboost.py
python scripts\train_graphsage.py
python scripts\train_hybrid.py
python scripts\train_gat.py
python scripts\model_comparison_report.py
python scripts\feature_importance_report.py
python scripts\temporal_error_analysis.py
python scripts\export_feature_store.py
python scripts\build_neo4j.py
```

For the operational demo:

```powershell
docker compose up -d --build
python scripts\smoke_test_api.py
Invoke-RestMethod "http://127.0.0.1:8000/cases/85087377?top_k=10&neighbor_limit=10" | ConvertTo-Json -Depth 8
```

## Modeling Rules

- PR-AUC is the primary fraud metric because the labeled classes are imbalanced.
- Random temporal mixing is not acceptable for the main experiment.
- Unknown labels are never silently converted into legitimate transactions.
- Test labels are never used for model selection or threshold tuning.
- A GNN is not considered superior unless measured forward-test performance supports that conclusion.
- Neo4j is an investigation layer, not a substitute for the GNN training graph.
- Explainability outputs must not be presented as causal explanations; they are local model contributions.
- The downloaded dataset remains outside GitHub.

## License

MIT
