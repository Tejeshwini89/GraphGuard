# Model Investigation Log

This document records model evidence and the hypotheses that follow from it. The goal is to understand whether graph structure adds robust predictive value under forward temporal evaluation, not to force a GNN into the final architecture.

## Evaluation protocol

Temporal split:

- Train: time steps 1–29
- Validation: time steps 30–34
- Test: time steps 35–49

Unknown labels are excluded from supervised loss and evaluation. Current GNN experiments use a **transductive temporal** protocol: future graph structure/features may participate in message passing, but future labels are not used for training, checkpoint selection, or threshold selection. Classification thresholds are selected on validation data only.

## Completed model comparison

| Model | Validation PR-AUC | Test PR-AUC | Test ROC-AUC | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost, all 165 features | 0.9826 | **0.7909** | 0.9240 | 0.6785 | 0.7405 | 0.7082 |
| Feature + GraphSAGE hybrid | 0.8174 | 0.4333 | 0.8690 | 0.3769 | 0.5826 | 0.4577 |
| GraphSAGE | 0.7846 | 0.4193 | 0.8482 | 0.3763 | 0.5282 | 0.4395 |
| GAT | 0.8466 | 0.3624 | 0.8584 | 0.5095 | 0.3721 | 0.4301 |

GAT was evaluated as a controlled hypothesis test with the same temporal protocol, feature scaling, class weighting, early stopping, and validation-only threshold selection.

### Interpretation

XGBoost is the strongest model on the forward test window by a large margin. The feature + GraphSAGE hybrid improves GraphSAGE only slightly, while GAT achieves the strongest GNN validation PR-AUC but the weakest forward-test PR-AUC of the four models.

The important conclusion is not that graph learning is useless. The evidence says that **the tested learned neighborhood aggregation approaches do not transfer robustly across the observed temporal shift**.

## Feature ablation

| Feature set | Features | Validation PR-AUC | Test PR-AUC |
|---|---:|---:|---:|
| Local transaction | 93 | 0.9889 | 0.7710 |
| One-hop aggregate | 72 | 0.8178 | 0.6201 |
| All features | 165 | 0.9826 | **0.7909** |

### Interpretation

The 93 local transaction features are the dominant predictive group. The 72 one-hop aggregate features are independently useful, and combining all 165 features slightly improves the test PR-AUC over local-only features.

This creates a concrete hypothesis for the graph investigation:

> **Engineered one-hop aggregate features may already encode much of the neighborhood information that a simple GraphSAGE layer would otherwise learn.**

This is a redundancy hypothesis, not a conclusion. The completed ablation supports investigating it, but does not establish complete redundancy.

## Graph forensics

The graph contains 203,769 nodes and 234,355 directed edges. Its mean total degree is 2.30 and its maximum total degree is 473. Every observed edge connects transactions within the same time step, and 36,624 edges have both endpoints labeled.

Among labeled endpoints, 95.37% of observed edges connect nodes with the same class globally. However, illicit-neighbor purity changes substantially over time:

| Window | Illicit-neighbor same-label rate |
|---|---:|
| Train | 52.92% |
| Validation | 71.43% |
| Test | 37.15% |

The global same-label rate therefore cannot be used by itself to claim that neighborhood aggregation will generalize. The test-period drop in illicit-neighbor purity is particularly relevant because GraphSAGE and GAT rely on local aggregation.

## Why the GNN result is informative

Four observations now fit together:

1. The graph has measurable class structure.
2. Engineered one-hop aggregate features are predictive.
3. Local transaction features dominate the predictive signal.
4. GraphSAGE, hybrid GraphSAGE, and GAT all show large forward-test degradation.

A reasonable interpretation is that the graph contains useful information, but the **form and stability of that information matter**. The models may also be learning information that overlaps with the anonymized one-hop aggregate features already supplied by the dataset.

The correct engineering response is therefore not to add increasingly complex GNNs without evidence. The selected detector is XGBoost, while the graph is retained for relationship-aware investigation.

## Feature importance

A post-hoc feature-importance report was executed for the frozen full XGBoost model using permutation importance on the untouched test set. The report ranks features by decrease in average precision when a feature is permuted.

Top features by test AP decrease were:

| Rank | Feature index | Mean AP decrease |
|---:|---:|---:|
| 1 | 52 | 0.02091 |
| 2 | 89 | 0.01461 |
| 3 | 58 | 0.01061 |
| 4 | 2 | 0.00763 |
| 5 | 17 | 0.00621 |
| 6 | 124 | 0.00466 |
| 7 | 4 | 0.00310 |
| 8 | 162 | 0.00292 |
| 9 | 126 | 0.00279 |
| 10 | 160 | 0.00278 |

The top 20 contain 15 local-feature indices and 5 one-hop aggregate-feature indices. Because the Elliptic model features are anonymized, GraphGuard does not invent semantic names for these indices.

This report is diagnostic only; the test labels did not influence model selection or threshold tuning.

## Temporal error analysis

Post-hoc timestep analysis shows that the aggregate XGBoost test PR-AUC of 0.7909 hides substantial temporal drift.

The model is strong in earlier test periods, including:

- t35: PR-AUC 0.9900
- t38: PR-AUC 0.9520
- t41: PR-AUC 0.9666
- t42: PR-AUC 0.8854

Performance then collapses from t43 onward:

- t43: PR-AUC 0.0332
- t44: PR-AUC 0.0279
- t45: PR-AUC 0.0068
- t47: PR-AUC 0.0401
- t48: PR-AUC 0.1874
- t49: PR-AUC 0.2059

This degradation is not explained by prevalence alone: t49 has an illicit rate of 11.76% but PR-AUC only 0.2059, while t35 has a similar 13.57% illicit rate and PR-AUC 0.9900. The evidence is consistent with a changing feature-to-label relationship or other temporal distribution shift.

These findings are post-hoc diagnostics and are not used to retune the frozen model.

## Final architecture decision

The forward-test benchmark selects **XGBoost** as the primary risk engine. Graph technology remains central to the product, but in a role where it provides direct operational value:

- **XGBoost:** illicit transaction risk scoring
- **Neo4j:** transaction relationships and suspicious-neighbor investigation
- **FastAPI:** inference, explanation, and investigation access
- **XGBoost contributions:** local model explanation

This separation is intentional. GraphGuard demonstrates that an engineering team should choose the predictive architecture from measured forward performance while still exploiting graph structure for investigation and context.

## Next engineering work

1. Populate Neo4j from the normalized Elliptic graph and frozen XGBoost risk scores.
2. Validate transaction and suspicious-neighbor queries against the local database.
3. Exercise the FastAPI `/predict`, `/explain`, `/transactions/{tx_id}`, and `/transactions/{tx_id}/neighbors` workflow.
4. Add end-to-end Docker validation and reproducibility checks.
5. Add a polished investigator-facing demo workflow and final limitations/threat-model documentation.
