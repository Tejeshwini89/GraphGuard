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
| GraphSAGE | 0.7846 | 0.4193 | 0.8482 | 0.3763 | 0.5282 | 0.4395 |
| Feature + GraphSAGE hybrid | 0.8174 | 0.4333 | 0.8690 | 0.3769 | 0.5826 | 0.4577 |

### Interpretation

XGBoost is currently the strongest completed model on the forward test window. GraphSAGE underperforms substantially, while the feature + GraphSAGE hybrid improves GraphSAGE only slightly and remains far behind the tabular baseline.

The important conclusion is not that graph learning is useless. The evidence says that **this simple learned neighborhood aggregation does not transfer robustly across the observed temporal shift**.

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

This is a redundancy hypothesis, not a conclusion. It needs a controlled experiment before being treated as established fact.

## Graph forensics

The graph contains 203,769 nodes and 234,355 directed edges. Its mean total degree is 2.30 and its maximum total degree is 473. Every observed edge connects transactions within the same time step, and 36,624 edges have both endpoints labeled.

Among labeled endpoints, 95.37% of observed edges connect nodes with the same class globally. However, illicit-neighbor purity changes substantially over time:

| Window | Illicit-neighbor same-label rate |
|---|---:|
| Train | 52.92% |
| Validation | 71.43% |
| Test | 37.15% |

The global same-label rate therefore cannot be used by itself to claim that neighborhood aggregation will generalize. The test-period drop in illicit-neighbor purity is particularly relevant because GraphSAGE relies on local aggregation.

## Why the GNN result is informative

Three observations now fit together:

1. The graph has measurable class structure.
2. Engineered one-hop aggregate features are predictive.
3. Learned GraphSAGE aggregation performs poorly on the forward test period.

A reasonable interpretation is that the graph contains useful information, but the **form and stability of that information matter**. The model may also be learning information that is redundant with the anonymized one-hop aggregate features already supplied by the dataset.

The current results therefore justify investigation rather than immediate architecture expansion.

## Feature importance

A post-hoc feature-importance script is implemented for the full XGBoost model. It is deliberately diagnostic: test labels may be used to measure importance after the model and threshold are frozen, but they must not influence model selection or tuning.

The intended next execution will rank features by permutation-based change in predictive behavior on the untouched test set.

## Next experiments

### 1. Temporal error analysis

Break the forward test period into individual time steps and inspect where XGBoost, GraphSAGE, and the hybrid degrade. The goal is to distinguish general temporal drift from specific periods where graph structure becomes less informative.

### 2. Graph-feature redundancy

Compare models using local features, aggregate features, graph-only information, and controlled combinations. The objective is to determine whether learned message passing contributes signal beyond the engineered neighborhood features.

### 3. GAT hypothesis test

Only after the above analysis, evaluate GAT under the same split and training protocol. The specific hypothesis is:

> **Attention-based aggregation may improve robustness by down-weighting misleading neighbors when illicit-neighbor purity shifts across time.**

A GAT result will be considered meaningful only if it improves the same forward test metric under the same leakage constraints.

## Decision rule

The final architecture will be selected from measured evidence. If XGBoost remains strongest, GraphGuard will use the GNN experiments as a documented negative/diagnostic result and focus the graph stack on investigation and explainability rather than pretending the GNN is superior.
