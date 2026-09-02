# GraphGuard Architecture Decision

## Decision

Use **XGBoost as the primary illicit-transaction risk engine** and use **Neo4j as the graph-native investigation layer**. Expose both through FastAPI, with local XGBoost contribution explanations available to investigators.

## Evidence

All candidate models were evaluated with the same chronological forward-test design:

- Train: time steps 1–29
- Validation: time steps 30–34
- Test: time steps 35–49
- Primary metric: PR-AUC
- Classification threshold selected on validation only

| Model | Test PR-AUC | Test ROC-AUC | Test F1 |
|---|---:|---:|---:|
| **XGBoost** | **0.7909** | **0.9240** | **0.7082** |
| Feature + GraphSAGE | 0.4333 | 0.8690 | 0.4577 |
| GraphSAGE | 0.4193 | 0.8482 | 0.4395 |
| GAT | 0.3624 | 0.8584 | 0.4301 |

XGBoost therefore wins the measured forward-test comparison by a substantial margin.

## Why the graph is still central

The architecture does not discard graph information. Graph forensics found measurable class structure, but illicit-neighbor purity fell from 52.92% in train to 37.15% in test. In parallel, the 72 engineered one-hop aggregate features achieved a test PR-AUC of 0.6201 on their own.

These findings suggest that graph-derived information is useful but that learned neighborhood aggregation is not temporally robust on this dataset. The product therefore uses the graph where it provides direct investigator value: relationship discovery, suspicious-neighbor ranking, and transaction context.

## Product workflow

```text
Transaction features
        |
        v
     XGBoost
        |
   Risk score
        |
   +----+----------------+
   |                     |
   v                     v
/explain              Neo4j
   |                     |
Feature-level       Relationships
contributions       + neighbors
   |                     |
   +----------+----------+
              |
              v
       Investigator
```

## Explainability contract

`POST /explain` returns the model's local additive feature contributions in XGBoost margin/log-odds space. Positive contributions increase the illicit-risk margin; negative contributions decrease it. The API reports the probability separately.

These are **model explanations, not causal explanations**. Elliptic feature semantics are anonymized, so the system reports feature indices rather than inventing human-readable meanings.

## Rejected alternatives

### GraphSAGE as primary detector

Rejected because its forward-test PR-AUC was 0.4193, substantially below XGBoost.

### Feature + GraphSAGE hybrid

Rejected as the primary detector because the hybrid reached 0.4333 test PR-AUC, only a small improvement over GraphSAGE and still far below XGBoost.

### GAT as primary detector

Rejected because its forward-test PR-AUC was 0.3624 despite a stronger validation PR-AUC of 0.8466. This validation-to-test gap is evidence against assuming attention-based aggregation solves the observed temporal shift.

## Consequences

### Positive

- Strongest measured forward-test detector is deployed.
- Graph technology remains a meaningful part of the product rather than decorative architecture.
- Model selection is defensible in interviews because alternatives were tested rather than assumed.
- Temporal drift becomes an explicit operational limitation and monitoring concern.

### Trade-off

The final detector is not a learned GNN. GraphGuard is therefore best described as a **graph-based financial-forensics platform with an evidence-selected XGBoost risk engine**, rather than as a GNN fraud detector.

## Status

**Accepted — September 2026**
