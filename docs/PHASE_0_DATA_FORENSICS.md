# Phase 0 — Dataset Forensics

## Objective

Establish exactly what is in the Elliptic Bitcoin dataset before selecting model inputs, graph construction rules, or training boundaries.

## Dataset Facts

The raw Elliptic feature file contains 203,769 transactions and 166 columns: `txId`, `time_step`, and 165 model features. The graph contains 234,355 directed payment-flow edges. The source labels are `1` = illicit, `2` = licit, and `unknown` = unlabeled. The temporal structure covers time steps 1 through 49.

## Critical Label-Encoding Finding

PyTorch Geometric's processed representation uses an internal label encoding that must not be confused with the raw dataset semantics. The first local forensic run exposed this: `graph.y` appeared to contain 203,769 non-negative labels even though the raw class file contained 157,205 `unknown`, 42,019 licit, and 4,545 illicit transactions.

GraphGuard therefore does **not** use the processed `graph.y` values as the source of truth for label semantics. `src/graphguard/elliptic.py` reads the raw class file and normalizes labels explicitly:

```text
raw class      GraphGuard label
unknown        -1
2 (licit)       0
1 (illicit)     1
```

This prevents unknown transactions from being silently treated as supervised examples.

## Questions We Must Answer

1. How many transactions are represented?
2. How many directed transaction-flow edges are present?
3. How many model features are available after excluding `txId` and `time_step`?
4. What are the exact raw label counts?
5. How many labels are unknown?
6. How are transactions distributed across time steps 1–49?
7. How are licit and illicit transactions distributed across each time step?
8. Which columns are identifiers or temporal metadata?
9. Which features are legitimately available at prediction time?
10. Do any engineered aggregate features use future/neighbor information that changes the interpretation of a transductive experiment?
11. What chronological train/validation/test boundaries are appropriate?

## Ground Rules

- Unknown labels are excluded from supervised loss and primary evaluation.
- Unknown nodes may remain in the graph for message passing when the experiment is explicitly transductive; this will be documented and ablated if necessary.
- No random split is used as the primary benchmark.
- `time_step` is treated as temporal metadata for splitting, not as a predictive feature in the main 165-feature representation.
- `txId` is an identifier and is never used as a model feature.
- Feature semantics and potential leakage must be documented before modeling.
- Every model uses the same main evaluation protocol.
- Any graph transformation must preserve the source transaction-flow semantics.

## Temporal Evaluation Design

The initial configuration is:

```text
Train      : timesteps 1–29
Validation : timesteps 30–34
Test       : timesteps 35–49
```

This is a project design decision, not a claim that the source dataset mandates these exact three partitions. We preserve the later 35–49 period as the final holdout while using 30–34 for model selection. Before freezing the split, GraphGuard records per-timestep class counts and checks that the validation and test windows contain usable labeled examples of both classes.

## Feature Leakage Concern

The dataset contains local transaction features plus 72 aggregated features derived from neighboring transaction information. This creates an important research-design question: graph message passing can itself aggregate neighbor information. We therefore preserve the 165-feature representation as the first baseline-compatible configuration, but we will explicitly document a second controlled experiment using only local transaction features if needed. That comparison will help determine whether GNN gains come from learned topology or from handcrafted neighborhood aggregates already present in the tabular features.

## Forensics Command

```powershell
python scripts\inspect_elliptic.py
```

The command loads the dataset, normalizes raw labels, records graph dimensions, feature quality, overall class distribution, and per-timestep class distribution, then writes a machine-readable report to `artifacts/forensics/dataset_report.json`.

## Output to Preserve

Keep the local JSON report as Phase 0 evidence. Do not commit the downloaded dataset itself unless licensing and repository-size constraints explicitly permit it.

## Next Gate

We do not begin XGBoost or GNN training until the corrected dataset audit has been run locally, the actual per-timestep class distribution has been checked, and the temporal split is frozen in code/configuration.
