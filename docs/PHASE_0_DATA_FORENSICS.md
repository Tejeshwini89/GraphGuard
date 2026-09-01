# Phase 0 — Dataset Forensics

## Objective

Establish exactly what is in the Elliptic Bitcoin dataset before selecting model inputs, graph construction rules, or training boundaries.

## Dataset Facts We Treat as Sanity Checks

PyTorch Geometric documents 203,769 transaction nodes, 234,355 directed payment-flow edges, and 165 model features. The original dataset has 166 columns per transaction when the `time_step` metadata column is included: 94 local features and 72 aggregated features. The remaining transactions are unknown rather than licit or illicit. The temporal structure covers time steps 1 through 49. citeturn350447search1turn812025search0

The original Elliptic release describes the labels as licit and illicit transaction activity and positions the data for financial-crime detection research. citeturn459467search0

## Questions We Must Answer

1. How many transactions are represented?
2. How many directed transaction-flow edges are present?
3. How many model features are available after excluding `txId` and `time_step` metadata?
4. What are the exact raw label counts?
5. How many labels are unknown?
6. How are transactions distributed across time steps 1–49?
7. Which columns are identifiers or temporal metadata?
8. Which features are legitimately available at prediction time?
9. Do any engineered aggregate features use future/neighbor information that changes the interpretation of a transductive experiment?
10. What chronological train/validation/test boundaries are appropriate?

## Ground Rules

- Unknown labels are excluded from supervised loss and primary evaluation.
- No random split is used as the primary benchmark.
- `time_step` is treated as temporal metadata for splitting, not as a predictive feature in the main 165-feature representation.
- `txId` is an identifier and is never used as a model feature.
- Feature semantics and potential leakage must be documented before modeling.
- Every model uses the same main evaluation protocol.
- Any graph transformation must preserve the source transaction-flow semantics.

## Temporal Evaluation Design

We preserve the commonly used chronological test boundary at time step 35 while carving out a validation window from the earlier period. Our initial configuration is:

```text
Train      : timesteps 1–29
Validation : timesteps 30–34
Test       : timesteps 35–49
```

This is a project design decision, not a claim that the source dataset mandates these exact three partitions. It keeps the later 35–49 period fully held out while giving model selection a validation window that does not contain future test activity. PyTorch Geometric's source implementation documents the classic 1–34 training / 35–49 test timestamp split. citeturn350447search1

## Feature Leakage Concern

The dataset contains 94 local features plus 72 one-hop aggregated features. Published descriptions note that the aggregated features are derived from neighboring transaction information. citeturn486588search0turn486588search6

This creates an important research-design question: graph message passing can itself aggregate neighbor information. We therefore preserve the 165-feature full representation as the first baseline-compatible configuration, but we will explicitly document a second controlled experiment using only local transaction features if needed. That comparison will help determine whether GNN gains come from learned topology or from information already encoded in handcrafted aggregated features.

## First Command

```powershell
python scripts\inspect_elliptic.py
```

The command loads the dataset through PyTorch Geometric, reads the raw feature/class files for the original time-step and label metadata, prints the graph dimensions and class distribution, prints counts for every time step, and writes a machine-readable report to `artifacts/forensics/dataset_report.json`.

## Output to Preserve

Save the terminal output and JSON report as the initial Phase 0 evidence. Do not commit the downloaded dataset itself unless licensing and repository-size constraints explicitly permit it. The public dataset is distributed under a Creative Commons Attribution-NonCommercial-NoDerivatives license on Kaggle. citeturn486588search1

## Next Gate

We do not begin XGBoost or GNN training until the dataset audit has been run locally, the actual time-step distribution has been checked, and the temporal split is frozen in code/configuration.
