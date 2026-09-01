# Model Investigation Log

## Feature ablation

Temporal split: train 1-29, validation 30-34, test 35-49.

| Feature set | Features | Validation PR-AUC | Test PR-AUC |
|---|---:|---:|---:|
| Local transaction | 93 | 0.9889 | 0.7710 |
| One-hop aggregate | 72 | 0.8178 | 0.6201 |
| All features | 165 | 0.9826 | 0.7909 |

Interpretation: local transaction features are the dominant predictive signal. One-hop aggregate features are independently useful and complementary because the full 165-feature model slightly exceeds the local-only test result. Therefore, poor GraphSAGE performance is not evidence that graph structure is irrelevant; it indicates that simple learned neighborhood aggregation is not transferring well over time and may overlap with information already represented by engineered aggregate features.

## Graph model comparison

| Model | Validation PR-AUC | Test PR-AUC |
|---|---:|---:|
| XGBoost, all 165 features | 0.9826 | 0.7909 |
| GraphSAGE | 0.7846 | 0.4193 |
| Feature + GraphSAGE hybrid | 0.8174 | 0.4333 |

## Working conclusion

Do not select a GNN merely because the dataset is a graph. Continue with controlled experiments that isolate temporal robustness, feature redundancy, and the value of graph topology. GAT should only be introduced if it tests a specific hypothesis that the current aggregation mechanism cannot test.
