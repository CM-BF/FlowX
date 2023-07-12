# XGNN_benchmark

This is an benchmark system for GNN explainable methods.

## GNN methods
- [x] GCN
- [x] GAT
- [x] GIN

## Explainability methods
- [x] GNNExplainer
- [] GNN_LRP
- [] GNN-3methods

## Datasets

- [x] Tox21
- [x] HIV
- [x] ClinTox

## GNN results

ROC_AUC:

| model | Tox21 | HIV | ClinTox|
| ---- | --- | --- |---|
|GCN|0.6968|0.5734|0.6151|
|GAT|0.7104|0.6290|0.6683|
|GIN|0.7879| 0.7191|0.9337|

## GNN one target results

| method | tox21(ROC_AUC)_2 | clintox(ROC_AUC)_0 | esol(RMSE lower is better)_0 |
| ------ | -------------- | ---------------- | -------------------------- |
| GCN    | 0.8266         | 0.7413           | 1.0343                     |
| GAT    | 0.8157         | 0.9103           | 1.1087                     |
| GIN    | 0.8626         | 0.8929           | 0.8512                     |

## Corresponding explainer results

|    method     | tox21  | clintox | esol |
| :-----------: | :----: | :-----: | :--: |
| GCN: Fidelity |  ->0   |   ->0   |      |
| Contrastivity | 0.9100 | 0.9490  |      |
|   Sparsity    | 0.3939 | 0.3481  |      |
| GAT: Fidelity |  ->0   |    0    |      |
| Contrastivity | 0.8976 | 0.9186  |      |
|   Sparsity    | 0.3885 | 0.5374  |      |
| GIN: Fidelity |  ->0   | -0.0403 |      |
| Contrastivity | 0.7393 | 0.8560  |      |
|   Sparsity    | 0.1705 | 0.3049  |      |

