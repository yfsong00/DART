# Mitigating Structural Overfitting: A Distribution-Aware Rectification Framework for Missing Feature Imputation

Implementation for paper: Mitigating Structural Overfitting: A Distribution-Aware Rectification Framework for Missing Feature Imputation.

## Dependencies
```bash
torch==2.1.2
torch_geometric==2.5.3
torch-scatter==2.1.2+pt21cu18
torch-sparse==0.6.18+pt21cu118
torch-sparse
dgl==2.2.1+cu118
ogb==1.3.6
pyyaml==6.0.1
```

## Run
Follow the commend to run the scripts:
```bash
sh scripts/node_classification_inductive.sh <dataset_name> <gpu_id>
# example: sh scripts/run_transductive.sh cora 0
sh scripts/node_classification_inductive.sh <dataset_name> <gpu_id>
# example: sh scripts/run_inductive.sh flickr 0
sh scripts/link_prediction.sh <dataset_name> <gpu_id>
# example: sh scripts/link_prediction.sh cora 0
```
