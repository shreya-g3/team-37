All 4 models are now trained and evaluated on 44 protein targets:

Lasso V2 (baseline): Train Pearson = 0.413, Valid Pearson = 0.001
ElasticNet V2: Train Pearson = 0.417, Valid Pearson = 0.002 — best classical model
MLP (30 epochs): Train Pearson = 0.192, Valid Pearson = 0.001
GNN (SpatialGraphSAGE): Train Pearson = 0.177, Valid Pearson = 0.001

The GNN is currently overfitting - strong training signal but validation doesn't generalize. actively working on fixes: adding dropout, weight decay, early stopping, and better train/valid graph isolation. Updated presentation slides are ready with full RMSE/MAE metrics for all models.
Next steps: tune GNN regularization, extend training with LR scheduling, and explore ElasticNet + GNN ensemble.
