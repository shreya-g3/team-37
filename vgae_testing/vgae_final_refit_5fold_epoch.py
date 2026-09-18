import pickle
import numpy as np
import pandas as pd
import torch
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from dual_graph_vgae import (
    SEED, N_SVD_COMPONENTS, MARKER_GENE_ALIASES,
    extract_marker_genes, build_node_features,
    build_spatial_knn_graph, build_expression_knn_graph_faiss,
    train_model, save_model, predict,
)
from preprocessing_final import inverse_transform_protein

N_EPOCHS_MEDIAN = 77  # median of [77, 148, 2, 197, 28] across the 5 folds, found from running separate dual_graph_vgae.py for cv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED)
torch.manual_seed(SEED)
print(f"Device: {device}  |  final refit epochs (median across 5 folds) = {N_EPOCHS_MEDIAN}")

print("\n--- Load ---")
rna_train = ad.read_h5ad("train_rna.h5ad")
rna_val = ad.read_h5ad("test_rna.h5ad")
pro_train = ad.read_h5ad("outputs/protein_train_processed.h5ad")  # already z-scored
with open("outputs/protein_stats.pkl", "rb") as f:
    protein_stats = pickle.load(f)

barcodes_val = rna_val.obs.index.values
coords_train = rna_train.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values.astype(np.float32)
coords_val = rna_val.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values.astype(np.float32)
protein_names = list(pro_train.var_names)
Y_train = np.asarray(
    pro_train.X.todense() if hasattr(pro_train.X, "todense") else pro_train.X
).astype(np.float32)
print(f"  Train RNA: {rna_train.shape}  Val RNA: {rna_val.shape}  Proteins: {len(protein_names)}")

print("\n--- Normalize + SVD ---")
sc.pp.normalize_total(rna_train, target_sum=1e4)
sc.pp.log1p(rna_train)
sc.pp.normalize_total(rna_val, target_sum=1e4)
sc.pp.log1p(rna_val)

X_train_sp = rna_train.X if sp.issparse(rna_train.X) else sp.csr_matrix(rna_train.X)
X_val_sp = rna_val.X if sp.issparse(rna_val.X) else sp.csr_matrix(rna_val.X)
svd = TruncatedSVD(n_components=N_SVD_COMPONENTS, random_state=SEED)
X_train_svd = svd.fit_transform(X_train_sp).astype(np.float32)
X_val_svd = svd.transform(X_val_sp).astype(np.float32)
print(f"  SVD explained variance: {svd.explained_variance_ratio_.sum():.3f}")

print("\n--- Marker genes ---")
X_m_train, matched = extract_marker_genes(rna_train, protein_names, MARKER_GENE_ALIASES)
X_m_val, _ = extract_marker_genes(rna_val, protein_names, MARKER_GENE_ALIASES)
X_train, mscaler = build_node_features(X_train_svd, X_m_train)
X_val, _ = build_node_features(X_val_svd, X_m_val, scaler=mscaler)
print(f"  in_dim = {X_train.shape[1]}")

print("\n--- Graphs (full train + full val) ---")
sp_ei_tr, sp_ew_tr = build_spatial_knn_graph(coords_train, k=6)
sp_ei_val, sp_ew_val = build_spatial_knn_graph(coords_val, k=6)
ex_ei_tr, ex_ew_tr = build_expression_knn_graph_faiss(X_train, k=6)
ex_ei_val, ex_ew_val = build_expression_knn_graph_faiss(X_val, k=6)

model_cfg = dict(sage_hidden=256, n_sage_layers=2, dropout=0.3,
                  gat_proj_dim=64, gat_heads=2, gat_out_per_head=16)
opt_cfg = dict(lr=3e-4, weight_decay=1e-3, warmup=10)

print(f"\n--- Final refit (n_epochs={N_EPOCHS_MEDIAN}, median across 5 folds) ---")
model, history = train_model(
    X_train, Y_train, sp_ei_tr, sp_ew_tr, ex_ei_tr, ex_ew_tr,
    N_EPOCHS_MEDIAN, model_cfg, opt_cfg, device, log_interval=10,
)
save_model(model, "outputs/gnn_v5_model_5fold.pt")
history.to_csv("outputs/history_5fold.csv", index=False)

print("\n--- Inference on test_rna.h5ad ---")
Z_pred = predict(model, X_val, sp_ei_val, sp_ew_val, ex_ei_val, ex_ew_val, device)
X_pred = inverse_transform_protein(Z_pred, protein_stats)
X_pred = np.clip(X_pred, 0, None)

submission = pd.DataFrame(X_pred, columns=protein_names)
submission.insert(0, "pxl_col_in_fullres", coords_val[:, 1])   # float, not int -- fix already applied
submission.insert(0, "pxl_row_in_fullres", coords_val[:, 0])
submission.insert(0, "barcode", barcodes_val)

out_csv = "outputs/pred_test_5fold_median.csv"
submission.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}  shape={submission.shape}")
print(f"any NaN: {submission.isna().any().any()}")