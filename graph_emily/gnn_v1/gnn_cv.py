"""
5-fold spatial CV for the GraphSAGE GNN

Inputs:
    rna_path: "data/preprocessed/rna_preprocessed.h5ad"
    protein_path: "data/preprocessed/pro_preprocessed.h5ad"
    cv_split_path: "outputs/cv_splits_patches.json"
    out_dir:"results"

Outputs (under out_dir):
    fold{N}_gnn.csv: per-fold, per-protein metrics
    gnn_svd_cv_results.csv: overall summary (one row)
    gnn_svd_cv_per_protein_metrics.csv: per-protein metrics averaged across folds
    gnn_model_fold{N}.pt: saved model weights + config per fold
    svd_model_gnn_fold{N}.pkl: saved SVD per fold (matches xgb's convention)

Returns: results_df, per_protein_results_df, gnn_params, fold_epochs
"""

import os
import gc
import pickle
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.decomposition import TruncatedSVD
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from cv_split_patches import load_cv_split

# ---------------- current tuned-ish config (mirrors DEFAULT_XGB_PARAMS pattern) ----------------
DEFAULT_GNN_PARAMS = dict(
    hidden_dim=256,
    n_layers=3,
    dropout=0.3,
    lr=1e-3,
    weight_decay=1e-5,
)

K_NEIGHBORS = 6
MAX_EPOCHS = 300
PATIENCE = 20
SEED = 0


def to_dense(X):
    return np.asarray(X.todense() if hasattr(X, "todense") else X)


def build_knn_graph(coords, k=K_NEIGHBORS):
    """
    kNN graph from spatial pixel coordinates via cKDTree
    """
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)  # first column is self
    n = coords.shape[0]
    src = np.repeat(np.arange(n), k)
    dst = idx[:, 1:].reshape(-1)
    edge_index = np.vstack([src, dst])
    edge_index = np.hstack([edge_index, edge_index[[1, 0]]])  # symmetrize
    edge_index = np.unique(edge_index, axis=1)
    return torch.tensor(edge_index, dtype=torch.long)


class ProteinGraphSAGE(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, n_layers=3, dropout=0.3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * n_layers
        self.convs = nn.ModuleList(SAGEConv(dims[i], dims[i + 1]) for i in range(n_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(dims[i + 1]) for i in range(n_layers))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self._config = dict(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim,
                             n_layers=n_layers, dropout=dropout)

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)


def save_model(model, model_path):
    torch.save({"state_dict": model.state_dict(), "config": model._config}, model_path)


def train_with_early_stopping(X, Y, edge_index, train_mask, holdout_mask, gnn_params,
                               device, max_epochs=MAX_EPOCHS, patience=PATIENCE, verbose=True):
    """
    One fold training loop, early-stops on fold's holdout set
    """
    x_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    edge_index = edge_index.to(device)

    # split combined params dict: hidden_dim/n_layers/dropout go to the model,
    # lr/weight_decay go to the optimizer - ProteinGraphSAGE doesn't accept lr/weight_decay
    model_keys = {"hidden_dim", "n_layers", "dropout"}
    model_kwargs = {k: v for k, v in gnn_params.items() if k in model_keys}
    model = ProteinGraphSAGE(in_dim=X.shape[1], out_dim=Y.shape[1], **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=gnn_params["lr"],
                                  weight_decay=gnn_params["weight_decay"])
    loss_fn = nn.MSELoss()

    best_val_loss, best_epoch, patience_ctr = float("inf"), 0, 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x_t, edge_index)
        loss = loss_fn(out[train_mask], y_t[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(x_t, edge_index)
            val_loss = loss_fn(out[holdout_mask], y_t[holdout_mask]).item()

        if val_loss < best_val_loss:
            best_val_loss, best_epoch, patience_ctr = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if verbose and (epoch % 10 == 0 or epoch == max_epochs - 1):
            print(f"    epoch {epoch:4d}  train_mse {loss.item():.4f}  holdout_mse {val_loss:.4f}")

        if patience_ctr >= patience:
            print(f"    early stop at epoch {epoch} (best epoch {best_epoch}, holdout_mse {best_val_loss:.4f})")
            break

    model.load_state_dict(best_state)  # roll back to best epoch, not the last one
    model.eval()
    with torch.no_grad():
        Y_pred_all = model(x_t, edge_index).cpu().numpy()

    return model, best_epoch + 1, Y_pred_all


def gnn_svd_cv(
        rna_path,
        protein_path,
        cv_split_path,
        out_dir,
        device=None,
        n_components=128,
        svd_random_state=0,
        gnn_params=None,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        max_folds=None,
):
    """
    Cross-validated GraphSAGE GNN with per-fold truncated SVD
    - Early stopping per fold on fold's own holdout - how many epochs to use on final run
    Returns results_df, per_protein_results_df, gnn_params, fold_epochs
    """
    os.makedirs(out_dir, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = dict(DEFAULT_GNN_PARAMS)
    if gnn_params:
        params.update(gnn_params)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 1. load preprocessed data
    print("Loading preprocessed data...")
    rna = ad.read_h5ad(rna_path)
    pro = ad.read_h5ad(protein_path)

    protein_names = list(pro.var_names)
    coords = rna.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values.astype(np.float32)

    X_raw = rna.X if sparse.issparse(rna.X) else sparse.csr_matrix(rna.X)
    X_raw = X_raw.astype(np.float32).copy()
    Y = pro.X.toarray() if sparse.issparse(pro.X) else np.asarray(pro.X).astype(np.float32)

    del rna, pro
    gc.collect()

    # 2. graph built once - edges don't depend on the fold split
    edge_index = build_knn_graph(coords)

    # 3. load cv split
    splits = load_cv_split(split_path=cv_split_path)
    n_splits = len(splits)
    if max_folds is not None:
        print(f"max_folds={max_folds}: running only {min(max_folds, n_splits)}/{n_splits} folds (smoke-test mode)")
        splits = splits[:max_folds]
        n_splits = len(splits)  # resize downstream arrays/means to match folds actually run

    fold_pearsonr = np.zeros((n_splits, Y.shape[1]))
    fold_r2 = np.zeros((n_splits, Y.shape[1]))
    fold_rmse = np.zeros((n_splits, Y.shape[1]))
    fold_svd_explained_var = np.zeros(n_splits)
    fold_epochs = []

    for split in splits:
        fold = split["fold"]
        train_idx, val_idx = np.array(split["train"]), np.array(split["test"])

        print(f"\nFold {fold + 1}/{n_splits} "
              f"({len(train_idx):,} train / {len(val_idx):,} val bins)")
 
        # 4. truncated SVD, fit on fold's train bins only, transform on all data
        svd = TruncatedSVD(n_components=n_components, random_state=svd_random_state)
        svd.fit(X_raw[train_idx])
        X_svd = svd.transform(X_raw).astype(np.float32)
        fold_svd_explained_var[fold] = svd.explained_variance_ratio_.sum()
        print(f"  SVD cumulative explained variance (train): {fold_svd_explained_var[fold]:.3f}")

        with open(f"{out_dir}/svd_model_gnn_fold{fold}.pkl", "wb") as f:
            pickle.dump(svd, f)

        # 5. masks for this fold
        n = X_svd.shape[0]
        train_mask = torch.zeros(n, dtype=torch.bool, device=device)
        train_mask[train_idx] = True
        holdout_mask = torch.zeros(n, dtype=torch.bool, device=device)
        holdout_mask[val_idx] = True

        # 6. train with early stopping on this fold's holdout
        model, n_epochs, Y_pred_all = train_with_early_stopping(
            X_svd, Y, edge_index, train_mask, holdout_mask, params, device,
            max_epochs=max_epochs, patience=patience,
        )
        fold_epochs.append(n_epochs)

        save_model(model, f"{out_dir}/gnn_model_fold{fold}.pt")

        Y_val_pred = Y_pred_all[val_idx]
        Y_val = Y[val_idx]

        # 7. per-fold metrics
        for j in range(Y.shape[1]):
            r, _ = pearsonr(Y_val[:, j], Y_val_pred[:, j])
            fold_pearsonr[fold, j] = r
            fold_r2[fold, j] = r2_score(Y_val[:, j], Y_val_pred[:, j])
            fold_rmse[fold, j] = np.sqrt(mean_squared_error(Y_val[:, j], Y_val_pred[:, j]))

        print(f"Fold {fold + 1}/{n_splits}, epochs={n_epochs}, "
              f"mean Pearson r={fold_pearsonr[fold].mean():.3f}, "
              f"mean R2={fold_r2[fold].mean():.3f}, "
              f"mean RMSE={fold_rmse[fold].mean():.3f}")

        fold_results = pd.DataFrame({
            "protein": protein_names,
            "pearsonr": fold_pearsonr[fold],
            "r2": fold_r2[fold],
            "rmse": fold_rmse[fold],
        })
        fold_results.to_csv(f"{out_dir}/fold{fold}_gnn.csv", index=False)

        del svd, X_svd, model, Y_pred_all, Y_val_pred, Y_val
        gc.collect()

    # 8. summary across folds
    mean_pearsonr_per_protein = fold_pearsonr.mean(axis=0)
    std_r_per_protein = fold_pearsonr.std(axis=0)
    mean_r2_per_protein = fold_r2.mean(axis=0)
    mean_rmse_per_protein = fold_rmse.mean(axis=0)

    print(
        f"\nOverall mean Pearson r across all proteins: "
        f"{mean_pearsonr_per_protein.mean():.3f} \u00b1 {mean_pearsonr_per_protein.std():.4f}")
    print(f"Overall mean R\u00b2 across all proteins: {mean_r2_per_protein.mean():.3f}")
    print(f"Overall mean RMSE across all proteins: {mean_rmse_per_protein.mean():.3f}")
    print(f"Mean SVD cumulative explained variance across folds: {fold_svd_explained_var.mean():.3f}")
    print(f"Epochs selected per fold: {fold_epochs} "
          f"(median={int(np.median(fold_epochs))}) - use this for a final full-data run")

    results_df = pd.DataFrame({
        "mean_pearsonr": [mean_pearsonr_per_protein.mean()],
        "mean_pearsonr_std": [mean_pearsonr_per_protein.std()],
        "mean_r2": [mean_r2_per_protein.mean()],
        "mean_rmse": [mean_rmse_per_protein.mean()],
        "mean_svd_explained_var": [fold_svd_explained_var.mean()],
        "n_svd_components": [n_components],
        "median_epochs": [int(np.median(fold_epochs))],
    })

    per_protein_results_df = pd.DataFrame({
        "protein": protein_names,
        "mean_pearsonr": mean_pearsonr_per_protein,
        "std_pearsonr": std_r_per_protein,
        "mean_r2": mean_r2_per_protein,
        "mean_rmse": mean_rmse_per_protein,
    }).sort_values("mean_pearsonr", ascending=False)

    print(f"\nTop 10 best-predicted proteins:")
    print(per_protein_results_df.head(10).to_string(index=False))

    results_df.to_csv(f"{out_dir}/gnn_svd_cv_results.csv", index=False)
    per_protein_results_df.to_csv(f"{out_dir}/gnn_svd_cv_per_protein_metrics.csv", index=False)

    print(f"\nSaved to {out_dir}")

    return results_df, per_protein_results_df, params, fold_epochs