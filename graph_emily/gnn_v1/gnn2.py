"""
GraphSAGE GNN:
predict CODEX protein expression (44 markers) from Visium HD RNA expression

preprocessing is imported
    preprocessing_final.py: inverse_transform_protein - for inverse tranform of protein predictions
truncated SVD is fit on rna_train_preprocessed and then applied to train and val

Epoch count: NOT re-derived from a single CV fold anymore. gnn_svd_cv.py already
ran the full 5-fold CV and found epochs=[33, 38, 86, 9, 34] (median=34) - that's a
more robust estimate than re-running find_n_epochs() on fold 0 alone (33, close to
the median by luck this time, but the fold-to-fold spread of 9-86 shows that isn't
reliable in general). N_EPOCHS below is hardcoded to that median and the model
trains directly on 100% of the data for that many epochs - no epoch-finder pass,
no cv_split_path dependency for this script anymore.

Inputs:
    rna_train_path: "rna_train_preprocessed.h5ad"
    rna_val_path: "rna_val_preprocessed.h5ad"
    pro_train_path: "pro_train_preprocessed.h5ad"

Output (saved into out_path):
    pred_val.csv: barcode, pxl_row_in_fullres, pxl_col_in_fullres, <44 markers>
    graphsage_model.pt: saved model weights + config, reloadable via load_model()
"""

import os
import numpy as np
import pandas as pd
import anndata as ad

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from scipy.spatial import cKDTree
from sklearn.decomposition import TruncatedSVD
import pickle

# modules
from preprocessing_final import inverse_transform_protein

# config
N_SVD_COMPONENTS = 128   # RNA dimensionality reduction, fit on train only
K_NEIGHBORS = 6          # kNN graph degree
HIDDEN_DIM = 256
N_LAYERS = 3
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-5
N_EPOCHS = 34             # median from gnn_svd_cv.py's 5-fold run: [33, 38, 86, 9, 34]
SEED = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED)
torch.manual_seed(SEED)


def to_dense(X):
    return np.asarray(X.todense() if hasattr(X, "todense") else X)


# RNA: SVD
def reduce_rna_svd(rna_train, rna_val, n_components=N_SVD_COMPONENTS):
    """
    rna_train & rna_val are preprocessed
    SVD is fit on train only then applied to both

    Kept SPARSE going into TruncatedSVD - sklearn supports sparse input directly,
    and densifying the full RNA matrix first (the previous version of this
    function did `to_dense(rna_train.X)` before SVD) can easily need 10-20+ GB
    for a full, un-filtered gene matrix at this many bins - more RAM than a
    g4dn.xlarge's 16GB has, which is what was killing the run.
    """
    import scipy.sparse as sp
    X_train = rna_train.X if sp.issparse(rna_train.X) else sp.csr_matrix(rna_train.X)
    X_val = rna_val.X if sp.issparse(rna_val.X) else sp.csr_matrix(rna_val.X)

    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    X_train_svd = svd.fit_transform(X_train).astype(np.float32)
    X_val_svd = svd.transform(X_val).astype(np.float32)
    return X_train_svd, X_val_svd, svd


# Graph construction
def build_knn_graph(coords, k=K_NEIGHBORS):
    """
    Symmetric kNN graph from spatial pixel coordinates via cKDTree
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


# Model
class ProteinGraphSAGE(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, dropout=DROPOUT):
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
        # stash config to rebuild identical architecture on load
        self._config = dict(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim,
                             n_layers=n_layers, dropout=dropout)

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)


# Train on 100% of the data for N_EPOCHS
def train_model(X, Y, edge_index, n_epochs=N_EPOCHS, verbose=True):
    """
    trains on all training data for specified epochs
    model is returned as pytorch object - can be saved and loaded again
    """
    x_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(Y, dtype=torch.float32, device=DEVICE)
    edge_index = edge_index.to(DEVICE)

    model = ProteinGraphSAGE(in_dim=X.shape[1], out_dim=Y.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = model(x_t, edge_index)
        loss = loss_fn(out, y_t)
        loss.backward()
        optimizer.step()

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"epoch {epoch:4d}  train_mse {loss.item():.4f}")

    return model


# Save & load model
def save_model(model, model_path):
    torch.save({"state_dict": model.state_dict(), "config": model._config}, model_path)
    print(f"saved model to {model_path}")


def load_model(model_path, device=DEVICE):
    """
    Rebuilds architecture from saved configurations
    loads weights and returns model
    """
    ckpt = torch.load(model_path, map_location=device)
    model = ProteinGraphSAGE(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# Prediction
def predict(model, X_val, edge_index_val):
    model.eval()
    x_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    edge_index_val = edge_index_val.to(DEVICE)
    with torch.no_grad():
        out = model(x_t, edge_index_val)
    return out.cpu().numpy()


# Main
def run_gnn(rna_train_path, pro_train_path, rna_val_path, pro_stats_path, out_path, n_epochs=N_EPOCHS):
    """
    Loads preprocessed data
    performs SVD, graph construction, trains on 100% of data for n_epochs
    predicts protein expression from rna_val and inverse transforms back to CODEX values
    """
    os.makedirs(out_path, exist_ok=True)  # out_path is a directory, not a file - create it up front

    rna_train = ad.read_h5ad(rna_train_path)
    rna_val = ad.read_h5ad(rna_val_path)
    pro_train = ad.read_h5ad(pro_train_path)

    with open(pro_stats_path, "rb") as f:
        protein_stats = pickle.load(f)

    barcodes_val = rna_val.obs.index.values
    coords_train = rna_train.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values.astype(np.float32)
    coords_val = rna_val.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values.astype(np.float32)
    marker_names = list(pro_train.var_names)

    X_train, X_val, svd = reduce_rna_svd(rna_train, rna_val)
    Y_train = to_dense(pro_train.X).astype(np.float32)  # already z-scored by preprocess_protein_train

    edge_index_train = build_knn_graph(coords_train)
    edge_index_val = build_knn_graph(coords_val)

    model = train_model(X_train, Y_train, edge_index_train, n_epochs=n_epochs)

    Z_pred = predict(model, X_val, edge_index_val)
    X_pred = inverse_transform_protein(Z_pred, protein_stats)
    X_pred = np.clip(X_pred, a_min=0, a_max=None)  # protein expression is non-negative

    submission = pd.DataFrame(X_pred, columns=marker_names)
    submission.insert(0, "pxl_col_in_fullres", coords_val[:, 1].astype(int))
    submission.insert(0, "pxl_row_in_fullres", coords_val[:, 0].astype(int))
    submission.insert(0, "barcode", barcodes_val)

    submission_path = os.path.join(out_path, "pred_val.csv")
    submission.to_csv(submission_path, index=False)
    print(f"saved {submission_path}  shape={submission.shape}")

    # save model - single save, with a real filename (out_path is a directory)
    model_path = os.path.join(out_path, "graphsage_model.pt")
    save_model(model, model_path=model_path)

    return model, submission