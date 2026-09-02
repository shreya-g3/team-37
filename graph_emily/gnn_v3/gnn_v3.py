"""
Residual GraphSAGE/GAT for RNA -> protein

ResidualGNNBlock: SAGEConv OR GATConv + LayerNorm + residual connection + dropout
conv_type="sage" (default)
conv_type="gat" swaps in GATConv - learns a per-neighbor attention weight instead of averaging all k neighbors uniformly
- GATConv's attention is scoped to the kNN graph's edges, not full attention over all nodes

input preprocessed rna_train_preprocessed, rna_val_preprocessed, and pro_train_preprocessed
apply truncated SVD
one kNN graph (not local and regional)

Inputs:
    preprocessed rna & protein data
    pro_stats_path: stats dict used for inverse transform back to CODEX values
    cv_split_path: cv_splits_patches.json
    out_path: "results"

Outputs:
    pred_val.csv: barcode, pxl_row_in_fullres, pxl_col_in_fullres, <n markers>
    gnn_v3_model.pt: model weights + config
    history.csv: per-epoch train loss / pearson r during the final refit
    svd_model.pkl: fitted TruncatedSVD, to reproduce exact features for inference re run
"""

import os
import pickle
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD

from preprocessing_final import inverse_transform_protein
from cv_split_patches import load_cv_split

SEED = 0
N_SVD_COMPONENTS = 128


def to_dense(X):
    return np.asarray(X.todense() if hasattr(X, "todense") else X)


# RNA: SVD
def reduce_rna_svd(rna_train, rna_val, n_components=N_SVD_COMPONENTS):
    """
    preprocessed rna_train & rna_val, sparse for TruncatedSVD
    """
    X_train = rna_train.X if sp.issparse(rna_train.X) else sp.csr_matrix(rna_train.X)
    X_val = rna_val.X if sp.issparse(rna_val.X) else sp.csr_matrix(rna_val.X)

    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    X_train_svd = svd.fit_transform(X_train).astype(np.float32)
    X_val_svd = svd.transform(X_val).astype(np.float32)
    print(f"SVD cumulative explained variance (train): {svd.explained_variance_ratio_.sum():.3f}")
    return X_train_svd, X_val_svd, svd


# Graph construction
def build_knn_graph(coords, k):
    """Symmetric kNN graph from spatial pixel coordinates via cKDTree."""
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
class ResidualGNNBlock(nn.Module):
    """
    SAGEConv or GATConv -> LayerNorm -> ReLU(h + x) residual -> Dropout
    conv_type="sage": uniform mean over the k neighbors (original behavior)
    conv_type="gat": learns a per-neighbor attention weight instead
    - uses heads=gat_heads with concat=False which averages the heads back down to
      exactly `hidden` dims so it stays compatible with the residual add below
      without any extra dimension bookkeeping
    """

    def __init__(self, hidden, dropout=0.3, conv_type="sage", gat_heads=2):
        super().__init__()
        if conv_type == "sage":
            self.conv = SAGEConv(hidden, hidden)
        elif conv_type == "gat":
            self.conv = GATConv(hidden, hidden, heads=gat_heads, concat=False)
        else:
            raise ValueError(f"unknown conv_type: {conv_type!r} (expected 'sage' or 'gat')")
        self.norm = nn.LayerNorm(hidden)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = self.norm(h)
        h = F.relu(h + x)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class ResidualGraphSAGE(nn.Module):
    """
    single-branch stack of ResidualGNNBlocks over ONE kNN graph.
    conv_type: "sage" (default) or "gat" - applies to every block
    use_jk: if True, concatenates every block's output (not just the last)
        before the head, via a learned linear projection back down to `hidden` -
        gives the head direct access to both fine (early layer, small receptive
        field) and broad (late layer, large receptive field) representations
        instead of only the last layer's output
    """

    def __init__(self, in_dim, out_dim, hidden=256, n_layers=4, dropout=0.3,
                 conv_type="sage", gat_heads=2, use_jk=False):
        super().__init__()
        self.use_jk = use_jk
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            ResidualGNNBlock(hidden, dropout, conv_type, gat_heads) for _ in range(n_layers)
        ])
        if use_jk:
            self.jk_proj = nn.Linear(hidden * n_layers, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim),
        )
        self._config = dict(in_dim=in_dim, out_dim=out_dim, hidden=hidden, n_layers=n_layers,
                            dropout=dropout, conv_type=conv_type, gat_heads=gat_heads, use_jk=use_jk)

    def forward(self, x, edge_index):
        h = self.input_proj(x)
        layer_outputs = []
        for blk in self.blocks:
            h = blk(h, edge_index)
            layer_outputs.append(h)
        if self.use_jk:
            h = self.jk_proj(torch.cat(layer_outputs, dim=-1))
        return self.head(h)


# Loss
def pearson_loss(yp, yt):
    vp = yp - yp.mean(0, keepdim=True)
    vt = yt - yt.mean(0, keepdim=True)
    r = (vp * vt).sum(0) / ((vp ** 2).sum(0) * (vt ** 2).sum(0) + 1e-8).sqrt()
    return (1 - r).mean()


def combined_loss(yp, yt, w=0.8):
    return w * F.mse_loss(yp, yt) + (1 - w) * pearson_loss(yp, yt)


def mean_pearson_r(pred, true):
    rs = [pearsonr(pred[:, j], true[:, j])[0] for j in range(true.shape[1])]
    rs = [r for r in rs if not np.isnan(r)]
    return float(np.mean(rs)) if rs else 0.0


# LR schedule
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr, min_lr=1e-6):
        self.opt, self.warmup_epochs, self.max_epochs = optimizer, warmup_epochs, max_epochs
        self.base_lr, self.min_lr, self.epoch = base_lr, min_lr, 0

    def step(self):
        self.epoch += 1
        e = self.epoch
        if e <= self.warmup_epochs:
            lr = self.base_lr * e / self.warmup_epochs
        else:
            progress = (e - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        for pg in self.opt.param_groups:
            pg["lr"] = lr
        return lr


# Step 1: epoch count via buffered CV fold
def find_n_epochs(X, Y, edge_index, cv_splits, fold, model_cfg, opt_cfg, device,
                  max_epochs, patience, verbose=True):
    split = cv_splits[fold]
    train_idx, holdout_idx = np.array(split["train"]), np.array(split["test"])

    x_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    edge_index_dev = edge_index.to(device)

    train_mask = torch.zeros(X.shape[0], dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    holdout_mask = torch.zeros(X.shape[0], dtype=torch.bool, device=device)
    holdout_mask[holdout_idx] = True

    loss_w = opt_cfg.get("loss_w", 0.8)
    model = ResidualGraphSAGE(in_dim=X.shape[1], out_dim=Y.shape[1], **model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
    sched = WarmupCosineScheduler(optimizer, opt_cfg["warmup"], max_epochs, opt_cfg["lr"])

    best_val_loss, best_epoch, patience_ctr = float("inf"), 0, 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x_t, edge_index_dev)
        loss = combined_loss(out[train_mask], y_t[train_mask], w=loss_w)
        loss.backward()
        optimizer.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            out = model(x_t, edge_index_dev)
            val_loss = combined_loss(out[holdout_mask], y_t[holdout_mask], w=loss_w).item()

        if val_loss < best_val_loss:
            best_val_loss, best_epoch, patience_ctr = val_loss, epoch, 0
        else:
            patience_ctr += 1

        if verbose and (epoch % 10 == 0 or epoch == max_epochs - 1):
            print(f"    [epoch-finder] epoch {epoch:4d}  train_loss {loss.item():.4f}  holdout_loss {val_loss:.4f}")

        if patience_ctr >= patience:
            print(f"    [epoch-finder] early stop at epoch {epoch} (best epoch {best_epoch})")
            break

    n_epochs = best_epoch + 1
    print(f"    [epoch-finder] selected n_epochs = {n_epochs} (fold {fold})")
    return n_epochs


# Step 2: refit on all data
def train_model(X, Y, edge_index, n_epochs, model_cfg, opt_cfg, device, verbose=True):
    x_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    edge_index = edge_index.to(device)

    loss_w = opt_cfg.get("loss_w", 0.8)
    model = ResidualGraphSAGE(in_dim=X.shape[1], out_dim=Y.shape[1], **model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
    sched = WarmupCosineScheduler(optimizer, opt_cfg["warmup"], n_epochs, opt_cfg["lr"])

    history = []
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = model(x_t, edge_index)
        loss = combined_loss(out, y_t, w=loss_w)
        loss.backward()
        optimizer.step()
        lr_now = sched.step()

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                pred_np = model(x_t, edge_index).cpu().numpy()
            train_r = mean_pearson_r(pred_np, Y)
            model.train()
            history.append({"epoch": epoch, "train_loss": loss.item(), "train_pearson": train_r, "lr": lr_now})
            if verbose:
                print(f"  epoch {epoch:4d}  train_loss {loss.item():.4f}  train_r {train_r:.4f}")

    return model, pd.DataFrame(history)


def save_model(model, path):
    torch.save({"state_dict": model.state_dict(), "config": model._config}, path)


def load_model(path, model_cls=ResidualGraphSAGE, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    model = model_cls(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def predict(model, X_val, edge_index_val, device):
    model.eval()
    x_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    edge_index_val = edge_index_val.to(device)
    with torch.no_grad():
        out = model(x_t, edge_index_val)
    return out.cpu().numpy()


# Run model
def run_gnn_v3(rna_train_path, pro_train_path, rna_val_path, pro_stats_path, cv_split_path,
               out_path, n_components=N_SVD_COMPONENTS, k=8, hidden=256, n_layers=4, dropout=0.1,
               conv_type="sage", gat_heads=2, use_jk=False,
               lr=0.0003, weight_decay=0.0001, warmup=10, loss_w=0.5, max_epochs=1000, patience=30,
               epoch_finder_fold=1, device=None):
    os.makedirs(out_path, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    rna_train = ad.read_h5ad(rna_train_path)
    rna_val = ad.read_h5ad(rna_val_path)
    pro_train = ad.read_h5ad(pro_train_path)

    with open(pro_stats_path, "rb") as f:
        protein_stats = pickle.load(f)

    barcodes_val = rna_val.obs.index.values
    coords_train = rna_train.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values.astype(np.float32)
    coords_val = rna_val.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values.astype(np.float32)
    marker_names = list(pro_train.var_names)

    X_train, X_val, svd = reduce_rna_svd(rna_train, rna_val, n_components=n_components)
    Y_train = to_dense(pro_train.X).astype(np.float32)  # already arcsinh/clip/z-scored

    with open(os.path.join(out_path, "svd_model.pkl"), "wb") as f:
        pickle.dump(svd, f)

    edge_index_train = build_knn_graph(coords_train, k=k)
    edge_index_val = build_knn_graph(coords_val, k=k)

    model_cfg = dict(hidden=hidden, n_layers=n_layers, dropout=dropout,
                     conv_type=conv_type, gat_heads=gat_heads, use_jk=use_jk)
    opt_cfg = dict(lr=lr, weight_decay=weight_decay, warmup=warmup, loss_w=loss_w)

    cv_splits = load_cv_split(cv_split_path)
    n_epochs = find_n_epochs(X_train, Y_train, edge_index_train, cv_splits, epoch_finder_fold,
                             model_cfg, opt_cfg, device, max_epochs, patience)

    model, history = train_model(X_train, Y_train, edge_index_train, n_epochs, model_cfg, opt_cfg, device)
    save_model(model, os.path.join(out_path, "gnn_v3_model.pt"))
    history.to_csv(os.path.join(out_path, "history.csv"), index=False)

    Z_pred = predict(model, X_val, edge_index_val, device)
    X_pred = inverse_transform_protein(Z_pred, protein_stats)
    X_pred = np.clip(X_pred, a_min=0, a_max=None)

    submission = pd.DataFrame(X_pred, columns=marker_names)
    submission.insert(0, "pxl_col_in_fullres", coords_val[:, 1].astype(int))
    submission.insert(0, "pxl_row_in_fullres", coords_val[:, 0].astype(int))
    submission.insert(0, "barcode", barcodes_val)

    submission_path = os.path.join(out_path, "pred_val.csv")
    submission.to_csv(submission_path, index=False)
    print(f"saved {submission_path}  shape={submission.shape}")

    return model, submission, n_epochs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_train_path", required=True)
    parser.add_argument("--rna_val_path", required=True)
    parser.add_argument("--pro_train_path", required=True)
    parser.add_argument("--pro_stats_path", required=True)
    parser.add_argument("--cv_split_path", required=True)
    parser.add_argument("--out_path", required=True)

    parser.add_argument("--n_components", type=int, default=128)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--conv_type", choices=["sage", "gat"], default="sage",
                        help="'sage' (default, matches the original fixed model) or 'gat'")
    parser.add_argument("--gat_heads", type=int, default=2, help="only used when --conv_type gat")
    parser.add_argument("--use_jk", action="store_true",
                        help="enable Jumping Knowledge (concat all layer outputs before the head)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--loss_w", type=float, default=0.8,
                        help="combined_loss MSE weight; (1 - loss_w) goes to the Pearson term")
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--epoch_finder_fold", type=int, default=1,
                        help="which fold of cv_split_path to use for early-stopping epoch selection")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None

    run_gnn_v3(
        rna_train_path=args.rna_train_path,
        rna_val_path=args.rna_val_path,
        pro_train_path=args.pro_train_path,
        pro_stats_path=args.pro_stats_path,
        cv_split_path=args.cv_split_path,
        out_path=args.out_path,
        n_components=args.n_components,
        k=args.k,
        hidden=args.hidden,
        n_layers=args.n_layers,
        dropout=args.dropout,
        conv_type=args.conv_type,
        gat_heads=args.gat_heads,
        use_jk=args.use_jk,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup=args.warmup,
        loss_w=args.loss_w,
        max_epochs=args.max_epochs,
        patience=args.patience,
        epoch_finder_fold=args.epoch_finder_fold,
        device=device,
    )


if __name__ == "__main__":
    main()