"""
DGAT (Dual Graph Attention Network) for RNA -> protein

Architecture:
    - Two graphs per modality: a spatial kNN graph (physical proximity)
      and a modality-specific feature-similarity kNN graph, unioned per
      modality
    - Two independent GAT encoders (mRNA, protein), each: input projection
      -> N x [GATConv -> LayerNorm -> feature-attention gate -> residual
      add -> LeakyReLU -> dropout] -> linear projection to a shared latent
      dim.
    - mRNA decoder: residual feedforward MLP, latent -> mRNA feature space.
    - Protein decoder: shared trunk -> one small branch per protein marker.
    - Five loss terms (paired training data only): align, recon_mrna,
      recon_protein, cross_protein (mRNA embedding decoded as protein), cross_mrna.
    - Inference (RNA-only data): X_mrna -> mrna_encoder -> z_mrna ->
      protein_decoder -> protein prediction. protein_encoder and
      mrna_decoder are trained but unused at inference.

Training: mini-batch, neighbor-sampled (PyTorch Geometric NeighborLoader)

Inputs: preprocessed rna_train/rna_val, pro_train, pro_stats_path,
cv_split_path, out_path

Outputs:
    pred_val.csv          : barcode, pxl_row_in_fullres, pxl_col_in_fullres, <markers>
    dgat_model.pt          : model weights + config
    history.csv            : per-eval-step holdout cross_protein RMSE during training
    svd_model.pkl            : fitted TruncatedSVD for RNA
    protein_pca_model.pkl : fitted PCA for protein feature graph (train-only)
"""

import os
import pickle
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD, PCA

from preprocessing_final import inverse_transform_protein
from cv_split_patches import load_cv_split

SEED = 0
N_SVD_COMPONENTS = 128       # RNA working representation (same as gnn_v3.py)
PROTEIN_PCA_VARIANCE = 0.85  # paper: retain 85% variance for protein feature graph
DEFAULT_NUM_NEIGHBORS = [15, 10, 10]  # per-layer sampling fanout, closest hop first
DEFAULT_BATCH_SIZE = 512


def to_dense(X):
    return np.asarray(X.todense() if hasattr(X, "todense") else X)


# ---------------------------------------------------------------------------
# Dimensionality reduction (feature spaces for the feature-similarity graphs)
# ---------------------------------------------------------------------------

def reduce_rna_svd(rna_train, rna_val, n_components=N_SVD_COMPONENTS):
    """Fold-safe: fit on train only."""
    X_train = rna_train.X if sp.issparse(rna_train.X) else sp.csr_matrix(rna_train.X)
    X_val = rna_val.X if sp.issparse(rna_val.X) else sp.csr_matrix(rna_val.X)

    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    X_train_svd = svd.fit_transform(X_train).astype(np.float32)
    X_val_svd = svd.transform(X_val).astype(np.float32)
    print(f"RNA SVD cumulative explained variance (train): {svd.explained_variance_ratio_.sum():.3f}")
    return X_train_svd, X_val_svd, svd


def reduce_protein_pca(pro_train_dense, variance=PROTEIN_PCA_VARIANCE):
    """
    PCA on training protein data only - to build the protein feature-similarity graph
    """
    pca = PCA(n_components=variance, random_state=SEED)
    Z = pca.fit_transform(pro_train_dense).astype(np.float32)
    print(f"Protein PCA: {pca.n_components_} components retain {variance:.0%} variance (train)")
    return Z, pca


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _knn_edge_index(X, k):
    """Symmetric kNN graph (edge_index) from any feature matrix via cKDTree"""
    tree = cKDTree(X)
    _, idx = tree.query(X, k=k + 1, workers=-1)  # first column is self
    n = X.shape[0]
    src = np.repeat(np.arange(n), k)
    dst = idx[:, 1:].reshape(-1)
    edge_index = np.vstack([src, dst])
    edge_index = np.hstack([edge_index, edge_index[[1, 0]]])  # symmetrize
    edge_index = np.unique(edge_index, axis=1)
    return torch.tensor(edge_index, dtype=torch.long).contiguous()


def build_spatial_graph(coords, k=6):
    """Physical-proximity kNN graph."""
    return _knn_edge_index(coords, k)


def build_feature_graph(X, k=10):
    """Feature-similarity kNN graph in a reduced feature space (SVD/PCA)."""
    return _knn_edge_index(X, k)


def union_edge_index(edge_index_a, edge_index_b):
    """A_mRNA+s or A_protein+s : union of a modality graph and the spatial graph"""
    combined = torch.cat([edge_index_a, edge_index_b], dim=1)
    return torch.unique(combined, dim=1).contiguous()


def build_pyg_data(X, edge_index):
    return Data(x=torch.tensor(X, dtype=torch.float32), edge_index=edge_index, num_nodes=X.shape[0])


# ---------------------------------------------------------------------------
# Encoder: GAT layers + feature-level attention gating + residual + norm
# ---------------------------------------------------------------------------

class FeatureAttention(nn.Module):
    """
    Multi-head sigmoid-gated channel attention over node features - decides how much each feature channel
    of the aggregated result matters
    """

    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_heads)])

    def forward(self, x):
        gates = torch.stack([torch.sigmoid(h(x)) for h in self.heads], dim=0).mean(0)
        return x * gates


class DGATEncoderBlock(nn.Module):
    """GATConv -> LayerNorm -> feature attention -> residual -> LeakyReLU -> dropout."""

    def __init__(self, hidden, dropout=0.3, gat_heads=2, feat_attn_heads=4):
        super().__init__()
        self.conv = GATConv(hidden, hidden, heads=gat_heads, concat=False)
        self.norm = nn.LayerNorm(hidden)
        self.feat_attn = FeatureAttention(hidden, feat_attn_heads)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = self.norm(h)
        h = self.feat_attn(h)
        h = F.leaky_relu(h + x)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GATEncoder(nn.Module):
    """Modality-specific encoder. mRNA and protein encoders share this class"""

    def __init__(self, in_dim, hidden=128, latent_dim=256, n_layers=3,
                 dropout=0.3, gat_heads=2, feat_attn_heads=4):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            DGATEncoderBlock(hidden, dropout, gat_heads, feat_attn_heads) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(hidden, latent_dim)

    def forward(self, x, edge_index):
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h, edge_index)
        return self.output_proj(h)


# Decoders

class ResidualDecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.dropout = dropout

    def forward(self, x):
        h = self.lin(x)
        h = self.norm(h)
        h = F.leaky_relu(h + self.proj(x))
        return F.dropout(h, p=self.dropout, training=self.training)


class MRNADecoder(nn.Module):
    """Residual feedforward MLP: latent -> RNA (SVD) feature space."""

    def __init__(self, latent_dim, out_dim, hidden=128, n_layers=2, dropout=0.3):
        super().__init__()
        dims = [latent_dim] + [hidden] * n_layers
        self.blocks = nn.ModuleList([
            ResidualDecoderBlock(dims[i], dims[i + 1], dropout) for i in range(n_layers)
        ])
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, z):
        h = z
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)


class ProteinDecoder(nn.Module):
    """Shared trunk -> one small branch per protein marker."""

    def __init__(self, latent_dim, n_proteins, trunk_hidden=128, branch_hidden=32, dropout=0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, trunk_hidden),
            nn.LayerNorm(trunk_hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.LayerNorm(trunk_hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trunk_hidden, branch_hidden),
                nn.LeakyReLU(),
                nn.Linear(branch_hidden, 1),
            ) for _ in range(n_proteins)
        ])

    def forward(self, z):
        shared = self.trunk(z)
        return torch.cat([branch(shared) for branch in self.branches], dim=-1)


# Full DGAT model

class DGAT(nn.Module):
    def __init__(self, rna_dim, protein_dim, hidden=128, latent_dim=256,
                 n_layers=3, dropout=0.3, gat_heads=2, feat_attn_heads=4,
                 decoder_hidden=128, branch_hidden=32):
        super().__init__()
        self.mrna_encoder = GATEncoder(rna_dim, hidden, latent_dim, n_layers,
                                       dropout, gat_heads, feat_attn_heads)
        self.protein_encoder = GATEncoder(protein_dim, hidden, latent_dim, n_layers,
                                          dropout, gat_heads, feat_attn_heads)
        self.mrna_decoder = MRNADecoder(latent_dim, rna_dim, decoder_hidden, dropout=dropout)
        self.protein_decoder = ProteinDecoder(latent_dim, protein_dim, decoder_hidden,
                                              branch_hidden, dropout=dropout)
        self._config = dict(rna_dim=rna_dim, protein_dim=protein_dim, hidden=hidden,
                            latent_dim=latent_dim, n_layers=n_layers, dropout=dropout,
                            gat_heads=gat_heads, feat_attn_heads=feat_attn_heads,
                            decoder_hidden=decoder_hidden, branch_hidden=branch_hidden)

    def forward_paired(self, x_mrna, edge_mrna, x_protein, edge_protein):
        """Full-graph forward pass (both modalities)
        Kept for small-graph/debug use - training loops use mini-batch subgraphs instead."""
        z_mrna = self.mrna_encoder(x_mrna, edge_mrna)
        z_protein = self.protein_encoder(x_protein, edge_protein)
        return dict(
            z_mrna=z_mrna, z_protein=z_protein,
            mrna_recon=self.mrna_decoder(z_mrna),
            protein_recon=self.protein_decoder(z_protein),
            protein_from_mrna=self.protein_decoder(z_mrna),
            mrna_from_protein=self.mrna_decoder(z_protein),
        )

    def predict_protein_from_rna(self, x_mrna, edge_mrna):
        """Full-graph inference (RNA-only)"""
        z_mrna = self.mrna_encoder(x_mrna, edge_mrna)
        return self.protein_decoder(z_mrna)


# Losses

def rmse(pred, target):
    return torch.sqrt(F.mse_loss(pred, target) + 1e-8)


def dgat_loss(outputs, x_mrna, x_protein, weights, align_zero_threshold=0.015):
    """
    weights: dict with keys 'align', 'recon_mrna', 'recon_protein',
             'cross_protein', 'cross_mrna'. Paper defaults:
             align=5, recon_mrna=1, recon_protein=1, cross_protein=3, cross_mrna=5
    """
    align = F.mse_loss(outputs["z_mrna"], outputs["z_protein"])
    recon_mrna = rmse(outputs["mrna_recon"], x_mrna)
    recon_protein = rmse(outputs["protein_recon"], x_protein)
    cross_protein = rmse(outputs["protein_from_mrna"], x_protein)
    cross_mrna = rmse(outputs["mrna_from_protein"], x_mrna)

    # dynamic zeroing: stop pushing the two latent spaces together once they're already close, to avoid modality collapse
    align_w = 0.0 if align.item() < align_zero_threshold else weights["align"]

    total = (align_w * align
             + weights["recon_mrna"] * recon_mrna
             + weights["recon_protein"] * recon_protein
             + weights["cross_protein"] * cross_protein
             + weights["cross_mrna"] * cross_mrna)

    parts = dict(align=align.item(), recon_mrna=recon_mrna.item(),
                recon_protein=recon_protein.item(), cross_protein=cross_protein.item(),
                cross_mrna=cross_mrna.item(), total=total.item())
    return total, parts


def mean_pearson_r(pred, true):
    rs = [pearsonr(pred[:, j], true[:, j])[0] for j in range(true.shape[1])]
    rs = [r for r in rs if not np.isnan(r)]
    return float(np.mean(rs)) if rs else 0.0


# Optimizer: single AdamW, per-module param groups

def build_optimizer(model, encoder_lr=5e-4, decoder_lr=1e-4, encoder_wd=2e-5):
    param_groups = [
        {"params": model.mrna_encoder.parameters(), "lr": encoder_lr, "weight_decay": encoder_wd},
        {"params": model.protein_encoder.parameters(), "lr": encoder_lr, "weight_decay": encoder_wd},
        {"params": model.mrna_decoder.parameters(), "lr": decoder_lr, "weight_decay": 0.0},
        {"params": model.protein_decoder.parameters(), "lr": decoder_lr, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups)


class StepDecayScheduler:
    """Paper: lr reduced by 20% every 10 epochs."""

    def __init__(self, optimizer, decay=0.8, every=10):
        self.opt, self.decay, self.every, self.epoch = optimizer, decay, every, 0

    def step(self):
        self.epoch += 1
        if self.epoch % self.every == 0:
            for pg in self.opt.param_groups:
                pg["lr"] *= self.decay


# Mini-batch (neighbor-sampled) training

def _make_loader(data, seed_idx, num_neighbors, batch_size, num_workers=0):
    """shuffle=False (caller controls ordering by pre-shuffling seed_idx,
    so two loaders fed the same seed_idx produce aligned batches)
    num_workers>0 parallelizes CPU-side neighbor sampling across processes"""
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=seed_idx,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def train_epoch_batched(model, data_mrna, data_protein, train_idx, optimizer,
                        loss_weights, device, num_neighbors=DEFAULT_NUM_NEIGHBORS,
                        batch_size=DEFAULT_BATCH_SIZE, grad_clip=1.0, num_workers=0):
    """One epoch of mini-batch training. Returns mean total loss across batches."""
    model.train()
    perm = train_idx[torch.randperm(train_idx.numel())]

    loader_mrna = _make_loader(data_mrna, perm, num_neighbors, batch_size, num_workers)
    loader_protein = _make_loader(data_protein, perm, num_neighbors, batch_size, num_workers)

    batch_losses = []
    for batch_mrna, batch_protein in zip(loader_mrna, loader_protein):
        batch_mrna = batch_mrna.to(device)
        batch_protein = batch_protein.to(device)
        bs = batch_mrna.batch_size
        assert bs == batch_protein.batch_size, "mRNA/protein batch seed counts diverged"

        optimizer.zero_grad()
        z_mrna_full = model.mrna_encoder(batch_mrna.x, batch_mrna.edge_index)
        z_protein_full = model.protein_encoder(batch_protein.x, batch_protein.edge_index)

        z_mrna = z_mrna_full[:bs]
        z_protein = z_protein_full[:bs]
        x_mrna_seed = batch_mrna.x[:bs]
        x_protein_seed = batch_protein.x[:bs]

        out = dict(
            z_mrna=z_mrna, z_protein=z_protein,
            mrna_recon=model.mrna_decoder(z_mrna),
            protein_recon=model.protein_decoder(z_protein),
            protein_from_mrna=model.protein_decoder(z_mrna),
            mrna_from_protein=model.mrna_decoder(z_protein),
        )
        loss, parts = dgat_loss(out, x_mrna_seed, x_protein_seed, loss_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_losses.append(parts["total"])

    return float(np.mean(batch_losses)) if batch_losses else float("nan")


@torch.no_grad()
def eval_cross_protein_rmse_batched(model, data_mrna, data_protein, eval_idx, device,
                                    num_neighbors=DEFAULT_NUM_NEIGHBORS,
                                    batch_size=DEFAULT_BATCH_SIZE, num_workers=0):
    """Holdout cross_protein RMSE, pooled over all eval_idx nodes (not averaged per-batch)."""
    model.eval()
    loader_mrna = _make_loader(data_mrna, eval_idx, num_neighbors, batch_size, num_workers)
    loader_protein = _make_loader(data_protein, eval_idx, num_neighbors, batch_size, num_workers)

    sq_errors, n_total = 0.0, 0
    for batch_mrna, batch_protein in zip(loader_mrna, loader_protein):
        batch_mrna = batch_mrna.to(device)
        batch_protein = batch_protein.to(device)
        bs = batch_mrna.batch_size

        z_mrna = model.mrna_encoder(batch_mrna.x, batch_mrna.edge_index)[:bs]
        pred = model.protein_decoder(z_mrna)
        target = batch_protein.x[:bs]

        sq_errors += ((pred - target) ** 2).sum().item()
        n_total += target.numel()

    return float(np.sqrt(sq_errors / n_total)) if n_total else float("nan")


@torch.no_grad()
def predict_batched(model, data_mrna, seed_idx, device,
                    num_neighbors=DEFAULT_NUM_NEIGHBORS, batch_size=DEFAULT_BATCH_SIZE, num_workers=0):
    """
    RNA-only inference path (mrna_encoder + protein_decoder), batched.
    Returns (predictions, node_ids): node_ids is the original node index
    each prediction row corresponds to
    """
    model.eval()
    loader = _make_loader(data_mrna, seed_idx, num_neighbors, batch_size, num_workers)

    all_preds, all_ids = [], []
    for batch in loader:
        batch = batch.to(device)
        bs = batch.batch_size
        z_mrna = model.mrna_encoder(batch.x, batch.edge_index)[:bs]
        pred = model.protein_decoder(z_mrna)
        all_preds.append(pred.cpu().numpy())
        all_ids.append(batch.n_id[:bs].cpu().numpy())

    return np.concatenate(all_preds, axis=0), np.concatenate(all_ids, axis=0)


def train_dgat_with_early_stopping(
        X_mrna, X_protein, edge_mrna, edge_protein, train_idx, holdout_idx,
        model_cfg, loss_weights, device,
        num_neighbors=DEFAULT_NUM_NEIGHBORS, batch_size=DEFAULT_BATCH_SIZE, num_workers=0,
        lr_encoder=5e-4, lr_decoder=1e-4, encoder_wd=2e-5,
        max_epochs=300, patience=20, grad_clip=1.0, verbose=True):
    """
    Mini-batch training loop with early stopping + best-state checkpointing,
    on a train_idx/holdout_idx split within ONE graph (used by both
    run_dgat's epoch-finder fold and dgat_cv.py's per-fold training).
    """
    data_mrna = build_pyg_data(X_mrna, edge_mrna)
    data_protein = build_pyg_data(X_protein, edge_protein)

    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long)
    holdout_idx_t = torch.as_tensor(holdout_idx, dtype=torch.long)

    model = DGAT(**model_cfg).to(device)
    optimizer = build_optimizer(model, encoder_lr=lr_encoder, decoder_lr=lr_decoder, encoder_wd=encoder_wd)
    sched = StepDecayScheduler(optimizer)

    best_val_loss, best_epoch, patience_ctr = float("inf"), 0, 0
    best_state = None
    history = []

    for epoch in range(max_epochs):
        train_loss = train_epoch_batched(
            model, data_mrna, data_protein, train_idx_t, optimizer, loss_weights,
            device, num_neighbors=num_neighbors, batch_size=batch_size, grad_clip=grad_clip,
            num_workers=num_workers,
        )
        sched.step()

        val_loss = eval_cross_protein_rmse_batched(
            model, data_mrna, data_protein, holdout_idx_t, device,
            num_neighbors=num_neighbors, batch_size=batch_size, num_workers=num_workers,
        )
        history.append(dict(epoch=epoch, train_loss=train_loss, holdout_cross_protein_rmse=val_loss))

        if val_loss < best_val_loss:
            best_val_loss, best_epoch, patience_ctr = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if verbose and (epoch % 5 == 0 or epoch == max_epochs - 1):
            print(f"    epoch {epoch:4d}  train_loss {train_loss:.4f}  "
                 f"holdout_cross_protein_rmse {val_loss:.4f}")

        if patience_ctr >= patience:
            print(f"    early stop at epoch {epoch} (best epoch {best_epoch}, "
                 f"holdout_cross_protein_rmse {best_val_loss:.4f})")
            break

    model.load_state_dict(best_state)
    model.eval()

    return model, best_epoch + 1, pd.DataFrame(history)


def save_model(model, path):
    torch.save({"state_dict": model.state_dict(), "config": model._config}, path)


def load_model(path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    model = DGAT(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# Run: single train/val run

def run_dgat(rna_train_path, pro_train_path, rna_val_path, pro_stats_path, cv_split_path,
            out_path, n_svd_components=N_SVD_COMPONENTS, protein_pca_variance=PROTEIN_PCA_VARIANCE,
            k_spatial=6, k_feature=10, hidden=128, latent_dim=256, n_layers=3, dropout=0.3,
            gat_heads=2, feat_attn_heads=4, decoder_hidden=128, branch_hidden=32,
            loss_weights=None, num_neighbors=None, batch_size=DEFAULT_BATCH_SIZE, num_workers=0,
            lr_encoder=5e-4, lr_decoder=1e-4, encoder_wd=2e-5,
            max_epochs=300, patience=20, epoch_finder_fold=1, device=None):
    os.makedirs(out_path, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_neighbors = num_neighbors or DEFAULT_NUM_NEIGHBORS
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    loss_weights = loss_weights or dict(align=5.0, recon_mrna=1.0, recon_protein=1.0,
                                        cross_protein=3.0, cross_mrna=5.0)

    rna_train = ad.read_h5ad(rna_train_path)
    rna_val = ad.read_h5ad(rna_val_path)
    pro_train = ad.read_h5ad(pro_train_path)

    with open(pro_stats_path, "rb") as f:
        protein_stats = pickle.load(f)

    barcodes_val = rna_val.obs.index.values
    coords_train = rna_train.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values.astype(np.float32)
    coords_val = rna_val.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values.astype(np.float32)
    marker_names = list(pro_train.var_names)

    # --- feature spaces (fold-safe: fit on train only) ---
    X_mrna_train, X_mrna_val, svd = reduce_rna_svd(rna_train, rna_val, n_components=n_svd_components)
    Y_protein_train = to_dense(pro_train.X).astype(np.float32)
    X_protein_pca, protein_pca = reduce_protein_pca(Y_protein_train, variance=protein_pca_variance)

    with open(os.path.join(out_path, "svd_model.pkl"), "wb") as f:
        pickle.dump(svd, f)
    with open(os.path.join(out_path, "protein_pca_model.pkl"), "wb") as f:
        pickle.dump(protein_pca, f)

    # graphs (train)
    spatial_edge_train = build_spatial_graph(coords_train, k=k_spatial)
    mrna_feat_edge_train = build_feature_graph(X_mrna_train, k=k_feature)
    protein_feat_edge_train = build_feature_graph(X_protein_pca, k=k_feature)
    edge_mrna_train = union_edge_index(mrna_feat_edge_train, spatial_edge_train)
    edge_protein_train = union_edge_index(protein_feat_edge_train, spatial_edge_train)

    # graph (val, RNA-only)
    spatial_edge_val = build_spatial_graph(coords_val, k=k_spatial)
    mrna_feat_edge_val = build_feature_graph(X_mrna_val, k=k_feature)
    edge_mrna_val = union_edge_index(mrna_feat_edge_val, spatial_edge_val)

    model_cfg = dict(rna_dim=n_svd_components, protein_dim=Y_protein_train.shape[1], hidden=hidden,
                     latent_dim=latent_dim, n_layers=n_layers, dropout=dropout, gat_heads=gat_heads,
                     feat_attn_heads=feat_attn_heads, decoder_hidden=decoder_hidden,
                     branch_hidden=branch_hidden)

    cv_splits = load_cv_split(cv_split_path)
    split = next(s for s in cv_splits if s["fold"] == epoch_finder_fold)
    train_idx, holdout_idx = np.array(split["train"]), np.array(split["test"])

    model, n_epochs, history = train_dgat_with_early_stopping(
        X_mrna_train, Y_protein_train, edge_mrna_train, edge_protein_train, train_idx, holdout_idx,
        model_cfg, loss_weights, device, num_neighbors=num_neighbors, batch_size=batch_size,
        num_workers=num_workers, lr_encoder=lr_encoder, lr_decoder=lr_decoder, encoder_wd=encoder_wd,
        max_epochs=max_epochs, patience=patience,
    )
    save_model(model, os.path.join(out_path, "dgat_model.pt"))
    history.to_csv(os.path.join(out_path, "history.csv"), index=False)

    data_mrna_val = build_pyg_data(X_mrna_val, edge_mrna_val)
    all_val_idx = torch.arange(X_mrna_val.shape[0], dtype=torch.long)
    Z_pred, node_ids = predict_batched(model, data_mrna_val, all_val_idx, device,
                                       num_neighbors=num_neighbors, batch_size=batch_size,
                                       num_workers=num_workers)
    order = np.argsort(node_ids)
    Z_pred = Z_pred[order]

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


# entry point

def main():
    import argparse

    parser = argparse.ArgumentParser()
    # data / paths
    parser.add_argument("--rna_train_path", default="data/rna_train_hvg.h5ad")
    parser.add_argument("--pro_train_path", default="data/protein_data_v2.h5ad")
    parser.add_argument("--rna_val_path", default="data/rna_val_hvg.h5ad")
    parser.add_argument("--pro_stats_path", default="data/protein_stats.pkl")
    parser.add_argument("--cv_split_path", default="data/cv_splits_patches.json")
    parser.add_argument("--out_path", default="results")

    # feature spaces
    parser.add_argument("--n_svd_components", type=int, default=N_SVD_COMPONENTS)
    parser.add_argument("--protein_pca_variance", type=float, default=PROTEIN_PCA_VARIANCE)

    # graphs
    parser.add_argument("--k_spatial", type=int, default=6)
    parser.add_argument("--k_feature", type=int, default=10)

    # model architecture -- defaults scaled down from the paper's 256/1024
    # for T4 memory constraints, see module docstring
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gat_heads", type=int, default=2)
    parser.add_argument("--feat_attn_heads", type=int, default=4)
    parser.add_argument("--decoder_hidden", type=int, default=128)
    parser.add_argument("--branch_hidden", type=int, default=32)

    # loss weights -- paper defaults: align=5, recon_mrna=1, recon_protein=1,
    # cross_protein=3, cross_mrna=5
    parser.add_argument("--w_align", type=float, default=5.0)
    parser.add_argument("--w_recon_mrna", type=float, default=1.0)
    parser.add_argument("--w_recon_protein", type=float, default=1.0)
    parser.add_argument("--w_cross_protein", type=float, default=3.0)
    parser.add_argument("--w_cross_mrna", type=float, default=5.0)

    # optimizer -- paper defaults: encoder_lr=5e-4, decoder_lr=1e-4, encoder_wd=2e-5
    parser.add_argument("--lr_encoder", type=float, default=5e-4)
    parser.add_argument("--lr_decoder", type=float, default=1e-4)
    parser.add_argument("--encoder_wd", type=float, default=2e-5)

    # mini-batch training
    parser.add_argument("--num_neighbors", type=int, nargs="+", default=None,
                        help="per-layer sampling fanout, e.g. --num_neighbors 15 10 10 "
                            "(defaults to dgat.DEFAULT_NUM_NEIGHBORS if omitted)")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="parallel CPU processes for graph neighbor sampling; "
                            "try 4 if training is sampling-bound (check nvidia-smi)")

    # training
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--epoch_finder_fold", type=int, default=1)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device == "cuda" and device.type == "cpu":
        print("cuda requested but not available -- falling back to cpu")

    loss_weights = dict(
        align=args.w_align,
        recon_mrna=args.w_recon_mrna,
        recon_protein=args.w_recon_protein,
        cross_protein=args.w_cross_protein,
        cross_mrna=args.w_cross_mrna,
    )

    model, submission, n_epochs = run_dgat(
        rna_train_path=args.rna_train_path,
        pro_train_path=args.pro_train_path,
        rna_val_path=args.rna_val_path,
        pro_stats_path=args.pro_stats_path,
        cv_split_path=args.cv_split_path,
        out_path=args.out_path,
        n_svd_components=args.n_svd_components,
        protein_pca_variance=args.protein_pca_variance,
        k_spatial=args.k_spatial,
        k_feature=args.k_feature,
        hidden=args.hidden,
        latent_dim=args.latent_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        gat_heads=args.gat_heads,
        feat_attn_heads=args.feat_attn_heads,
        decoder_hidden=args.decoder_hidden,
        branch_hidden=args.branch_hidden,
        loss_weights=loss_weights,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr_encoder=args.lr_encoder,
        lr_decoder=args.lr_decoder,
        encoder_wd=args.encoder_wd,
        max_epochs=args.max_epochs,
        patience=args.patience,
        epoch_finder_fold=args.epoch_finder_fold,
        device=device,
    )

    print(f"\nDone. n_epochs={n_epochs}  submission shape={submission.shape}")


if __name__ == "__main__":
    main()