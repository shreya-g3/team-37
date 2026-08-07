"""
dual_branch_model_v3.py
note:
1. Marker genes concatenated onto SVD node features before either branch.
   - Both VGAE (spatial) and GraphSAGE (expression) branches see marker
     gene values during neighbour aggregation:
       "my neighbour has high CD8A" not "my neighbour has a generally similar expression profile."
   - VGAE reconstruction objective reconstructs marker gene
     signal, so spatial latent space is shaped by protein-relevant
     biology.

3. n_layers set to 2 (GraphSAGE branch). <- can change

Outputs:
    pred_val.csv          submission predictions
    gnn_v5_model.pt       model weights + config
    history.csv           per-epoch metrics during final refit
    svd_model.pkl         fitted TruncatedSVD
    marker_scaler.pkl     fitted StandardScaler for marker gene columns
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
from torch_geometric.nn import SAGEConv, VGAE
from torch_geometric.nn.models import InnerProductDecoder
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("WARNING: faiss not installed. Falling back to sklearn NearestNeighbors.")
    print("Install with: pip install faiss-gpu  (or faiss-cpu)")
    from sklearn.neighbors import NearestNeighbors

from preprocessing_final import inverse_transform_protein
from cv_split_patches import load_cv_split

# Constants
SEED             = 0
N_SVD_COMPONENTS = 256

PRED_WEIGHT      = 0.85
RECON_WEIGHT     = 0.10
KL_BETA_MAX      = 0.05
KL_WARMUP_EPOCHS = 50

# Maps protein panel names → RNA gene names.
# Only genes present in the RNA var_names will be used
# unmatched proteins contribute a zero column so in_dim stays stable.
MARKER_GENE_ALIASES = {
    "synd":        "SDC1",
    "FOXP3":       "FOXP3",
    "CD16":        "FCGR3A",
    "CD31":        "PECAM1",
    "CXCL13":      "CXCL13",
    "Ki67":        "MKI67",
    "OLIG2":       "OLIG2",
    "CXCR5":       "CXCR5",
    "HLA-A":       "HLA-A",
    "PD-L1":       "CD274",
    "PSD95":       "DLG4",
    "CD20":        "MS4A1",
    "CD68":        "CD68",
    "CD44":        "CD44",
    "SMA":         "ACTA2",
    "MSH6":        "MSH6",
    "CD23":        "FCER2",
    "GFAP":        "GFAP",
    "SYNA":        "SYP",
    "Podoplanin":  "PDPN",
    "Vimentin":    "VIM",
    "CD47":        "CD47",
    "CD74":        "CD74",
    "SIRP":        "SIRPA",
    "Granzyme B":  "GZMB",
    "IDH1":        "IDH1",
    "MPO":         "MPO",
    "CD45":        "PTPRC",
    "CD21":        "CR2",
    "FIBR":        "FN1",
    "C-KIT":       "KIT",
    "CD3e":        "CD3E",
    "TOX":         "TOX",
    "PD-1":        "PDCD1",
    "PDGFR":       "PDGFRB",
    "CD4":         "CD4",
    "MAP2":        "MAP2",
    "CD8":         "CD8A",
    "MGMT":        "MGMT",
    "CD38":        "CD38",
    "HLA-DR":      "HLA-DRA",
    "CD14":        "CD14",
    "ICOS":        "ICOS",
    "Granzyme K":  "GZMK",
}


def to_dense(X):
    return np.asarray(X.todense() if hasattr(X, "todense") else X)

# SVD

def reduce_rna_svd(rna_train, rna_val, n_components=N_SVD_COMPONENTS):
    """
    TruncatedSVD on sparse input.
    Fit on train, transform both.
    """
    X_train = rna_train.X if sp.issparse(rna_train.X) else sp.csr_matrix(rna_train.X)
    X_val   = rna_val.X   if sp.issparse(rna_val.X)   else sp.csr_matrix(rna_val.X)

    svd          = TruncatedSVD(n_components=n_components, random_state=SEED)
    X_train_svd  = svd.fit_transform(X_train).astype(np.float32)
    X_val_svd    = svd.transform(X_val).astype(np.float32)
    print(f"  SVD explained variance: {svd.explained_variance_ratio_.sum():.3f}  "
          f"shape: {X_train_svd.shape}")
    return X_train_svd, X_val_svd, svd


# Marker gene extraction

def extract_marker_genes(rna_adata, protein_names, alias_map):
    """
    Slice only the 40 matched gene columns from the sparse .h5ad.
    The full gene matrix is not densified - only this slice

    Returns
    X_marker : float32 ndarray, shape (n_spots, len(protein_names))
        One column per protein. Zero where gene is absent from RNA.
    matched  : list of (protein_name, gene_name) pairs that were found.
    """
    var_names    = list(rna_adata.var_names)
    var_name_set = set(var_names)
    n_spots      = rna_adata.n_obs
    n_proteins   = len(protein_names)

    X_marker = np.zeros((n_spots, n_proteins), dtype=np.float32)
    matched  = []

    for col_idx, prot in enumerate(protein_names):
        # look up via alias map first, then case-insensitive fallback
        gene = alias_map.get(prot)
        if gene is None:
            gene = next((v for v in var_names if v.lower() == prot.lower()), None)
        if gene is not None and gene in var_name_set:
            gene_col = var_names.index(gene)
            # slice single column, stays sparse until .toarray()
            col = rna_adata.X[:, gene_col]
            X_marker[:, col_idx] = (
                np.asarray(col.todense()).flatten()
                if sp.issparse(col)
                else np.asarray(col).flatten()
            )
            matched.append((prot, gene))

    n_matched = len(matched)
    n_missing = n_proteins - n_matched
    print(f"  Marker genes: {n_matched}/{n_proteins} matched  "
          f"({n_missing} proteins have no RNA match → zero column)")
    return X_marker, matched


def build_node_features(X_svd, X_marker, scaler=None):
    """
    Z-score marker columns (fit on train, apply to val/test) then
    concatenate onto SVD features.

    Parameters
    ----------
    X_svd    : (n, 128) float32
    X_marker : (n, 44)  float32
    scaler   : fitted StandardScaler or None (fit here if None)

    Returns
    -------
    X_combined : (n, 128 + 44) float32
    scaler     : fitted StandardScaler (pass to val call)
    """
    if X_marker.shape[1] == 0:
        return X_svd, None

    if scaler is None:
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_marker).astype(np.float32)
    else:
        X_scaled = scaler.transform(X_marker).astype(np.float32)

    X_combined = np.hstack([X_svd, X_scaled])
    print(f"  Node features: {X_svd.shape[1]} SVD + {X_marker.shape[1]} marker "
          f"= {X_combined.shape[1]} dims  "
          f"({X_combined.nbytes / 1e6:.1f} MB)")
    return X_combined, scaler


# Graph construction

def build_spatial_knn_graph(coords, k):
    """
    Exact kNN on 2-D pixel coordinates via cKDTree.
    Feeds VGAE branch.
    """
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)   # k+1 because first hit = self
    n   = coords.shape[0]
    src = np.repeat(np.arange(n), k)
    dst = idx[:, 1:].reshape(-1)
    ei  = np.vstack([src, dst])
    ei  = np.hstack([ei, ei[[1, 0]]])      # symmetrise
    ei  = np.unique(ei, axis=1)
    return torch.tensor(ei, dtype=torch.long)


def build_expression_knn_graph_faiss(X_features, k):
    """
    Approximate kNN in high-dimensional SVD+marker space using faiss.

    Why faiss instead of sklearn:
        sklearn NearestNeighbors falls back to brute-force in >20 dims.
        For 166k nodes × 168 dims, the distance matrix computation
        spikes to 4-8 GB RAM — the OOM this file is designed to prevent.

        faiss.IndexFlatL2 processes the same query in batches internally,
        keeping peak RAM < 500 MB regardless of n_nodes or n_dims.
        With a GPU resource attached it runs on VRAM instead

    Falls back to sklearn if faiss is not installed (triggers a warning
    at import time).
    """
    if not FAISS_AVAILABLE:
        # fallback
        print("  WARNING: using sklearn fallback — may OOM on full dataset")
        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
        nn_model.fit(X_features)
        _, idx = nn_model.kneighbors(X_features)
        n   = X_features.shape[0]
        src = np.repeat(np.arange(n), k)
        dst = idx[:, 1:].reshape(-1)
        ei  = np.vstack([src, dst])
        ei  = np.hstack([ei, ei[[1, 0]]])
        ei  = np.unique(ei, axis=1)
        return torch.tensor(ei, dtype=torch.long)

    n, d  = X_features.shape
    X_f32 = np.ascontiguousarray(X_features, dtype=np.float32)

    # try GPU index first; fall back to CPU if no GPU or VRAM is tight
    index = faiss.IndexFlatL2(d)
    try:
        res   = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        print(f"  faiss: GPU index  ({n} nodes, {d} dims, k={k})")
    except Exception:
        print(f"  faiss: CPU index  ({n} nodes, {d} dims, k={k})")

    index.add(X_f32)
    _, idx = index.search(X_f32, k + 1)   # k+1 → first column is self

    src = np.repeat(np.arange(n), k)
    dst = idx[:, 1:].reshape(-1)
    ei  = np.vstack([src, dst])
    ei  = np.hstack([ei, ei[[1, 0]]])     # symmetrise
    ei  = np.unique(ei, axis=1)
    return torch.tensor(ei, dtype=torch.long)

# Model - in_dim = N_SVD_COMPONENTS + n_marker_genes automatically

class ResidualSAGEBlock(nn.Module):
    """SAGEConv → LayerNorm → ReLU(h + x) residual → Dropout."""
    def __init__(self, hidden, dropout=0.3):
        super().__init__()
        self.conv    = SAGEConv(hidden, hidden)
        self.norm    = nn.LayerNorm(hidden)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = self.norm(h)
        h = F.relu(h + x)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class VGAEEncoder(nn.Module):
    """
    Two-layer GraphSAGE encoder for the VGAE branch (spatial graph).

    in_dim = SVD dims + marker gene dims.
    The encoder aggregates marker gene values from spatial neighbours,
    so the spatial latent space z reflects transcriptomic
    context and local protein-relevant gene expression.
    """
    def __init__(self, in_dim, hidden_dim=256, latent_dim=64):
        super().__init__()
        self.conv1       = SAGEConv(in_dim,     hidden_dim)
        self.conv_mu     = SAGEConv(hidden_dim, latent_dim)
        self.conv_logvar = SAGEConv(hidden_dim, latent_dim)

    def forward(self, x, edge_index):
        h      = F.relu(self.conv1(x, edge_index))
        mu     = self.conv_mu(h, edge_index)
        logvar = self.conv_logvar(h, edge_index)
        return mu, logvar


class DualGraphModel(nn.Module):
    """
    Branch A - VGAE encoder (spatial kNN graph)
        Sees marker genes during spatial neighbour aggregation.
        Produces z: latent spatial embedding per spot.
        Reconstruction loss: can the spatial graph be recovered from z?

    Branch B - residual GraphSAGE (expression kNN graph, built with faiss)
        Sees marker genes during expression-similarity aggregation.
        Spots with similar transcriptomic AND marker gene profiles share
        information, regardless of physical distance in the tissue.
        Produces h: expression-context embedding per spot.

    Fusion: concat([h, z]) → MLP head → 44 protein predictions.
    """

    def __init__(
        self,
        in_dim,          # SVD dims + marker gene dims
        out_dim,
        hidden=256,
        n_layers=2,      # change number of layers later to test
        dropout=0.3,
        vgae_hidden=256,
        vgae_latent=64,
    ):
        super().__init__()

        # Branch A - VGAE on spatial graph
        self.vgae_encoder = VGAEEncoder(in_dim, vgae_hidden, vgae_latent)
        self.vgae         = VGAE(self.vgae_encoder, decoder=InnerProductDecoder())

        # Branch B - residual GraphSAGE on expression graph
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [ResidualSAGEBlock(hidden, dropout) for _ in range(n_layers)]
        )

        # Fusion head
        self.head = nn.Sequential(
            nn.Linear(hidden + vgae_latent, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim),
        )

        self._config = dict(
            in_dim=in_dim, out_dim=out_dim, hidden=hidden, n_layers=n_layers,
            dropout=dropout, vgae_hidden=vgae_hidden, vgae_latent=vgae_latent,
        )

    def forward(self, x, spatial_edge_index, expression_edge_index):
        # Branch A: spatial VGAE
        # x includes marker genes → spatial neighbours share marker signal
        z = self.vgae.encode(x, spatial_edge_index)

        # Branch B: expression GraphSAGE
        # x includes marker genes → expression neighbours share marker signal
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h, expression_edge_index)

        out = self.head(torch.cat([h, z], dim=1))
        return out, z


# Loss

def pearson_loss(yp, yt):
    vp = yp - yp.mean(0, keepdim=True)
    vt = yt - yt.mean(0, keepdim=True)
    r  = (vp * vt).sum(0) / ((vp ** 2).sum(0) * (vt ** 2).sum(0) + 1e-8).sqrt()
    return (1 - r).mean()


def combined_loss(yp, yt, w=0.8):
    return w * F.mse_loss(yp, yt) + (1 - w) * pearson_loss(yp, yt)


def mean_pearson_r(pred, true):
    rs = [pearsonr(pred[:, j], true[:, j])[0] for j in range(true.shape[1])]
    rs = [r for r in rs if not np.isnan(r)]
    return float(np.mean(rs)) if rs else 0.0


def total_loss(pred_loss, recon_loss, kl_loss, epoch,
               warmup_epochs=KL_WARMUP_EPOCHS, beta_max=KL_BETA_MAX,
               pred_weight=PRED_WEIGHT, recon_weight=RECON_WEIGHT):
    beta = beta_max * min(1.0, epoch / warmup_epochs)
    return pred_weight * pred_loss + recon_weight * recon_loss + beta * kl_loss, beta


# LR schedule

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr, min_lr=1e-6):
        self.opt           = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs    = max_epochs
        self.base_lr       = base_lr
        self.min_lr        = min_lr
        self.epoch         = 0

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


# Epoch finder - receives X automatically

def find_n_epochs(X, Y, spatial_edge_index, expression_edge_index,
                  cv_splits, fold, model_cfg, opt_cfg, device,
                  max_epochs, patience, verbose=True):
    split       = cv_splits[fold]
    train_idx   = np.array(split["train"])
    holdout_idx = np.array(split["test"])

    x_t  = torch.tensor(X, dtype=torch.float32, device=device)
    y_t  = torch.tensor(Y, dtype=torch.float32, device=device)
    sp_e = spatial_edge_index.to(device)
    ex_e = expression_edge_index.to(device)

    train_mask   = torch.zeros(X.shape[0], dtype=torch.bool, device=device)
    holdout_mask = torch.zeros(X.shape[0], dtype=torch.bool, device=device)
    train_mask[train_idx]   = True
    holdout_mask[holdout_idx] = True

    model     = DualGraphModel(in_dim=X.shape[1], out_dim=Y.shape[1], **model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"],
                                  weight_decay=opt_cfg["weight_decay"])
    sched     = WarmupCosineScheduler(optimizer, opt_cfg["warmup"], max_epochs, opt_cfg["lr"])

    best_val, best_epoch, patience_ctr = float("inf"), 0, 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        out, z     = model(x_t, sp_e, ex_e)
        pred_loss  = combined_loss(out[train_mask], y_t[train_mask])
        recon_loss = model.vgae.recon_loss(z, sp_e)
        kl_loss    = model.vgae.kl_loss()
        loss, beta = total_loss(pred_loss, recon_loss, kl_loss, epoch)
        loss.backward()
        optimizer.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            out, _ = model(x_t, sp_e, ex_e)
            val_loss = combined_loss(out[holdout_mask], y_t[holdout_mask]).item()

        if val_loss < best_val:
            best_val, best_epoch, patience_ctr = val_loss, epoch, 0
        else:
            patience_ctr += 1

        if verbose and (epoch % 10 == 0 or epoch == max_epochs - 1):
            print(f"    [epoch-finder] {epoch:4d}  "
                  f"train={loss.item():.4f}  holdout={val_loss:.4f}  beta={beta:.3f}")

        if patience_ctr >= patience:
            print(f"    [epoch-finder] early stop epoch {epoch}  best={best_epoch}")
            break

    print(f"    [epoch-finder] → n_epochs = {best_epoch + 1}")
    return best_epoch + 1


# refit

def train_model(X, Y, spatial_edge_index, expression_edge_index,
                n_epochs, model_cfg, opt_cfg, device, verbose=True):
    x_t  = torch.tensor(X, dtype=torch.float32, device=device)
    y_t  = torch.tensor(Y, dtype=torch.float32, device=device)
    sp_e = spatial_edge_index.to(device)
    ex_e = expression_edge_index.to(device)

    model     = DualGraphModel(in_dim=X.shape[1], out_dim=Y.shape[1], **model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"],
                                  weight_decay=opt_cfg["weight_decay"])
    sched     = WarmupCosineScheduler(optimizer, opt_cfg["warmup"], n_epochs, opt_cfg["lr"])

    history = []
    model.train()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out, z     = model(x_t, sp_e, ex_e)
        pred_loss  = combined_loss(out, y_t)
        recon_loss = model.vgae.recon_loss(z, sp_e)
        kl_loss    = model.vgae.kl_loss()
        loss, beta = total_loss(pred_loss, recon_loss, kl_loss, epoch)
        loss.backward()
        optimizer.step()
        lr_now = sched.step()

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                pred_np, _ = model(x_t, sp_e, ex_e)
            train_r = mean_pearson_r(pred_np.cpu().numpy(), Y)
            model.train()
            history.append({
                "epoch": epoch, "total_loss": loss.item(),
                "pred_loss": pred_loss.item(), "recon_loss": recon_loss.item(),
                "kl_loss": kl_loss.item(), "beta": beta,
                "train_pearson": train_r, "lr": lr_now,
            })
            if verbose:
                print(f"  epoch {epoch:4d}  total={loss.item():.4f}  "
                      f"pred={pred_loss.item():.4f}  recon={recon_loss.item():.4f}  "
                      f"kl={kl_loss.item():.4f}  beta={beta:.3f}  r={train_r:.4f}")

    return model, pd.DataFrame(history)


def save_model(model, path):
    torch.save({"state_dict": model.state_dict(), "config": model._config}, path)


def load_model(path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(path, map_location=device)
    model  = DualGraphModel(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def predict(model, X_val, spatial_ei_val, expression_ei_val, device):
    model.eval()
    with torch.no_grad():
        out, _ = model(
            torch.tensor(X_val, dtype=torch.float32, device=device),
            spatial_ei_val.to(device),
            expression_ei_val.to(device),
        )
    return out.cpu().numpy()

# Main entry point

def run_gnn_v5(
    rna_train_path, pro_train_path, rna_val_path,
    pro_stats_path, cv_split_path, out_path,
    n_components=N_SVD_COMPONENTS,
    k_spatial=8, k_expression=8,
    hidden=256, n_layers=2, dropout=0.3,
    lr=3e-4, weight_decay=1e-3, warmup=10,
    max_epochs=1000, patience=30,
    epoch_finder_fold=1, device=None,
):
    os.makedirs(out_path, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Device: {device}")
    print(f"faiss available: {FAISS_AVAILABLE}")

    # Load data
    print("\n--- Loading data ---")
    rna_train = ad.read_h5ad(rna_train_path)
    rna_val   = ad.read_h5ad(rna_val_path)
    pro_train = ad.read_h5ad(pro_train_path)

    with open(pro_stats_path, "rb") as f:
        protein_stats = pickle.load(f)

    barcodes_val  = rna_val.obs.index.values
    coords_train  = rna_train.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values.astype(np.float32)
    coords_val    = rna_val.obs[  ["pxl_row_in_fullres", "pxl_col_in_fullres"]].values.astype(np.float32)
    protein_names = list(pro_train.var_names)
    Y_train       = to_dense(pro_train.X).astype(np.float32)

    print(f"  RNA train: {rna_train.shape}  RNA val: {rna_val.shape}")
    print(f"  Proteins:  {len(protein_names)}")

    # Step 1 — SVD (sparse → compact, never densifies full gene matrix)
    # ------------------------------------------------------------------
    print("\n--- TruncatedSVD ---")
    X_train_svd, X_val_svd, svd = reduce_rna_svd(rna_train, rna_val, n_components)
    with open(os.path.join(out_path, "svd_model.pkl"), "wb") as f:
        pickle.dump(svd, f)

    # Step 2 — Marker gene extraction
    # ------------------------------------------------------------------
    print("\n--- Marker gene extraction ---")
    X_marker_train, matched = extract_marker_genes(rna_train, protein_names, MARKER_GENE_ALIASES)
    X_marker_val,   _       = extract_marker_genes(rna_val,   protein_names, MARKER_GENE_ALIASES)

    # Step 3 — Build combined node features: SVD + z-scored marker genes
    # ------------------------------------------------------------------
    print("\n--- Building node features ---")
    X_train, marker_scaler = build_node_features(X_train_svd, X_marker_train, scaler=None)
    X_val,   _             = build_node_features(X_val_svd,   X_marker_val,   scaler=marker_scaler)

    with open(os.path.join(out_path, "marker_scaler.pkl"), "wb") as f:
        pickle.dump({"scaler": marker_scaler, "matched": matched}, f)

    # in_dim = (n_components + n_marker_gene_cols)
    in_dim = X_train.shape[1]
    print(f"  Final in_dim: {in_dim}  ({n_components} SVD + {in_dim - n_components} marker)")

    # Step 4 — Build 2 graphs
    # ------------------------------------------------------------------
    print("\n--- Building spatial kNN graphs (cKDTree, 2-D coords) ---")
    spatial_ei_train = build_spatial_knn_graph(coords_train, k=k_spatial)
    spatial_ei_val   = build_spatial_knn_graph(coords_val,   k=k_spatial)
    print(f"  Train edges: {spatial_ei_train.shape[1]:,}  "
          f"Val edges: {spatial_ei_val.shape[1]:,}")

    print("\n--- Building expression kNN graphs (faiss, SVD+marker features) ---")
    expression_ei_train = build_expression_knn_graph_faiss(X_train, k=k_expression)
    expression_ei_val   = build_expression_knn_graph_faiss(X_val,   k=k_expression)
    print(f"  Train edges: {expression_ei_train.shape[1]:,}  "
          f"Val edges: {expression_ei_val.shape[1]:,}")

    # Step 5 — Find optimal epoch count via one buffered CV fold
    # ------------------------------------------------------------------
    model_cfg = dict(
        hidden=hidden, n_layers=n_layers, dropout=dropout,
        vgae_hidden=256, vgae_latent=64,
    )
    opt_cfg = dict(lr=lr, weight_decay=weight_decay, warmup=warmup)

    cv_splits = load_cv_split(cv_split_path)
    print(f"\n--- Epoch finder (fold {epoch_finder_fold}) ---")
    n_epochs = find_n_epochs(
        X_train, Y_train,
        spatial_ei_train, expression_ei_train,
        cv_splits, epoch_finder_fold,
        model_cfg, opt_cfg, device, max_epochs, patience,
    )

    # Step 6 — Refit on all training data
    # ------------------------------------------------------------------
    print(f"\n--- Final refit  (n_epochs={n_epochs}) ---")
    model, history = train_model(
        X_train, Y_train,
        spatial_ei_train, expression_ei_train,
        n_epochs, model_cfg, opt_cfg, device,
    )
    save_model(model, os.path.join(out_path, "gnn_v5_model.pt"))
    history.to_csv(os.path.join(out_path, "history.csv"), index=False)

    # Step 7 — Inference and inverse transform
    # ------------------------------------------------------------------
    print("\n--- Inference on validation ---")
    Z_pred = predict(model, X_val, spatial_ei_val, expression_ei_val, device)
    X_pred = inverse_transform_protein(Z_pred, protein_stats)
    X_pred = np.clip(X_pred, a_min=0, a_max=None)

    submission = pd.DataFrame(X_pred, columns=protein_names)
    submission.insert(0, "pxl_col_in_fullres", coords_val[:, 1].astype(int))
    submission.insert(0, "pxl_row_in_fullres", coords_val[:, 0].astype(int))
    submission.insert(0, "barcode", barcodes_val)

    out_csv = os.path.join(out_path, "pred_val.csv")
    submission.to_csv(out_csv, index=False)
    print(f"  Saved {out_csv}  shape={submission.shape}")

    return model, submission, n_epochs