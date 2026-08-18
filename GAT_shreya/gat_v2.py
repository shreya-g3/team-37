"""

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
from torch_geometric.nn import SAGEConv, VGAE, GATv2Conv
from torch_geometric.nn.models import InnerProductDecoder
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import faiss
import scanpy as sc

from preprocessing_final import inverse_transform_protein
from cv_split_patches import load_cv_split

# ---------------------------------------------------------
# Global constants
# ---------------------------------------------------------
SEED             = 0
N_SVD_COMPONENTS = 128   # kept at 128 — pre-projection handles GAT memory
GAT_PROJ_DIM     = 128   # project node features to this before GATv2
GAT_HEADS        = 4     # 4 heads × 32-dim = 128-dim output per layer
GAT_OUT_PER_HEAD = 32    # output dim per head in GATv2 layers

PRED_WEIGHT      = 0.85
RECON_WEIGHT     = 0.10
KL_BETA_MAX      = 0.05
KL_WARMUP_EPOCHS = 50

LOG_INTERVAL     = 10    # print metrics every N epochs

MARKER_GENE_ALIASES = {
    "synd":        "SDC1",    "FOXP3":       "FOXP3",
    "CD16":        "FCGR3A",  "CD31":        "PECAM1",
    "CXCL13":      "CXCL13",  "Ki67":        "MKI67",
    "OLIG2":       "OLIG2",   "CXCR5":       "CXCR5",
    "HLA-A":       "HLA-A",   "PD-L1":       "CD274",
    "PSD95":       "DLG4",    "CD20":        "MS4A1",
    "CD68":        "CD68",    "CD44":        "CD44",
    "SMA":         "ACTA2",   "MSH6":        "MSH6",
    "CD23":        "FCER2",   "GFAP":        "GFAP",
    "SYNA":        "SYP",     "Podoplanin":  "PDPN",
    "Vimentin":    "VIM",     "CD47":        "CD47",
    "CD74":        "CD74",    "SIRP":        "SIRPA",
    "Granzyme B":  "GZMB",   "IDH1":        "IDH1",
    "MPO":         "MPO",     "CD45":        "PTPRC",
    "CD21":        "CR2",     "FIBR":        "FN1",
    "C-KIT":       "KIT",     "CD3e":        "CD3E",
    "TOX":         "TOX",     "PD-1":        "PDCD1",
    "PDGFR":       "PDGFRB",  "CD4":         "CD4",
    "MAP2":        "MAP2",    "CD8":         "CD8A",
    "MGMT":        "MGMT",    "CD38":        "CD38",
    "HLA-DR":      "HLA-DRA", "CD14":        "CD14",
    "ICOS":        "ICOS",    "Granzyme K":  "GZMK",
}


def to_dense(X):
    return np.asarray(X.todense() if hasattr(X, "todense") else X)

# ---------------------------------------------------------
# Marker gene extraction
# Sparse column slice — 166k x 44 x float32 ≈ 24 MB
# ---------------------------------------------------------

def extract_marker_genes(rna_adata, protein_names, alias_map):
    """
    Pull ~40 matched gene columns from sparse .h5ad.
    Full gene matrix is NEVER densified — only the matched slice is.
    Unmatched proteins receive a zero column so output shape is stable.
    """
    var_names    = list(rna_adata.var_names)
    var_name_set = set(var_names)
    X_marker     = np.zeros((rna_adata.n_obs, len(protein_names)), dtype=np.float32)
    matched      = []

    for col_idx, prot in enumerate(protein_names):
        gene = alias_map.get(prot) or next(
            (v for v in var_names if v.lower() == prot.lower()), None
        )
        if gene and gene in var_name_set:
            col = rna_adata.X[:, var_names.index(gene)]
            X_marker[:, col_idx] = (
                np.asarray(col.todense()).flatten()
                if sp.issparse(col)
                else np.asarray(col).flatten()
            )
            matched.append((prot, gene))

    print(f"  Marker genes matched: {len(matched)}/{len(protein_names)}")
    return X_marker, matched


def build_node_features(X_svd, X_marker, scaler=None):
    """
    Z-score marker genes (fit on train, apply to val) then
    concatenate onto SVD features.
    Final node feature dim = N_SVD_COMPONENTS + n_marker_cols.
    """
    if X_marker.shape[1] == 0:
        return X_svd, None
    if scaler is None:
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_marker).astype(np.float32)
    else:
        X_scaled = scaler.transform(X_marker).astype(np.float32)
    X_combined = np.hstack([X_svd, X_scaled])
    print(f"  Node features: {X_svd.shape[1]} SVD + "
          f"{X_marker.shape[1]} marker = {X_combined.shape[1]} dims  "
          f"({X_combined.nbytes/1e6:.1f} MB)")
    return X_combined, scaler


# ---------------------------------------------------------
# Graph construction — both graphs return (edge_index, edge_weight)
# ---------------------------------------------------------

def build_spatial_knn_graph(coords, k):
    """
    Exact kNN on 2-D pixel coordinates via cKDTree.
    Returns symmetric edge_index and inverse-distance edge weights.
    Weights are passed as edge_attr into GATv2Conv so the attention
    mechanism can bias towards physically closer neighbours.
    """
    tree       = cKDTree(coords)
    dists, idx = tree.query(coords, k=k + 1)
    n   = coords.shape[0]
    src = np.repeat(np.arange(n), k)
    dst = idx[:, 1:].reshape(-1)
    d   = dists[:, 1:].reshape(-1)
    w   = (1.0 / (1.0 + d)).astype(np.float32)
    ei  = np.vstack([src, dst])
    ei  = np.hstack([ei, ei[[1, 0]]])
    w   = np.hstack([w, w])
    print(f"  Spatial graph: {ei.shape[1]:,} edges  "
          f"(k={k}, {n:,} nodes)")
    return torch.tensor(ei, dtype=torch.long), torch.tensor(w, dtype=torch.float32)


def build_expression_knn_graph_faiss(X_features, k):
    """
    Approximate kNN in SVD+marker feature space using faiss GPU.
    Returns symmetric edge_index and inverse-L2-distance edge weights.
    Feeds the SAGEConv expression branch.
    """
    n, d  = X_features.shape
    X_f32 = np.ascontiguousarray(X_features, dtype=np.float32)

    index = faiss.IndexFlatL2(d)
    try:
        res   = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        print(f"  Expression graph: faiss GPU  ({n:,} nodes, {d} dims, k={k})")
    except Exception:
        print(f"  Expression graph: faiss CPU  ({n:,} nodes, {d} dims, k={k})")

    index.add(X_f32)
    dists_sq, idx = index.search(X_f32, k + 1)
    dists = np.sqrt(np.clip(dists_sq[:, 1:], 0, None))
    idx   = idx[:, 1:]

    src = np.repeat(np.arange(n), k)
    dst = idx.reshape(-1)
    d   = dists.reshape(-1)
    w   = (1.0 / (1.0 + d)).astype(np.float32)
    ei  = np.vstack([src, dst])
    ei  = np.hstack([ei, ei[[1, 0]]])
    w   = np.hstack([w, w])
    print(f"  Expression graph: {ei.shape[1]:,} edges")
    return torch.tensor(ei, dtype=torch.long), torch.tensor(w, dtype=torch.float32)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

class GATv2Encoder(nn.Module):
    """
    Spatial VGAE encoder using GATv2Conv.

    Memory analysis (why pre-projection is essential):
        GATv2 materialises [hᵢ || hⱼ] for every edge.
        With raw node features (172 dims) and 2.65M edges:
            2.65M × 344 × 4 bytes × 4 heads ≈ 14 GB  ← OOM
        With pre-projection to 128 dims:
            2.65M × 256 × 4 bytes × 4 heads ≈ 10 GB  ← still tight
        With pre-projection to GAT_PROJ_DIM=64:
            2.65M × 128 × 4 bytes × 4 heads ≈ 5 GB   ← safe

    The pre-projection is a simple linear layer applied to all node
    features before any graph operation — no edges involved,
    no memory spike.

    Distance weights are passed as edge_attr (shape [E, 1]) so
    GATv2Conv incorporates them into the attention score computation:
        e(i,j) = a · W·[hᵢ || hⱼ || dist_weight(i,j)]
    This lets the model learn whether physical proximity matters for
    each attention head independently.
    """

    def __init__(self, in_dim,
                 proj_dim=GAT_PROJ_DIM,
                 heads=GAT_HEADS,
                 out_per_head=GAT_OUT_PER_HEAD,
                 dropout=0.3):
        super().__init__()

        # Pre-projection: compress node features before GAT
        # This is the key memory-saving step
        self.pre_proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ELU(),
        )

        gat_hidden = heads * out_per_head   # e.g. 4 × 32 = 128

        # Layer 1: proj_dim → gat_hidden (concat heads)
        # edge_dim=1 tells GATv2Conv to include the distance weight
        # as an additional feature in the attention computation
        self.gat1 = GATv2Conv(
            in_channels=proj_dim,
            out_channels=out_per_head,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=1,        # one scalar edge weight per edge
        )
        self.norm1 = nn.LayerNorm(gat_hidden)

        # Layer 2: gat_hidden → latent (single head, no concat)
        latent_dim = gat_hidden   # keep latent = gat_hidden for mu/logvar

        self.gat_mu = GATv2Conv(
            in_channels=gat_hidden,
            out_channels=latent_dim // heads,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=1,
        )
        self.gat_logvar = GATv2Conv(
            in_channels=gat_hidden,
            out_channels=latent_dim // heads,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=1,
        )

        self.latent_dim = latent_dim
        self.dropout    = dropout

    def forward(self, x, edge_index, edge_attr):
        # edge_attr: [E] → reshape to [E, 1] for GATv2Conv edge_dim
        ea = edge_attr.unsqueeze(1) if edge_attr.dim() == 1 else edge_attr

        # pre-project: no graph op, no memory spike
        h = self.pre_proj(x)

        # GAT layer 1
        h = self.gat1(h, edge_index, edge_attr=ea)
        h = self.norm1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # mu and logvar — both use attention on the layer-1 output
        mu     = self.gat_mu(h, edge_index, edge_attr=ea)
        logvar = self.gat_logvar(h, edge_index, edge_attr=ea)
        return mu, logvar


class ResidualSAGEBlock(nn.Module):
    """SAGEConv → LayerNorm → ELU(h + x) residual → Dropout."""
    def __init__(self, hidden, dropout=0.3):
        super().__init__()
        self.conv    = SAGEConv(hidden, hidden)
        self.norm    = nn.LayerNorm(hidden)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = self.norm(h)
        h = F.elu(h + x)
        return F.dropout(h, p=self.dropout, training=self.training)


class DualGraphModel(nn.Module):
    """
    Branch A — GATv2 VGAE on spatial kNN graph (distance-weighted attention)
    Branch B — Residual SAGEConv on expression kNN graph (distance-weighted pre-agg)
    Fusion   — concat([h_sage, z_vgae]) → MLP → 44 protein predictions

    in_dim is set automatically from the node feature matrix passed in.
    """

    def __init__(self, in_dim, out_dim,
                 sage_hidden=256,
                 n_sage_layers=2,
                 dropout=0.3,
                 gat_proj_dim=GAT_PROJ_DIM,
                 gat_heads=GAT_HEADS,
                 gat_out_per_head=GAT_OUT_PER_HEAD):
        super().__init__()

        gat_latent = gat_heads * gat_out_per_head   # VGAE latent dim

        # Branch A — GATv2 VGAE (spatial, distance-weighted attention)
        self.vgae_encoder = GATv2Encoder(
            in_dim=in_dim,
            proj_dim=gat_proj_dim,
            heads=gat_heads,
            out_per_head=gat_out_per_head,
            dropout=dropout,
        )
        self.vgae = VGAE(self.vgae_encoder, decoder=InnerProductDecoder())

        # Branch B — Residual SAGEConv (expression, distance pre-aggregation)
        # WeightedMessagePassing pre-aggregates using distance weights
        # before the SAGEConv mean-aggregation.
        # FIX: input_proj now takes [own_features || weighted_neighbour_avg]
        # (2 * in_dim), not just the neighbour average alone — the node's
        # own raw signal was previously discarded entirely before ever
        # reaching this branch (see forward() below).
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim * 2, sage_hidden),
            nn.LayerNorm(sage_hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.sage_blocks = nn.ModuleList(
            [ResidualSAGEBlock(sage_hidden, dropout) for _ in range(n_sage_layers)]
        )

        # Fusion MLP
        fusion_in = sage_hidden + gat_latent
        self.head = nn.Sequential(
            nn.Linear(fusion_in, sage_hidden // 2),
            nn.LayerNorm(sage_hidden // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(sage_hidden // 2, out_dim),
        )

        self._config = dict(
            in_dim=in_dim, out_dim=out_dim,
            sage_hidden=sage_hidden, n_sage_layers=n_sage_layers,
            dropout=dropout, gat_proj_dim=gat_proj_dim,
            gat_heads=gat_heads, gat_out_per_head=gat_out_per_head,
        )

    def forward(self, x,
                spatial_ei, spatial_ew,
                expr_ei,    expr_ew):
        # Branch A: GATv2 spatial VGAE
        # edge_attr (spatial_ew) is fed into GATv2Conv attention
        # encode with edge_attr passed correctly to GATv2Encoder
        mu, logvar = self.vgae_encoder(x, spatial_ei, spatial_ew)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        # store on vgae so recon_loss() and kl_loss() still work correctly
        self.vgae.__mu__ = mu
        self.vgae.__logvar__ = logvar

        # Branch B: expression SAGEConv
        # FIX: previously x_w REPLACED x entirely, so this branch never
        # saw a node's own untransformed features — only a neighbour-
        # smoothed version, compounded again by SAGEConv's own internal
        # aggregation. Now both own features AND the distance-weighted
        # neighbour average are concatenated, so the branch has access
        # to both signals (matches how the RF pipeline used neighbour
        # features as an ADDITION, never a replacement).
        x_w = self._weighted_agg(x, expr_ei, expr_ew)
        x_combined = torch.cat([x, x_w], dim=1)
        h   = self.input_proj(x_combined)
        for blk in self.sage_blocks:
            h = blk(h, expr_ei)

        return self.head(torch.cat([h, z], dim=1)), z

    @staticmethod
    def _weighted_agg(x, edge_index, edge_weight):
        """Weighted mean of neighbour features before SAGEConv."""
        src, dst = edge_index
        weighted = x[dst] * edge_weight.unsqueeze(1)
        out      = torch.zeros_like(x)
        out.scatter_add_(0, src.unsqueeze(1).expand_as(weighted), weighted)
        wsum = torch.zeros(x.shape[0], 1, device=x.device)
        wsum.scatter_add_(0, src.unsqueeze(1), edge_weight.unsqueeze(1))
        return out / wsum.clamp(min=1e-8)


# ---------------------------------------------------------
# Loss functions
# ---------------------------------------------------------

def pearson_loss(yp, yt):
    vp = yp - yp.mean(0, keepdim=True)
    vt = yt - yt.mean(0, keepdim=True)
    r  = (vp * vt).sum(0) / ((vp**2).sum(0) * (vt**2).sum(0) + 1e-8).sqrt()
    return (1 - r).mean()


def combined_loss(yp, yt, w=0.8):
    return w * F.mse_loss(yp, yt) + (1 - w) * pearson_loss(yp, yt)


def total_loss(pred_loss, recon_loss, kl_loss, epoch,
               warmup=KL_WARMUP_EPOCHS, beta_max=KL_BETA_MAX,
               pw=PRED_WEIGHT, rw=RECON_WEIGHT):
    beta = beta_max * min(1.0, epoch / max(warmup, 1))
    return pw * pred_loss + rw * recon_loss + beta * kl_loss, beta


def compute_r2(pred_np, true_np):
    """
    Per-protein R² then mean and median.
    """
    r2_vals = r2_score(true_np, pred_np, multioutput="raw_values")
    return float(r2_vals.mean()), float(np.median(r2_vals)), r2_vals


def mean_pearson_r(pred, true):
    rs = [pearsonr(pred[:, j], true[:, j])[0] for j in range(true.shape[1])]
    rs = [r for r in rs if not np.isnan(r)]
    return float(np.mean(rs)) if rs else 0.0


# ---------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------

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
            p  = (e - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * p))
        for pg in self.opt.param_groups:
            pg["lr"] = lr
        return lr


# ---------------------------------------------------------
# Training helpers
# ---------------------------------------------------------

def _forward_eval(model, x_t, sp_ei, sp_ew, ex_ei, ex_ew):
    """Run model in eval mode, return numpy predictions."""
    model.eval()
    with torch.no_grad():
        pred, _ = model(x_t, sp_ei, sp_ew, ex_ei, ex_ew)
    return pred.cpu().numpy()


def _log_metrics(epoch, loss, pl, rl, kl, beta, lr,
                 pred_np, true_np, mask_np=None, prefix="train"):
    """
    Compute and print R², mean Pearson r, and loss components.
    mask_np: boolean array - if provided, evaluate on masked rows only.
    """
    p = pred_np[mask_np] if mask_np is not None else pred_np
    t = true_np[mask_np] if mask_np is not None else true_np
    mean_r2, med_r2, _ = compute_r2(p, t)
    pear_r              = mean_pearson_r(p, t)
    print(
        f"  epoch {epoch:4d}  "
        f"total={loss:.4f}  pred={pl:.4f}  recon={rl:.4f}  "
        f"kl={kl:.4f}  beta={beta:.3f}  lr={lr:.2e}  "
        f"[{prefix}] mean_R²={mean_r2:.4f}  med_R²={med_r2:.4f}  "
        f"pearson_r={pear_r:.4f}"
    )
    return mean_r2, med_r2


# ---------------------------------------------------------
# Epoch finder — uses one CV fold, early stop on val loss
# Prints R² on held-out fold every LOG_INTERVAL epochs
# ---------------------------------------------------------

def find_n_epochs(X, Y, coords, k_spatial, k_expression,
                  cv_splits, fold, model_cfg, opt_cfg,
                  device, max_epochs, patience,
                  log_interval=LOG_INTERVAL, verbose=True):
    """
    """
    split   = cv_splits[fold]
    tr_idx  = np.array(split["train"])
    ho_idx  = np.array(split["test"])

    X_tr, X_ho = X[tr_idx], X[ho_idx]
    Y_tr, Y_ho = Y[tr_idx], Y[ho_idx]
    coords_tr, coords_ho = coords[tr_idx], coords[ho_idx]

    print(f"  [epoch-finder] fold {fold}: building fold-local graphs "
          f"(train={len(tr_idx):,}  holdout={len(ho_idx):,}, no cross-edges)")

    sp_ei_tr, sp_ew_tr = build_spatial_knn_graph(coords_tr, k=k_spatial)
    sp_ei_ho, sp_ew_ho = build_spatial_knn_graph(coords_ho, k=k_spatial)
    ex_ei_tr, ex_ew_tr = build_expression_knn_graph_faiss(X_tr, k=k_expression)
    ex_ei_ho, ex_ew_ho = build_expression_knn_graph_faiss(X_ho, k=k_expression)

    x_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(Y_tr, dtype=torch.float32, device=device)
    x_ho_t = torch.tensor(X_ho, dtype=torch.float32, device=device)
    y_ho_t = torch.tensor(Y_ho, dtype=torch.float32, device=device)

    sp_ei_tr_d = sp_ei_tr.to(device); sp_ew_tr_d = sp_ew_tr.to(device)
    ex_ei_tr_d = ex_ei_tr.to(device); ex_ew_tr_d = ex_ew_tr.to(device)
    sp_ei_ho_d = sp_ei_ho.to(device); sp_ew_ho_d = sp_ew_ho.to(device)
    ex_ei_ho_d = ex_ei_ho.to(device); ex_ew_ho_d = ex_ew_ho.to(device)

    model = DualGraphModel(in_dim=X.shape[1], out_dim=Y.shape[1],
                           **model_cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"],
                               weight_decay=opt_cfg["weight_decay"])
    sched = WarmupCosineScheduler(opt, opt_cfg["warmup"],
                                  max_epochs, opt_cfg["lr"])

    best_val = float("inf"); best_epoch = 0; ctr = 0

    for epoch in range(max_epochs):
        # --- train step: TRAIN graph only ---
        model.train()
        opt.zero_grad()
        out, z    = model(x_tr_t, sp_ei_tr_d, sp_ew_tr_d, ex_ei_tr_d, ex_ew_tr_d)
        pl        = combined_loss(out, y_tr_t)
        rl        = model.vgae.recon_loss(z, sp_ei_tr_d)
        kl        = model.vgae.kl_loss()
        loss, beta = total_loss(pl, rl, kl, epoch)
        loss.backward()
        opt.step()
        lr_now = sched.step()

        # --- eval step: HOLDOUT graph only — zero connection to train ---
        model.eval()
        with torch.no_grad():
            out_ho, _ = model(x_ho_t, sp_ei_ho_d, sp_ew_ho_d, ex_ei_ho_d, ex_ew_ho_d)
            val_loss  = combined_loss(out_ho, y_ho_t).item()

        if val_loss < best_val:
            best_val = val_loss; best_epoch = epoch; ctr = 0
        else:
            ctr += 1

        # --- logging with R² ---
        if verbose and (epoch % log_interval == 0 or epoch == max_epochs - 1):
            pred_np = out_ho.cpu().numpy()
            _log_metrics(epoch, loss.item(), pl.item(), rl.item(), kl.item(),
                         beta, lr_now, pred_np, Y_ho, mask_np=None, prefix="holdout")

        if ctr >= patience:
            print(f"  [epoch-finder] early stop at epoch {epoch}  "
                  f"best_epoch={best_epoch}  best_val={best_val:.4f}")
            break

    print(f"  [epoch-finder] -> n_epochs = {best_epoch + 1}")
    return best_epoch + 1


# ---------------------------------------------------------
# Final refit on all training data
# Prints R² on full training set every LOG_INTERVAL epochs
# ---------------------------------------------------------

def train_model(X, Y, sp_ei, sp_ew, ex_ei, ex_ew,
                n_epochs, model_cfg, opt_cfg, device,
                log_interval=LOG_INTERVAL, verbose=True):

    Y_np   = Y
    x_t    = torch.tensor(X, dtype=torch.float32, device=device)
    y_t    = torch.tensor(Y, dtype=torch.float32, device=device)
    sp_ei_d = sp_ei.to(device); sp_ew_d = sp_ew.to(device)
    ex_ei_d = ex_ei.to(device); ex_ew_d = ex_ew.to(device)

    model = DualGraphModel(in_dim=X.shape[1], out_dim=Y.shape[1],
                           **model_cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"],
                               weight_decay=opt_cfg["weight_decay"])
    sched = WarmupCosineScheduler(opt, opt_cfg["warmup"],
                                  n_epochs, opt_cfg["lr"])
    history = []

    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad()
        out, z     = model(x_t, sp_ei_d, sp_ew_d, ex_ei_d, ex_ew_d)
        pl         = combined_loss(out, y_t)
        rl         = model.vgae.recon_loss(z, sp_ei_d)
        kl         = model.vgae.kl_loss()
        loss, beta = total_loss(pl, rl, kl, epoch)
        loss.backward()
        opt.step()
        lr_now = sched.step()

        if verbose and (epoch % log_interval == 0 or epoch == n_epochs - 1):
            pred_np         = _forward_eval(model, x_t, sp_ei_d, sp_ew_d,
                                            ex_ei_d, ex_ew_d)
            mean_r2, med_r2 = _log_metrics(
                epoch, loss.item(), pl.item(), rl.item(), kl.item(),
                beta, lr_now, pred_np, Y_np, prefix="train"
            )
            pear_r = mean_pearson_r(pred_np, Y_np)
            history.append({
                "epoch": epoch, "total_loss": loss.item(),
                "pred_loss": pl.item(), "recon_loss": rl.item(),
                "kl_loss": kl.item(), "beta": beta,
                "mean_r2": mean_r2, "median_r2": med_r2,
                "train_pearson": pear_r, "lr": lr_now,
            })
            model.train()   # restore train mode after eval

    return model, pd.DataFrame(history)


# ---------------------------------------------------------
# Model persistence
# ---------------------------------------------------------

def save_model(model, path):
    torch.save({"state_dict": model.state_dict(),
                "config":     model._config}, path)


def load_model(path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(path, map_location=device)
    model  = DualGraphModel(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def predict(model, X_val, sp_ei, sp_ew, ex_ei, ex_ew, device):
    return _forward_eval(
        model,
        torch.tensor(X_val, dtype=torch.float32, device=device),
        sp_ei.to(device), sp_ew.to(device),
        ex_ei.to(device), ex_ew.to(device),
    )


# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------

def run_gnn_v5(
    rna_train_path, pro_train_path, rna_val_path,
    pro_stats_path, cv_split_path, out_path,
    n_components=N_SVD_COMPONENTS,
    k_spatial=8, k_expression=8,
    hidden=256, n_layers=2, dropout=0.1,
    gat_proj_dim=GAT_PROJ_DIM,
    gat_heads=GAT_HEADS,
    gat_out_per_head=GAT_OUT_PER_HEAD,
    lr=3e-4, weight_decay=1e-3, warmup=10,
    max_epochs=1000, patience=30,
    epoch_finder_fold=1, device=None,
    log_interval=LOG_INTERVAL,
):
    os.makedirs(out_path, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"Device: {device}")

    print("\n--- Load ---")
    rna_train = ad.read_h5ad(rna_train_path)
    rna_val = ad.read_h5ad(rna_val_path)
    pro_train = ad.read_h5ad(pro_train_path)
    with open(pro_stats_path, "rb") as f:
        protein_stats = pickle.load(f)

    barcodes_val = rna_val.obs.index.values
    coords_train = rna_train.obs[["pxl_row_in_fullres",
                                  "pxl_col_in_fullres"]].values.astype(np.float32)
    coords_val = rna_val.obs[["pxl_row_in_fullres",
                              "pxl_col_in_fullres"]].values.astype(np.float32)
    protein_names = list(pro_train.var_names)
    Y_train = to_dense(pro_train.X).astype(np.float32)
    print(f"  Train RNA: {rna_train.shape}  Val RNA: {rna_val.shape}  "
          f"Proteins: {len(protein_names)}")

    print("\n--- SVD ---")
    X_train_sp = rna_train.X if sp.issparse(rna_train.X) else sp.csr_matrix(rna_train.X)
    X_val_sp = rna_val.X if sp.issparse(rna_val.X) else sp.csr_matrix(rna_val.X)
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    X_train_svd = svd.fit_transform(X_train_sp).astype(np.float32)
    X_val_svd = svd.transform(X_val_sp).astype(np.float32)
    exp_var = svd.explained_variance_ratio_.sum()
    print(f"  SVD: {n_components} components, "
          f"explained variance = {exp_var * 100:.1f}%  "
          f"({X_train_svd.nbytes / 1e6:.0f} MB train)")
    with open(os.path.join(out_path, "svd_model.pkl"), "wb") as f:
        pickle.dump(svd, f)

    # Marker genes
    print("\n--- Marker genes ---")
    X_m_train, matched = extract_marker_genes(rna_train, protein_names, MARKER_GENE_ALIASES)
    X_m_val,   _       = extract_marker_genes(rna_val,   protein_names, MARKER_GENE_ALIASES)

    # Node features: SVD + marker genes
    print("\n--- Node features ---")
    X_train, mscaler = build_node_features(X_train_svd, X_m_train)
    X_val,   _       = build_node_features(X_val_svd,   X_m_val, scaler=mscaler)
    with open(os.path.join(out_path, "marker_scaler.pkl"), "wb") as f:
        pickle.dump({"scaler": mscaler, "matched": matched}, f)

    in_dim = X_train.shape[1]
    print(f"  in_dim = {in_dim}  "
          f"({n_components} SVD + {in_dim - n_components} marker)")

    # Graphs
    print("\n--- Spatial graph ---")
    sp_ei_tr,  sp_ew_tr  = build_spatial_knn_graph(coords_train, k=k_spatial)
    sp_ei_val, sp_ew_val = build_spatial_knn_graph(coords_val,   k=k_spatial)

    print("\n--- Expression graph ---")
    ex_ei_tr,  ex_ew_tr  = build_expression_knn_graph_faiss(X_train, k=k_expression)
    ex_ei_val, ex_ew_val = build_expression_knn_graph_faiss(X_val,   k=k_expression)

    model_cfg = dict(
        sage_hidden=hidden,
        n_sage_layers=n_layers,
        dropout=dropout,
        gat_proj_dim=gat_proj_dim,
        gat_heads=gat_heads,
        gat_out_per_head=gat_out_per_head,
    )
    opt_cfg = dict(lr=lr, weight_decay=weight_decay, warmup=warmup)

    total_params = DualGraphModel(in_dim=in_dim, out_dim=len(protein_names),
                                   **model_cfg).to("cpu")
    n_params = sum(p.numel() for p in total_params.parameters())
    print(f"\nModel parameters: {n_params:,}")
    del total_params

    # Epoch finder — builds its OWN fold-local graphs internally (see FIX
    # note in find_n_epochs), so it takes raw coords/k values here rather
    # than the pre-built full graph. train_model below still correctly
    # uses the full pre-built graph — that's the final refit on ALL
    # training data, no held-out split involved there.
    cv_splits = load_cv_split(cv_split_path)
    print(f"\n--- Epoch finder (fold {epoch_finder_fold}) ---")
    n_epochs = find_n_epochs(
        X_train, Y_train, coords_train, k_spatial, k_expression,
        cv_splits, epoch_finder_fold, model_cfg, opt_cfg,
        device, max_epochs, patience, log_interval=log_interval,
    )

    # Final refit
    print(f"\n--- Final refit (n_epochs={n_epochs}) ---")
    model, history = train_model(
        X_train, Y_train, sp_ei_tr, sp_ew_tr, ex_ei_tr, ex_ew_tr,
        n_epochs, model_cfg, opt_cfg, device, log_interval=log_interval,
    )
    save_model(model, os.path.join(out_path, "gnn_v5_model.pt"))
    history.to_csv(os.path.join(out_path, "history.csv"), index=False)

    # Final R² summary on training data
    print("\n--- Final R² on training data ---")
    pred_train = predict(model, X_train, sp_ei_tr, sp_ew_tr,
                         ex_ei_tr, ex_ew_tr, device)
    mean_r2, med_r2, r2_vals = compute_r2(pred_train, Y_train)
    df_r2 = pd.DataFrame({
        "protein":  protein_names,
        "r2":       r2_vals,
    }).sort_values("r2", ascending=False)
    print(df_r2.to_string(index=False))
    print(f"\n  Mean R²   = {mean_r2:.4f}")
    print(f"  Median R² = {med_r2:.4f}")
    df_r2.to_csv(os.path.join(out_path, "train_r2_per_protein.csv"), index=False)

    # Inference on validation
    print("\n--- Inference on validation ---")
    Z_pred = predict(model, X_val, sp_ei_val, sp_ew_val,
                     ex_ei_val, ex_ew_val, device)
    X_pred = inverse_transform_protein(Z_pred, protein_stats)
    X_pred = np.clip(X_pred, 0, None)

    submission = pd.DataFrame(X_pred, columns=protein_names)
    submission.insert(0, "pxl_col_in_fullres", coords_val[:, 1].astype(int))
    submission.insert(0, "pxl_row_in_fullres", coords_val[:, 0].astype(int))
    submission.insert(0, "barcode", barcodes_val)

    out_csv = os.path.join(out_path, "pred_val.csv")
    submission.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}  shape={submission.shape}")

    return model, submission, n_epochs, df_r2