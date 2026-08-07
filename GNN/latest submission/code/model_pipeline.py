"""
model_pipeline.py — Multi-Scale Spatial GNN for RNA → Protein (Team 37)
=========================================================================
Standalone. No repo clone needed.

Default mode trains ONE deployable model on 100% of the data - no held-out
fold, no internal split of any kind. Saves every fitted preprocessing
artifact needed to reproduce identical preprocessing at inference time (see
predict.py), which then runs the trained model on valid_rna.h5ad/
test_rna.h5ad, also with no split or sampling - every row gets a prediction.

Pass --cv to instead run the 5-fold spatially-blocked CV (genuinely disjoint
train/test graphs per fold) - this is the honest generalization estimate
used for architecture/hyperparameter comparisons (e.g. --n_layers), not for
producing a deployable model. --val_split is a third, TEMPORARY mode that
holds out a fraction of bins within a single 100%-data-style run, only kept
around to reproduce one specific historical result.

Architecture (MultiScaleSpatialGNN):
    Input SVD/marker/H&E features → Linear → LayerNorm → ReLU → Dropout
    → two parallel branches, each [SAGEConv + LayerNorm + ReLU + Residual] x n_layers:
        local branch:    k-NN graph (fine detail)
        regional branch: radius graph (density-varying neighborhoods)
    → concat + fuse → Linear → LN → ReLU → Dropout → Linear(hidden/2 → 44)

Loss: 0.8*MSE + 0.2*(1 - mean Pearson r)
Opt:  AdamW + warmup + CosineAnnealingLR

Quick start (CPU — verifies pipeline end-to-end):
    python model_pipeline.py --rna train_rna.h5ad --pro train_pro.h5ad --out out/ --epochs 50 --patience 10

AWS GPU (recommended):
    python model_pipeline.py --rna train_rna.h5ad --pro train_pro.h5ad --out out/ --device cuda --amp
"""

import argparse
import json
import os
import time
import scipy.sparse as sp

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD  # sparse-friendly, allows more genes
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import SAGEConv
from torch.utils.checkpoint import checkpoint


# ═══════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════

class ResidualSAGEBlock(nn.Module):
    """
    One graph-conv residual block:
        h = SAGEConv(x, edge_index)
        h = LayerNorm(h)
        h = ReLU(h + x)          ← skip connection (same hidden dim)
        h = Dropout(h)
    """
    def __init__(self, hidden: int, dropout: float = 0.3):
        super().__init__()
        self.conv    = SAGEConv(hidden, hidden)
        self.norm    = nn.LayerNorm(hidden)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index)
        h = self.norm(h)
        h = F.relu(h + x)                              # residual
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class MultiScaleSpatialGNN(nn.Module):
    """Two parallel SAGE stacks - one over a local k-NN graph (fine detail),
    one over a regional radius graph (density-varying neighborhoods) - fused
    before the prediction head.

    Parameters
    ----------
    in_channels  : SVD + marker feature dimension
    hidden       : hidden size per branch (default 384)
    out_channels : number of protein targets (default 44)
    n_layers     : ResidualSAGEBlock layers PER branch (default 4)
    dropout      : dropout rate (default 0.3)
    """
    def __init__(self,
                 in_channels: int = 298,
                 hidden: int = 384,
                 out_channels: int = 44,
                 n_layers: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.local_blocks = nn.ModuleList(
            [ResidualSAGEBlock(hidden, dropout) for _ in range(n_layers)]
        )
        self.regional_blocks = nn.ModuleList(
            [ResidualSAGEBlock(hidden, dropout) for _ in range(n_layers)]
        )

        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_channels),
        )

    def forward(self, x: torch.Tensor, edge_local: torch.Tensor, edge_regional: torch.Tensor) -> torch.Tensor:
        h0 = self.input_proj(x)

        # Gradient checkpointing during training: recomputes each block's
        # activations on the backward pass instead of keeping them all in
        # memory. Costs some extra compute but saves a lot of GPU memory,
        # which is what lets this run at full size without running out of VRAM.
        hl = h0
        for blk in self.local_blocks:
            if self.training:
                hl = checkpoint(blk, hl, edge_local, use_reentrant=False)
            else:
                hl = blk(hl, edge_local)

        hr = h0
        for blk in self.regional_blocks:
            if self.training:
                hr = checkpoint(blk, hr, edge_regional, use_reentrant=False)
            else:
                hr = blk(hr, edge_regional)

        h = self.fuse(torch.cat([hl, hr], dim=-1))
        return self.output_head(h)

# ═══════════════════════════════════════════════════════════════════
#  LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def pearson_loss(yp: torch.Tensor, yt: torch.Tensor) -> torch.Tensor:
    """1 minus the mean Pearson r across all protein columns."""
    vp = yp - yp.mean(0, keepdim=True)
    vt = yt - yt.mean(0, keepdim=True)
    r  = (vp * vt).sum(0) / ((vp**2).sum(0) * (vt**2).sum(0) + 1e-8).sqrt()
    return (1 - r).mean()


def combined_loss(yp: torch.Tensor, yt: torch.Tensor, w: float = 0.8) -> torch.Tensor:
    """Weighted mix of MSE and (1 - Pearson r). Pearson-based loss only
    rewards getting the correlation *shape* right, not the actual scale of
    the values, so MSE is weighted more heavily here to make sure the model
    also learns the right absolute magnitude, not just the right trend."""
    return w * F.mse_loss(yp, yt) + (1 - w) * pearson_loss(yp, yt)


# ═══════════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════════

def mean_pearson_r(pred: np.ndarray, true: np.ndarray) -> float:
    rs = [pearsonr(pred[:, j], true[:, j])[0] for j in range(true.shape[1])]
    rs = [r for r in rs if not np.isnan(r)]
    return float(np.mean(rs)) if rs else 0.0


def mean_spearman_r(pred: np.ndarray, true: np.ndarray) -> float:
    rs = [spearmanr(pred[:, j], true[:, j]).correlation for j in range(true.shape[1])]
    rs = [r for r in rs if not np.isnan(r)]
    return float(np.mean(rs)) if rs else 0.0


def per_protein_metrics(pred: np.ndarray, true: np.ndarray):
    """Per-protein Pearson r, Spearman, RMSE, MAE - used by the 5-fold CV path."""
    n = true.shape[1]
    pearson  = np.zeros(n)
    spearman = np.zeros(n)
    rmse     = np.zeros(n)
    mae      = np.zeros(n)
    for j in range(n):
        r, _ = pearsonr(pred[:, j], true[:, j])
        pearson[j] = r if not np.isnan(r) else 0.0
        sr = spearmanr(pred[:, j], true[:, j]).correlation
        spearman[j] = sr if not np.isnan(sr) else 0.0
        rmse[j] = float(np.sqrt(np.mean((pred[:, j] - true[:, j]) ** 2)))
        mae[j]  = float(np.mean(np.abs(pred[:, j] - true[:, j])))
    return pearson, spearman, rmse, mae


# ═══════════════════════════════════════════════════════════════════
#  GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════

def build_knn_graph(coords: np.ndarray, k: int = 8) -> torch.Tensor:
    """
    Undirected k-NN spatial graph from (array_row, array_col) coordinates.
    Returns edge_index of shape (2, E).
    """
    nn_ = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1)
    nn_.fit(coords)
    _, indices = nn_.kneighbors(coords)

    src, dst = [], []
    for i, row in enumerate(indices):
        for j in row[1:]:          # skip self
            src += [i, j]
            dst += [j, i]          # undirected

    ei = torch.tensor([src, dst], dtype=torch.long)
    return torch.unique(ei, dim=1)


def build_radius_graph(coords: np.ndarray, radius: float, k_fallback: int = 4) -> torch.Tensor:
    """Regional graph - connect all bins within `radius` (captures density-varying
    neighborhoods that a fixed-k graph misses). Isolated nodes (nothing within radius)
    fall back to their nearest k_fallback neighbors so the graph stays connected."""
    nn_ = NearestNeighbors(radius=radius, algorithm="auto", n_jobs=-1)
    nn_.fit(coords)
    neigh = nn_.radius_neighbors(coords, return_distance=False)

    knn_ = NearestNeighbors(n_neighbors=k_fallback + 1, algorithm="auto", n_jobs=-1)
    knn_.fit(coords)
    _, knn_idx = knn_.kneighbors(coords)

    src, dst = [], []
    for i, row in enumerate(neigh):
        row = row[row != i]
        if len(row) == 0:
            row = knn_idx[i, 1:]
        for j in row:
            src += [i, int(j)]
            dst += [int(j), i]           # undirected

    ei = torch.tensor([src, dst], dtype=torch.long)
    return torch.unique(ei, dim=1)


def get_coords(obs, coord_space: str = "grid", microns_per_pixel: float = 0.8820219467631594) -> np.ndarray:
    """Returns the (row, col) coordinate array used for both graphs.

    coord_space="grid" (default, unchanged from before): raw array_row/array_col
    grid-index units. A --radius value here is measured in grid steps, NOT
    physical distance - adjacent bins are ~1 unit apart regardless of their
    real spacing in microns.

    coord_space="microns": pxl_row_in_fullres/pxl_col_in_fullres (the full-
    resolution image pixel coordinates) scaled by microns_per_pixel, so a
    --radius value is now real physical distance in microns. Empirically,
    for this dataset one array-grid step is ~16 microns (measured directly:
    median pixel distance between array-grid-adjacent bins is ~18.1px,
    18.1 * 0.882 microns/px =~ 16 microns) - so a microns-radius of e.g. 30
    is roughly equivalent to ~1.9 grid-unit steps, tighter than the previous
    "auto" grid-space default (~3.5 steps =~ 56 microns).

    microns_per_pixel default (0.8820219467631594) is this dataset's train-
    split calibration constant (used consistently for train/valid/test here
    for simplicity - the true valid/test constant is ~0.883, a ~0.1%
    difference, immaterial at these radii)."""
    if coord_space == "microns":
        return np.column_stack([
            obs["pxl_row_in_fullres"].to_numpy(dtype=np.float64) * microns_per_pixel,
            obs["pxl_col_in_fullres"].to_numpy(dtype=np.float64) * microns_per_pixel,
        ]).astype(np.float32)
    else:
        return np.column_stack([
            obs["array_row"].to_numpy(dtype=np.float32),
            obs["array_col"].to_numpy(dtype=np.float32),
        ])


# ═══════════════════════════════════════════════════════════════════
#  PREPROCESSING  (inline, mirrors preprocessing.py)
# ═══════════════════════════════════════════════════════════════════

def preprocess(rna_path: str, pro_path: str,
               out_dir: str, n_hvg: int = 2000):
    os.makedirs(out_dir, exist_ok=True)
    rna_hvg_path = os.path.join(out_dir, "rna_hvg.h5ad")
    pro_raw_path = os.path.join(out_dir, "protein_raw.h5ad")

    if os.path.exists(rna_hvg_path) and os.path.exists(pro_raw_path):
        print("  Preprocessed files found — skipping.")
        return rna_hvg_path, pro_raw_path

    print("  Loading RNA...")
    rna = ad.read_h5ad(rna_path).copy()
    print(f"  RNA: {rna.shape}")
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    if n_hvg <= 0:
        # Use ALL genes - no HVG pre-filtering. HVG selection is normally
        # computed once on the full dataset before the fold split happens,
        # so which genes are "kept" would otherwise be informed by bins
        # that later become each fold's held-out test set. Skipping it keeps
        # every downstream step (SVD, markers, everything) 100% fold-local.
        rna_hvg = rna.copy()
        print("  n_hvg<=0: using ALL genes (no HVG filtering, no global-fit step)")
    else:
        sc.pp.highly_variable_genes(rna, n_top_genes=n_hvg,
                                    flavor="cell_ranger", subset=False)
        hvgs    = list(rna.var_names[rna.var.highly_variable])
        rna_hvg = rna[:, hvgs].copy()
    rna_hvg.write_h5ad(rna_hvg_path)
    print(f"  Saved rna_hvg.h5ad  ({rna_hvg.shape})")

    print("  Loading protein (raw)...")
    pro = ad.read_h5ad(pro_path).copy()
    pro.write_h5ad(pro_raw_path)
    print(f"  Saved protein_raw.h5ad  ({pro.shape})")
    return rna_hvg_path, pro_raw_path


def make_cv_splits(rna_hvg_path: str, out_dir: str,
                    n_splits: int = 5, block_size: int = 10,
                    random_state: int = 42) -> str:
    """Spatially-blocked 5-fold CV split: bins are grouped into 10x10
    coordinate blocks, then GroupKFold assigns whole blocks to folds (never
    splits a block across train/test) - this is what makes the resulting
    fold-vs-fold comparison an honest, spatially-disjoint generalization
    estimate rather than a plain random split, which would let nearby
    (highly spatially-correlated) bins leak between train and test.
    Same methodology that produced the 0.6566 Pearson / 0.5625 Spearman
    baseline. Reused as-is if cv_splits.json already exists in out_dir, so
    re-running with a different --n_layers etc. still compares against the
    exact same fold assignment."""
    cv_path = os.path.join(out_dir, "cv_splits.json")
    if os.path.exists(cv_path):
        print(f"  cv_splits.json found — reusing existing split ({cv_path}).")
        return cv_path

    rna = ad.read_h5ad(rna_hvg_path)
    n_bins = rna.n_obs
    row = rna.obs["array_row"].to_numpy()
    col = rna.obs["array_col"].to_numpy()

    block_id = (row // block_size).astype(np.int64) * 1_000_003 \
             + (col // block_size).astype(np.int64)
    uniq, inv = np.unique(block_id, return_inverse=True)
    n_blocks = len(uniq)

    rng = np.random.RandomState(random_state)
    rank = np.empty(n_blocks, dtype=int)
    rank[rng.permutation(n_blocks)] = np.arange(n_blocks)
    groups = rank[inv]

    splits = []
    for fold, (tr, te) in enumerate(
            GroupKFold(n_splits).split(np.arange(n_bins), groups=groups)):
        splits.append({"fold": fold, "train": tr.tolist(), "test": te.tolist()})
        print(f"  Fold {fold + 1}: {len(tr):,} train | {len(te):,} test")

    with open(cv_path, "w") as f:
        json.dump({"n_splits": n_splits, "n_bins": n_bins,
                   "n_blocks": int(n_blocks), "block_size": block_size,
                   "random_state": random_state, "splits": splits}, f)
    print("  Saved cv_splits.json")
    return cv_path


# ═══════════════════════════════════════════════════════════════════
#  LR SCHEDULER WITH WARMUP
# ═══════════════════════════════════════════════════════════════════

class WarmupCosineScheduler:
    """
    Linear warmup for `warmup_epochs` then CosineAnnealingLR.
    Wraps a single optimizer; call .step() once per epoch.
    """
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int,
                 base_lr: float, min_lr: float = 1e-6):
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
            progress = (e - self.warmup_epochs) / max(
                1, self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress))
        for pg in self.opt.param_groups:
            pg["lr"] = lr
        return lr


# ======================================================================
#  FINAL FULL-DATA TRAINING  (no held-out CV fold - one deployable model
#  trained on 100% of train_rna.h5ad/train_pro.h5ad, for running inference
#  on valid_rna.h5ad / test_rna.h5ad. Saves every fitted preprocessing
#  artifact (SVD, marker/H&E scalers, target mean/std, used gene list)
#  needed to reproduce identical preprocessing at inference time.
# ======================================================================

def train_final(args, rna_hvg_path: str, pro_raw_path: str, out_dir: str,
                 device: torch.device, use_amp: bool):
    print(f"\n{'='*65}")
    print("FINAL FULL-DATA TRAINING  (no held-out fold - for deployment)")
    print(f"{'='*65}")
    print(f"Device   : {device}   AMP: {use_amp}")

    rna = ad.read_h5ad(rna_hvg_path)
    pro = ad.read_h5ad(pro_raw_path)

    X_raw = rna.X.tocsr() if sp.issparse(rna.X) else sp.csr_matrix(rna.X)
    Y_raw = (pro.X.toarray() if hasattr(pro.X, "toarray") else pro.X).astype(np.float32)

    if args.target_transform == "clr":
        _Xp = np.nan_to_num(Y_raw.astype(np.float64))
        _log = np.log(_Xp + 1.0)
        _geom_log_mean = _log.mean(axis=1, keepdims=True)
        Y_arc = (_log - _geom_log_mean).astype(np.float32)
        print("  Target transform: CLR (centered log-ratio)")
    else:
        Y_arc = np.arcsinh(Y_raw / args.protein_cofactor).astype(np.float32)
        print(f"  Target transform: arcsinh(x/{args.protein_cofactor:g})")

    coords = get_coords(rna.obs, args.coord_space, args.microns_per_pixel)
    print(f"  Coordinate space: {args.coord_space}"
          + (f"  (microns_per_pixel={args.microns_per_pixel})" if args.coord_space == "microns" else ""))
    protein_names = list(pro.var_names)
    n_proteins = Y_arc.shape[1]
    n_bins = X_raw.shape[0]

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _mk = os.path.join(_script_dir, "marker_features.npy")
    if not os.path.exists(_mk):
        _mk = os.path.join(_script_dir, "..", "shared_preprocessing", "marker_features.npy")
    if os.path.exists(_mk):
        _MK_raw = np.load(_mk).astype(np.float32)
        assert _MK_raw.shape[0] == n_bins, f"marker/bin mismatch {_MK_raw.shape} vs {n_bins}"
    else:
        _MK_raw = np.zeros((n_bins, 0), dtype=np.float32)
        print("  final: marker_features.npy NOT found - continuing without markers")

    _he = os.path.join(_script_dir, "he_features.npy")
    if not os.path.exists(_he):
        _he = os.path.join(_script_dir, "..", "shared_preprocessing", "he_features.npy")
    if os.path.exists(_he):
        _HE_raw = np.load(_he).astype(np.float32)
        assert _HE_raw.shape[0] == n_bins, f"he/bin mismatch {_HE_raw.shape} vs {n_bins}"
    else:
        _HE_raw = np.zeros((n_bins, 0), dtype=np.float32)
        print("  final: he_features.npy NOT found - continuing without histology features")

    _FEATDIM = args.svd_dims + _MK_raw.shape[1] + _HE_raw.shape[1]
    print(f"Bins     : {n_bins:,}")
    print(f"Proteins : {n_proteins}")
    print(f"Features : SVD({args.svd_dims}) + {_MK_raw.shape[1]} markers + {_HE_raw.shape[1]} "
          f"H&E -> {_FEATDIM} dims  (fit on ALL {n_bins:,} bins - this IS the deployment fit)")

    if args.radius <= 0:
        _nn = NearestNeighbors(n_neighbors=args.k + 1, algorithm="auto", n_jobs=-1).fit(coords)
        _dist, _ = _nn.kneighbors(coords)
        radius = float(3.0 * np.median(_dist[:, -1]))
        print(f"  Auto regional radius = {radius:.2f}  (3x median {args.k}-NN distance)")
    else:
        radius = args.radius

    # ---- fit preprocessing on ALL data - this is the artifact inference reuses ----
    _svd = TruncatedSVD(n_components=args.svd_dims, random_state=42)
    _X = _svd.fit_transform(X_raw).astype(np.float32)

    _mmu = _msd = _hmu = _hsd = None
    if _MK_raw.shape[1] > 0:
        _mmu = _MK_raw.mean(0, keepdims=True); _msd = _MK_raw.std(0, keepdims=True) + 1e-8
        _X = np.concatenate([_X, (_MK_raw - _mmu) / _msd], axis=1).astype(np.float32)
    if _HE_raw.shape[1] > 0:
        _hmu = _HE_raw.mean(0, keepdims=True); _hsd = _HE_raw.std(0, keepdims=True) + 1e-8
        _X = np.concatenate([_X, (_HE_raw - _hmu) / _hsd], axis=1).astype(np.float32)

    _ymu = Y_arc.mean(0, keepdims=True)
    _ysd = Y_arc.std(0, keepdims=True) + 1e-8
    _Y = ((Y_arc - _ymu) / _ysd).astype(np.float32)

    t0 = time.time()
    print(f"  Building multi-scale graphs over all {n_bins:,} bins (k={args.k}, radius={radius:.1f})...")
    ei_local    = build_knn_graph(coords, k=args.k).to(device)
    ei_regional = build_radius_graph(coords, radius=radius, k_fallback=args.k_fallback).to(device)
    print(f"  {ei_local.shape[1]:,} local / {ei_regional.shape[1]:,} regional edges  ({time.time()-t0:.1f}s)")

    X_t = torch.tensor(_X).to(device)
    Y_t = torch.tensor(_Y).to(device)

    # TEMPORARY, opt-in only: --val_split > 0 carves out a held-out fraction
    # of bins to reproduce a specific historical run for comparison. Default
    # (--val_split 0.0) is unchanged from before: every one of the n_bins
    # rows goes into the loss, no held-out slice at all (that's what the
    # separate 5-fold CV run is for). The graph itself is still built over
    # ALL bins either way (message-passing can see held-out bins' RNA
    # features, just not their protein labels) - so with a split enabled,
    # val_r/val_sr is a softer, transductive estimate, not as strict as the
    # disjoint 5-fold CV.
    if args.val_split > 0:
        rng = np.random.RandomState(42)
        perm = rng.permutation(n_bins)
        n_val = int(round(n_bins * args.val_split))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        train_idx_t = torch.tensor(train_idx, dtype=torch.long).to(device)
        val_idx_t = torch.tensor(val_idx, dtype=torch.long).to(device)
        _Y_np = _Y[train_idx]
        _Yval_np = _Y[val_idx]
        print(f"  TEMPORARY internal validation split enabled ({args.val_split:.0%}): "
              f"{len(train_idx):,} train / {len(val_idx):,} val bins. Reproducing a "
              f"historical run for comparison - default (--val_split 0.0) trains on "
              f"100% of the data with no split at all.")
    else:
        train_idx_t = None
        val_idx_t = None
        _Y_np = _Y
        print(f"  Training on all {n_bins:,} bins - no internal train/val split")

    model = MultiScaleSpatialGNN(
        in_channels=_FEATDIM, hidden=args.hidden, out_channels=n_proteins,
        n_layers=args.n_layers, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: MultiScaleSpatialGNN  params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = WarmupCosineScheduler(opt, warmup_epochs=args.warmup, max_epochs=args.epochs,
                                   base_lr=args.lr, min_lr=args.lr / 30)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_train = -1e9
    best_state = None
    patience_ctr = 0
    history = []
    print(f"\n  Training: max_epochs={args.epochs}  patience={args.patience}  (100% of data, no split)")

    has_val = train_idx_t is not None
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(X_t, ei_local, ei_regional)
            loss = combined_loss(pred[train_idx_t], Y_t[train_idx_t]) if has_val else combined_loss(pred, Y_t)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        lr_now = sched.step()

        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_all = model(X_t, ei_local, ei_regional)
            _pred_np = pred_all.float().cpu().numpy()

            if has_val:
                # Genuine held-out metric (val bins' labels never entered the loss).
                tl = combined_loss(pred_all[val_idx_t].float(), Y_t[val_idx_t]).item()
                train_pr = mean_pearson_r(_pred_np[train_idx], _Y_np)
                train_sr = mean_spearman_r(_pred_np[train_idx], _Y_np)
                val_pr = mean_pearson_r(_pred_np[val_idx], _Yval_np)
                val_sr = mean_spearman_r(_pred_np[val_idx], _Yval_np)
                # Checkpoint selection matches the historical run: pick the
                # epoch with the best held-out val metric, not train fit.
                train_metric = 0.5 * val_pr + 0.5 * val_sr
            else:
                # Diagnostic-only: how well the model currently fits the training
                # data it just saw. Not a generalization estimate - there is no
                # held-out data in this run to measure that against.
                tl = combined_loss(pred_all.float(), Y_t).item()
                train_pr = mean_pearson_r(_pred_np, _Y_np)
                train_sr = mean_spearman_r(_pred_np, _Y_np)
                val_pr = val_sr = None
                train_metric = 0.5 * train_pr + 0.5 * train_sr

        improved = train_metric > best_train
        if improved:
            best_train = train_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        history.append({"epoch": epoch, "train_loss": loss.item(), "eval_loss": tl,
                         "train_pearson": train_pr, "train_spearman": train_sr,
                         "val_pearson": val_pr, "val_spearman": val_sr, "lr": lr_now})

        if epoch == 1 or epoch % 20 == 0 or improved:
            elapsed = (time.time() - t_start) / 60
            eta = elapsed / epoch * (args.epochs - epoch)
            star = " *" if improved else ""
            if has_val:
                print(f"  Ep {epoch:>5}/{args.epochs} | train={loss.item():.4f}  val={tl:.4f}  "
                      f"val_r={val_pr:.4f}  val_sr={val_sr:.4f}  patience={patience_ctr}/{args.patience}  "
                      f"lr={lr_now:.2e}  [{elapsed:.1f}min, ~{eta:.0f}min left]{star}")
            else:
                print(f"  Ep {epoch:>5}/{args.epochs} | train={loss.item():.4f}  eval={tl:.4f}  "
                      f"train_r={train_pr:.4f}  train_sr={train_sr:.4f}  patience={patience_ctr}/{args.patience}  "
                      f"lr={lr_now:.2e}  [{elapsed:.1f}min, ~{eta:.0f}min left]{star}")

        if patience_ctr >= args.patience:
            print(f"  Early stop at epoch {epoch} (best {'val' if has_val else 'train'}_metric[0.5*r+0.5*sr]={best_train:.4f})")
            break

    model.load_state_dict(best_state)

    final_dir = os.path.join(out_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)

    torch.save(best_state, os.path.join(final_dir, "model.pt"))
    pd.DataFrame(history).to_csv(os.path.join(final_dir, "history_final.csv"), index=False)

    import pickle
    with open(os.path.join(final_dir, "svd.pkl"), "wb") as f:
        pickle.dump(_svd, f)

    _z = np.zeros((1, 0), dtype=np.float32)
    np.savez(os.path.join(final_dir, "preprocessing_stats.npz"),
              y_mean=_ymu, y_std=_ysd,
              marker_mean=(_mmu if _mmu is not None else _z),
              marker_std=(_msd if _msd is not None else _z),
              he_mean=(_hmu if _hmu is not None else _z),
              he_std=(_hsd if _hsd is not None else _z))

    meta = {
        "protein_names": protein_names,
        "used_genes": list(rna.var_names),
        "target_transform": args.target_transform,
        "protein_cofactor": args.protein_cofactor,
        "svd_dims": args.svd_dims,
        "n_markers": int(_MK_raw.shape[1]),
        "n_he": int(_HE_raw.shape[1]),
        "feat_dim": int(_FEATDIM),
        "hidden": args.hidden,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "k": args.k,
        "radius": radius,
        "k_fallback": args.k_fallback,
        "coord_space": args.coord_space,
        "microns_per_pixel": args.microns_per_pixel,
        "n_hvg": args.n_hvg,
        "n_train_bins": int(n_bins),
        "val_split": args.val_split,
    }
    if has_val:
        meta["best_val_metric"] = float(best_train)
        meta["note"] = ("TEMPORARY reproduction run with an internal val_split held out "
                         "(not the default methodology). best_val_metric is a genuine "
                         "held-out estimate (val bins' labels never entered the loss), "
                         "but the graph was still built over all bins, so message-passing "
                         "could see held-out bins' RNA features - a softer, transductive "
                         "estimate, not as strict as the disjoint 5-fold CV. Compare "
                         "against the 5-fold CV's 0.6566 Pearson / 0.5625 Spearman and "
                         "the default (val_split=0) deployment run, not just each other.")
    else:
        meta["best_train_fit_metric"] = float(best_train)
        meta["note"] = ("best_train_fit_metric is measured on the SAME 100% of data the model "
                         "trained on (no internal split at all in this run) - it is NOT a "
                         "generalization estimate. Use the separate 5-fold CV run's honest "
                         "Pearson/Spearman/RMSE (0.6566 / 0.5625 / 0.7451) for that.")
    with open(os.path.join(final_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  FINAL MODEL SAVED to {final_dir}")
    if has_val:
        print(f"  TEMPORARY val_split={args.val_split:.0%} run. Best held-out val metric = {best_train:.4f}")
        print(f"  Softer/transductive estimate - not as strict as the 5-fold CV.")
    else:
        print(f"  Trained on all {n_bins:,} bins, no internal split. Best train-fit metric = {best_train:.4f}")
        print(f"  This is NOT the generalization estimate - use the 5-fold CV numbers for that.")
    print(f"{'='*65}")


# ======================================================================
#  5-FOLD SPATIALLY-BLOCKED CV  (opt-in via --cv; NOT the deployment path -
#  train_final() above is still what runs by default / for deployment)
#
#  Genuinely disjoint fold graphs: unlike train_final()'s optional
#  --val_split (one graph over ALL bins, so message-passing can still see
#  held-out bins' RNA features), here each fold builds SEPARATE train-only
#  and test-only graphs - the test bins have zero connectivity to train
#  bins. This is the stricter, "honest held-out" estimate referenced
#  throughout this project (0.6566 Pearson / 0.5625 Spearman baseline).
# ======================================================================

def run_cv(args, rna_hvg_path: str, pro_raw_path: str, out_dir: str,
            device: torch.device, use_amp: bool):
    print(f"\n{'='*65}")
    print(f"5-FOLD SPATIALLY-BLOCKED CV  (honest held-out test per fold)")
    print(f"{'='*65}")
    print(f"Device   : {device}   AMP: {use_amp}")

    preproc_dir = os.path.join(out_dir, "preprocessed")
    os.makedirs(preproc_dir, exist_ok=True)
    if args.cv_split_path and os.path.exists(args.cv_split_path):
        cv_path = args.cv_split_path
        print(f"  Using explicit CV split: {cv_path}")
    else:
        cv_path = make_cv_splits(rna_hvg_path, preproc_dir)

    rna = ad.read_h5ad(rna_hvg_path)
    pro = ad.read_h5ad(pro_raw_path)

    X_raw = rna.X.tocsr() if sp.issparse(rna.X) else sp.csr_matrix(rna.X)
    Y_raw = (pro.X.toarray() if hasattr(pro.X, "toarray") else pro.X).astype(np.float32)

    if args.target_transform == "clr":
        _Xp = np.nan_to_num(Y_raw.astype(np.float64))
        _log = np.log(_Xp + 1.0)
        _geom_log_mean = _log.mean(axis=1, keepdims=True)
        Y_arc = (_log - _geom_log_mean).astype(np.float32)
        print("  Target transform: CLR (centered log-ratio)")
    else:
        Y_arc = np.arcsinh(Y_raw / args.protein_cofactor).astype(np.float32)
        print(f"  Target transform: arcsinh(x/{args.protein_cofactor:g})")

    coords = get_coords(rna.obs, args.coord_space, args.microns_per_pixel)
    print(f"  Coordinate space: {args.coord_space}"
          + (f"  (microns_per_pixel={args.microns_per_pixel})" if args.coord_space == "microns" else ""))
    protein_names = list(pro.var_names)
    n_proteins = Y_arc.shape[1]
    n_bins = X_raw.shape[0]

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _mk = os.path.join(_script_dir, "marker_features.npy")
    if not os.path.exists(_mk):
        _mk = os.path.join(_script_dir, "..", "shared_preprocessing", "marker_features.npy")
    if os.path.exists(_mk):
        _MK_raw = np.load(_mk).astype(np.float32)
        assert _MK_raw.shape[0] == n_bins, f"marker/bin mismatch {_MK_raw.shape} vs {n_bins}"
    else:
        _MK_raw = np.zeros((n_bins, 0), dtype=np.float32)
        print("  cv: marker_features.npy NOT found - continuing without markers")

    _he = os.path.join(_script_dir, "he_features.npy")
    if not os.path.exists(_he):
        _he = os.path.join(_script_dir, "..", "shared_preprocessing", "he_features.npy")
    if os.path.exists(_he):
        _HE_raw = np.load(_he).astype(np.float32)
        assert _HE_raw.shape[0] == n_bins, f"he/bin mismatch {_HE_raw.shape} vs {n_bins}"
    else:
        _HE_raw = np.zeros((n_bins, 0), dtype=np.float32)
        print("  cv: he_features.npy NOT found - continuing without histology features")

    _FEATDIM = args.svd_dims + _MK_raw.shape[1] + _HE_raw.shape[1]
    print(f"Bins     : {n_bins:,}")
    print(f"Proteins : {n_proteins}")
    print(f"Features : SVD({args.svd_dims}) + {_MK_raw.shape[1]} markers + {_HE_raw.shape[1]} "
          f"H&E -> {_FEATDIM} dims  (refit per fold, train-only)")

    if args.radius <= 0:
        _nn = NearestNeighbors(n_neighbors=args.k + 1, algorithm="auto", n_jobs=-1).fit(coords)
        _dist, _ = _nn.kneighbors(coords)
        radius = float(3.0 * np.median(_dist[:, -1]))
        print(f"  Auto regional radius = {radius:.2f}  (3x median {args.k}-NN distance)")
    else:
        radius = args.radius

    with open(cv_path) as f:
        splits = json.load(f)["splits"]
    if args.max_folds and args.max_folds > 0:
        splits = splits[: args.max_folds]

    fold_pearson  = np.zeros((len(splits), n_proteins))
    fold_spearman = np.zeros((len(splits), n_proteins))
    fold_rmse     = np.zeros((len(splits), n_proteins))
    fold_mae      = np.zeros((len(splits), n_proteins))

    for split in splits:
        fold = split["fold"]
        train_idx = np.array(split["train"])
        test_idx  = np.array(split["test"])

        print(f"\n{'-'*65}")
        print(f"  FOLD {fold + 1}/{len(splits)}   train={len(train_idx):,}   test={len(test_idx):,}")
        print(f"{'-'*65}")

        # ---- fold-safe feature extraction: fit on TRAIN rows only ----
        _svd = TruncatedSVD(n_components=args.svd_dims, random_state=42)
        _Xtr = _svd.fit_transform(X_raw[train_idx]).astype(np.float32)
        _Xte = _svd.transform(X_raw[test_idx]).astype(np.float32)

        if _MK_raw.shape[1] > 0:
            _mmu = _MK_raw[train_idx].mean(0, keepdims=True)
            _msd = _MK_raw[train_idx].std(0, keepdims=True) + 1e-8
            _Xtr = np.concatenate([_Xtr, (_MK_raw[train_idx] - _mmu) / _msd], axis=1).astype(np.float32)
            _Xte = np.concatenate([_Xte, (_MK_raw[test_idx]  - _mmu) / _msd], axis=1).astype(np.float32)
        if _HE_raw.shape[1] > 0:
            _hmu = _HE_raw[train_idx].mean(0, keepdims=True)
            _hsd = _HE_raw[train_idx].std(0, keepdims=True) + 1e-8
            _Xtr = np.concatenate([_Xtr, (_HE_raw[train_idx] - _hmu) / _hsd], axis=1).astype(np.float32)
            _Xte = np.concatenate([_Xte, (_HE_raw[test_idx]  - _hmu) / _hsd], axis=1).astype(np.float32)

        _ymu = Y_arc[train_idx].mean(0, keepdims=True)
        _ysd = Y_arc[train_idx].std(0, keepdims=True) + 1e-8
        _Ytr = ((Y_arc[train_idx] - _ymu) / _ysd).astype(np.float32)
        _Yte = ((Y_arc[test_idx]  - _ymu) / _ysd).astype(np.float32)

        # ---- SEPARATE graphs for train and test - genuinely disjoint ----
        t0 = time.time()
        print(f"  Building multi-scale graphs (k={args.k}, radius={radius:.1f})...")
        ei_tr_local    = build_knn_graph(coords[train_idx], k=args.k).to(device)
        ei_te_local    = build_knn_graph(coords[test_idx],  k=args.k).to(device)
        ei_tr_regional = build_radius_graph(coords[train_idx], radius=radius, k_fallback=args.k_fallback).to(device)
        ei_te_regional = build_radius_graph(coords[test_idx],  radius=radius, k_fallback=args.k_fallback).to(device)
        print(f"  Train: {ei_tr_local.shape[1]:,} local / {ei_tr_regional.shape[1]:,} regional edges | "
              f"Test: {ei_te_local.shape[1]:,} local / {ei_te_regional.shape[1]:,} regional edges  ({time.time()-t0:.1f}s)")

        X_tr = torch.tensor(_Xtr).to(device)
        X_te = torch.tensor(_Xte).to(device)
        Y_tr = torch.tensor(_Ytr).to(device)
        Y_te = torch.tensor(_Yte).to(device)

        model = MultiScaleSpatialGNN(
            in_channels=_FEATDIM, hidden=args.hidden, out_channels=n_proteins,
            n_layers=args.n_layers, dropout=args.dropout,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n  Model: MultiScaleSpatialGNN  params={n_params:,}")

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = WarmupCosineScheduler(opt, warmup_epochs=args.warmup, max_epochs=args.epochs,
                                       base_lr=args.lr, min_lr=args.lr / 30)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_val = -1e9
        best_state = None
        patience_ctr = 0
        history = []
        print(f"\n  Training: max_epochs={args.epochs}  patience={args.patience}")

        t_start = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train()
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(X_tr, ei_tr_local, ei_tr_regional)
                loss = combined_loss(pred, Y_tr)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            lr_now = sched.step()

            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred_te = model(X_te, ei_te_local, ei_te_regional)
                vl = combined_loss(pred_te.float(), Y_te).item()
                _pred_te_np = pred_te.float().cpu().numpy()
                val_pr = mean_pearson_r(_pred_te_np, _Yte)
                val_sr = mean_spearman_r(_pred_te_np, _Yte)
                val_metric = 0.5 * val_pr + 0.5 * val_sr

            improved = val_metric > best_val
            if improved:
                best_val = val_metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1

            history.append({"epoch": epoch, "train_loss": loss.item(), "val_loss": vl,
                             "val_pearson": val_pr, "val_spearman": val_sr, "lr": lr_now})

            if epoch == 1 or epoch % 20 == 0 or improved:
                elapsed = (time.time() - t_start) / 60
                eta = elapsed / epoch * (args.epochs - epoch)
                star = " *" if improved else ""
                print(f"  Ep {epoch:>5}/{args.epochs} | train={loss.item():.4f}  val={vl:.4f}  "
                      f"val_r={val_pr:.4f}  val_sr={val_sr:.4f}  patience={patience_ctr}/{args.patience}  "
                      f"lr={lr_now:.2e}  [{elapsed:.1f}min, ~{eta:.0f}min left]{star}")

            if patience_ctr >= args.patience:
                print(f"  Early stop at epoch {epoch} (best val_metric[0.5*r+0.5*sr]={best_val:.4f})")
                break

        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(out_dir, f"model_fold{fold + 1}.pt"))
        pd.DataFrame(history).to_csv(os.path.join(out_dir, f"history_fold{fold + 1}.csv"), index=False)

        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_np = model(X_te, ei_te_local, ei_te_regional).float().cpu().numpy()
        np.save(os.path.join(out_dir, f"pred_fold{fold + 1}.npy"), pred_np)
        np.save(os.path.join(out_dir, f"true_fold{fold + 1}.npy"), _Yte)

        pr, sr, rmse, mae = per_protein_metrics(pred_np, _Yte)
        fold_pearson[fold]  = pr
        fold_spearman[fold] = sr
        fold_rmse[fold]     = rmse
        fold_mae[fold]      = mae
        print(f"  Fold {fold + 1} done | mean Pearson r = {pr.mean():.4f} | mean Spearman = {sr.mean():.4f} | "
              f"mean RMSE = {rmse.mean():.4f} | time = {(time.time()-t_start)/60:.1f}min")

    mean_pr = fold_pearson.mean(0)
    std_pr  = fold_pearson.std(0)
    mean_sr = fold_spearman.mean(0)
    mean_rmse = fold_rmse.mean(0)
    mean_mae  = fold_mae.mean(0)

    print(f"\n{'='*65}")
    print("  5-FOLD CV FINAL RESULTS")
    print(f"{'='*65}")
    print(f"  Mean Pearson r : {mean_pr.mean():.4f}")
    print(f"  Mean Spearman  : {mean_sr.mean():.4f}")
    print(f"  Mean RMSE      : {mean_rmse.mean():.4f}")

    per_protein_df = pd.DataFrame({
        "protein": protein_names,
        "mean_pearsonr": mean_pr,
        "std_pearsonr": std_pr,
        "mean_spearmanr": mean_sr,
        "mean_rmse": mean_rmse,
        "mean_mae": mean_mae,
    }).sort_values("mean_pearsonr", ascending=False)
    print("\nTop 10:")
    print(per_protein_df.head(10).to_string(index=False))
    print("\nBottom 5:")
    print(per_protein_df.tail(5).to_string(index=False))
    per_protein_df.to_csv(os.path.join(out_dir, "per_protein_metrics.csv"), index=False)

    pd.DataFrame({
        "model": ["GNN_V4_MultiScale"],
        "mean_pearsonr": [mean_pr.mean()],
        "std_pearsonr": [std_pr.mean()],
        "mean_spearmanr": [mean_sr.mean()],
        "mean_rmse": [mean_rmse.mean()],
        "mean_mae": [mean_mae.mean()],
        "epochs": [args.epochs], "hidden": [args.hidden], "n_layers": [args.n_layers],
        "k_local": [args.k], "radius": [radius], "svd_dims": [args.svd_dims], "n_hvg": [args.n_hvg],
    }).to_csv(os.path.join(out_dir, "results.csv"), index=False)
    print("\n  Saved per_protein_metrics.csv and results.csv")
    print(f"  Compare against the 0.6566 Pearson / 0.5625 Spearman baseline (n_layers=4).")


# ======================================================================
#  MAIN  (always trains on 100% of the data - no CV, no internal split)
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", required=True)
    parser.add_argument("--pro", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target_transform", default="clr", choices=["clr", "arcsinh"],
        help="protein target normalisation: 'clr' (centered log-ratio, matches the "
             "team's original reference preprocessing.py - now deprecated, CLR is "
             "not appropriate for CODEX protein data) or 'arcsinh' (arcsinh(x/cofactor), "
             "the current team standard - see --protein_cofactor)")
    parser.add_argument("--protein_cofactor", type=float, default=150.0,
        help="cofactor for target_transform=arcsinh, i.e. arcsinh(x/cofactor). "
             "Default 150.0 (this pipeline's own original invertible config). Set to "
             "5.0 to match the team's preprocessingV3.py exactly (arcsinh(x/5.0), the "
             "current documented standard on GitHub). The cofactor itself does not "
             "affect invertibility either way (raw = cofactor*sinh(z) always exactly "
             "recovers the pre-clip value) - only preprocessingV3.py's additional "
             "post-transform percentile clip would break exact invertibility, and "
             "that clip is deliberately NOT implemented here so real submissions stay "
             "exactly convertible back to raw CODEX scale.")

    parser.add_argument("--n_hvg",     type=int,   default=5000, help="HVGs; 0 = use ALL genes, no HVG filtering")
    parser.add_argument("--svd_dims",  type=int,   default=384,  help="TruncatedSVD components")
    parser.add_argument("--k",         type=int,   default=8,    help="local k-NN neighbours")
    parser.add_argument("--radius",    type=float, default=0.0,  help="regional graph radius; 0 = auto (3x median 8-NN distance). Units depend on --coord_space: grid steps if 'grid', microns if 'microns'.")
    parser.add_argument("--coord_space", default="grid", choices=["grid", "microns"],
        help="'grid' (default, unchanged): array_row/array_col grid-index units - "
             "--radius is measured in grid steps, not real distance. 'microns': use "
             "true physical distance (pxl_row_in_fullres/pxl_col_in_fullres * "
             "--microns_per_pixel) so --radius is real microns. One grid step is "
             "empirically ~16 microns for this dataset, so e.g. --radius 30 in "
             "microns space is tighter than the old grid-space auto-radius (~56um).")
    parser.add_argument("--microns_per_pixel", type=float, default=0.8820219467631594,
        help="only used when --coord_space=microns. Default is this dataset's "
             "train-split calibration constant.")
    parser.add_argument("--k_fallback",type=int,   default=4,    help="fallback k for isolated regional nodes")

    parser.add_argument("--hidden",        type=int,   default=384)
    parser.add_argument("--n_layers",      type=int,   default=4, help="ResidualSAGEBlock layers PER branch")
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--weight_decay",  type=float, default=1e-3)
    parser.add_argument("--warmup",        type=int,   default=10)
    parser.add_argument("--epochs",        type=int,   default=1000)
    parser.add_argument("--patience",      type=int,   default=30)
    parser.add_argument("--device",        default="cpu")
    parser.add_argument("--amp",           action="store_true", help="mixed-precision training (CUDA only) - faster/bigger")
    parser.add_argument("--final_model",   action="store_true",
        help="Kept for backward compatibility with existing deploy scripts - has no "
             "effect. Training always runs on 100%% of the data now.")
    parser.add_argument("--val_split",     type=float, default=0.0,
        help="TEMPORARY, opt-in only: fraction of bins to hold out as an internal "
             "validation set (e.g. 0.1 for 90/10), to reproduce a specific historical "
             "run for comparison. Default 0.0 trains on 100%% of the data with no "
             "internal split at all - that is the recommended, currently-deployed "
             "methodology. Only set this if you specifically want to reproduce the "
             "old 90/10-split run's numbers.")
    parser.add_argument("--cv",            action="store_true",
        help="Run the 5-fold spatially-blocked CV instead of final-model training. "
             "This is the honest, disjoint-graph generalization estimate (what "
             "produced the 0.6566 Pearson / 0.5625 Spearman baseline) - use this for "
             "architecture/hyperparameter comparisons, not for the deployed model.")
    parser.add_argument("--cv_split_path", default=None,
        help="Path to an existing cv_splits.json to reuse (same fold assignment as "
             "a previous CV run, for a clean apples-to-apples comparison). If not "
             "given, a fresh split is generated deterministically (same block_size=10, "
             "n_splits=5, random_state=42 as the original baseline run) and reused "
             "automatically on subsequent runs writing to the same --out directory.")
    parser.add_argument("--max_folds",     type=int, default=0,
        help="If >0, only run this many folds (for quick smoke-testing --cv).")
    args = parser.parse_args()

    device = torch.device(args.device)
    use_amp = bool(args.amp) and device.type == "cuda"
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    preproc_dir = os.path.join(out_dir, "preprocessed")

    print(f"\n{'='*65}")
    print("STEP 1/2  Preprocessing")
    print(f"{'='*65}")
    rna_hvg_path, pro_raw_path = preprocess(args.rna, args.pro, preproc_dir, n_hvg=args.n_hvg)

    if args.cv:
        print(f"\n{'='*65}")
        print("STEP 2/2  5-fold spatially-blocked CV")
        print(f"{'='*65}")
        run_cv(args, rna_hvg_path, pro_raw_path, out_dir, device, use_amp)
    else:
        print(f"\n{'='*65}")
        print("STEP 2/2  Training on 100% of the data")
        print(f"{'='*65}")
        train_final(args, rna_hvg_path, pro_raw_path, out_dir, device, use_amp)


if __name__ == "__main__":
    main()
