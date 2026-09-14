"""
gnn_pipeline_v4_final.py — Multi-Scale Spatial GNN for RNA → Protein (Team 37)
=========================================================================
Standalone. No repo clone needed.

Trains ONE deployable model on 100% of the data - no held-out fold, no
internal split of any kind. Saves every fitted preprocessing artifact
needed to reproduce identical preprocessing at inference time (see
infer_v4_final.py), which then runs the trained model on valid_rna.h5ad/
test_rna.h5ad, also with no split or sampling - every row gets a prediction.

Architecture (MultiScaleSpatialGNN):
    Input SVD/marker/H&E features → Linear → LayerNorm → ReLU → Dropout
    → two parallel branches, each [SAGEConv + LayerNorm + ReLU + Residual] x n_layers:
        local branch:    k-NN graph (fine detail)
        regional branch: radius graph (density-varying neighborhoods)
    → concat + fuse → Linear → LN → ReLU → Dropout → Linear(hidden/2 → 44)

Loss: 0.8*MSE + 0.2*(1 - mean Pearson r)
Opt:  AdamW + warmup + CosineAnnealingLR

Quick start (CPU — verifies pipeline end-to-end):
    python gnn_pipeline_v4_final.py --rna train_rna.h5ad --pro train_pro.h5ad --out out/ --epochs 50 --patience 10

AWS GPU (recommended):
    python gnn_pipeline_v4_final.py --rna train_rna.h5ad --pro train_pro.h5ad --out out/ --device cuda --amp
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
        Y_arc = np.arcsinh(Y_raw / 150.0).astype(np.float32)
        print("  Target transform: arcsinh(x/150)")

    coords = np.column_stack([
        rna.obs["array_row"].to_numpy(dtype=np.float32),
        rna.obs["array_col"].to_numpy(dtype=np.float32),
    ])
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

    # No internal split here - every one of the n_bins rows goes into the
    # loss. There is no held-out slice at all in this run (that's what the
    # separate 5-fold CV run is for). Checkpoint selection below tracks the
    # model's fit to this same training data, not a generalization estimate.
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

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(X_t, ei_local, ei_regional)
            loss = combined_loss(pred, Y_t)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        lr_now = sched.step()

        # Diagnostic-only: how well the model currently fits the training
        # data it just saw. Not a generalization estimate - there is no
        # held-out data in this run to measure that against.
        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_all = model(X_t, ei_local, ei_regional)
            tl = combined_loss(pred_all.float(), Y_t).item()
            _pred_np = pred_all.float().cpu().numpy()
            train_pr = mean_pearson_r(_pred_np, _Y_np)
            train_sr = mean_spearman_r(_pred_np, _Y_np)
            train_metric = 0.5 * train_pr + 0.5 * train_sr

        improved = train_metric > best_train
        if improved:
            best_train = train_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        history.append({"epoch": epoch, "train_loss": loss.item(), "eval_loss": tl,
                         "train_pearson": train_pr, "train_spearman": train_sr, "lr": lr_now})

        if epoch == 1 or epoch % 20 == 0 or improved:
            elapsed = (time.time() - t_start) / 60
            eta = elapsed / epoch * (args.epochs - epoch)
            star = " *" if improved else ""
            print(f"  Ep {epoch:>5}/{args.epochs} | train={loss.item():.4f}  eval={tl:.4f}  "
                  f"train_r={train_pr:.4f}  train_sr={train_sr:.4f}  patience={patience_ctr}/{args.patience}  "
                  f"lr={lr_now:.2e}  [{elapsed:.1f}min, ~{eta:.0f}min left]{star}")

        if patience_ctr >= args.patience:
            print(f"  Early stop at epoch {epoch} (best train_metric[0.5*r+0.5*sr]={best_train:.4f})")
            break

    model.load_state_dict(best_state)

    final_dir = os.path.join(out_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)

    torch.save(best_state, os.path.join(final_dir, "gnn_v4_final_model.pt"))
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
        "n_hvg": args.n_hvg,
        "n_train_bins": int(n_bins),
        "best_train_fit_metric": float(best_train),
        "note": ("best_train_fit_metric is measured on the SAME 100% of data the model "
                 "trained on (no internal split at all in this run) - it is NOT a "
                 "generalization estimate. Use the separate 5-fold CV run's honest "
                 "Pearson/Spearman/RMSE (0.6566 / 0.5625 / 0.7451) for that."),
    }
    with open(os.path.join(final_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  FINAL MODEL SAVED to {final_dir}")
    print(f"  Trained on all {n_bins:,} bins, no internal split. Best train-fit metric = {best_train:.4f}")
    print(f"  This is NOT the generalization estimate - use the 5-fold CV numbers for that.")
    print(f"{'='*65}")


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
             "team's reference preprocessing.py + the ridge-regression baseline, "
             "default) or 'arcsinh' (arcsinh(x/150))")

    parser.add_argument("--n_hvg",     type=int,   default=5000, help="HVGs; 0 = use ALL genes, no HVG filtering")
    parser.add_argument("--svd_dims",  type=int,   default=384,  help="TruncatedSVD components")
    parser.add_argument("--k",         type=int,   default=8,    help="local k-NN neighbours")
    parser.add_argument("--radius",    type=float, default=0.0,  help="regional graph radius; 0 = auto (3x median 8-NN distance)")
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

    print(f"\n{'='*65}")
    print("STEP 2/2  Training on 100% of the data")
    print(f"{'='*65}")
    train_final(args, rna_hvg_path, pro_raw_path, out_dir, device, use_amp)


if __name__ == "__main__":
    main()
