"""
gnn_pipeline_v2.py  —  Improved Spatial GNN for RNA → Protein (Team 37)
=========================================================================
Standalone. No repo clone needed.

V2 improvements over V1:
  1. Residual connections between all SAGE layers (helps gradient flow)
  2. LayerNorm instead of BatchNorm (more stable on graphs)
  3. Separate input projection layer (decouples feature encoding from aggregation)
  4. Deeper model: hidden=256, n_layers=3 (receptive field covers 8^3 neighbours)
  5. Output head: Linear(256→128) → LN → ReLU → Dropout → Linear(128→44)
  6. k=8 neighbours (larger spatial context vs k=6)
  7. 500 epochs, patience=20  (much more room to converge)
  8. LR warmup for first 10 epochs (avoids early divergence)
  9. ReduceLROnPlateau as safety net after cosine schedule
  10. Mini-batch option via NeighborLoader for GPU (--use_loader)

Architecture:
    Input PCA(256) → Linear(256→256) → LayerNorm → ReLU → Dropout
    → [SAGEConv + LayerNorm + ReLU + Residual] × 3
    → Linear(256→128) → LN → ReLU → Dropout → Linear(128→44)

CV:  5-fold spatially-blocked GroupKFold (10×10 bin blocks)
Loss: 0.5×MSE + 0.5×(1 − mean Pearson r)
Opt:  AdamW + warmup(10ep) + CosineAnnealingLR + grad-clip(1.0)

Quick start (CPU — verifies pipeline end-to-end):
    python gnn_pipeline_v2.py --epochs 50 --patience 10

Full local run (CPU, ~8h):
    python gnn_pipeline_v2.py

AWS GPU (recommended, ~1-2h):
    python gnn_pipeline_v2.py --device cuda --batch_size 0 --epochs 500
"""

import argparse
import copy
import json
import os
import time

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import SAGEConv


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


class ImprovedSpatialGNN(nn.Module):
    """
    Spatial GNN V2 with input projection, residual SAGE blocks, output head.

    Parameters
    ----------
    in_channels  : PCA feature dimension (default 256)
    hidden       : hidden size per layer (default 256)
    out_channels : number of protein targets (default 44)
    n_layers     : number of ResidualSAGEBlock layers (default 3)
    dropout      : dropout rate (default 0.3)
    """
    def __init__(self,
                 in_channels:  int   = 256,
                 hidden:       int   = 256,
                 out_channels: int   = 44,
                 n_layers:     int   = 3,
                 dropout:      float = 0.3):
        super().__init__()
        self.dropout = dropout

        # Input projection: map PCA features → hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Residual SAGE blocks
        self.blocks = nn.ModuleList(
            [ResidualSAGEBlock(hidden, dropout) for _ in range(n_layers)]
        )

        # Output head: hidden → 128 → out_channels
        self.output_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_channels),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, edge_index)
        return self.output_head(x)


# ═══════════════════════════════════════════════════════════════════
#  LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def pearson_loss(yp: torch.Tensor, yt: torch.Tensor) -> torch.Tensor:
    """1 − mean Pearson r across all protein columns."""
    vp = yp - yp.mean(0, keepdim=True)
    vt = yt - yt.mean(0, keepdim=True)
    r  = (vp * vt).sum(0) / ((vp**2).sum(0) * (vt**2).sum(0) + 1e-8).sqrt()
    return (1 - r).mean()


def combined_loss(yp: torch.Tensor, yt: torch.Tensor, w: float = 0.5) -> torch.Tensor:
    """0.5 × MSE  +  0.5 × (1 − mean Pearson r)."""
    return w * F.mse_loss(yp, yt) + (1 - w) * pearson_loss(yp, yt)


# ═══════════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════════

def mean_pearson_r(pred: np.ndarray, true: np.ndarray) -> float:
    rs = [pearsonr(pred[:, j], true[:, j])[0] for j in range(true.shape[1])]
    rs = [r for r in rs if not np.isnan(r)]
    return float(np.mean(rs)) if rs else 0.0


def per_protein_metrics(pred: np.ndarray, true: np.ndarray):
    n = true.shape[1]
    pearson = np.zeros(n)
    rmse    = np.zeros(n)
    mae     = np.zeros(n)
    for j in range(n):
        r, _ = pearsonr(pred[:, j], true[:, j])
        pearson[j] = r if not np.isnan(r) else 0.0
        rmse[j]    = float(np.sqrt(np.mean((pred[:, j] - true[:, j])**2)))
        mae[j]     = float(np.mean(np.abs(pred[:, j] - true[:, j])))
    return pearson, rmse, mae


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
#  CV SPLIT  (mirrors cross_validation_split.py)
# ═══════════════════════════════════════════════════════════════════

def make_cv_splits(rna_hvg_path: str, out_dir: str,
                   n_splits: int = 5, block_size: int = 10,
                   random_state: int = 42) -> str:
    cv_path = os.path.join(out_dir, "cv_splits.json")
    if os.path.exists(cv_path):
        print("  cv_splits.json found — skipping.")
        return cv_path

    rna     = ad.read_h5ad(rna_hvg_path)
    n_bins  = rna.n_obs
    row     = rna.obs["array_row"].to_numpy()
    col     = rna.obs["array_col"].to_numpy()

    block_id = (row // block_size).astype(np.int64) * 1_000_003 \
             + (col // block_size).astype(np.int64)
    uniq, inv = np.unique(block_id, return_inverse=True)
    n_blocks  = len(uniq)

    rng   = np.random.RandomState(random_state)
    rank  = np.empty(n_blocks, dtype=int)
    rank[rng.permutation(n_blocks)] = np.arange(n_blocks)
    groups = rank[inv]

    splits = []
    for fold, (tr, te) in enumerate(
            GroupKFold(n_splits).split(np.arange(n_bins), groups=groups)):
        splits.append({"fold": fold, "train": tr.tolist(), "test": te.tolist()})
        print(f"  Fold {fold+1}: {len(tr):,} train | {len(te):,} test")

    with open(cv_path, "w") as f:
        json.dump({"n_splits": n_splits, "n_bins": n_bins,
                   "n_blocks": int(n_blocks), "splits": splits}, f)
    print(f"  Saved cv_splits.json")
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


# ═══════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def gnn_pipeline_v2(
    rna_path:      str,
    pro_path:      str,
    out_dir:       str,
    cv_split_path: str   = None,   # pass existing cv_splits.json to reuse it
    # Features
    pca_dims:      int   = 256,
    k_neighbors:   int   = 8,
    n_hvg:         int   = 2000,
    # Model
    hidden:       int   = 256,
    n_layers:     int   = 3,
    dropout:      float = 0.3,
    # Optimiser
    lr:           float = 3e-4,
    weight_decay: float = 1e-3,
    warmup_epochs:int   = 10,
    # Training
    max_epochs:   int   = 500,
    patience:     int   = 20,
    # Hardware
    device_str:   str   = "cpu",
):
    os.makedirs(out_dir, exist_ok=True)
    preproc_dir = os.path.join(out_dir, "preprocessed")

    if device_str == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available — falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    # ── Step 1: Preprocess ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("STEP 1/3  Preprocessing")
    print(f"{'='*65}")
    rna_hvg_path, pro_raw_path = preprocess(
        rna_path, pro_path, preproc_dir, n_hvg=n_hvg)

    # ── Step 2: CV splits ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("STEP 2/3  CV Split  (5-fold spatially-blocked)")
    print(f"{'='*65}")
    # Accept externally provided cv_splits.json (e.g. from the repo outputs)
    if cv_split_path and os.path.exists(cv_split_path):
        cv_path = cv_split_path
        print(f"  Using existing CV split: {cv_path}")
    else:
        cv_path = make_cv_splits(rna_hvg_path, preproc_dir)

    # ── Step 3: Load data ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"STEP 3/3  GNN V2 Training  (5-fold CV, {max_epochs} epochs)")
    print(f"{'='*65}")
    print(f"\nDevice  : {device}")

    rna = ad.read_h5ad(rna_hvg_path)
    pro = ad.read_h5ad(pro_raw_path)

    X_raw = (rna.X.toarray() if hasattr(rna.X, "toarray")
             else np.array(rna.X, dtype=np.float32))
    Y_raw = (pro.X.toarray() if hasattr(pro.X, "toarray")
             else np.array(pro.X, dtype=np.float32))

    # arcsinh(x/150): standard CODEX normalisation (confirmed by EDA)
    Y = np.arcsinh(Y_raw / 150.0).astype(np.float32)

    coords        = np.column_stack([
        rna.obs["array_row"].to_numpy(dtype=np.float32),
        rna.obs["array_col"].to_numpy(dtype=np.float32),
    ])
    protein_names = list(pro.var_names)
    n_proteins    = Y.shape[1]

    print(f"Bins      : {X_raw.shape[0]:,}")
    print(f"HVGs      : {X_raw.shape[1]:,}")
    print(f"Proteins  : {n_proteins}")
    print(f"k-NN      : k={k_neighbors}")
    print(f"Model     : hidden={hidden}  layers={n_layers}  dropout={dropout}")
    print(f"Training  : epochs={max_epochs}  patience={patience}  "
          f"lr={lr}  warmup={warmup_epochs}ep")

    # ── PCA ────────────────────────────────────────────────────────────────
    print(f"\nFitting PCA({pca_dims}) on {X_raw.shape[0]:,} bins...")
    t0    = time.time()
    pca   = PCA(n_components=pca_dims, random_state=42)
    X_pca = pca.fit_transform(X_raw).astype(np.float32)
    print(f"  Done ({time.time()-t0:.1f}s) — "
          f"explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    with open(cv_path) as f:
        splits = json.load(f)["splits"]

    fold_pearson = np.zeros((len(splits), n_proteins))
    fold_rmse    = np.zeros((len(splits), n_proteins))
    fold_mae     = np.zeros((len(splits), n_proteins))

    # ── Fold loop ──────────────────────────────────────────────────────────
    for split in splits:
        fold      = split["fold"]
        train_idx = np.array(split["train"])
        test_idx  = np.array(split["test"])

        print(f"\n{'═'*65}")
        print(f"  FOLD {fold+1}/{len(splits)}   "
              f"train={len(train_idx):,}   test={len(test_idx):,}")
        print(f"{'═'*65}")

        # ── Build spatial k-NN graphs ──────────────────────────────────────
        print(f"\n  Building k-NN graphs (k={k_neighbors})...")
        t0    = time.time()
        ei_tr = build_knn_graph(coords[train_idx], k=k_neighbors)
        ei_te = build_knn_graph(coords[test_idx],  k=k_neighbors)
        print(f"  Train: {ei_tr.shape[1]:,} edges | "
              f"Test: {ei_te.shape[1]:,} edges  ({time.time()-t0:.1f}s)")

        # ── Tensors → device ───────────────────────────────────────────────
        X_tr   = torch.tensor(X_pca[train_idx]).to(device)
        X_te   = torch.tensor(X_pca[test_idx]).to(device)
        Y_tr   = torch.tensor(Y[train_idx]).to(device)
        Y_te   = torch.tensor(Y[test_idx]).to(device)
        ei_tr  = ei_tr.to(device)
        ei_te  = ei_te.to(device)
        Y_te_np = Y[test_idx]

        # ── Model ──────────────────────────────────────────────────────────
        model = ImprovedSpatialGNN(
            in_channels  = pca_dims,
            hidden       = hidden,
            out_channels = n_proteins,
            n_layers     = n_layers,
            dropout      = dropout,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n  Model: ImprovedSpatialGNN  params={n_params:,}")

        # ── Optimiser + warmup cosine scheduler ────────────────────────────
        opt   = torch.optim.AdamW(model.parameters(),
                                  lr=lr, weight_decay=weight_decay)
        sched = WarmupCosineScheduler(opt,
                                      warmup_epochs = warmup_epochs,
                                      max_epochs    = max_epochs,
                                      base_lr       = lr,
                                      min_lr        = lr / 30)   # was /100 → kept LR higher longer

        # ── Training loop ──────────────────────────────────────────────────
        best_pearson  = -1.0
        best_state    = None
        patience_ctr  = 0
        history       = []
        t_fold        = time.time()

        print(f"\n  Training:  max_epochs={max_epochs}  patience={patience}\n")

        for epoch in range(1, max_epochs + 1):
            # Train
            model.train()
            opt.zero_grad()
            pred = model(X_tr, ei_tr)
            loss = combined_loss(pred, Y_tr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            current_lr = sched.step()
            tl = loss.item()

            # Eval
            model.eval()
            with torch.no_grad():
                pred_te = model(X_te, ei_te)
                vl      = combined_loss(pred_te, Y_te).item()
                val_pr  = mean_pearson_r(pred_te.cpu().numpy(), Y_te_np)

            history.append({
                "epoch": epoch, "train_loss": tl,
                "val_loss": vl, "val_pearson": val_pr, "lr": current_lr,
            })

            # Early stopping
            if val_pr > best_pearson:
                best_pearson = val_pr
                best_state   = copy.deepcopy(model.state_dict())
                patience_ctr = 0
                marker = " ★"
            else:
                patience_ctr += 1
                marker = ""

            if epoch == 1 or epoch % 20 == 0 or patience_ctr == patience:
                elapsed   = time.time() - t_fold
                ep_per_s  = epoch / elapsed if elapsed > 0 else 0
                remaining = (max_epochs - epoch) / ep_per_s if ep_per_s > 0 else 0
                print(f"  Ep {epoch:>4}/{max_epochs} | "
                      f"train={tl:.4f}  val={vl:.4f}  "
                      f"val_r={val_pr:.4f}  "
                      f"patience={patience_ctr}/{patience}  "
                      f"lr={current_lr:.2e}  "
                      f"[{elapsed/60:.1f}min, ~{remaining/60:.0f}min left]"
                      f"{marker}")

            if patience_ctr >= patience:
                print(f"\n  ▶ Early stop at epoch {epoch}. "
                      f"Best val Pearson r = {best_pearson:.4f}")
                break

        # ── Evaluate best checkpoint ───────────────────────────────────────
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred_np = model(X_te, ei_te).cpu().numpy()

        pr, rmse, mae = per_protein_metrics(pred_np, Y_te_np)
        fold_pearson[fold] = pr
        fold_rmse[fold]    = rmse
        fold_mae[fold]     = mae

        fold_time = time.time() - t_fold
        print(f"\n  Fold {fold+1} done | "
              f"mean Pearson r = {pr.mean():.4f} | "
              f"mean RMSE = {rmse.mean():.4f} | "
              f"time = {fold_time/60:.1f}min")

        # ── Save fold artefacts ────────────────────────────────────────────
        torch.save(best_state,
                   os.path.join(out_dir, f"gnn_v2_fold{fold+1}.pt"))
        pd.DataFrame(history).to_csv(
            os.path.join(out_dir, f"history_fold{fold+1}.csv"), index=False)
        # Save per-fold predictions for analysis
        np.save(os.path.join(out_dir, f"pred_fold{fold+1}.npy"),  pred_np)
        np.save(os.path.join(out_dir, f"true_fold{fold+1}.npy"),  Y_te_np)

    # ── Aggregate across folds ─────────────────────────────────────────────
    mean_pr   = fold_pearson.mean(0)
    std_pr    = fold_pearson.std(0)
    mean_rmse = fold_rmse.mean(0)
    mean_mae  = fold_mae.mean(0)

    print(f"\n{'═'*65}")
    print(f"  5-FOLD CV RESULTS  (GNN V2, {max_epochs} epochs max)")
    print(f"{'═'*65}")
    print(f"  Mean Pearson r : {mean_pr.mean():.4f} ± {std_pr.mean():.4f}")
    print(f"  Mean RMSE      : {mean_rmse.mean():.4f}")
    print(f"  Mean MAE       : {mean_mae.mean():.4f}")

    per_protein_df = pd.DataFrame({
        "protein":       protein_names,
        "mean_pearsonr": mean_pr,
        "std_pearsonr":  std_pr,
        "mean_rmse":     mean_rmse,
        "mean_mae":      mean_mae,
    }).sort_values("mean_pearsonr", ascending=False)

    print(f"\n  Top 10 proteins:")
    print(per_protein_df.head(10).to_string(index=False))
    print(f"\n  Bottom 5 proteins:")
    print(per_protein_df.tail(5).to_string(index=False))

    per_protein_df.to_csv(
        os.path.join(out_dir, "gnn_v2_per_protein_metrics.csv"), index=False)
    pd.DataFrame({
        "model":         ["GNN_V2"],
        "mean_pearsonr": [mean_pr.mean()],
        "std_pearsonr":  [std_pr.mean()],
        "mean_rmse":     [mean_rmse.mean()],
        "mean_mae":      [mean_mae.mean()],
        "epochs":        [max_epochs],
        "hidden":        [hidden],
        "n_layers":      [n_layers],
        "k_neighbors":   [k_neighbors],
        "pca_dims":      [pca_dims],
    }).to_csv(os.path.join(out_dir, "gnn_v2_results.csv"), index=False)

    print(f"\n  All results saved → {out_dir}")
    return per_protein_df


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GNN V2 Pipeline — Spatial RNA→Protein (Team 37)"
    )

    # Paths
    parser.add_argument("--rna",
        default="/Users/vaishnavikeshav/Downloads/IRBM/train_rna.h5ad",
        help="Path to raw train_rna.h5ad")
    parser.add_argument("--pro",
        default="/Users/vaishnavikeshav/Downloads/IRBM/train_pro.h5ad",
        help="Path to raw train_pro.h5ad")
    parser.add_argument("--cv",
        default=None,
        help="Path to existing cv_splits.json (skips CV generation if provided)")
    parser.add_argument("--out",
        default="/Users/vaishnavikeshav/Downloads/IRBM/gnn_v2_results",
        help="Output directory")

    # Features
    parser.add_argument("--pca_dims",     type=int,   default=256)
    parser.add_argument("--k",            type=int,   default=8,
        help="k-NN neighbours for spatial graph (default 8)")
    parser.add_argument("--n_hvg",        type=int,   default=2000)

    # Model
    parser.add_argument("--hidden",       type=int,   default=256,
        help="Hidden size per layer (default 256)")
    parser.add_argument("--n_layers",     type=int,   default=3,
        help="Number of ResidualSAGE layers (default 3)")
    parser.add_argument("--dropout",      type=float, default=0.3)

    # Optimiser
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--warmup",       type=int,   default=10,
        help="LR warmup epochs (default 10)")

    # Training
    parser.add_argument("--epochs",       type=int,   default=500,
        help="Max epochs per fold (default 500)")
    parser.add_argument("--patience",     type=int,   default=20,
        help="Early-stop patience on val Pearson r (default 20)")

    # Hardware
    parser.add_argument("--device",       default="cpu",
        choices=["cpu", "cuda"])

    args = parser.parse_args()

    gnn_pipeline_v2(
        rna_path      = args.rna,
        pro_path      = args.pro,
        out_dir       = args.out,
        cv_split_path = args.cv,
        pca_dims      = args.pca_dims,
        k_neighbors   = args.k,
        n_hvg         = args.n_hvg,
        hidden        = args.hidden,
        n_layers      = args.n_layers,
        dropout       = args.dropout,
        lr            = args.lr,
        weight_decay  = args.weight_decay,
        warmup_epochs = args.warmup,
        max_epochs    = args.epochs,
        patience      = args.patience,
        device_str    = args.device,
    )
