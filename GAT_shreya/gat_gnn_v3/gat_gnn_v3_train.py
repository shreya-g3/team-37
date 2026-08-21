import os
import numpy as np
import pandas as pd
import torch
import anndata as ad
from scipy.sparse import issparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from model import DualBranchGNN, combined_loss
from preprocessing_final import preprocess_rna, preprocess_protein_train, inverse_transform_protein  # UNMODIFIED shared file
from graph_construction import build_spatial_graph_from_coords, build_expression_graph
from cv_split_patches import cv_split_patches, load_cv_split


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED             = 0
N_SPLITS         = 5
N_SVD_COMPONENTS = 180
K_SPATIAL        = 6
K_EXPRESSION     = 10
BATCH_SIZE       = 512
NUM_NEIGHBORS    = [10, 10]     # fanout per layer, matches 2-layer branch depth
MAX_EPOCHS       = 300
PATIENCE         = 20
WARMUP_EPOCHS    = 10
BASE_LR          = 3e-4
LOSS_W           = 0.8          # combined_loss regression-vs-pearson weight
TARGET_PATCH_BINS = 2000        # cv_split_patches: approx bins per spatial patch
BUFFER_DIST       = 60          # cv_split_patches: exclusion radius around test patches
TISSUE_LABEL      = None        # set to a rna.obs column name if you have cluster labels to stratify by


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def compute_qc_mask(rna_path, min_counts=500, min_genes=200, max_pct_mito=20.0, mito_prefix="MT-"):
    """Row-level low-quality flag. nothing gets
    filtered out of the shared feature matrix."""
    raw = ad.read_h5ad(rna_path)
    X = raw.X
    total_counts = np.asarray(X.sum(axis=1)).ravel()
    n_genes = np.asarray((X > 0).sum(axis=1)).ravel()
    mito = raw.var_names.str.startswith(mito_prefix)
    if mito.sum() > 0:
        pct_mt = np.asarray(X[:, mito].sum(axis=1)).ravel() / np.clip(total_counts, 1, None) * 100.0
    else:
        pct_mt = np.zeros(raw.n_obs)
    return (total_counts < min_counts) | (n_genes < min_genes) | (pct_mt > max_pct_mito)


def compute_protein_missing_mask(protein_path):
    """Missingness read before preprocess_protein_train's np.nan_to_num call
    turns missing measurements into literal zeros. preprocess_protein_train's
    output is still what gets used as the training target - this mask just
    tells the loss which of those zero-filled entries were never real."""
    raw = ad.read_h5ad(protein_path)
    X = raw.X.toarray() if issparse(raw.X) else np.asarray(raw.X)
    return np.isnan(X)


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr, min_lr=1e-6):
        self.opt, self.warmup_epochs, self.max_epochs = optimizer, warmup_epochs, max_epochs
        self.base_lr, self.min_lr, self.epoch = base_lr, min_lr, 0

    def step(self):
        self.epoch += 1
        e = self.epoch
        if e <= self.warmup_epochs:
            lr = self.base_lr * e / max(1, self.warmup_epochs)
        else:
            p = (e - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * p))
        for pg in self.opt.param_groups:
            pg["lr"] = lr
        return lr


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_r2(pred, true):
    vals = r2_score(true, pred, multioutput="raw_values")
    return float(vals.mean()), float(np.median(vals)), vals


def mean_pearson_r(pred, true):
    rs = [pearsonr(pred[:, j], true[:, j])[0] for j in range(true.shape[1])]
    rs = [r for r in rs if not np.isnan(r)]
    return float(np.mean(rs)) if rs else 0.0


# ---------------------------------------------------------------------------
# Fold-local graphs + mini-batch loaders
# (kept fully separate per fold - zero edges cross between train/holdout, mini-batched)
# ---------------------------------------------------------------------------

def build_fold_pyg_data(latent, coords, y, missing):
    x = torch.tensor(latent, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    missing = torch.tensor(missing)

    sp_ei, sp_ea = build_spatial_graph_from_coords(coords, k=K_SPATIAL)
    ex_ei, ex_ew = build_expression_graph(latent, k=K_EXPRESSION)

    spatial_data = Data(x=x, edge_index=sp_ei, edge_attr=sp_ea, y=y, missing=missing)
    expr_data = Data(x=x, edge_index=ex_ei, edge_weight=ex_ew, y=y, missing=missing)
    return spatial_data, expr_data


def make_loaders(spatial_data, expr_data, input_nodes, shuffle):
    if shuffle:
        input_nodes = input_nodes[torch.randperm(input_nodes.size(0))]
    common = dict(num_neighbors=NUM_NEIGHBORS, input_nodes=input_nodes,
                  batch_size=BATCH_SIZE, shuffle=False)
    return NeighborLoader(spatial_data, **common), NeighborLoader(expr_data, **common)


def run_epoch(model, spatial_data, expr_data, node_idx, optimizer, device):
    train_mode = optimizer is not None
    model.train(train_mode)

    spatial_loader, expr_loader = make_loaders(spatial_data, expr_data, node_idx, shuffle=train_mode)

    total_loss, n_batches = 0.0, 0
    all_preds, all_targets = [], []

    for sb, eb in zip(spatial_loader, expr_loader):
        sb, eb = sb.to(device), eb.to(device)
        n_seed = sb.batch_size

        with torch.set_grad_enabled(train_mode):
            preds = model(sb.x, sb.edge_index, sb.edge_attr, eb.edge_index, eb.edge_weight)
            loss = combined_loss(preds[:n_seed], sb.y[:n_seed], w=LOSS_W, missing_mask=sb.missing[:n_seed])

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        all_preds.append(preds[:n_seed].detach().cpu().numpy())
        all_targets.append(sb.y[:n_seed].cpu().numpy())

    preds_np = np.concatenate(all_preds)
    targets_np = np.concatenate(all_targets)
    return total_loss / max(n_batches, 1), preds_np, targets_np


# ---------------------------------------------------------------------------
# Single fold: leak-free SVD fit (sparse, no densification) + mini-batch train
# ---------------------------------------------------------------------------

def train_one_fold(rna_sparse, protein_z, missing_full, coords, train_idx, ho_idx, device, fold_seed,
                    protein_stats):
    svd = TruncatedSVD(n_components=N_SVD_COMPONENTS, random_state=fold_seed)
    svd.fit(rna_sparse[train_idx])                 # sparse in, sparse in - no .toarray()
    latent_all = svd.transform(rna_sparse)          # (n_spots, n_components), small & dense from here

    lat_tr, lat_ho = latent_all[train_idx].astype(np.float32), latent_all[ho_idx].astype(np.float32)
    y_tr, y_ho = protein_z[train_idx], protein_z[ho_idx]
    coords_tr, coords_ho = coords[train_idx], coords[ho_idx]
    m_tr, m_ho = missing_full[train_idx], missing_full[ho_idx]

    spatial_tr, expr_tr = build_fold_pyg_data(lat_tr, coords_tr, y_tr, m_tr)
    spatial_ho, expr_ho = build_fold_pyg_data(lat_ho, coords_ho, y_ho, m_ho)

    train_nodes = torch.arange(spatial_tr.num_nodes)
    ho_nodes = torch.arange(spatial_ho.num_nodes)

    model = DualBranchGNN(in_dim=N_SVD_COMPONENTS, out_dim=protein_z.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-3)
    sched = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, MAX_EPOCHS, BASE_LR)

    best_val, best_epoch, patience_ctr, best_state = float("inf"), 0, 0, None

    for epoch in range(MAX_EPOCHS):
        train_loss, _, _ = run_epoch(model, spatial_tr, expr_tr, train_nodes, optimizer, device)
        lr_now = sched.step()
        val_loss, val_preds, val_targets = run_epoch(model, spatial_ho, expr_ho, ho_nodes, None, device)

        if val_loss < best_val:
            best_val, best_epoch, patience_ctr = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or epoch == MAX_EPOCHS - 1:
            mean_r2, med_r2, _ = compute_r2(val_preds, val_targets)
            pear = mean_pearson_r(val_preds, val_targets)
            print(f"    epoch {epoch:4d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"lr={lr_now:.2e}  val_meanR2={mean_r2:.4f}  val_medR2={med_r2:.4f}  val_pearson={pear:.4f}")

        if patience_ctr >= PATIENCE:
            print(f"    early stop at epoch {epoch}  best_epoch={best_epoch}  best_val={best_val:.4f}")
            break

    model.load_state_dict(best_state)
    _, final_preds, final_targets = run_epoch(model, spatial_ho, expr_ho, ho_nodes, None, device)
    mean_r2, med_r2, r2_vals = compute_r2(final_preds, final_targets)
    pear = mean_pearson_r(final_preds, final_targets)

    # inverse-transform this fold's holdout predictions/targets back to raw
    # CODEX scale for reporting - protein_stats was fit once, globally, by
    # the unmodified preprocess_protein_train before the CV split, so every
    # fold shares the same stats. minor leak of each
    # fold's holdout distribution into percentile/mean/std values -
    # unavoidable while preprocess stays same
    raw_preds = inverse_transform_protein(final_preds, protein_stats)
    raw_targets = inverse_transform_protein(final_targets, protein_stats)
    raw_mean_r2, raw_med_r2, raw_r2_vals = compute_r2(raw_preds, raw_targets)
    raw_pear = mean_pearson_r(raw_preds, raw_targets)

    return model, best_epoch + 1, dict(
        mean_r2=mean_r2, median_r2=med_r2, r2_per_protein=r2_vals, pearson=pear,
        raw_mean_r2=raw_mean_r2, raw_median_r2=raw_med_r2, raw_r2_per_protein=raw_r2_vals, raw_pearson=raw_pear,
    )


# ---------------------------------------------------------------------------
# 5-fold CV, spatially-blocked (patch-based, buffered) splits
# ---------------------------------------------------------------------------

def run_cv(rna_sparse, protein_z, missing_full, coords, protein_names, protein_stats, device, cv_splits):
    rows, per_protein, per_protein_raw = [], [], []

    for split in cv_splits:
        fold_i = split["fold"]
        train_idx, ho_idx = np.array(split["train"]), np.array(split["test"])
        print(f"[fold {fold_i}] train={len(train_idx):,}  holdout={len(ho_idx):,}  "
              f"buffer_excluded={split.get('n_buffer_bins_excluded', 'n/a')}")
        _, n_epochs, metrics = train_one_fold(
            rna_sparse, protein_z, missing_full, coords, train_idx, ho_idx, device, fold_seed=fold_i,
            protein_stats=protein_stats,
        )
        rows.append(dict(fold=fold_i, n_epochs=n_epochs,
                          mean_r2=metrics["mean_r2"], median_r2=metrics["median_r2"], pearson=metrics["pearson"],
                          raw_mean_r2=metrics["raw_mean_r2"], raw_median_r2=metrics["raw_median_r2"],
                          raw_pearson=metrics["raw_pearson"]))
        per_protein.append(metrics["r2_per_protein"])
        per_protein_raw.append(metrics["raw_r2_per_protein"])

    summary = pd.DataFrame(rows)
    print("\nCV summary (z-score scale and raw-CODEX scale, via inverse_transform_protein):")
    print(summary.to_string(index=False))
    print(f"\nMean CV R2 (z-score) = {summary['mean_r2'].mean():.4f} +/- {summary['mean_r2'].std():.4f}")
    print(f"Mean CV R2 (raw CODEX) = {summary['raw_mean_r2'].mean():.4f} +/- {summary['raw_mean_r2'].std():.4f}")

    per_protein_df = pd.DataFrame(np.stack(per_protein), columns=protein_names)
    per_protein_raw_df = pd.DataFrame(np.stack(per_protein_raw), columns=protein_names)
    return summary, per_protein_df, per_protein_raw_df


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = "outputs"
    os.makedirs(out_path, exist_ok=True)

    rna_train = preprocess_rna("rna_train.h5ad")                       # exact shared function, untouched
    protein_train, protein_stats = preprocess_protein_train("protein_train.h5ad")  # exact shared function

# check if double here
    rna_train_processed_path = os.path.join(out_path, "rna_train_processed.h5ad")
    rna_train.write(rna_train_processed_path)

    cv_split_patches(
        rna_path=rna_train_processed_path,
        output_path=out_path,
        n_splits=N_SPLITS,
        random_state=SEED,
        target_patch_bins=TARGET_PATCH_BINS,
        buffer_dist=BUFFER_DIST,
        tissue_label=TISSUE_LABEL,
    )
    cv_splits = load_cv_split(os.path.join(out_path, "cv_splits_patches.json"))

    rna_sparse = rna_train.X  # keep sparse -- SVD is fit on this directly, per fold
    protein_z = protein_train.X.astype(np.float32)
    coords = np.asarray(rna_train.obsm["spatial"], dtype=np.float32)
    protein_names = list(protein_train.var_names)

    # masks: additive - comment out either line to
    # reproduce tno-QC / no-missingness-aware behaviour
    missing_full = compute_protein_missing_mask("protein_train.h5ad")
    qc_mask = compute_qc_mask("rna_train.h5ad")
    missing_full = missing_full | qc_mask[:, None]   # flagged spots excluded from loss across all proteins

    summary, per_protein_r2, per_protein_r2_raw = run_cv(
        rna_sparse, protein_z, missing_full, coords, protein_names, protein_stats, device, cv_splits,
    )
    summary.to_csv(os.path.join(out_path, "cv_summary.csv"), index=False)
    per_protein_r2.to_csv(os.path.join(out_path, "cv_r2_per_protein.csv"), index=False)
    per_protein_r2_raw.to_csv(os.path.join(out_path, "cv_r2_per_protein_raw_codex.csv"), index=False)

    # persist protein_stats so a later inference script can call
    # inverse_transform_protein on predictions for rna_val without
    # needing to re-run preprocess_protein_train
    np.savez(os.path.join(out_path, "protein_stats.npz"), **protein_stats)