"""Improved MLP pipeline for Visium HD RNA to CODEX protein prediction.

This version is structured for reassessment feedback:
- spatial cross-validation is configurable;
- HVG selection, scaling, and PCA are fitted inside each training fold;
- defaults are portable instead of machine-specific;
- metrics and checkpoints are saved in a reproducible project layout.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class PipelineConfig:
    rna: str = "data/train_rna.h5ad"
    protein: str = "data/train_pro.h5ad"
    valid_rna: str = "data/valid_rna.h5ad"
    out_dir: str = "reports/mlp_results"
    n_splits: int = 5
    split_strategy: str = "spatial"
    block_size: int = 10
    feature_selection: str = "variance"
    n_hvg: int = 2000
    pca_dims: int = 256
    hidden: int = 512
    dropout: float = 0.30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 500
    patience: int = 25
    batch_size: int = 4096
    target_scale: float = 150.0
    standardize_targets: bool = True
    loss_mse_weight: float = 0.50
    device: str = "auto"
    seed: int = 42
    max_folds: int | None = None
    sample_n: int | None = None
    export_valid_predictions: bool = True


class ResidualBlock(nn.Module):
    """Residual MLP block with batch normalization and dropout."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class MLPRegressor(nn.Module):
    """Encoder, residual trunk, and decoder for multi-output regression."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.trunk = nn.Sequential(
            ResidualBlock(hidden, dropout),
            ResidualBlock(hidden, dropout),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.trunk(x)
        return self.decoder(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU.")
        device = "cpu"
    return torch.device(device)


def dense_float32(x) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def _to_memory(adata: ad.AnnData) -> ad.AnnData:
    if hasattr(adata, "to_memory"):
        return adata.to_memory()
    return adata.copy()


def load_aligned_data(
    rna_path: Path,
    protein_path: Path,
    sample_n: int | None = None,
    seed: int = 42,
) -> tuple[ad.AnnData, ad.AnnData]:
    backed = sample_n is not None and sample_n > 0
    rna = ad.read_h5ad(rna_path, backed="r" if backed else None)
    protein = ad.read_h5ad(protein_path, backed="r" if backed else None)

    if rna.obs_names.equals(protein.obs_names):
        print("RNA/protein observations already aligned; avoiding full matrix copy.")
        if backed:
            n = min(sample_n, rna.n_obs)
            rng = np.random.RandomState(seed)
            sample_idx = np.sort(rng.choice(rna.n_obs, size=n, replace=False))
            print(f"Using sample_n={n:,} observations from data folder for smoke test.")
            rna_sample = _to_memory(rna[sample_idx])
            protein_sample = _to_memory(protein[sample_idx])
            rna.file.close()
            protein.file.close()
            return rna_sample, protein_sample
        return rna, protein

    common = rna.obs_names.intersection(protein.obs_names)
    if len(common) == 0:
        if rna.n_obs != protein.n_obs:
            raise ValueError(
                "RNA and protein AnnData objects have no shared obs_names and "
                f"different row counts ({rna.n_obs} vs {protein.n_obs})."
            )
        print("No shared obs_names found; assuming both files are already row-aligned.")
        if backed:
            n = min(sample_n, rna.n_obs)
            rng = np.random.RandomState(seed)
            sample_idx = np.sort(rng.choice(rna.n_obs, size=n, replace=False))
            print(f"Using sample_n={n:,} row-aligned observations from data folder for smoke test.")
            rna_sample = _to_memory(rna[sample_idx])
            protein_sample = _to_memory(protein[sample_idx])
            rna.file.close()
            protein.file.close()
            return rna_sample, protein_sample
        return rna, protein

    if len(common) != rna.n_obs or len(common) != protein.n_obs:
        print(f"Aligning RNA/protein to {len(common):,} shared spatial observations.")

    print("RNA/protein obs_names are not in the same order; creating aligned views.")
    if backed:
        n = min(sample_n, len(common))
        rng = np.random.RandomState(seed)
        common_sample = common[np.sort(rng.choice(len(common), size=n, replace=False))]
        print(f"Using sample_n={n:,} aligned observations from data folder for smoke test.")
        rna_sample = _to_memory(rna[common_sample])
        protein_sample = _to_memory(protein[common_sample])
        rna.file.close()
        protein.file.close()
        return rna_sample, protein_sample
    return rna[common], protein[common]


def normalize_rna(rna: ad.AnnData) -> ad.AnnData:
    if rna.is_view:
        print("Materializing aligned RNA view before normalization.")
        rna = rna.copy()
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    return rna


def make_spatial_splits(
    obs: pd.DataFrame,
    n_splits: int,
    block_size: int,
    seed: int,
    split_strategy: str = "spatial",
) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(obs))
    if split_strategy == "kfold":
        print("Using shuffled KFold split strategy.")
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(indices))
    if split_strategy != "spatial":
        raise ValueError("split_strategy must be 'spatial' or 'kfold'.")

    coords = None
    for row_col in [("array_row", "array_col"), ("row", "col"), ("y", "x")]:
        if row_col[0] in obs.columns and row_col[1] in obs.columns:
            coords = row_col
            break

    if coords is None:
        print("Spatial coordinate columns not found; using shuffled KFold.")
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(indices))

    row = pd.to_numeric(obs[coords[0]], errors="coerce").to_numpy()
    col = pd.to_numeric(obs[coords[1]], errors="coerce").to_numpy()
    if np.isnan(row).any() or np.isnan(col).any():
        print("Spatial coordinates contain missing values; using shuffled KFold.")
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(indices))

    block_id = (row // block_size).astype(np.int64) * 1_000_003 + (
        col // block_size
    ).astype(np.int64)
    unique_blocks = np.unique(block_id)
    if len(unique_blocks) < n_splits:
        print("Too few spatial blocks for GroupKFold; using shuffled KFold.")
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(indices))

    print(
        f"Using spatial GroupKFold with {len(unique_blocks):,} blocks "
        f"from {coords[0]}/{coords[1]}."
    )
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(indices, groups=block_id))


def select_hvg_train_only(
    rna_norm: ad.AnnData,
    train_idx: np.ndarray,
    n_hvg: int,
) -> list[str]:

    train_rna = rna_norm[train_idx].copy()

    # Remove genes expressed in very few cells
    sc.pp.filter_genes(train_rna, min_cells=3)

    X = dense_float32(train_rna.X)

    if X.shape[1] == 0:
        raise RuntimeError("No genes remain after filtering.")

    gene_var = np.var(X, axis=0)

    ranking = np.argsort(gene_var)[::-1]

    top = ranking[: min(n_hvg, len(ranking))]

    return train_rna.var_names[top].tolist()


def select_supervised_genes_train_only(
    rna_norm: ad.AnnData,
    y: np.ndarray,
    train_idx: np.ndarray,
    n_hvg: int,
) -> list[str]:
    train_rna = rna_norm[train_idx].copy()
    sc.pp.filter_genes(train_rna, min_cells=3)
    x_train = dense_float32(train_rna.X)
    y_train = y[train_idx]

    if x_train.shape[1] == 0:
        raise RuntimeError("No genes remain after filtering.")

    x_centered = x_train - x_train.mean(axis=0, keepdims=True)
    y_centered = y_train - y_train.mean(axis=0, keepdims=True)
    x_norm = np.sqrt(np.square(x_centered).sum(axis=0)).clip(min=1e-8)
    y_norm = np.sqrt(np.square(y_centered).sum(axis=0)).clip(min=1e-8)
    corr = (x_centered.T @ y_centered) / (x_norm[:, None] * y_norm[None, :])
    scores = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0).max(axis=1)
    ranking = np.argsort(scores)[::-1]
    top = ranking[: min(n_hvg, len(ranking))]
    return train_rna.var_names[top].tolist()


def make_fold_features(
    rna_norm: ad.AnnData,
    y: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    feature_selection: str,
    n_hvg: int,
    pca_dims: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, float], StandardScaler, PCA]:
    if feature_selection == "variance":
        hvgs = select_hvg_train_only(rna_norm, train_idx, n_hvg)
    elif feature_selection == "supervised_correlation":
        hvgs = select_supervised_genes_train_only(rna_norm, y, train_idx, n_hvg)
    else:
        raise ValueError(
            "feature_selection must be 'variance' or 'supervised_correlation'."
        )

    x_train = dense_float32(rna_norm[train_idx, hvgs].X)
    x_valid = dense_float32(rna_norm[valid_idx, hvgs].X)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_valid = scaler.transform(x_valid).astype(np.float32)

    n_components = min(pca_dims, x_train.shape[0] - 1, x_train.shape[1])
    if n_components < 1:
        raise ValueError("Not enough training samples/features for PCA.")
    pca = PCA(n_components=n_components, random_state=seed)
    x_train = pca.fit_transform(x_train).astype(np.float32)
    x_valid = pca.transform(x_valid).astype(np.float32)

    info = {
        "feature_selection": feature_selection,
        "n_hvg_selected": float(len(hvgs)),
        "pca_components": float(n_components),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
    }
    return x_train, x_valid, hvgs, info, scaler, pca


def transform_rna_features(
    rna_norm: ad.AnnData,
    hvgs: list[str],
    scaler: StandardScaler,
    pca: PCA,
) -> np.ndarray:
    missing = [gene for gene in hvgs if gene not in rna_norm.var_names]
    if missing:
        raise ValueError(
            f"valid_rna is missing {len(missing)} selected genes. "
            f"First missing gene: {missing[0]}"
        )
    x = dense_float32(rna_norm[:, hvgs].X)
    x = scaler.transform(x).astype(np.float32)
    return pca.transform(x).astype(np.float32)


def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_centered = pred - pred.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=0)
    denominator = torch.sqrt(
        (pred_centered.square().sum(dim=0) * target_centered.square().sum(dim=0)).clamp_min(1e-8)
    )
    corr = numerator / denominator
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return 1.0 - corr.mean()


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float,
) -> torch.Tensor:
    return mse_weight * F.mse_loss(pred, target) + (1.0 - mse_weight) * pearson_loss(
        pred, target
    )


def batch_predict(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


def metric_frame(pred: np.ndarray, true: np.ndarray, protein_names: Iterable[str]) -> pd.DataFrame:
    rows = []
    for j, protein in enumerate(protein_names):
        y_pred = pred[:, j]
        y_true = true[:, j]
        if np.std(y_pred) == 0 or np.std(y_true) == 0:
            pearson = 0.0
        else:
            pearson = float(np.corrcoef(y_pred, y_true)[0, 1])
            if math.isnan(pearson):
                pearson = 0.0
        err = y_pred - y_true
        rows.append(
            {
                "protein": protein,
                "pearson_r": pearson,
                "rmse": float(np.sqrt(np.mean(err**2))),
                "mae": float(np.mean(np.abs(err))),
            }
        )
    return pd.DataFrame(rows)


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._")
    return cleaned or "protein"


def first_existing_column(obs: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    for column in candidates:
        if column in obs.columns:
            return obs[column]
    return None


def build_submission_frame(
    obs: pd.DataFrame,
    obs_names: Iterable[str],
    pred_expression: np.ndarray,
    protein_names: Iterable[str],
) -> pd.DataFrame:
    obs_names = pd.Index(list(map(str, obs_names)))
    barcode = first_existing_column(obs, ["barcode", "Barcode", "spot_id", "Spot_ID"])
    pxl_row = first_existing_column(obs, ["pxl_row_in_fullres", "array_row", "row", "y"])
    pxl_col = first_existing_column(obs, ["pxl_col_in_fullres", "array_col", "col", "x"])

    submission = pd.DataFrame(
        {
            "barcode": obs_names if barcode is None else barcode.astype(str).to_numpy(),
            "pxl_row_in_fullres": np.nan if pxl_row is None else pxl_row.to_numpy(),
            "pxl_col_in_fullres": np.nan if pxl_col is None else pxl_col.to_numpy(),
        }
    )

    for i, protein in enumerate(protein_names):
        submission[str(protein)] = pred_expression[:, i]
    return submission


def export_valid_protein_predictions(
    predictions_dir: Path,
    submission_path: Path,
    valid_obs: pd.DataFrame,
    obs_names: Iterable[str],
    pred: np.ndarray,
    protein_names: Iterable[str],
    target_scale: float,
) -> None:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    spot_ids = list(map(str, obs_names))
    pred_expression = np.clip(np.sinh(pred) * target_scale, a_min=0.0, a_max=None)

    submission = build_submission_frame(
        valid_obs,
        spot_ids,
        pred_expression,
        protein_names,
    )
    submission.to_csv(submission_path, index=False)
    submission.to_csv(predictions_dir / "final_submission.csv", index=False)

    for i, protein in enumerate(protein_names):
        df = pd.DataFrame(
            {
                "Spot_ID": spot_ids,
                "Predicted": pred_expression[:, i],
            }
        )
        df.to_csv(
            predictions_dir / f"{safe_filename(protein)}_predicted.csv",
            index=False,
        )


def train_one_fold(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    config: PipelineConfig,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame, np.ndarray]:
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(
        train_ds,
        batch_size=min(config.batch_size, len(train_ds)),
        shuffle=True,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_state: dict[str, torch.Tensor] | None = None
    best_pearson = -float("inf")
    patience_count = 0
    history: list[dict[str, float]] = []
    started = time.time()

    y_valid_tensor = torch.from_numpy(y_valid).to(device)
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = combined_loss(model(xb), yb, config.loss_mse_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        valid_pred = batch_predict(model, x_valid, config.batch_size, device)
        valid_pred_tensor = torch.from_numpy(valid_pred).to(device)
        valid_loss = float(
            combined_loss(valid_pred_tensor, y_valid_tensor, config.loss_mse_weight)
            .detach()
            .cpu()
        )
        valid_metric = metric_frame(valid_pred, y_valid, range(y_valid.shape[1]))
        valid_pearson = float(valid_metric["pearson_r"].mean())

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "valid_loss": valid_loss,
                "valid_mean_pearson": valid_pearson,
                "lr": float(scheduler.get_last_lr()[0]),
            }
        )

        if valid_pearson > best_pearson:
            best_pearson = valid_pearson
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        if epoch == 1 or epoch % 10 == 0:
            elapsed = time.time() - started
            print(
                f"    epoch {epoch:4d}/{config.epochs} | "
                f"train={history[-1]['train_loss']:.4f} | "
                f"valid={valid_loss:.4f} | "
                f"r={valid_pearson:.4f} | "
                f"patience={patience_count}/{config.patience} | {elapsed:.0f}s"
            )

        if patience_count >= config.patience:
            print(f"    early stop at epoch {epoch}; best valid r={best_pearson:.4f}")
            break

    if best_state is None:
        raise RuntimeError("Training ended without a checkpoint.")
    model.load_state_dict(best_state)
    best_pred = batch_predict(model, x_valid, config.batch_size, device)
    return best_state, pd.DataFrame(history), best_pred


def run_pipeline(config: PipelineConfig) -> pd.DataFrame:
    set_seed(config.seed)
    out_dir = Path(config.out_dir)
    tables_dir = out_dir / "tables"
    models_dir = out_dir / "models"
    valid_predictions_dir = out_dir / "valid_rna_predicted_proteins"
    tables_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    valid_predictions_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(config.device)
    print(f"Device: {device}")
    print(f"Output: {out_dir.resolve()}")

    rna, protein = load_aligned_data(
        Path(config.rna),
        Path(config.protein),
        sample_n=config.sample_n,
        seed=config.seed,
    )
    rna_norm = normalize_rna(rna)
    y_raw = dense_float32(protein.X)
    y = np.arcsinh(y_raw / config.target_scale).astype(np.float32)
    protein_names = list(map(str, protein.var_names))
    valid_rna_norm = None
    if config.export_valid_predictions:
        valid_rna = ad.read_h5ad(Path(config.valid_rna))
        valid_rna_norm = normalize_rna(valid_rna)

    splits = make_spatial_splits(
        rna_norm.obs,
        config.n_splits,
        config.block_size,
        config.seed,
        config.split_strategy,
    )
    if config.max_folds is not None:
        splits = splits[: config.max_folds]

    (out_dir / "run_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    all_metrics = []
    fold_summaries = []
    valid_fold_predictions = []
    for fold_no, (train_idx, valid_idx) in enumerate(splits, start=1):
        print("=" * 72)
        print(
            f"Fold {fold_no}/{len(splits)} | "
            f"train={len(train_idx):,} valid={len(valid_idx):,}"
        )
        x_train, x_valid, hvgs, feature_info, feature_scaler, feature_pca = make_fold_features(
            rna_norm,
            y,
            train_idx,
            valid_idx,
            config.feature_selection,
            config.n_hvg,
            config.pca_dims,
            config.seed,
        )
        pd.Series(hvgs, name="gene").to_csv(
            tables_dir / f"fold{fold_no}_selected_hvgs.csv", index=False
        )
        print(
            f"Features: {int(feature_info['n_hvg_selected'])} HVGs, "
            f"{int(feature_info['pca_components'])} PCA dims, "
            f"variance={feature_info['pca_explained_variance']:.3f}"
        )

        model = MLPRegressor(
            in_dim=x_train.shape[1],
            hidden=config.hidden,
            out_dim=y.shape[1],
            dropout=config.dropout,
        ).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        y_train = y[train_idx]
        y_valid = y[valid_idx]
        target_scaler = None
        y_train_model = y_train
        y_valid_model = y_valid
        if config.standardize_targets:
            target_scaler = StandardScaler()
            y_train_model = target_scaler.fit_transform(y_train).astype(np.float32)
            y_valid_model = target_scaler.transform(y_valid).astype(np.float32)
        state, history, pred = train_one_fold(
            model, x_train, y_train_model, x_valid, y_valid_model, config, device
        )
        if target_scaler is not None:
            pred = target_scaler.inverse_transform(pred).astype(np.float32)
        torch.save(state, models_dir / f"mlp_fold{fold_no}.pt")
        history.to_csv(tables_dir / f"fold{fold_no}_training_history.csv", index=False)

        if valid_rna_norm is not None:
            x_valid_rna = transform_rna_features(
                valid_rna_norm,
                hvgs,
                feature_scaler,
                feature_pca,
            )
            valid_pred = batch_predict(model, x_valid_rna, config.batch_size, device)
            if target_scaler is not None:
                valid_pred = target_scaler.inverse_transform(valid_pred).astype(np.float32)
            valid_fold_predictions.append(valid_pred)

        metrics = metric_frame(pred, y_valid, protein_names)
        metrics.insert(0, "fold", fold_no)
        metrics.to_csv(tables_dir / f"fold{fold_no}_protein_metrics.csv", index=False)
        all_metrics.append(metrics)

        fold_summary = {
            "fold": fold_no,
            "mean_pearson_r": float(metrics["pearson_r"].mean()),
            "mean_rmse": float(metrics["rmse"].mean()),
            "mean_mae": float(metrics["mae"].mean()),
            **feature_info,
        }
        fold_summaries.append(fold_summary)
        print(
            f"Fold {fold_no} result: mean Pearson r={fold_summary['mean_pearson_r']:.4f}, "
            f"RMSE={fold_summary['mean_rmse']:.4f}, MAE={fold_summary['mean_mae']:.4f}"
        )

    metrics_all = pd.concat(all_metrics, ignore_index=True)
    metrics_all.to_csv(tables_dir / "all_fold_protein_metrics.csv", index=False)

    aggregate = (
        metrics_all.groupby("protein", as_index=False)
        .agg(
            mean_pearson_r=("pearson_r", "mean"),
            std_pearson_r=("pearson_r", "std"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
        )
        .sort_values("mean_pearson_r", ascending=False)
    )
    aggregate.to_csv(tables_dir / "mlp_cv_protein_metrics.csv", index=False)
    fold_summary_df = pd.DataFrame(fold_summaries)
    fold_summary_df.to_csv(tables_dir / "mlp_cv_fold_summary.csv", index=False)

    valid_predictions_written = 0
    if valid_rna_norm is not None and valid_fold_predictions:
        valid_pred_mean = np.mean(np.stack(valid_fold_predictions, axis=0), axis=0)
        export_valid_protein_predictions(
            valid_predictions_dir,
            out_dir / "final_submission.csv",
            valid_rna_norm.obs,
            valid_rna_norm.obs_names,
            valid_pred_mean,
            protein_names,
            config.target_scale,
        )
        valid_predictions_written = len(protein_names)

    summary = {
        "mean_cv_pearson_r": float(aggregate["mean_pearson_r"].mean()),
        "mean_cv_rmse": float(aggregate["mean_rmse"].mean()),
        "mean_cv_mae": float(aggregate["mean_mae"].mean()),
        "n_observations": int(rna.n_obs),
        "n_rna_genes": int(rna.n_vars),
        "n_proteins": int(protein.n_vars),
        "folds_run": len(splits),
        "valid_rna_predictions_written": valid_predictions_written,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Cross-validation summary")
    print(json.dumps(summary, indent=2))
    print("Top proteins")
    print(aggregate.head(10).to_string(index=False))
    return aggregate


def load_config(path: str | None) -> dict:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args(argv: list[str] | None = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Improved MLP RNA-to-protein pipeline")
    parser.add_argument("--config", help="Optional JSON config file.")
    parser.add_argument("--rna", help="Path to train_rna.h5ad.")
    parser.add_argument("--protein", "--pro", dest="protein", help="Path to train_pro.h5ad.")
    parser.add_argument("--valid_rna", help="Path to valid_rna.h5ad for final protein prediction exports.")
    parser.add_argument("--out_dir", "--out", dest="out_dir", help="Output directory.")
    parser.add_argument("--n_splits", type=int)
    parser.add_argument("--split_strategy", choices=["spatial", "kfold"])
    parser.add_argument("--block_size", type=int)
    parser.add_argument("--feature_selection", choices=["variance", "supervised_correlation"])
    parser.add_argument("--n_hvg", type=int)
    parser.add_argument("--pca_dims", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--target_scale", type=float)
    parser.add_argument("--standardize_targets", action=argparse.BooleanOptionalAction)
    parser.add_argument("--loss_mse_weight", type=float)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max_folds", type=int)
    parser.add_argument(
        "--sample_n",
        type=int,
        help="Use only N observations from data/ for a quick smoke test.",
    )
    parser.add_argument("--export_valid_predictions", action=argparse.BooleanOptionalAction)

    args = parser.parse_args(argv)
    values = asdict(PipelineConfig())
    values.update(load_config(args.config))
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            values[key] = value
    return PipelineConfig(**values)


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    run_pipeline(config)


if __name__ == "__main__":
    main()
