from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parent))

from data import (
    build_spatial_edges,
    choose_gene_indices,
    make_features,
    make_split,
    normalize_targets,
    read_protein_h5ad,
    read_rna,
    read_target_stats,
    save_metadata,
)
from model import GINRegressor


def pearson_by_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt((yt * yt).sum(axis=0) * (yp * yp).sum(axis=0)) + 1e-8
    return (yt * yp).sum(axis=0) / denom


def evaluate(model, x, edge_index, y, valid_idx):
    model.eval()
    with torch.no_grad():
        pred = model(x, edge_index)[valid_idx].cpu().numpy()
    truth = y[valid_idx.cpu().numpy()]
    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    mae = float(np.mean(np.abs(pred - truth)))
    pearson = pearson_by_target(truth, pred)
    return {"rmse": rmse, "mae": mae, "mean_pearson": float(np.nanmean(pearson))}


def train(config, overrides=None):
    overrides = overrides or {}
    paths = config["paths"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    for key, value in overrides.items():
        if value is not None:
            if key in data_cfg:
                data_cfg[key] = value
            if key in train_cfg:
                train_cfg[key] = value

    torch.manual_seed(int(data_cfg["seed"]))
    np.random.seed(int(data_cfg["seed"]))
    device_name = train_cfg.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))

    artifacts_dir = Path(paths["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Loading RNA and protein matrices...")
    train_adata = read_rna(paths["train_rna"])
    raw_y, protein_names, protein_obs = read_protein_h5ad(paths["train_pro"])
    if list(train_adata.obs_names.astype(str))[:10] != protein_obs[:10]:
        raise ValueError("RNA and protein spot order does not appear to match.")

    supplied_y_mean, supplied_y_std = read_target_stats(paths["preprocessing_stats"])
    log_y = np.log1p(raw_y.astype(np.float32))
    fitted_y_mean = log_y.mean(axis=0, keepdims=True).astype(np.float32)
    fitted_y_std = (log_y.std(axis=0, keepdims=True).astype(np.float32) + 1e-6)
    if float(np.mean(np.abs(fitted_y_mean - supplied_y_mean))) > 1.0:
        print("Supplied y statistics do not match train_pro.X; fitting target normalization from train_pro.X.")
        y_mean, y_std = fitted_y_mean, fitted_y_std
    else:
        y_mean, y_std = supplied_y_mean, supplied_y_std
    gene_indices = choose_gene_indices(train_adata, paths["train_rna"], protein_names, int(data_cfg["max_genes"]))
    x_np, feature_mean, feature_std = make_features(train_adata, paths["train_rna"], gene_indices)
    y_np = normalize_targets(raw_y, y_mean, y_std)
    edge_index = build_spatial_edges(train_adata, int(data_cfg["spatial_neighbors"]))
    train_ids, valid_ids = make_split(x_np.shape[0], float(data_cfg["validation_fraction"]), int(data_cfg["seed"]))

    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    edge_index = edge_index.to(device)
    train_idx = torch.from_numpy(train_ids).to(device)
    valid_idx = torch.from_numpy(valid_ids).to(device)

    model = GINRegressor(
        input_dim=x.shape[1],
        output_dim=y.shape[1],
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    loss_fn = torch.nn.SmoothL1Loss()

    best_rmse = float("inf")
    best_metrics = None
    stale = 0
    epochs = int(train_cfg["epochs"])
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, edge_index)
        loss = loss_fn(pred[train_idx], y[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        metrics = evaluate(model, x, edge_index, y_np, valid_idx)
        print(f"epoch={epoch:03d} loss={loss.item():.5f} val_rmse={metrics['rmse']:.5f} val_pearson={metrics['mean_pearson']:.5f}")
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_metrics = metrics | {"epoch": epoch, "train_loss": float(loss.item())}
            torch.save({"model": model.state_dict(), "config": config}, artifacts_dir / "best_gin.pt")
            stale = 0
        else:
            stale += 1
            if stale >= int(train_cfg["patience"]):
                break

    metadata = {
        "gene_indices": gene_indices,
        "gene_names": np.asarray(train_adata.var_names.astype(str))[gene_indices].tolist(),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "protein_names": protein_names,
        "input_dim": int(x.shape[1]),
    }
    save_metadata(artifacts_dir / "metadata.json", metadata)
    (artifacts_dir / "metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    return best_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-genes", dest="max_genes", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train(config, {"epochs": args.epochs, "max_genes": args.max_genes})


if __name__ == "__main__":
    main()
