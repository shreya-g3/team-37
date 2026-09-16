import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial import cKDTree

from src.data import (
    read_rna,
    read_protein_h5ad,
    choose_gene_indices,
    make_features,
    normalize_targets,
    build_spatial_edges,
)
from src.model import GINRegressor


with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
data_cfg = config["data"]
model_cfg = config["model"]
train_cfg = config["training"]

seed = int(data_cfg["seed"])
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)
print("Loading data...")

rna = read_rna(paths["train_rna"])
y_raw, protein_names, protein_obs = read_protein_h5ad(
    paths["train_pro"]
)

coords = rna.obs[
    ["array_row", "array_col"]
].to_numpy(dtype=np.float32)

# ----------------------------
# 5 spatial folds
# ----------------------------

target_patch_bins = 2000
buffer_dist = 60
n_splits = 5

n_patches = max(
    n_splits,
    int(np.ceil(len(coords) / target_patch_bins))
)

nr = max(1, int(np.sqrt(n_patches)))
nc = int(np.ceil(n_patches / nr))

row_edges = np.linspace(
    coords[:, 0].min(),
    coords[:, 0].max() + 1e-6,
    nr + 1
)

col_edges = np.linspace(
    coords[:, 1].min(),
    coords[:, 1].max() + 1e-6,
    nc + 1
)

row_bin = np.digitize(
    coords[:, 0],
    row_edges[1:-1]
)

col_bin = np.digitize(
    coords[:, 1],
    col_edges[1:-1]
)

patch_id = row_bin * nc + col_bin

patches = np.unique(patch_id)

rng = np.random.default_rng(seed)
rng.shuffle(patches)

fold_patches = np.array_split(
    patches,
    n_splits
)

all_idx = np.arange(len(coords))

results = []

Path("gin_cv_results").mkdir(
    exist_ok=True
)

for fold in range(n_splits):

    holdout_mask = np.isin(
        patch_id,
        fold_patches[fold]
    )

    valid_idx = all_idx[holdout_mask]
    train_candidates = all_idx[~holdout_mask]

    tree = cKDTree(
        coords[valid_idx]
    )

    distance, _ = tree.query(
        coords[train_candidates],
        k=1
    )

    train_idx = train_candidates[
        distance > buffer_dist
    ]

    print("\n============================")
    print(f"FOLD {fold + 1}/5")
    print(
        "train =", len(train_idx),
        "validation =", len(valid_idx),
        "buffer removed =",
        len(train_candidates) - len(train_idx)
    )
    print("============================")

    train_rna = rna[train_idx]
    valid_rna = rna[valid_idx]

    genes = choose_gene_indices(
        train_rna,
        paths["train_rna"],
        protein_names,
        int(data_cfg["max_genes"])
    )

    x_train_np, mean, std = make_features(
        train_rna,
        paths["train_rna"],
        genes
    )

    x_valid_np, _, _ = make_features(
        valid_rna,
        paths["train_rna"],
        genes,
        mean,
        std
    )

    y_train_raw = y_raw[train_idx]
    y_valid_raw = y_raw[valid_idx]

    log_train = np.log1p(
        y_train_raw.astype(np.float32)
    )

    y_mean = log_train.mean(
        axis=0,
        keepdims=True
    ).astype(np.float32)

    y_std = (
        log_train.std(
            axis=0,
            keepdims=True
        ).astype(np.float32)
        + 1e-6
    )

    y_train_np = normalize_targets(
        y_train_raw,
        y_mean,
        y_std
    )

    y_valid_np = normalize_targets(
        y_valid_raw,
        y_mean,
        y_std
    )

    train_edges = build_spatial_edges(
        train_rna,
        int(data_cfg["spatial_neighbors"])
    ).to(device)

    valid_edges = build_spatial_edges(
        valid_rna,
        int(data_cfg["spatial_neighbors"])
    ).to(device)

    x_train = torch.from_numpy(
        x_train_np
    ).to(device)

    y_train = torch.from_numpy(
        y_train_np
    ).to(device)

    x_valid = torch.from_numpy(
        x_valid_np
    ).to(device)

    model = GINRegressor(
        input_dim=x_train.shape[1],
        output_dim=y_train.shape[1],
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"])
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(
            train_cfg["weight_decay"]
        )
    )

    loss_fn = torch.nn.SmoothL1Loss()

    best_rmse = float("inf")
    best_state = None

    for epoch in range(
        1,
        int(train_cfg["epochs"]) + 1
    ):

        model.train()
        optimizer.zero_grad()

        pred = model(
            x_train,
            train_edges
        )

        loss = loss_fn(
            pred,
            y_train
        )

        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            val_pred = model(
                x_valid,
                valid_edges
            ).cpu().numpy()

        rmse = float(
            np.sqrt(
                np.mean(
                    (val_pred - y_valid_np) ** 2
                )
            )
        )

        yt = (
            y_valid_np
            - y_valid_np.mean(
                axis=0,
                keepdims=True
            )
        )

        yp = (
            val_pred
            - val_pred.mean(
                axis=0,
                keepdims=True
            )
        )

        r = (
            (yt * yp).sum(axis=0)
            /
            (
                np.sqrt(
                    (yt ** 2).sum(axis=0)
                    * (yp ** 2).sum(axis=0)
                )
                + 1e-8
            )
        )

        pearson = float(
            np.nanmean(r)
        )

        print(
            f"epoch {epoch}: "
            f"RMSE={rmse:.4f}, "
            f"Pearson={pearson:.4f}"
        )

        if rmse < best_rmse:
            best_rmse = rmse
            best_state = copy.deepcopy(
                model.state_dict()
            )

    model.load_state_dict(
        best_state
    )

    model.eval()

    with torch.no_grad():
        val_pred = model(
            x_valid,
            valid_edges
        ).cpu().numpy()

    rmse = float(
        np.sqrt(
            np.mean(
                (val_pred - y_valid_np) ** 2
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(
                val_pred - y_valid_np
            )
        )
    )

    yt = y_valid_np - y_valid_np.mean(
        axis=0,
        keepdims=True
    )

    yp = val_pred - val_pred.mean(
        axis=0,
        keepdims=True
    )

    r = (
        (yt * yp).sum(axis=0)
        /
        (
            np.sqrt(
                (yt ** 2).sum(axis=0)
                * (yp ** 2).sum(axis=0)
            )
            + 1e-8
        )
    )

    pearson = float(
        np.nanmean(r)
    )

    results.append({
        "fold": fold + 1,
        "pearson": pearson,
        "rmse": rmse,
        "mae": mae
    })

    print(
        f"Fold {fold + 1}: "
        f"Pearson={pearson:.4f}, "
        f"RMSE={rmse:.4f}, "
        f"MAE={mae:.4f}"
    )


df = pd.DataFrame(results)

df.to_csv(
    "gin_cv_results/fold_results.csv",
    index=False
)

summary = {
    "mean_cv_pearson":
        float(df["pearson"].mean()),

    "std_cv_pearson":
        float(df["pearson"].std()),

    "mean_cv_rmse":
        float(df["rmse"].mean()),

    "mean_cv_mae":
        float(df["mae"].mean()),

    "folds": 5,

    "buffer_dist": 60,

    "target_patch_bins": 2000
}

with open(
    "gin_cv_results/summary.json",
    "w"
) as f:
    json.dump(
        summary,
        f,
        indent=2
    )

print("\nFINAL CV RESULTS")
print(json.dumps(summary, indent=2))