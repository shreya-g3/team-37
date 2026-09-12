from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parent))

from data import build_spatial_edges, invert_targets, load_metadata, make_features, read_rna
from model import GINRegressor


def predict(config):
    paths = config["paths"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    artifacts_dir = Path(paths["artifacts_dir"])
    predictions_dir = Path(paths["predictions_dir"])
    predictions_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(artifacts_dir / "metadata.json")
    gene_indices = np.asarray(metadata["gene_indices"], dtype=np.int64)
    feature_mean = np.asarray(metadata["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(metadata["feature_std"], dtype=np.float32)
    y_mean = np.asarray(metadata["y_mean"], dtype=np.float32)
    y_std = np.asarray(metadata["y_std"], dtype=np.float32)
    protein_names = metadata["protein_names"]

    device_name = train_cfg.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    test_adata = read_rna(paths["test_rna"])
    x_np, _, _ = make_features(test_adata, paths["test_rna"], gene_indices, feature_mean, feature_std)
    edge_index = build_spatial_edges(test_adata, int(config["data"]["spatial_neighbors"]))

    model = GINRegressor(
        input_dim=int(metadata["input_dim"]),
        output_dim=len(protein_names),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)
    checkpoint = torch.load(artifacts_dir / "best_gin.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        pred_norm = model(torch.from_numpy(x_np).to(device), edge_index.to(device)).cpu().numpy()
    pred_raw = invert_targets(pred_norm, y_mean, y_std)

    out = pd.DataFrame(pred_raw, columns=protein_names)
    out.insert(0, "spot_id", test_adata.obs_names.astype(str))
    output_path = predictions_dir / "gin_test_predictions.csv"
    out.to_csv(output_path, index=False)
    print(f"Wrote {output_path} with shape {out.shape}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    predict(config)


if __name__ == "__main__":
    main()
