"""
infer_v4_final.py — run the trained final GNN model on new, unlabeled RNA
data (valid_rna.h5ad / test_rna.h5ad) and produce protein predictions.

Reuses every fitted preprocessing artifact saved by `gnn_pipeline_v4_final.py
--final_model` (SVD, marker/H&E z-score stats, target mean/std, exact gene
list) so preprocessing here is bit-for-bit consistent with training - no
refitting happens on the new data.

Marker features (RNA-gene-derived) are computed fresh for the new dataset if
not already provided, since they only need the RNA file itself. H&E features
require a full-resolution tissue image aligned to this dataset's pixel
coordinates - if none is supplied, they're zero-filled (matching how the
training script treats "not found"), which is the same behaviour, just
possibly leaving useful signal on the table for that dataset.

Usage:
    python3 infer_v4_final.py \
        --rna /path/to/valid_rna.h5ad \
        --model_dir /path/to/results/final_model \
        --out /path/to/valid_predictions.csv \
        [--he_features /path/to/he_features_valid.npy] \
        [--marker_features /path/to/marker_features_valid.npy]   # auto-built if omitted
"""
import argparse
import json
import os
import pickle
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn_pipeline_v4_final import (
    MultiScaleSpatialGNN, build_knn_graph, build_radius_graph,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rna", required=True, help="path to valid_rna.h5ad or test_rna.h5ad")
    ap.add_argument("--model_dir", required=True, help="the final_model/ dir saved by --final_model training")
    ap.add_argument("--out", required=True, help="output CSV path for predictions")
    ap.add_argument("--marker_features", default=None,
                     help="precomputed marker_features .npy for this RNA file; auto-built if omitted")
    ap.add_argument("--he_features", default=None,
                     help="precomputed he_features .npy for this RNA file; zero-filled if omitted")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    with open(os.path.join(args.model_dir, "model_meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(args.model_dir, "svd.pkl"), "rb") as f:
        svd = pickle.load(f)
    stats = np.load(os.path.join(args.model_dir, "preprocessing_stats.npz"))

    print(f"Loaded model_meta.json: feat_dim={meta['feat_dim']}  "
          f"n_markers={meta['n_markers']}  n_he={meta['n_he']}  "
          f"n_genes_used={len(meta['used_genes'])}")

    print(f"Loading RNA: {args.rna}")
    rna = ad.read_h5ad(args.rna).copy()
    print(f"  RNA: {rna.shape}")
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)

    # subset/reindex to the EXACT gene list + order used at training time
    used_genes = meta["used_genes"]
    missing = [g for g in used_genes if g not in rna.var_names]
    if missing:
        print(f"  WARNING: {len(missing)} training genes missing from this RNA file "
              f"(zero-filled): {missing[:10]}{'...' if len(missing) > 10 else ''}")
    rna = rna[:, [g for g in used_genes if g in rna.var_names]].copy()
    if missing:
        # pad missing genes back in as zero columns, in the exact saved order
        pad = sp.csr_matrix((rna.n_obs, len(missing)), dtype=np.float32)
        full = sp.hstack([rna.X, pad]).tocsc()
        order = [used_genes.index(g) for g in list(rna.var_names) + missing]
        full = full[:, np.argsort(order)]
        X_raw = full.tocsr()
    else:
        X_raw = rna.X.tocsr() if sp.issparse(rna.X) else sp.csr_matrix(rna.X)

    n_bins = X_raw.shape[0]
    coords = np.column_stack([
        rna.obs["array_row"].to_numpy(dtype=np.float32),
        rna.obs["array_col"].to_numpy(dtype=np.float32),
    ])

    print(f"  Applying fitted SVD({meta['svd_dims']})...")
    X = svd.transform(X_raw).astype(np.float32)

    if meta["n_markers"] > 0:
        mk_path = args.marker_features
        if mk_path is None:
            mk_path = os.path.splitext(args.rna)[0] + "_marker_features.npy"
            if not os.path.exists(mk_path):
                print(f"  Auto-building marker features -> {mk_path}")
                gen = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "shared_preprocessing", "build_marker_features_for.py")
                subprocess.run([sys.executable, gen, "--rna", args.rna, "--out", mk_path], check=True)
        MK = np.load(mk_path).astype(np.float32)
        assert MK.shape[0] == n_bins, f"marker/bin mismatch {MK.shape} vs {n_bins}"
        assert MK.shape[1] == meta["n_markers"], f"marker dim mismatch {MK.shape[1]} vs {meta['n_markers']}"
        MK = (MK - stats["marker_mean"]) / stats["marker_std"]
        X = np.concatenate([X, MK.astype(np.float32)], axis=1)
    elif meta["n_markers"] == 0 and stats["marker_mean"].shape[1] == 0:
        pass  # model was trained without markers, nothing to add

    if meta["n_he"] > 0:
        if args.he_features and os.path.exists(args.he_features):
            HE = np.load(args.he_features).astype(np.float32)
            assert HE.shape[0] == n_bins, f"he/bin mismatch {HE.shape} vs {n_bins}"
            HE = (HE - stats["he_mean"]) / stats["he_std"]
        else:
            print("  No --he_features supplied for this dataset - zero-filling "
                  "(same fallback the training script uses for 'not found').")
            HE = np.zeros((n_bins, meta["n_he"]), dtype=np.float32)
        X = np.concatenate([X, HE.astype(np.float32)], axis=1)

    assert X.shape[1] == meta["feat_dim"], f"feature dim mismatch: built {X.shape[1]}, expected {meta['feat_dim']}"

    print(f"  Building multi-scale graphs (k={meta['k']}, radius={meta['radius']:.2f}) "
          f"over {n_bins:,} bins...")
    ei_local = build_knn_graph(coords, k=meta["k"]).to(device)
    ei_regional = build_radius_graph(coords, radius=meta["radius"], k_fallback=meta["k_fallback"]).to(device)

    model = MultiScaleSpatialGNN(
        in_channels=meta["feat_dim"], hidden=meta["hidden"],
        out_channels=len(meta["protein_names"]), n_layers=meta["n_layers"],
        dropout=meta["dropout"],
    ).to(device)
    state = torch.load(os.path.join(args.model_dir, "gnn_v4_final_model.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()

    X_t = torch.tensor(X).to(device)
    with torch.no_grad():
        pred_z = model(X_t, ei_local, ei_regional).cpu().numpy()

    # inverse the z-score standardisation to get back to CLR-scale predictions
    # (note: CLR itself is only invertible up to a per-row scale factor, so
    # these are CLR-space predictions, not raw protein counts - fine for
    # correlation-based scoring, which is scale/shift invariant)
    y_mean = stats["y_mean"]; y_std = stats["y_std"]
    pred_clr = pred_z * y_std + y_mean

    out_df = pd.DataFrame(pred_clr, columns=meta["protein_names"], index=rna.obs_names)
    out_df.to_csv(args.out)
    print(f"\nSaved predictions: {args.out}  shape={out_df.shape}")
    print(f"(values are in CLR space, matching the training target's scale - "
          f"use as-is for Pearson/Spearman scoring)")


if __name__ == "__main__":
    main()
