"""
5-fold spatial CV for DGAT (mini-batch, neighbor-sampled training)

Per fold:
    - RNA: TruncatedSVD fit on train bins only, transform all
    - Protein: PCA fit on train bins only, transform all
    - Spatial graph built once (coords fixed across folds); feature graphs
      rebuilt per fold since SVD/PCA are fold-specific; each unioned with
      the shared spatial graph
    - Training: dgat.train_dgat_with_early_stopping (mini-batch,
      NeighborLoader-based)
    - Early-stopping criterion: holdout cross_protein RMSE
    - Fold predictions: dgat.predict_batched over all nodes (mRNA-only
      path), sliced to val_idx
"""

import os
import gc
import pickle
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from sklearn.decomposition import TruncatedSVD, PCA
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error

import torch

from dgat import (
    build_spatial_graph, build_feature_graph, union_edge_index, build_pyg_data,
    predict_batched, train_dgat_with_early_stopping,
    DEFAULT_NUM_NEIGHBORS, DEFAULT_BATCH_SIZE,
)
from cv_split_patches import load_cv_split

# current config
DEFAULT_DGAT_PARAMS = dict(
    hidden=128,
    latent_dim=256,
    n_layers=3,
    dropout=0.3,
    gat_heads=2,
    feat_attn_heads=4,
    decoder_hidden=128,
    branch_hidden=32,
    loss_weights=dict(align=5.0, recon_mrna=1.0, recon_protein=1.0,
                      cross_protein=3.0, cross_mrna=5.0),
    lr_encoder=5e-4,
    lr_decoder=1e-4,
    encoder_wd=2e-5,
    grad_clip=1.0,
)

N_SVD_COMPONENTS = 128
PROTEIN_PCA_VARIANCE = 0.85
K_SPATIAL = 6
K_FEATURE = 10
MAX_EPOCHS = 300
PATIENCE = 20
SEED = 0


def save_model(model, model_path):
    torch.save({"state_dict": model.state_dict(), "config": model._config}, model_path)


def run_dgat_cv(
        rna_path,
        protein_path,
        cv_split_path,
        out_dir,
        device=None,
        n_svd_components=N_SVD_COMPONENTS,
        protein_pca_variance=PROTEIN_PCA_VARIANCE,
        svd_random_state=SEED,
        params=None,
        k_spatial=K_SPATIAL,
        k_feature=K_FEATURE,
        num_neighbors=None,
        batch_size=DEFAULT_BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        max_folds=None,
):
    """
    Cross-validated DGAT, mini-batch training. Per-fold RNA SVD + protein
    PCA (train-only fits).
    """
    os.makedirs(out_dir, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_neighbors = num_neighbors or DEFAULT_NUM_NEIGHBORS

    run_params = dict(DEFAULT_DGAT_PARAMS)
    if params:
        run_params.update(params)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading preprocessed data...")
    rna = ad.read_h5ad(rna_path)
    pro = ad.read_h5ad(protein_path)

    protein_names = list(pro.var_names)
    coords = rna.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values.astype(np.float32)

    X_raw = rna.X if sparse.issparse(rna.X) else sparse.csr_matrix(rna.X)
    X_raw = X_raw.astype(np.float32).copy()
    Y = pro.X.toarray() if sparse.issparse(pro.X) else np.asarray(pro.X).astype(np.float32)

    del rna, pro
    gc.collect()

    # spatial graph built once, edges depend only on coordinates, shared across folds
    spatial_edge = build_spatial_graph(coords, k=k_spatial)

    splits = load_cv_split(split_path=cv_split_path)
    n_splits = len(splits)
    if max_folds is not None:
        print(f"max_folds={max_folds}: running only {min(max_folds, n_splits)}/{n_splits} folds (partial run)")
        splits = splits[:max_folds]
        n_splits = len(splits)

    fold_pearsonr = np.zeros((n_splits, Y.shape[1]))
    fold_r2 = np.zeros((n_splits, Y.shape[1]))
    fold_rmse = np.zeros((n_splits, Y.shape[1]))
    fold_svd_explained_var = np.zeros(n_splits)
    fold_epochs = []

    model_cfg_base = dict(
        hidden=run_params["hidden"], latent_dim=run_params["latent_dim"],
        n_layers=run_params["n_layers"], dropout=run_params["dropout"],
        gat_heads=run_params["gat_heads"], feat_attn_heads=run_params["feat_attn_heads"],
        decoder_hidden=run_params["decoder_hidden"], branch_hidden=run_params["branch_hidden"],
    )

    for split in splits:
        fold = split["fold"]
        train_idx, val_idx = np.array(split["train"]), np.array(split["test"])

        print(f"\nFold {fold + 1}/{n_splits} "
              f"({len(train_idx):,} train / {len(val_idx):,} val bins)")

        svd = TruncatedSVD(n_components=n_svd_components, random_state=svd_random_state)
        svd.fit(X_raw[train_idx])
        X_svd = svd.transform(X_raw).astype(np.float32)
        fold_svd_explained_var[fold] = svd.explained_variance_ratio_.sum()
        print(f"  RNA SVD cumulative explained variance (train): {fold_svd_explained_var[fold]:.3f}")

        with open(f"{out_dir}/svd_model_dgat_fold{fold}.pkl", "wb") as f:
            pickle.dump(svd, f)

        pca = PCA(n_components=protein_pca_variance, random_state=svd_random_state)
        pca.fit(Y[train_idx])
        Y_pca = pca.transform(Y).astype(np.float32)
        print(f"  Protein PCA: {pca.n_components_} components retain "
             f"{protein_pca_variance:.0%} variance (train)")

        with open(f"{out_dir}/protein_pca_dgat_fold{fold}.pkl", "wb") as f:
            pickle.dump(pca, f)

        mrna_feat_edge = build_feature_graph(X_svd, k=k_feature)
        protein_feat_edge = build_feature_graph(Y_pca, k=k_feature)
        edge_mrna = union_edge_index(mrna_feat_edge, spatial_edge)
        edge_protein = union_edge_index(protein_feat_edge, spatial_edge)
        print(f"  edge_mrna: {edge_mrna.shape[1]:,} edges  "
             f"edge_protein: {edge_protein.shape[1]:,} edges")

        model_cfg = dict(model_cfg_base, rna_dim=n_svd_components, protein_dim=Y.shape[1])

        model, n_epochs, history = train_dgat_with_early_stopping(
            X_svd, Y, edge_mrna, edge_protein, train_idx, val_idx,
            model_cfg, run_params["loss_weights"], device,
            num_neighbors=num_neighbors, batch_size=batch_size,
            lr_encoder=run_params["lr_encoder"], lr_decoder=run_params["lr_decoder"],
            encoder_wd=run_params["encoder_wd"], max_epochs=max_epochs, patience=patience,
            grad_clip=run_params.get("grad_clip", 1.0),
        )
        fold_epochs.append(n_epochs)

        save_model(model, f"{out_dir}/dgat_model_fold{fold}.pt")
        history.to_csv(f"{out_dir}/history_fold{fold}.csv", index=False)

        data_mrna = build_pyg_data(X_svd, edge_mrna)
        all_idx = torch.arange(X_svd.shape[0], dtype=torch.long)
        preds, node_ids = predict_batched(model, data_mrna, all_idx, device,
                                          num_neighbors=num_neighbors, batch_size=batch_size)
        order = np.argsort(node_ids)
        Protein_pred_all = preds[order]

        Y_val_pred = Protein_pred_all[val_idx]
        Y_val = Y[val_idx]

        for j in range(Y.shape[1]):
            r, _ = pearsonr(Y_val[:, j], Y_val_pred[:, j])
            fold_pearsonr[fold, j] = r
            fold_r2[fold, j] = r2_score(Y_val[:, j], Y_val_pred[:, j])
            fold_rmse[fold, j] = np.sqrt(mean_squared_error(Y_val[:, j], Y_val_pred[:, j]))

        print(f"Fold {fold + 1}/{n_splits}, epochs={n_epochs}, "
              f"mean Pearson r={fold_pearsonr[fold].mean():.3f}, "
              f"mean R2={fold_r2[fold].mean():.3f}, "
              f"mean RMSE={fold_rmse[fold].mean():.3f}")

        fold_results = pd.DataFrame({
            "protein": protein_names,
            "pearsonr": fold_pearsonr[fold],
            "r2": fold_r2[fold],
            "rmse": fold_rmse[fold],
        })
        fold_results.to_csv(f"{out_dir}/fold{fold}_dgat.csv", index=False)

        del svd, pca, X_svd, Y_pca, model, Protein_pred_all, Y_val_pred, Y_val
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_pearsonr_per_protein = fold_pearsonr.mean(axis=0)
    std_r_per_protein = fold_pearsonr.std(axis=0)
    mean_r2_per_protein = fold_r2.mean(axis=0)
    mean_rmse_per_protein = fold_rmse.mean(axis=0)

    print(
        f"\nOverall mean Pearson r across all proteins: "
        f"{mean_pearsonr_per_protein.mean():.3f} \u00b1 {mean_pearsonr_per_protein.std():.4f}")
    print(f"Overall mean R\u00b2 across all proteins: {mean_r2_per_protein.mean():.3f}")
    print(f"Overall mean RMSE across all proteins: {mean_rmse_per_protein.mean():.3f}")
    print(f"Mean RNA SVD cumulative explained variance across folds: {fold_svd_explained_var.mean():.3f}")
    print(f"Epochs selected per fold: {fold_epochs} (median={int(np.median(fold_epochs))})")

    results_df = pd.DataFrame({
        "mean_pearsonr": [mean_pearsonr_per_protein.mean()],
        "mean_pearsonr_std": [mean_pearsonr_per_protein.std()],
        "mean_r2": [mean_r2_per_protein.mean()],
        "mean_rmse": [mean_rmse_per_protein.mean()],
        "mean_svd_explained_var": [fold_svd_explained_var.mean()],
        "n_svd_components": [n_svd_components],
        "protein_pca_variance": [protein_pca_variance],
        "median_epochs": [int(np.median(fold_epochs))],
    })

    per_protein_results_df = pd.DataFrame({
        "protein": protein_names,
        "mean_pearsonr": mean_pearsonr_per_protein,
        "std_pearsonr": std_r_per_protein,
        "mean_r2": mean_r2_per_protein,
        "mean_rmse": mean_rmse_per_protein,
    }).sort_values("mean_pearsonr", ascending=False)

    print(f"\nTop 10 best-predicted proteins:")
    print(per_protein_results_df.head(10).to_string(index=False))

    results_df.to_csv(f"{out_dir}/dgat_cv_results.csv", index=False)
    per_protein_results_df.to_csv(f"{out_dir}/dgat_cv_per_protein_metrics.csv", index=False)

    print(f"\nSaved to {out_dir}")

    return results_df, per_protein_results_df, run_params, fold_epochs


# entry point

def main():
    import argparse

    parser = argparse.ArgumentParser()
    # data / paths
    parser.add_argument("--rna_path", default="data/rna_hvg.h5ad")
    parser.add_argument("--protein_path", default="data/protein_data_v2.h5ad")
    parser.add_argument("--cv_split_path", default="data/cv_splits_patches.json")
    parser.add_argument("--out_dir", default="results")

    # feature spaces
    parser.add_argument("--n_svd_components", type=int, default=N_SVD_COMPONENTS)
    parser.add_argument("--protein_pca_variance", type=float, default=PROTEIN_PCA_VARIANCE)
    parser.add_argument("--svd_random_state", type=int, default=SEED)

    # graphs
    parser.add_argument("--k_spatial", type=int, default=K_SPATIAL)
    parser.add_argument("--k_feature", type=int, default=K_FEATURE)

    # model architecture -- defaults scaled down from the paper's 256/1024
    # for T4 memory constraints, see dgat.py's module docstring
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gat_heads", type=int, default=2)
    parser.add_argument("--feat_attn_heads", type=int, default=4)
    parser.add_argument("--decoder_hidden", type=int, default=128)
    parser.add_argument("--branch_hidden", type=int, default=32)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lr_encoder", type=float, default=5e-4)
    parser.add_argument("--lr_decoder", type=float, default=1e-4)
    parser.add_argument("--encoder_wd", type=float, default=2e-5)

    # loss weights -- paper defaults: align=5, recon_mrna=1, recon_protein=1,
    # cross_protein=3, cross_mrna=5
    parser.add_argument("--w_align", type=float, default=5.0)
    parser.add_argument("--w_recon_mrna", type=float, default=1.0)
    parser.add_argument("--w_recon_protein", type=float, default=1.0)
    parser.add_argument("--w_cross_protein", type=float, default=3.0)
    parser.add_argument("--w_cross_mrna", type=float, default=5.0)

    # mini-batch training
    parser.add_argument("--num_neighbors", type=int, nargs="+", default=None,
                        help="per-layer sampling fanout, e.g. num_neighbors 15 10 10 "
                            "(defaults to dgat.DEFAULT_NUM_NEIGHBORS if left out)")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)

    # training / CV
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--max_folds", type=int, default=None,
                        help="run only the first N folds (partial run, e.g. for a timing/memory check)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device == "cuda" and device.type == "cpu":
        print("cuda requested but not available, falling back to cpu")

    loss_weights = dict(
        align=args.w_align,
        recon_mrna=args.w_recon_mrna,
        recon_protein=args.w_recon_protein,
        cross_protein=args.w_cross_protein,
        cross_mrna=args.w_cross_mrna,
    )

    # merged into DEFAULT_DGAT_PARAMS by run_dgat_cv (params overrides defaults)
    params = dict(
        hidden=args.hidden,
        latent_dim=args.latent_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        gat_heads=args.gat_heads,
        feat_attn_heads=args.feat_attn_heads,
        decoder_hidden=args.decoder_hidden,
        branch_hidden=args.branch_hidden,
        loss_weights=loss_weights,
        lr_encoder=args.lr_encoder,
        lr_decoder=args.lr_decoder,
        encoder_wd=args.encoder_wd,
        grad_clip=args.grad_clip,
    )

    results_df, per_protein_results_df, run_params, fold_epochs = run_dgat_cv(
        rna_path=args.rna_path,
        protein_path=args.protein_path,
        cv_split_path=args.cv_split_path,
        out_dir=args.out_dir,
        device=device,
        n_svd_components=args.n_svd_components,
        protein_pca_variance=args.protein_pca_variance,
        svd_random_state=args.svd_random_state,
        params=params,
        k_spatial=args.k_spatial,
        k_feature=args.k_feature,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        max_folds=args.max_folds,
    )

    print(f"\nDone. fold_epochs={fold_epochs}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()