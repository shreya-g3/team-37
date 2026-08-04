import gc
import os
import pickle

import numpy as np
import pandas as pd
import anndata as ad
import xgboost as xgb
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import TruncatedSVD

from cv_split_patches import load_cv_split

# current tuned config
DEFAULT_XGB_PARAMS = dict(
    n_estimators=550,
    max_depth=6,
    learning_rate=0.019,
    subsample=0.76,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)


def xgb_svd_cv(
        rna_path,               # preprocessed RNA (full dataset, all bins) "data/preprocessed/rna_preprocessed.h5ad"
        protein_path,           # preprocessed protein data, z-scored (full dataset) "data/preprocessed/pro_preprocessed.h5ad"
        cv_split_path,          # cv splits json from cv_split_patches() "outputs/cv_splits_patches.json"
        hop,                    # neighbourhood radius in bin units (3, 30, 60), or None, for file naming
        out_dir,                # "results"
        device="cpu",
        n_components=50,        # number of truncated SVD components
        svd_random_state=0,
        xgb_params=None,        # dict to override DEFAULT_XGB_PARAMS
        A=None,                 # adjacency/weight matrix, subset per fold via A[np.ix_(idx, idx)], or None for no spatial features
        A_path=None,            # .npz adjacency matrix
):
    """
    cross-validated xgboost with truncated svd

    preprocessed input: normalize, log1p RNA, arcsinh, clip, z-score protein

    truncated SVD is refit per fold on the fold's train bins only
    adjacency matrix is subset to train/train and val/val blocks per fold

    saves per-fold per-protein metrics (pearsonr, r2, rmse) and overall summary CSV

    returns (results_df, per_protein_results_df, xgb_params)
    """
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_hop{hop}" if hop is not None else "_nohop"

    default_params = dict(DEFAULT_XGB_PARAMS)
    if xgb_params:
        default_params.update(xgb_params)

    # 1. load preprocessed data
    print("Loading preprocessed data...")
    rna = ad.read_h5ad(rna_path)
    pro = ad.read_h5ad(protein_path)

    protein_names = list(pro.var_names)

    X = rna.X if sparse.issparse(rna.X) else sparse.csr_matrix(rna.X)
    X = X.astype(np.float32).copy()
    Y = pro.X.toarray() if sparse.issparse(pro.X) else np.asarray(pro.X)

    del rna, pro
    gc.collect()

    # 2. adjacency matrix, subset per fold
    if A is None and A_path is not None:
        A = sparse.load_npz(A_path)

    use_spatial = A is not None

    # 3. load cv split
    splits = load_cv_split(split_path=cv_split_path)
    n_splits = len(splits)

    fold_pearsonr = np.zeros((n_splits, Y.shape[1]))
    fold_r2 = np.zeros((n_splits, Y.shape[1]))
    fold_rmse = np.zeros((n_splits, Y.shape[1]))
    fold_explained_var = np.zeros(n_splits)

    for split in splits:
        fold = split["fold"]
        train_idx, val_idx = split["train"], split["test"]

        print(f"\n Fold {fold + 1}/{n_splits} "
              f"({len(train_idx):,} train / {len(val_idx):,} val bins)")

        X_tr_raw, X_val_raw = X[train_idx], X[val_idx]
        Y_tr, Y_val = Y[train_idx, :], Y[val_idx, :]

        # 4. truncated SVD, fit on fold's train bins only, transform val
        svd = TruncatedSVD(n_components=n_components, random_state=svd_random_state)
        X_tr_svd = svd.fit_transform(X_tr_raw)
        X_val_svd = svd.transform(X_val_raw)
        fold_explained_var[fold] = svd.explained_variance_ratio_.sum()
        print(f"SVD cumulative explained variance (train): {fold_explained_var[fold]:.3f}")

        with open(f"{out_dir}/svd_model{suffix}_fold{fold}.pkl", "wb") as f:
            pickle.dump(svd, f)

        if use_spatial:
            # subset full adjacency to within-fold train/train and val/val blocks
            A_tr = A[np.ix_(train_idx, train_idx)]
            A_val = A[np.ix_(val_idx, val_idx)]

            X_tr_neighbour = A_tr @ X_tr_svd
            X_val_neighbour = A_val @ X_val_svd

            X_tr_final = np.hstack([X_tr_svd, X_tr_neighbour])
            X_val_final = np.hstack([X_val_svd, X_val_neighbour])

            del A_tr, A_val, X_tr_neighbour, X_val_neighbour
        else:
            X_tr_final = X_tr_svd
            X_val_final = X_val_svd

        del X_tr_raw, X_val_raw, X_tr_svd, X_val_svd
        gc.collect()

        # 5. train one XGBoost model per protein on this fold's train bins, predict val
        models_dir = f"{out_dir}/xgb_models{suffix}_fold{fold}"
        os.makedirs(models_dir, exist_ok=True)

        Y_val_pred = np.zeros_like(Y_val, dtype=np.float32)

        for j, protein in enumerate(protein_names):
            model_path = f"{models_dir}/{protein}.pkl"

            if os.path.exists(model_path):
                print(f"  [{j + 1}/{len(protein_names)}] {protein} already trained, loading")
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            else:
                model = xgb.XGBRegressor(
                    **default_params, device=device, early_stopping_rounds=30
                )
                model.fit(
                    X_tr_final,
                    Y_tr[:, j],
                    eval_set=[(X_val_final, Y_val[:, j])],
                    verbose=False,
                )
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                print(f"  [{j + 1}/{len(protein_names)}] {protein} done")

            Y_val_pred[:, j] = model.predict(X_val_final)
            del model
            gc.collect()

        # 6. per-fold metrics
        for j in range(Y.shape[1]):
            r, _ = pearsonr(Y_val[:, j], Y_val_pred[:, j])
            fold_pearsonr[fold, j] = r
            fold_r2[fold, j] = r2_score(Y_val[:, j], Y_val_pred[:, j])
            fold_rmse[fold, j] = np.sqrt(mean_squared_error(Y_val[:, j], Y_val_pred[:, j]))

        print(f"Fold {fold + 1}/{n_splits}, "
              f"mean Pearson r={fold_pearsonr[fold].mean():.3f}, "
              f"mean R2={fold_r2[fold].mean():.3f}, "
              f"mean RMSE={fold_rmse[fold].mean():.3f}")

        fold_results = pd.DataFrame({
            "protein": protein_names,
            "pearsonr": fold_pearsonr[fold],
            "r2": fold_r2[fold],
            "rmse": fold_rmse[fold],
        })
        fold_results.to_csv(f"{out_dir}/fold{fold}{suffix}.csv", index=False)

        del X_tr_final, X_val_final, Y_tr, Y_val, Y_val_pred
        gc.collect()

    # 7. summary across folds
    mean_pearsonr_per_protein = fold_pearsonr.mean(axis=0)
    std_r_per_protein = fold_pearsonr.std(axis=0)
    mean_r2_per_protein = fold_r2.mean(axis=0)
    mean_rmse_per_protein = fold_rmse.mean(axis=0)

    print(
        f"\nOverall mean Pearson r across all proteins: "
        f"{mean_pearsonr_per_protein.mean():.3f} ± {mean_pearsonr_per_protein.std():.4f}")
    print(f"Overall mean R² across all proteins: {mean_r2_per_protein.mean():.3f}")
    print(f"Overall mean RMSE across all proteins: {mean_rmse_per_protein.mean():.3f}")
    print(f"Mean SVD cumulative explained variance across folds: {fold_explained_var.mean():.3f}")

    results_df = pd.DataFrame({
        "mean_pearsonr": [mean_pearsonr_per_protein.mean()],
        "mean_pearsonr_std": [mean_pearsonr_per_protein.std()],
        "mean_r2": [mean_r2_per_protein.mean()],
        "mean_rmse": [mean_rmse_per_protein.mean()],
        "mean_svd_explained_var": [fold_explained_var.mean()],
        "n_svd_components": [n_components],
        "hop": [hop],
    })

    xgb_params = default_params

    per_protein_results_df = pd.DataFrame({
        "protein": protein_names,
        "mean_pearsonr": mean_pearsonr_per_protein,
        "std_pearsonr": std_r_per_protein,
        "mean_r2": mean_r2_per_protein,
        "mean_rmse": mean_rmse_per_protein,
    }).sort_values("mean_pearsonr", ascending=False)

    print(f"\nTop 10 best-predicted proteins:")
    print(per_protein_results_df.head(10).to_string(index=False))

    results_df.to_csv(f"{out_dir}/xgb_svd_cv_results{suffix}.csv", index=False)
    per_protein_results_df.to_csv(f"{out_dir}/xgb_svd_cv_per_protein_metrics{suffix}.csv", index=False)

    print(f"\nSaved to {out_dir}:")

    return results_df, per_protein_results_df, xgb_params