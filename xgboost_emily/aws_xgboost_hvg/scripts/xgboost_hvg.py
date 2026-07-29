import numpy as np
import anndata as ad
import pandas as pd
import xgboost as xgb
import time

from scipy import sparse
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error

from cv_split_patches import load_cv_split

def xgboost_hvg(
    rna_path,               # preprocessed RNA - with spatial features "../preprocessing/outputs/rna_hvg_aug_3.h5ad"
    protein_path,           # z-normalised protein data "../preprocessing/outputs/protein_data_V2.h5ad"
    cv_split_path,          # shared cv split "../preprocessing/outputs/cv_splits_patches.json"
    hop,                    # for file output naming (None, 3, 30 or 60)
    out_path,               # "results"
    device,                 # device setting for "cpu" or "cuda"
    xgb_params=None,        # dict to override default XGBRegressor params
    ):
    """
    predict z-normalised protein expression from rna - with or without spatial features
    returns per-protein and overall mean pearson r across folds
    """

    # load preprocessed data
    rna_hvg = ad.read_h5ad(rna_path)
    pro_data = ad.read_h5ad(protein_path)

    X = rna_hvg.X
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    Y = pro_data.X.toarray() if hasattr(pro_data.X, "toarray") else pro_data.X.copy()

    # load cv split
    splits = load_cv_split(split_path=cv_split_path)  # 5 splits
    n_splits = len(splits)

    # XGBoost params - defaults for ~2000-feature tabular regression
    default_params = dict(
        n_estimators=550,            # default:300, tuning:550
        max_depth=6,                 # default:4, tuning:6
        learning_rate=0.053,         # default:0.05, tuning:0.053
        subsample=0.86,              # default:0.8, tuning:0.86
        colsample_bytree=0.68,       # default:0.8, tuning: 0.68
        tree_method="hist",
        device=device,
        random_state=42,
        n_jobs=-1)

    if xgb_params:
        default_params.update(xgb_params)

    # cross-validated xgboost regression
    fold_pearsonr = np.zeros((n_splits, Y.shape[1]))  # per fold pearson r values
    fold_r2 = np.zeros((n_splits, Y.shape[1]))  # per fold r2 values
    fold_rmse = np.zeros((n_splits, Y.shape[1]))  # per fold rmse values

    protein_names = list(pro_data.var_names)

    for split in splits:
        fold = split["fold"]
        train_idx, val_idx = split["train"], split["test"]

        X_tr, X_val = X[train_idx], X[val_idx]
        Y_tr, Y_val = Y[train_idx, :], Y[val_idx, :]

        Y_val_pred = np.zeros_like(Y_val, dtype=float)

        for j in range(Y.shape[1]):
            start = time.time()
            model = xgb.XGBRegressor(**default_params,
                                     early_stopping_rounds=30) # to stop adding trees if no improvement
            model.fit(
                X_tr,
                Y_tr[:, j],
                eval_set=[(X_val, Y_val[:, j])],
                verbose=False
            )
            Y_val_pred[:, j] = model.predict(X_val)

            print(
                f"Protein {j + 1}/{Y.shape[1]} "
                f"took {time.time() - start:.1f} seconds"
            )

        for j in range(Y.shape[1]):
            r, _ = pearsonr(Y_val[:, j], Y_val_pred[:, j])
            fold_pearsonr[fold, j] = r
            fold_r2[fold, j] = r2_score(Y_val[:, j], Y_val_pred[:, j])
            fold_rmse[fold, j] = np.sqrt(mean_squared_error(Y_val[:, j], Y_val_pred[:, j]))

        print(f"Fold {fold + 1}/{n_splits}, "
              f"\nmean Pearson r={fold_pearsonr[fold].mean():.3f}, "
              f"\nmean R2={fold_r2[fold].mean():.3f}, "
              f"\nmean RMSE={fold_rmse[fold].mean():.3f}")

        fold_results = pd.DataFrame({
        "protein": protein_names,
        "pearsonr": fold_pearsonr[fold],
        "r2": fold_r2[fold],
        "rmse": fold_rmse[fold]})

        suffix = f"_hop{hop}" if hop is not None else "_nohop"
        fold_results.to_csv(f"{out_path}/fold{fold}{suffix}.csv", index=False)

    # Summary

    mean_pearsonr_per_protein = fold_pearsonr.mean(axis=0)
    std_r_per_protein = fold_pearsonr.std(axis=0)
    mean_r2_per_protein = fold_r2.mean(axis=0)
    mean_rmse_per_protein = fold_rmse.mean(axis=0)

    print(
        f"\nOverall mean Pearson r across all proteins: {mean_pearsonr_per_protein.mean():.3f} ± {mean_pearsonr_per_protein.std():.4f}")
    print(f"Overall mean R² across all proteins: {mean_r2_per_protein.mean():.3f}")
    print(f"Overall mean RMSE across all proteins: {mean_rmse_per_protein.mean():.3f}")

    results_df = pd.DataFrame({
        "mean_pearsonr": [mean_pearsonr_per_protein.mean()],
        "mean_pearsonr_std": [mean_pearsonr_per_protein.std()],
        "mean_r2": [mean_r2_per_protein.mean()],
        "mean_rmse": [mean_rmse_per_protein.mean()],
        "hop": [hop],
    })

    # save parameters
    xgb_params=default_params

    # Per protein analysis

    per_protein_results_df = pd.DataFrame({
        "protein": protein_names,
        "mean_pearsonr": mean_pearsonr_per_protein,
        "std_pearsonr": std_r_per_protein,
        "mean_r2": mean_r2_per_protein,
        "mean_rmse": mean_rmse_per_protein,
    }).sort_values("mean_pearsonr", ascending=False)

    print(f"\nTop 10 best-predicted proteins:")
    print(per_protein_results_df.head(10).to_string(index=False))

    # save results
    suffix = f"_hop{hop}" if hop is not None else "_nohop"
    results_df.to_csv(f"{out_path}/xgb_hvg_results{suffix}.csv", index=False)
    per_protein_results_df.to_csv(f"{out_path}/xgb_hvg_per_protein_metrics{suffix}.csv", index=False)

    return results_df, per_protein_results_df, xgb_params

