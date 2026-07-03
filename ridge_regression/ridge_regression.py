import anndata as ad
import numpy as np
import pandas as pd

from cross_validation_split import load_cv_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


def ridge_regression(
    rna_path,                   # "../preprocessing/outputs/rna_hvg.h5ad"
    pro_path,                   # "../preprocessing/outputs/protein_data.h5ad"
    cv_split_path,              # "../preprocessing/outputs/cv_splits.json"
    alphas,                     # [0.01, 0.1, 1, 10, 100, 1000, 5000, 10000, 50000, 100000]
    out_path                    # output folder
    ):
    """
    ridge regression on preprocessed rna and protein data
    returns
    """

    # load preprocessed data
    rna_hvg = ad.read_h5ad(rna_path)
    pro_data = ad.read_h5ad(pro_path)

    X = rna_hvg.X.toarray() if hasattr(rna_hvg.X, "toarray") else rna_hvg.X.copy()
    Y = pro_data.X.toarray() if hasattr(pro_data.X, "toarray") else pro_data.X.copy()

    # load cv split
    splits = load_cv_split(split_path=cv_split_path)  # 5 splits
    n_splits = len(splits)

    # cross-validated ridge regression
    fold_pearsonr = np.zeros((n_splits, Y.shape[1]))  # per fold pearson r values
    fold_r2 = np.zeros((n_splits, Y.shape[1]))  # per fold r2 values
    chosen_alphas = []

    Y_pred_oof = np.zeros_like(Y, dtype=float)  # out-of-fold predictions

    for split in splits:
        fold = split["fold"]
        train_idx, val_idx = split["train"], split["test"]

        X_tr, X_val = X[train_idx], X[val_idx]
        Y_tr, Y_val = Y[train_idx], Y[val_idx]

        # fit scalers on train, apply to train and val
        scaler_X = StandardScaler()
        X_tr_scaled = scaler_X.fit_transform(X_tr)
        X_val_scaled = scaler_X.transform(X_val)

        scaler_Y = StandardScaler()
        Y_tr_scaled = scaler_Y.fit_transform(Y_tr)
        Y_val_scaled = scaler_Y.transform(Y_val)

        model = RidgeCV(alphas=alphas, cv=5, scoring="r2")
        model.fit(X_tr_scaled, Y_tr_scaled)
        chosen_alphas.append(model.alpha_)

        Y_val_pred_scaled = model.predict(X_val_scaled)

        # inverse-transform predictions back to original protein expression units
        Y_val_pred = scaler_Y.inverse_transform(Y_val_pred_scaled)
        Y_pred_oof[val_idx] = Y_val_pred

        for j in range(Y.shape[1]):
            r, _ = pearsonr(Y_val[:, j], Y_val_pred[:, j])
            fold_pearsonr[fold, j] = r
            fold_r2[fold, j] = r2_score(Y_val[:, j], Y_val_pred[:, j])

        print(f"Fold {fold + 1}/{n_splits}, "
            f"\nbest alpha={model.alpha_:.2f}, "
            f"\nmean Pearson r={fold_pearsonr[fold].mean():.3f}")

    # Summary

    mean_pearsonr_per_protein = fold_pearsonr.mean(axis=0)
    std_r_per_protein = fold_pearsonr.std(axis=0)
    mean_r2_per_protein = fold_r2.mean(axis=0)

    print(f"\nOverall mean Pearson r across all proteins: {mean_pearsonr_per_protein.mean():.3f} ± {mean_pearsonr_per_protein.std():.4f}")
    print(f"Overall mean R² across all proteins: {mean_r2_per_protein.mean():.3f}")
    print(f"Chosen alphas per fold: {chosen_alphas}")

    results_df = pd.DataFrame({
        "mean_pearsonr": mean_pearsonr_per_protein.mean(),
        "mean_pearsonr_std": mean_pearsonr_per_protein.std(),
        "mean_r2": mean_r2_per_protein.mean(),
        "chosen_alphas": chosen_alphas
    })

    # Per protein analysis

    protein_names = list(pro_data.var_names)

    per_protein_results_df = pd.DataFrame({
        "protein": protein_names,
        "mean_pearsonr": mean_pearsonr_per_protein,
        "std_pearsonr": std_r_per_protein,
        "mean_r2": mean_r2_per_protein,
    }).sort_values("mean_pearsonr", ascending=False)

    print(f"\nTop 10 best-predicted proteins:")
    print(per_protein_results_df.head(10).to_string(index=False))

    # save results
    results_df.to_csv(f"{out_path}/results.csv", index=False)
    per_protein_results_df.to_csv(f"{out_path}/per_protein_metrics.csv", index=False)

    return results_df, per_protein_results_df