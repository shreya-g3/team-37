from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def safe_corr(func, actual, pred):
    if np.std(actual) == 0 or np.std(pred) == 0:
        return np.nan
    return float(func(actual, pred).statistic)


def evaluate(y_true, y_pred, protein_names, fold):
    rows = []
    for index, protein in enumerate(protein_names):
        actual = y_true[:, index]
        pred = y_pred[:, index]
        rows.append(
            {
                "fold": fold,
                "protein": protein,
                "pearson": safe_corr(pearsonr, actual, pred),
                "spearman": safe_corr(spearmanr, actual, pred),
                "r2": float(r2_score(actual, pred)),
                "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
                "mae": float(mean_absolute_error(actual, pred)),
            }
        )
    return rows


def run_lasso_cv(x_path, y_path, protein_names_path, splits, output_dir, alpha=0.01, max_iter=500):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading preprocessed sparse RNA and protein arrays...")
    X = sparse.load_npz(x_path).tocsr()
    Y = np.load(y_path).astype(np.float32)
    protein_names = pd.read_csv(protein_names_path)["protein"].tolist()

    protein_rows = []
    fold_rows = []

    for split in splits:
        fold = int(split["fold"])
        train_idx = np.array(split["train"], dtype=np.int64)
        test_idx = np.array(split["test"], dtype=np.int64)

        X_train = X[train_idx]
        X_test = X[test_idx]
        Y_train = Y[train_idx]
        Y_test = Y[test_idx]

        x_scaler = StandardScaler(with_mean=False)
        X_train_scaled = x_scaler.fit_transform(X_train)
        X_test_scaled = x_scaler.transform(X_test)

        y_scaler = StandardScaler()
        Y_train_scaled = y_scaler.fit_transform(Y_train)

        print(f"Training LASSO fold {fold + 1}/{len(splits)}...")
        model = Lasso(alpha=alpha, max_iter=max_iter, random_state=42, selection="random")
        model.fit(X_train_scaled, Y_train_scaled)
        pred_scaled = model.predict(X_test_scaled)
        pred = y_scaler.inverse_transform(pred_scaled)

        rows = evaluate(Y_test, pred, protein_names, fold)
        protein_rows.extend(rows)
        fold_df = pd.DataFrame(rows)
        fold_rows.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "alpha": alpha,
                "mean_pearson": fold_df["pearson"].mean(),
                "mean_spearman": fold_df["spearman"].mean(),
                "mean_r2": fold_df["r2"].mean(),
                "mean_rmse": fold_df["rmse"].mean(),
                "mean_mae": fold_df["mae"].mean(),
            }
        )
        print(
            f"Fold {fold + 1}: Pearson={fold_rows[-1]['mean_pearson']:.3f}, "
            f"Spearman={fold_rows[-1]['mean_spearman']:.3f}, "
            f"R2={fold_rows[-1]['mean_r2']:.3f}"
        )

    fold_metrics = pd.DataFrame(fold_rows)
    per_fold_protein = pd.DataFrame(protein_rows)
    per_protein = (
        per_fold_protein.groupby("protein", as_index=False)
        .agg(
            mean_pearson=("pearson", "mean"),
            std_pearson=("pearson", "std"),
            mean_spearman=("spearman", "mean"),
            mean_r2=("r2", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
        )
        .sort_values("mean_pearson", ascending=False)
    )

    summary = pd.DataFrame(
        [
            {
                "model": "LASSO",
                "preprocessing": "Team 37 HVG + library-size normalisation + log1p + protein CLR",
                "validation": "spatially blocked GroupKFold",
                "n_folds": len(splits),
                "alpha": alpha,
                "mean_pearson": fold_metrics["mean_pearson"].mean(),
                "std_pearson": fold_metrics["mean_pearson"].std(),
                "mean_spearman": fold_metrics["mean_spearman"].mean(),
                "mean_r2": fold_metrics["mean_r2"].mean(),
                "mean_rmse": fold_metrics["mean_rmse"].mean(),
                "mean_mae": fold_metrics["mean_mae"].mean(),
            }
        ]
    )

    summary.to_csv(output_dir / "results_summary.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    per_protein.to_csv(output_dir / "per_protein_metrics.csv", index=False)
    per_fold_protein.to_csv(output_dir / "per_fold_protein_metrics.csv", index=False)
    print(summary.to_string(index=False))
    return summary, fold_metrics, per_protein

