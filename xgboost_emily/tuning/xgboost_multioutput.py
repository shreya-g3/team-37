import time
import numpy as np
import anndata as ad
import pandas as pd

from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from cross_validation_split import load_cv_split
from sklearn.model_selection import train_test_split


def run_loop_benchmark(rna_hvg_path, protein_path, cv_split_path, xgb_params=None,
                        fold_to_test=0, test_size=0.2, random_state=42):
    """
    per-protein loop
    Returns predictions, pearsonr, r2, runtime

    uses a random train_test_split instead of CV split
    """
    rna_hvg = ad.read_h5ad(rna_hvg_path)
    pro_data = ad.read_h5ad(protein_path)

    X = rna_hvg.X.toarray() if hasattr(rna_hvg.X, "toarray") else rna_hvg.X.copy()
    Y = pro_data.X.toarray() if hasattr(pro_data.X, "toarray") else pro_data.X.copy()

    if cv_split_path is None:
        idx = np.arange(X.shape[0])
        train_idx, val_idx = train_test_split(idx, test_size=test_size, random_state=random_state)
    else:
        splits = load_cv_split(split_path=cv_split_path)
        split = splits[fold_to_test]
        train_idx, val_idx = split["train"], split["test"]

    X_tr, X_val = X[train_idx], X[val_idx]
    Y_tr, Y_val = Y[train_idx], Y[val_idx]

    protein_names = list(pro_data.var_names)
    n_proteins = Y.shape[1]

    base_params = dict(
        n_estimators=550, max_depth=6, learning_rate=0.053,
        subsample=0.86, colsample_bytree=0.68,
        tree_method="hist", device="cuda",
        random_state=42, n_jobs=-1,
    )
    if xgb_params:
        base_params.update(xgb_params)

    print(f"Fold {fold_to_test}: running per-protein loop ({n_proteins} fits)...")
    t0 = time.time()
    Y_val_pred_loop = np.zeros_like(Y_val, dtype=float)
    for j in range(n_proteins):
        model = XGBRegressor(**base_params)
        model.fit(X_tr, Y_tr[:, j])
        Y_val_pred_loop[:, j] = model.predict(X_val)
    loop_time = time.time() - t0

    loop_pearsonr = np.array([pearsonr(Y_val[:, j], Y_val_pred_loop[:, j])[0] for j in range(n_proteins)])
    loop_r2 = np.array([r2_score(Y_val[:, j], Y_val_pred_loop[:, j]) for j in range(n_proteins)])
    print(f"  loop time: {loop_time:.1f}s, mean Pearson r: {loop_pearsonr.mean():.4f}")

    loop_output = {
        "protein_names": protein_names,
        "Y_val": Y_val,
        "Y_val_pred": Y_val_pred_loop,
        "pearsonr": loop_pearsonr,
        "r2": loop_r2,
        "runtime_sec": loop_time}

    return loop_output


def run_multioutput_benchmark(rna_hvg_path, protein_path, cv_split_path, xgb_params=None,
                               fold_to_test=0, device="cpu", test_size=0.2, random_state=42):
    """
    multi_output_tree
    Returns predictions, pearsonr, r2, runtime

    uses a random train_test_split
    """
    rna_hvg = ad.read_h5ad(rna_hvg_path)
    pro_data = ad.read_h5ad(protein_path)

    X = rna_hvg.X.toarray() if hasattr(rna_hvg.X, "toarray") else rna_hvg.X.copy()
    Y = pro_data.X.toarray() if hasattr(pro_data.X, "toarray") else pro_data.X.copy()

    if cv_split_path is None:
        idx = np.arange(X.shape[0])
        train_idx, val_idx = train_test_split(idx, test_size=test_size, random_state=random_state)
    else:
        splits = load_cv_split(split_path=cv_split_path)
        split = splits[fold_to_test]
        train_idx, val_idx = split["train"], split["test"]

    X_tr, X_val = X[train_idx], X[val_idx]
    Y_tr, Y_val = Y[train_idx], Y[val_idx]

    protein_names = list(pro_data.var_names)

    mo_params = dict(
        n_estimators=550, max_depth=6, learning_rate=0.053,
        subsample=0.86, colsample_bytree=0.68,
        tree_method="hist", device=device,
        random_state=42, n_jobs=-1,
        multi_strategy="multi_output_tree",
    )
    if xgb_params:
        mo_params.update(xgb_params)

    print(f"Fold {fold_to_test}: running multi_output_tree on {device} (1 fit)...")
    t0 = time.time()
    mo_model = XGBRegressor(**mo_params)
    mo_model.fit(X_tr, Y_tr)
    Y_val_pred_mo = mo_model.predict(X_val)
    mo_time = time.time() - t0

    mo_pearsonr = np.array([pearsonr(Y_val[:, j], Y_val_pred_mo[:, j])[0] for j in range(Y.shape[1])])
    mo_r2 = np.array([r2_score(Y_val[:, j], Y_val_pred_mo[:, j]) for j in range(Y.shape[1])])
    print(f"  multi_output_tree time: {mo_time:.1f}s, mean Pearson r: {mo_pearsonr.mean():.4f}")

    mo_output = {
        "protein_names": protein_names,
        "Y_val": Y_val,
        "Y_val_pred": Y_val_pred_mo,
        "pearsonr": mo_pearsonr,
        "r2": mo_r2,
        "runtime_sec": mo_time}

    return mo_output

def compare_results(loop_result, mo_result):
    comparison_df = pd.DataFrame({
        "protein": loop_result["protein_names"],
        "pearsonr_loop": loop_result["pearsonr"],
        "pearsonr_multioutput": mo_result["pearsonr"],
        "pearsonr_diff": mo_result["pearsonr"] - loop_result["pearsonr"],
        "r2_loop": loop_result["r2"],
        "r2_multioutput": mo_result["r2"],
    }).sort_values("pearsonr_diff")

    speedup = loop_result["runtime_sec"] / mo_result["runtime_sec"]
    print(f"Speedup: {speedup:.1f}x")
    print(comparison_df)
    return comparison_df