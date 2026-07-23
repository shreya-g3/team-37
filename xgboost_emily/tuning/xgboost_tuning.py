import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from cross_validation_split import load_cv_split

def xgboost_tuning(
        rna_path,  # "../preprocessing/outputs/rna_hvg.h5ad"
        pro_path,  # "../preprocessing/outputs/protein_data.h5ad"
        cv_split_path,  # "../preprocessing/outputs/cv_splits.json"
        n_proteins_to_sample=8,  # how many proteins to tune on
        n_iter=20,  # number of random param combos to try per protein
        random_state=42
):
    """
    global hyperparameter search for xgboost_regression.py
    runs a small RandomizedSearchCV on a few representative proteins, using only training data from fold 1
    best parameters across sampled proteins are averaged - parameters for final model
    proteins are selected to include a variation of expression so avoid overfitting to easy/hard proteins
    """

    # load preprocessed data
    rna_hvg = ad.read_h5ad(rna_path)
    pro_data = ad.read_h5ad(pro_path)

    X = rna_hvg.X.toarray() if hasattr(rna_hvg.X, "toarray") else rna_hvg.X.copy()
    Y = pro_data.X.toarray() if hasattr(pro_data.X, "toarray") else pro_data.X.copy()
    protein_names = list(pro_data.var_names)

    # use only fold 1's training split - not test
    splits = load_cv_split(split_path=cv_split_path)
    train_idx = splits[0]["train"]
    X_tr, Y_tr = X[train_idx], Y[train_idx]

    # sample a few proteins by variance - low, mid, high expression variability
    protein_var = Y_tr.var(axis=0)
    sorted_idx = np.argsort(protein_var)
    sample_positions = np.linspace(0, len(sorted_idx) - 1, n_proteins_to_sample).astype(int)
    sampled_protein_idx = sorted_idx[sample_positions]

    param_distributions = {
        "n_estimators": randint(100, 600),
        "max_depth": randint(3, 8),
        "learning_rate": uniform(0.01, 0.19),  # 0.01 - 0.20
        "subsample": uniform(0.6, 0.4),  # 0.6 - 1.0
        "colsample_bytree": uniform(0.6, 0.4),  # 0.6 - 1.0
    }

    search_results = []
    best_params_per_protein = []

    for j in sampled_protein_idx:
        base_model = XGBRegressor(
            tree_method="hist",
            device="cuda",          # for kaggle, to use GPU
            random_state=random_state,
            n_jobs=1,
        )
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring="r2",
            cv=3,
            random_state=random_state,
            n_jobs=1,
        )
        search.fit(X_tr, Y_tr[:, j])

        best_params_per_protein.append(search.best_params_)
        search_results.append({
            "protein": protein_names[j],
            "best_cv_r2": search.best_score_,
            **search.best_params_,
        })
        print(f"Protein {protein_names[j]}: best CV R2={search.best_score_:.3f}, "
              f"params={search.best_params_}")

    search_results_df = pd.DataFrame(search_results)

    # average numeric hyperparams across sampled proteins
    best_params = {
        "n_estimators": int(round(search_results_df["n_estimators"].mean())),
        "max_depth": int(round(search_results_df["max_depth"].mean())),
        "learning_rate": float(search_results_df["learning_rate"].mean()),
        "subsample": float(search_results_df["subsample"].mean()),
        "colsample_bytree": float(search_results_df["colsample_bytree"].mean()),
    }

    print(f"\nRecommended global xgb_params (hard-code into xgboost_regression.py):")
    print(best_params)

    return best_params, search_results_df