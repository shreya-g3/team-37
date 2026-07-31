import gc
import os
import pickle

import numpy as np
import pandas as pd
import anndata as ad
import xgboost as xgb
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from spatial_features import build_neighbourhood_adjacency
from preprocessing_final import inverse_transform_protein

# current tuned config (from CV experiments)
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


def xgb_svd_final(
        rna_train_path,  # PREPROCESSED "data/preprocessed/rna_train_preprocessed.h5ad"
        rna_val_path,  # PREPROCESSED "data/preprocessed/rna_val_preprocessed.h5ad"
        pro_train_path,  # PREPROCESSED "data/preprocessed/pro_train_preprocessed.h5ad"
        protein_stats_path,  # "data/preprocessed/protein_normalisation_stats.pkl"
        hop,  # neighbourhood radius in bin units (3, 30, 60), or None for no spatial features
        out_dir,  # "results"
        device="cpu",
        n_components=50,  # number of truncated SVD components
        svd_random_state=0,
        xgb_params=None,  # dict to override DEFAULT_XGB_PARAMS
        A_train_path=None,  # optional: precomputed adjacency .npz to skip in-process build
        A_val_path=None,  # optional: precomputed adjacency .npz for val bins
):
    """
    final train-to-val pipeline
    fits truncated SVD + one XGBoost model per protein on rna_train/pro_train
    predicts pro_val

    expects preprocessed input - normalize_total/log1p on RNA, arcsinh->clip->z-score fit on train protein

    if A_train_path/A_val_path are given, loads those precomputed adjacency
    matrices instead of building them here -- useful for hop=30/60, where
    building them in-process is memory-heavy enough to have crashed the
    instance before. Build them elsewhere (e.g. Kaggle) with build_adjacency.py,
    download, upload to S3, then pass the local paths here.

    saves per-protein models, the fitted SVD, both adjacency matrices, and the z-scored predictions to out_dir
    saves CSV with pro_val predictions in CODEX values

    returns (pred_pro_val, pred_adata_zscored, models_dir)
    """
    suffix = f"_hop{hop}" if hop is not None else "_nohop"

    default_params = dict(DEFAULT_XGB_PARAMS)
    if xgb_params:
        default_params.update(xgb_params)

    # 1. load already-preprocessed data
    print("Loading preprocessed data...")
    rna_train = ad.read_h5ad(rna_train_path)
    rna_val = ad.read_h5ad(rna_val_path)
    pro_train = ad.read_h5ad(pro_train_path)

    with open(protein_stats_path, "rb") as f:
        protein_stats = pickle.load(f)

    if list(rna_train.var_names) != list(rna_val.var_names):
        raise ValueError(
            "rna_train and rna_val gene sets/order do not match -- "
            "reindex rna_val to rna_train.var_names before continuing"
        )

    if list(pro_train.obs_names) != list(rna_train.obs_names):
        raise ValueError("pro_train and rna_train obs are not aligned -- check cell ordering")

    protein_names = list(pro_train.var_names)

    # pull out rna_val's obs before deleting rna_val
    val_obs_names = rna_val.obs_names
    val_pxl_row = rna_val.obs["pxl_row_in_fullres"].values
    val_pxl_col = rna_val.obs["pxl_col_in_fullres"].values
    val_coords = rna_val.obs[["array_row", "array_col"]].to_numpy()
    train_coords = rna_train.obs[["array_row", "array_col"]].to_numpy()

    # 3. RNA matrices for SVD (sparse)
    X_train = rna_train.X if sparse.issparse(rna_train.X) else sparse.csr_matrix(rna_train.X)
    X_val = rna_val.X if sparse.issparse(rna_val.X) else sparse.csr_matrix(rna_val.X)
    X_train = X_train.astype(np.float32).copy()
    X_val = X_val.astype(np.float32).copy()

    Y_train = pro_train.X.toarray() if sparse.issparse(pro_train.X) else np.asarray(pro_train.X)

    # done with the full AnnData objects - delete & free overhead
    del rna_train, rna_val, pro_train
    gc.collect()

    # 4. truncated SVD - fit on train only, transform val
    print("Fitting truncated SVD...")
    svd = TruncatedSVD(n_components=n_components, random_state=svd_random_state)
    X_train_svd = svd.fit_transform(X_train)
    X_val_svd = svd.transform(X_val)
    print(f"SVD cumulative explained variance (train): {svd.explained_variance_ratio_.sum():.3f}")

    with open(f"{out_dir}/svd_model{suffix}.pkl", "wb") as f:
        pickle.dump(svd, f)

    use_spatial = hop is not None
    if use_spatial:
        if A_train_path is not None and A_val_path is not None:
            # precomputed
            print(f"Loading precomputed adjacency matrices (hop={hop})...")
            A_train = sparse.load_npz(A_train_path)
            A_val = sparse.load_npz(A_val_path)

            if A_train.shape[0] != len(train_coords) or A_train.shape[1] != len(train_coords):
                raise ValueError(
                    f"A_train shape {A_train.shape} doesn't match n_train_bins="
                    f"{len(train_coords)} -- was this built from a different/"
                    f"differently-ordered rna_train file than the one used here?"
                )
            if A_val.shape[0] != len(val_coords) or A_val.shape[1] != len(val_coords):
                raise ValueError(
                    f"A_val shape {A_val.shape} doesn't match n_val_bins="
                    f"{len(val_coords)} -- was this built from a different/"
                    f"differently-ordered rna_val file than the one used here?"
                )
        else:
            # 5. spatial neighbourhood adjacency - built separately for train and val
            print(f"Building spatial adjacency (hop={hop})...")
            A_train = build_neighbourhood_adjacency(train_coords, radius=hop)
            A_val = build_neighbourhood_adjacency(val_coords, radius=hop)

            sparse.save_npz(f"{out_dir}/A_train{suffix}.npz", A_train)
            sparse.save_npz(f"{out_dir}/A_val{suffix}.npz", A_val)

        # 6. neighbourhood-mean features, post-SVD (uncentered-SVD commutativity)
        X_train_neighbour = A_train @ X_train_svd
        X_val_neighbour = A_val @ X_val_svd

        X_train_final = np.hstack([X_train_svd, X_train_neighbour])
        X_val_final = np.hstack([X_val_svd, X_val_neighbour])

        del A_train, A_val, X_train_neighbour, X_val_neighbour
    else:
        X_train_final = X_train_svd
        X_val_final = X_val_svd

    # raw sparse RNA matrices and per-hop SVD arrays - only X_train_final/X_val_final for training loop
    del X_train, X_val, X_train_svd, X_val_svd
    gc.collect()

    # 7. train one XGBoost model per protein on ALL training data, predict val.
    # each model per protein is written to disk + trained proteins are skipped on a rerun
    models_dir = f"{out_dir}/xgb_models{suffix}"
    os.makedirs(models_dir, exist_ok=True)

    print(f"Training {len(protein_names)} per-protein XGBoost models...")
    Y_val_pred = np.zeros((X_val_final.shape[0], Y_train.shape[1]), dtype=np.float32)

    for j, protein in enumerate(protein_names):
        model_path = f"{models_dir}/{protein}.pkl"

        if os.path.exists(model_path):
            print(f"  [{j + 1}/{len(protein_names)}] {protein} already trained, loading")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        else:
            model = xgb.XGBRegressor(**default_params, device=device)
            model.fit(X_train_final, Y_train[:, j], verbose=False)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"  [{j + 1}/{len(protein_names)}] {protein} done")

        Y_val_pred[:, j] = model.predict(X_val_final)
        del model
        gc.collect()

    # 8. convert predictions back to raw CODEX-intensity units
    Y_val_pred_raw = inverse_transform_protein(Y_val_pred, protein_stats)

    # keep the z-scored version around as an AnnData
    pred_adata_zscored = ad.AnnData(
        X=Y_val_pred,
        obs=pd.DataFrame(
            {"pxl_row_in_fullres": val_pxl_row, "pxl_col_in_fullres": val_pxl_col},
            index=val_obs_names,
        ),
        var=pd.DataFrame(index=protein_names),
    )
    pred_adata_zscored.write_h5ad(f"{out_dir}/protein_val_predicted_zscored{suffix}.h5ad")

    # 9. build CSV: barcode,pxl_row_in_fullres,
    # pxl_col_in_fullres,<protein columns>, raw units, column order = protein_names
    pred_pro_val = pd.DataFrame(index=val_obs_names)
    pred_pro_val.insert(0, "barcode", val_obs_names)
    pred_pro_val["pxl_row_in_fullres"] = val_pxl_row
    pred_pro_val["pxl_col_in_fullres"] = val_pxl_col
    for j, protein in enumerate(protein_names):
        pred_pro_val[protein] = Y_val_pred_raw[:, j]

    submission_path = f"{out_dir}/submission{suffix}.csv"
    pred_pro_val.to_csv(submission_path, index=False)

    print(f"\nDone. Saved to {out_dir}:")
    print(f"  - xgb_models{suffix}/ (one .pkl per protein, checkpointed as training progressed)")
    print(f"  - svd_model{suffix}.pkl")
    if use_spatial:
        print(f"  - A_train{suffix}.npz / A_val{suffix}.npz")
    print(f"  - protein_val_predicted_zscored{suffix}.h5ad  (z-scored, for sanity checks)")
    print(f"  - submission{suffix}.csv  <-- upload this ({pred_pro_val.shape[0]} rows, "
          f"{pred_pro_val.shape[1]} columns)")

    return pred_pro_val, pred_adata_zscored, models_dir