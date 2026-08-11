import anndata as ad
import scanpy as sc
import numpy as np
import pickle

def preprocessing(
    rna_train_path,             # rna input path "train_rna.h5ad"
    rna_val_path,               # "valid_rna.h5ad"
    rna_test_path,              # "test_rna.h5ad"
    protein_path,               # protein input path "train_pro.h5ad"
    output_path,                # output directory "outputs"
    protein_cofactor=150,       # arcsinh cofactor per protein (CODEX/CyTOF typical range ~5-150)
    ):
    """
    preprocessing of rna and protein data
    1. load data
    2. rna data: normalise and log transform
    3. protein data: z-normalization + arcsinh transform
    4. save data

    returns:
        rna_data: normalised RNA, all genes
        protein_data: arcsinh-transformed, z-normalised
    """
    # 1. Load data
    rna_train_data = ad.read_h5ad(rna_train_path).copy()
    rna_val_data = ad.read_h5ad(rna_val_path).copy()
    rna_test_data = ad.read_h5ad(rna_test_path).copy()
    protein_data = ad.read_h5ad(protein_path).copy()

    # Copy spatial coordinates from RNA to protein data
    protein_data.obs["array_row"] = rna_train_data.obs["array_row"].values
    protein_data.obs["array_col"] = rna_train_data.obs["array_col"].values
    protein_data.obs["pxl_row_in_fullres"] = rna_train_data.obs["pxl_row_in_fullres"].values
    protein_data.obs["pxl_col_in_fullres"] = rna_train_data.obs["pxl_col_in_fullres"].values

    # 2. RNA: normalise and log transform

    # train data
    sc.pp.normalize_total(rna_train_data, target_sum=1e4)
    sc.pp.log1p(rna_train_data)

    # val data
    sc.pp.normalize_total(rna_val_data, target_sum=1e4)
    sc.pp.log1p(rna_val_data)

    # test data
    sc.pp.normalize_total(rna_test_data, target_sum=1e4)
    sc.pp.log1p(rna_test_data)

    # 3. Proteins:

    X_pro = (protein_data.X.toarray()
        if hasattr(protein_data.X, "toarray")
        else protein_data.X.copy())
    X_pro = np.nan_to_num(X_pro.astype(float))

    # arcsinh transform - stabilises variance and compresses right-skew (standard for CODEX protein intensity data)
    X_pro_arcsinh = np.arcsinh(X_pro / protein_cofactor)

    # z-score per marker
    marker_mean = X_pro_arcsinh.mean(axis=0, keepdims=True)
    marker_std = X_pro_arcsinh.std(axis=0, keepdims=True)
    marker_std[marker_std == 0] = 1.0  # avoid divide-by-zero for constant markers
    protein_data.X = (X_pro_arcsinh - marker_mean) / marker_std

    stats = dict(
        cofactor=protein_cofactor,
        marker_mean=marker_mean,
        marker_std=marker_std,
        marker_names=protein_data.var_names.to_numpy(),
    )

    # 4. Save output

    rna_train_data.write_h5ad(f"{output_path}/rna_train_preprocessed.h5ad")
    rna_val_data.write_h5ad(f"{output_path}/rna_val_preprocessed.h5ad")
    rna_test_data.write_h5ad(f"{output_path}/rna_test_preprocessed.h5ad")
    protein_data.write_h5ad(f"{output_path}/pro_train_preprocessed.h5ad")

    with open(f"{output_path}/pro_normalisation_stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    return rna_train_data, rna_val_data, rna_test_data, protein_data, stats

def inverse_transform_protein(z_scored_values, stats_path):
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    x_arcsinh = z_scored_values * stats["marker_std"] + stats["marker_mean"]
    x_original = np.sinh(x_arcsinh) * stats["cofactor"]
    return x_original
