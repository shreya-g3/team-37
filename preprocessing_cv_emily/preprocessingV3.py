import anndata as ad
import scanpy as sc
import numpy as np

def preprocessing(
    rna_path,                   # rna input path "train_rna.h5ad"
    protein_path,               # protein input path "train_pro.h5ad"
    hvg_path,                   # highly variable gene list input path "outputs/highly_variable_genes.txt"
    n_hvg,                      # number of highly variable genes (2000)
    output_path,                # output directory "outputs"
    protein_cofactor=5.0,       # arcsinh cofactor per protein (CODEX/CyTOF typical range ~5-150)
    protein_cap_pct=(1, 99)     # per protein percentile cap for z-score, applied post-arcsinh
    ):
    """
    preprocessing of rna and protein data
    1. load data
    2. normalise and log transform (RNA) & arcsinh transform + z-normalization (protein)
    3. select highly variable genes
    4. save data

    returns rna_hvg: normalised RNA, HVG subset
           rna_data: normalised RNA, all genes
           protein_data: arcsinh-transformed, z-normalised proteins
    """
    # 1. Load data
    rna_data = ad.read_h5ad(rna_path).copy()
    protein_data = ad.read_h5ad(protein_path).copy()

    # Copy spatial coordinates from RNA to protein data
    protein_data.obs["array_row"] = rna_data.obs["array_row"].values
    protein_data.obs["array_col"] = rna_data.obs["array_col"].values
    protein_data.obs["pxl_row_in_fullres"] = rna_data.obs["pxl_row_in_fullres"].values
    protein_data.obs["pxl_col_in_fullres"] = rna_data.obs["pxl_col_in_fullres"].values

    # 2. Normalise

    # RNA: normalise and log transform

    sc.pp.normalize_total(rna_data, target_sum=1e4)
    sc.pp.log1p(rna_data)

    # Proteins:

    X_pro = (protein_data.X.toarray()
        if hasattr(protein_data.X, "toarray")
        else protein_data.X.copy())
    X_pro = np.nan_to_num(X_pro.astype(float))

    # arcsinh transform - stabilises variance and compresses right-skew,
    # standard for CODEX/CyTOF-style protein intensity data
    X_pro_arcsinh = np.arcsinh(X_pro / protein_cofactor)

    # cap each marker post-transform - remove residual outliers
    lower_pct, upper_pct = protein_cap_pct
    p_low = np.percentile(X_pro_arcsinh, lower_pct, axis=0, keepdims=True)
    p_high = np.percentile(X_pro_arcsinh, upper_pct, axis=0, keepdims=True)
    X_pro_capped = np.clip(X_pro_arcsinh, p_low, p_high)

    # z-score per marker
    marker_mean = X_pro_capped.mean(axis=0, keepdims=True)
    marker_std = X_pro_capped.std(axis=0, keepdims=True)
    marker_std[marker_std == 0] = 1.0  # avoid divide-by-zero for constant markers
    protein_data.X = (X_pro_capped - marker_mean) / marker_std

    # 3. Highly variable gene selection - RNA only

    if hvg_path is not None:
        # load fixed list of hvgs
        with open(hvg_path, "r") as f:
            hvgs = [line.strip() for line in f if line.strip()]
        hvgs = [g for g in hvgs if g in rna_data.var_names]
    else:
        sc.pp.highly_variable_genes(
            rna_data,
            n_top_genes=n_hvg,
            flavor="cell_ranger",
            subset=False)
        hvgs = list(rna_data.var_names[rna_data.var.highly_variable])
        with open(f"{output_path}/highly_variable_genes_{n_hvg}.txt", "w") as hvg_text_file:
            hvg_text_file.write("\n".join(hvgs))

    rna_hvg = rna_data[:, hvgs].copy()

    # 4. Save output

    rna_hvg.write_h5ad(f"{output_path}/rna_hvg_{n_hvg}.h5ad")
    rna_data.write_h5ad(f"{output_path}/rna_data.h5ad")
    protein_data.write_h5ad(f"{output_path}/protein_data_v3.h5ad")

    return rna_hvg, rna_data, protein_data