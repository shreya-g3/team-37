import anndata as ad
import scanpy as sc
import numpy as np


def preprocess_rna(rna_path):
    """
    normalise + log1p RNA counts - per-cell
    apply to rna_train and rna_val
    """
    rna = ad.read_h5ad(rna_path).copy()
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    return rna


def preprocess_protein_train(protein_path, protein_cap_pct=(1, 99), protein_cofactor=5.0):
    """
    preprocess protein data: arcsinh -> percentile clip -> z-score.
    computed on train data only
    returns processed h5ad file and stats dict for inverse z-scoring of predictions (back to raw CODEX) for submission
    """
    protein = ad.read_h5ad(protein_path).copy()

    X = protein.X.toarray() if hasattr(protein.X, "toarray") else protein.X.copy()
    X = np.nan_to_num(X.astype(float))

    # arcsinh transform (fixed cofactor -- not fit, so no leakage risk either way)
    X_asinh = np.arcsinh(X / protein_cofactor)

    # percentile clip -- fit on train
    lower_pct, upper_pct = protein_cap_pct
    p_low = np.percentile(X_asinh, lower_pct, axis=0, keepdims=True)
    p_high = np.percentile(X_asinh, upper_pct, axis=0, keepdims=True)
    X_clipped = np.clip(X_asinh, p_low, p_high)

    # z-score -- fit on train
    marker_mean = X_clipped.mean(axis=0, keepdims=True)
    marker_std = X_clipped.std(axis=0, keepdims=True)
    marker_std[marker_std == 0] = 1.0  # avoid divide-by-zero for constant markers

    protein.X = (X_clipped - marker_mean) / marker_std

    stats = dict(
        cofactor=protein_cofactor,
        p_low=p_low,
        p_high=p_high,
        marker_mean=marker_mean,
        marker_std=marker_std,
    )
    return protein, stats


def inverse_transform_protein(Z, stats):
    """
    inverse of preprocess_protein_train's z-score + arcsinh, using stats fit on train
    inverts the z-score and arcsinh exactly, but not percentile clip
    """
    X_clipped_approx = Z * stats["marker_std"] + stats["marker_mean"]
    X_raw_approx = np.sinh(X_clipped_approx) * stats["cofactor"]
    return X_raw_approx