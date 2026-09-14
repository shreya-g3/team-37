"""
Evaluation utilities - Pearson correlation per protein, summary tables,
and comparison across models.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import os

from config import PROTEIN_COLS, OUTPUT_DIR


def pearson_per_protein(Y_true: np.ndarray,
                         Y_pred: np.ndarray,
                         model_name: str = "model",
                         ) -> pd.DataFrame:
    """
    Compute per-protein Pearson r between predictions and ground truth.

    Returns a DataFrame with columns:
        protein | pearson_r | spearman_r | p_value | model
    sorted by pearson_r descending.
    """
    rows = []
    for i, prot in enumerate(PROTEIN_COLS):
        r_p, p_p = pearsonr(Y_true[:, i], Y_pred[:, i])
        r_s, _   = spearmanr(Y_true[:, i], Y_pred[:, i])
        rows.append({
            "protein":    prot,
            "pearson_r":  r_p,
            "spearman_r": r_s,
            "p_value":    p_p,
            "model":      model_name,
        })
    df = pd.DataFrame(rows).sort_values("pearson_r", ascending=False).reset_index(drop=True)
    return df


def print_summary(df: pd.DataFrame, model_name: str = ""):
    tag = f"[{model_name}] " if model_name else ""
    mean_r = df["pearson_r"].mean()
    med_r  = df["pearson_r"].median()
    print(f"{tag}Mean Pearson r : {mean_r:.4f}")
    print(f"{tag}Median Pearson r: {med_r:.4f}")
    print(f"{tag}Top-5 proteins:")
    print(df.head(5)[["protein","pearson_r"]].to_string(index=False))
    print(f"{tag}Bottom-5 proteins:")
    print(df.tail(5)[["protein","pearson_r"]].to_string(index=False))


def compare_models(*args: tuple[str, pd.DataFrame]) -> pd.DataFrame:
    """
    args: (model_name, per_protein_df), ...

    Returns a wide DataFrame:
        protein | lasso_r | elasticnet_r | gnn_r | best_model
    """
    merged = None
    for name, df in args:
        sub = df[["protein", "pearson_r"]].rename(columns={"pearson_r": f"{name}_r"})
        merged = sub if merged is None else merged.merge(sub, on="protein")

    r_cols = [c for c in merged.columns if c.endswith("_r")]
    merged["best_model"] = merged[r_cols].idxmax(axis=1).str.replace("_r", "")
    merged["best_r"]     = merged[r_cols].max(axis=1)
    merged = merged.sort_values("best_r", ascending=False).reset_index(drop=True)
    return merged


def save_predictions(Y_pred: np.ndarray, barcodes: list[str],
                      model_name: str, split: str = "valid"):
    """Save predictions as CSV with barcode index."""
    df = pd.DataFrame(Y_pred, index=barcodes, columns=PROTEIN_COLS)
    df.index.name = "barcode"
    path = os.path.join(OUTPUT_DIR, f"predictions_{model_name}_{split}.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(path)
    print(f"[eval] Predictions saved -> {path}")
    return path


def save_eval_table(df: pd.DataFrame, model_name: str):
    """Save per-protein evaluation table."""
    path = os.path.join(OUTPUT_DIR, f"eval_{model_name}.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[eval] Eval table saved -> {path}")
    return path
