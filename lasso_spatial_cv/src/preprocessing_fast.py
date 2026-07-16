from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.h5ad_sparse import (
    load_hvg_list,
    read_normalized_selected_rna,
    read_protein_matrix,
    read_var_names,
    save_gene_names,
    selected_gene_indices,
)


def clr_normalize_protein(protein_x):
    protein_x = np.nan_to_num(protein_x.astype(np.float32))
    log_x = np.log(protein_x + 1.0)
    geom_log_mean = np.mean(log_x, axis=1, keepdims=True)
    return (log_x - geom_log_mean).astype(np.float32)


def run_preprocessing(rna_path, protein_path, hvg_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Team 37 HVG list...")
    hvg_genes = load_hvg_list(hvg_path)
    var_names = read_var_names(rna_path)
    selected_indices, selected_names = selected_gene_indices(var_names, hvg_genes)
    print(f"Using {len(selected_names)} HVG genes found in RNA file.")

    print("Reading and normalising selected RNA genes without copying full RNA matrix...")
    x_rna = read_normalized_selected_rna(rna_path, selected_indices)
    sparse.save_npz(output_dir / "rna_hvg_normalized.npz", x_rna)
    save_gene_names(output_dir / "highly_variable_genes_used.csv", selected_names)

    print("Reading and CLR-normalising protein matrix...")
    protein_x, protein_names = read_protein_matrix(protein_path)
    protein_clr = clr_normalize_protein(protein_x)
    np.save(output_dir / "protein_clr.npy", protein_clr)
    pd.Series(protein_names, name="protein").to_csv(output_dir / "protein_names.csv", index=False)

    return x_rna, protein_clr, protein_names

