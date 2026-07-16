from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


def decode(values):
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def read_var_names(path):
    with h5py.File(path, "r") as handle:
        return decode(handle["var"]["_index"][:])


def read_obs_columns(path, columns):
    with h5py.File(path, "r") as handle:
        obs = {}
        for column in columns:
            values = handle["obs"][column][:]
            if values.dtype.kind in {"S", "O"}:
                obs[column] = decode(values)
            else:
                obs[column] = values
    return pd.DataFrame(obs)


def read_protein_matrix(path):
    with h5py.File(path, "r") as handle:
        x = handle["X"][:].astype(np.float32)
        protein_names = decode(handle["var"]["_index"][:])
    return x, protein_names


def load_hvg_list(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def selected_gene_indices(var_names, selected_genes):
    lookup = {gene.upper(): index for index, gene in enumerate(var_names)}
    indices = []
    names = []
    for gene in selected_genes:
        index = lookup.get(gene.upper())
        if index is not None:
            indices.append(index)
            names.append(var_names[index])
    if not indices:
        raise ValueError("None of the selected HVG genes were found in train_rna.h5ad.")
    return np.array(indices, dtype=np.int64), names


def read_normalized_selected_rna(rna_path, selected_indices):
    """Read selected columns from CSR H5AD and apply Team 37 RNA normalisation."""
    selected_indices = np.array(selected_indices, dtype=np.int64)
    with h5py.File(rna_path, "r") as handle:
        x_node = handle["X"]
        shape = tuple(int(v) for v in x_node.attrs["shape"])
        data_ds = x_node["data"]
        indices_ds = x_node["indices"]
        indptr = x_node["indptr"][:]

        row_totals = np.add.reduceat(data_ds[:], indptr[:-1]).astype(np.float32)
        empty_rows = np.diff(indptr) == 0
        row_totals[empty_rows] = 0.0

        col_map = np.full(shape[1], -1, dtype=np.int64)
        col_map[selected_indices] = np.arange(len(selected_indices), dtype=np.int64)

        out_data = []
        out_indices = []
        out_indptr = [0]

        chunk_rows = 2048
        for row_start in range(0, shape[0], chunk_rows):
            row_end = min(row_start + chunk_rows, shape[0])
            value_start = int(indptr[row_start])
            value_end = int(indptr[row_end])
            row_lengths = np.diff(indptr[row_start : row_end + 1])

            if value_start == value_end:
                out_indptr.extend([out_indptr[-1]] * (row_end - row_start))
                continue

            block_indices = indices_ds[value_start:value_end]
            block_data = data_ds[value_start:value_end].astype(np.float32)
            mapped_cols = col_map[block_indices]
            keep = mapped_cols >= 0

            if np.any(keep):
                local_rows = np.repeat(np.arange(row_end - row_start), row_lengths)
                kept_rows = local_rows[keep]
                global_rows = kept_rows + row_start
                scale = np.divide(
                    10000.0,
                    row_totals[global_rows],
                    out=np.zeros_like(row_totals[global_rows], dtype=np.float32),
                    where=row_totals[global_rows] > 0,
                )
                normalized = np.log1p(block_data[keep] * scale).astype(np.float32)
                kept_counts = np.bincount(kept_rows, minlength=row_end - row_start)
                out_data.append(normalized)
                out_indices.append(mapped_cols[keep].astype(np.int32))
            else:
                kept_counts = np.zeros(row_end - row_start, dtype=np.int64)

            cumulative = np.cumsum(kept_counts) + out_indptr[-1]
            out_indptr.extend(cumulative.tolist())

    data = np.concatenate(out_data) if out_data else np.array([], dtype=np.float32)
    indices = np.concatenate(out_indices) if out_indices else np.array([], dtype=np.int32)
    indptr_out = np.array(out_indptr, dtype=np.int64)
    return sparse.csr_matrix((data, indices, indptr_out), shape=(shape[0], len(selected_indices)))


def save_gene_names(path, genes):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.Series(genes, name="gene").to_csv(path, index=False)

