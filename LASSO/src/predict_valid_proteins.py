import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


def decode(values):
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def read_var_names(path):
    with h5py.File(path, "r") as handle:
        return decode(handle["var"]["_index"][:])


def read_obs_index(path):
    with h5py.File(path, "r") as handle:
        return decode(handle["obs"]["_index"][:])


def selected_gene_indices(var_names, selected_genes):
    lookup = {gene.upper(): index for index, gene in enumerate(var_names)}
    indices = [lookup[gene.upper()] for gene in selected_genes if gene.upper() in lookup]
    if not indices:
        raise ValueError("None of the selected HVG genes were found in valid_rna.h5ad.")
    return np.array(indices, dtype=np.int64)


def read_normalized_selected_rna(rna_path, selected_indices):
    selected_indices = np.array(selected_indices, dtype=np.int64)
    with h5py.File(rna_path, "r") as handle:
        x_node = handle["X"]
        shape = tuple(int(value) for value in x_node.attrs["shape"])
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


def create_valid_predictions(
    preprocessed_dir="outputs/preprocessed",
    valid_rna_path="data/valid_rna.h5ad",
    results_dir="results/lasso",
    alpha=0.002,
    max_iter=1000,
):
    preprocessed_dir = Path(preprocessed_dir)
    valid_rna_path = Path(valid_rna_path)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading saved training data...")
    x_train = sparse.load_npz(preprocessed_dir / "rna_hvg_normalized.npz").tocsr()
    y_train = np.load(preprocessed_dir / "protein_clr.npy").astype(np.float32)
    protein_names = pd.read_csv(preprocessed_dir / "protein_names.csv")["protein"].tolist()
    selected_genes = pd.read_csv(preprocessed_dir / "highly_variable_genes_used.csv")["gene"].tolist()

    print("Reading and normalising valid RNA data...")
    valid_var_names = read_var_names(valid_rna_path)
    selected_indices = selected_gene_indices(valid_var_names, selected_genes)
    x_valid = read_normalized_selected_rna(valid_rna_path, selected_indices)
    spot_ids = read_obs_index(valid_rna_path)

    if x_valid.shape[1] != x_train.shape[1]:
        raise ValueError(f"Feature mismatch: train={x_train.shape[1]}, valid={x_valid.shape[1]}")

    x_scaler = StandardScaler(with_mean=False)
    x_train_scaled = x_scaler.fit_transform(x_train)
    x_valid_scaled = x_scaler.transform(x_valid)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)

    print("Training final LASSO model and predicting validation proteins...")
    model = Lasso(alpha=alpha, max_iter=max_iter, random_state=42, selection="random")
    model.fit(x_train_scaled, y_train_scaled)
    predictions = y_scaler.inverse_transform(model.predict(x_valid_scaled))

    prediction_df = pd.DataFrame(predictions, columns=protein_names, index=spot_ids)
    prediction_df.index.name = "spot_id"
    output_path = results_dir / "valid_predicted_proteins.csv"
    prediction_df.to_csv(output_path)

    print("Saved valid prediction CSV to:", output_path)
    save_predicted_proteins_by_marker(prediction_df, results_dir)
    return prediction_df


def safe_marker_filename(marker_name):
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", marker_name).strip("_")
    return safe_name or "protein_marker"


def save_predicted_proteins_by_marker(prediction_df, results_dir="results/lasso"):
    results_dir = Path(results_dir)
    marker_dir = results_dir / "predicted_proteins_by_marker"
    marker_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for protein in prediction_df.columns:
        protein_output = prediction_df[[protein]].rename(columns={protein: "predicted_protein_value"})
        output_path = marker_dir / f"{safe_marker_filename(protein)}_predicted_protein.csv"
        protein_output.to_csv(output_path)
        output_paths.append(output_path)

    print(f"Saved {len(output_paths)} separate protein prediction files to:", marker_dir)
    return output_paths


if __name__ == "__main__":
    create_valid_predictions()

