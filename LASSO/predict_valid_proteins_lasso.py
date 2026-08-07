import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


def main():
    args = parse_args()
    project_root = Path(args.project_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preprocessed_dir = project_root / "outputs" / "preprocessed"
    valid_rna_path = project_root / "data" / "valid_rna.h5ad"

    print("Loading saved training data...")
    x_train = sparse.load_npz(preprocessed_dir / "rna_hvg_normalized.npz").tocsr()
    y_train = np.load(preprocessed_dir / "protein_clr.npy").astype(np.float32)
    protein_names = pd.read_csv(preprocessed_dir / "protein_names.csv")["protein"].tolist()
    selected_genes = pd.read_csv(preprocessed_dir / "highly_variable_genes_used.csv")["gene"].tolist()

    print("Reading valid_rna.h5ad and selecting the same HVG genes...")
    valid_var_names = read_var_names(valid_rna_path)
    selected_indices = selected_gene_indices(valid_var_names, selected_genes)
    x_valid = read_normalized_selected_rna(valid_rna_path, selected_indices)
    spot_ids = read_obs_index(valid_rna_path)

    if x_valid.shape[1] != x_train.shape[1]:
        raise ValueError(f"Feature mismatch: train={x_train.shape[1]}, valid={x_valid.shape[1]}")

    print("Training final LASSO model on all training rows...")
    x_scaler = StandardScaler(with_mean=False)
    x_train_scaled = x_scaler.fit_transform(x_train)
    x_valid_scaled = x_scaler.transform(x_valid)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)

    model = Lasso(alpha=args.alpha, max_iter=args.max_iter, random_state=42, selection="random")
    model.fit(x_train_scaled, y_train_scaled)

    print("Predicting proteins for valid_rna.h5ad...")
    predictions = y_scaler.inverse_transform(model.predict(x_valid_scaled))

    submission = pd.DataFrame(predictions, columns=protein_names, index=spot_ids)
    submission.index.name = "spot_id"
    submission.to_csv(output_path)

    print(f"Saved prediction CSV: {output_path}")
    print(f"Rows: {submission.shape[0]}")
    print(f"Protein columns: {submission.shape[1]}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create Pallavi LASSO valid_rna protein prediction CSV.")
    parser.add_argument(
        "--project-root",
        default=r"C:\Users\shana\Downloads\Pallavi (LASSO)",
        help="Path to the extracted Pallavi (LASSO) project folder.",
    )
    parser.add_argument(
        "--output",
        default=r"C:\Users\shana\Documents\Codex\2026-08-01\please-update-also-for-pallvi-in\outputs\valid_predicted_proteins.csv",
        help="Output CSV path for the valid_rna protein predictions.",
    )
    parser.add_argument("--alpha", type=float, default=0.01, help="LASSO alpha value.")
    parser.add_argument("--max-iter", type=int, default=500, help="Maximum LASSO iterations.")
    return parser.parse_args()


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
        row_totals[np.diff(indptr) == 0] = 0.0

        col_map = np.full(shape[1], -1, dtype=np.int64)
        col_map[selected_indices] = np.arange(len(selected_indices), dtype=np.int64)

        out_data = []
        out_indices = []
        out_indptr = [0]

        for row_start in range(0, shape[0], 2048):
            row_end = min(row_start + 2048, shape[0])
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


if __name__ == "__main__":
    main()
