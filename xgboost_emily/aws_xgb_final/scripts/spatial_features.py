import numpy as np
import anndata as ad
from scipy import sparse
from scipy.spatial import cKDTree


def build_neighbourhood_adjacency(coords, radius, chunk_size=5000, verbose=True):
    """
    build a sparse row-normalised adjacency/weight matrix connecting bins
    within `radius` bin-units of each other (circular Euclidean neighbourhood)

    coords: (n_bins, 2) array of array_row/array_col values
    radius: neighbourhood radius in bin units (e.g. 3, 30, 60)
    chunk_size: number of bins to query per batch, for progress reports
    verbose: print progress as chunks complete

    returns: sparse (n_bins x n_bins) CSR matrix
        Row i holds uniform weights (1/n_neighbours) over bins within `radius` of bin i (self included,
        since query_ball_point returns the point itself at distance 0). A @ X gives the neighbourhood mean of any per-bin matrix X.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]

    tree = cKDTree(coords)

    rows_chunks, cols_chunks, vals_chunks = [], [], []
    n_chunks = int(np.ceil(n / chunk_size))

    for chunk_i in range(n_chunks):
        start = chunk_i * chunk_size
        end = min(start + chunk_size, n)

        neighbour_lists = tree.query_ball_point(coords[start:end], r=radius, p=2)
        counts = np.array([len(nb) for nb in neighbour_lists], dtype=np.int64)

        # int32 for indices, float32 for weights
        chunk_rows = np.repeat(np.arange(start, end, dtype=np.int32), counts)
        chunk_cols = np.concatenate(neighbour_lists).astype(np.int32) if len(neighbour_lists) else np.array([],
                                                                                                            dtype=np.int32)
        chunk_vals = (np.repeat(1.0, len(chunk_cols)) / np.repeat(counts, counts)).astype(np.float32)

        rows_chunks.append(chunk_rows)
        cols_chunks.append(chunk_cols)
        vals_chunks.append(chunk_vals)

        if verbose:
            avg_neighbours = counts.mean() if len(counts) else 0
            pct = 100 * end / n
            print(f"    adjacency (radius={radius}): {end}/{n} bins ({pct:.1f}%), "
                  f"avg {avg_neighbours:.0f} neighbours/bin, "
                  f"{sum(len(c) for c in cols_chunks):,} nonzeros so far")

    rows = np.concatenate(rows_chunks)
    cols = np.concatenate(cols_chunks)
    vals = np.concatenate(vals_chunks)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return A


def build_adjacency_matrices(rna_train_path, rna_val_path, hop, out_path):
    """
    build train and val spatial neighbourhood adjacency matrices in one call

    saves A_train_hop{hop}.npz and A_val_hop{hop}.npz to out_path
    returns A_train, A_val
    """
    rna_train = ad.read_h5ad(rna_train_path)
    rna_val = ad.read_h5ad(rna_val_path)

    train_coords = rna_train.obs[["array_row", "array_col"]].to_numpy()
    val_coords = rna_val.obs[["array_row", "array_col"]].to_numpy()

    print(f"Building train adjacency (hop={hop}, n_bins={len(train_coords)})...")
    A_train = build_neighbourhood_adjacency(train_coords, radius=hop)
    sparse.save_npz(f"{out_path}/A_train_hop{hop}.npz", A_train)
    print(f"Saved A_train_hop{hop}.npz ({A_train.nnz:,} nonzeros)")

    print(f"Building val adjacency (hop={hop}, n_bins={len(val_coords)})...")
    A_val = build_neighbourhood_adjacency(val_coords, radius=hop)
    sparse.save_npz(f"{out_path}/A_val_hop{hop}.npz", A_val)
    print(f"Saved A_val_hop{hop}.npz ({A_val.nnz:,} nonzeros)")

    print(f"\nDone. Saved both files to {out_path}. Upload them to S3, then pass their "
          f"paths as A_train_path/A_val_path into xgb_svd_final().")

    return A_train, A_val