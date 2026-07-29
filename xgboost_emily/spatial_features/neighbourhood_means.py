import anndata as ad
import numpy as np

from scipy import sparse
from scipy.spatial import cKDTree


def neighbourhood_means(
        rna_path,               # rna data path "../preprocessing/outputs/rna_data.h5ad"
        out_path,               # "results"
        hop=3,                  # radius (in bin units) for circular neighbourhood mean
):
    """
    build circular (Euclidean-radius) neighbourhood-averaging adjacency matrix
    at a single radius, from spatial bin coordinates only.
    saves a sparse (bins x bins) row-normalised weight matrix as .npz --
    row i holds uniform weights (1/n_neighbours) over the bins within
    radius `hop` of bin i. apply to any per-bin matrix (raw expression,
    SVD embeddings, etc.) via A @ X to get that matrix's neighbourhood mean.
    """
    rna = ad.read_h5ad(rna_path)

    array_row, array_col = rna.obs['array_row'], rna.obs['array_col']
    coords = np.column_stack([array_row, array_col]).astype(np.float64)
    n_bins = coords.shape[0]

    tree = cKDTree(coords)  # builds spatial index

    neighbour_lists = tree.query_ball_point(coords, r=hop, p=2)  # p=2: circular (Euclidean) radius

    # build sparse row-normalised adjacency/weight matrix from neighbour_lists - (vectorised instead of looping)
    counts = np.array([len(n) for n in neighbour_lists])
    rows = np.repeat(np.arange(n_bins), counts)
    cols = np.concatenate(neighbour_lists)
    vals = np.repeat(1.0, len(cols)) / np.repeat(counts, counts)
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n_bins, n_bins))

    suffix = f"_hop{hop}" if hop is not None else "_nohop"
    adjacency_out_file = f"{out_path}/A_adjacency{suffix}.npz"
    sparse.save_npz(adjacency_out_file, A)

    return A