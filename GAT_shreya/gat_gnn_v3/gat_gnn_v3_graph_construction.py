import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import to_undirected, coalesce


# ---------------------------------------------------------------------------
# Branch 2 input: truncated SVD on preprocessed RNA
# ---------------------------------------------------------------------------

def fit_truncated_svd(rna_train, n_components=100, random_state=0):
    """
    Fit TruncatedSVD on train RNA only (output of preprocess_rna).
    Works directly on sparse .X, no densification needed.
    Returns fitted svd model + train latent (n_spots_train, n_components).
    """
    X_train = rna_train.X
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    latent_train = svd.fit_transform(X_train)
    return svd, latent_train


def transform_svd(rna, svd):
    """
    Project val (or any other split) into the train-fitted SVD space.
    """
    return svd.transform(rna.X)


# ---------------------------------------------------------------------------
# Branch A: spatial k-NN graph from Visium spot coordinates
# ---------------------------------------------------------------------------

def build_spatial_graph(rna, k=6, coord_key="spatial"):
    """Convenience wrapper: k-NN graph straight from an AnnData's coordinates."""
    coords = np.asarray(rna.obsm[coord_key], dtype=float)
    return build_spatial_graph_from_coords(coords, k=k)


def build_spatial_graph_from_coords(coords, k=6):
    """
    k-NN graph over raw physical coordinates. Split out from build_spatial_graph
    so it can be called on a coords SUBSET (e.g. one CV fold's rows) without
    needing to slice an AnnData object first.

    Returns:
        edge_index: torch.LongTensor (2, n_edges), src -> dst
        edge_attr:  torch.FloatTensor (n_edges, 1), normalised physical distance
    """
    coords = np.asarray(coords, dtype=float)

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(coords)
    dist, idx = nn.kneighbors(coords)

    # drop self-loop (first column is always the point itself)
    dist, idx = dist[:, 1:], idx[:, 1:]

    n_spots = coords.shape[0]
    src = np.repeat(np.arange(n_spots), k)
    dst = idx.reshape(-1)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    edge_dist = dist.reshape(-1)
    edge_attr = torch.tensor(edge_dist / (edge_dist.max() + 1e-8), dtype=torch.float32).unsqueeze(1)

    return edge_index, edge_attr


# ---------------------------------------------------------------------------
# Branch B: expression similarity k-NN graph, built in SVD latent space
# (feeds a GCN/SAGE-style GNN, not GAT -> symmetric edges + scalar edge_weight
# not directed attention-style edge_attr used for the spatial branch)
# ---------------------------------------------------------------------------

def build_expression_graph(latent, k=10):
    """
    k-NN graph over transcriptomic similarity (cosine distance in SVD latent space).
    Symmetrised (undirected) since a GCN/SAGE-style layer expects a proper
    adjacency rather than the directed src->dst edges GAT's attention consumes.

    Returns:
        edge_index:  torch.LongTensor (2, n_edges), undirected, deduplicated
        edge_weight: torch.FloatTensor (n_edges,), cosine similarity (higher = more similar)
                     compatible with e.g. PyG's GCNConv(edge_weight=...)
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(latent)
    dist, idx = nn.kneighbors(latent)

    # drop self-loop, convert cosine distance -> similarity
    dist, idx = dist[:, 1:], idx[:, 1:]
    sim = 1.0 - dist

    n_spots = latent.shape[0]
    src = np.repeat(np.arange(n_spots), k)
    dst = idx.reshape(-1)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_weight = torch.tensor(sim.reshape(-1), dtype=torch.float32)

    # make undirected: reciprocal (i,j)/(j,i) pairs get averaged into one weight
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce="mean")
    edge_index, edge_weight = coalesce(edge_index, edge_weight, reduce="mean")

    return edge_index, edge_weight


# ---------------------------------------------------------------------------
# Example wiring (train split shown; apply the same svd/transform to val)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from preprocessing_final import preprocess_rna

    rna_train = preprocess_rna("rna_train.h5ad")

    svd, latent_train = fit_truncated_svd(rna_train, n_components=100)

    spatial_edge_index, spatial_edge_attr = build_spatial_graph(rna_train, k=6)     # -> GATv2Conv(edge_attr=...)
    expr_edge_index, expr_edge_weight = build_expression_graph(latent_train, k=10)  # -> GCNConv(edge_weight=...)