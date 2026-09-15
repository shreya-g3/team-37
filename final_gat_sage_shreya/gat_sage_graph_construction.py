import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import to_undirected, coalesce

# Branch 2 input: truncated SVD on preprocessed RNA (log1p, CPM-normed)

def fit_truncated_svd(rna_train, n_components=100, random_state=0):
    """
    Fit TruncatedSVD on train RNA.
    Returns fitted svd model + train latent (n_spots_train, n_components).
    """
    X_train = rna_train.X
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    latent_train = svd.fit_transform(X_train)
    return svd, latent_train


def transform_svd(rna, svd):
    """
    Project val into the train-fitted SVD space.
    """
    return svd.transform(rna.X)


# Branch A: spatial k-NN graph from Visium spot coordinates

def build_spatial_graph(rna, k=6, obs_cols=("array_row", "array_col")):
    """
    Convenience wrapper: k-NN graph from AnnData's grid coordinates.
    Defaults to obs["array_row"]/["array_col"]
    """
    coords = np.asarray(rna.obs[list(obs_cols)], dtype=float)
    return build_spatial_graph_from_coords(coords, k=k)


def build_spatial_graph_from_coords(coords, k=6):
    """
    k-NN graph over raw physical coordinates.

    edge_index[0] = source (the neighbour whose features get sent as a
    message),
    edge_index[1] = target (the node that aggregates, each
    node's own kNN point into it).
    GATv2Conv/SAGEConv aggregate from edge_index[0] into edge_index[1]

    Returns:
        edge_index: torch.LongTensor (2, n_edges), [0]=neighbour(source), [1]=own(target)
        edge_attr:  torch.FloatTensor (n_edges, 1), normalised physical distance
    """
    coords = np.asarray(coords, dtype=float)

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(coords)
    dist, idx = nn.kneighbors(coords)

    # drop self-loop
    dist, idx = dist[:, 1:], idx[:, 1:]

    n_spots = coords.shape[0]
    own = np.repeat(np.arange(n_spots), k)
    neighbor = idx.reshape(-1)
    edge_index = torch.tensor(np.stack([neighbor, own]), dtype=torch.long)  # [source, target]

    edge_dist = dist.reshape(-1)
    edge_attr = torch.tensor(edge_dist / (edge_dist.max() + 1e-8), dtype=torch.float32).unsqueeze(1)

    return edge_index, edge_attr


# Branch B: expression similarity k-NN graph, built in SVD latent space
# (feeds the residual SAGEConv expression branch -> symmetric edges +
# scalar edge_weight)

def build_spatial_graph_from_coords_asymmetric(all_coords, good_mask, k=6):
    """
    every node in all_coords gets edges, while only good_mask=True nodes can
    be a neighbour source. QC-flagged node can receive context from
    good neighbours, but cannot be context for anyone else.

    edge_index[0] = source (neighbour, always good_mask=True), edge_index[1] = target
    (query, any node).

    Returns edge_index/edge_attr in the local indexing as all_coords.
    """
    all_coords = np.asarray(all_coords, dtype=float)
    good_idx = np.where(good_mask)[0]
    if len(good_idx) < k:
        raise ValueError(f"Only {len(good_idx)} good-quality nodes, need at least k={k}")

    pool_coords = all_coords[good_idx]
    n_all = all_coords.shape[0]

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")  # +1 in case a good query matches itself
    nn.fit(pool_coords)
    dist, idx_in_pool = nn.kneighbors(all_coords)  # (n_all, k+1), indices into pool_coords

    src_list, dst_list, dist_list = [], [], []
    for i in range(n_all):
        neighbours = good_idx[idx_in_pool[i]]
        row_dist = dist[i]
        keep = neighbours != i          # drop self-match (only possible when node i is itself good)
        neighbours, row_dist = neighbours[keep][:k], row_dist[keep][:k]
        src_list.append(neighbours)     # source = good neighbour
        dst_list.append(np.full(len(neighbours), i))  # target = query node i
        dist_list.append(row_dist)

    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)
    edge_dist = np.concatenate(dist_list)

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)  # [source(good), target(query)]
    edge_attr = torch.tensor(edge_dist / (edge_dist.max() + 1e-8), dtype=torch.float32).unsqueeze(1)
    return edge_index, edge_attr


def build_expression_graph_asymmetric(all_latent, good_mask, k=10):
    """
    Flagged node may receive aggregated context from good neighbours, but must not
    be sent back out as one.

    Returns edge_index/edge_weight in same local indexing as all_latent.
    """
    all_latent = np.asarray(all_latent, dtype=np.float32)
    good_idx = np.where(good_mask)[0]
    if len(good_idx) < k:
        raise ValueError(f"Only {len(good_idx)} good-quality nodes, need at least k={k}")

    pool_latent = all_latent[good_idx]
    n_all = all_latent.shape[0]

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(pool_latent)
    dist, idx_in_pool = nn.kneighbors(all_latent)

    src_list, dst_list, sim_list = [], [], []
    for i in range(n_all):
        neighbours = good_idx[idx_in_pool[i]]
        row_dist = dist[i]
        keep = neighbours != i
        neighbours, row_dist = neighbours[keep][:k], row_dist[keep][:k]
        src_list.append(neighbours)
        dst_list.append(np.full(len(neighbours), i))
        sim_list.append(1.0 - row_dist)

    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)
    sim = np.concatenate(sim_list)

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)  # [source(good), target(query)]
    edge_weight = torch.tensor(sim, dtype=torch.float32)
    return edge_index, edge_weight


def build_expression_graph(latent, k=10):
    """
    k-NN graph over transcriptomic similarity (cosine distance in SVD latent space).
    Symmetrised (undirected)

    Returns:
        edge_index:  torch.LongTensor (2, n_edges), undirected, deduplicated
        edge_weight: torch.FloatTensor (n_edges,), cosine similarity
                     feeds ExpressionBranch's weighted neighbour aggregation
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


def select_supervised_genes(rna_sparse, protein_z, missing_mask, train_idx, top_k_per_protein=20):
    """
    Per-gene Pearson correlation against each protein, computed via sparse
    matrix-vector products

    Returns: sorted list of selected global gene column indices (union
    across all proteins' top-k), and a dict of per-protein top gene
    indices for logging.
    """
    X = rna_sparse[train_idx]  # sparse, stays sparse
    n = X.shape[0]

    mean_x = np.asarray(X.mean(axis=0)).ravel()
    X2 = X.multiply(X)
    mean_x2 = np.asarray(X2.mean(axis=0)).ravel()
    var_x = np.clip(mean_x2 - mean_x ** 2, 0, None)
    std_x = np.sqrt(var_x)
    std_x_safe = std_x.copy()
    std_x_safe[std_x_safe == 0] = np.inf

    Y = protein_z[train_idx]
    M = ~missing_mask[train_idx]

    selected = set()
    per_protein_top = {}

    for p in range(Y.shape[1]):
        valid = M[:, p]
        if valid.sum() < 10:
            continue
        y = Y[valid, p]
        std_y = y.std()
        if std_y == 0:
            continue
        y_c = y - y.mean()
        Xv = X[valid]
        cov = np.asarray(Xv.T.dot(y_c)).ravel() / valid.sum()
        r = cov / (std_x_safe * std_y)
        top = np.argsort(-np.abs(r))[:top_k_per_protein]
        selected.update(top.tolist())
        per_protein_top[p] = top

    return sorted(selected), per_protein_top


def build_supervised_gene_features(rna_sparse, gene_indices, train_idx, scaler_stats=None):
    """
    Extract + z-score the selected gene columns for all rows.

    Pass scaler_stats=None to fit (mean/std on train_idx only); pass a
    previously-fit dict back in to transform val/holdout rows without
    refitting on them.
    """
    if len(gene_indices) == 0:
        return np.zeros((rna_sparse.shape[0], 0), dtype=np.float32), scaler_stats or dict(mean=np.zeros((1, 0)), std=np.ones((1, 0)))

    X_sub = np.asarray(rna_sparse[:, gene_indices].todense(), dtype=np.float32)
    if scaler_stats is None:
        mean = X_sub[train_idx].mean(axis=0, keepdims=True)
        std = X_sub[train_idx].std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        scaler_stats = dict(mean=mean, std=std)
    X_scaled = (X_sub - scaler_stats["mean"]) / scaler_stats["std"]
    return X_scaled.astype(np.float32), scaler_stats


if __name__ == "__main__":
    from preprocessing_final import preprocess_rna

    rna_train = preprocess_rna("train_rna.h5ad")

    svd, latent_train = fit_truncated_svd(rna_train, n_components=100)

    spatial_edge_index, spatial_edge_attr = build_spatial_graph(rna_train, k=6)     # -> GATv2Conv(edge_attr=...)
    expr_edge_index, expr_edge_weight = build_expression_graph(latent_train, k=10)  # -> SAGEConv weighted pre-aggregation