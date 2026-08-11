import os
import matplotlib
matplotlib.use("Agg")   # headless backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import scanpy as sc


def rna_clustering(
    rna_data,                       # preprocessed RNA
    resolution,                     # 0.3, 0.5, 1.0
    output_dir,                     # "results"
    n_pcs=30
    ):
    """
    Run PCA, UMAP, leiden clustering on preprocessed RNA data
    Saves a spatial plot and a UMAP plot
    Saves the RNA data with cluster labels

    Returns RNA data with cluster label added in .obs["rna_cluster"].
    """

    os.makedirs(output_dir, exist_ok=True)

    # dimensionality reduction
    n_pcs = min(n_pcs, rna_data.n_vars - 1)
    sc.tl.pca(rna_data, n_comps=n_pcs)
    sc.pp.neighbors(rna_data, n_neighbors=15, n_pcs=n_pcs)
    sc.tl.umap(rna_data)

    # Leiden clustering (resolution controls number of clusters)
    sc.tl.leiden(
        rna_data,
        resolution=resolution,
        key_added="rna_cluster",
        flavor="igraph",
        n_iterations=2,
        directed=False)

    rna_data.obs["rna_cluster"] = rna_data.obs["rna_cluster"].astype("category")
    cluster_categories = rna_data.obs["rna_cluster"].cat.categories
    n_clusters = len(cluster_categories)

    cluster_ids = rna_data.obs["rna_cluster"].cat.codes.values

    cmap = plt.get_cmap("tab20", n_clusters)
    bounds = np.arange(n_clusters + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    print(f"Leiden resolution: {resolution}, Number of clusters: {n_clusters}")
    print(f"Clusters: {rna_data.obs['rna_cluster'].value_counts()}")

    # spatial plot (to show if clusters correspond with regions)
    fig, ax = plt.subplots(figsize=(10, 10))

    scatter = ax.scatter(
        rna_data.obs["array_col"],
        rna_data.obs["array_row"],
        c=cluster_ids,
        cmap=cmap,
        norm=norm,
        s=1.5,
        linewidths=0)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, ticks=np.arange(n_clusters))
    cbar.ax.set_yticklabels(cluster_categories)
    cbar.set_label("Cluster")

    ax.set_title("Spatial map of RNA clusters", fontsize=14)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()

    spatial_path = os.path.join(output_dir, f"rna_spatial_clusters_res{resolution}.png")
    fig.savefig(spatial_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # UMAP coloured by cluster
    umap_path = os.path.join(output_dir, f"rna_umap_clusters_res{resolution}.png")
    sc.pl.umap(
        rna_data,
        color="rna_cluster",
        title="RNA clusters (Leiden)",
        legend_loc="on data",
        show=False,
        save=False  # handle saving manually to control path
        )
    plt.savefig(umap_path, dpi=300, bbox_inches="tight")
    plt.close()

    # save RNA data with cluster labels
    h5ad_path = os.path.join(output_dir, f"rna_data_clustered_res{resolution}.h5ad")
    rna_data.write_h5ad(h5ad_path)
    print(f"Saved clustered RNA data to {h5ad_path}")

    return rna_data

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--n_pcs", type=int, default=30)
    args = parser.parse_args()

    rna_data = sc.read_h5ad(args.input)
    print(f"{rna_data.shape[0]} bins x {rna_data.shape[1]} genes")

    rna_clustering(
        rna_data=rna_data,
        resolution=args.resolution,
        output_dir=args.output_dir,
        n_pcs=args.n_pcs
    )

if __name__ == "__main__":
    main()