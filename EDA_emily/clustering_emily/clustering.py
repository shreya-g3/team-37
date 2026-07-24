
import scanpy as sc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from cross_validation_split import load_cv_split

# stage 1: cluster proteins

def protein_clustering(
    protein_data,                   # preprocessed protein data "protein_data.h5ad"
    resolution                      # 0.3, 0.5, 1.0
    ):
    """
    Run PCA, UMAP, leiden; produce spatial and UMAP plot
    Returns protein data with assigned cluster
    """

    # dimensionality reduction
    n_pcs = min(30, protein_data.n_vars - 1)
    sc.tl.pca(protein_data, n_comps=n_pcs)
    sc.pp.neighbors(protein_data, n_neighbors=15, n_pcs=n_pcs)
    sc.tl.umap(protein_data)

    # Leiden clustering (resolution controls number of clusters)
    sc.tl.leiden(
        protein_data,
        resolution=resolution,
        key_added="protein_cluster",
        flavor="igraph",
        n_iterations=2,
        directed=False)

    protein_data.obs["protein_cluster"] = protein_data.obs["protein_cluster"].astype("category")
    cluster_categories = protein_data.obs["protein_cluster"].cat.categories
    n_clusters = len(cluster_categories)

    cluster_ids = protein_data.obs["protein_cluster"].cat.codes.values

    cmap = plt.get_cmap("tab20", n_clusters)
    bounds = np.arange(n_clusters + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    print(f"Leiden resolution: {resolution}, Number of clusters: {n_clusters}")
    print(f"Clusters: {protein_data.obs['protein_cluster'].value_counts()}")

    # spatial plot - to show if clusters correspond with regions
    fig, ax = plt.subplots(figsize=(10, 10))

    scatter = ax.scatter(
        protein_data.obs["array_col"],
        protein_data.obs["array_row"],
        c=cluster_ids,
        cmap=cmap,
        norm=norm,
        s=1.5,
        linewidths=0)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, ticks=np.arange(n_clusters))
    cbar.ax.set_yticklabels(cluster_categories)
    cbar.set_label("Cluster")

    ax.set_title("Spatial map of protein clusters", fontsize=14)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    # UMAP coloured by cluster
    sc.pl.umap(
        protein_data,
        color="protein_cluster",
        title="Protein clusters (Leiden)",
        legend_loc="on data"
        )

    # dot plot
    sc.pl.dotplot(
        protein_data,
        var_names=protein_data.var_names.tolist(),
        groupby="protein_cluster",
        standard_scale="var"
        )

    return protein_data

# stage 2

def cluster_genes(
    rna_data,               # preprocessed rna data
    protein_clustered,
    n_markers
    ):
    """
    Transfers protein cluster labels to rna bins; find marker genes per cluster using Wilcoxon rank-sum test, return rna data with cluster labels
    """
    # add cluster labels to rna data
    rna_data.obs["protein_cluster"] = protein_clustered.obs["protein_cluster"].values

    # find marker genes per cluster using Wilcoxon rank-sum
    sc.tl.rank_genes_groups(
        rna_data, groupby="protein_cluster",
        method="wilcoxon", key_added="rank_genes_protein_clusters",
        pts=True)  # include fraction of cells expressing each gene

    # table for genes per cluster
    result = rna_data.uns["rank_genes_protein_clusters"]  # cluster results
    clusters = result["names"].dtype.names  # cluster names
    rows = []
    for cl in clusters:
        names = result["names"][cl][:n_markers]
        scores = result["scores"][cl][:n_markers]  # Wilxocon test statistic
        pvals = result["pvals_adj"][cl][:n_markers]  # p-value (Benjamini-Hochberg)
        lfc = result["logfoldchanges"][cl][:n_markers]  # change vs the rest
        for rank, (g, s, p, l) in enumerate(zip(names, scores, pvals, lfc), 1):
            rows.append({"cluster": cl, "rank": rank, "gene": g,
                         "score": round(s, 4), "log2FC": round(l, 4),
                         "pval_adj": f"{p:.2e}"})

    markers_df = pd.DataFrame(rows)

    return rna_data, markers_df


# stage 3: rna classifier

def rna_classifier(
    rna_data,
    split_path,                 # "../preprocessing/outputs/cv_splits.json"
    ):
    """
    train random forest on highly variable genes from rna data to predict protein clusters
    return trained model and feature gene list
    """

    X = rna_data.X
    X = X.toarray() if hasattr(X, "toarray") else X

    # convert string labels to integers for random forest model
    le = LabelEncoder()
    y = le.fit_transform(rna_data.obs["protein_cluster"].astype(str))
    print(f"Classes: {list(le.classes_)}")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        n_jobs=1,
        random_state=42,
        class_weight="balanced")

    # load CV split and convert to (train_idx, test_idx) tuples
    splits = load_cv_split(split_path)
    cv_folds = [(np.array(s["train"]), np.array(s["test"])) for s in splits]

    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="f1_macro", n_jobs=-1)
    print(f"{len(cv_folds)}-fold CV F1-macro: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    y_pred_cv = cross_val_predict(model, X, y, cv=cv_folds, n_jobs=-1)
    print(classification_report(y, y_pred_cv, target_names=le.classes_))

    # fit final model on all data
    model.fit(X, y)

    return model, le