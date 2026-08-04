import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import squidpy as sq
import scipy.sparse as sp
import os
import sys
from scipy.sparse import diags
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# PATHS
BASE_PATH = '/home/ubuntu'
RNA_PATH = f'{BASE_PATH}/train_rna.h5ad'
PROTEIN_PATH = f'{BASE_PATH}/train_pro.h5ad'
VALID_PATH = f'{BASE_PATH}/valid_rna.h5ad'
OUTPUT_PATH = f'{BASE_PATH}/outputs'
os.makedirs(OUTPUT_PATH, exist_ok=True)

COORD_COLS = ['pxl_col_in_fullres', 'pxl_row_in_fullres']

# CONFIGURATION (for testing, config_1_baseline worked the best)

BEST_PARAMS = dict(n_estimators=300, max_depth=None, min_samples_leaf=2, max_features='sqrt')
BEST_CONFIG_NAME = 'config_1_baseline'

RF_N_JOBS = 4
N_HVG = 2000
PLS_N_COMPONENTS_RNA = 50
PLS_N_COMPONENTS_NEIGH = 30
MIN_CELLS_EXPRESSING = 10

MARKER_GENE_ALIASES = {
    'CD16': 'FCGR3A', 'CD31': 'PECAM1', 'CD20': 'MS4A1', 'CD68': 'CD68',
    'CD44': 'CD44', 'CD23': 'FCER2', 'CD47': 'CD47', 'CD74': 'CD74',
    'CD45': 'PTPRC', 'CD21': 'CR2', 'CD3e': 'CD3E', 'CD4': 'CD4',
    'CD8': 'CD8A', 'CD38': 'CD38', 'CD14': 'CD14', 'FOXP3': 'FOXP3',
    'CXCL13': 'CXCL13', 'CXCR5': 'CXCR5', 'Ki67': 'MKI67', 'OLIG2': 'OLIG2',
    'HLA-A': 'HLA-A', 'PD-L1': 'CD274', 'PD-1': 'PDCD1', 'PSD95': 'DLG4',
    'Vimentin': 'VIM', 'SIRP': 'SIRPA', 'Granzyme B': 'GZMB',
    'Granzyme K': 'GZMK', 'IDH1': 'IDH1', 'MPO': 'MPO', 'C-KIT': 'KIT',
    'TOX': 'TOX', 'MAP2': 'MAP2', 'MGMT': 'MGMT', 'Podoplanin': 'PDPN',
    'GFAP': 'GFAP', 'ICOS': 'ICOS', 'SMA': 'ACTA2', 'HLA-DR': 'HLA-DRA',
}

# HELPER FUNCTIONS

def preprocess_protein(X_pro_raw, cofactor=5.0, cap_pct=(1, 99)):
    X = np.nan_to_num(X_pro_raw.astype(float))
    X_arcsinh = np.arcsinh(X / cofactor)
    p_low = np.percentile(X_arcsinh, cap_pct[0], axis=0)
    p_high = np.percentile(X_arcsinh, cap_pct[1], axis=0)
    X_capped = np.clip(X_arcsinh, p_low, p_high)
    marker_mean = X_capped.mean(axis=0)
    marker_std = X_capped.std(axis=0)
    marker_std[marker_std == 0] = 1.0
    X_processed = (X_capped - marker_mean) / marker_std
    params = dict(cofactor=cofactor, p_low=p_low, p_high=p_high,
                  marker_mean=marker_mean, marker_std=marker_std)
    return X_processed, params


def select_supervised_genes(adata_rna, y_zscored, n_top, min_cells=10):
    X = adata_rna.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    n = X.shape[0]

    cells_expressing = np.asarray((X > 0).sum(axis=0)).flatten()
    eligible_idx = np.where(cells_expressing >= min_cells)[0]
    print(f"Genes eligible (expressed in >= {min_cells} spots): {len(eligible_idx)} / {X.shape[1]}")

    X_elig = X[:, eligible_idx]
    gene_mean = np.asarray(X_elig.mean(axis=0)).flatten()
    gene_sq_mean = np.asarray(X_elig.multiply(X_elig).mean(axis=0)).flatten()
    gene_var = np.clip(gene_sq_mean - gene_mean ** 2, 1e-12, None)
    gene_std = np.sqrt(gene_var)

    cov = (X_elig.T @ y_zscored) / n
    corr = cov / gene_std[:, None]
    max_abs_corr = np.max(np.abs(corr), axis=1)

    order = np.argsort(max_abs_corr)[::-1]
    top_local_idx = order[:n_top]
    top_global_idx = eligible_idx[top_local_idx]
    selected_genes = list(adata_rna.var_names[top_global_idx])

    print("Top 15 genes by max |correlation| with any protein:")
    for g, s in list(zip(selected_genes, max_abs_corr[top_local_idx]))[:15]:
        print(f"  {g}: {s:.3f}")

    return selected_genes


def build_marker_gene_matrix(adata_rna_subset, protein_names, alias_map):
    matched_cols, matched_names, unmatched = [], [], []
    lower_map = {v.lower(): v for v in adata_rna_subset.var_names}

    for prot in protein_names:
        gene = alias_map.get(prot) or lower_map.get(prot.lower())
        if gene is not None and gene in adata_rna_subset.var_names:
            matched_cols.append(gene)
            matched_names.append(prot)
        else:
            unmatched.append(prot)

    print(f"Marker-gene features matched: {len(matched_cols)} / {len(protein_names)}")
    if unmatched:
        print(f"  No gene match found for: {unmatched}")

    if len(matched_cols) == 0:
        return np.zeros((adata_rna_subset.n_obs, 0))

    sub = adata_rna_subset[:, matched_cols].copy()
    return sub.X.toarray() if sp.issparse(sub.X) else np.asarray(sub.X)


def build_rna_and_neighbour_features(adata_rna_subset, hvgs):
    present = [g for g in hvgs if g in adata_rna_subset.var_names]
    sub = adata_rna_subset[:, present].copy()

    sub.obsm['spatial'] = sub.obs[COORD_COLS].values
    sq.gr.spatial_neighbors(sub, coord_type="generic", n_neighs=6)

    X = sub.X.toarray() if sp.issparse(sub.X) else np.asarray(sub.X)

    if len(present) < len(hvgs):
        X_full = np.zeros((X.shape[0], len(hvgs)))
        idx = [hvgs.index(g) for g in present]
        X_full[:, idx] = X
        X = X_full

    dist = sub.obsp['spatial_distances']
    weights = dist.copy()
    weights.data = 1.0 / (1.0 + weights.data)
    row_sums = np.array(weights.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    norm_weights = diags(1 / row_sums) @ weights
    X_neigh = norm_weights @ X

    return X, X_neigh


def cluster_full_data(protein_subset, coords_df, resolution=0.5):
    pclust = protein_subset.copy()
    pclust.obsm['spatial'] = coords_df.values
    sc.pp.pca(pclust)
    sc.pp.neighbors(pclust)
    sq.gr.spatial_neighbors(pclust, coord_type="generic", n_neighs=6)
    expr_graph = pclust.obsp['connectivities']
    spatial_graph = pclust.obsp['spatial_connectivities']
    pclust.obsp['connectivities'] = 0.5 * expr_graph + 0.5 * spatial_graph
    sc.tl.leiden(pclust, resolution=resolution, flavor='igraph', n_iterations=2, directed=False)
    labels = pclust.obs['leiden'].values
    le = LabelEncoder()
    encoded = le.fit_transform(labels).reshape(-1, 1)
    return encoded, labels, le


def fit_pls(X, y, n_components):
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X, y)
    return pls, pls.transform(X)


# Load data (train + valid). No split

print("=" * 60)
print(f"FINALIZE RUN — {BEST_CONFIG_NAME}")
print("=" * 60)

rna_data = ad.read_h5ad(RNA_PATH).copy()
protein_data = ad.read_h5ad(PROTEIN_PATH).copy()
valid_rna = ad.read_h5ad(VALID_PATH).copy()

print(f"Train RNA:     {rna_data.shape}")
print(f"Train protein: {protein_data.shape}")
print(f"Valid RNA:     {valid_rna.shape}")

protein_data.obs["array_row"] = rna_data.obs["array_row"].values
protein_data.obs["array_col"] = rna_data.obs["array_col"].values
protein_data.obs["pxl_row_in_fullres"] = rna_data.obs["pxl_row_in_fullres"].values
protein_data.obs["pxl_col_in_fullres"] = rna_data.obs["pxl_col_in_fullres"].values

sc.pp.normalize_total(rna_data, target_sum=1e4)
sc.pp.log1p(rna_data)
sc.pp.normalize_total(valid_rna, target_sum=1e4)
sc.pp.log1p(valid_rna)

X_pro_raw = protein_data.X.toarray() if hasattr(protein_data.X, 'toarray') else protein_data.X.copy()
X_pro_processed, protein_params = preprocess_protein(X_pro_raw)
protein_data.X = X_pro_processed
y_full = X_pro_processed

print("\nSupervised gene selection...")
hvgs_full = select_supervised_genes(rna_data, y_full, N_HVG, min_cells=MIN_CELLS_EXPRESSING)

print("\nBuilding RNA + neighbour features...")
X_rna_full, X_neigh_full = build_rna_and_neighbour_features(rna_data, hvgs_full)

protein_names_all = protein_data.var_names.tolist()
print("\nBuilding marker-gene features...")
X_marker_full = build_marker_gene_matrix(rna_data, protein_names_all, MARKER_GENE_ALIASES)

print("\nSpatial Leiden clustering...")
full_coords = rna_data.obs[COORD_COLS]
cluster_encoded_full, cluster_labels_full, le_full = cluster_full_data(protein_data, full_coords)

print("\nPLS fitting...")
scaler_rna_full = StandardScaler()
X_rna_full_scaled = scaler_rna_full.fit_transform(X_rna_full)
pls_rna_full, X_pls_full = fit_pls(X_rna_full_scaled, y_full, PLS_N_COMPONENTS_RNA)

scaler_neigh_full = StandardScaler()
X_neigh_full_scaled = scaler_neigh_full.fit_transform(X_neigh_full)
pls_neigh_full, X_pls_neigh_full = fit_pls(X_neigh_full_scaled, y_full, PLS_N_COMPONENTS_NEIGH)

scaler_marker_full = StandardScaler()
if X_marker_full.shape[1] > 0:
    X_marker_full_scaled = scaler_marker_full.fit_transform(X_marker_full)
else:
    X_marker_full_scaled = X_marker_full

X_full_combined = np.hstack([X_pls_full, X_pls_neigh_full, cluster_encoded_full, X_marker_full_scaled])
scaler_final_full = StandardScaler()
X_full_final = scaler_final_full.fit_transform(X_full_combined)
print(f"Full feature matrix: {X_full_final.shape}")

print(f"\nTraining final RandomForest ({BEST_CONFIG_NAME}) on full data...")
rf_final = RandomForestRegressor(n_jobs=RF_N_JOBS, random_state=42, verbose=1, **BEST_PARAMS)
rf_final.fit(X_full_final, y_full)
print("Final model trained.")

# Build valid_rna features using the same fitted transformers

print("\nBuilding validation features...")
X_rna_valid, X_neigh_valid = build_rna_and_neighbour_features(valid_rna, hvgs_full)
X_marker_valid = build_marker_gene_matrix(valid_rna, protein_names_all, MARKER_GENE_ALIASES)

X_rna_valid_scaled = scaler_rna_full.transform(X_rna_valid)
X_pls_valid = pls_rna_full.transform(X_rna_valid_scaled)
X_neigh_valid_scaled = scaler_neigh_full.transform(X_neigh_valid)
X_pls_neigh_valid = pls_neigh_full.transform(X_neigh_valid_scaled)

if X_marker_valid.shape[1] > 0:
    X_marker_valid_scaled = scaler_marker_full.transform(X_marker_valid)
else:
    X_marker_valid_scaled = X_marker_valid

knn_cluster_full = KNeighborsClassifier(n_neighbors=5)
knn_cluster_full.fit(X_pls_full, cluster_labels_full)
valid_cluster_labels = knn_cluster_full.predict(X_pls_valid)
valid_cluster_encoded = le_full.transform(valid_cluster_labels).reshape(-1, 1)

X_valid_combined = np.hstack([X_pls_valid, X_pls_neigh_valid, valid_cluster_encoded, X_marker_valid_scaled])
X_valid_scaled_final = scaler_final_full.transform(X_valid_combined)

print("Predicting...")
y_valid_pred_processed = rf_final.predict(X_valid_scaled_final)

print("Applying inverse transform to get raw CODEX values...")
marker_mean = protein_params['marker_mean']
marker_std = protein_params['marker_std']
cofactor = protein_params['cofactor']

y_inv_zscore = (y_valid_pred_processed * marker_std) + marker_mean
y_inv_arcsinh = np.sinh(y_inv_zscore) * cofactor
y_raw = np.clip(y_inv_arcsinh, 0, None)

print(f"Prediction range: {y_raw.min():.1f} to {y_raw.max():.1f}")
print(f"Raw training range: {X_pro_raw.min():.1f} to {X_pro_raw.max():.1f}")

required_protein_order = [
    'synd', 'FOXP3', 'CD16', 'CD31', 'CXCL13', 'Ki67', 'OLIG2',
    'CXCR5', 'HLA-A', 'PD-L1', 'PSD95', 'CD20', 'CD68', 'CD44',
    'SMA', 'MSH6', 'CD23', 'GFAP', 'SYNA', 'Podoplanin', 'Vimentin',
    'CD47', 'CD74', 'SIRP', 'Granzyme B', 'IDH1', 'MPO', 'CD45',
    'CD21', 'FIBR', 'C-KIT', 'CD3e', 'TOX', 'PD-1', 'PDGFR',
    'CD4', 'MAP2', 'CD8', 'MGMT', 'CD38', 'HLA-DR', 'CD14',
    'ICOS', 'Granzyme K'
]

df_predictions = pd.DataFrame(y_raw, columns=protein_names_all)
df_predictions.insert(0, 'barcode', valid_rna.obs.index.values)
df_predictions.insert(1, 'pxl_row_in_fullres', valid_rna.obs['pxl_row_in_fullres'].values)
df_predictions.insert(2, 'pxl_col_in_fullres', valid_rna.obs['pxl_col_in_fullres'].values)
df_predictions = df_predictions[['barcode', 'pxl_row_in_fullres', 'pxl_col_in_fullres'] + required_protein_order]

output_file = f'{OUTPUT_PATH}/valid_predictions_FINAL_{BEST_CONFIG_NAME}.csv'
df_predictions.to_csv(output_file, index=False)

print(f"\nDone. Predictions saved to: {output_file}")
print(f"Shape: {df_predictions.shape} (should be {valid_rna.n_obs} rows)")
print("\nFirst row (first 6 columns):")
print(df_predictions.iloc[:, :6].head(1).to_string())