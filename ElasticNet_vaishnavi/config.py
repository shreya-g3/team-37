"""
IRBM Pipeline Configuration
Task: Predict 44 protein markers from spatial RNA expression
"""

import os

# --- Data Paths ---
DATA_DIR = os.environ.get("IRBM_DATA_DIR", "./IRBM")  # override with env var

TRAIN_RNA_PATH   = os.path.join(DATA_DIR, "train_rna.h5ad")
TRAIN_PRO_PATH   = os.path.join(DATA_DIR, "train_pro.h5ad")
VALID_RNA_PATH   = os.path.join(DATA_DIR, "valid_rna.h5ad")
VALID_CSV_PATH   = os.path.join(DATA_DIR, "valid_uniform_range.csv")
TEST_RNA_PATH    = os.path.join(DATA_DIR, "test_rna.h5ad")

OUTPUT_DIR = os.environ.get("IRBM_OUTPUT_DIR", "./outputs")

# --- Protein Targets (44 markers) ---
PROTEIN_COLS = [
    'synd', 'FOXP3', 'CD16', 'CD31', 'CXCL13', 'Ki67', 'OLIG2', 'CXCR5',
    'HLA-A', 'PD-L1', 'PSD95', 'CD20', 'CD68', 'CD44', 'SMA', 'MSH6',
    'CD23', 'GFAP', 'SYNA', 'Podoplanin', 'Vimentin', 'CD47', 'CD74',
    'SIRP', 'Granzyme B', 'IDH1', 'MPO', 'CD45', 'CD21', 'FIBR', 'C-KIT',
    'CD3e', 'TOX', 'PD-1', 'PDGFR', 'CD4', 'MAP2', 'CD8', 'MGMT',
    'CD38', 'HLA-DR', 'CD14', 'ICOS', 'Granzyme K',
]
N_PROTEINS = len(PROTEIN_COLS)  # 44

# --- Preprocessing ---
NORM_TARGET   = 1e4      # counts-per-cell normalization target
LOG1P         = True     # apply log1p after normalization

# --- Linear Models (Lasso / ElasticNet) ---
# We use MultiTaskElasticNet / MultiTaskLasso (joint L1/L2 on all 44 targets)
# Alpha controls total regularization strength.
LASSO_ALPHA       = 0.01
ELASTIC_ALPHA     = 0.01
ELASTIC_L1_RATIO  = 0.5      # 0 = Ridge, 1 = Lasso
LINEAR_MAX_ITER   = 1000
LINEAR_TOL        = 1e-4
LINEAR_N_JOBS     = -1       # use all CPU cores (for per-protein parallelism)

# --- GNN ---
# Input features: PCA-reduced RNA (memory constraint: 18K genes x 166K spots
# would require ~50GB as float32; we compress to PCA_N_COMPONENTS first).
PCA_N_COMPONENTS  = 256      # RNA -> PCA -> GNN input
GNN_KNN_K         = 6        # k nearest spatial neighbors per spot
GNN_HIDDEN_DIM    = 256
GNN_N_LAYERS      = 3        # number of GraphSAGE conv layers
GNN_DROPOUT       = 0.1
GNN_LR            = 1e-3
GNN_WEIGHT_DECAY  = 1e-5
GNN_EPOCHS        = 50
GNN_BATCH_SIZE    = 1024     # spots per mini-batch (NeighborLoader)
GNN_NUM_NEIGHBORS = [10, 5, 5]  # neighbors sampled per layer (len == N_LAYERS)

# Random seed
SEED = 42
