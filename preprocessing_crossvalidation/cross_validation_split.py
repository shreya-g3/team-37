import json
import numpy as np
import anndata as ad
from sklearn.model_selection import GroupKFold

def cv_split(
    rna_path,               # path to preprocessed rna data "rna_hvg.h5ad"
    output_path,            # "outputs"
    n_splits=5,             # number of cross validation folds
    random_state=42,        # fixed seed
    block_size=10,          # spatial patches assigned to each fold
    ):
    """
    Generate and save cross-validation split.
    Bins are grouped into patches via KMeans. These are then split into folds
    outputs:    cv_splits.json
                cv_splits_info.txt
    The split is saved as a JSON file containing train/test bin indices for each fold
    """
    # 1. Load data

    rna_data = ad.read_h5ad(rna_path)
    n_bins = rna_data.n_obs

    array_row = rna_data.obs["array_row"].to_numpy()
    array_col = rna_data.obs["array_col"].to_numpy()

    # 2. Assign bins to block
    block_row = (array_row // block_size).astype(np.int64)
    block_col = (array_col // block_size).astype(np.int64)
    block_id = block_row * 1_000_003 + block_col  # unique id per (block_row, block_col)

    unique_blocks, block_groups = np.unique(block_id, return_inverse=True)
    n_blocks = len(unique_blocks)

    # 3. Split at block level - shuffles block order, then GroupKFold
    rng = np.random.RandomState(random_state)
    shuffled_block_order = rng.permutation(n_blocks)
    block_rank = np.empty(n_blocks, dtype=int)
    block_rank[shuffled_block_order] = np.arange(n_blocks)
    groups = block_rank[block_groups]  # per-bin group label, shuffled

    gkf = GroupKFold(n_splits=n_splits)

    splits = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(np.arange(n_bins), groups=groups)):
        splits.append({
            "fold": fold,
            "train": train_idx.tolist(),
            "test": test_idx.tolist(),
        })

    # 4. Save
    split_path = f"{output_path}/cv_splits.json"
    with open(split_path, "w") as f:
        json.dump({
            "n_splits": n_splits,
            "n_bins": n_bins,
            "n_blocks": int(n_blocks),
            "block_size": block_size,
            "random_state": random_state,
            "method": "GroupKFold on spatial grid blocks (array_row/array_col)",
            "splits": splits,
        }, f)

    info_path = f"{output_path}/cv_splits_info.txt"
    with open(info_path, "w") as f:
        f.write(f"Spatially-blocked CV split summary\n")
        f.write(f"{'=' * 40}\n")
        f.write(f"n_bins       : {n_bins:,}\n")
        f.write(f"n_blocks     : {n_blocks:,}\n")
        f.write(f"block_size   : {block_size} bins\n")
        f.write(f"n_splits     : {n_splits}\n")
        f.write(f"random_state : {random_state}\n\n")
        for s in splits:
            f.write(f"Fold {s['fold'] + 1}: {len(s['train']):,} train | {len(s['test']):,} test\n")

    return splits


def load_cv_split(split_path):
    """
    example use:
    rna = ad.read_h5ad("outputs/rna_hvg.h5ad")
    protein = ad.read_h5ad("outputs/protein_data.h5ad")
    X = rna.X.toarray()
    y = protein.X
    splits = load_cv_split("outputs/cv_splits.json")
    for split in splits:
        X_train = X[split["train"]]
        X_test  = X[split["test"]]
        y_train = y[split["train"]]
        y_test  = y[split["test"]]
    """
    with open(split_path, "r") as f:
        data = json.load(f)

    return data["splits"]
