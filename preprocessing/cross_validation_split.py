import json
import numpy as np
import anndata as ad
from sklearn.model_selection import KFold

def cv_split(
    rna_path,               # path to preprocessed rna data "rna_hvg.h5ad"
    output_path,            # "outputs"
    n_splits=5,             # number of cross validation folds
    random_state=42         # fixed seed
    ):
    """
    Generate and save cross-validation split.
    outputs:    cv_splits.json
                cv_splits_info.txt
    The split is saved as a JSON file containing train/test bin indices for each fold
    """
    # 1. Load data

    rna_data = ad.read_h5ad(rna_path)
    n_bins = rna_data.n_obs

    # 2. Generate splits

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    splits = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_bins))):
        splits.append({
            "fold": fold,
            "train": train_idx.tolist(),
            "test": test_idx.tolist(),
        })

    # 3. Save
    split_path = f"{output_path}/cv_splits.json"
    with open(split_path, "w") as f:
        json.dump({
            "n_splits": n_splits,
            "n_bins": n_bins,
            "random_state": random_state,
            "splits": splits,
        }, f)

    #
    info_path = f"{output_path}/cv_splits_info.txt"
    with open(info_path, "w") as f:
        f.write(f"CV split summary\n")
        f.write(f"{'=' * 40}\n")
        f.write(f"n_bins       : {n_bins:,}\n")
        f.write(f"n_splits     : {n_splits}\n")
        f.write(f"random_state : {random_state}\n\n")
        for s in splits:
            f.write(f"Fold {s['fold'] + 1}: {len(s['train']):,} train | {len(s['test']):,}test\n")

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