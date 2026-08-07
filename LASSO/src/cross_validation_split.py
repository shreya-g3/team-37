import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from src.h5ad_sparse import read_obs_columns


def create_spatial_cv_splits(rna_path, output_dir, n_splits=5, block_size=10, random_state=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    obs = read_obs_columns(rna_path, ["array_row", "array_col"])
    array_row = obs["array_row"].to_numpy()
    array_col = obs["array_col"].to_numpy()
    n_bins = len(obs)

    block_row = (array_row // block_size).astype(np.int64)
    block_col = (array_col // block_size).astype(np.int64)
    block_id = block_row * 1_000_003 + block_col
    unique_blocks, block_groups = np.unique(block_id, return_inverse=True)
    n_blocks = len(unique_blocks)

    rng = np.random.RandomState(random_state)
    shuffled_block_order = rng.permutation(n_blocks)
    block_rank = np.empty(n_blocks, dtype=int)
    block_rank[shuffled_block_order] = np.arange(n_blocks)
    groups = block_rank[block_groups]

    splitter = GroupKFold(n_splits=n_splits)
    splits = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.arange(n_bins), groups=groups)):
        splits.append({"fold": fold, "train": train_idx.tolist(), "test": test_idx.tolist()})

    with open(output_dir / "cv_splits.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "n_splits": n_splits,
                "n_bins": n_bins,
                "n_blocks": int(n_blocks),
                "block_size": block_size,
                "random_state": random_state,
                "method": "GroupKFold on spatial grid blocks using array_row and array_col",
                "splits": splits,
            },
            handle,
        )

    with open(output_dir / "cv_splits_info.txt", "w", encoding="utf-8") as handle:
        handle.write("Spatially blocked CV split summary\n")
        handle.write("=" * 40 + "\n")
        handle.write(f"n_bins       : {n_bins:,}\n")
        handle.write(f"n_blocks     : {n_blocks:,}\n")
        handle.write(f"block_size   : {block_size} bins\n")
        handle.write(f"n_splits     : {n_splits}\n")
        handle.write(f"random_state : {random_state}\n\n")
        for split in splits:
            handle.write(f"Fold {split['fold'] + 1}: {len(split['train']):,} train | {len(split['test']):,} test\n")

    return splits


def load_cv_split(split_path):
    with open(split_path, "r", encoding="utf-8") as handle:
        return json.load(handle)["splits"]

