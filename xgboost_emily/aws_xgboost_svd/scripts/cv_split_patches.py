import json
import numpy as np
import anndata as ad
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from collections import defaultdict


def cv_split_patches(
    rna_path,                       # preprocessed rna data path "../clustering/results/rna_hvg_cl0.5.h5ad"
    output_path,                    # "outputs"
    n_splits=5,                     # number of cross validation folds
    random_state=42,
    target_patch_bins=2000,         # approx. bins per spatial patch
    buffer_dist=60,                 # radius to remove around test patches
    tissue_label=None,              # tissue labels column from clustering - "protein_cluster"
    ):
    """
    spatially-blocked cross-validation split, with buffer around test patches
    uses preprocessed rna data with cluster labels added

    1. partition bins into patches via KMeans
    2. assign patched to folds - with balanced tissue label composition
    3. remove buffer around training bins for each fold

    outputs:    cv_splits.json
                cv_splits_info.txt

    split is saved as a JSON file containing train/test bin indices per fold
    """

    # 1. Load data

    rna_data = ad.read_h5ad(rna_path)
    n_bins = rna_data.n_obs

    array_row = rna_data.obs["array_row"].to_numpy()
    array_col = rna_data.obs["array_col"].to_numpy()
    coords = np.column_stack([array_row, array_col]).astype(np.float64)

    # load column
    if tissue_label is not None:
        if tissue_label not in rna_data.obs.columns:
            raise ValueError(
                f"tissue_label='{tissue_label}' not found in rna_data.obs columns: "
                f"{rna_data.obs.columns.tolist()}"
            )
        tissue_label_arr = rna_data.obs[tissue_label].to_numpy()
    else:
        tissue_label_arr = None

    # 2. Partition spatial patches via KMeans

    n_patches = max(n_splits, round(n_bins / target_patch_bins))
    patch_id = KMeans(n_clusters=n_patches, random_state=random_state, n_init=10).fit_predict(coords)
    n_patches = len(np.unique(patch_id))  # in case KMeans collapses empty clusters

    if n_patches < n_splits:
        raise ValueError(
            f"n_patches ({n_patches}) < n_splits ({n_splits}) — target_patch_bins is big")

    # 3. Assign patches to folds

    rng = np.random.RandomState(random_state)
    patches = np.unique(patch_id)
    patch_to_fold = {}

    if tissue_label_arr is not None:
        groups = defaultdict(list)
        for p in patches:
            mask = patch_id == p
            labels, counts = np.unique(tissue_label_arr[mask], return_counts=True)
            dominant = labels[np.argmax(counts)]
            groups[dominant].append(p)

        fold_counter = 0
        for dom, plist in groups.items():
            plist = list(plist)
            rng.shuffle(plist)
            for p in plist:
                patch_to_fold[p] = fold_counter % n_splits
                fold_counter += 1
        assignment_method = f"stratified round-robin by tissue_label='{tissue_label}'"
    else:
        shuffled_patches = rng.permutation(patches)
        for i, p in enumerate(shuffled_patches):
            patch_to_fold[p] = i % n_splits
        assignment_method = "shuffled round-robin"

    bin_fold = np.array([patch_to_fold[p] for p in patch_id])

    # 4. Train/test split per fold with buffers removed

    splits = []
    for fold in range(n_splits):
        test_mask = bin_fold == fold
        test_idx = np.where(test_mask)[0]
        train_candidates = np.where(~test_mask)[0]

        tree = cKDTree(coords[test_idx])
        dist, _ = tree.query(coords[train_candidates], k=1)
        train_idx = train_candidates[dist >= buffer_dist]

        n_buffer_excluded = len(train_candidates) - len(train_idx)

        splits.append({
            "fold": fold,
            "train": train_idx.tolist(),
            "test": test_idx.tolist(),
            "n_buffer_bins_excluded": int(n_buffer_excluded),
        })

    # 5. Save

    split_path = f"{output_path}/cv_splits_patches.json"
    with open(split_path, "w") as f:
        json.dump({
            "n_splits": n_splits,
            "n_bins": n_bins,
            "n_patches": int(n_patches),
            "target_patch_bins": target_patch_bins,
            "buffer_dist": buffer_dist,
            "random_state": random_state,
            "patch_assignment_method": assignment_method,
            "method": "KMeans spatial patches, buffered GroupKFold-style split (array_row/array_col)",
            "splits": splits,
        }, f)

    info_path = f"{output_path}/cv_splits_patches_info.txt"
    with open(info_path, "w") as f:
        f.write(f"Spatially-patched, buffered CV split summary\n")
        f.write(f"{'=' * 45}\n")
        f.write(f"n_bins              : {n_bins:,}\n")
        f.write(f"n_patches           : {n_patches:,}\n")
        f.write(f"target_patch_bins   : {target_patch_bins}\n")
        f.write(f"buffer_dist         : {buffer_dist} bins\n")
        f.write(f"n_splits            : {n_splits}\n")
        f.write(f"random_state        : {random_state}\n")
        f.write(f"patch assignment    : {assignment_method}\n\n")
        for s in splits:
            f.write(
                f"Fold {s['fold'] + 1}: {len(s['train']):,} train | "
                f"{len(s['test']):,} test | {s['n_buffer_bins_excluded']:,} buffer excluded\n"
            )

    return splits


def load_cv_split(split_path):
    """
    example use:
    rna = ad.read_h5ad("outputs/rna_hvg.h5ad")
    protein = ad.read_h5ad("outputs/protein_data.h5ad")
    X = rna.X.toarray()
    y = protein.X
    splits = load_cv_split("outputs/cv_splits_patches.json")
    for split in splits:
        X_train = X[split["train"]]
        X_test  = X[split["test"]]
        y_train = y[split["train"]]
        y_test  = y[split["test"]]
    """
    with open(split_path, "r") as f:
        data = json.load(f)
    return data["splits"]