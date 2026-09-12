from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree


def _decode(values: Iterable) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def read_rna(path: str):
    return ad.read_h5ad(path, backed="r")


def read_protein_h5ad(path: str) -> Tuple[np.ndarray, list[str], list[str]]:
    """Read protein h5ad through h5py to avoid local anndata/pandas string dtype issues."""
    with h5py.File(path, "r") as handle:
        y = handle["X"][:].astype(np.float32)
        proteins = _decode(handle["var"]["_index"][:])
        obs_key = "Name" if "Name" in handle["obs"] else "_index"
        obs_names = _decode(handle["obs"][obs_key][:])
    return y, proteins, obs_names


def read_target_stats(path: str) -> Tuple[np.ndarray, np.ndarray]:
    stats = np.load(path, allow_pickle=True)
    return stats["y_mean"].astype(np.float32), stats["y_std"].astype(np.float32)


def _sparse_shape(path: str) -> tuple[int, int]:
    with h5py.File(path, "r") as handle:
        return tuple(int(v) for v in handle["X"].attrs["shape"])


def choose_gene_indices(train_adata, train_rna_path: str, protein_names: list[str], max_genes: int) -> np.ndarray:
    genes = np.asarray(train_adata.var_names.astype(str))
    selected = set()
    lower_to_idx = {g.lower(): i for i, g in enumerate(genes)}
    aliases = {
        "cd3e": "CD3E",
        "ki67": "MKI67",
        "vimentin": "VIM",
        "podoplanin": "PDPN",
        "cd31": "PECAM1",
        "cd16": "FCGR3A",
        "cd20": "MS4A1",
        "cd45": "PTPRC",
        "cd21": "CR2",
        "c-kit": "KIT",
        "pd-1": "PDCD1",
        "pd-l1": "CD274",
        "hla-a": "HLA-A",
        "hla-dr": "HLA-DRA",
        "granzyme b": "GZMB",
        "granzyme k": "GZMK",
        "sma": "ACTA2",
        "fibr": "COL1A1",
        "synd": "SDC1",
        "sirp": "SIRPA",
        "psd95": "DLG4",
    }
    for protein in protein_names:
        key = protein.lower()
        for candidate in (protein, aliases.get(key, "")):
            idx = lower_to_idx.get(candidate.lower())
            if idx is not None:
                selected.add(idx)

    sample_size = min(train_adata.n_obs, 2000)
    sample_x = train_adata[:sample_size, :].X
    if sp.issparse(sample_x):
        sample_x = sample_x.toarray()
    sample_x = np.asarray(sample_x, dtype=np.float32)
    variance = np.var(np.log1p(sample_x), axis=0)
    top_count = max(0, max_genes - len(selected))
    if top_count:
        top = np.argpartition(variance, -top_count)[-top_count:]
        selected.update(int(i) for i in top)
    return np.asarray(sorted(selected), dtype=np.int64)


def _load_selected_sparse(adata, gene_indices: np.ndarray) -> np.ndarray:
    x = adata[:, np.asarray(gene_indices, dtype=np.int64)].X
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def make_features(adata, rna_path: str, gene_indices: np.ndarray, mean=None, std=None):
    x = _load_selected_sparse(adata, gene_indices)
    x = np.log1p(x)
    coords = adata.obs[["array_row", "array_col"]].to_numpy(dtype=np.float32)
    coords = (coords - coords.mean(axis=0, keepdims=True)) / (coords.std(axis=0, keepdims=True) + 1e-6)
    x = np.concatenate([x, coords], axis=1)
    if mean is None:
        mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        std = x.std(axis=0, keepdims=True).astype(np.float32) + 1e-6
    x = (x - mean) / std
    return x.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def build_spatial_edges(adata, k: int) -> torch.Tensor:
    coords = adata.obs[["array_row", "array_col"]].to_numpy(dtype=np.float32)
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    src = np.repeat(np.arange(coords.shape[0], dtype=np.int64), k)
    dst = idx[:, 1:].reshape(-1).astype(np.int64)
    edges = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
    return torch.from_numpy(edges)


def make_split(n: int, valid_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_valid = max(1, int(round(n * valid_fraction)))
    valid_idx = np.sort(order[:n_valid])
    train_idx = np.sort(order[n_valid:])
    return train_idx, valid_idx


def normalize_targets(raw_y: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    return ((np.log1p(raw_y.astype(np.float32)) - y_mean) / (y_std + 1e-6)).astype(np.float32)


def invert_targets(norm_y: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    y = np.expm1(norm_y * (y_std + 1e-6) + y_mean)
    return np.clip(y, 0.0, None).astype(np.float32)


def save_metadata(path: Path, metadata: Dict):
    serializable = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in metadata.items()
    }
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def load_metadata(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))
