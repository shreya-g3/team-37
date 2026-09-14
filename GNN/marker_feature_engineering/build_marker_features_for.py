"""Build marker-gene feature matrix for ANY rna h5ad file (train, valid, or
test), reusing the exact gene list + column order recorded in
marker_meta.json (from the original train_rna.h5ad run) so the output is
guaranteed feature-compatible with the trained model.

Usage:
    python3 build_marker_features_for.py --rna <path_to_rna.h5ad> --out <out.npy> [--meta marker_meta.json]

Normalisation matches build_marker_features.py: normalize_total(1e4) -> log1p,
using the full-panel library size (computed from ALL genes in that RNA file,
not just the marker subset).
"""
import argparse
import json
import os

import h5py
import numpy as np


def names(path, grp):
    with h5py.File(path, 'r') as f:
        g = f[grp]
        idx = g.attrs.get('_index', '_index')
        if isinstance(idx, bytes):
            idx = idx.decode()
        key = idx if idx in g else ('_index' if '_index' in g else list(g.keys())[0])
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[key][:]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rna", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "marker_meta.json"))
    args = ap.parse_args()

    with open(args.meta) as f:
        meta = json.load(f)
    marker_genes = [m["gene"] for m in meta["markers"]]
    print(f"Reusing {len(marker_genes)} marker genes from {args.meta}")

    genes = names(args.rna, 'var')
    gidx = {g: i for i, g in enumerate(genes)}
    obs = names(args.rna, 'obs')
    n = len(obs)

    missing = [g for g in marker_genes if g not in gidx]
    if missing:
        print(f"  WARNING: {len(missing)} marker genes not found in this panel, will be zero-filled: {missing}")
    cols = [gidx.get(g, -1) for g in marker_genes]

    with h5py.File(args.rna, 'r') as f:
        X = f['X']
        data = X['data']; indices = X['indices']; indptr = X['indptr'][:]
        assert dict(X.attrs).get('encoding-type') == 'csr_matrix'
        raw = np.zeros((n, len(marker_genes)), dtype=np.float64)
        tot = np.zeros(n, dtype=np.float64)
        col_pos = {c: j for j, c in enumerate(cols) if c >= 0}
        CH = 20000
        for r0 in range(0, n, CH):
            r1 = min(n, r0 + CH)
            s, e = int(indptr[r0]), int(indptr[r1])
            d = data[s:e].astype(np.float64); ix = indices[s:e]
            rows = np.repeat(np.arange(r0, r1), np.diff(indptr[r0:r1 + 1]))
            np.add.at(tot, rows, d)
            m = np.isin(ix, list(col_pos.keys()))
            if m.any():
                rr = rows[m]; jj = np.array([col_pos[c] for c in ix[m]]); dd = d[m]
                np.add.at(raw, (rr, jj), dd)
            print(f"  rows {r1}/{n}", end="\r")
    print()

    tot[tot == 0] = 1.0
    feat = np.log1p(raw / tot[:, None] * 1e4).astype(np.float32)
    np.save(args.out, feat)
    print(f"Saved {args.out}  shape={feat.shape}  mean={feat.mean():.3f}  nonzero%={100*(feat > 0).mean():.1f}")


if __name__ == "__main__":
    main()
