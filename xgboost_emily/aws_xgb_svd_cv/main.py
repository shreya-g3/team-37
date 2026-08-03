import argparse
from scipy import sparse

from xgb_svd_cv import xgb_svd_cv


def parse_hop(value):
    """allow --hop to be omitted/None, or an int radius (3, 30, 60)"""
    if value is None or value.lower() == "none":
        return None
    return int(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_path", default="data/preprocessed/rna_preprocessed.h5ad")
    parser.add_argument("--protein_path", default="data/preprocessed/pro_preprocessed.h5ad")
    parser.add_argument("--cv_split_path", default="data/cv_splits_patches.json")
    parser.add_argument("--hop", type=parse_hop, default=None,
                        help="neighbourhood radius in bin units (3, 30, 60), or omit for no spatial features")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_components", type=int, default=50)
    parser.add_argument("--svd_random_state", type=int, default=0)
    parser.add_argument("--A_path", default=None,
                        help="path to a precomputed full (bins x bins) adjacency .npz -- "
                             "required if --hop is set, ignored otherwise")
    args = parser.parse_args()

    if args.hop is not None and args.A_path is None:
        parser.error("--A_path is required when --hop is set")

    results_df, per_protein_results_df, xgb_params = xgb_svd_cv(
        rna_path=args.rna_path,
        protein_path=args.protein_path,
        cv_split_path=args.cv_split_path,
        hop=args.hop,
        out_dir=args.out_dir,
        device=args.device,
        n_components=args.n_components,
        svd_random_state=args.svd_random_state,
        xgb_params=None,
        A=None,
        A_path=args.A_path,
    )


if __name__ == "__main__":
    main()
