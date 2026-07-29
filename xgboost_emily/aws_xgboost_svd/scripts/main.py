import argparse
from xgboost_truncsvd import xgboost_svd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_path", default="data/rna_hvg.h5ad")
    parser.add_argument("--protein_path", default="data/protein_data_v2.h5ad")
    parser.add_argument("--cv_split_path", default="data/cv_splits_patches.json")
    parser.add_argument("--hop", default=None)
    parser.add_argument("--out_path", default="results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_components", default="50")
    args = parser.parse_args()

    results_df, per_protein_results_df, xgb_params  = xgboost_svd(
        rna_path=args.rna_path,
        protein_path=args.protein_path,
        cv_split_path=args.cv_split_path,
        hop=args.hop,
        out_path=args.out_path,
        device=args.device,
        n_components=args.n_components,
        xgb_params=None,
        )

if __name__ == "__main__":
    main()