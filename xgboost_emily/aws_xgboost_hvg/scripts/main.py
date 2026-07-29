import argparse
from xgboost_hvg import xgboost_hvg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_path", default="data/rna_hvg.h5ad")
    parser.add_argument("--protein_path", default="data/protein_data_v2.h5ad")
    parser.add_argument("--cv_split_path", default="data/cv_splits.json")
    parser.add_argument("--hop", default=None)
    parser.add_argument("--out_path", default="results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    results_df, per_protein_results_df, xgb_params  = xgboost_hvg(
        rna_path=args.rna_path,
        protein_path=args.protein_path,
        cv_split_path=args.cv_split_path,
        hop=args.hop,
        out_path=args.out_path,
        xgb_params=None,
        device=args.device)

if __name__ == "__main__":
    main()