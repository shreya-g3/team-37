import argparse
from gnn_cv import gnn_svd_cv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_path", default="data/rna_train_preprocessed.h5ad")
    parser.add_argument("--protein_path", default="data/pro_train_preprocessed.h5ad")
    parser.add_argument("--cv_split_path", default="data/cv_splits_patches.json")
    parser.add_argument("--out_path", default="results")
    args = parser.parse_args()

    gnn_svd_cv(
        rna_path=args.rna_path,
        protein_path=args.protein_path,
        cv_split_path=args.cv_split_path,
        out_dir=args.out_path
    )


if __name__ == "__main__":
    main()