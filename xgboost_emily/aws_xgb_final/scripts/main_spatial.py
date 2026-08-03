import argparse
from spatial_features import build_adjacency_matrices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_train_path", default="data/final_run/rna_train_preprocessed.h5ad")
    parser.add_argument("--rna_val_path", default="data/final_run/rna_val_preprocessed.h5ad")
    parser.add_argument("--out_path", default="results")
    parser.add_argument("--hop", type=int, default=3,
                        help="neighbourhood radius in bin units (3, 30, 60)")
    args = parser.parse_args()

    build_adjacency_matrices(
        rna_train_path=args.rna_train_path,
        rna_val_path=args.rna_val_path,
        hop=args.hop,
        out_path=args.out_path
    )


if __name__ == "__main__":
    main()