import argparse
from gnn2 import run_gnn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_train_path", default="data/rna_train_preprocessed.h5ad")
    parser.add_argument("--rna_val_path", default="data/rna_val_preprocessed.h5ad")
    parser.add_argument("--pro_train_path", default="data/pro_train_preprocessed.h5ad")
    parser.add_argument("--pro_stats_path", default="data/protein_normalisation_stats.pkl")
    parser.add_argument("--out_path", default="results")
    args = parser.parse_args()

    run_gnn(
        rna_train_path=args.rna_train_path,
        rna_val_path=args.rna_val_path,
        pro_train_path=args.pro_train_path,
        pro_stats_path=args.pro_stats_path,
        out_path=args.out_path
    )


if __name__ == "__main__":
    main()