import argparse
from xgb_svd_final import xgb_svd_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_train_path", default="data/train_rna.h5ad")
    parser.add_argument("--rna_val_path", default="data/valid_rna.h5ad")
    parser.add_argument("--pro_train_path", default="data/train_pro.h5ad")
    parser.add_argument("--protein_stats_path", default="data/protein_normalisation_stats.pkl")
    parser.add_argument("--out_path", default="results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hop", type=int, default=3,
                        help="neighbourhood radius in bin units (3, 30, 60)")
    parser.add_argument("--n_components", type=int, default=50)
    parser.add_argument("--A_train_path", default="data/A_train_hop30.npz")
    parser.add_argument("--A_val_path", default="data/A_val_hop30.npz")
    args = parser.parse_args()

    xgb_svd_final(
        rna_train_path=args.rna_train_path,
        rna_val_path=args.rna_val_path,
        pro_train_path=args.pro_train_path,
        protein_stats_path=args.protein_stats_path,
        hop=args.hop,
        out_dir=args.out_path,
        device=args.device,
        n_components=args.n_components,
        A_train_path=args.A_train_path,
        A_val_path=args.A_val_path
    )


if __name__ == "__main__":
    main()