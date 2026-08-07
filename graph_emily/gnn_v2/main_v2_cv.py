import argparse
import torch

from gnn_v_cv import gnn_v4_svd_cv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_path", required=True)
    parser.add_argument("--protein_path", required=True)
    parser.add_argument("--cv_split_path", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--n_components", type=int, default=128)
    parser.add_argument("--svd_random_state", type=int, default=0)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max_folds", type=int, default=None,
                        help="run only the first N folds instead of all 5 (for a partial test)")
    parser.add_argument("--device", default=None)

    # optional overrides for DEFAULT_V4_PARAMS
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=None)

    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None

    param_overrides = {
        k: v for k, v in dict(
            hidden=args.hidden,
            n_layers=args.n_layers,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup=args.warmup,
        ).items() if v is not None
    }

    results_df, per_protein_df, params_used, fold_epochs = gnn_v4_svd_cv(
        rna_path=args.rna_path,
        protein_path=args.protein_path,
        cv_split_path=args.cv_split_path,
        out_dir=args.out_dir,
        device=device,
        n_components=args.n_components,
        svd_random_state=args.svd_random_state,
        params=param_overrides or None,
        k=args.k,
        max_epochs=args.max_epochs,
        patience=args.patience,
        max_folds=args.max_folds,
    )

    print("\nFinal params used:", params_used)
    print("Epochs selected per fold:", fold_epochs)


if __name__ == "__main__":
    main()
