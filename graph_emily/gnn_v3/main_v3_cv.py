import argparse
import torch

from gnn_v3_cv import run_gnn_v3_cv


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

    # optional overrides for DEFAULT_V3_PARAMS
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--loss_w", type=float, default=None,
                        help="combined_loss MSE weight; (1 - loss_w) goes to the Pearson term")
    parser.add_argument("--conv_type", choices=["sage", "gat"], default=None,
                        help="'sage' (default, matches the original fixed model) or 'gat'")
    parser.add_argument("--gat_heads", type=int, default=None,
                        help="only used when --conv_type gat")
    parser.add_argument("--use_jk", action="store_true",
                        help="enable Jumping Knowledge (concat all layer outputs before the head)")

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
            loss_w=args.loss_w,
            conv_type=args.conv_type,
            gat_heads=args.gat_heads,
            use_jk=args.use_jk if args.use_jk else None,  # only override when explicitly passed
        ).items() if v is not None
    }

    results_df, per_protein_df, params_used, fold_epochs = run_gnn_v3_cv(
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