"""
5-fold spatial CV
SAGE + autoencoder
"""

import argparse
import torch

from gnn_v4_cv import run_gnn_v4_cv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_path", required=True)
    parser.add_argument("--protein_path", required=True)
    parser.add_argument("--cv_split_path", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--ae_hidden_dims", type=int, nargs="+", default=[1024, 512])
    parser.add_argument("--ae_epochs", type=int, default=100)
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
    parser.add_argument("--loss_w", type=float, default=None,
                         help="combined_loss MSE weight; (1 - loss_w) goes to the Pearson term")
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
            use_jk=args.use_jk if args.use_jk else None,  # only override when explicitly passed
        ).items() if v is not None
    }

    results_df, per_protein_df, params_used, fold_epochs = run_gnn_v4_cv(
        rna_path=args.rna_path,
        protein_path=args.protein_path,
        cv_split_path=args.cv_split_path,
        out_dir=args.out_dir,
        device=device,
        latent_dim=args.latent_dim,
        ae_hidden_dims=tuple(args.ae_hidden_dims),
        ae_epochs=args.ae_epochs,
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