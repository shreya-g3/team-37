import argparse
import torch
from gnn_v import run_gnn_v4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_train_path", required=True)
    parser.add_argument("--rna_val_path", required=True)
    parser.add_argument("--pro_train_path", required=True)
    parser.add_argument("--pro_stats_path", required=True)
    parser.add_argument("--cv_split_path", required=True)
    parser.add_argument("--out_path", required=True)

    parser.add_argument("--n_components", type=int, default=128)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--epoch_finder_fold", type=int, default=1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None

    run_gnn_v4(
        rna_train_path=args.rna_train_path,
        rna_val_path=args.rna_val_path,
        pro_train_path=args.pro_train_path,
        pro_stats_path=args.pro_stats_path,
        cv_split_path=args.cv_split_path,
        out_path=args.out_path,
        n_components=args.n_components,
        k=args.k, hidden=args.hidden, n_layers=args.n_layers, dropout=args.dropout,
        lr=args.lr, weight_decay=args.weight_decay, warmup=args.warmup,
        max_epochs=args.max_epochs, patience=args.patience,
        epoch_finder_fold=args.epoch_finder_fold, device=device,
    )


if __name__ == "__main__":
    main()