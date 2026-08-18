import os
import pickle
import argparse

from preprocessing_final import (
    preprocess_rna,
    preprocess_protein_train,
)

from cv_split_patches import cv_split_patches

from gat_v2 import run_gnn_v5


def main():

    parser = argparse.ArgumentParser()

    # Raw input data
    parser.add_argument("--rna_train_path", required=True)
    parser.add_argument("--rna_val_path", required=True)
    parser.add_argument("--pro_train_path", required=True)

    # Output directory
    parser.add_argument("--out_path", required=True)

    # CV split parameters
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--target_patch_bins", type=int, default=2000)
    parser.add_argument("--buffer_dist", type=float, default=60)
    parser.add_argument("--tissue_label", default=None)

    # Model parameters
    parser.add_argument("--n_components", type=int, default=128)
    parser.add_argument("--k_spatial", type=int, default=8)
    parser.add_argument("--k_expression", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--epoch_finder_fold", type=int, default=1)
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    print("=== Preprocessing RNA ===")

    rna_train = preprocess_rna(args.rna_train_path)
    rna_val = preprocess_rna(args.rna_val_path)

    rna_train_processed = os.path.join(
        args.out_path,
        "rna_train_processed.h5ad",
    )

    rna_val_processed = os.path.join(
        args.out_path,
        "rna_val_processed.h5ad",
    )

    rna_train.write(rna_train_processed)
    rna_val.write(rna_val_processed)

    print("=== Preprocessing protein ===")

    protein_train, protein_stats = preprocess_protein_train(
        args.pro_train_path
    )

    protein_train_processed = os.path.join(
        args.out_path,
        "protein_train_processed.h5ad",
    )

    protein_train.write(protein_train_processed)

    protein_stats_path = os.path.join(
        args.out_path,
        "protein_stats.pkl",
    )

    with open(protein_stats_path, "wb") as f:
        pickle.dump(protein_stats, f)

    print("=== Creating CV splits ===")

    cv_split_patches(
        rna_path=rna_train_processed,
        output_path=args.out_path,
        n_splits=args.n_splits,
        random_state=args.random_state,
        target_patch_bins=args.target_patch_bins,
        buffer_dist=args.buffer_dist,
        tissue_label=args.tissue_label,
    )

    cv_split_path = os.path.join(
        args.out_path,
        "cv_splits_patches.json",
    )

    print("=== Training model ===")

    run_gnn_v5(
        rna_train_path=rna_train_processed,
        pro_train_path=protein_train_processed,
        rna_val_path=rna_val_processed,
        pro_stats_path=protein_stats_path,
        cv_split_path=cv_split_path,
        out_path=args.out_path,
        n_components=args.n_components,
        k_spatial=args.k_spatial,
        k_expression=args.k_expression,
        hidden=args.hidden,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup=args.warmup,
        max_epochs=args.max_epochs,
        patience=args.patience,
        epoch_finder_fold=args.epoch_finder_fold,
        device=args.device,
    )


if __name__ == "__main__":
    main()