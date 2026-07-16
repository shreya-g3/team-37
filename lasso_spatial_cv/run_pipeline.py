import argparse
import json
from pathlib import Path

from src.cross_validation_split import create_spatial_cv_splits
from src.lasso_sparse import run_lasso_cv
from src.preprocessing_fast import run_preprocessing


def main():
    args = parse_args()
    config = load_config(args.config)
    preprocessed_dir = Path(config["preprocessed_output_dir"])
    results_dir = Path(config["results_output_dir"])

    print("Step 1: Running memory-efficient Team 37 preprocessing")
    run_preprocessing(
        rna_path=config["rna_path"],
        protein_path=config["protein_path"],
        hvg_path=config["hvg_path"],
        output_dir=preprocessed_dir,
    )

    print("Step 2: Creating spatial cross-validation splits")
    splits = create_spatial_cv_splits(
        rna_path=config["rna_path"],
        output_dir=preprocessed_dir,
        n_splits=int(config["n_splits"]),
        block_size=int(config["block_size"]),
        random_state=int(config["random_state"]),
    )

    print("Step 3: Training LASSO and saving metrics")
    run_lasso_cv(
        x_path=preprocessed_dir / "rna_hvg_normalized.npz",
        y_path=preprocessed_dir / "protein_clr.npy",
        protein_names_path=preprocessed_dir / "protein_names.csv",
        splits=splits,
        output_dir=results_dir,
        alpha=float(config["lasso_alpha"]),
        max_iter=int(config["lasso_max_iter"]),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run Pallavi final memory-efficient LASSO project.")
    parser.add_argument("--config", default="config/config.json")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()

