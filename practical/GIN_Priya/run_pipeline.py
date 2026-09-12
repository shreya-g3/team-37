from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.predict import predict
from src.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-genes", dest="max_genes", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    overrides = {"epochs": args.epochs, "max_genes": args.max_genes}
    metrics = train(config, overrides)
    output_path = predict(config)
    print("Best validation metrics:", metrics)
    print("Prediction file:", output_path)


if __name__ == "__main__":
    main()
