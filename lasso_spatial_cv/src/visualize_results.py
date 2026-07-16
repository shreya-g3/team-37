from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_fold_pearson_plot(fold_metrics, output_path):
    plt.figure(figsize=(8, 5))
    x = fold_metrics["fold"] + 1
    plt.plot(x, fold_metrics["mean_pearson"], marker="o", linewidth=2)
    plt.axhline(fold_metrics["mean_pearson"].mean(), linestyle="--", linewidth=1.5)
    plt.xticks(x)
    plt.xlabel("Spatial CV fold")
    plt.ylabel("Mean Pearson correlation")
    plt.title("LASSO performance across spatial CV folds")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_summary_metrics_plot(summary, output_path):
    metrics = {
        "Pearson": summary.loc[0, "mean_pearson"],
        "Spearman": summary.loc[0, "mean_spearman"],
        "R2": summary.loc[0, "mean_r2"],
        "RMSE": summary.loc[0, "mean_rmse"],
        "MAE": summary.loc[0, "mean_mae"],
    }
    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics.keys(), metrics.values())
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.3f}", ha="center", va="bottom")
    plt.ylabel("Metric value")
    plt.title("Overall LASSO model metrics")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_top_protein_pearson_plot(per_protein, output_path, top_n=15):
    top = per_protein.sort_values("mean_pearson", ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    plt.barh(top["protein"], top["mean_pearson"])
    plt.gca().invert_yaxis()
    plt.xlabel("Mean Pearson correlation")
    plt.ylabel("Protein marker")
    plt.title(f"Top {top_n} predicted protein markers")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_metric_heatmap(per_protein, output_path, top_n=20):
    columns = ["mean_pearson", "mean_spearman", "mean_r2", "mean_rmse", "mean_mae"]
    top = per_protein.sort_values("mean_pearson", ascending=False).head(top_n).set_index("protein")
    data = top[columns].to_numpy(dtype=float)

    plt.figure(figsize=(9, max(6, top_n * 0.32)))
    plt.imshow(data, aspect="auto", cmap="viridis")
    plt.colorbar(label="Metric value")
    plt.xticks(np.arange(len(columns)), ["Pearson", "Spearman", "R2", "RMSE", "MAE"], rotation=30, ha="right")
    plt.yticks(np.arange(len(top.index)), top.index)
    plt.title(f"Metric heatmap for top {top_n} proteins")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_error_vs_correlation_plot(per_protein, output_path):
    plt.figure(figsize=(8, 5))
    plt.scatter(per_protein["mean_rmse"], per_protein["mean_pearson"], alpha=0.75)
    for _, row in per_protein.sort_values("mean_pearson", ascending=False).head(5).iterrows():
        plt.text(row["mean_rmse"], row["mean_pearson"], row["protein"], fontsize=8)
    plt.xlabel("Mean RMSE")
    plt.ylabel("Mean Pearson correlation")
    plt.title("Protein-level error versus correlation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def create_visualizations(results_dir="results/lasso", output_dir="results/figures"):
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(results_dir / "results_summary.csv")
    fold_metrics = pd.read_csv(results_dir / "fold_metrics.csv")
    per_protein = pd.read_csv(results_dir / "per_protein_metrics.csv")

    save_fold_pearson_plot(fold_metrics, output_dir / "01_fold_pearson_trend.png")
    save_summary_metrics_plot(summary, output_dir / "02_overall_metrics_barplot.png")
    save_top_protein_pearson_plot(per_protein, output_dir / "03_top15_protein_pearson.png")
    save_metric_heatmap(per_protein, output_dir / "04_top20_protein_metric_heatmap.png")
    save_error_vs_correlation_plot(per_protein, output_dir / "05_rmse_vs_pearson_scatter.png")

    print("Saved 5 visualizations to:", output_dir)


if __name__ == "__main__":
    create_visualizations()
