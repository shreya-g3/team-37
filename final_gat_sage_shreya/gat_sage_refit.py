import os
import json
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.decomposition import TruncatedSVD
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from gat_sage_model import DualBranchGNN
from preprocessing_final import preprocess_rna, preprocess_protein_train, inverse_transform_protein
from gat_sage_graph_construction import (
    build_spatial_graph_from_coords, build_expression_graph, transform_svd,
    build_spatial_graph_from_coords_asymmetric, build_expression_graph_asymmetric,
    select_supervised_genes, build_supervised_gene_features,
)
from gat_sage_train import (
    SEED, N_SVD_COMPONENTS, K_SPATIAL, K_EXPRESSION, BASE_LR, WEIGHT_DECAY, WARMUP_EPOCHS,
    BATCH_SIZE, EVAL_NUM_NEIGHBORS, SUPERVISED_TOP_K_PER_PROTEIN,
    WarmupCosineScheduler,
    compute_qc_mask, compute_protein_missing_mask, build_fold_pyg_data, run_epoch,
)


# submission column order (from valid_uniform_range.csv)
SUBMISSION_PROTEIN_ORDER = [
    "synd", "FOXP3", "CD16", "CD31", "CXCL13", "Ki67", "OLIG2", "CXCR5", "HLA-A",
    "PD-L1", "PSD95", "CD20", "CD68", "CD44", "SMA", "MSH6", "CD23", "GFAP", "SYNA",
    "Podoplanin", "Vimentin", "CD47", "CD74", "SIRP", "Granzyme B", "IDH1", "MPO",
    "CD45", "CD21", "FIBR", "C-KIT", "CD3e", "TOX", "PD-1", "PDGFR", "CD4", "MAP2",
    "CD8", "MGMT", "CD38", "HLA-DR", "CD14", "ICOS", "Granzyme K",
]

# Final CV-based performance report

def report_final_metrics(out_path):
    cv_summary = pd.read_csv(os.path.join(out_path, "cv_summary.csv"))

    report = dict(
        n_folds=len(cv_summary),
        mean_r2_zscore=float(cv_summary["mean_r2"].mean()),
        std_r2_zscore=float(cv_summary["mean_r2"].std()),
        median_r2_zscore=float(cv_summary["median_r2"].mean()),
        mean_rmse_zscore=float(cv_summary["mean_rmse"].mean()),
        std_rmse_zscore=float(cv_summary["mean_rmse"].std()),
        median_rmse_zscore=float(cv_summary["median_rmse"].mean()),
        mean_pearson_zscore=float(cv_summary["pearson"].mean()),
        mean_r2_raw_codex=float(cv_summary["raw_mean_r2"].mean()),
        std_r2_raw_codex=float(cv_summary["raw_mean_r2"].std()),
        mean_rmse_raw_codex=float(cv_summary["raw_mean_rmse"].mean()),
        std_rmse_raw_codex=float(cv_summary["raw_mean_rmse"].std()),
        mean_pearson_raw_codex=float(cv_summary["raw_pearson"].mean()),
        n_epochs_per_fold=cv_summary["n_epochs"].tolist(),
    )

    print("\n" + "=" * 60)
    print("FINAL MODEL PERFORMANCE (5-fold spatially-blocked CV)")
    print("=" * 60)
    print(f"  Mean R2 (z-score)    : {report['mean_r2_zscore']:.4f} +/- {report['std_r2_zscore']:.4f}")
    print(f"  Median R2 (z-score)  : {report['median_r2_zscore']:.4f}")
    print(f"  Mean RMSE (z-score)  : {report['mean_rmse_zscore']:.4f} +/- {report['std_rmse_zscore']:.4f}")
    print(f"  Mean Pearson r       : {report['mean_pearson_zscore']:.4f}")
    print(f"  Mean R2 (raw CODEX)  : {report['mean_r2_raw_codex']:.4f} +/- {report['std_r2_raw_codex']:.4f}")
    print(f"  Mean RMSE (raw CODEX): {report['mean_rmse_raw_codex']:.4f} +/- {report['std_rmse_raw_codex']:.4f}")
    print(f"  Mean Pearson (raw)   : {report['mean_pearson_raw_codex']:.4f}")
    print("=" * 60 + "\n")

    with open(os.path.join(out_path, "final_metrics_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    return report

# Mini-batch prediction on rna_val
def predict_val(model, latent_val, coords_val, device, good_mask=None):
    """
    good_mask: optional bool array (True = QC-passing)
    When given, uses the asymmetric graph builders
    """
    if good_mask is not None:
        val_sp_ei, val_sp_ea = build_spatial_graph_from_coords_asymmetric(coords_val, good_mask, k=K_SPATIAL)
        val_ex_ei, val_ex_ew = build_expression_graph_asymmetric(latent_val, good_mask, k=K_EXPRESSION)
    else:
        val_sp_ei, val_sp_ea = build_spatial_graph_from_coords(coords_val, k=K_SPATIAL)
        val_ex_ei, val_ex_ew = build_expression_graph(latent_val, k=K_EXPRESSION)

    x = torch.tensor(latent_val, dtype=torch.float32)
    spatial_data = Data(x=x, edge_index=val_sp_ei, edge_attr=val_sp_ea)
    expr_data = Data(x=x, edge_index=val_ex_ei, edge_weight=val_ex_ew)

    n_val = x.shape[0]
    input_nodes = torch.arange(n_val)  # shuffle=False below -> batches stay in this order
    common = dict(num_neighbors=EVAL_NUM_NEIGHBORS, input_nodes=input_nodes, batch_size=BATCH_SIZE, shuffle=False)
    spatial_loader = NeighborLoader(spatial_data, **common)
    expr_loader = NeighborLoader(expr_data, **common)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for sb, eb in zip(spatial_loader, expr_loader):
            sb, eb = sb.to(device), eb.to(device)
            n_seed = sb.batch_size
            preds = model(sb.x, sb.edge_index, sb.edge_attr, eb.x, eb.edge_index, eb.edge_weight, n_seed=n_seed)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds)


def final_refit_and_predict(
    rna_train_path="train_rna.h5ad",
    protein_train_path="train_pro.h5ad",
    rna_val_path="valid_rna.h5ad",
    out_path="outputs",
    device=None,
):
    """
    Reports final CV performance metrics, refits on 100% of training data
    for the CV-determined epoch count
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_path, exist_ok=True)

    final_metrics = report_final_metrics(out_path)

    cv_summary = pd.read_csv(os.path.join(out_path, "cv_summary.csv"))
    n_epochs_final = int(round(cv_summary["n_epochs"].median()))
    print(f"Final refit epoch count (median across CV folds): {n_epochs_final}")

    rna_train = preprocess_rna(rna_train_path)
    protein_train, protein_stats = preprocess_protein_train(protein_train_path)

    protein_z = protein_train.X.astype(np.float32)
    coords = np.asarray(rna_train.obs[["array_row", "array_col"]], dtype=np.float32)
    protein_names = list(protein_train.var_names)

    missing_full = compute_protein_missing_mask(protein_train_path)
    qc_mask = compute_qc_mask(rna_train_path)
    missing_full = missing_full | qc_mask[:, None]

    train_idx_all = np.arange(rna_train.n_obs)
    train_idx_all = train_idx_all[~qc_mask[train_idx_all]]  # QC-flagged spots excluded from the
                                                              # graph

    svd = TruncatedSVD(n_components=N_SVD_COMPONENTS, random_state=SEED)
    svd.fit(rna_train.X[train_idx_all])                        # fit on good-quality rows only
    latent_all = svd.transform(rna_train.X).astype(np.float32)  # transform all rows

    gene_idx, _ = select_supervised_genes(
        rna_train.X, protein_z, missing_full, train_idx_all, top_k_per_protein=SUPERVISED_TOP_K_PER_PROTEIN,
    )
    supervised_all, gene_scaler_stats = build_supervised_gene_features(rna_train.X, gene_idx, train_idx_all)
    print(f"Supervised genes selected for final refit: {len(gene_idx)}")
    print(f"Training graph: {len(train_idx_all):,} of {rna_train.n_obs:,} spots "
          f"({rna_train.n_obs - len(train_idx_all):,} QC-flagged spots excluded)")

    feat_all = np.hstack([latent_all, supervised_all])
    in_dim = feat_all.shape[1]

    # only good-quality rows go into the training graph
    feat_train = feat_all[train_idx_all]
    coords_train = coords[train_idx_all]
    protein_z_train = protein_z[train_idx_all]
    missing_train = missing_full[train_idx_all]

    spatial_data, expr_data = build_fold_pyg_data(feat_train, coords_train, protein_z_train, missing_train)
    train_nodes = torch.arange(spatial_data.num_nodes)

    model = DualBranchGNN(in_dim=in_dim, out_dim=protein_z.shape[1],
                          dropout=0.35, edge_dropout=0.15).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    sched = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, n_epochs_final, BASE_LR)

    for epoch in range(n_epochs_final):
        train_loss, _, _ = run_epoch(model, spatial_data, expr_data, train_nodes, optimizer, device)
        lr_now = sched.step()
        if epoch % 10 == 0 or epoch == n_epochs_final - 1:
            print(f"  [final refit] epoch {epoch:4d}  train_loss={train_loss:.4f}  lr={lr_now:.2e}")

    torch.save({"state_dict": model.state_dict(), "config": model._config},
               os.path.join(out_path, "final_model.pt"))
    joblib.dump(svd, os.path.join(out_path, "final_svd.joblib"))
    joblib.dump({"gene_idx": gene_idx, "scaler_stats": gene_scaler_stats},
                os.path.join(out_path, "final_supervised_genes.joblib"))

    rna_val = preprocess_rna(rna_val_path)
    latent_val = transform_svd(rna_val, svd).astype(np.float32)
    coords_val = np.asarray(rna_val.obs[["array_row", "array_col"]], dtype=np.float32)

    supervised_val, _ = build_supervised_gene_features(
        rna_val.X, gene_idx, train_idx=np.arange(rna_val.n_obs), scaler_stats=gene_scaler_stats,
    )
    feat_val = np.hstack([latent_val, supervised_val])

    val_qc_mask = compute_qc_mask(rna_val_path)
    val_good_mask = ~val_qc_mask
    print(f"Prediction graph: {val_good_mask.sum():,} of {len(val_good_mask):,} spots pass QC "
          f"({(~val_good_mask).sum():,} flagged spots still get predictions, "
          f"but never serve as neighbour context)")

    preds_z = predict_val(model, feat_val, coords_val, device, good_mask=val_good_mask)
    preds_raw = inverse_transform_protein(preds_z, protein_stats)

    # protein_names comes from train_pro.h5ad's var_names order - reindexed to match
    preds_raw_df = pd.DataFrame(preds_raw, columns=protein_names)
    missing_targets = set(SUBMISSION_PROTEIN_ORDER) - set(protein_names)
    if missing_targets:
        raise ValueError(
            f"train_pro.h5ad is missing protein(s) required by the submission "
            f"format: {sorted(missing_targets)}. Check var_names spelling/casing."
        )
    preds_raw_df = preds_raw_df[SUBMISSION_PROTEIN_ORDER]

    # barcode / pxl_row_in_fullres / pxl_col_in_fullres pulled from valid_rna.h5ad's obs
    for col in ("pxl_row_in_fullres", "pxl_col_in_fullres"):
        if col not in rna_val.obs.columns:
            raise KeyError(
                f"'{col}' not found in valid_rna.h5ad's obs -- required for the "
                f"submission format. Check the actual column name in your file."
            )

    submission_df = pd.DataFrame({
        "barcode": rna_val.obs_names.to_numpy(),
        "pxl_row_in_fullres": rna_val.obs["pxl_row_in_fullres"].to_numpy(),
        "pxl_col_in_fullres": rna_val.obs["pxl_col_in_fullres"].to_numpy(),
    })
    submission_df = pd.concat([submission_df, preds_raw_df.reset_index(drop=True)], axis=1)

    submission_path = os.path.join(out_path, "predictions_raw_codex.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Wrote {submission_path}  ({submission_df.shape[0]:,} spots x "
          f"{len(SUBMISSION_PROTEIN_ORDER)} proteins, matching submission format)")
    print(f"Final metrics saved to {os.path.join(out_path, 'final_metrics_report.json')}")

    return submission_df, final_metrics


if __name__ == "__main__":
    final_refit_and_predict()