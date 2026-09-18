import os
import numpy as np
import pandas as pd
import anndata as ad
import torch
from sklearn.metrics import r2_score

from dual_graph_vgae import (
    SEED, N_SVD_COMPONENTS, MARKER_GENE_ALIASES,
    to_dense, reduce_rna_svd, extract_marker_genes, build_node_features,
    build_spatial_knn_graph, build_expression_knn_graph_faiss,
    DualGraphModel, combined_loss, total_loss, mean_pearson_r, WarmupCosineScheduler,
)
from cv_split_patches import load_cv_split


def _edge_index_only(result):
    return result[0] if isinstance(result, tuple) else result


def train_one_fold_gnn_v5(X_all, Y_all, coords_all, train_idx, ho_idx,
                           k_spatial, k_expression, model_cfg, opt_cfg,
                           device, max_epochs, patience, verbose=True):
    X_tr, X_ho = X_all[train_idx], X_all[ho_idx]
    Y_tr, Y_ho = Y_all[train_idx], Y_all[ho_idx]
    coords_tr, coords_ho = coords_all[train_idx], coords_all[ho_idx]

    sp_ei_tr = _edge_index_only(build_spatial_knn_graph(coords_tr, k=k_spatial))
    sp_ei_ho = _edge_index_only(build_spatial_knn_graph(coords_ho, k=k_spatial))
    ex_ei_tr = _edge_index_only(build_expression_knn_graph_faiss(X_tr, k=k_expression))
    ex_ei_ho = _edge_index_only(build_expression_knn_graph_faiss(X_ho, k=k_expression))

    x_tr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr = torch.tensor(Y_tr, dtype=torch.float32, device=device)
    x_ho = torch.tensor(X_ho, dtype=torch.float32, device=device)
    y_ho = torch.tensor(Y_ho, dtype=torch.float32, device=device)
    sp_ei_tr, ex_ei_tr = sp_ei_tr.to(device), ex_ei_tr.to(device)
    sp_ei_ho, ex_ei_ho = sp_ei_ho.to(device), ex_ei_ho.to(device)

    model = DualGraphModel(in_dim=X_tr.shape[1], out_dim=Y_tr.shape[1], **model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
    sched = WarmupCosineScheduler(optimizer, opt_cfg["warmup"], max_epochs, opt_cfg["lr"])

    best_val, best_epoch, patience_ctr, best_state = float("inf"), 0, 0, None

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        out, z = model(x_tr, sp_ei_tr, ex_ei_tr)
        pred_loss = combined_loss(out, y_tr)
        recon_loss = model.vgae.recon_loss(z, sp_ei_tr)
        kl_loss = model.vgae.kl_loss()
        loss, beta = total_loss(pred_loss, recon_loss, kl_loss, epoch)
        loss.backward()
        optimizer.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            out_ho, _ = model(x_ho, sp_ei_ho, ex_ei_ho)
            val_loss = combined_loss(out_ho, y_ho).item()

        if val_loss < best_val - 1e-4:  # min_delta - doesn't reset patience
            best_val, best_epoch, patience_ctr = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if verbose and (epoch % 10 == 0 or epoch == max_epochs - 1):
            print(f"    epoch {epoch:4d}  train={loss.item():.4f}  val={val_loss:.4f}")

        if patience_ctr >= patience:
            print(f"    early stop epoch {epoch}  best_epoch={best_epoch}  best_val={best_val:.4f}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_pred, _ = model(x_ho, sp_ei_ho, ex_ei_ho)
    final_pred_np = final_pred.cpu().numpy()

    r2_vals = r2_score(Y_ho, final_pred_np, multioutput="raw_values")
    rmse_vals = np.sqrt(np.mean((final_pred_np - Y_ho) ** 2, axis=0))
    pear = mean_pearson_r(final_pred_np, Y_ho)

    metrics = dict(mean_r2=float(r2_vals.mean()), median_r2=float(np.median(r2_vals)),
                   r2_per_protein=r2_vals, mean_rmse=float(rmse_vals.mean()),
                   median_rmse=float(np.median(rmse_vals)), rmse_per_protein=rmse_vals,
                   pearson=pear)

    del model, optimizer, x_tr, y_tr, x_ho, y_ho, sp_ei_tr, sp_ei_ho, ex_ei_tr, ex_ei_ho
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_epoch + 1, metrics


def run_cv_gnn_v5(rna_train_path, pro_train_path, cv_split_path, out_path,
                   n_components=N_SVD_COMPONENTS, k_spatial=8, k_expression=8,
                   hidden=256, n_layers=2, dropout=0.3,
                   lr=3e-4, weight_decay=1e-3, warmup=10,
                   max_epochs=200, patience=20, device=None):
    os.makedirs(out_path, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f"Device: {device}")

    rna_train = ad.read_h5ad(rna_train_path)
    pro_train = ad.read_h5ad(pro_train_path)
    protein_names = list(pro_train.var_names)
    Y_all = to_dense(pro_train.X).astype(np.float32)
    coords_all = rna_train.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values.astype(np.float32)

    print("TruncatedSVD (fit on full train set)")
    X_svd, _, svd = reduce_rna_svd(rna_train, rna_train, n_components)
    X_marker, matched = extract_marker_genes(rna_train, protein_names, MARKER_GENE_ALIASES)
    X_all, _ = build_node_features(X_svd, X_marker, scaler=None)

    model_cfg = dict(hidden=hidden, n_layers=n_layers, dropout=dropout, vgae_hidden=256, vgae_latent=64)
    opt_cfg = dict(lr=lr, weight_decay=weight_decay, warmup=warmup)

    cv_splits = load_cv_split(cv_split_path)
    rows, per_protein_r2, per_protein_rmse = [], [], []

    for split in cv_splits:
        fold_i = split["fold"]
        train_idx, ho_idx = np.array(split["train"]), np.array(split["test"])
        print(f"\n[fold {fold_i}] train={len(train_idx):,}  holdout={len(ho_idx):,}")

        n_epochs, metrics = train_one_fold_gnn_v5(
            X_all, Y_all, coords_all, train_idx, ho_idx,
            k_spatial, k_expression, model_cfg, opt_cfg, device, max_epochs, patience,
        )
        rows.append(dict(fold=fold_i, n_epochs=n_epochs, mean_r2=metrics["mean_r2"],
                          median_r2=metrics["median_r2"], mean_rmse=metrics["mean_rmse"],
                          median_rmse=metrics["median_rmse"], pearson=metrics["pearson"]))
        per_protein_r2.append(metrics["r2_per_protein"])
        per_protein_rmse.append(metrics["rmse_per_protein"])

    summary = pd.DataFrame(rows)
    print("\nCV summary:")
    print(summary.to_string(index=False))
    print(f"\nMean CV R2 = {summary['mean_r2'].mean():.4f} +/- {summary['mean_r2'].std():.4f}")
    print(f"Mean CV Pearson = {summary['pearson'].mean():.4f}")

    summary.to_csv(os.path.join(out_path, "vgae_cv_summary.csv"), index=False)
    pd.DataFrame(np.stack(per_protein_r2), columns=protein_names).to_csv(
        os.path.join(out_path, "vgae_cv_r2_per_protein.csv"), index=False)
    pd.DataFrame(np.stack(per_protein_rmse), columns=protein_names).to_csv(
        os.path.join(out_path, "vgae_cv_rmse_per_protein.csv"), index=False)

    return summary


if __name__ == "__main__":
    run_cv_gnn_v5(
        rna_train_path="train_rna.h5ad",
        pro_train_path="train_pro.h5ad",
        cv_split_path="outputs/cv_splits_patches.json",
        out_path="outputs",
    )