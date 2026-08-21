import os
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.decomposition import TruncatedSVD

from model import DualBranchGNN, combined_loss
from preprocessing_final import preprocess_rna, preprocess_protein_train, inverse_transform_protein
from graph_construction import build_spatial_graph_from_coords, build_expression_graph, transform_svd
from train import (
    SEED, N_SVD_COMPONENTS, K_SPATIAL, K_EXPRESSION, BASE_LR, WARMUP_EPOCHS,
    LOSS_W, WarmupCosineScheduler, compute_qc_mask, compute_protein_missing_mask,
)


def final_refit_and_predict(
    rna_train_path="rna_train.h5ad",
    protein_train_path="protein_train.h5ad",
    rna_val_path="rna_val.h5ad",
    out_path="outputs",
    device=None,
):
    """
    Refit on 100% of the training data for the CV-determined epoch count,
    then predict on the external rna_val set and write both a
    raw-CODEX-scale and z-score-scale predictions file.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_path, exist_ok=True)

    cv_summary = pd.read_csv(os.path.join(out_path, "cv_summary.csv"))
    n_epochs_final = int(round(cv_summary["n_epochs"].median()))
    print(f"Final refit epoch count (median across CV folds): {n_epochs_final}")

    rna_train = preprocess_rna(rna_train_path)
    protein_train, protein_stats = preprocess_protein_train(protein_train_path)

    protein_z = protein_train.X.astype(np.float32)
    coords = np.asarray(rna_train.obsm["spatial"], dtype=np.float32)
    protein_names = list(protein_train.var_names)

    # same additive, loss-only masks as CV - comment out for no QC
    missing_full = compute_protein_missing_mask(protein_train_path)
    qc_mask = compute_qc_mask(rna_train_path)
    missing_full = missing_full | qc_mask[:, None]

    # fit SVD on ALL training spots - no fold held out at this stage
    svd = TruncatedSVD(n_components=N_SVD_COMPONENTS, random_state=SEED)
    latent_train = svd.fit_transform(rna_train.X).astype(np.float32)

    sp_ei, sp_ea = build_spatial_graph_from_coords(coords, k=K_SPATIAL)
    ex_ei, ex_ew = build_expression_graph(latent_train, k=K_EXPRESSION)

    x = torch.tensor(latent_train, device=device)
    y = torch.tensor(protein_z, device=device)
    missing = torch.tensor(missing_full, device=device)
    sp_ei, sp_ea = sp_ei.to(device), sp_ea.to(device)
    ex_ei, ex_ew = ex_ei.to(device), ex_ew.to(device)

    model = DualBranchGNN(in_dim=N_SVD_COMPONENTS, out_dim=protein_z.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-3)
    sched = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, n_epochs_final, BASE_LR)

    model.train()
    for epoch in range(n_epochs_final):
        optimizer.zero_grad()
        preds = model(x, sp_ei, sp_ea, ex_ei, ex_ew)
        loss = combined_loss(preds, y, w=LOSS_W, missing_mask=missing)
        loss.backward()
        optimizer.step()
        lr_now = sched.step()
        if epoch % 10 == 0 or epoch == n_epochs_final - 1:
            print(f"  [final refit] epoch {epoch:4d}  train_loss={loss.item():.4f}  lr={lr_now:.2e}")

    torch.save({"state_dict": model.state_dict(), "config": model._config},
               os.path.join(out_path, "final_model.pt"))
    joblib.dump(svd, os.path.join(out_path, "final_svd.joblib"))

    # predict on the external validation set
    rna_val = preprocess_rna(rna_val_path)
    latent_val = transform_svd(rna_val, svd).astype(np.float32)
    coords_val = np.asarray(rna_val.obsm["spatial"], dtype=np.float32)

    val_sp_ei, val_sp_ea = build_spatial_graph_from_coords(coords_val, k=K_SPATIAL)
    val_ex_ei, val_ex_ew = build_expression_graph(latent_val, k=K_EXPRESSION)

    x_val = torch.tensor(latent_val, device=device)
    model.eval()
    with torch.no_grad():
        preds_z = model(
            x_val, val_sp_ei.to(device), val_sp_ea.to(device),
            val_ex_ei.to(device), val_ex_ew.to(device),
        ).cpu().numpy()

    preds_raw = inverse_transform_protein(preds_z, protein_stats)

    spot_ids = rna_val.obs_names.to_numpy()
    preds_raw_df = pd.DataFrame(preds_raw, index=spot_ids, columns=protein_names)
    preds_z_df = pd.DataFrame(preds_z, index=spot_ids, columns=protein_names)

    preds_raw_path = os.path.join(out_path, "predictions_raw_codex.csv")
    preds_z_path = os.path.join(out_path, "predictions_zscore.csv")
    preds_raw_df.to_csv(preds_raw_path)
    preds_z_df.to_csv(preds_z_path)

    print(f"Wrote {preds_raw_path}  ({preds_raw_df.shape[0]:,} spots x {preds_raw_df.shape[1]} proteins)")
    print(f"Wrote {preds_z_path}")
    return preds_raw_df, preds_z_df


if __name__ == "__main__":
    final_refit_and_predict()