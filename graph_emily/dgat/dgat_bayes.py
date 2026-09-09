"""
Bayesian hyperparameter optimisation for DGAT (Optuna, TPE sampler).

Searches architecture (hidden, latent_dim, n_layers, gat_heads,
feat_attn_heads, decoder_hidden, branch_hidden), training (batch_size,
lr_encoder, lr_decoder, encoder_wd), and loss-weight hyperparameters
against a single held-out CV fold's holdout cross_protein RMSE
"""

import argparse
import gc
import time
import numpy as np
import anndata as ad
from scipy import sparse
from sklearn.decomposition import TruncatedSVD, PCA

import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from dgat import (
    DGAT, build_spatial_graph, build_feature_graph, union_edge_index, build_pyg_data,
    build_optimizer, StepDecayScheduler, train_epoch_batched, eval_cross_protein_rmse_batched,
    DEFAULT_NUM_NEIGHBORS,
)
from cv_split_patches import load_cv_split

SEED = 0
N_SVD_COMPONENTS = 128
PROTEIN_PCA_VARIANCE = 0.85
K_SPATIAL = 6
K_FEATURE = 10


def prepare_fold_data(rna_path, protein_path, cv_split_path, opt_fold,
                      n_svd_components=N_SVD_COMPONENTS, protein_pca_variance=PROTEIN_PCA_VARIANCE,
                      k_spatial=K_SPATIAL, k_feature=K_FEATURE):
    """One-time setup: fit SVD/PCA on the chosen fold's train split, build graphs."""
    rna = ad.read_h5ad(rna_path)
    pro = ad.read_h5ad(protein_path)

    coords = rna.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values.astype(np.float32)
    X_raw = rna.X if sparse.issparse(rna.X) else sparse.csr_matrix(rna.X)
    X_raw = X_raw.astype(np.float32).copy()
    Y = pro.X.toarray() if sparse.issparse(pro.X) else np.asarray(pro.X).astype(np.float32)
    del rna, pro

    splits = load_cv_split(split_path=cv_split_path)
    split = next(s for s in splits if s["fold"] == opt_fold)
    train_idx, holdout_idx = np.array(split["train"]), np.array(split["test"])

    svd = TruncatedSVD(n_components=n_svd_components, random_state=SEED)
    svd.fit(X_raw[train_idx])
    X_svd = svd.transform(X_raw).astype(np.float32)

    pca = PCA(n_components=protein_pca_variance, random_state=SEED)
    pca.fit(Y[train_idx])
    Y_pca = pca.transform(Y).astype(np.float32)

    spatial_edge = build_spatial_graph(coords, k=k_spatial)
    mrna_feat_edge = build_feature_graph(X_svd, k=k_feature)
    protein_feat_edge = build_feature_graph(Y_pca, k=k_feature)
    edge_mrna = union_edge_index(mrna_feat_edge, spatial_edge)
    edge_protein = union_edge_index(protein_feat_edge, spatial_edge)

    print(f"opt_fold={opt_fold}: {len(train_idx):,} train / {len(holdout_idx):,} holdout bins")
    print(f"edge_mrna: {edge_mrna.shape[1]:,} edges  edge_protein: {edge_protein.shape[1]:,} edges")

    return dict(X_svd=X_svd, Y=Y, edge_mrna=edge_mrna, edge_protein=edge_protein,
               train_idx=train_idx, holdout_idx=holdout_idx,
               rna_dim=n_svd_components, protein_dim=Y.shape[1])


def make_objective(fold_data, device, max_epochs, patience, report_every=5,
                   num_workers=0, subsample_frac=1.0):
    """
    subsample_frac < 1.0 shrinks the seed-node set used per epoch during the search
    """
    data_mrna = build_pyg_data(fold_data["X_svd"], fold_data["edge_mrna"])
    data_protein = build_pyg_data(fold_data["Y"], fold_data["edge_protein"])

    train_idx = fold_data["train_idx"]
    holdout_idx = fold_data["holdout_idx"]
    if subsample_frac < 1.0:
        rng = np.random.default_rng(SEED)
        train_idx = rng.choice(train_idx, size=int(len(train_idx) * subsample_frac), replace=False)
        holdout_idx = rng.choice(holdout_idx, size=int(len(holdout_idx) * subsample_frac), replace=False)
        print(f"subsample_frac={subsample_frac}: using {len(train_idx):,} train / "
             f"{len(holdout_idx):,} holdout seed nodes per epoch (search only)")

    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long)
    holdout_idx_t = torch.as_tensor(holdout_idx, dtype=torch.long)

    def objective(trial):
        hidden = trial.suggest_categorical("hidden", [64, 128, 256])
        latent_dim = trial.suggest_categorical("latent_dim", [128, 256, 512])
        n_layers = trial.suggest_int("n_layers", 2, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        gat_heads = trial.suggest_categorical("gat_heads", [1, 2, 4])
        feat_attn_heads = trial.suggest_categorical("feat_attn_heads", [2, 4, 8])
        decoder_hidden = trial.suggest_categorical("decoder_hidden", [64, 128, 256])
        branch_hidden = trial.suggest_categorical("branch_hidden", [16, 32, 64])
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
        lr_encoder = trial.suggest_float("lr_encoder", 1e-4, 1e-3, log=True)
        lr_decoder = trial.suggest_float("lr_decoder", 5e-5, 5e-4, log=True)
        encoder_wd = trial.suggest_float("encoder_wd", 1e-6, 1e-4, log=True)
        w_align = trial.suggest_float("w_align", 1.0, 10.0)
        w_recon_mrna = trial.suggest_float("w_recon_mrna", 0.1, 3.0)
        w_recon_protein = trial.suggest_float("w_recon_protein", 0.1, 3.0)
        w_cross_protein = trial.suggest_float("w_cross_protein", 1.0, 10.0)
        w_cross_mrna = trial.suggest_float("w_cross_mrna", 1.0, 10.0)

        model_cfg = dict(
            rna_dim=fold_data["rna_dim"], protein_dim=fold_data["protein_dim"],
            hidden=hidden, latent_dim=latent_dim, n_layers=n_layers, dropout=dropout,
            gat_heads=gat_heads, feat_attn_heads=feat_attn_heads,
            decoder_hidden=decoder_hidden, branch_hidden=branch_hidden,
        )
        loss_weights = dict(align=w_align, recon_mrna=w_recon_mrna, recon_protein=w_recon_protein,
                            cross_protein=w_cross_protein, cross_mrna=w_cross_mrna)

        torch.manual_seed(SEED)
        model = DGAT(**model_cfg).to(device)
        optimizer = build_optimizer(model, encoder_lr=lr_encoder, decoder_lr=lr_decoder, encoder_wd=encoder_wd)
        sched = StepDecayScheduler(optimizer)

        best_val = float("inf")
        patience_ctr = 0
        trial_start = time.time()
        print(f"[trial {trial.number}] hidden={hidden} latent_dim={latent_dim} n_layers={n_layers} "
             f"batch_size={batch_size} -- starting", flush=True)

        try:
            for epoch in range(max_epochs):
                epoch_start = time.time()
                train_epoch_batched(model, data_mrna, data_protein, train_idx_t, optimizer,
                                    loss_weights, device, num_neighbors=DEFAULT_NUM_NEIGHBORS,
                                    batch_size=batch_size, grad_clip=1.0, num_workers=num_workers)
                sched.step()

                if epoch % report_every == 0 or epoch == max_epochs - 1:
                    val_loss = eval_cross_protein_rmse_batched(
                        model, data_mrna, data_protein, holdout_idx_t, device,
                        num_neighbors=DEFAULT_NUM_NEIGHBORS, batch_size=batch_size,
                        num_workers=num_workers,
                    )
                    epoch_time = time.time() - epoch_start
                    print(f"[trial {trial.number}] epoch {epoch:3d}  "
                         f"holdout_cross_protein_rmse {val_loss:.4f}  "
                         f"({epoch_time:.1f}s/epoch, {time.time() - trial_start:.0f}s elapsed)", flush=True)

                    if not np.isfinite(val_loss):
                        # diverged -- NaN/Inf loss, treat as pruned rather than
                        # crashing the whole study
                        print(f"[trial {trial.number}] diverged (NaN/Inf) -- pruning", flush=True)
                        raise optuna.TrialPruned()

                    if val_loss < best_val:
                        best_val = val_loss
                        patience_ctr = 0
                    else:
                        patience_ctr += report_every

                    trial.report(val_loss, step=epoch)
                    if trial.should_prune():
                        print(f"[trial {trial.number}] pruned at epoch {epoch}", flush=True)
                        raise optuna.TrialPruned()

                    if patience_ctr >= patience:
                        print(f"[trial {trial.number}] early stop at epoch {epoch}", flush=True)
                        break

            print(f"[trial {trial.number}] done -- best_val={best_val:.4f} "
                 f"({time.time() - trial_start:.0f}s total)", flush=True)
            return best_val

        except torch.cuda.OutOfMemoryError:
            # trial's sampled config (hidden/latent_dim/batch_size) didn't fit in memory - report as pruned
            print(f"[trial {trial.number}] CUDA OOM pruning this config "
                 f"(hidden={hidden}, latent_dim={latent_dim}, batch_size={batch_size})", flush=True)
            raise optuna.TrialPruned()

        finally:
            # cleanup between trials
            del model, optimizer, sched
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_path", default="data/rna_hvg.h5ad")
    parser.add_argument("--protein_path", default="data/protein_data_v2.h5ad")
    parser.add_argument("--cv_split_path", default="data/cv_splits_patches.json")
    parser.add_argument("--out_path", default="results/dgat_bayesopt_results.csv")
    parser.add_argument("--study_db", default="results/dgat_bayesopt.db",
                        help="sqlite path -- lets the study resume if interrupted or extended")
    parser.add_argument("--opt_fold", type=int, default=1)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--n_startup_trials", type=int, default=5,
                        help="trials run before the pruner starts acting (random exploration phase)")
    parser.add_argument("--n_warmup_steps", type=int, default=20,
                        help="epochs each trial gets before it becomes eligible for pruning")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="parallel CPU processes for graph neighbor sampling; "
                            "try 4 if training is sampling-bound (check nvidia-smi)")
    parser.add_argument("--subsample_frac", type=float, default=1.0,
                        help="fraction of train/holdout seed nodes used per epoch during "
                            "search (search-only speedup; re-validate winner on full data)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device == "cuda" and device.type == "cpu":
        print("cuda requested but not available -- falling back to cpu")

    fold_data = prepare_fold_data(args.rna_path, args.protein_path, args.cv_split_path, args.opt_fold)
    objective = make_objective(fold_data, device, args.max_epochs, args.patience,
                               num_workers=args.num_workers, subsample_frac=args.subsample_frac)

    sampler = TPESampler(seed=SEED)
    pruner = MedianPruner(n_startup_trials=args.n_startup_trials, n_warmup_steps=args.n_warmup_steps)
    study = optuna.create_study(
        study_name="dgat_hpo",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{args.study_db}",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=args.n_trials)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\n{len(completed)} completed, {len(pruned)} pruned, {len(study.trials)} total trials")

    print("\nBest trial:")
    print(f"  value (holdout cross_protein RMSE): {study.best_trial.value:.4f}")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    trials_df = study.trials_dataframe()
    trials_df.to_csv(args.out_path, index=False)
    print(f"\nAll trial results saved to {args.out_path}")
    print(f"Study database saved to {args.study_db} (resumable via load_if_exists)")


if __name__ == "__main__":
    main()