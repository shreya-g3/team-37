"""
Bayesian hyperparameter search (Optuna TPE sampler)
for gnn_v4 SAGE + autoencoder

one fold used for each trial, uses fold 1 by default because fold 0 has more excluded buffer

requires optuna: pip3 install optuna

Outputs under out_dir:
    bayes_search_trials.csv: every trial's params + score, sorted best-first
    best_config.json: winning trial's params
    optuna_study.pkl: full Optuna study object
"""

import os
import gc
import json
import pickle
import argparse
import numpy as np

import torch
import optuna
from optuna.samplers import TPESampler
from scipy import sparse
from scipy.stats import pearsonr
import anndata as ad

from gnn_v4 import ResidualGraphSAGE, build_knn_graph, combined_loss, WarmupCosineScheduler
from rna_autoencoder import fit_autoencoder, encode
from cv_split_patches import load_cv_split

FIXED_CONV_TYPE = "sage"  # gnn_v4 is the SAGE + autoencoder version
SEED = 0


def _mean_pearson_r(pred, true):
    rs = [pearsonr(pred[:, j], true[:, j])[0] for j in range(true.shape[1])]
    rs = [r for r in rs if not np.isnan(r)]
    return float(np.mean(rs)) if rs else 0.0


def _train_and_score(X, Y, edge_index, train_idx, val_idx, params, device, max_epochs, patience):
    """
    trains one fold with early stopping
    returns holdout mean Pearson r and epoch count
    rolls back to the best epoch's weights before scoring
    """
    x_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    edge_index_dev = edge_index.to(device)

    train_mask = torch.zeros(X.shape[0], dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    holdout_mask = torch.zeros(X.shape[0], dtype=torch.bool, device=device)
    holdout_mask[val_idx] = True

    model_keys = {"hidden", "n_layers", "dropout", "conv_type", "gat_heads", "use_jk"}
    model_kwargs = {k: v for k, v in params.items() if k in model_keys}
    model = ResidualGraphSAGE(in_dim=X.shape[1], out_dim=Y.shape[1], **model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    sched = WarmupCosineScheduler(optimizer, params["warmup"], max_epochs, params["lr"])
    loss_w = params.get("loss_w", 0.8)

    best_val_loss, best_epoch, patience_ctr, best_state = float("inf"), 0, 0, None

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x_t, edge_index_dev)
        loss = combined_loss(out[train_mask], y_t[train_mask], w=loss_w)
        loss.backward()
        optimizer.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            out = model(x_t, edge_index_dev)
            val_loss = combined_loss(out[holdout_mask], y_t[holdout_mask], w=loss_w).item()

        if val_loss < best_val_loss:
            best_val_loss, best_epoch, patience_ctr = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        Y_pred_all = model(x_t, edge_index_dev).cpu().numpy()

    r = _mean_pearson_r(Y_pred_all[val_idx], Y[val_idx])

    del model, optimizer, sched, x_t, y_t, edge_index_dev, train_mask, holdout_mask, best_state, out, Y_pred_all
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return r, best_epoch + 1


def run_bayes_search(rna_path, protein_path, cv_split_path, out_dir, n_trials=30, fold=1,
                      latent_dim=128, search_latent_dim=True, ae_hidden_dims=(1024, 512),
                      ae_epochs=100, k=8, search_k=True, max_epochs=150, patience=15,
                      device=None, seed=SEED):
    os.makedirs(out_dir, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Loading preprocessed data...")
    rna = ad.read_h5ad(rna_path)
    pro = ad.read_h5ad(protein_path)
    coords = rna.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values.astype(np.float32)

    X_raw = rna.X if sparse.issparse(rna.X) else sparse.csr_matrix(rna.X)
    X_raw = X_raw.astype(np.float32).copy()
    Y = pro.X.toarray() if sparse.issparse(pro.X) else np.asarray(pro.X).astype(np.float32)
    del rna, pro
    gc.collect()

    all_splits = load_cv_split(split_path=cv_split_path)
    split = all_splits[fold]
    train_idx, val_idx = np.array(split["train"]), np.array(split["test"])
    print(f"Using fold {fold} only ({len(train_idx):,} train / {len(val_idx):,} val bins) "
          f"for the search - NOT the full 5-fold CV, see gnn_v4_cv.py for that")

    def objective(trial):
        params = dict(
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            loss_w=trial.suggest_float("loss_w", 0.2, 1.0),
            hidden=trial.suggest_categorical("hidden", [128, 256, 384]),
            n_layers=trial.suggest_int("n_layers", 2, 5),
            warmup=trial.suggest_int("warmup", 0, 20),
            use_jk=trial.suggest_categorical("use_jk", [False, True]),
            conv_type=FIXED_CONV_TYPE,
        )

        trial_latent_dim = trial.suggest_categorical("latent_dim", [64, 128, 256]) if search_latent_dim else latent_dim
        trial_k = trial.suggest_int("k", 4, 12) if search_k else k

        ae = fit_autoencoder(X_raw[train_idx], latent_dim=trial_latent_dim, hidden_dims=ae_hidden_dims,
                              epochs=ae_epochs, device=device, verbose=False)
        X_feat = encode(ae, X_raw, device=device)
        del ae
        gc.collect()

        edge_index = build_knn_graph(coords, k=trial_k)
        score, n_epochs = _train_and_score(X_feat, Y, edge_index, train_idx, val_idx,
                                            params, device, max_epochs, patience)
        del X_feat
        gc.collect()

        trial.set_user_attr("n_epochs", n_epochs)
        trial.set_user_attr("latent_dim", trial_latent_dim)
        trial.set_user_attr("k", trial_k)

        print(f"[trial {trial.number}] pearsonr={score:.4f}  latent_dim={trial_latent_dim}  "
              f"k={trial_k}  epochs={n_epochs}")
        return score

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    trials_df = study.trials_dataframe().sort_values("value", ascending=False)
    trials_df.to_csv(os.path.join(out_dir, "bayes_search_trials.csv"), index=False)

    with open(os.path.join(out_dir, "optuna_study.pkl"), "wb") as f:
        pickle.dump(study, f)

    best_config = dict(study.best_trial.params)
    with open(os.path.join(out_dir, "best_config.json"), "w") as f:
        json.dump(best_config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Best trial: #{study.best_trial.number}  pearsonr = {study.best_value:.4f}  (fold {fold} only)")
    print(f"Best params: {json.dumps(best_config, indent=2)}")
    print(f"Saved to {out_dir}/bayes_search_trials.csv and {out_dir}/best_config.json")
    print(f"This is a single-fold estimate - now run gnn_v4_cv.py's full 5-fold CV "
          f"with these params before trusting this number.")
    print(f"{'='*60}")

    return study, trials_df, best_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna_path", required=True)
    parser.add_argument("--protein_path", required=True)
    parser.add_argument("--cv_split_path", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--fold", type=int, default=1,
                         help="which single fold to evaluate each trial on (default 1, not 0 - "
                              "fold 0 is the smallest fold)")
    parser.add_argument("--latent_dim", type=int, default=128, help="used when --no_search_latent_dim is set")
    parser.add_argument("--no_search_latent_dim", dest="search_latent_dim", action="store_false")
    parser.add_argument("--ae_hidden_dims", type=int, nargs="+", default=[1024, 512])
    parser.add_argument("--ae_epochs", type=int, default=100)
    parser.add_argument("--k", type=int, default=8, help="used when --no_search_k is set")
    parser.add_argument("--no_search_k", dest="search_k", action="store_false")
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.set_defaults(search_latent_dim=True, search_k=True)

    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None

    run_bayes_search(
        rna_path=args.rna_path,
        protein_path=args.protein_path,
        cv_split_path=args.cv_split_path,
        out_dir=args.out_dir,
        n_trials=args.n_trials,
        fold=args.fold,
        latent_dim=args.latent_dim,
        search_latent_dim=args.search_latent_dim,
        ae_hidden_dims=tuple(args.ae_hidden_dims),
        ae_epochs=args.ae_epochs,
        k=args.k,
        search_k=args.search_k,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()