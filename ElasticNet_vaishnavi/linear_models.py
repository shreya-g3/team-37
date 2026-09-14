"""
Lasso and ElasticNet baselines for RNA to protein prediction.

Why MultiTask* variants?
  - MultiTaskLasso / MultiTaskElasticNet jointly regularise across all 44
    protein targets, which helps when many proteins share gene regulators.
  - They are more efficient than fitting 44 independent models.
  - The group L1 norm encourages sparsity in the same gene subset for all
    proteins simultaneously.

Memory strategy for train_rna (166K x 18K):
  - If the full matrix fits in RAM (~50 GB for float32), use fit_full().
  - If RAM is constrained, use fit_sgd() which trains with SGDRegressor
    (elastic net penalty) via incremental partial_fit on mini-batches.

Testing note:
  - The test split (test_rna.h5ad, 84K spots) has NO protein labels.
  - Therefore Lasso/ElasticNet are *trained* on train split, *evaluated*
    on valid split (8786 spots with valid_uniform_range.csv as ground truth),
    and *predictions* are generated for the test split.
  - Pearson correlation on the validation set is the reported metric.
"""

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import (
    MultiTaskLasso,
    MultiTaskElasticNet,
    SGDRegressor,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import joblib

from config import (
    LASSO_ALPHA, ELASTIC_ALPHA, ELASTIC_L1_RATIO,
    LINEAR_MAX_ITER, LINEAR_TOL, LINEAR_N_JOBS,
    PROTEIN_COLS, SEED, OUTPUT_DIR,
)
import os

# Import shared evaluation helper (avoids duplication)
from evaluate import pearson_per_protein


# --- Approach 1: MultiTask (fits when full matrix fits in RAM) ---

class MultiTaskLinear:
    """
    Wraps MultiTaskLasso and MultiTaskElasticNet.

    Both models accept X of shape (n_samples, n_features) and
    Y of shape (n_samples, n_targets) directly.
    """

    def __init__(self, model: str = "elasticnet"):
        assert model in ("lasso", "elasticnet")
        self.model_name = model
        self.scaler = StandardScaler(with_mean=False)  # sparse-safe

        if model == "lasso":
            self.model = MultiTaskLasso(
                alpha=LASSO_ALPHA,
                max_iter=LINEAR_MAX_ITER,
                tol=LINEAR_TOL,
                selection="random",  # faster than cyclic
                random_state=SEED,
            )
        else:
            self.model = MultiTaskElasticNet(
                alpha=ELASTIC_ALPHA,
                l1_ratio=ELASTIC_L1_RATIO,
                max_iter=LINEAR_MAX_ITER,
                tol=LINEAR_TOL,
                selection="random",
                random_state=SEED,
            )

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        print(f"[{self.model_name}] Scaling features ...", flush=True)
        X_s = self.scaler.fit_transform(X_train)
        print(f"[{self.model_name}] Fitting model on {X_train.shape} ...", flush=True)
        t0 = time.time()
        self.model.fit(X_s, Y_train)
        print(f"[{self.model_name}] Done in {time.time()-t0:.1f}s "
              f"(n_iter={self.model.n_iter_})", flush=True)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X)).astype(np.float32)

    def evaluate(self, X_valid: np.ndarray, Y_valid: np.ndarray) -> pd.DataFrame:
        Y_pred = self.predict(X_valid)
        df = pearson_per_protein(Y_valid, Y_pred)
        mean_r = df["pearson_r"].mean()
        print(f"[{self.model_name}] Validation mean Pearson r = {mean_r:.4f}")
        return df, Y_pred

    def save(self, path: str | None = None):
        path = path or os.path.join(OUTPUT_DIR, f"{self.model_name}_model.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)
        print(f"[{self.model_name}] Saved to {path}")

    @classmethod
    def load(cls, model: str, path: str):
        obj = cls.__new__(cls)
        obj.model_name = model
        ckpt = joblib.load(path)
        obj.model  = ckpt["model"]
        obj.scaler = ckpt["scaler"]
        return obj


# --- Approach 2: SGD-based (memory-efficient, partial_fit mini-batches) ---

class SGDLinear:
    """
    Per-protein SGDRegressor with elastic-net penalty, supporting partial_fit.
    Use this when the full training matrix does NOT fit in RAM.

    Note: MultiOutputRegressor parallelises across proteins with n_jobs.
    """

    def __init__(self, l1_ratio: float = ELASTIC_L1_RATIO, alpha: float = ELASTIC_ALPHA):
        self.l1_ratio   = l1_ratio
        self.alpha      = alpha
        self.scaler     = StandardScaler(with_mean=False)
        self.model_name = "sgd_elasticnet"

        single = SGDRegressor(
            loss="squared_error",
            penalty="elasticnet",
            l1_ratio=l1_ratio,
            alpha=alpha,
            max_iter=1,             # we drive epochs manually via partial_fit
            tol=None,
            random_state=SEED,
        )
        self.model = MultiOutputRegressor(single, n_jobs=LINEAR_N_JOBS)
        self._scaler_fitted = False

    # -- Phase 1: fit scaler on first pass, collect stats --
    def fit_scaler_on_chunks(self, chunk_iter):
        """chunk_iter: yields (X_chunk, (start, end)) pairs."""
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler(with_mean=False)
        for chunk, _ in chunk_iter:
            self.scaler.partial_fit(chunk)
        self._scaler_fitted = True
        print("[sgd] Scaler fitted.", flush=True)

    # -- Phase 2: one or more epochs of partial_fit --
    def partial_fit_epoch(self, X_chunk: np.ndarray, Y_chunk: np.ndarray):
        if not self._scaler_fitted:
            raise RuntimeError("Call fit_scaler_on_chunks first.")
        X_s = self.scaler.transform(X_chunk)
        self.model.partial_fit(X_s, Y_chunk)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [est.predict(self.scaler.transform(X)) for est in self.model.estimators_]
        ).astype(np.float32)

    def evaluate(self, X_valid: np.ndarray, Y_valid: np.ndarray) -> pd.DataFrame:
        Y_pred = self.predict(X_valid)
        df = pearson_per_protein(Y_valid, Y_pred)
        mean_r = df["pearson_r"].mean()
        print(f"[{self.model_name}] Validation mean Pearson r = {mean_r:.4f}")
        return df, Y_pred

    def save(self, path: str | None = None):
        path = path or os.path.join(OUTPUT_DIR, f"{self.model_name}_model.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)
        print(f"[{self.model_name}] Saved to {path}")


# --- Convenience wrappers ---

def train_lasso(X_train, Y_train, X_valid, Y_valid):
    """Full-matrix Lasso training + validation."""
    m = MultiTaskLinear("lasso")
    m.fit(X_train, Y_train)
    df, Y_pred = m.evaluate(X_valid, Y_valid)
    m.save()
    return m, df, Y_pred


def train_elasticnet(X_train, Y_train, X_valid, Y_valid):
    """Full-matrix ElasticNet training + validation."""
    m = MultiTaskLinear("elasticnet")
    m.fit(X_train, Y_train)
    df, Y_pred = m.evaluate(X_valid, Y_valid)
    m.save()
    return m, df, Y_pred


def train_sgd_elasticnet_chunked(adata_rna_backed, Y_train,
                                  X_valid, Y_valid,
                                  n_epochs: int = 5,
                                  chunk_size: int = 10_000):
    """
    Memory-efficient ElasticNet using SGD + partial_fit.
    Use when train_rna does not fit in RAM.

    adata_rna_backed: backed AnnData opened with backed='r'
    Y_train: full protein matrix (166186, 44) - fits in RAM
    """
    from data_utils import iter_train_rna_chunks
    m = SGDLinear()

    print("[sgd] Phase 1: fit scaler ...", flush=True)
    m.fit_scaler_on_chunks(iter_train_rna_chunks(adata_rna_backed, chunk_size))

    print(f"[sgd] Phase 2: {n_epochs} epochs of partial_fit ...", flush=True)
    for epoch in range(n_epochs):
        print(f"  Epoch {epoch+1}/{n_epochs}", flush=True)
        for chunk, (s, e) in iter_train_rna_chunks(adata_rna_backed, chunk_size):
            m.partial_fit_epoch(chunk, Y_train[s:e])

    df, Y_pred = m.evaluate(X_valid, Y_valid)
    m.save()
    return m, df, Y_pred
