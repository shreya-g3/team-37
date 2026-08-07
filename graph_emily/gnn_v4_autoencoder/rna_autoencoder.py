"""
nonlinear RNA embedding, alternative to truncated SVD dimensionality reduction
truncatedSVD is linear projection
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

SEED = 0
DEFAULT_LATENT_DIM = 128
DEFAULT_HIDDEN_DIMS = (1024, 512)
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 4096
DEFAULT_LR = 1e-3
DEFAULT_PATIENCE = 10


class RNAAutoencoder(nn.Module):
    """
    simple symmetric MLP autoencoder
    Encoder: in_dim -> hidden_dims -> latent_dim
    Decoder mirrors the encoder back up to in_dim.
    LayerNorm + ReLU + Dropout at each hidden layer,
    same building blocks as the GNN's own layers to match
    """

    def __init__(self, in_dim, latent_dim=DEFAULT_LATENT_DIM, hidden_dims=DEFAULT_HIDDEN_DIMS, dropout=0.1):
        super().__init__()
        enc_dims = [in_dim] + list(hidden_dims) + [latent_dim]
        enc_layers = []
        for i in range(len(enc_dims) - 1):
            enc_layers.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            if i < len(enc_dims) - 2:  # no norm/activation on the latent output itself
                enc_layers += [nn.LayerNorm(enc_dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        self.encoder = nn.Sequential(*enc_layers)

        dec_dims = [latent_dim] + list(reversed(hidden_dims)) + [in_dim]
        dec_layers = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:  # no activation on the reconstruction output
                dec_layers += [nn.LayerNorm(dec_dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        self.decoder = nn.Sequential(*dec_layers)

        self._config = dict(in_dim=in_dim, latent_dim=latent_dim, hidden_dims=list(hidden_dims), dropout=dropout)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def _to_dense_batch(X_sparse, idx):
    """
    densify only the rows needed for this batch
    """
    batch = X_sparse[idx]
    return np.asarray(batch.todense() if hasattr(batch, "todense") else batch, dtype=np.float32)


def fit_autoencoder(X_train_sparse, latent_dim=DEFAULT_LATENT_DIM, hidden_dims=DEFAULT_HIDDEN_DIMS,
                     epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, lr=DEFAULT_LR,
                     patience=DEFAULT_PATIENCE, val_frac=0.1, device=None, verbose=True):
    """
    trains autoencoder on train bins only, using small internal holdout (val_frac) for early stop on reconstruction loss
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    n, in_dim = X_train_sparse.shape
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    model = RNAAutoencoder(in_dim=in_dim, latent_dim=latent_dim, hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss, patience_ctr, best_state = float("inf"), 0, None

    for epoch in range(epochs):
        model.train()
        shuffled = rng.permutation(train_idx)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(shuffled), batch_size):
            batch_idx = shuffled[start:start + batch_size]
            x_batch = torch.tensor(_to_dense_batch(X_train_sparse, batch_idx), device=device)
            optimizer.zero_grad()
            x_hat, _ = model(x_batch)
            loss = loss_fn(x_hat, x_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            x_val = torch.tensor(_to_dense_batch(X_train_sparse, val_idx), device=device)
            x_val_hat, _ = model(x_val)
            val_loss = loss_fn(x_val_hat, x_val).item()

        if val_loss < best_val_loss:
            best_val_loss, patience_ctr = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  [autoencoder] epoch {epoch:4d}  train_mse {epoch_loss / n_batches:.4f}  "
                  f"val_mse {val_loss:.4f}")

        if patience_ctr >= patience:
            print(f"  [autoencoder] early stop at epoch {epoch} (best val_mse {best_val_loss:.4f})")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model


def encode(model, X_sparse, device=None, batch_size=DEFAULT_BATCH_SIZE):
    """
    encodes X_sparse in mini-batches
    """
    device = device or next(model.parameters()).device
    model.eval()
    n = X_sparse.shape[0]
    out = np.zeros((n, model._config["latent_dim"]), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            idx = np.arange(start, min(start + batch_size, n))
            x_batch = torch.tensor(_to_dense_batch(X_sparse, idx), device=device)
            z = model.encode(x_batch)
            out[idx] = z.cpu().numpy()
    return out


def reduce_rna_autoencoder(rna_train, rna_val, latent_dim=DEFAULT_LATENT_DIM,
                            hidden_dims=DEFAULT_HIDDEN_DIMS, epochs=DEFAULT_EPOCHS,
                            batch_size=DEFAULT_BATCH_SIZE, lr=DEFAULT_LR,
                            patience=DEFAULT_PATIENCE, device=None, verbose=True):
    """
    fit on train only, applied to both
    """
    X_train = rna_train.X if sp.issparse(rna_train.X) else sp.csr_matrix(rna_train.X)
    X_val = rna_val.X if sp.issparse(rna_val.X) else sp.csr_matrix(rna_val.X)

    model = fit_autoencoder(X_train, latent_dim=latent_dim, hidden_dims=hidden_dims,
                             epochs=epochs, batch_size=batch_size, lr=lr,
                             patience=patience, device=device, verbose=verbose)

    X_train_emb = encode(model, X_train, device=device, batch_size=batch_size)
    X_val_emb = encode(model, X_val, device=device, batch_size=batch_size)
    return X_train_emb, X_val_emb, model


def save_autoencoder(model, path):
    torch.save({"state_dict": model.state_dict(), "config": model._config}, path)


def load_autoencoder(path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    model = RNAAutoencoder(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model