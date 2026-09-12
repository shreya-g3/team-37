# Spatial Protein Prediction with a Graph Isomorphism Network

This project trains a PyTorch Graph Isomorphism Network (GIN) to predict protein abundance from spatial RNA data stored in `.h5ad` files.

## Files

- `src/data.py` - `.h5ad`/`.npz` loading, sparse feature selection, graph construction.
- `src/model.py` - self-contained GIN implementation in plain PyTorch.
- `src/train.py` - trains the model and writes artifacts/metrics.
- `src/predict.py` - loads the trained model and writes the prediction CSV.
- `run_pipeline.py` - train + predict in one command.
- `config/config.yaml` - default paths and hyperparameters.

## Default Dataset Paths

The default config points to the attached files:

- `C:\Users\shana\Downloads\New folder (19)\train_rna.h5ad`
- `C:\Users\shana\Downloads\New folder (19)\train_pro.h5ad`
- `C:\Users\shana\Downloads\New folder (19)\test_rna.h5ad`
- `C:\Users\shana\Downloads\preprocessing_stats.npz\preprocessing_stats.npz`

Edit `config/config.yaml` if you move the data.

## Run

```bash
python run_pipeline.py --config config/config.yaml
```

For a quicker smoke test:

```bash
python run_pipeline.py --config config/config.yaml --epochs 3 --max-genes 256
```

## Outputs

- `artifacts/best_gin.pt` - best validation checkpoint.
- `artifacts/metrics.json` - validation RMSE, MAE, and mean Pearson correlation.
- `predictions/gin_test_predictions.csv` - raw-scale protein predictions for `test_rna.h5ad`.

## Notes

The target values are trained as standardized `log1p(protein)` values. The trainer first checks `y_mean` and `y_std` from `preprocessing_stats.npz`; if those statistics do not match the attached `train_pro.h5ad`, it fits target normalization from `train_pro.X` and records the fitted values in `artifacts/metadata.json`. Prediction export applies the inverse transform and clips negative numerical noise to zero.
