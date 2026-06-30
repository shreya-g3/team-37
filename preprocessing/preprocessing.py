import anndata as ad
import scanpy as sc
import numpy as np

def preprocessing(
   rna_path,           # rna input path "train_rna.h5ad"
   protein_path,       # protein input path "train_pro.h5ad"
   hvg_path,           # highly variable gene list input path "outputs/highly_variable_genes.txt"
   n_hvg,              # number of highly variable genes (2000)
   output_path         # output directory "outputs"
   ):
   """
   preprocessing of rna and protein data
   1. load data
   2. normalise and log transform
   3. select highly variable genes

   returns rna_hvg: normalised RNA, HVG subset
           rna_data: normalised RNA, all genes
           protein_data: CLR-normalised proteins
   """
   # 1. Load data
   rna_data = ad.read_h5ad(rna_path).copy()
   protein_data = ad.read_h5ad(protein_path).copy()

   # 2. Normalise

   # RNA: normalise and log transform

   sc.pp.normalize_total(rna_data, target_sum=1e4)
   sc.pp.log1p(rna_data)

   # Proteins: CLR normalisation

   X_pro = (protein_data.X.toarray()
            if hasattr(protein_data.X, "toarray")
            else protein_data.X.copy())

   X_pro = np.nan_to_num(X_pro.astype(float))
   geom_mean = np.expm1(np.mean(np.log1p(X_pro + 1e-8), axis=1, keepdims=True))
   protein_data.X = np.log1p(X_pro / (geom_mean + 1e-8))

   # 3. Highly variable gene selection - RNA only

   if hvg_path is not None:
       # load fixed list of hvgs
       with open(hvg_path, "r") as f:
           hvgs = [line.strip() for line in f if line.strip()]
       hvgs = [g for g in hvgs if g in rna_data.var_names]
   else:
       sc.pp.highly_variable_genes(
           rna_data,
           n_top_genes=n_hvg,
           flavor="cell_ranger",
           subset=False)
       hvgs = list(rna_data.var_names[rna_data.var.highly_variable])
       with open(f"{output_path}/highly_variable_genes.txt", "w") as hvg_text_file:
           hvg_text_file.write("\n".join(hvgs))

   rna_hvg = rna_data[:, hvgs].copy()

   # 4. Save output

   rna_hvg.write_h5ad(f"{output_path}/rna_hvg.h5ad")
   rna_data.write_h5ad(f"{output_path}/rna_data.h5ad")
   protein_data.write_h5ad(f"{output_path}/protein_data.h5ad")

   return rna_hvg, rna_data, protein_data