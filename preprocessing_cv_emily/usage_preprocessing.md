How to use preprocessing files:  
- preprocessing_final.py contains preprocessing() and inverse_transform_protein() modules
- run preprocessing() to produce preprocessed rna_train, rna_val, rna_test, pro_train and protein_normalisation_stats
- preprocessed files can be input directly into models (this ensures all models use the same preprocessed data)
- protein_normalisation_stats.pkl can be used in inverse_transform_proteins() module to inverse transform predicted protein expression values back to CODEX values
