Experimental pipeline

Raw data 
-> Preprocessing (highly variable gene selection, save reduced features for cross validation)
-> Cross-validation split (used for all models)
-> Cross-validated model (Ridge regression, Clustering + Random Forest, ElasticNet, XGBoost, MLP, etc.)
-> Cross-validation evaluation (Pearson R per protein, mean across folds)
-> Final model scripts – ready for submission for validation
