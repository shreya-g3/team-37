GitHub uploads and changes 23rd July 

- Added updated preprocessing script - use preprocessingV2 (now has z-normalisation instead of CLR normalisation, more appropriate for CODEX data)
- Run preprocessingV2 on your data and then use outputs “protein_data_v2.h5ad”


- Added “cv_split_patches.py” which uses larger areas and buffer zones of 60 bin radius in order to prevent leakage (buffer zone was picked based on autocorrelation in EDA)
- Output “cv_split_patches.json” and “cv_split_patches_info.txt” - use this cv split instead of previous one   


- Added EDA to justify my preprocessing and cv_split choices
- Outputs with diagrams that might be useful for write-up   
   
