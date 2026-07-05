EDA Notes:
- RNA zeroes: 2828070514 (94.1%) -> high zero values due to VISIUM fails to capture many proteins per spot (inc in lims)
- RNA shape: 166186 spots 18085 genes
- RNA range: 0-95
- Protein shape: 166186 spots 44 proteins
- Protein value range: 0.0 to 76675.51745972
- Protein zeros: 13584 (0.2%)
- Protein range: 0 - 76675 (diff in ranges of RNA & protein so normalisation is required)

LIMITATIONS/CHALLENGES/PROBLEMS SLIDE:
1) RNA-protein decoupling: RNA levels don't reflect protein lvls because of post-translational regulation
   - ie. proteins can be modified/degraded/activated after being made in ways that RNA can't capture
   - low R^2 vals in models show this
   - therefore, RNA explains little of variation in protein expression
   - J. Joshi and S. Kate, "Subcellular Localization Constrains Protein Detectability and Reveals Systematic RNA-Protein Discordance Across Cancers," bioRxiv, Mar. 2026. [Online]. Available: https://www.biorxiv.org/content/10.64898/2026.03.30.713919v1
2) sparsity
   - 94.1% of RNA values are zeroes because VISIUM fails to capture many proteins per spot
   - dropout = harder for model to learn reliable RNA-protein relationships because input signal => noisy
   - https://biocellgen-public.svi.edu.au/mig_2019_scrnaseq-workshop/handling-sparsity.html 
3) single patient training
   - our models trained entirely on patient A
4) spatiality
   - using GNNs to solve
5) computational constraints
   - training on full dataset exceeds available memory -> RAM crashes
   - only use subsets of around 10,000 sometimes which only represents 3-6% of available data -> low R^2
   - solve by AWS, limited credit so must choose models to run and finetune them beforehand by testsing on subsets
