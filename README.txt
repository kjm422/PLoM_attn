````plaintext
```plaintext
PLoM with Attention-Informed Manifold Learning: Option A and Option B

================================================================================
OPTION A: Attention Preconditioning + Standard Kernel
================================================================================

WORKFLOW:
1. Robust scaling in band space (median/IQR centering)
2. Attention preconditioning: S = diag((w^α)^{1/2}) applied to scaled data
3. Standard PCA to reduce dimensionality
4. Diffusion maps with isotropic (standard) kernel
5. Sampling via projected Itô SDE on manifold
6. Inverse transformations: inverse PCA → inverse preconditioning → inverse scaling

PARAMETERS:
- attn_enabled=True
- attn_weights: Attention weight vector w ∈ ℝ^ν
- attn_alpha: Exponent α ∈ [0.5, 1]
- pca_whiten=False  ← Distinguishes Option A
- scaling_method='RobustIQR'

KEY PRINCIPLE:
- Attention weights modify data in band space via preconditioning matrix
- Standard Euclidean distances in preconditioned space
- Preserves data while emphasizing/de-emphasizing features

================================================================================
OPTION B: Full-Rank PCA Whitening + Anisotropic Kernel
================================================================================

WORKFLOW:
1. Robust scaling in band space (median/IQR centering)
2. Full-rank PCA whitening (pca_dim = n_features, scale_evecs=True)
   - No attention preconditioning in band space
   - Whitened data has normalized covariance structure
3. Compute attention metric: M_η = Λ^{1/2} Φ^T diag(w^α) Φ Λ^{1/2}
4. Diffusion maps with anisotropic (metric-informed) kernel
   - Distances computed using metric: d² = (η_i - η_j)^T M_η (η_i - η_j)
5. Sampling via projected Itô SDE on manifold
6. Inverse transformations: inverse PCA whitening → inverse scaling
   - No inverse preconditioning (not applied!)

PARAMETERS:
- attn_enabled=True
- attn_weights: Attention weight vector w ∈ ℝ^ν
- attn_alpha: Exponent α ∈ [0.5, 1]
- pca_whiten=True  ← Distinguishes Option B
- pca_cum_energy=1-1e-5 (keep all dimensions)
- scaling_method='RobustIQR'

KEY PRINCIPLE:
- Attention weights inform the metric tensor (distance measurement) after PCA
- Preserves original geometry while incorporating attention in whitened space
- Metric-informed kernel provides flexible attention incorporation

================================================================================
COMPARISON TABLE
================================================================================

Feature              | Option A                    | Option B
---------------------|-----------------------------|---------------------------------
Trigger Parameters   | attn_enabled=True           | attn_enabled=True
                     | pca_whiten=False            | pca_whiten=True
---------------------|-----------------------------|---------------------------------
Band Space Modify    | Preconditioning S matrix    | None
---------------------|-----------------------------|---------------------------------
Scaling              | Robust (median/IQR)         | Robust (median/IQR)
---------------------|-----------------------------|---------------------------------
PCA Step             | Standard dimensionality     | Full-rank whitening
                     | reduction                   | (scale_evecs=True)
---------------------|-----------------------------|---------------------------------
DMAPS Kernel         | Isotropic (Gaussian)        | Anisotropic (metric-informed)
---------------------|-----------------------------|---------------------------------
Distance Metric      | Euclidean L2 norm           | d² = (η-μ)^T M_η (η-μ)
---------------------|-----------------------------|---------------------------------
Attention Effect     | Modifies data before PCA    | Modifies metric after PCA
---------------------|-----------------------------|---------------------------------
Geometry Preservation| Modified in band space      | Original preserved; metric shifts
---------------------|-----------------------------|---------------------------------
Inverse Precond      | Yes (to recover band space) | No (not applied in forward)
---------------------|-----------------------------|---------------------------------
CSV Loading Support  | Yes (band_attention_...)    | Yes (band_attention_...)
---------------------|-----------------------------|---------------------------------

================================================================================
IMPLEMENTATION DETAILS
================================================================================

PCA WHITENING (Option B Only):
- Uses _pca() with pca_dim = n_features and scale_evecs = True
- Performs: X_white = (X - mean) @ scaled_eigvecs
- Result: Normalized covariance structure (each direction has unit variance)
- Eigenvalues and eigenvectors preserved for metric computation

METRIC COMPUTATION (Option B Only):
- M_x = diag(w^α) in band space
- M_η = Λ^{1/2} Φ^T M_x Φ Λ^{1/2} in whitened space
- Cholesky factorization: M_η = R^T @ R
- Distances: d² = ||R(η_i - η_j)||²

ATTENTION WEIGHTS:
- Supply via attn_weights (numpy array) or attn_weights_file (CSV path)
- CSV format: single row or column of floats
- Typical range: 0.5 to 2.5 (weights < 1 de-emphasize, > 1 emphasize)
- attn_alpha ∈ [0.5, 1]: controls strength (higher = stronger effect)

ROBUST SCALING:
- Center: median instead of mean
- Scale: IQR (Q75 - Q25) instead of std
- More resistant to outliers than standard scaling

================================================================================
USAGE EXAMPLES
================================================================================

Option A (Attention Preconditioning):
------
from plom import initialize, run

plom_dict = initialize(
    training=data,
    attn_enabled=True,
    attn_weights=weights,
    attn_alpha=0.75,
    pca_whiten=False,  # Option A
    scaling_method='RobustIQR',
    dmaps=True,
    sampling=True,
    num_samples=100
)
run(plom_dict)


Option B (Anisotropic Kernel):
------
from plom import initialize, run

plom_dict = initialize(
    training=data,
    attn_enabled=True,
    attn_weights_file='band_attention_global.csv',
    attn_alpha=0.75,
    pca_whiten=True,  # Option B
    pca_cum_energy=1-1e-5,
    scaling_method='RobustIQR',
    dmaps=True,
    sampling=True,
    num_samples=100
)
run(plom_dict)

================================================================================
BRANCHES
================================================================================

- main: Original PLoM algorithm (no attention)
- dev_optionA: Option A implementation with attention preconditioning
- dev_optionB: Option B implementation with anisotropic kernel

All features (scaling, sampling, etc.) are available on all branches.
Attention features only activate when attn_enabled=True.

================================================================================
REFERENCES
================================================================================

- Soize, C. & Ghanem, R. (2016). Data-driven probability concentration and 
  sampling on manifold. Journal of Computational Physics, 321, 242-258.
- Diffusion Maps: Coifman, R. R. & Lafon, S. (2006). Diffusion maps. 
  Applied and Computational Harmonic Analysis, 21(1), 5-30.
```
````