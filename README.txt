```plaintext
PLoM with Option A: Attention-Informed Manifold Learning

OPTION A WORKFLOW:
1. Robust scaling in band space (median/IQR centering and normalization)
2. Attention preconditioning in band space (apply weight matrix S = diag((w^α)^{1/2}))
3. PCA whitening to data space (full-rank PCA with eigenvector scaling)
4. Standard kernel and diffusion maps basis computation
5. Reduced representation and sampling via projected Itô SDE
6. Inverse transformations to recover original band space

IMPLEMENTATION:
- attn_enabled: Set to True to activate Option A workflow
- attn_weights: Attention weight vector w ∈ ℝ^ν (set to uniform if not provided)
- attn_alpha: Exponent α ∈ [0.5, 1] controlling attention strength
- pca_whiten: Set to True to use full-rank PCA (pca_dim = n_features with eigenvector scaling)
- scaling_method: Use 'RobustIQR' for robust scaling (median/IQR)

HOW PCA WHITENING WORKS:
- Uses standard _pca() function with pca_dim=n_features to keep all dimensions
- Applies eigenvector scaling (scale_evecs=True) to normalize by eigenvalues
- Result: Full-rank PCA with whitened covariance structure
- Equivalent to: X_pca = (X - mean) @ scaled_eigvecs where scaled_eigvecs = eigvecs / sqrt(eigvals)

SUPPORTING FEATURES:
- Robust scaling using median and interquartile range
- Attention-based preconditioning matrix
- Full-rank PCA whitening using standard _pca() function
- Inverse transformations maintaining mathematical consistency
- C++ library: Yes
- Variance constraints: Optional via Option A weighting

BRANCHES:
- main: Original PLoM algorithm
- dev_optionA: Option A implementation (attention-informed workflow)
- dev_optionB: Alternative implementation approach
```