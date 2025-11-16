```plaintext
PLoM with Option A: Attention-Informed Manifold Learning

OPTION A WORKFLOW:
1. Robust scaling in band space (median/IQR centering and normalization)
2. Attention preconditioning in band space (apply weight matrix S = diag((w^α)^{1/2}))
3. PCA whitening to data space (full-rank eigendecomposition and normalization)
4. Standard kernel and diffusion maps basis computation
5. Reduced representation and sampling via projected Itô SDE
6. Inverse transformations to recover original band space

IMPLEMENTATION:
- attn_enabled: Set to True to activate Option A workflow
- attn_weights: Attention weight vector w ∈ ℝ^ν (set to uniform if not provided)
- attn_alpha: Exponent α ∈ [0.5, 1] controlling attention strength
- pca_whiten: Set to True to use full-rank PCA whitening
- scaling_method: Use 'RobustIQR' for robust scaling (median/IQR)

SUPPORTING FEATURES:
- Robust scaling using median and interquartile range
- Attention-based preconditioning matrix
- Full-rank PCA whitening with eigenvalue normalization
- Inverse transformations maintaining mathematical consistency
- C++ library: Yes
- Variance constraints: Optional via Option A weighting

BRANCHES:
- main: Original PLoM algorithm
- dev_optionA: Option A implementation (attention-informed workflow)
- dev_optionB: Alternative implementation approach
```