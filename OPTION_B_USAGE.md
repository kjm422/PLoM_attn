# Option B: Anisotropic Kernel with Attention Weights

## Quick Start

Option B uses **attention-informed anisotropic kernel** instead of attention preconditioning. This means:
- Attention weights inform the *metric* used for distance computation in DMAPS
- No preconditioning applied in band space
- Better preserves original data geometry while incorporating attention

```python
from plom import initialize, run

# Create PLoM solution with Option B
plom_dict = initialize(
    training=your_data,
    
    # Enable attention
    attn_enabled=True,
    attn_weights_file='band_attention_global.csv',  # or numpy array
    attn_alpha=0.75,  # Control attention strength
    
    # Enable whitening (differentiates Option B from Option A)
    pca_whiten=True,
    
    # Rest of configuration
    dmaps=True,
    dmaps_epsilon='auto',
    sampling=True,
    num_samples=100,
    verbose=True
)

# Run analysis
run(plom_dict)

# Access results
print(f"RMSE: {plom_dict['data']['rmse']:.6e}")
print(f"Metric matrix:\n{plom_dict['dmaps']['metric_matrix']}")
```

## Option A vs Option B Comparison

| Aspect | Option A | Option B |
|--------|----------|----------|
| **Parameters** | `attn_enabled=True, pca_whiten=False` | `attn_enabled=True, pca_whiten=True` |
| **Attention Application** | Preconditioning matrix in band space | Metric tensor in whitened space |
| **Preconditioning Matrix** | $S = \text{diag}((w^α)^{1/2})$ in band | None (handled by metric) |
| **Workflow** | Scaling → Preconditioning → PCA → DMAPS | Scaling → PCA (whitening) → DMAPS |
| **Kernel** | Isotropic (standard Gaussian) | Anisotropic (attention-informed) |
| **Metric** | Standard L2 distance | $M_η = Λ^{1/2}Φ^T M_x Φ Λ^{1/2}$ |
| **Geometry** | Modified data in band space | Preserved data geometry with metric |

## Parameter Reference

### Attention Parameters

```python
attn_enabled=True              # Enable attention workflow
attn_weights=np.array([...])   # Weight vector (direct) or None
attn_weights_file='path.csv'   # Alternative: load from CSV
attn_alpha=0.75                # Exponent α in [0.5, 1]
```

**attn_alpha** controls how strongly attention weights affect the metric:
- `α = 0.5`: Mild emphasis (√w)
- `α = 1.0`: Strong emphasis (w)
- Recommended: `0.5` to `1.0`

### PCA Parameters

```python
pca=True                       # Enable PCA
pca_whiten=True                # Enable full-rank whitening (OPTION B)
pca_method='cum_energy'        # Standard method
pca_cum_energy=1-1e-5          # Keep all dimensions for whitening
```

For Option B, always use:
- `pca_whiten=True` (distinguishes from Option A)
- `pca_cum_energy=1-1e-5` or similar (keep most/all dimensions)

### DMAPS Parameters

```python
dmaps=True                     # Enable DMAPS
dmaps_epsilon='auto'           # Auto-select kernel bandwidth
dmaps_kappa=1                  # Diffusion power
dmaps_L=0.1                    # Eigenvalue cutoff for dimension
dmaps_first_evec=False         # Include first eigenvector
dmaps_m_override=0             # Manual dimension override
```

## Understanding the Metric Matrix

The key difference in Option B is the **metric matrix**:

$$M_η = Λ_r^{1/2} Φ_r^T M_x Φ_r Λ_r^{1/2}$$

Where:
- $M_x = \text{diag}(w^α)$ (attention metric in band space)
- $Φ_r$ (PCA eigenvectors)
- $Λ_r$ (PCA eigenvalues)

**Interpretation:**
- Weights $w > 1.0$ emphasize certain directions
- Weights $w < 1.0$ de-emphasize directions
- Metric is computed *after* PCA whitening
- Distances in DMAPS: $d²_M = (η_i - η_j)^T M_η (η_i - η_j)$

### Accessing the Metric Matrix

```python
# After run()
M_eta = plom_dict['dmaps']['metric_matrix']
R = plom_dict['dmaps']['R']  # Cholesky factor (M_eta = R^T @ R)

print(f"Metric condition number: {np.linalg.cond(M_eta):.4e}")
print(f"Metric eigenvalues: {np.linalg.eigvals(M_eta)}")

# Use Cholesky factor for distance computations
transformed_data = data @ R.T
distances = np.linalg.norm(transformed_data[:, None, :] - 
                           transformed_data[None, :, :], axis=2)
```

## Workflow Steps

```
1. Load Data
   └─ shape: (n_samples, n_features)

2. Robust Scaling (Median/IQR)
   └─ Replace mean/std with median/IQR for outlier robustness
   └─ Output: scaled_training (mean=0, IQR normalized)

3. PCA Whitening (Full-rank, scale_evecs=True)
   └─ No attention preconditioning in band space
   └─ Apply to scaled data only
   └─ Full-rank whitening: pca_dim = n_features
   └─ Output: whitened_training (normalized covariance)

4. Compute Attention Metric
   └─ M_x = diag(w^α) in band space
   └─ Transform to whitened space: M_η = Λ^{1/2}Φ^T M_x Φ Λ^{1/2}
   └─ Compute Cholesky: M_η = R^T @ R
   └─ Output: M_eta, R

5. Anisotropic DMAPS
   └─ Compute distances using metric: d² = ||R(η_i - η_j)||²
   └─ Build kernel: K = exp(-d²/(4ε))
   └─ Eigendecompose for basis vectors
   └─ Output: dmaps_training (n_samples, m)

6. Sampling (Itô SDE)
   └─ Sample on manifold with invariant measure
   └─ Generate augmented data
   └─ Output: augmented_data (n_aug, n_features in original space)

7. Inverse Transformations
   └─ Inverse DMAPS projection
   └─ Inverse PCA whitening
   └─ Inverse scaling
   └─ **Skip** inverse attention preconditioning (not applied!)
   └─ Output: reconst_training (n_samples, n_features)

8. Compute Reconstruction RMSE
   └─ RMSE = sqrt(mean((training - reconst)²))
```

## Full Example: Synthetic Data

```python
import numpy as np
from plom import initialize, run

# Generate synthetic 2D circles
np.random.seed(42)
n_samples = 200
theta = np.linspace(0, 2*np.pi, n_samples, endpoint=False)

# Two concentric circles
circle1 = np.column_stack([1.0 * np.cos(theta), 1.0 * np.sin(theta)])
circle2 = np.column_stack([2.5 * np.cos(theta), 2.5 * np.sin(theta)])
training_data = np.vstack([circle1, circle2])

# Add small noise
noise = 0.05 * np.random.randn(*training_data.shape)
training_data = training_data + noise

# Create attention weights
# Emphasize x-direction, de-emphasize y-direction
attn_weights = np.array([1.3, 0.7])

# Initialize Option B
plom_dict = initialize(
    training=training_data,
    
    scaling=True,
    scaling_method='RobustIQR',
    
    attn_enabled=True,
    attn_weights=attn_weights,
    attn_alpha=0.8,
    
    pca=True,
    pca_whiten=True,              # Option B: Enable whitening
    pca_cum_energy=1-1e-5,        # Keep all dimensions
    
    dmaps=True,
    dmaps_epsilon='auto',
    dmaps_L=0.1,
    
    projection=True,
    sampling=True,
    num_samples=100,
    ito_steps='auto',
    
    verbose=True
)

# Run analysis
run(plom_dict)

# Analyze results
print(f"\n=== Option B Results ===")
print(f"Original data shape: {training_data.shape}")
print(f"Augmented data shape: {plom_dict['data']['augmented'].shape}")
print(f"Reconstruction RMSE: {plom_dict['data']['rmse']:.6e}")
print(f"Manifold dimension: {plom_dict['dmaps']['dimension']}")

# Examine metric matrix
M_eta = plom_dict['dmaps']['metric_matrix']
print(f"Metric matrix condition number: {np.linalg.cond(M_eta):.4e}")
print(f"Metric matrix eigenvalues: {np.linalg.eigvals(M_eta)}")

# Verify attention was incorporated
attn_weights_used = plom_dict['attn']['weights']
print(f"Attention weights used: {attn_weights_used}")
print(f"Attention alpha: {plom_dict['attn']['alpha']}")
```

## Advanced: Custom Attention Weights

```python
# Load transformer output
attention_file = 'path/to/band_attention_global.csv'

# Option 1: Automatic loading
plom_dict = initialize(
    training=data,
    attn_enabled=True,
    attn_weights_file=attention_file,  # Auto-loaded
    attn_alpha=0.75,
    pca_whiten=True,
    # ... other parameters
)

# Option 2: Manual loading with normalization
import numpy as np
weights = np.loadtxt(attention_file, delimiter=',')
weights = weights.flatten()  # Handle 1D/2D arrays

# Normalize if desired
weights = weights / np.mean(weights)  # Mean-center to 1.0

plom_dict = initialize(
    training=data,
    attn_enabled=True,
    attn_weights=weights,
    attn_alpha=0.75,
    pca_whiten=True,
    # ... other parameters
)

run(plom_dict)
```

## Troubleshooting

### "Metric matrix not positive definite"

If you see: `Warning: Metric matrix not positive definite, using eigenvalue decomposition`

This is expected sometimes due to numerical precision. The code automatically uses eigenvalue decomposition as fallback.

**Solution:** Usually fine to ignore. Monitor condition number:
```python
cond_num = np.linalg.cond(plom_dict['dmaps']['metric_matrix'])
if cond_num > 1e4:
    print(f"Warning: Metric ill-conditioned ({cond_num:.2e})")
```

### Large RMSE (> 0.1)

Option B reconstruction should be comparable to Option A. If RMSE is very large:

1. Check attention weights are reasonable (mostly in 0.5-2.0 range)
2. Verify `attn_alpha` is in [0.5, 1.0]
3. Try reducing `attn_alpha` if weights are extreme
4. Ensure `pca_cum_energy` is high (≥ 1-1e-3)

### "No augmented data generated"

If `plom_dict['data']['augmented']` is None:

1. Ensure `sampling=True` in initialize
2. Check `num_samples > 0`
3. Verify `dmaps_dimension > 0` (manifold dimension must be detected)

## Differences from Option A

**Option A (with preconditioning):**
```python
initialize(
    attn_enabled=True,
    attn_weights=weights,
    attn_alpha=0.75,
    pca_whiten=False,  # ← Option A: No whitening
    # DMAPS uses isotropic kernel on preconditioned data
)
```

**Option B (with anisotropic kernel):**
```python
initialize(
    attn_enabled=True,
    attn_weights=weights,
    attn_alpha=0.75,
    pca_whiten=True,   # ← Option B: Enable whitening
    # DMAPS uses anisotropic metric-informed kernel
)
```

The workflow is determined automatically by the `pca_whiten` flag when `attn_enabled=True`.

## References

- Soize, C. & Ghanem, R. (2016). *Data-driven probability concentration and sampling on manifold*. Journal of Computational Physics, 321, 242-258.
- Diffusion Maps: Coifman, R. R. & Lafon, S. (2006). *Diffusion maps*. Applied and Computational Harmonic Analysis, 21(1), 5-30.
