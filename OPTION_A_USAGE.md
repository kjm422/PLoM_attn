# Option A: Attention-Informed Manifold Learning - Usage Guide

## Quick Start with Transformer Attention Weights

### Step 1: Load Your Data and Transformer Weights

```python
import numpy as np
from plom import initialize, run

# Your training spectral data (n_samples x n_features, e.g., 100 x 285)
training_data = np.loadtxt('training_spectra.txt')

# Option A: Load attention weights from transformer output
sol_dict = initialize(
    training=training_data,
    attn_enabled=True,
    attn_weights_file='band_attention_global.csv',  # From your transformer
    attn_alpha=0.8,
    pca_whiten=True,
    scaling_method='RobustIQR',
    dmaps=True,
    sampling=True,
    num_samples=100,
    job_desc="With transformer attention weights",
    verbose=True
)

# Step 2: Run the workflow
run(sol_dict)

# Step 3: Access results
synthetic_samples = sol_dict['data']['augmented']  # Generated samples
rmse = sol_dict['data']['rmse']  # Reconstruction error

# Save samples
np.savetxt('generated_samples.txt', synthetic_samples)
```

## Option A Workflow Steps

When `attn_enabled=True`, the algorithm executes:

1. **Robust Scaling** → Median/IQR normalization (outlier-resistant)
2. **Attention Preconditioning** → Apply $S = \text{diag}((w^\alpha)^{1/2})$
3. **PCA Whitening** → Full-rank eigenvector scaling (`pca_whiten=True`)
4. **Diffusion Maps** → Learn manifold structure
5. **Itô Sampling** → Generate new samples on manifold
6. **Inverse Transformations** → Map back to original space

## CSV File Format

`band_attention_global.csv` should contain:
- A **1D vector** of attention weights (one per spectral band)
- Can be 1×285, 285×1, or even just comma-separated values
- Example:
  ```
  0.5, 0.6, 0.8, 1.2, 0.9, ...
  ```
  or
  ```
  0.5
  0.6
  0.8
  ...
  ```

## Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `attn_enabled` | bool | False | Activate Option A workflow |
| `attn_weights_file` | str | None | Path to CSV with attention weights |
| `attn_weights` | ndarray | None | Direct array of weights (alternative to file) |
| `attn_alpha` | float | 1.0 | Exponent α ∈ [0.5, 1] controlling attention strength |
| `pca_whiten` | bool | False | Use full-rank PCA whitening |
| `scaling_method` | str | 'Normalization' | Use 'RobustIQR' for robust scaling |

## Advanced: Manual Weight Specification

If you don't have a CSV file, you can provide weights directly:

```python
# Create attention weights that emphasize certain bands
attn_weights = np.ones(285) * 0.5
attn_weights[50:100] = 2.0  # Emphasize bands 50-100
attn_weights[:20] = 0.3      # De-emphasize first 20 bands

sol_dict = initialize(
    training=training_data,
    attn_enabled=True,
    attn_weights=attn_weights,  # Direct array instead of file
    attn_alpha=0.8,
    ...
)
```

## Understanding Attention Weights

- **w > 1.0**: Emphasize these bands (give them more importance)
- **w = 1.0**: Standard weight (no emphasis or de-emphasis)
- **w < 1.0**: De-emphasize these bands

The transformation applied is: $\tilde{x}^{(\text{att})} = S \tilde{x}$ where $S = \text{diag}((w^\alpha)^{1/2})$

Higher `attn_alpha` → stronger attention effect
Lower `attn_alpha` → attention effect is dampened

## Full Example: Spectral Analysis Workflow

```python
import numpy as np
import pandas as pd
from plom import initialize, run

# Load spectra
spectra = np.loadtxt('spectra.txt')  # (1000, 285)

# Load transformer attention weights
attn = np.loadtxt('band_attention_global.csv', delimiter=',')

# Configure Option A
params = {
    'training': spectra,
    'attn_enabled': True,
    'attn_weights_file': 'band_attention_global.csv',
    'attn_alpha': 0.8,
    'pca_whiten': True,
    'scaling_method': 'RobustIQR',
    'dmaps': True,
    'dmaps_epsilon': 'auto',
    'dmaps_L': 0.1,
    'sampling': True,
    'num_samples': 500,
    'job_desc': 'Spectral synthesis with transformer attention'
}

sol_dict = initialize(**params)
run(sol_dict)

# Results
print(f"Generated {sol_dict['data']['augmented'].shape[0]} synthetic spectra")
print(f"Reconstruction RMSE: {sol_dict['data']['rmse']:.6e}")

# Save
np.savetxt('synthetic_spectra.txt', sol_dict['data']['augmented'])
print("Synthetic spectra saved to synthetic_spectra.txt")
```

## Troubleshooting

### Error: "Failed to load attention weights from CSV"
- Check file path is correct and file exists
- Verify CSV contains only numeric values (no headers)
- For 1×N or N×1 formats, ensure no extra spaces/newlines

### Warning: "Attention weights not provided. Using uniform weights"
- This happens if `attn_weights=None` and `attn_weights_file=None`
- All bands will be weighted equally (α has no effect)

### NaN in results
- May indicate too many features vs. samples
- Consider reducing PCA dimensions with `pca_cum_energy < 1.0`
- Or provide more training samples

---

For more details, see README.txt and the mathematical formulation in the paper.
