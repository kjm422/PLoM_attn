#!/usr/bin/env python3
"""
Test Option B: Attention-informed anisotropic kernel workflow.

This test verifies that:
1. Robust scaling works correctly
2. PCA whitening produces full-rank whitened data
3. Attention weights are properly loaded
4. Metric matrix is computed correctly
5. Anisotropic DMAPS kernel is used
6. Reconstruction works end-to-end
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/kmccoy/Documents/USC/Research/Dissertation/kelli_scripts/PLoM-1')

from plom import initialize, run

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("Testing Option B: Attention-informed Anisotropic Kernel Workflow")
print("=" * 70)

# Generate synthetic training data (2D circles)
n_samples = 100
theta = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
r1 = 1.0
r2 = 2.0
# Create two circle data
circle1 = np.column_stack([r1 * np.cos(theta), r1 * np.sin(theta)])
circle2 = np.column_stack([r2 * np.cos(theta), r2 * np.sin(theta)])
training_data = np.vstack([circle1, circle2])

# Add small noise
noise = 0.05 * np.random.randn(*training_data.shape)
training_data = training_data + noise

print(f"\nTraining data shape: {training_data.shape}")
print(f"Training data range: [{training_data.min():.3f}, {training_data.max():.3f}]")

# Create simple attention weights
# Let's emphasize the first feature more than the second
attn_weights = np.array([1.5, 0.8])
print(f"\nAttention weights: {attn_weights}")
print(f"  Feature 1: {attn_weights[0]:.2f} (emphasis)")
print(f"  Feature 2: {attn_weights[1]:.2f} (de-emphasis)")

# Initialize PLoM for Option B
print("\n" + "-" * 70)
print("Initializing PLoM with Option B settings...")
print("-" * 70)

plom_dict = initialize(
    training=training_data,
    
    # Scaling
    scaling=True,
    scaling_method='RobustIQR',
    
    # Option B parameters
    attn_enabled=True,
    attn_weights=attn_weights,
    attn_alpha=0.75,  # 0 < alpha <= 1
    
    # PCA with whitening (Option B)
    pca=True,
    pca_method='cum_energy',
    pca_cum_energy=1-1e-5,  # Keep all dimensions
    pca_whiten=True,  # Enable full-rank whitening
    
    # DMAPS with anisotropic kernel
    dmaps=True,
    dmaps_epsilon='auto',  # Auto-select epsilon
    dmaps_kappa=1,
    dmaps_L=0.1,
    dmaps_first_evec=False,
    
    # Projection and sampling
    projection=True,
    sampling=True,
    num_samples=50,
    ito_steps='auto',
    ito_pot_method=2,
    ito_kde_bw_factor=1,
    
    # Utilities
    verbose=True,
    parallel=False,
    save_samples=False,
)

print("\nPLoM initialization complete")
print(f"Attn weights shape: {plom_dict['attn']['weights'].shape}")
print(f"Attn alpha: {plom_dict['attn']['alpha']}")

# Run PLoM with Option B
print("\n" + "=" * 70)
print("Running PLoM with Option B workflow...")
print("=" * 70)

run(plom_dict)

# Verify results
print("\n" + "=" * 70)
print("Option B Results Verification")
print("=" * 70)

# Check scaling
print("\nScaling:")
print(f"  Scaled training shape: {plom_dict['scaling']['training'].shape}")
print(f"  Scaled training range: [{plom_dict['scaling']['training'].min():.3f}, "
      f"{plom_dict['scaling']['training'].max():.3f}]")

# Check PCA whitening
print("\nPCA Whitening:")
print(f"  PCA training shape: {plom_dict['pca']['training'].shape}")
print(f"  PCA mean: {plom_dict['pca']['mean']}")
print(f"  PCA eigenvalues: {plom_dict['pca']['eigvals'][:5]}...")
print(f"  Number of PCA dimensions: {len(plom_dict['pca']['eigvals'])}")

# Check that attention preconditioning was NOT applied (Option B)
if plom_dict['attn']['training'] is None:
    print("\nAttention Preconditioning:")
    print("  ✓ Correctly skipped for Option B (handled by anisotropic kernel)")
else:
    print("\nAttention Preconditioning:")
    print("  ⚠ WARNING: Attention preconditioning was applied (should be None for Option B)")

# Check DMAPS
print("\nDMAPS Analysis:")
print(f"  DMAPS dimension: {plom_dict['dmaps']['dimension']}")
print(f"  DMAPS epsilon: {plom_dict['dmaps']['epsilon']:.4f}")
print(f"  DMAPS eigenvalues (top 5): {plom_dict['dmaps']['eigenvalues'][:5]}")

# Check metric matrix
if plom_dict['dmaps']['metric_matrix'] is not None:
    print(f"\n  Metric matrix shape: {plom_dict['dmaps']['metric_matrix'].shape}")
    print(f"  Metric matrix condition number: {np.linalg.cond(plom_dict['dmaps']['metric_matrix']):.4e}")
    print("  ✓ Anisotropic kernel metric matrix computed")
else:
    print("\n  ⚠ WARNING: Metric matrix is None (should be computed for Option B)")

# Check sampling
print("\nSampling:")
if plom_dict['data']['augmented'] is not None:
    print(f"  Augmented data shape: {plom_dict['data']['augmented'].shape}")
    print(f"  Total data (training + augmented): {plom_dict['data']['training'].shape[0] + plom_dict['data']['augmented'].shape[0]}")
else:
    print("  No augmented data generated")

# Check reconstruction
print("\nReconstruction:")
print(f"  Reconstruction RMSE: {plom_dict['data']['rmse']:.6e}")
if plom_dict['data']['rmse'] < 0.1:
    print("  ✓ RMSE is small (good reconstruction)")
else:
    print("  ⚠ RMSE is large (may indicate issues)")

print("\n" + "=" * 70)
print("Option B Test Complete")
print("=" * 70)
