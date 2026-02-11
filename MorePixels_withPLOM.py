import random
import numpy as np
from plom import PLoM

# 1. Load Data (N samples x n_features)

data = np.load('/Users/kmccoy/Documents/USC/Research/Dissertation/Data/TOApixel_balanced_W_gb1ID_gb2ID_train7500_1000.npy')

# Save as .txt
# 'fmt' controls the number format (e.g., %.4f for 4 decimal places)
# 'delimiter' separates your columns
np.savetxt('/Users/kmccoy/Documents/USC/Research/Dissertation/Data/TOApixel_balanced_W_gb1ID_gb2ID_train7500_1000.txt', data, fmt='%s', delimiter='\t')

X_train = np.loadtxt('/Users/kmccoy/Documents/USC/Research/Dissertation/Data/TOApixel_balanced_W_gb1ID_gb2ID_train7500_1000.txt')
X_train_small= list(X_train[:,:-1])

# 2. Initialize Model
model = PLoM(
    integer_columns=[-1],  # Specify which columns are integers (e.g., class labels)
    use_pca=True,              # Enable/Disable pipeline stages
    use_dmaps=True,
    pca_method='cum_energy',   # PCA configuration
    pca_cum_energy=0.99,
    dmaps_epsilon='auto',      # Auto-tune kernel bandwidth
    ito_steps='auto',          # Number of SDE integration steps
    n_jobs=-2,                  # Parallel processing option
    verbose=1                  # 0, 1, 2
)

# 3. Fit the Manifold
model.fit(X_train_small)

# 4. Generate New Samples
# Returns (n_samples * N) points
X_new = model.sample(n_samples=1000)

print(f"Generated data shape: {X_new.shape}")
# Save generated samples
np.save("/Users/kmccoy/Documents/USC/Research/Dissertation/Data/X_new_plom_N1000.npy", X_new)
print("X_new saved to X_new_plom_N1000.npy")

### 2. Saving and Loading Models

# Save the trained model
model.save("/Users/kmccoy/Documents/USC/Research/Dissertation/Data/plom_model_N1000.pkl")

# Load it later
#from plom import PLoM
#loaded_model = PLoM.load("my_plom_model.pkl")
#samples = loaded_model.sample(n_samples=5)
