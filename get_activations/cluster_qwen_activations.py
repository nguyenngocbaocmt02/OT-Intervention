import os
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Constants
HF_NAMES = {
    'qwen_2.5_1.5B': 'Qwen/Qwen2.5-1.5B',
    'qwen_2.5_1.5B-math': 'Qwen/Qwen2.5-Math-1.5B',
}


def load_hidden_states(file_path, layer_idx):
    """Load hidden states from .npy file and select a specific layer."""
    hidden_states = np.load(file_path)  # Shape: (num_samples, num_layers, hidden_dim)
    return hidden_states[:, layer_idx, :].squeeze(1)  # Shape: (num_samples, hidden_dim)


def plot_pca_2d_with_kmedoids(data, labels, medoid_indices, title, output_path=None):
    """Perform PCA and plot the 2D projection with all data and medoids highlighted."""
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    plt.figure(figsize=(8, 6))
    # Plot all points
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='lightblue', label='Data Points', alpha=0.7)

    # Highlight medoids
    medoid_points = reduced_data[medoid_indices]
    plt.scatter(medoid_points[:, 0], medoid_points[:, 1], c='red', label='Medoids', edgecolors='black', s=100)

    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()

    if output_path:
        plt.savefig(f"{output_path}.png")
        plt.savefig(f"{output_path}.pdf")
    else:
        plt.show()


# Load stuffs
hidden_states_path = '../features/prm800k_test'
layer_idx = 15
num_clusters = 20

# Paths for the hidden states
qwen_file = os.path.join(hidden_states_path, 'qwen_2.5_1.5B_hidden_states.npy')
qwen_math_file = os.path.join(hidden_states_path, 'qwen_2.5_1.5B-math_hidden_states.npy')

if not os.path.exists(qwen_file) or not os.path.exists(qwen_math_file):
    raise FileNotFoundError("Hidden states files not found. Ensure the paths are correct.")

# Load hidden states for the specified layer
print(f"Loading hidden states from layer {layer_idx}...")
qwen_hidden_states = load_hidden_states(qwen_file, layer_idx)  # Shape: (num_samples, hidden_dim)
qwen_math_hidden_states = load_hidden_states(qwen_math_file, layer_idx)

# Compute the difference between hidden states
print("Computing differences between hidden states...")
differences = qwen_hidden_states - qwen_math_hidden_states  # Shape: (num_samples, hidden_dim)

# Run K-Medoids clustering
print(f"Clustering with {num_clusters} clusters using K-Medoids...")
kmedoids = KMedoids(n_clusters=num_clusters, metric='euclidean', random_state=42)
kmedoids.fit(differences)

# Plot PCA 2D of the dataset with medoids highlighted
print("Plotting PCA 2D visualization with medoids highlighted...")
plot_pca_2d_with_kmedoids(
    differences,
    kmedoids.labels_,
    kmedoids.medoid_indices_,
    title=f"K-Medoids Clusters with Medoids (Layer {layer_idx})",
    output_path=os.path.join(hidden_states_path, f"kmedoids_layer_{layer_idx}_pca")
)
