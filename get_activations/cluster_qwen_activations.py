import os
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_random_state
from scipy.spatial.distance import cdist
from typing import Callable, Union
from tqdm import tqdm

# Constants
HF_NAMES = {
    'qwen_2.5_1.5B': 'Qwen/Qwen2.5-1.5B',
    'qwen_2.5_1.5B-math': 'Qwen/Qwen2.5-Math-1.5B',
}

class MiniBatchKMedoids(BaseEstimator, ClusterMixin, TransformerMixin):
    """Mini-Batch K-Medoids clustering using sklearn_extra.cluster.KMedoids.
    
    This implementation closely follows the original MSMBuilder implementation
    while using sklearn_extra.cluster.KMedoids as the base clusterer.
    
    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form and medoids to generate.
    max_iter : int, default=5
        Maximum number of iterations over the complete dataset.
    batch_size : int, default=100
        Size of the mini batches.
    metric : str or callable, default='euclidean'
        The distance metric to use. Can be:
        - str: Any metric supported by scipy.spatial.distance.cdist
        - callable: A function that takes two arrays X and Y as input
          and returns a distance matrix of shape (len(X), len(Y))
    max_no_improvement : int, default=10
        Early stopping when no improvement is seen for this many consecutive mini-batches.
    random_state : int or RandomState, default=None
        Controls randomness for initialization.
    """
    
    def __init__(self, 
                 n_clusters: int = 8, 
                 max_iter: int = 5, 
                 batch_size: int = 100,
                 metric: Union[str, Callable] = 'euclidean',
                 max_no_improvement: int = 10, 
                 random_state: Union[int, np.random.RandomState, None] = None):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.max_no_improvement = max_no_improvement
        self.metric = metric
        self.random_state = random_state
    
    def _compute_distances(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute distances between X and Y using the specified metric."""
        if callable(self.metric):
            return self.metric(X, Y)
        else:
            return cdist(X, Y, metric=self.metric)
    
    def fit(self, X: np.ndarray, y=None):
        """Fit mini-batch k-medoids clustering."""
        X = np.asarray(X)
        n_samples = len(X)
        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        n_iter = int(self.max_iter * n_batches)
        random_state = check_random_state(self.random_state)
        
        # Initialize exactly as in original implementation
        self.cluster_ids_ = random_state.randint(0, n_samples, size=self.n_clusters)
        self.labels_ = random_state.randint(0, self.n_clusters, size=n_samples)
        
        n_iters_no_improvement = 0
        
        for iteration in tqdm(range(n_iter)):
            # Select minibatch exactly as in original implementation
            minibatch_indices = np.concatenate([
                self.cluster_ids_,
                random_state.randint(0, n_samples, self.batch_size),
            ])
            
            # Extract mini-batch data
            batch_X = X[minibatch_indices]
            
            # Initialize batch labels as in original
            batch_labels = np.concatenate([
                np.arange(self.n_clusters),
                self.labels_[minibatch_indices[self.n_clusters:]]
            ])
            
            # Fit KMedoids on the batch
            kmedoids = KMedoids(
                n_clusters=self.n_clusters,
                metric=self.metric,
                init='k-medoids++',
                random_state=random_state,
                method='alternate'
            )
            
            # Fit and get new labels
            kmedoids.fit(batch_X)
            new_batch_labels = kmedoids.labels_
            
            # Update cluster centers (medoids)
            new_medoid_indices = np.array([minibatch_indices[kmedoids.medoid_indices_[i]] 
                                         for i in range(self.n_clusters)])
            
            # Check for changes in labels (following original implementation)
            n_changed = np.sum(self.labels_[minibatch_indices] != new_batch_labels)
            if n_changed == 0:
                n_iters_no_improvement += 1
            else:
                self.cluster_ids_ = new_medoid_indices
                self.labels_[minibatch_indices] = new_batch_labels
                n_iters_no_improvement = 0
            
            # Early stopping as in original
            if n_iters_no_improvement >= self.max_no_improvement:
                break
        
        # Set final cluster centers
        self.cluster_centers_ = X[self.cluster_ids_]
        
        # Compute final labels and inertia for all points
        distances = self._compute_distances(X, self.cluster_centers_)
        self.labels_ = np.argmin(distances, axis=1)
        self.inertia_ = np.sum(np.min(distances, axis=1))
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the closest cluster each sample in X belongs to."""
        X = np.asarray(X)
        distances = self._compute_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and predict in one step."""
        return self.fit(X).labels_


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
# print("Computing differences between hidden states...")
# differences = qwen_hidden_states - qwen_math_hidden_states  # Shape: (num_samples, hidden_dim)

# Compute the mean and covariance of the hidden states, then sample use these two to sample 100k sampels from the multivariate normal
mean_hidden_states = np.mean(qwen_hidden_states, axis=0)
cov_hidden_states = np.cov(qwen_hidden_states, rowvar=False)

# Sample 100k samples from the multivariate normal distribution
num_samples = 100000
sampled_hidden_states = np.random.multivariate_normal(
    mean_hidden_states, cov_hidden_states, num_samples)


# Run K-Medoids clustering
print(f"Clustering with {num_clusters} clusters using K-Medoids...")

minibatch_kmedoids = MiniBatchKMedoids(
    n_clusters=num_clusters,
    metric='euclidean', 
    batch_size=10000,
    random_state=42)

minibatch_kmedoids.fit(sampled_hidden_states)

# Plot PCA 2D of the dataset with medoids highlighted
print("Plotting PCA 2D visualization with medoids highlighted...")
plot_pca_2d_with_kmedoids(
    sampled_hidden_states,
    minibatch_kmedoids.labels_,
    minibatch_kmedoids.medoid_indices_,
    title=f"Minibatch K-Medoids Clusters with Medoids (Layer {layer_idx})",
    output_path=os.path.join(hidden_states_path, f"kmedoids_layer_{layer_idx}_pca")
)
