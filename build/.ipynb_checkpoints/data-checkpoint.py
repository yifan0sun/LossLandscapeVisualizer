import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from torch.utils.data import TensorDataset

# --- Dataset Generators ---
def make_two_blobs(std=0.5, centers=[[-2, 0], [2, 0]], n_samples=1000):
    data, labels = make_blobs(n_samples=n_samples, centers=centers, cluster_std=std, random_state=42)
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def make_half_moons(n_samples=1000, noise=0.1):
    data, labels = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def make_spirals(n_samples=500, noise=0.5):
    n = np.sqrt(np.random.rand(n_samples, 1)) * 580 * (2 * np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_samples, 1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_samples, 1) * noise
    X1 = np.hstack((d1x, d1y))
    X2 = np.hstack((-d1x, -d1y))
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros(n_samples), np.ones(n_samples)))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def make_four_corners(n_samples_per_class=250):
    corner_1 = np.random.randn(n_samples_per_class, 2) * 0.2 + [-2, -2]
    corner_2 = np.random.randn(n_samples_per_class, 2) * 0.2 + [2, 2]
    corner_3 = np.random.randn(n_samples_per_class, 2) * 0.2 + [-2, 2]
    corner_4 = np.random.randn(n_samples_per_class, 2) * 0.2 + [2, -2]
    data = np.vstack([corner_1, corner_2, corner_3, corner_4])
    labels = np.array([0]*n_samples_per_class + [0]*n_samples_per_class + [1]*n_samples_per_class + [1]*n_samples_per_class)
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# --- Dataset Dictionary ---
datasets = {
    "two_blobs_separate": make_two_blobs(std=0.5),
    "two_blobs_overlap": make_two_blobs(std=1.5),
    "two_blobs_fully_overlap": make_two_blobs(std=3.0),
    "half_moons": make_half_moons(),
    "spirals": make_spirals(),
    "four_corners": make_four_corners()
}
 
def get_blob_dataset(name):
    if name == "two_blobs_separate":
        return make_two_blobs(std=0.5)
    elif name == "two_blobs_overlap":
        return make_two_blobs(std=1.5)
    elif name == "two_blobs_fully_overlap":
        return make_two_blobs(std=3.0)
    elif name == "half_moons":
        return make_half_moons()
    elif name == "spirals":
        return make_spirals()
    elif name == "four_corners":
        return make_four_corners()
    else:
        raise ValueError(f"Unknown dataset name: {name}")

        
        

# --- Plotting ---
def plot_all_datasets(datasets):
    n = len(datasets)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 10))
    axs = axs.flatten()
    
    for i, (name, (data, labels)) in enumerate(datasets.items()):
        axs[i].scatter(data[:, 0], data[:, 1], c=labels, cmap='coolwarm', s=10, edgecolor='k', alpha=0.7)
        axs[i].set_title(name)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    for j in range(i+1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    plot_all_datasets(datasets)
