# data.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, random_split
from sklearn.datasets import make_blobs, make_moons

# --- Dataset Generators ---

def make_two_blobs(std=0.5, centers=[[-2, 0], [2, 0]], n_samples=1000):
    data, labels = make_blobs(n_samples=n_samples, centers=centers, cluster_std=std, random_state=42)
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def make_half_moons(n_samples=1000, noise=0.025):
    data, labels = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def make_spirals(n_samples=500, noise=0.5):
    n = np.sqrt(np.random.rand(n_samples, 1)) * 580 * (2 * np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_samples, 1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_samples, 1) * noise
    X = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y))))
    y = np.hstack((np.zeros(n_samples), np.ones(n_samples)))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def make_four_corners(n_samples_per_class=250):
    corners = [
        np.random.randn(n_samples_per_class, 2) * 0.2 + offset
        for offset in [[-2, -2], [2, 2], [-2, 2], [2, -2]]
    ]
    data = np.vstack(corners)
    labels = np.hstack([
        np.zeros(n_samples_per_class),
        np.zeros(n_samples_per_class),
        np.ones(n_samples_per_class),
        np.ones(n_samples_per_class),
    ])
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


# --- Save and Load ---

def save_datasets(save_dir="../data", train_frac=0.5):
    os.makedirs(save_dir, exist_ok=True)

    dataset_generators = {
        "two_blobs_separate": lambda: make_two_blobs(std=0.5),
        "two_blobs_overlap": lambda: make_two_blobs(std=1.5), 
        "half_moons": make_half_moons,
        "spirals": make_spirals,
        "four_corners": make_four_corners,
    }

    for name, gen_fn in dataset_generators.items():
        data, labels = gen_fn()
        dataset = TensorDataset(data, labels)

        n_samples = len(dataset)
        n_train = int(train_frac * n_samples)
        n_test = n_samples - n_train

        train_set, test_set = random_split(dataset, [n_train, n_test])

        # Save datasets
        torch.save((train_set.dataset.tensors[0][train_set.indices], train_set.dataset.tensors[1][train_set.indices]),
                   os.path.join(save_dir, f"{name}_train.pt"))
        torch.save((test_set.dataset.tensors[0][test_set.indices], test_set.dataset.tensors[1][test_set.indices]),
                   os.path.join(save_dir, f"{name}_test.pt"))

        print(f"✅ Saved {name} train/test splits.")

def load_dataset(dataset_name: str, train=True, save_dir="../data"):
    split = "train" if train else "test"
    path = os.path.join(save_dir, f"{dataset_name}_{split}.pt")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file {path} not found.")

    data, labels = torch.load(path)
    labels = labels.float().unsqueeze(1)  # For BCE loss shape compatibility
    return TensorDataset(data, labels)

# --- Plotting ---

def plot_all_datasets(save_dir="../data"):
    dataset_names = [
        "two_blobs_separate",
        "two_blobs_overlap",
        "half_moons",
        "spirals",
        "four_corners",
    ]

    splits = ["train", "test"]
    n = len(dataset_names) * len(splits)
    ncols = 1
    nrows = 1

    for idx, name in enumerate(dataset_names):
        fig, ax = plt.subplots(nrows, ncols, figsize=(3, 3))
         

        for split_idx, split in enumerate(splits):
            dataset = load_dataset(name, train=(split == "train"), save_dir=save_dir)
            data, labels = dataset.tensors
             
            ax.scatter(data[:, 0], data[:, 1], c=labels.squeeze(), cmap="coolwarm", s=10, edgecolor='k', alpha=0.7-.3*float(split=="test"))
            #ax.set_title(f"{name}")
            ax.set_xticks([])
            ax.set_yticks([])

        ax.axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, "previews", f"{name}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.show()

# --- Main Execution ---

if __name__ == "__main__":
    save_datasets()
    plot_all_datasets()
