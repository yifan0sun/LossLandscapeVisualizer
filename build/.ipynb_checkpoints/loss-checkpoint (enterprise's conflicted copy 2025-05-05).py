# loss.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
from data import get_blob_dataset  # <-- You should implement this in blob_datasets.py
from model import Blob_MLP  # Replaces MNIST_MLP

loss_fn = nn.BCEWithLogitsLoss()

def load_dataset(dataset_name: str, train=True):
    """Load selected blob dataset."""
    data, labels = get_blob_dataset(dataset_name)
    # reshape labels to (N, 1) and convert to float for BCE loss
    labels = labels.float().unsqueeze(1)
    return TensorDataset(data, labels)

def compute_loss(model: nn.Module, dataset_name: str) -> float:
    """Compute average BCE loss on a dataset."""
    dataset = load_dataset(dataset_name)
    loader = DataLoader(dataset, batch_size=1000, shuffle=False)

    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches

def debug_loss(dataset_name: str, path="./mlp_blob.pth"):
    """CLI wrapper to compute and print loss."""
    if not os.path.exists(path):
        raise RuntimeError(f"Model file {path} not found.")

    model = Blob_MLP()
    model.load_state_dict(torch.load(path))

    avg_loss = compute_loss(model, dataset_name)
    print(f"📉 Average loss on '{dataset_name}': {avg_loss:.4f}")

# --- CLI Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute average loss for a trained MLP on blob datasets.")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["two_blobs_separate", "two_blobs_overlap", "two_blobs_fully_overlap",
                                 "half_moons", "spirals", "four_corners"],
                        help="Dataset to evaluate on")
    args = parser.parse_args()
    debug_loss(dataset_name=args.dataset)
