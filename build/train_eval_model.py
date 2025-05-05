# train_eval_model.py

import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from model import BlobMLP, arch_to_name  
from data import load_dataset

# Constants
BASE_MODELS_DIR = "../backend/models"
os.makedirs(BASE_MODELS_DIR, exist_ok=True)

def compute_accuracy(model: nn.Module, dataset_name: str, train=True) -> float:
    """Compute accuracy of a model on train or test set."""
    dataset = load_dataset(dataset_name, train=train)
    loader = DataLoader(dataset, batch_size=1000, shuffle=False)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            preds = (outputs > 0).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total

 
def latest_checkpoint(arch, dataset_name, base_dir=BASE_MODELS_DIR):
    arch_name = arch_to_name(arch)
    model_dir = os.path.join(base_dir, arch_name, dataset_name)
    if not os.path.exists(model_dir):
        return model_dir, None, 0  # No models yet
    files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    if not files:
        return model_dir, None, 0
    files.sort(key=lambda f: int(f.split("ep")[-1].split(".")[0]))
    latest = files[-1]
    epoch = int(latest.split("ep")[-1].split(".")[0])
    path = os.path.join(model_dir, latest)
    return model_dir, path, epoch


def train_model(arch, epochs, dataset_name: str):
    model_dir, latest_model_path, existing_epochs = latest_checkpoint(arch, dataset_name)

        

    model = BlobMLP(hidden_layers=arch)
    if existing_epochs > 0:
        print(f"🔁 Resuming from {existing_epochs} epochs: {latest_model_path}")
        model.load_state_dict(torch.load(latest_model_path))

    os.makedirs(model_dir, exist_ok=True)
    epoch0_path = os.path.join(model_dir, "ep0.pth")
    if not os.path.exists(epoch0_path):
        torch.save(model.state_dict(), epoch0_path)
        print(f"🚀 Saved initial untrained model at {epoch0_path}")

        
    if existing_epochs >= epochs:
        print(f"✅ Model already trained for {existing_epochs} epochs: {latest_model_path}")
    else:
        train_data = load_dataset(dataset_name, train=True)
        train_loader = DataLoader(train_data, batch_size=1000, shuffle=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.BCEWithLogitsLoss()

        os.makedirs(model_dir, exist_ok=True)

        for epoch in range(existing_epochs + 1, epochs + 1):
            model.train()
            total_loss = 0.
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                output = model(inputs)
                loss = loss_fn(output, labels)
                total_loss += loss.detach()
                loss.backward()
                optimizer.step()

            model_path = os.path.join(model_dir, f"ep{epoch}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"📦 Epoch {epoch} complete. Loss {total_loss}. Saved to {model_path}")

    acc = compute_accuracy(model, dataset_name, train=True)
    print(f"📈 Final train eval after ({epochs} epochs) [{dataset_name}]: Accuracy = {acc:.2f}%")
    acc = compute_accuracy(model, dataset_name, train=False)
    print(f"📈 Final test eval after ({epochs} epochs) [{dataset_name}]: Accuracy = {acc:.2f}%")



def eval_model(arch, epochs, dataset_name: str):
    model_dir, latest_model_path, existing_epochs = latest_checkpoint(arch, dataset_name)

        

    model = BlobMLP(hidden_layers=arch)
    if existing_epochs > 0:
        print(f"🔁 Resuming from {existing_epochs} epochs: {latest_model_path}")
        model.load_state_dict(torch.load(latest_model_path))

    if existing_epochs < epochs:
        print(f"✅ Model not yet trained for {existing_epochs} epochs: {latest_model_path}")
        return



    acc = compute_accuracy(model, dataset_name, train=True)
    print(f"📈 Final train eval after ({epochs} epochs) [{dataset_name}]: Accuracy = {acc:.2f}%")
    acc = compute_accuracy(model, dataset_name, train=False)
    print(f"📈 Final test eval after ({epochs} epochs) [{dataset_name}]: Accuracy = {acc:.2f}%")

def plot_model(arch, epochs, dataset_name: str):
    model_dir, latest_model_path, existing_epochs = latest_checkpoint(arch, dataset_name)

    if existing_epochs < epochs:
        print(f"❌ Model not trained up to {epochs} epochs yet. Latest: {existing_epochs} epochs.")
        return

    model = BlobMLP(hidden_layers=arch)
    model.load_state_dict(torch.load(latest_model_path))
    model.eval()

    # Prepare datasets
    train_dataset = load_dataset(dataset_name, train=True)
    test_dataset = load_dataset(dataset_name, train=False)

    # Plot setup
    plt.figure(figsize=(8, 6))
    
    # Grid for decision boundary
    grid_points = 200
    # Combine train and test inputs to compute plot limits
    all_inputs = torch.cat([train_dataset.tensors[0], test_dataset.tensors[0]], dim=0)

    x_min = all_inputs[:, 0].min().item() - 0.5
    x_max = all_inputs[:, 0].max().item() + 0.5
    y_min = all_inputs[:, 1].min().item() - 0.5
    y_max = all_inputs[:, 1].max().item() + 0.5
    xx, yy = torch.meshgrid(
        torch.linspace(x_min, x_max, grid_points),
        torch.linspace(y_min, y_max, grid_points),
        indexing='ij'  # PyTorch >=1.10
    )
    grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)

    with torch.no_grad():
        logits = model(grid)
        preds = (logits > 0).float().reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx.numpy(), yy.numpy(), preds.numpy(), levels=1, cmap="coolwarm", alpha=0.3)

    # Plot train points
    train_inputs, train_labels = train_dataset.tensors
    plt.scatter(train_inputs[:, 0], train_inputs[:, 1], c=train_labels.squeeze(), cmap="coolwarm",
                marker='o', edgecolor='k', label="Train", s=40, alpha=0.8)

    # Plot test points
    test_inputs, test_labels = test_dataset.tensors
    plt.scatter(test_inputs[:, 0], test_inputs[:, 1], c=test_labels.squeeze(), cmap="coolwarm",
                marker='x', edgecolor='k', label="Test", s=40, alpha=0.8)

    plt.title(f"Decision Boundary ({dataset_name}, {epochs} epochs)")
    plt.legend()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True)
    plt.show()

# --- CLI ---
if __name__ == "__main__":
    dataset_choices=["two_blobs_separate", "two_blobs_overlap", "two_blobs_fully_overlap",
                                 "half_moons", "spirals", "four_corners"]
    parser = argparse.ArgumentParser(description="Train or evaluate a blob MLP.")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--plot", action="store_true", help="Plot the dataset and model decision boundary")

    parser.add_argument("--arch", nargs="+", type=int, required=True, help="Hidden layer sizes (e.g. --arch 128 64)")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs to train")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["two_blobs_separate", "two_blobs_overlap",
                                 "half_moons", "spirals", "four_corners", "all"],
                        help="Dataset name")
    

    args = parser.parse_args()

    if args.dataset == 'all':
        choices = ["two_blobs_separate", "two_blobs_overlap", "half_moons", "spirals", "four_corners"]
    else:
        choices = [args.dataset]

    """
    for dataset in  choices:
        if args.train:
            train_model(args.arch, args.epochs, dataset)

        if args.eval:
            eval_model(args.arch, args.epochs, dataset)

        if args.plot:
            plot_model(args.arch, args.epochs, dataset)
    """
    
    for width in [5]:#[5,10,25,50,100]:
        
        for depth in [16]:#[1,2,4,8,16]:
            arch = [width for k in range(depth)]
            for dataset in  choices:
                if args.train:
                    train_model(arch, args.epochs, dataset)

                if args.eval:
                    eval_model(arch, args.epochs, dataset)

                #if args.plot:
                #    plot_model(arch, args.epochs, dataset)
