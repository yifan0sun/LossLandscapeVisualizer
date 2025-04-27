# train_eval_model.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
from model import Blob_MLP, model_filename, latest_checkpoint  # assumes blob models only now
from loss import load_dataset  # will use updated load_dataset with dataset_name argument
import matplotlib.pyplot as plt
import numpy as np
BASE_MODELS_DIR = "models"  # Clean relative path for saving
os.makedirs(BASE_MODELS_DIR, exist_ok=True)


def compute_accuracy(model: nn.Module, dataset_name: str) -> float:
    dataset = load_dataset(dataset_name)
    loader = DataLoader(dataset, batch_size=1000, shuffle=False)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            preds = (model(inputs) > 0).long()  # binary thresholding for BCE
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total

def model_path(dataset_name: str, arch: list[int], epoch: int) -> str:
    arch_str = "_".join(str(h) for h in arch)
    return os.path.join(BASE_MODELS_DIR, dataset_name, f"mlp_{arch_str}_ep{epoch}.pth")

def train_model(arch, epochs, dataset_name: str):
    latest_path, existing_epochs = latest_checkpoint(arch, dataset_name, model_path)
    if existing_epochs >= epochs:
        print(f"✅ Model already trained for {existing_epochs} epochs: {latest_path}")
        return

    model = Blob_MLP(hidden_layers=arch)
    if existing_epochs > 0:
        print(f"🔁 Resuming from {existing_epochs} epochs: {latest_path}")
        model.load_state_dict(torch.load(latest_path))

    train_data = load_dataset(dataset_name, train=True)
    train_loader = DataLoader(train_data, batch_size=1000, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(existing_epochs + 1, epochs + 1):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
        path = model_path(dataset_name, arch, epoch)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"📦 Epoch {epoch} complete. Saved to {path}")
    acc = compute_accuracy(model, dataset_name)
    print(f"📈 Train eval on checkpoint ({epoch} epochs) [{dataset_name}]: Accuracy = {acc:.2f}%")

def evaluate_model(arch, dataset_name: str, epochs=None):
    if epochs is not None:
        path = model_path(dataset_name, arch, epochs)
        if not os.path.exists(path):
            print(f"❌ No checkpoint found for {epochs} epochs: {path}")
            return
        target_epoch = epochs
    else:
        path, target_epoch = latest_checkpoint(arch, dataset_name, model_path)
        if target_epoch == 0:
            print("❌ No checkpoint found for this architecture.")
            return

    model = Blob_MLP(hidden_layers=arch)
    model.load_state_dict(torch.load(path))
    acc = compute_accuracy(model, dataset_name)
    print(f"📈 Eval on checkpoint ({target_epoch} epochs) [{dataset_name}]: Accuracy = {acc:.2f}%")

    
    
@torch.no_grad()
def plot_decision_boundary(model: nn.Module, dataset_name: str):
    model.eval()
    data, labels = load_dataset(dataset_name)
    X = data.numpy()
    Y = labels.numpy().squeeze()

    # Create meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        probs = torch.sigmoid(model(grid)).numpy().reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, probs, levels=50, cmap='RdBu', alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='bwr', edgecolor='k', s=20)
    plt.title(f"Decision Boundary on '{dataset_name}'")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    
    
# --- CLI ---
if __name__ == "__main__":
    dataset_choices=["two_blobs_separate", "two_blobs_overlap", "two_blobs_fully_overlap",
                                 "half_moons", "spirals", "four_corners"]
    parser = argparse.ArgumentParser(description="Train or evaluate a blob MLP.")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--arch", nargs="+", type=int, required=True, help="Hidden layer sizes (e.g. --arch 128 64)")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs to train")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["two_blobs_separate", "two_blobs_overlap", "two_blobs_fully_overlap",
                                 "half_moons", "spirals", "four_corners", "all"],
                        help="Dataset name")
    parser.add_argument("--plot", action="store_true", help="Plot the dataset and model decision boundary")

    args = parser.parse_args()

    if args.dataset == 'all':
        for dataset in dataset_choices:
            if args.train:
                train_model(args.arch, args.epochs, dataset)
            if args.eval or args.plot:
                path, target_epoch = latest_checkpoint(args.arch, dataset, model_path)
                if target_epoch == 0:
                    print("❌ No checkpoint found for evaluation/plotting.")
                else:
                    model = Blob_MLP(hidden_layers=args.arch)
                    model.load_state_dict(torch.load(path))
                    if args.eval:
                        acc = compute_accuracy(model, dataset)
                        print(f"📈 Eval on checkpoint ({target_epoch} epochs) [{dataset}]: Accuracy = {acc:.2f}%")
                    if args.plot:
                        plot_decision_boundary(model, dataset)
    else:
        if args.train:
            train_model(args.arch, args.epochs, args.dataset)
        if args.eval or args.plot:
            path, target_epoch = latest_checkpoint(args.arch, args.dataset, model_path)
            if target_epoch == 0:
                print("❌ No checkpoint found for evaluation/plotting.")
            else:
                model = Blob_MLP(hidden_layers=args.arch)
                model.load_state_dict(torch.load(path))
                if args.eval:
                    acc = compute_accuracy(model, args.dataset)
                    print(f"📈 Eval on checkpoint ({target_epoch} epochs) [{args.dataset}]: Accuracy = {acc:.2f}%")
                if args.plot:
                    plot_decision_boundary(model, args.dataset)
    