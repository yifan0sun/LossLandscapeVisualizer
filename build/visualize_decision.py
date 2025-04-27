# visualize.py
import torch
import torch.nn as nn
import os, pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import BlobMLP, arch_to_name, get_model_path
from train_eval_model import latest_checkpoint
from data import load_dataset

CACHE_ROOT = "../decisionboundaries"
MODELS_DIR = "../models"
 
 

 


def compute_decision(arch, dataset, train=True, epoch=1, ab_range=1.0, gridsize=10):
    """Compute or load decision boundary outputs for visualization."""
    train_str = 'train' if train else 'test'
    model_path = os.path.join(MODELS_DIR, arch_to_name(arch), dataset, f"ep{epoch}.pth")
    cache_dir = os.path.join(CACHE_ROOT, arch_to_name(arch), dataset)
    cache_file = os.path.join(cache_dir, f"ep{epoch}.pkl")
    print(cache_dir, cache_file)

    os.makedirs(cache_dir, exist_ok=True)

    # If cached file exists, load and return
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            decision_outputs = pickle.load(f)
        return decision_outputs

    # Otherwise compute
    model = BlobMLP(hidden_layers=arch)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    train_dataset = load_dataset(dataset, train=True)
    test_dataset = load_dataset(dataset, train=False)

    # Create a meshgrid for decision boundary
    grid_points = 200
    all_inputs = torch.cat([train_dataset.tensors[0], test_dataset.tensors[0]], dim=0)

    x_min, x_max = all_inputs[:, 0].min().item() - 0.5, all_inputs[:, 0].max().item() + 0.5
    y_min, y_max = all_inputs[:, 1].min().item() - 0.5, all_inputs[:, 1].max().item() + 0.5

    xx, yy = torch.meshgrid(
        torch.linspace(x_min, x_max, grid_points),
        torch.linspace(y_min, y_max, grid_points),
        indexing='ij'
    )
    grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)

    with torch.no_grad():
        logits = model(grid)
        preds = logits.reshape(xx.shape)

    # Collect outputs
    train_inputs, train_labels = train_dataset.tensors
    test_inputs, test_labels = test_dataset.tensors

    decision_outputs = {
        'xx': xx,
        'yy': yy,
        'preds': preds,
        'train_inputs': train_inputs,
        'train_labels': train_labels,
        'test_inputs': test_inputs,
        'test_labels': test_labels
    }

    # Save outputs for future use
    with open(cache_file, 'wb') as f:
        pickle.dump(decision_outputs, f)

    return decision_outputs

 


def plot_decision_boundary(decision_outputs, title,  animate=False, fig=None):
    if fig is None:
        fig = plt.figure()
    else:
        fig.clf()  # Clear only this figure if reusing

    ax = fig.add_subplot(111)

    xx,yy,preds = decision_outputs['xx'],decision_outputs['yy'],decision_outputs['preds']
  
    ax.contourf(xx.numpy(), yy.numpy(), preds.numpy(), levels=1, cmap="coolwarm", alpha=0.8)
  
    train_inputs,train_labels = decision_outputs['train_inputs'],decision_outputs['train_labels'] 
    ax.scatter(train_inputs[:, 0], train_inputs[:, 1], c=train_labels.squeeze(), cmap="coolwarm",
                marker='o', edgecolor='k', label="Train", s=40, alpha=0.8)
    test_inputs,test_labels = decision_outputs['test_inputs'],decision_outputs['test_labels'] 
    ax.scatter(test_inputs[:, 0], test_inputs[:, 1], c=test_labels.squeeze(), cmap="coolwarm",
                marker='o', edgecolor='k', label="test", s=40, alpha=0.8)
 
    ax.set_title(title)
   

 

    if animate:
        plt.pause(0.1)
    else:
        plt.show()

    return fig

# CLI interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", nargs="+", type=int, required=True, help="Architecture (e.g. --arch 128 64)")
    parser.add_argument("--epochs",  required = True, help="Checkpoint epoch to use")
    parser.add_argument("--plot", action="store_true", help="Display 3D plot")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["two_blobs_separate", "two_blobs_overlap",
                                 "half_moons", "spirals", "four_corners", "all"],
                        help="Dataset name")

    args = parser.parse_args()
    
    print(args)

    if args.epochs.startswith('*'):
        epochs = range(1,int(args.epochs.strip('*')))
    else:
        epochs = [int(args.epochs)]
    
    
    if args.dataset == 'all':
        datasets = ["two_blobs_separate", "two_blobs_overlap",    "half_moons", "spirals", "four_corners"]
    else:
        datasets = [args.dataset]
         
    for dataset in datasets:
        fig = None
        for epoch in epochs:
            decision_outputs = compute_decision( args.arch,  dataset, train=True, epoch=epoch, ab_range=1.0, gridsize=10)
            if args.plot:
                fig = plot_decision_boundary(decision_outputs,  title='Decision boundary, ep%d' % epoch, animate=epoch < epochs[-1], fig=fig)
