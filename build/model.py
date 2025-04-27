# model.py

import torch
import torch.nn as nn
import os

MODELS_DIR = "../models"  # Base directory for model checkpoints

def arch_to_name(arch):
    """Convert architecture list to a string name like 'mlp_64_64'."""
    return "mlp_" + "_".join(str(x) for x in arch)

def get_model_path(arch, epochs, dataset_name):
    """Get full path for a model checkpoint."""
    arch_name = arch_to_name(arch)
    model_dir = os.path.join(MODELS_DIR, arch_name, dataset_name)
    model_name = f"ep{epochs}.pth"
    return os.path.join(model_dir, model_name)

class BlobMLP(nn.Module):
    """Simple MLP for 2D input blobs."""
    def __init__(self, hidden_layers=[128]):
        super().__init__()
        layers = []
        input_dim = 2
        for h in hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))  # Binary output
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.view(-1, 2))
