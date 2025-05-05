# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

MODELS_DIR = "../models"  # You can override this in scripts if needed

def get_arch_folder(arch, prefix="mlp"):
    arch_str = "_".join(str(x) for x in arch)
    return f"{prefix}_{arch_str}"

def model_filename(arch, epoch, models_root="../models", prefix="mlp"):
    subfolder = get_arch_folder(arch, prefix=prefix)
    filename = f"{subfolder}_ep{epoch}.pth"
    return os.path.join(models_root, subfolder, filename)


def latest_checkpoint(arch, models_root="../models", prefix="mlp"):
    subfolder = get_arch_folder(arch, prefix=prefix)
    dir_path = os.path.join(models_root, subfolder)
    if not os.path.exists(dir_path):
        return None, 0
    files = [f for f in os.listdir(dir_path) if f.startswith(subfolder)]
    if not files:
        return None, 0
    files.sort(key=lambda f: int(f.split("_ep")[-1].split(".")[0]))
    latest = files[-1]
    epoch = int(latest.split("_ep")[-1].split(".")[0])
    return os.path.join(dir_path, latest), epoch



class Blob_MLP(nn.Module):
    def __init__(self, hidden_layers=[128]):
        """
        Args:
            hidden_layers (list[int]): Sizes of hidden layers.
        """
        super().__init__()
        layers = []

        input_dim = 2
        for h in hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h

        layers.append(nn.Linear(input_dim, 1))  # Output layer

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 2)
        return self.network(x)