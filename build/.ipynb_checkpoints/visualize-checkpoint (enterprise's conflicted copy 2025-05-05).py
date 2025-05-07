# visualize.py

import torch
import os
import itertools
import argparse
import numpy as np
import matplotlib.pyplot as plt
from model import MNIST_MLP
from loss import compute_loss
from model import MNIST_MLP, model_filename, latest_checkpoint

CACHE_ROOT = "../landscapes"
MODELS_DIR = "../models"
def get_model_id(arch):
    return "mlp_" + "_".join(str(x) for x in arch)

def generate_random_directions(arch, instance_name, model_id):
    """Create random x/y directions and save them."""
    save_dir = os.path.join(CACHE_ROOT, model_id, instance_name)
    os.makedirs(save_dir, exist_ok=True)

    # Create and normalize random directions
    def random_direction():
        rand_model = MNIST_MLP(hidden_layers=arch)
        for param in rand_model.parameters():
            param.data = torch.randn_like(param)
        return rand_model

    x = random_direction()
    y = random_direction()

    torch.save(x.state_dict(), os.path.join(save_dir, "x.pt"))
    torch.save(y.state_dict(), os.path.join(save_dir, "y.pt"))

    print(f"✅ Random directions saved in {save_dir}")
    return save_dir

def load_direction(path,arch):
    model = MNIST_MLP(hidden_layers=arch)

    model.load_state_dict(torch.load(path))
    return model

def combine_models(base, dx, dy, a, b,arch):
    new_model = MNIST_MLP(hidden_layers=arch)

    for p, p_base, p_x, p_y in zip(new_model.parameters(), base.parameters(), dx.parameters(), dy.parameters()):
        p.data = p_base.data + a * p_x.data + b * p_y.data
    return new_model

#in this version, do not change grid size or ab range.
def compute_loss_grid(base_path, arch, model_id, instance_name, epoch, ab_range=1.0, gridsize=10):

    cache_dir = os.path.join(CACHE_ROOT, model_id, instance_name, f"ep{epoch}", f"range{ab_range}")
    cache_model_dir = os.path.join(CACHE_ROOT, model_id, instance_name)
    os.makedirs(cache_dir, exist_ok=True)

    loss_file = os.path.join(cache_dir, "loss.npy")
    a_file = os.path.join(cache_dir, "a_vals.npy")
    b_file = os.path.join(cache_dir, "b_vals.npy")

    

    # Load or initialize
    acand = np.linspace(-ab_range, ab_range, gridsize)
    if os.path.exists(loss_file) and os.path.exists(a_file) and os.path.exists(b_file):
        a = np.load(a_file)
        if len(a) < len(acand):
            print('Previously saved model is of lower resolution; recomputing')
        else:
            a = np.load(a_file)
            b = np.load(b_file)
            losses = np.load(loss_file)
            print(f"📂 Loaded cached grid {losses.shape} from {cache_dir}")
            return a, b, losses
    

    # New request
    a = np.linspace(-ab_range, ab_range, gridsize)
    b = np.linspace(-ab_range, ab_range, gridsize)
 
    # Create full new grid
    losses = torch.full((gridsize, gridsize), float('nan'))
    
      
    # Load model and directions
    base_model = MNIST_MLP(hidden_layers=arch)
    base_model.load_state_dict(torch.load(base_path))
    x = load_direction(os.path.join(cache_model_dir, "x.pt"), arch)
    y = load_direction(os.path.join(cache_model_dir, "y.pt"), arch)

    # Compute missing values
    for i, aa in enumerate(a):
        for j, bb in enumerate(b): 
            perturbed_model = combine_models(base_model, x, y, aa, bb, arch)
            loss = compute_loss(perturbed_model, train=False)
            losses[i, j] = loss
    print(a_file, b_file, loss_file)
    
    # Save updated grid
    np.save(loss_file,losses)
    np.save(a_file, a)
    np.save(b_file, b)

    print(f"✅ Updated and saved new grid at {cache_dir}")
    return a, b, losses


def plot_surface(a_vals, b_vals, losses, title="Loss Landscape"):
    A, B = np.meshgrid(a_vals, b_vals, indexing='ij')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(A, B, losses, cmap='viridis')
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('Loss')
    ax.set_title(title)
    plt.show()

# CLI interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", nargs="+", type=int, required=True, help="Architecture (e.g. --arch 128 64)")
    parser.add_argument("--epochs",  help="Checkpoint epoch to use")
    parser.add_argument("--gridsize",  help="Grid size", type=int)
    parser.add_argument("--regen", action="store_true", help="Force generate new random directions")
    parser.add_argument("--plot", action="store_true", help="Display 3D plot")
    parser.add_argument("--range", nargs = "+", type = float, default="1.0", help="a,b range (e.g. [-range, range])")
    parser.add_argument("--clean", action="store_true", help="Remove all cached random directions")

    args = parser.parse_args()
    
    if args.clean:
        if not os.path.exists(CACHE_ROOT):
            print("📁 visualize_cache/ does not exist.")
        else:
            deleted = False
            for subdir in os.listdir(CACHE_ROOT):
                full_path = os.path.join(CACHE_ROOT, subdir)
                if os.path.isdir(full_path):
                    for subsub in os.listdir(full_path):
                        if subsub.startswith("random_instance"):
                            path = os.path.join(full_path, subsub)
                            shutil.rmtree(path)
                            print(f"🧹 Removed {path}")
                            deleted = True
            if not deleted:
                print("📂 No random_instance directories found.")
        exit(0)
    
    if args.gridsize is None:
        gridsize = 10
    else:
        gridsize = args.gridsize
    # Resolve checkpoint path
    if args.epochs is not None:
        print(args.epochs)
        if str(args.epochs).startswith("*"):
            max_epoch = int(str(args.epochs)[1:])
            epoch_range = range(1, max_epoch + 1)
        else:
            epoch_range = [int(args.epochs)]
    else:
        # fallback to latest checkpoint
        base_path, used_epoch = latest_checkpoint(args.arch, models_dir=MODELS_DIR)
        if base_path is None:
            print("❌ No checkpoint found for this architecture.")
            exit(1)
        epoch_range = [used_epoch]
 
 
    # Create a new instance if --regen
    model_id = get_model_id(args.arch)

    # Auto-generate new instance number
    
    if args.regen:
        i = 1
        while os.path.exists(os.path.join(CACHE_ROOT, model_id, f"random_instance_{i}")):
            i += 1
        instance = f"random_instance_{i}"
    else:
    
        instance = "random_instance_1"

    cache_path = os.path.join(CACHE_ROOT, model_id, instance)
    if args.regen or not os.path.exists(os.path.join(cache_path, "x.pt")):
        generate_random_directions(args.arch, instance, model_id)

    for epoch in epoch_range:
        base_path = model_filename(args.arch, epoch, models_root=MODELS_DIR)
        if not os.path.exists(base_path):
            print(f"❌ Checkpoint not found for epoch {epoch}: {base_path}")
            continue

         
        for drange in args.range:
            print(f"📊 Processing epoch {epoch}, disc {drange}...")
            a_vals, b_vals, losses = compute_loss_grid(base_path, args.arch, model_id, instance, epoch, drange, gridsize)

            if args.plot:
                plot_surface(a_vals, b_vals, losses, title=f"Loss Landscape (epoch {epoch})")
 