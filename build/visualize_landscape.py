# visualize.py
import torch
import torch.nn as nn
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import BlobMLP, arch_to_name, get_model_path
from train_eval_model import latest_checkpoint
from data import load_dataset

CACHE_ROOT = "../landscapes"
MODELS_DIR = "../models"
 
def compute_loss(model: nn.Module, dataset_name: str, train=True) -> float:
    """Compute average BCE loss of a model on train or test set."""
    dataset = load_dataset(dataset_name, train=train)
    loader = DataLoader(dataset, batch_size=1000, shuffle=False)

    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction='sum')  # Sum losses first

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            total_samples += labels.size(0)

    return total_loss / total_samples


def generate_random_directions(arch):
    
    # Create and normalize random directions
    def random_direction():
        rand_model = BlobMLP(hidden_layers=arch)
        for param in rand_model.parameters():
            param.data = torch.randn_like(param)
        return rand_model

    x = random_direction()
    y = random_direction()
    return x,y
 

def load_direction(path, arch):
    model = BlobMLP(hidden_layers=arch)
    model.load_state_dict(torch.load(path))
    return model

def combine_models(base, dx, dy, a, b,arch):
    new_model = BlobMLP(hidden_layers=arch)

    for p, p_base, p_x, p_y in zip(new_model.parameters(), base.parameters(), dx.parameters(), dy.parameters()):
        p.data = p_base.data + a * p_x.data + b * p_y.data
    return new_model
 


def compute_loss_grid( arch,  dataset, train=True, epoch=1, ab_range=1.0, gridsize=10):

    train_str = 'train' if train else 'test'
    base_path = os.path.join(MODELS_DIR,  arch_to_name(arch), dataset,   f"ep{epoch}.pth")
    cache_dir = os.path.join(CACHE_ROOT, arch_to_name(arch), dataset, train_str, f"range{ab_range}",  f"ep{epoch}")
    cache_xy_dir = os.path.join(CACHE_ROOT, arch_to_name(arch))
    

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(cache_xy_dir, exist_ok=True)
    if not os.path.exists(os.path.join(cache_xy_dir, "x.pt")):
        x,y = generate_random_directions(arch)
        torch.save(x.state_dict(), os.path.join(cache_xy_dir, "x.pt"))
        torch.save(y.state_dict(), os.path.join(cache_xy_dir, "y.pt"))

      
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
    base_model = BlobMLP(hidden_layers=arch)
    base_model.load_state_dict(torch.load(base_path))
    x = load_direction(os.path.join(cache_xy_dir, "x.pt"), arch)
    y = load_direction(os.path.join(cache_xy_dir, "y.pt"), arch)

    # Compute missing values
    for i, aa in enumerate(a):
        for j, bb in enumerate(b): 
            perturbed_model = combine_models(base_model, x, y, aa, bb, arch)
            loss = compute_loss(perturbed_model, dataset, train=False)
            losses[i, j] = loss
     
    # Save updated grid
    np.save(loss_file,losses)
    np.save(a_file, a)
    np.save(b_file, b)

    print(f"✅ Updated and saved new grid at {cache_dir}")
    return a, b, losses


def compute_and_save_zrange(arch, dataset, train=True, ab_range=1.0, gridsize=10):
    """Compute global [zmin, zmax] across all available epochs and save it."""
    train_str = 'train' if train else 'test'
    cache_base_dir = os.path.join(CACHE_ROOT, arch_to_name(arch), dataset, train_str, f"range{ab_range}")
    zrange_file = os.path.join(cache_base_dir, f"zrange.npy")


    # --- If zrange already exists, skip ---
    if os.path.exists(zrange_file):
        print(f"✅ zrange already computed at {zrange_file}. Skipping computation.")
        return

    if not os.path.exists(cache_base_dir):
        print(f"⚠️ No cache directory found at {cache_base_dir}")
        return


    epoch_max_losses = []

    # Go through all saved epochs
    for ep_folder in sorted(os.listdir(cache_base_dir)):

        
        # Parse epoch number
        try:
            epoch_num = int(ep_folder.replace("ep", ""))
        except ValueError:
            continue  # Skip non-epoch folders

        if epoch_num > 250:
            continue  # Only consider epochs <= 250

        epoch_dir = os.path.join(cache_base_dir, ep_folder)
        loss_file = os.path.join(epoch_dir, "loss.npy")

        losses = np.load(loss_file)
        epoch_max_losses.append(losses.max())
        print(ep_folder)



    if not epoch_max_losses:
        print(f"❌ No loss files found to compute zrange in {cache_base_dir}")
        return


    # --- Now do robust zrange calculation ---
    zmin = 0.0
    sorted_max_losses = np.sort(epoch_max_losses)
    percentile_index = int(0.8 * len(sorted_max_losses)) - 1
    zmax = sorted_max_losses[percentile_index]

    # Save it
    np.save(zrange_file, np.array([zmin, zmax]))
    print(f"✅ Saved robust zrange [0.0, {zmax:.4f}] to {zrange_file}")


def plot_surface(a_vals, b_vals, losses, title="Loss Landscape", animate=False, fig=None):
    A, B = np.meshgrid(a_vals, b_vals, indexing='ij')
    if fig is None:
        fig = plt.figure()
    else:
        fig.clf()  # Clear only this figure if reusing

    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(A, B, losses, cmap='viridis')
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('Loss')
    ax.set_title(title)




    ax.plot([0, 0], [0, 0], [losses.min(), losses.max()], color='red', linestyle='-', linewidth=2, label='(0,0)', zorder=10)


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
    parser.add_argument("--gridsize",  help="Grid size", type=int)
    parser.add_argument("--plot", action="store_true", help="Display 3D plot")
    parser.add_argument("--range", type = float, default="1.0", help="a,b range (e.g. [-range, range])")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["two_blobs_separate", "two_blobs_overlap",
                                 "half_moons", "spirals", "four_corners", "all"],
                        help="Dataset name")
    parser.add_argument("--train", action="store_true", help="Train the model", default=False) 
    parser.add_argument("--save_zrange", action="store_true", help="Compute and save global z-range for given settings")

    args = parser.parse_args()
    
    print(args)

    if args.epochs.startswith('*'):
        epochs = range(1,int(args.epochs.strip('*')))
    else:
        epochs = [int(args.epochs)]
    
    
    if args.gridsize is None:
        gridsize = 10
    else:
        gridsize = args.gridsize
    
    if args.dataset == 'all':
        datasets = ["two_blobs_separate", "two_blobs_overlap",    "half_moons", "spirals", "four_corners"]
    else:
        datasets = [args.dataset]
        

    for ranges in [.1,.01,1]:
        for width in [10,25,50,100]:
            
            for depth in [1,2,3,4]:
                arch = [width for k in range(depth)]
                
                for dataset in datasets:
                    fig = None
                    for epoch in epochs:
                        a_vals, b_vals, losses = compute_loss_grid(arch,  dataset, train=args.train, epoch=epoch, ab_range=ranges, gridsize=args.gridsize)
                
                    if args.save_zrange:
                        print('compute_and_save_zrange...')
                        compute_and_save_zrange(
                            arch=arch,
                            dataset=dataset,
                            train=args.train,
                            ab_range=ranges,
                            gridsize=args.gridsize
                        )
                        print('done.')
                    