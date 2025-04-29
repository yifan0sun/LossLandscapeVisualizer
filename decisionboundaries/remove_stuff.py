import os
import shutil


def remove_files_with_prefix_suffix(root_path,prefix, suffix):
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.startswith(prefix) and filename.endswith(suffix):
                full_path = os.path.join(dirpath, filename)
                print(full_path)
                os.remove(full_path)

if __name__ == "__main__":
    # Change this to your root directory
    root_dir = "."          # Starting point
    prefix = "ep"
    suffix = "1.pkl"

    remove_files_with_prefix_suffix(root_dir, prefix, suffix)
