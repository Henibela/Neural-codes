"""
setup_local_folders.py
======================
Run this ONCE on your local machine after cloning the repo.
Creates the results/local_cpu/ and checkpoints/local_cpu/ folder structure
that mirrors what Colab creates on Drive under results/colab_t4/ and checkpoints/colab_t4/.

Usage:
    python setup_local_folders.py

Run from inside your Neural-codes/ folder.
"""

import os

FOLDERS = [
    os.path.join('results',     'local_cpu'),
    os.path.join('checkpoints', 'local_cpu'),
]

print("Setting up local folder structure...")
print()

for folder in FOLDERS:
    os.makedirs(folder, exist_ok=True)
    print(f"  ✓ {folder}/")

print()
print("Done. Your local structure now mirrors Colab:")
print()
print("  Neural-codes/")
print("  ├── results/")
print("  │   ├── local_cpu/   ← local training results go here")
print("  │   └── colab_t4/   ← pushed here by upload_to_github.py")
print("  └── checkpoints/")
print("      ├── local_cpu/   ← local checkpoints go here")
print("      └── colab_t4/   ← pushed here by upload_to_github.py")
print()
print("Next: run  python ae_lite_train.py  to start training.")
