"""
data.py
-------
Dataset loading and splitting for MNIST / Fashion-MNIST / CIFAR-10.

Responsibilities
----------------
  - Download and normalise the dataset.
  - Deterministically split the training set into train / validation subsets
    using a fixed generator seed so every run reproduces the same split.
  - Return DataLoaders and the raw train_dataset Subset (passed to the
    uncertainty analysis phase to select augmentation candidates).

IMPORTANT
---------
  The validation and test DataLoaders returned here are NEVER used for
  augmentation or retraining.  They are evaluation-only.
"""

import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Per-channel normalisation statistics (precomputed from the training sets)
STATS = {
    "FashionMNIST": {"mean": (0.2860,),          "std": (0.3530,)},
    "MNIST":        {"mean": (0.1307,),          "std": (0.3081,)},
    "CIFAR10":      {"mean": (0.4914, 0.4822, 0.4465),
                     "std":  (0.2023, 0.1994, 0.2010)},
}

# Dataset metadata (used by main.py to configure model architecture)
DATASET_META = {
    "FashionMNIST": {"in_channels": 1, "image_size": 28},
    "MNIST":        {"in_channels": 1, "image_size": 28},
    "CIFAR10":      {"in_channels": 3, "image_size": 32},
}


def get_dataloaders(
    dataset_name: str = "FashionMNIST",
    batch_size:   int   = 64,
    val_split:    float = 0.10,
    seed:         int   = 42,
    data_root:    str   = "./data",
):
    """
    Build DataLoaders for train / val / test splits.

    Parameters
    ----------
    dataset_name : "FashionMNIST" or "MNIST"
    batch_size   : Mini-batch size for all loaders
    val_split    : Fraction of training data reserved for validation
    seed         : Controls the train/val split (reproducible across runs)
    data_root    : Local folder where the raw dataset is cached

    Returns
    -------
    train_loader    : DataLoader  – shuffled training batches
    val_loader      : DataLoader  – ordered validation batches
    test_loader     : DataLoader  – ordered test batches
    train_dataset   : Subset     – raw training subset (for uncertainty analysis)
    """
    if dataset_name not in STATS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            "Choose 'FashionMNIST', 'MNIST', or 'CIFAR10'."
        )

    mean, std = STATS[dataset_name]["mean"], STATS[dataset_name]["std"]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if dataset_name == "CIFAR10":
        # Support both:
        #   (a) ImageFolder layout: data/CIFAR-10-images-master/train/<class>/*.jpg
        #   (b) Torchvision pickle layout (auto-download)
        img_root = os.path.join(data_root, "CIFAR-10-images-master")
        if os.path.isdir(img_root):
            # (a) Load from pre-extracted image folders (Kaggle format)
            full_train = datasets.ImageFolder(
                root=os.path.join(img_root, "train"), transform=transform
            )
            test_data  = datasets.ImageFolder(
                root=os.path.join(img_root, "test"),  transform=transform
            )
        else:
            # (b) Fall back to torchvision auto-download (requires internet)
            full_train = datasets.CIFAR10(
                root=data_root, train=True,  download=True, transform=transform
            )
            test_data  = datasets.CIFAR10(
                root=data_root, train=False, download=True, transform=transform
            )
    else:
        Dataset    = getattr(datasets, dataset_name)
        full_train = Dataset(root=data_root, train=True,  download=True, transform=transform)
        test_data  = Dataset(root=data_root, train=False, download=True, transform=transform)

    n_val   = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train, [n_train, n_val], generator=generator
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )

    print(f"  Dataset     : {dataset_name}")
    print(f"  Train size  : {n_train:,}")
    print(f"  Val size    : {n_val:,}")
    print(f"  Test size   : {len(test_data):,}")

    return train_loader, val_loader, test_loader, train_dataset
