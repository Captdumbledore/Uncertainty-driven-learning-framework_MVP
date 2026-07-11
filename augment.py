"""
augment.py
----------
Targeted data augmentation — Phase 4.

Only the selected uncertain (or random/least-uncertain) training samples
are augmented.  The rest of the training data is used as-is.

Augmentations used (torchvision.transforms.functional only):
  1. Random rotation        ± 15°
  2. Random translation     ± 10% of image size (pixel-level)
  3. Brightness jitter      factor ∈ [0.8, 1.2]
  4. Gaussian noise         σ = 0.05

Each selected sample produces n_aug_per_image new images.
Returns a TensorDataset that is concatenated with the original training
data inside retrain.py.
"""

import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import TensorDataset


def augment_samples(
    dataset,
    indices:         np.ndarray,
    n_aug_per_image: int = 3,
    seed:            int = 42,
) -> TensorDataset:
    """
    Generate augmented copies of selected training samples.

    Parameters
    ----------
    dataset          : Training Subset — supports dataset[i] → (image, label)
    indices          : 1-D array of integer indices into `dataset`
    n_aug_per_image  : Number of augmented copies produced per selected sample
    seed             : Random seed for reproducible augmentations

    Returns
    -------
    TensorDataset containing (augmented_images, labels) pairs.
    Shape: ( len(indices) * n_aug_per_image, 1, 28, 28 )
    """
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    aug_images: list = []
    aug_labels: list = []

    for idx in indices:
        image, label = dataset[int(idx)]   # (1, 28, 28), scalar

        for _ in range(n_aug_per_image):
            img = image.clone()

            # 1. Random rotation ±15°
            angle = py_rng.uniform(-15.0, 15.0)
            img   = TF.rotate(img, angle)

            # 2. Random translation ±10% of image width
            max_shift = max(1, int(0.10 * img.shape[-1]))   # 2 px for 28×28
            tx = py_rng.randint(-max_shift, max_shift)
            ty = py_rng.randint(-max_shift, max_shift)
            img = TF.affine(
                img, angle=0.0, translate=[tx, ty], scale=1.0, shear=0.0
            )

            # 3. Brightness jitter
            brightness_factor = py_rng.uniform(0.8, 1.2)
            img = TF.adjust_brightness(img, brightness_factor)

            # 4. Gaussian noise (σ = 0.05)
            noise = torch.tensor(
                np_rng.normal(0.0, 0.05, img.shape), dtype=torch.float32
            )
            img = img + noise

            aug_images.append(img)
            aug_labels.append(int(label))

    aug_images_t = torch.stack(aug_images)                      # (M, 1, 28, 28)
    aug_labels_t = torch.tensor(aug_labels, dtype=torch.long)   # (M,)

    n_generated = len(aug_images)
    print(
        f"    Augmented {len(indices):,} samples × {n_aug_per_image} "
        f"= {n_generated:,} new training images"
    )
    return TensorDataset(aug_images_t, aug_labels_t)
