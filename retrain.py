"""
retrain.py
----------
Offline retraining — Phase 5.

Combines the full original training dataset with either:
  - an augmented TensorDataset (Pipelines B and D), or
  - an AKRM experience pool Subset (Pipeline C)
then runs a small number of additional training epochs starting from
the baseline model's saved weights.

Design notes
------------
  - A deep copy of the baseline model is made to avoid modifying the
    original.  Pipelines B, C, and D therefore all start from exactly
    the same initial weights.
  - ConcatDataset ensures original samples are re-seen alongside the
    new experience pool.  The combined loader is shuffled so the model
    does not overfit to the ordering.
  - This function is intentionally identical for all three retraining
    pipelines.  The only variation is the experience_pool passed in.

Type-compatibility note
-----------------------
  torchvision Subsets return labels as plain Python ints while
  TensorDataset returns labels as LongTensors.  PyTorch's default
  collate function cannot mix the two types in one batch.
  TensorLabelDataset wraps any dataset so that its labels are always
  returned as LongTensors, making it safe to concatenate with a
  TensorDataset.
"""

import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset


class TensorLabelDataset(Dataset):
    """
    Thin wrapper around any dataset whose __getitem__ returns (image, label).

    Converts the label to a torch.LongTensor so that it can be safely
    concatenated with a TensorDataset (which also returns tensor labels)
    inside a ConcatDataset without causing collation errors.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        return image, label


def retrain_model(
    model,
    original_train_dataset,
    augmented_dataset,
    val_loader,
    epochs:     int   = 8,
    lr:         float = 0.001,
    batch_size: int   = 64,
    device            = "cpu",
):
    """
    Offline retraining: original training data + experience pool.

    Parameters
    ----------
    model                  : Baseline model — deepcopy is made internally
    original_train_dataset : Full original training Subset (not modified)
    augmented_dataset      : Either a TensorDataset of augmented images
                             (Pipelines B/D) or an AKRM experience pool
                             Subset (Pipeline C).  Both are accepted.
    val_loader             : Validation DataLoader (monitoring only —
                             never used for selection or retraining)
    epochs                 : Retraining epochs (default 8)
    lr                     : Adam learning rate
    batch_size             : Mini-batch size for the combined loader
    device                 : torch.device or string

    Returns
    -------
    retrained_model : Updated model after offline retraining
    history         : Dict with per-epoch train/val metrics
    """
    retrained_model = copy.deepcopy(model).to(device)

    # Both datasets are wrapped with TensorLabelDataset so that labels are
    # always LongTensors regardless of whether the input is a TensorDataset
    # (returns tensor labels) or a torchvision/AKRM Subset (returns int labels).
    wrapped_train = TensorLabelDataset(original_train_dataset)
    wrapped_pool  = TensorLabelDataset(augmented_dataset)
    combined_dataset = ConcatDataset([wrapped_train, wrapped_pool])
    combined_loader  = DataLoader(
        combined_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(retrained_model.parameters(), lr=lr)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    print(
        f"    Combined dataset: {len(original_train_dataset):,} original "
        f"+ {len(augmented_dataset):,} augmented = {len(combined_dataset):,} total"
    )

    for epoch in range(1, epochs + 1):

        # ── Training ──────────────────────────────────────────────────────
        retrained_model.train()
        train_loss = 0.0
        correct = total = 0

        for images, labels in combined_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = retrained_model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds    = outputs.max(1)
            correct    += preds.eq(labels).sum().item()
            total      += labels.size(0)

        # ── Validation ────────────────────────────────────────────────────
        retrained_model.eval()
        val_loss = 0.0
        val_correct = val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs      = retrained_model(images)
                loss         = criterion(outputs, labels)
                val_loss    += loss.item()
                _, preds     = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total   += labels.size(0)

        t_loss = train_loss / len(combined_loader)
        t_acc  = correct    / total
        v_loss = val_loss   / len(val_loader)
        v_acc  = val_correct / val_total

        history["train_loss"].append(t_loss)
        history["train_acc" ].append(t_acc)
        history["val_loss"  ].append(v_loss)
        history["val_acc"   ].append(v_acc)

        print(
            f"    Epoch {epoch:>2}/{epochs}  "
            f"| Train Acc {t_acc:.4f}  | Val Acc {v_acc:.4f}"
        )

    return retrained_model, history
