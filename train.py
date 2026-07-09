"""
train.py
--------
Standard supervised training loop — Phase 1.

Responsibilities
----------------
  - Train SimpleCNN on the training DataLoader using cross-entropy + Adam.
  - Track per-epoch train/val loss and accuracy.
  - Print a concise progress line each epoch.

Returns the trained model and a history dict used later for plotting.
"""

import torch
import torch.nn as nn
import torch.optim as optim


def train_model(
    model:        nn.Module,
    train_loader,
    val_loader,
    epochs:       int   = 15,
    lr:           float = 0.001,
    device              = "cpu",
):
    """
    Standard supervised training.

    Parameters
    ----------
    model        : Uninitialised SimpleCNN (moved to device internally)
    train_loader : DataLoader for the training subset
    val_loader   : DataLoader for the validation subset (monitoring only —
                   NEVER used for augmentation or retraining)
    epochs       : Number of training epochs
    lr           : Adam learning rate
    device       : torch.device or string ("cpu" / "cuda")

    Returns
    -------
    model   : Trained model (on the same device)
    history : Dict with keys "train_loss", "train_acc", "val_loss", "val_acc"
              Each value is a list of per-epoch floats.
    """
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    for epoch in range(1, epochs + 1):

        # ── Training pass ────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        correct = total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds    = outputs.max(1)
            correct    += preds.eq(labels).sum().item()
            total      += labels.size(0)

        # ── Validation pass ──────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs      = model(images)
                loss         = criterion(outputs, labels)
                val_loss    += loss.item()
                _, preds     = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total   += labels.size(0)

        # ── Record ───────────────────────────────────────────────────────
        t_loss = train_loss / len(train_loader)
        t_acc  = correct    / total
        v_loss = val_loss   / len(val_loader)
        v_acc  = val_correct / val_total

        history["train_loss"].append(t_loss)
        history["train_acc" ].append(t_acc)
        history["val_loss"  ].append(v_loss)
        history["val_acc"   ].append(v_acc)

        print(
            f"    Epoch {epoch:>2}/{epochs}  "
            f"| Train Loss {t_loss:.4f}  Acc {t_acc:.4f}"
            f"  | Val Loss {v_loss:.4f}  Acc {v_acc:.4f}"
        )

    return model, history
