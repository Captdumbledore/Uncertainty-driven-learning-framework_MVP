"""
mixed_replay.py
---------------
Protocol 3: Mixed Replay

Every mini-batch is explicitly constructed to contain a configurable
ratio of original-to-curated samples. Unlike the Replay Buffer
(stochastic injection), this guarantees exact proportional
representation in every batch.

Research Question:
Does balanced replay outperform replaying only difficult samples?
"""

import random as pyrandom
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from akrm.protocols.base import LearningProtocol
from retrain import TensorLabelDataset


class MixedReplay(LearningProtocol):
    """
    Constructs every mini-batch with a guaranteed ratio of
    original-to-curated samples.
    """

    def __init__(self, curated_ratio: float = 0.2, **kwargs):
        """
        Parameters
        ----------
        curated_ratio : Fraction of each batch that should be curated samples.
                        Default 0.2 means every batch is 80% original, 20% curated.
        """
        super().__init__(**kwargs)
        self.curated_ratio = curated_ratio

    def update(
        self,
        model: nn.Module,
        original_dataset: Dataset,
        curated_dataset: Dataset,
        val_loader: DataLoader,
    ) -> Tuple[nn.Module, dict]:

        retrained = self._make_copy(model)

        n_curated_per_batch = max(1, int(self.batch_size * self.curated_ratio))
        n_original_per_batch = self.batch_size - n_curated_per_batch

        original_loader = DataLoader(
            TensorLabelDataset(original_dataset),
            batch_size=n_original_per_batch, shuffle=True,
            num_workers=0, drop_last=True
        )
        curated_wrapped = TensorLabelDataset(curated_dataset)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(retrained.parameters(), lr=self.lr)

        history = {"train_loss": [], "train_acc": [],
                   "val_loss": [], "val_acc": []}

        print(f"    [Mixed Replay] Batch composition: "
              f"{n_original_per_batch} original + {n_curated_per_batch} curated "
              f"(ratio={self.curated_ratio})")

        for epoch in range(1, self.epochs + 1):
            retrained.train()
            train_loss = 0.0
            correct = total = 0

            for orig_images, orig_labels in original_loader:
                # Sample curated portion
                cur_indices = [pyrandom.randint(0, len(curated_wrapped) - 1)
                               for _ in range(n_curated_per_batch)]
                cur_imgs = []
                cur_lbls = []
                for idx in cur_indices:
                    img, lbl = curated_wrapped[idx]
                    cur_imgs.append(img)
                    cur_lbls.append(lbl)

                cur_imgs = torch.stack(cur_imgs)
                cur_lbls = torch.stack(cur_lbls)

                # Combine into a single mixed batch
                images = torch.cat([orig_images, cur_imgs], dim=0)
                labels = torch.cat([orig_labels, cur_lbls], dim=0)

                # Shuffle the combined batch
                perm = torch.randperm(images.size(0))
                images, labels = images[perm], labels[perm]

                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = retrained(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

            t_loss = train_loss / len(original_loader)
            t_acc = correct / total
            v_loss, v_acc = self._evaluate(retrained, val_loader)

            history["train_loss"].append(t_loss)
            history["train_acc"].append(t_acc)
            history["val_loss"].append(v_loss)
            history["val_acc"].append(v_acc)

            print(f"    Epoch {epoch:>2}/{self.epochs}  "
                  f"| Train Acc {t_acc:.4f}  | Val Acc {v_acc:.4f}")

        return retrained, history
