"""
curriculum_replay.py
--------------------
Protocol 4: Curriculum Replay

Introduces curated experiences gradually over three training phases:
  Phase 1 (epochs 1-3): 95% original, 5% curated (gentle introduction)
  Phase 2 (epochs 4-6): 80% original, 20% curated (moderate exposure)
  Phase 3 (epochs 7-8): 70% original, 30% curated (focused learning)

The phase schedule is configurable.

Research Question:
Does progressive integration improve stability?
"""

import random as pyrandom
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from akrm.protocols.base import LearningProtocol
from retrain import TensorLabelDataset


class CurriculumReplay(LearningProtocol):
    """
    Gradually increases the proportion of curated samples across training phases.
    """

    def __init__(self, schedule: List[Tuple[int, float]] = None, **kwargs):
        """
        Parameters
        ----------
        schedule : List of (last_epoch, curated_ratio) tuples.
                   Default: [(3, 0.05), (6, 0.20), (8, 0.30)]
                   Meaning: epochs 1-3 use 5%, epochs 4-6 use 20%, epochs 7-8 use 30%.
        """
        super().__init__(**kwargs)
        self.schedule = schedule or [(3, 0.05), (6, 0.20), (self.epochs, 0.30)]

    def _get_ratio(self, epoch: int) -> float:
        """Return the curated ratio for a given epoch."""
        for last_epoch, ratio in self.schedule:
            if epoch <= last_epoch:
                return ratio
        return self.schedule[-1][1]

    def update(
        self,
        model: nn.Module,
        original_dataset: Dataset,
        curated_dataset: Dataset,
        val_loader: DataLoader,
    ) -> Tuple[nn.Module, dict]:

        retrained = self._make_copy(model)

        curated_wrapped = TensorLabelDataset(curated_dataset)
        original_wrapped = TensorLabelDataset(original_dataset)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(retrained.parameters(), lr=self.lr)

        history = {"train_loss": [], "train_acc": [],
                   "val_loss": [], "val_acc": []}

        print(f"    [Curriculum Replay] Schedule: "
              + ", ".join(f"epochs ≤{e}: {r:.0%}" for e, r in self.schedule))

        for epoch in range(1, self.epochs + 1):
            ratio = self._get_ratio(epoch)
            n_curated_per_batch = max(1, int(self.batch_size * ratio))
            n_original_per_batch = self.batch_size - n_curated_per_batch

            original_loader = DataLoader(
                original_wrapped,
                batch_size=n_original_per_batch, shuffle=True,
                num_workers=0, drop_last=True
            )

            retrained.train()
            train_loss = 0.0
            correct = total = 0

            for orig_images, orig_labels in original_loader:
                # Sample curated portion for this batch
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

                images = torch.cat([orig_images, cur_imgs], dim=0)
                labels = torch.cat([orig_labels, cur_lbls], dim=0)

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
                  f"| Curated ratio {ratio:.0%}  "
                  f"| Train Acc {t_acc:.4f}  | Val Acc {v_acc:.4f}")

        return retrained, history
