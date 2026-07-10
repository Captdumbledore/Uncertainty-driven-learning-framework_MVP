"""
replay_buffer.py
----------------
Protocol 2: Replay Buffer

Trains on the full original dataset, but injects curated samples
into random batch positions at a fixed probability. Curated samples
are cycled through a replay buffer, ensuring they appear throughout
training rather than as a separate concatenated block.

Research Question:
Can replay reduce catastrophic forgetting compared to offline retraining?
"""

import random as pyrandom
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from akrm.protocols.base import LearningProtocol
from retrain import TensorLabelDataset


class ReplayBuffer(LearningProtocol):
    """
    Trains on original data with stochastic injection of curated samples.
    Each batch has a configurable probability of replacing some samples
    with curated replay samples.
    """

    def __init__(self, replay_ratio: float = 0.2, **kwargs):
        """
        Parameters
        ----------
        replay_ratio : Fraction of each batch to replace with curated samples.
                       Default 0.2 means ~20% of each batch comes from replay.
        """
        super().__init__(**kwargs)
        self.replay_ratio = replay_ratio

    def update(
        self,
        model: nn.Module,
        original_dataset: Dataset,
        curated_dataset: Dataset,
        val_loader: DataLoader,
    ) -> Tuple[nn.Module, dict]:

        retrained = self._make_copy(model)

        original_loader = DataLoader(
            TensorLabelDataset(original_dataset),
            batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        curated_wrapped = TensorLabelDataset(curated_dataset)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(retrained.parameters(), lr=self.lr)

        history = {"train_loss": [], "train_acc": [],
                   "val_loss": [], "val_acc": []}

        n_replace = max(1, int(self.batch_size * self.replay_ratio))

        print(f"    [Replay Buffer] Training on {len(original_dataset):,} original, "
              f"injecting ~{n_replace} curated samples per batch "
              f"(ratio={self.replay_ratio})")

        for epoch in range(1, self.epochs + 1):
            retrained.train()
            train_loss = 0.0
            correct = total = 0

            for images, labels in original_loader:
                # Inject curated replay samples into this batch
                replay_indices = [pyrandom.randint(0, len(curated_wrapped) - 1)
                                  for _ in range(n_replace)]
                replay_imgs = []
                replay_lbls = []
                for idx in replay_indices:
                    img, lbl = curated_wrapped[idx]
                    replay_imgs.append(img)
                    replay_lbls.append(lbl)

                replay_imgs = torch.stack(replay_imgs)
                replay_lbls = torch.stack(replay_lbls)

                # Replace last n_replace samples in the batch with replay
                images = torch.cat([images[:-n_replace], replay_imgs], dim=0)
                labels = torch.cat([labels[:-n_replace], replay_lbls], dim=0)

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
