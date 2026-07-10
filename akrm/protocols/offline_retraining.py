"""
offline_retraining.py
---------------------
Protocol 1: Offline Retraining (Control)

Wraps the existing retrain.py logic in the LearningProtocol interface.
Concatenates original + curated datasets and trains for N epochs.
This serves as the experimental control for Stage 2.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from akrm.protocols.base import LearningProtocol
from retrain import TensorLabelDataset


class OfflineRetraining(LearningProtocol):
    """
    Control protocol: simple offline retraining on original + curated data.
    This is identical to the existing retrain_model() behavior.
    """

    def update(
        self,
        model: nn.Module,
        original_dataset: Dataset,
        curated_dataset: Dataset,
        val_loader: DataLoader,
    ) -> Tuple[nn.Module, dict]:

        retrained = self._make_copy(model)

        wrapped_train = TensorLabelDataset(original_dataset)
        wrapped_pool = TensorLabelDataset(curated_dataset)
        combined = ConcatDataset([wrapped_train, wrapped_pool])
        loader = DataLoader(combined, batch_size=self.batch_size,
                            shuffle=True, num_workers=0)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(retrained.parameters(), lr=self.lr)

        history = {"train_loss": [], "train_acc": [],
                   "val_loss": [], "val_acc": []}

        print(f"    [Offline Retraining] {len(original_dataset):,} original "
              f"+ {len(curated_dataset):,} curated = {len(combined):,} total")

        for epoch in range(1, self.epochs + 1):
            retrained.train()
            train_loss = 0.0
            correct = total = 0

            for images, labels in loader:
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

            t_loss = train_loss / len(loader)
            t_acc = correct / total
            v_loss, v_acc = self._evaluate(retrained, val_loader)

            history["train_loss"].append(t_loss)
            history["train_acc"].append(t_acc)
            history["val_loss"].append(v_loss)
            history["val_acc"].append(v_acc)

            print(f"    Epoch {epoch:>2}/{self.epochs}  "
                  f"| Train Acc {t_acc:.4f}  | Val Acc {v_acc:.4f}")

        return retrained, history
