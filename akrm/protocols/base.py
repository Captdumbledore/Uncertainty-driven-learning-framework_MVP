"""
base.py
-------
Abstract Base Class for all Learning Protocols.

Every learning strategy implements this interface, making them
perfectly swappable via a CLI flag without modifying any upstream
module (Diagnosis, Planner, Provider).
"""

import copy
from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class LearningProtocol(ABC):
    """
    Abstract interface for integrating curated experiences into a model.

    All protocols receive the same inputs:
      - model:            The baseline model (a deep copy is made internally).
      - original_dataset: The full original training dataset.
      - curated_dataset:  The curated experience pool from the AKRM pipeline.
      - val_loader:       Validation DataLoader (monitoring only).

    All protocols return:
      - updated_model:    The model after the learning protocol has been applied.
      - history:          Dict with per-epoch train/val metrics.
    """

    def __init__(self, epochs: int = 8, lr: float = 0.001,
                 batch_size: int = 64, device: str = "cpu"):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

    @abstractmethod
    def update(
        self,
        model: nn.Module,
        original_dataset: Dataset,
        curated_dataset: Dataset,
        val_loader: DataLoader,
    ) -> Tuple[nn.Module, dict]:
        """
        Integrate curated experiences into the model.

        Returns
        -------
        updated_model : nn.Module
        history       : dict with keys train_loss, train_acc, val_loss, val_acc
        """
        pass

    def _make_copy(self, model: nn.Module) -> nn.Module:
        """Deep-copy the model to avoid modifying the baseline."""
        return copy.deepcopy(model).to(self.device)

    def _evaluate(self, model: nn.Module, val_loader: DataLoader) -> Tuple[float, float]:
        """Run a validation pass and return (loss, accuracy)."""
        model.eval()
        criterion = nn.CrossEntropyLoss()
        val_loss = 0.0
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
        return val_loss / len(val_loader), correct / total
