"""
knowledge_distillation.py
-------------------------
Protocol 5: Knowledge Distillation

Freezes the baseline model as a "teacher". Trains a student (deep copy)
using a combined loss:

    L = alpha * L_CE(y, y_hat) + (1 - alpha) * L_KD(p_teacher, p_student)

where L_KD is the KL divergence between softened teacher and student logits.
This explicitly prevents forgetting by anchoring the student to the
teacher's knowledge.

Research Question:
Can preserving the teacher outputs reduce forgetting?
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from akrm.protocols.base import LearningProtocol
from retrain import TensorLabelDataset


class KnowledgeDistillation(LearningProtocol):
    """
    Trains a student model using both supervised loss and distillation loss
    from a frozen teacher (the baseline model).
    """

    def __init__(self, alpha: float = 0.5, temperature: float = 4.0, **kwargs):
        """
        Parameters
        ----------
        alpha       : Weight for the supervised CE loss.
                      (1 - alpha) is the weight for the distillation KD loss.
        temperature : Softmax temperature for distillation.
                      Higher T -> softer probability distributions -> more knowledge transfer.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.temperature = temperature

    def update(
        self,
        model: nn.Module,
        original_dataset: Dataset,
        curated_dataset: Dataset,
        val_loader: DataLoader,
    ) -> Tuple[nn.Module, dict]:

        # Teacher: frozen baseline
        teacher = self._make_copy(model)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        # Student: trainable copy
        student = self._make_copy(model)

        wrapped_train = TensorLabelDataset(original_dataset)
        wrapped_pool = TensorLabelDataset(curated_dataset)
        combined = ConcatDataset([wrapped_train, wrapped_pool])
        loader = DataLoader(combined, batch_size=self.batch_size,
                            shuffle=True, num_workers=0)

        criterion_ce = nn.CrossEntropyLoss()
        optimizer = optim.Adam(student.parameters(), lr=self.lr)

        history = {"train_loss": [], "train_acc": [],
                   "val_loss": [], "val_acc": []}

        T = self.temperature

        print(f"    [Knowledge Distillation] alpha={self.alpha}, "
              f"temperature={T}, dataset={len(combined):,}")

        for epoch in range(1, self.epochs + 1):
            student.train()
            train_loss = 0.0
            correct = total = 0

            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Student forward pass
                student_logits = student(images)

                # Teacher forward pass (no grad)
                with torch.no_grad():
                    teacher_logits = teacher(images)

                # Supervised loss (hard labels)
                loss_ce = criterion_ce(student_logits, labels)

                # Distillation loss (soft labels from teacher)
                loss_kd = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction="batchmean"
                ) * (T * T)

                # Combined loss
                loss = self.alpha * loss_ce + (1.0 - self.alpha) * loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, preds = student_logits.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

            t_loss = train_loss / len(loader)
            t_acc = correct / total
            v_loss, v_acc = self._evaluate(student, val_loader)

            history["train_loss"].append(t_loss)
            history["train_acc"].append(t_acc)
            history["val_loss"].append(v_loss)
            history["val_acc"].append(v_acc)

            print(f"    Epoch {epoch:>2}/{self.epochs}  "
                  f"| Train Acc {t_acc:.4f}  | Val Acc {v_acc:.4f}")

        return student, history
