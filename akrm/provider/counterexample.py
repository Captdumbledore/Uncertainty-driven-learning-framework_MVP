import numpy as np
from typing import List, Any
from akrm.provider.base import BaseProvider
from akrm.objective import LearningObjective

class CounterexampleProvider(BaseProvider):
    """
    Retrieves confusing neighbouring classes to improve class separation.
    """
    def __init__(self, train_labels: np.ndarray, embeddings: np.ndarray, n_retrieve: int = 8):
        self.train_labels = train_labels
        self.embeddings = embeddings
        self.n_retrieve = n_retrieve

    def provide(self, sample_idx: int, objective: LearningObjective, true_class: int = None, confused_class: int = None) -> List[int]:
        if true_class is None or confused_class is None:
            return []
            
        n_per_class = self.n_retrieve // 2
        n_extra = self.n_retrieve - 2 * n_per_class
        sample_emb = self.embeddings[sample_idx]
        retrieved = []

        for cls, n in [(true_class, n_per_class + n_extra), (confused_class, n_per_class)]:
            cls_mask = (self.train_labels == cls)
            if cls == true_class:
                cls_mask[sample_idx] = False
            cls_idx = np.where(cls_mask)[0]

            if len(cls_idx) == 0:
                continue

            cls_embs = self.embeddings[cls_idx]
            dists = np.linalg.norm(cls_embs - sample_emb, axis=1)
            n_take = min(n, len(cls_idx))
            nearest = cls_idx[np.argsort(dists)[:n_take]]
            retrieved.extend(nearest.tolist())

        return retrieved
