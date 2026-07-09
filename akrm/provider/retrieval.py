import numpy as np
from typing import List, Any
from akrm.provider.base import BaseProvider
from akrm.objective import LearningObjective

class RetrievalProvider(BaseProvider):
    """
    Retrieves diverse real samples from the dataset spanning the full visual 
    diversity of the true class.
    """
    def __init__(self, train_labels: np.ndarray, embeddings: np.ndarray, n_retrieve: int = 8):
        self.train_labels = train_labels
        self.embeddings = embeddings
        self.n_retrieve = n_retrieve

    def provide(self, sample_idx: int, objective: LearningObjective, true_label: int = None) -> List[int]:
        if true_label is None:
            true_label = self.train_labels[sample_idx]
            
        same_class_mask = (self.train_labels == true_label)
        same_class_mask[sample_idx] = False
        same_class_idx = np.where(same_class_mask)[0]

        if len(same_class_idx) == 0:
            return []

        sample_emb = self.embeddings[sample_idx]
        same_class_embs = self.embeddings[same_class_idx]
        dists = np.linalg.norm(same_class_embs - sample_emb, axis=1)
        sorted_order = np.argsort(dists)

        n_take = min(self.n_retrieve, len(same_class_idx))
        if n_take == 1:
            chosen = sorted_order[[0]]
        else:
            percentile_positions = np.linspace(0, len(sorted_order) - 1, n_take, dtype=int)
            chosen = sorted_order[percentile_positions]

        return same_class_idx[chosen].tolist()
