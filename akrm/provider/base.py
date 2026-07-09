from abc import ABC, abstractmethod
from typing import List, Any
import torch
from akrm.objective import LearningObjective

class BaseProvider(ABC):
    """
    Abstract Base Class for all Experience Providers.
    """
    @abstractmethod
    def provide(self, sample_idx: int, objective: LearningObjective) -> List[Any]:
        """
        Provides a list of task-appropriate learning experiences (tensors or indices)
        intended to satisfy the given learning objective.
        
        Args:
            sample_idx: Index of the uncertain sample.
            objective: The learning objective derived from diagnosis.
            
        Returns:
            List of experiences (can be indices into a dataset or actual generated Tensors).
        """
        pass
