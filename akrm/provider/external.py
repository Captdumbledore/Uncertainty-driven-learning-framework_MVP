from typing import List, Any
from akrm.provider.base import BaseProvider
from akrm.objective import LearningObjective

class ExternalProvider(BaseProvider):
    """
    Retrieves semantically similar samples not present in the original training set
    from an external dataset.
    """
    def provide(self, sample_idx: int, objective: LearningObjective) -> List[Any]:
        # MVP Placeholder: Future extension.
        return []
