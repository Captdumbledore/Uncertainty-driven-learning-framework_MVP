from enum import Enum
from akrm.diagnosis import KnowledgeGap
from akrm.objective import LearningObjective

class ProviderType(str, Enum):
    """
    Types of experience providers available in the framework.
    """
    RETRIEVAL = "retrieval"
    COUNTEREXAMPLE = "counterexample"
    CONTEXTUAL = "contextual"
    SYNTHETIC = "synthetic"
    EXTERNAL = "external"

import random

class KnowledgeGuidedExperiencePlanner:
    """
    The decision engine that maps a learning objective and the sample's context
    to one or more candidate learning strategies (ProviderTypes).
    
    This uses deterministic heuristics to guide selection.
    """
    
    def __init__(self, policy_type: str = "heuristic"):
        self.policy_type = policy_type

    def select_strategy(self, objective: LearningObjective, gap_context: KnowledgeGap, history: dict = None) -> ProviderType:
        """
        Selects the most appropriate experience provider.
        """
        if self.policy_type == "random":
            # For Stage 1 testing, only choose between existing providers
            return random.choice([ProviderType.RETRIEVAL, ProviderType.COUNTEREXAMPLE])
            
        # MVP Heuristics:
        if objective == LearningObjective.IMPROVE_CLASS_SEPARATION:
            if gap_context.margin < 0.10:
                return ProviderType.COUNTEREXAMPLE
            else:
                return ProviderType.SYNTHETIC
                
        elif objective == LearningObjective.INCREASE_INTRA_CLASS_DIVERSITY:
            return ProviderType.CONTEXTUAL
            
        elif objective == LearningObjective.STRENGTHEN_CONCEPT_UNDERSTANDING:
            return ProviderType.RETRIEVAL
            
        elif objective == LearningObjective.IMPROVE_ROBUSTNESS:
            return ProviderType.CONTEXTUAL
            
        elif objective == LearningObjective.CALIBRATE_CONFIDENCE:
            return ProviderType.RETRIEVAL
            
        return ProviderType.RETRIEVAL
