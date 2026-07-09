from enum import Enum
from akrm.diagnosis import KnowledgeGap, KnowledgeGapType

class LearningObjective(str, Enum):
    """
    Defines WHAT must be learned based on the diagnosed gap.
    Does NOT define HOW it will be learned.
    """
    IMPROVE_CLASS_SEPARATION = "improve_class_separation"
    INCREASE_INTRA_CLASS_DIVERSITY = "increase_intra_class_diversity"
    STRENGTHEN_CONCEPT_UNDERSTANDING = "strengthen_concept_understanding"
    IMPROVE_ROBUSTNESS = "improve_robustness"
    CALIBRATE_CONFIDENCE = "calibrate_confidence"

class LearningObjectiveGenerator:
    """
    Translates a diagnosis into a learning objective.
    """
    _GAP_TO_OBJECTIVE = {
        KnowledgeGapType.DECISION_BOUNDARY_CONFUSION: LearningObjective.IMPROVE_CLASS_SEPARATION,
        KnowledgeGapType.SPARSE_CONCEPT_REPRESENTATION: LearningObjective.INCREASE_INTRA_CLASS_DIVERSITY,
        KnowledgeGapType.GENERAL_UNCERTAINTY: LearningObjective.STRENGTHEN_CONCEPT_UNDERSTANDING,
        KnowledgeGapType.OOD_CONCEPT: LearningObjective.IMPROVE_ROBUSTNESS,
        KnowledgeGapType.LOW_CONFIDENCE_CORRECT: LearningObjective.CALIBRATE_CONFIDENCE,
    }

    def generate(self, gap: KnowledgeGap) -> LearningObjective:
        """
        Translates a knowledge gap into a learning objective.
        """
        return self._GAP_TO_OBJECTIVE[gap.gap_type]
