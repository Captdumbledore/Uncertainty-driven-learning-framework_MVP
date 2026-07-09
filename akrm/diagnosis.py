import numpy as np
from dataclasses import dataclass
from enum import Enum

class KnowledgeGapType(str, Enum):
    """
    Structural gap types diagnosed by the AKRM.
    """
    DECISION_BOUNDARY_CONFUSION = "decision_boundary_confusion"
    SPARSE_CONCEPT_REPRESENTATION = "sparse_concept_representation"
    GENERAL_UNCERTAINTY = "general_uncertainty"
    OOD_CONCEPT = "ood_concept"
    LOW_CONFIDENCE_CORRECT = "low_confidence_correct"

@dataclass
class KnowledgeGap:
    """
    Output of the diagnosis phase.
    """
    sample_idx: int
    gap_type: KnowledgeGapType
    entropy: float
    confidence: float
    margin: float
    true_class: int
    predicted_class: int
    top2_class: int
    top2_prob: float
    density_dist: float
    explanation: str

@dataclass
class AKRMConfig:
    """
    Hyperparameters governing the AKRM reasoning pipeline.
    """
    # Policy type: "heuristic" or "random"
    policy_type: str = "heuristic"
    
    # Entropy thresholds for knowledge gap diagnosis
    high_entropy_threshold: float = 0.50   # strongly uncertain
    medium_entropy_threshold: float = 0.20   # moderately uncertain

    # Margin threshold (top-1 prob - top-2 prob)
    low_margin_threshold: float = 0.20

    # Neighbourhood density (embedding space)
    density_k: int = 20
    sparse_density_percentile: float = 66.7

    # Retrieval parameters
    n_retrieve_retrieval: int = 8
    n_retrieve_counterexample: int = 8

class UncertaintyAnalysisEngine:
    """
    Extracts per-sample uncertainty features from MC Dropout results.
    """
    def __init__(self, unc_results: dict):
        self.unc_results = unc_results

    def analyse(self, sample_idx: int) -> dict:
        mean_probs  = self.unc_results["mean_probs"][sample_idx]
        true_label  = int(self.unc_results["true_labels"][sample_idx])
        pred_label  = int(self.unc_results["predicted_classes"][sample_idx])
        entropy     = float(self.unc_results["entropy"][sample_idx])
        confidence  = float(self.unc_results["confidence"][sample_idx])

        sorted_classes = np.argsort(mean_probs)[::-1]
        top2_class     = int(sorted_classes[1])
        top2_prob      = float(mean_probs[top2_class])
        margin         = float(mean_probs[sorted_classes[0]] - top2_prob)

        return {
            "sample_idx":      sample_idx,
            "true_label":      true_label,
            "predicted_label": pred_label,
            "entropy":         entropy,
            "confidence":      confidence,
            "margin":          margin,
            "top2_class":      top2_class,
            "top2_prob":       top2_prob,
            "is_correct":      (true_label == pred_label),
        }

class KnowledgeGapDiagnoser:
    """
    Classifies WHY a training sample is uncertain.
    """
    def __init__(self, config: AKRMConfig):
        self.config = config

    def diagnose(self, analysis: dict, density_dist: float, density_threshold: float, class_names: list) -> KnowledgeGap:
        cfg       = self.config
        tc_name   = class_names[analysis["true_label"]]
        top2_name = class_names[analysis["top2_class"]]
        ent       = analysis["entropy"]
        margin    = analysis["margin"]
        is_correct = analysis["is_correct"]

        if is_correct and ent > cfg.medium_entropy_threshold:
            gap_type = KnowledgeGapType.LOW_CONFIDENCE_CORRECT
            explanation = f"Correct prediction but with moderate/high uncertainty ({ent:.3f})."
        elif ent > cfg.high_entropy_threshold and margin < cfg.low_margin_threshold:
            gap_type = KnowledgeGapType.DECISION_BOUNDARY_CONFUSION
            explanation = (f"High entropy ({ent:.3f}) and low margin ({margin:.3f}). "
                           f"Model confuses '{tc_name}' with '{top2_name}'.")
        elif ent > cfg.medium_entropy_threshold and density_dist > density_threshold:
            gap_type = KnowledgeGapType.SPARSE_CONCEPT_REPRESENTATION
            explanation = (f"Medium entropy ({ent:.3f}) with sparse neighbourhood (dist={density_dist:.3f}). "
                           f"Lacks intra-class diversity.")
        else:
            gap_type = KnowledgeGapType.GENERAL_UNCERTAINTY
            explanation = f"General uncertainty ({ent:.3f})."

        return KnowledgeGap(
            sample_idx=analysis["sample_idx"],
            gap_type=gap_type,
            entropy=ent,
            confidence=analysis["confidence"],
            margin=margin,
            true_class=analysis["true_label"],
            predicted_class=analysis["predicted_label"],
            top2_class=analysis["top2_class"],
            top2_prob=analysis["top2_prob"],
            density_dist=density_dist,
            explanation=explanation
        )
