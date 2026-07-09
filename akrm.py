"""
akrm.py
-------
Adaptive Knowledge Reasoning Module (AKRM) — Version 1 (MVP)

Research Framework: Uncertainty-Driven Learning
B.Tech Final Year Project — Minimum Viable Research Prototype

Philosophy
----------
  Traditional uncertainty-guided learning:
      Uncertainty → Fixed Action → Retraining

  This framework:
      Uncertainty → Knowledge Gap Diagnosis → Learning Objective
                  → Strategy Selection → Experience Pool → Retraining

  The novelty is NOT uncertainty estimation.
  The novelty is the reasoning process that converts uncertainty into
  targeted learning experiences.

Components
----------
  1. UncertaintyAnalysisEngine  — extracts per-sample features from MC Dropout
  2. KnowledgeGapDiagnoser      — classifies WHY the model is uncertain
  3. LearningObjectiveGenerator — converts diagnosis into an educational goal
  4. AdaptiveStrategySelector   — maps objective to a retrieval strategy
  5. ExperiencePoolBuilder      — retrieves real training samples; builds pool

  AdaptiveKnowledgeReasoningModule orchestrates all five components.
  The main pipeline communicates only with this facade class.

Version 1 implements two learning strategies:
  - Retrieval Learning     : diverse same-class examples
  - Counterexample Learning: boundary examples from true & confused class

Future strategy placeholders (raise NotImplementedError):
  - retrieve_external_experience()
  - retrieve_contextual_experience()
  - retrieve_generated_experience()

Outputs per seed
----------------
  outputs/seed_<N>/akrm_reasoning.csv         — per-sample reasoning log
  outputs/seed_<N>/diagnosis_validation.txt   — diagnosis sanity-check report
"""

import csv
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


# =============================================================================
# Enumerations
# =============================================================================

class KnowledgeGapType(str, Enum):
    """
    Three knowledge gap types diagnosed in Version 1.

    Future versions may add:
      DISTRIBUTION_SHIFT   — sample from a shifted input distribution
      CONTEXT_DEFICIENCY   — missing contextual information
      NOVEL_CONCEPT        — class never seen clearly in training
    """
    DECISION_BOUNDARY_CONFUSION = "decision_boundary_confusion"
    LOW_INTRA_CLASS_DIVERSITY    = "low_intra_class_diversity"
    GENERAL_UNCERTAINTY          = "general_uncertainty"


class LearningObjective(str, Enum):
    """Educational objective generated from the diagnosed knowledge gap."""
    INCREASE_CLASS_SEPARATION          = "increase_class_separation"
    INCREASE_INTRA_CLASS_DIVERSITY     = "increase_intra_class_diversity"
    STRENGTHEN_CONCEPT_REPRESENTATION  = "strengthen_concept_representation"


class Strategy(str, Enum):
    """Learning experience strategy selected by the rule-based engine."""
    RETRIEVAL      = "RETRIEVAL"
    COUNTEREXAMPLE = "COUNTEREXAMPLE"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AKRMConfig:
    """
    Hyperparameters governing the AKRM reasoning pipeline.

    All thresholds are consolidated here so the decision logic remains
    transparent, reproducible, and easy to adjust for future experiments.

    Calibrated against observed Fashion-MNIST baseline run:
      Mean entropy — correct predictions   : 0.13
      Mean entropy — incorrect predictions : 0.75
    """

    # Entropy thresholds for knowledge gap diagnosis
    high_entropy_threshold:   float = 0.50   # strongly uncertain
    medium_entropy_threshold: float = 0.20   # moderately uncertain

    # Margin threshold (top-1 prob − top-2 prob)
    # Small margin → model barely distinguishes the top two classes
    low_margin_threshold: float = 0.20

    # Neighbourhood density (embedding space)
    density_k:                 int   = 20    # k nearest same-class neighbours
    sparse_density_percentile: float = 66.7  # above this percentile = sparse

    # Retrieval parameters (per uncertain sample)
    n_retrieve_retrieval:      int = 8   # diverse same-class samples
    n_retrieve_counterexample: int = 8   # 4 from true class + 4 from confused class


# =============================================================================
# Component 1 — Uncertainty Analysis Engine
# =============================================================================

class UncertaintyAnalysisEngine:
    """
    Extracts per-sample uncertainty features from MC Dropout results.

    The engine does NOT make any decisions.
    It only transforms raw MC Dropout outputs into a structured feature dict
    that subsequent components can reason about.

    Extracted features
    ------------------
    entropy          : Predictive entropy H = −Σ p̄_c log(p̄_c)
    confidence       : max p̄_c (highest class probability)
    margin           : top-1 prob − top-2 prob (decision margin)
    top2_class       : index of the second most probable class
    top2_prob        : probability of the second most probable class
    true_label       : ground-truth class index
    predicted_label  : MC-Dropout argmax prediction
    is_correct       : whether prediction matches true label
    """

    def __init__(self, unc_results: dict) -> None:
        """
        Parameters
        ----------
        unc_results : Output of mc_dropout_predict() — covers the full
                      training dataset (NOT just uncertain samples).
        """
        self.unc_results = unc_results

    def analyse(self, sample_idx: int) -> dict:
        """
        Extract features for one training sample.

        Parameters
        ----------
        sample_idx : Index into the training dataset (matches unc_results index)

        Returns
        -------
        Feature dict with keys listed in the class docstring.
        """
        mean_probs  = self.unc_results["mean_probs"][sample_idx]   # (C,)
        true_label  = int(self.unc_results["true_labels"][sample_idx])
        pred_label  = int(self.unc_results["predicted_classes"][sample_idx])
        entropy     = float(self.unc_results["entropy"][sample_idx])
        confidence  = float(self.unc_results["confidence"][sample_idx])

        # Identify top-2 classes by probability
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


# =============================================================================
# Component 2 — Knowledge Gap Diagnoser
# =============================================================================

class KnowledgeGapDiagnoser:
    """
    Classifies WHY a training sample is uncertain.

    Instead of reacting directly to uncertainty magnitude, the diagnoser
    attempts to identify the structural reason behind the uncertainty.

    Diagnosis rules (applied in order; first match wins)
    ────────────────────────────────────────────────────
    DECISION_BOUNDARY_CONFUSION
      IF  entropy > high_entropy_threshold
      AND margin  < low_margin_threshold
      → Model struggles to separate the true class from another class.

    LOW_INTRA_CLASS_DIVERSITY
      IF  entropy > medium_entropy_threshold
      AND density_dist > sparse_density_percentile (global threshold)
      → Model has few similar examples of this class in feature space.

    GENERAL_UNCERTAINTY
      Catch-all for all remaining uncertain samples.

    Note on density_threshold
    ─────────────────────────
    The density threshold is computed globally across all uncertain samples
    (66.7th percentile), making it relative and scale-invariant.
    This requires a two-pass approach in the AKRM orchestrator.
    """

    def __init__(self, config: AKRMConfig) -> None:
        self.config = config

    def diagnose(
        self,
        analysis:          dict,
        density_dist:      float,
        density_threshold: float,
        class_names:       list,
    ) -> tuple:
        """
        Diagnose the knowledge gap for one uncertain sample.

        Parameters
        ----------
        analysis          : Feature dict from UncertaintyAnalysisEngine
        density_dist      : L2 distance to k-th nearest same-class neighbour
        density_threshold : Global density threshold (66.7th percentile)
        class_names       : List of class name strings (for explanations)

        Returns
        -------
        gap_type    : KnowledgeGapType
        explanation : Human-readable textual explanation
        """
        cfg       = self.config
        tc_name   = class_names[analysis["true_label"]]
        top2_name = class_names[analysis["top2_class"]]
        ent       = analysis["entropy"]
        margin    = analysis["margin"]

        # Rule 1 — Decision Boundary Confusion
        if ent > cfg.high_entropy_threshold and margin < cfg.low_margin_threshold:
            gap_type    = KnowledgeGapType.DECISION_BOUNDARY_CONFUSION
            explanation = (
                f"High entropy ({ent:.3f}) with low decision margin ({margin:.3f}). "
                f"The model confuses '{tc_name}' with '{top2_name}' "
                f"(p={analysis['top2_prob']:.3f}). "
                f"The model's decision boundary between these two classes is weak."
            )

        # Rule 2 — Low Intra-Class Diversity
        elif ent > cfg.medium_entropy_threshold and density_dist > density_threshold:
            gap_type    = KnowledgeGapType.LOW_INTRA_CLASS_DIVERSITY
            explanation = (
                f"Medium entropy ({ent:.3f}) with sparse feature-space neighbourhood "
                f"for '{tc_name}' (density_dist={density_dist:.3f} > "
                f"threshold {density_threshold:.3f}). "
                f"This sample lies in an underrepresented region of the class distribution."
            )

        # Rule 3 — General Uncertainty (default)
        else:
            gap_type    = KnowledgeGapType.GENERAL_UNCERTAINTY
            correct_str = "correct" if analysis["is_correct"] else "incorrect"
            explanation = (
                f"Moderate uncertainty ({ent:.3f}). Prediction was {correct_str} "
                f"(predicted '{class_names[analysis['predicted_label']]}', "
                f"true '{tc_name}'). No specific structural gap identified. "
                f"General concept reinforcement is indicated."
            )

        return gap_type, explanation


# =============================================================================
# Component 3 — Learning Objective Generator
# =============================================================================

class LearningObjectiveGenerator:
    """
    Converts a diagnosed knowledge gap into a learning objective.

    This component is intentionally separate from strategy selection.
    The objective describes WHAT the model should learn; the strategy
    selector decides HOW to teach it.  This separation allows future
    versions to map the same objective to a different strategy without
    changing the diagnosis layer.

    Mappings
    --------
    DECISION_BOUNDARY_CONFUSION   → INCREASE_CLASS_SEPARATION
    LOW_INTRA_CLASS_DIVERSITY     → INCREASE_INTRA_CLASS_DIVERSITY
    GENERAL_UNCERTAINTY           → STRENGTHEN_CONCEPT_REPRESENTATION
    """

    _GAP_TO_OBJECTIVE = {
        KnowledgeGapType.DECISION_BOUNDARY_CONFUSION: LearningObjective.INCREASE_CLASS_SEPARATION,
        KnowledgeGapType.LOW_INTRA_CLASS_DIVERSITY:    LearningObjective.INCREASE_INTRA_CLASS_DIVERSITY,
        KnowledgeGapType.GENERAL_UNCERTAINTY:          LearningObjective.STRENGTHEN_CONCEPT_REPRESENTATION,
    }

    _OBJECTIVE_DESCRIPTIONS = {
        LearningObjective.INCREASE_CLASS_SEPARATION: (
            "Sharpen the decision boundary between the confused classes by presenting "
            "discriminative examples from both sides of the boundary."
        ),
        LearningObjective.INCREASE_INTRA_CLASS_DIVERSITY: (
            "Expose the model to a wider range of visual appearances within the true "
            "class to improve robustness of the class representation."
        ),
        LearningObjective.STRENGTHEN_CONCEPT_REPRESENTATION: (
            "Reinforce the model's general understanding of the true class by "
            "revisiting representative training examples."
        ),
    }

    def generate(self, gap_type: KnowledgeGapType) -> tuple:
        """
        Parameters
        ----------
        gap_type : Diagnosed knowledge gap type

        Returns
        -------
        objective   : LearningObjective
        description : Human-readable description of the objective
        """
        objective   = self._GAP_TO_OBJECTIVE[gap_type]
        description = self._OBJECTIVE_DESCRIPTIONS[objective]
        return objective, description


# =============================================================================
# Component 4 — Adaptive Strategy Selector
# =============================================================================

class AdaptiveStrategySelector:
    """
    Maps a learning objective to a retrieval strategy.

    Version 1: rule-based mapping (transparent, reproducible).
    Version 2: this class can be replaced with a learned policy
    (RL / contextual bandits / meta-learning) without modifying any
    other component of the AKRM.

    The selector receives a LEARNING OBJECTIVE, not raw uncertainty.
    This ensures the strategy is always grounded in an interpretation of
    WHY the model is uncertain.

    Mappings
    --------
    INCREASE_CLASS_SEPARATION         → COUNTEREXAMPLE
    INCREASE_INTRA_CLASS_DIVERSITY    → RETRIEVAL
    STRENGTHEN_CONCEPT_REPRESENTATION → RETRIEVAL
    """

    _OBJECTIVE_TO_STRATEGY = {
        LearningObjective.INCREASE_CLASS_SEPARATION:         Strategy.COUNTEREXAMPLE,
        LearningObjective.INCREASE_INTRA_CLASS_DIVERSITY:    Strategy.RETRIEVAL,
        LearningObjective.STRENGTHEN_CONCEPT_REPRESENTATION: Strategy.RETRIEVAL,
    }

    def select(
        self,
        objective:   LearningObjective,
        analysis:    dict,
        class_names: list,
    ) -> tuple:
        """
        Parameters
        ----------
        objective   : Learning objective from LearningObjectiveGenerator
        analysis    : Feature dict from UncertaintyAnalysisEngine
        class_names : Class name strings for human-readable reasons

        Returns
        -------
        strategy : Strategy (RETRIEVAL or COUNTEREXAMPLE)
        reason   : Textual justification for the selection
        """
        strategy  = self._OBJECTIVE_TO_STRATEGY[objective]
        tc_name   = class_names[analysis["true_label"]]
        top2_name = class_names[analysis["top2_class"]]

        if strategy == Strategy.COUNTEREXAMPLE:
            reason = (
                f"Objective '{objective.value}' requires exposing discriminative pairs. "
                f"Counterexample Learning selected: will retrieve boundary examples "
                f"from both '{tc_name}' (true class) and '{top2_name}' (confused class)."
            )
        else:
            reason = (
                f"Objective '{objective.value}' requires broadening class coverage. "
                f"Retrieval Learning selected: will retrieve diverse examples "
                f"spanning the full visual range of '{tc_name}'."
            )

        return strategy, reason


# =============================================================================
# Component 5 — Experience Pool Builder
# =============================================================================

class ExperiencePoolBuilder:
    """
    Retrieves real training samples based on the selected strategy and
    builds a deduplicated experience pool.

    No synthetic images are generated.
    No external datasets are used.
    Only existing training data is retrieved.

    Strategy A — Retrieval Learning
    ────────────────────────────────
    Retrieve samples that SPAN the full visual diversity of the true class.
    Selection uses evenly distributed percentiles of the L2 distance
    from the uncertain sample to all same-class training samples
    (nearest → farthest), maximising intra-class coverage.

    Strategy B — Counterexample Learning
    ─────────────────────────────────────
    Retrieve samples near the decision boundary from BOTH the true class
    and the confused class.  "Near the boundary" is approximated by
    proximity to the uncertain sample in embedding space (the uncertain
    sample is itself near the boundary by definition).
    """

    def __init__(
        self,
        train_dataset,
        embeddings:    np.ndarray,
        train_labels:  np.ndarray,
        config:        AKRMConfig,
    ) -> None:
        self.train_dataset = train_dataset
        self.embeddings    = embeddings
        self.train_labels  = train_labels
        self.config        = config

    # ── Strategy A: Retrieval Learning ──────────────────────────────────────

    def retrieve_diverse_same_class(
        self,
        sample_idx: int,
        true_label: int,
    ) -> np.ndarray:
        """
        Retrieve same-class training samples spanning the class distribution.

        Parameters
        ----------
        sample_idx : Uncertain sample's training dataset index
        true_label : Ground-truth class of the uncertain sample

        Returns
        -------
        Integer array of training dataset indices (may be empty).
        """
        same_class_mask             = (self.train_labels == true_label)
        same_class_mask[sample_idx] = False          # exclude self
        same_class_idx              = np.where(same_class_mask)[0]

        if len(same_class_idx) == 0:
            return np.array([], dtype=int)

        sample_emb      = self.embeddings[sample_idx]
        same_class_embs = self.embeddings[same_class_idx]
        dists           = np.linalg.norm(same_class_embs - sample_emb, axis=1)
        sorted_order    = np.argsort(dists)

        n_take = min(self.config.n_retrieve_retrieval, len(same_class_idx))

        # Select n_take samples at evenly-spaced percentiles of the distance
        # distribution (nearest → farthest).  This maximises diversity by
        # covering both common and rare appearances of the class.
        if n_take == 1:
            chosen = sorted_order[[0]]
        else:
            percentile_positions = np.linspace(0, len(sorted_order) - 1,
                                               n_take, dtype=int)
            chosen = sorted_order[percentile_positions]

        return same_class_idx[chosen]

    # ── Strategy B: Counterexample Learning ─────────────────────────────────

    def retrieve_counterexamples(
        self,
        sample_idx:     int,
        true_class:     int,
        confused_class: int,
    ) -> np.ndarray:
        """
        Retrieve boundary examples from the true class AND the confused class.

        The nearest neighbours in embedding space to the uncertain sample
        are used as a proxy for "close to the decision boundary".

        Parameters
        ----------
        sample_idx     : Uncertain sample's index
        true_class     : Ground-truth class
        confused_class : Class most confused with true_class (top-2)

        Returns
        -------
        Integer array of training dataset indices.
        """
        n_per_class = self.config.n_retrieve_counterexample // 2
        n_extra     = self.config.n_retrieve_counterexample - 2 * n_per_class
        sample_emb  = self.embeddings[sample_idx]
        retrieved   = []

        for cls, n in [(true_class, n_per_class + n_extra),
                       (confused_class, n_per_class)]:
            cls_mask             = (self.train_labels == cls)
            if cls == true_class:
                cls_mask[sample_idx] = False  # avoid returning self
            cls_idx              = np.where(cls_mask)[0]

            if len(cls_idx) == 0:
                continue

            cls_embs  = self.embeddings[cls_idx]
            dists     = np.linalg.norm(cls_embs - sample_emb, axis=1)
            n_take    = min(n, len(cls_idx))
            nearest   = cls_idx[np.argsort(dists)[:n_take]]
            retrieved.extend(nearest.tolist())

        return np.array(retrieved, dtype=int)

    # ── Pool construction ────────────────────────────────────────────────────

    def build(self, experience_batches: List[np.ndarray]) -> Subset:
        """
        Merge all retrieved experiences, deduplicate, and return a Subset.

        Parameters
        ----------
        experience_batches : List of integer index arrays (one per uncertain sample)

        Returns
        -------
        torch.utils.data.Subset of train_dataset containing the experience pool
        """
        pool: set = set()
        for batch in experience_batches:
            pool.update(batch.tolist())
        pool_indices = sorted(pool)
        return Subset(self.train_dataset, pool_indices)


# =============================================================================
# Main Facade — Adaptive Knowledge Reasoning Module
# =============================================================================

class AdaptiveKnowledgeReasoningModule:
    """
    Orchestrates the full AKRM reasoning pipeline.

    This is the only class that the main training pipeline interacts with.
    All internal components can be upgraded or replaced without changing
    the calling code in main.py.

    Pipeline per uncertain sample
    ─────────────────────────────
      MC Dropout results
          ↓ UncertaintyAnalysisEngine
      entropy, confidence, margin, top-2, is_correct
          ↓ KnowledgeGapDiagnoser
      Gap type + explanation
          ↓ LearningObjectiveGenerator
      Learning objective + description
          ↓ AdaptiveStrategySelector
      Strategy (RETRIEVAL | COUNTEREXAMPLE) + reason
          ↓ ExperiencePoolBuilder
      Retrieved training sample indices
          ↓ build()
      Deduplicated experience pool (Subset)

    Parameters
    ----------
    model        : Trained SimpleCNN (used only for embedding extraction)
    train_dataset: Training Subset — NEVER val or test data
    unc_results  : Full training-set MC Dropout output (from Phase 2)
    device       : torch.device or string
    class_names  : List of class name strings (for logging and reporting)
    config       : AKRMConfig instance (uses defaults if not provided)
    """

    def __init__(
        self,
        model,
        train_dataset,
        unc_results: dict,
        device,
        class_names: list,
        config:      Optional[AKRMConfig] = None,
    ) -> None:
        self.model         = model
        self.train_dataset = train_dataset
        self.unc_results   = unc_results
        self.device        = device
        self.class_names   = class_names
        self.config        = config or AKRMConfig()

        # Populated during run()
        self._embeddings:     Optional[np.ndarray] = None
        self._train_labels:   Optional[np.ndarray] = None
        self._reasoning_log:  List[dict]            = []

        # Components (instantiated once embeddings are ready)
        self._analysis_engine    = UncertaintyAnalysisEngine(unc_results)
        self._diagnoser          = KnowledgeGapDiagnoser(self.config)
        self._objective_gen      = LearningObjectiveGenerator()
        self._strategy_selector  = AdaptiveStrategySelector()
        self._pool_builder:      Optional[ExperiencePoolBuilder] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, uncertain_indices: np.ndarray) -> Subset:
        """
        Execute the full AKRM reasoning pipeline.

        Parameters
        ----------
        uncertain_indices : Indices of top-uncertain training samples
                            (from get_selection_indices mode="highest")

        Returns
        -------
        Subset of train_dataset forming the curated experience pool.
        """
        # ── Step 0: Extract CNN embeddings ────────────────────────────────────
        print(f"\n    AKRM: Extracting CNN embeddings from "
              f"{len(self.train_dataset):,} training samples...")
        self._extract_embeddings()

        self._pool_builder = ExperiencePoolBuilder(
            self.train_dataset, self._embeddings,
            self._train_labels, self.config,
        )

        # ── Pass 1: Analyse all uncertain samples + compute density distances ─
        print(f"    AKRM: Pass 1 — Uncertainty analysis "
              f"({len(uncertain_indices):,} samples)...")
        analyses      = []
        density_dists = []

        for idx in uncertain_indices:
            analysis     = self._analysis_engine.analyse(int(idx))
            density_dist = self._compute_density_dist(
                int(idx), analysis["true_label"]
            )
            analyses.append(analysis)
            density_dists.append(density_dist)

        # Global density threshold (relative, percentile-based)
        density_threshold = float(
            np.percentile(density_dists, self.config.sparse_density_percentile)
        )
        print(f"    AKRM: Density threshold = {density_threshold:.4f} "
              f"({self.config.sparse_density_percentile:.1f}th percentile)")

        # ── Pass 2: Diagnose → Objective → Strategy → Retrieve ───────────────
        print(f"    AKRM: Pass 2 — Reasoning pipeline...")
        experience_batches: List[np.ndarray] = []
        strategy_counts = Counter()
        gap_counts      = Counter()

        for analysis, density_dist in zip(analyses, density_dists):
            gap_type, gap_explanation = self._diagnoser.diagnose(
                analysis, density_dist, density_threshold, self.class_names
            )
            objective, obj_description = self._objective_gen.generate(gap_type)
            strategy, strategy_reason  = self._strategy_selector.select(
                objective, analysis, self.class_names
            )

            if strategy == Strategy.RETRIEVAL:
                retrieved = self._pool_builder.retrieve_diverse_same_class(
                    analysis["sample_idx"], analysis["true_label"]
                )
            else:
                retrieved = self._pool_builder.retrieve_counterexamples(
                    analysis["sample_idx"],
                    analysis["true_label"],
                    analysis["top2_class"],
                )

            experience_batches.append(retrieved)
            strategy_counts[strategy.value] += 1
            gap_counts[gap_type.value]      += 1

            self._reasoning_log.append({
                **analysis,
                "density_dist":          density_dist,
                "knowledge_gap":         gap_type.value,
                "gap_explanation":       gap_explanation,
                "learning_objective":    objective.value,
                "objective_description": obj_description,
                "selected_strategy":     strategy.value,
                "strategy_reason":       strategy_reason,
                "n_experiences":         len(retrieved),
            })

        # ── Build deduplicated experience pool ────────────────────────────────
        experience_pool = self._pool_builder.build(experience_batches)

        print(
            f"    AKRM: Strategy split — "
            f"RETRIEVAL: {strategy_counts[Strategy.RETRIEVAL.value]:,}  "
            f"COUNTEREXAMPLE: {strategy_counts[Strategy.COUNTEREXAMPLE.value]:,}"
        )
        print(
            f"    AKRM: Experience pool — {len(experience_pool):,} unique training "
            f"samples (deduplicated from "
            f"{sum(len(b) for b in experience_batches):,} total retrievals)"
        )

        return experience_pool

    def save_reasoning_report(self, save_path: str) -> None:
        """
        Write the per-sample reasoning log to a CSV file.

        Every decision made by the AKRM is recorded, making the full
        reasoning pipeline explainable and auditable for the research paper.
        """
        if not self._reasoning_log:
            print("    AKRM: No reasoning records to save.")
            return

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        fieldnames = [
            "sample_id",
            "true_label", "true_label_name",
            "predicted_label", "predicted_label_name",
            "entropy", "confidence", "margin",
            "top2_class", "top2_class_name", "top2_prob",
            "density_dist", "is_correct",
            "knowledge_gap", "gap_explanation",
            "learning_objective", "objective_description",
            "selected_strategy", "strategy_reason",
            "n_experiences",
        ]

        with open(save_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self._reasoning_log:
                writer.writerow({
                    "sample_id":              rec["sample_idx"],
                    "true_label":             rec["true_label"],
                    "true_label_name":        self._cname(rec["true_label"]),
                    "predicted_label":        rec["predicted_label"],
                    "predicted_label_name":   self._cname(rec["predicted_label"]),
                    "entropy":                f"{rec['entropy']:.6f}",
                    "confidence":             f"{rec['confidence']:.6f}",
                    "margin":                 f"{rec['margin']:.6f}",
                    "top2_class":             rec["top2_class"],
                    "top2_class_name":        self._cname(rec["top2_class"]),
                    "top2_prob":              f"{rec['top2_prob']:.6f}",
                    "density_dist":           f"{rec['density_dist']:.6f}",
                    "is_correct":             rec["is_correct"],
                    "knowledge_gap":          rec["knowledge_gap"],
                    "gap_explanation":        rec["gap_explanation"],
                    "learning_objective":     rec["learning_objective"],
                    "objective_description":  rec["objective_description"],
                    "selected_strategy":      rec["selected_strategy"],
                    "strategy_reason":        rec["strategy_reason"],
                    "n_experiences":          rec["n_experiences"],
                })

        print(f"    AKRM: Reasoning report saved "
              f"({len(self._reasoning_log):,} rows) → {save_path}")

    def save_diagnosis_validation(self, save_path: str) -> None:
        """
        Generate a plain-text validation report for the AKRM's diagnosis.

        Purpose: verify that the diagnosis is meaningful before interpreting
        results.  Includes gap distribution, confusion statistics, per-gap
        entropy statistics, and example reasoning records.

        This report should be included in the research paper appendix as
        evidence that the diagnosis reflects genuine model behaviour.
        """
        if not self._reasoning_log:
            print("    AKRM: No records for diagnosis validation.")
            return

        recs       = self._reasoning_log
        n          = len(recs)
        gap_counts = Counter(r["knowledge_gap"] for r in recs)

        # Per-gap entropy statistics
        gap_entropy = defaultdict(list)
        for r in recs:
            gap_entropy[r["knowledge_gap"]].append(r["entropy"])

        # Top confusion pairs (for DECISION_BOUNDARY_CONFUSION only)
        pair_counts: Counter = Counter()
        for r in recs:
            if r["knowledge_gap"] == KnowledgeGapType.DECISION_BOUNDARY_CONFUSION.value:
                pair = (
                    self._cname(r["true_label"]),
                    self._cname(r["top2_class"]),
                )
                # Normalise pair order for counting
                pair_counts[tuple(sorted(pair))] += 1

        # Sample examples (up to 3 per gap type)
        gap_examples: dict = defaultdict(list)
        for r in recs:
            if len(gap_examples[r["knowledge_gap"]]) < 3:
                gap_examples[r["knowledge_gap"]].append(r)

        sep  = "=" * 70
        sep2 = "-" * 70

        lines = [
            sep,
            "  AKRM DIAGNOSIS VALIDATION REPORT",
            "  Purpose: Verify that the knowledge gap diagnosis reflects",
            "  genuine model behaviour (not random classification).",
            sep,
            "",
            f"  Total uncertain samples analysed : {n:,}",
            "",
            sep2,
            "  1. KNOWLEDGE GAP DISTRIBUTION",
            sep2,
        ]

        gap_display = {
            KnowledgeGapType.DECISION_BOUNDARY_CONFUSION.value:
                "Decision Boundary Confusion",
            KnowledgeGapType.LOW_INTRA_CLASS_DIVERSITY.value:
                "Low Intra-class Diversity",
            KnowledgeGapType.GENERAL_UNCERTAINTY.value:
                "General Uncertainty",
        }
        for gv, gname in gap_display.items():
            cnt = gap_counts.get(gv, 0)
            pct = cnt / n * 100
            lines.append(f"  {gname:<35} {cnt:>5}  ({pct:5.1f}%)")

        lines += [
            "",
            sep2,
            "  2. MEAN ENTROPY PER GAP TYPE",
            sep2,
        ]
        for gv, gname in gap_display.items():
            vals = gap_entropy.get(gv, [0.0])
            lines.append(
                f"  {gname:<35} mean={np.mean(vals):.4f}  "
                f"std={np.std(vals):.4f}  "
                f"min={np.min(vals):.4f}  max={np.max(vals):.4f}"
            )
        lines.append(
            "  (Expected: DECISION_BOUNDARY_CONFUSION > LOW_DIVERSITY > GENERAL)"
        )

        lines += [
            "",
            sep2,
            "  3. TOP CONFUSION PAIRS  (Decision Boundary Confusion only)",
            sep2,
        ]
        if pair_counts:
            for (c1, c2), cnt in pair_counts.most_common(10):
                pct = cnt / gap_counts.get(
                    KnowledgeGapType.DECISION_BOUNDARY_CONFUSION.value, 1
                ) * 100
                lines.append(f"  {c1:<15} vs {c2:<15} : {cnt:>4}  ({pct:5.1f}%)")
        else:
            lines.append("  (No Decision Boundary Confusion samples found)")

        lines += [
            "",
            sep2,
            "  4. EXAMPLE REASONING RECORDS (up to 3 per gap type)",
            sep2,
        ]
        for gv, gname in gap_display.items():
            lines += ["", f"  [{gname}]"]
            examples = gap_examples.get(gv, [])
            if not examples:
                lines.append("    (none)")
                continue
            for ex in examples:
                lines += [
                    f"    Sample {ex['sample_idx']:>6}  "
                    f"True: {self._cname(ex['true_label']):<15}  "
                    f"Pred: {self._cname(ex['predicted_label']):<15}",
                    f"    Entropy: {ex['entropy']:.4f}  "
                    f"Margin: {ex['margin']:.4f}  "
                    f"Correct: {ex['is_correct']}",
                    f"    Explanation: {ex['gap_explanation'][:120]}...",
                    f"    Objective  : {ex['learning_objective']}",
                    f"    Strategy   : {ex['selected_strategy']}",
                    f"    Experiences: {ex['n_experiences']}",
                    "",
                ]

        lines += [
            sep2,
            "  5. STRATEGY BREAKDOWN",
            sep2,
        ]
        strat_counts = Counter(r["selected_strategy"] for r in recs)
        for sv in [Strategy.RETRIEVAL.value, Strategy.COUNTEREXAMPLE.value]:
            cnt = strat_counts.get(sv, 0)
            lines.append(f"  {sv:<20} {cnt:>5}  ({cnt/n*100:5.1f}%)")

        lines += ["", sep]

        report = "\n".join(lines)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as fh:
            fh.write(report)

        print(f"    AKRM: Diagnosis validation saved → {save_path}")

    def print_summary(self) -> None:
        """Print a concise AKRM summary to stdout."""
        recs = self._reasoning_log
        if not recs:
            print("    AKRM: No records.")
            return

        n          = len(recs)
        gap_counts = Counter(r["knowledge_gap"] for r in recs)
        str_counts = Counter(r["selected_strategy"] for r in recs)
        total_exp  = sum(r["n_experiences"] for r in recs)

        print(f"\n    AKRM Summary:")
        print(f"      Uncertain samples : {n:,}")
        print(f"      Knowledge gaps    :")
        for gv, gname in [
            (KnowledgeGapType.DECISION_BOUNDARY_CONFUSION.value, "  Boundary Confusion"),
            (KnowledgeGapType.LOW_INTRA_CLASS_DIVERSITY.value,    "  Low Diversity     "),
            (KnowledgeGapType.GENERAL_UNCERTAINTY.value,          "  General Uncertainty"),
        ]:
            cnt = gap_counts.get(gv, 0)
            print(f"        {gname} : {cnt:>5}  ({cnt/n*100:5.1f}%)")
        print(f"      Strategies        :")
        for sv in [Strategy.RETRIEVAL.value, Strategy.COUNTEREXAMPLE.value]:
            cnt = str_counts.get(sv, 0)
            print(f"        {sv:<20} : {cnt:>5}  ({cnt/n*100:5.1f}%)")
        print(f"      Total retrievals  : {total_exp:,}")

    # ── Future strategy placeholders ─────────────────────────────────────────

    def retrieve_external_experience(self, *args, **kwargs):
        """
        Placeholder — External Experience.
        Planned for Version 2: retrieve samples from an external dataset.
        """
        raise NotImplementedError(
            "External Experience retrieval is planned for a future version "
            "of the AKRM. Version 1 uses only the original training dataset."
        )

    def retrieve_contextual_experience(self, *args, **kwargs):
        """
        Placeholder — Contextual Experience.
        Planned for Version 2: retrieve samples conditioned on the current
        training context (curriculum stage, task difficulty, etc.).
        """
        raise NotImplementedError(
            "Contextual Experience retrieval is planned for a future version "
            "of the AKRM."
        )

    def retrieve_generated_experience(self, *args, **kwargs):
        """
        Placeholder — Generated Experience.
        Planned for Version 2: generate synthetic samples using a generative
        model targeted at the diagnosed knowledge gap.
        """
        raise NotImplementedError(
            "Generated Experience retrieval is planned for a future version "
            "of the AKRM."
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_embeddings(self) -> None:
        """
        Extract 128-dim FC1 activations from all training samples.

        The forward pass stops after ReLU(FC1) — before Dropout and FC2.
        These features live in the model's learned representation space,
        where L2 distance is a meaningful proxy for semantic similarity.

        Architecture note: this accesses SimpleCNN layer names directly
        (conv1, conv2, pool, fc1).  If the model architecture changes,
        update this method accordingly.
        """
        self.model.eval()
        loader = DataLoader(
            self.train_dataset, batch_size=256,
            shuffle=False, num_workers=0,
        )

        all_embeddings: List[np.ndarray] = []
        all_labels:     List[np.ndarray] = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)

                # Manual forward: Conv1 → Conv2 → Flatten → FC1 (ReLU)
                x = self.model.pool(F.relu(self.model.conv1(images)))
                x = self.model.pool(F.relu(self.model.conv2(x)))
                x = x.view(x.size(0), -1)
                x = F.relu(self.model.fc1(x))          # (B, 128)

                all_embeddings.append(x.cpu().numpy())
                if isinstance(labels, torch.Tensor):
                    all_labels.append(labels.numpy())
                else:
                    all_labels.append(np.array(labels))

        self._embeddings   = np.concatenate(all_embeddings, axis=0)  # (N, 128)
        self._train_labels = np.concatenate(all_labels,     axis=0)  # (N,)

        print(f"    AKRM: Embeddings ready — "
              f"shape {self._embeddings.shape}  "
              f"classes {len(np.unique(self._train_labels))}")

    def _compute_density_dist(self, sample_idx: int, true_label: int) -> float:
        """
        Compute the L2 distance to the k-th nearest same-class neighbour.

        A large value indicates the sample is isolated in the feature space
        (sparse region of the class distribution).
        A small value indicates the sample is in a densely populated region.

        Using the k-th neighbour (rather than the mean) gives a stable
        measure of local density that is less sensitive to outliers.
        """
        same_class_mask             = (self._train_labels == true_label)
        same_class_mask[sample_idx] = False     # exclude self
        same_class_idx              = np.where(same_class_mask)[0]

        if len(same_class_idx) == 0:
            return 0.0

        sample_emb      = self._embeddings[sample_idx]
        diff            = self._embeddings[same_class_idx] - sample_emb
        # einsum is faster than np.sum(diff**2, axis=1) for large arrays
        dists_sq        = np.einsum("ij,ij->i", diff, diff)

        k        = min(self.config.density_k, len(same_class_idx))
        kth_sq   = np.partition(dists_sq, k - 1)[k - 1]

        return float(np.sqrt(kth_sq))

    def _cname(self, class_idx: int) -> str:
        """Return class name string or numeric fallback."""
        if 0 <= class_idx < len(self.class_names):
            return self.class_names[class_idx]
        return str(class_idx)
