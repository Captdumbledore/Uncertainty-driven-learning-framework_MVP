import numpy as np
import torch
from torch.utils.data import Subset
from typing import Optional, List, Dict, Any

from akrm.diagnosis import (
    AKRMConfig, 
    UncertaintyAnalysisEngine, 
    KnowledgeGapDiagnoser
)
from akrm.objective import LearningObjectiveGenerator
from akrm.planner import KnowledgeGuidedExperiencePlanner, ProviderType
from akrm.provider.retrieval import RetrievalProvider
from akrm.provider.counterexample import CounterexampleProvider
from akrm.provider.contextual import ContextualProvider
from akrm.provider.synthetic import SyntheticProvider
from akrm.provider.external import ExternalProvider
from akrm.validator import ExperienceValidator
from akrm.executor import LearningStrategyExecutor

class AdaptiveKnowledgeReasoningModule:
    """
    Orchestrates the Adaptive Learning Experience Framework pipeline.
    
    Pipeline:
    MC Dropout -> Diagnosis -> Objective Generation -> Adaptive Experience Planner 
    -> Experience Provider -> Experience Validator -> Learning Strategy Executor
    """
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Subset,
        unc_results: dict,
        device: torch.device,
        class_names: list,
        config: Optional[AKRMConfig] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.unc_results = unc_results
        self.device = device
        self.class_names = class_names
        self.config = config or AKRMConfig()

        self._embeddings: Optional[np.ndarray] = None
        self._train_labels: Optional[np.ndarray] = None
        self._reasoning_log: List[dict] = []

        # Core Modules
        self._analysis_engine = UncertaintyAnalysisEngine(unc_results)
        self._diagnoser = KnowledgeGapDiagnoser(self.config)
        self._objective_gen = LearningObjectiveGenerator()
        self._planner = KnowledgeGuidedExperiencePlanner(policy_type=self.config.policy_type)
        self._validator = ExperienceValidator()
        self._executor = LearningStrategyExecutor()
        
        # Providers (instantiated after embeddings are ready)
        self._providers: Dict[ProviderType, Any] = {}

    def _extract_embeddings(self):
        """Extract embeddings for the training dataset."""
        self.model.eval()
        embeddings, labels = [], []
        # Fallback to dummy extraction if not fully implemented in this MVP
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        
        loader = DataLoader(self.train_dataset, batch_size=256, shuffle=False)
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                emb = self.model.extract_features(x)
                embeddings.append(emb.cpu().numpy())
                labels.append(y.numpy())
        self._embeddings = np.concatenate(embeddings)
        self._train_labels = np.concatenate(labels)

    def _compute_density_dist(self, sample_idx: int, true_label: int) -> float:
        """Computes distance to k-th nearest same-class neighbour."""
        same_class_mask = (self._train_labels == true_label)
        same_class_mask[sample_idx] = False
        same_class_idx = np.where(same_class_mask)[0]
        
        if len(same_class_idx) < self.config.density_k:
            return 999.0
            
        sample_emb = self._embeddings[sample_idx]
        same_class_embs = self._embeddings[same_class_idx]
        dists = np.linalg.norm(same_class_embs - sample_emb, axis=1)
        kth_dist = float(np.sort(dists)[self.config.density_k - 1])
        return kth_dist

    def run(self, uncertain_indices: np.ndarray) -> Subset:
        """
        Execute the full framework.
        """
        print(f"\n    AKRM v2: Extracting embeddings...")
        self._extract_embeddings()
        
        # Initialize Providers
        self._providers = {
            ProviderType.RETRIEVAL: RetrievalProvider(self._train_labels, self._embeddings, self.config.n_retrieve_retrieval),
            ProviderType.COUNTEREXAMPLE: CounterexampleProvider(self._train_labels, self._embeddings, self.config.n_retrieve_counterexample),
            ProviderType.CONTEXTUAL: ContextualProvider(),
            ProviderType.SYNTHETIC: SyntheticProvider(),
            ProviderType.EXTERNAL: ExternalProvider(),
        }

        print(f"    AKRM v2: Pass 1 — Diagnosis ({len(uncertain_indices):,} samples)...")
        analyses, density_dists = [], []
        for idx in uncertain_indices:
            analysis = self._analysis_engine.analyse(int(idx))
            density_dist = self._compute_density_dist(int(idx), analysis["true_label"])
            analyses.append(analysis)
            density_dists.append(density_dist)

        density_threshold = float(np.percentile(density_dists, self.config.sparse_density_percentile))

        print(f"    AKRM v2: Pass 2 — Generating Objectives & Planning...")
        experience_pool_indices = set()
        
        for analysis, density_dist in zip(analyses, density_dists):
            # 1. Diagnosis
            gap = self._diagnoser.diagnose(analysis, density_dist, density_threshold, self.class_names)
            
            # 2. Objective Generation
            objective = self._objective_gen.generate(gap)
            
            # 3. Adaptive Planning
            provider_type = self._planner.select_strategy(objective, gap)
            provider = self._providers[provider_type]
            
            # 4. Experience Provision
            if provider_type == ProviderType.COUNTEREXAMPLE:
                raw_experiences = provider.provide(
                    gap.sample_idx, objective, true_class=gap.true_class, confused_class=gap.top2_class
                )
            else:
                raw_experiences = provider.provide(gap.sample_idx, objective)
                
            # 5. Experience Validation
            valid_experiences = self._validator.validate(raw_experiences, objective)
            
            # For MVP: Add validated indices to pool
            experience_pool_indices.update(valid_experiences)
            
            # Logging
            self._reasoning_log.append({
                **analysis,
                "knowledge_gap": gap.gap_type.value,
                "gap_explanation": gap.explanation,
                "learning_objective": objective.value,
                "selected_provider": provider_type.value,
                "n_valid_experiences": len(valid_experiences),
            })

        pool_list = sorted(list(experience_pool_indices))
        final_subset = Subset(self.train_dataset, pool_list)
        
        print(f"    AKRM v2: Curated Experience Dataset size: {len(final_subset):,}")
        
        # 6. Learning Strategy Executor
        # In a fully integrated system, the executor would take over the training loop here.
        # For this prototype, we return the subset to the main pipeline.
        self._executor.execute(self.model, final_subset)
        
        return final_subset

    def print_summary(self):
        print(f"    AKRM v2: Strategy selection summary:")
        from collections import Counter
        counts = Counter(log['selected_provider'] for log in self._reasoning_log)
        for provider, count in counts.items():
            print(f"      {provider}: {count:,}")
        
    def save_reasoning_report(self, save_path: str):
        import pandas as pd
        import os
        if not self._reasoning_log:
            return
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        df = pd.DataFrame(self._reasoning_log)
        df.to_csv(save_path, index=False)
        print(f"    AKRM v2: Reasoning report saved to {save_path}")
