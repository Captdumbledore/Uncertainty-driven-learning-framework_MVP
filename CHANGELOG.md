# Changelog

All notable changes to this project will be documented in this file.

## [v1.0.0-mvp] - 2026-07-11

### Added
- **Initial stable MVP release.**
- **AKRM Reasoning Pipeline:** Introduced the Adaptive Knowledge Reasoning Module to diagnose uncertainty.
- **Knowledge-Guided Planner:** Implements strategies to translate knowledge gaps into learning objectives.
- **Experience Providers:**
  - `Retrieval Provider` (Operational)
  - `Counterexample Provider` (Operational)
  - `Synthetic Provider` (Prototype)
  - `Contextual Provider` (Prototype)
- **Learning Protocols:**
  - Offline Retraining
  - Replay Buffer
  - Mixed Replay
  - Curriculum Replay
  - Knowledge Distillation
- **Evaluation Framework:** Automated metric computation (Accuracy, F1, ECE, Entropy) and plotting tools.
- **Multi-seed Experiments:** Deterministic sequential pipeline execution across multiple random seeds for robustness.
- **CIFAR-10 Evaluation:** Comprehensive validated results demonstrating the superiority of AKRM + Knowledge Distillation in mitigating catastrophic forgetting.
