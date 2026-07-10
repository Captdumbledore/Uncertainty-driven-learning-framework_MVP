<div align="center">
  
# 🧠 Uncertainty-Guided Learning Framework (AKRM)
**A Research Prototype for Adaptive Knowledge Reasoning in Neural Networks**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-EE4C2C?logo=pytorch&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#)

*Investigating whether prediction uncertainty can be interpreted as knowledge gaps to guide targeted learning experiences.*

</div>

---

## 📖 Overview

Traditional uncertainty-guided learning often follows a simple reactive pipeline: **Predict Uncertainty → Apply Fixed Augmentation → Retrain**. 

This research project proposes and evaluates a fundamental redesign: the **Adaptive Knowledge Reasoning Module (AKRM)**. Instead of blindly augmenting uncertain samples, the AKRM introduces a reasoning layer that asks *why* the model is uncertain, diagnosing specific knowledge gaps before selecting a targeted learning strategy.

### Core Philosophy
1. **Never directly react to uncertainty.** First, interpret it.
2. **Convert uncertainty into knowledge gaps** (e.g., *Decision Boundary Confusion*, *Low Intra-Class Diversity*).
3. **Map gaps to targeted learning experiences** (e.g., *Counterexample Learning*, *Targeted Retrieval*).

---

## 🔬 Experimental Pipelines

To isolate the effect of reasoning, this repository implements four parallel retraining pipelines under identical experimental conditions (using Monte Carlo Dropout for uncertainty estimation):

* **Pipeline A (Baseline):** Standard CNN trained on the dataset without retraining.
* **Pipeline B (Random Augmentation):** Selects the most uncertain samples and applies standard data augmentation (rotation, translation, jitter, noise).
* **Pipeline C (AKRM - Our Approach):** Passes uncertain samples through the reasoning module to diagnose knowledge gaps and build a curated experience pool from real training samples.
* **Pipeline D (Least-Uncertain Control):** A sanity check that augments the *most confident* (least uncertain) samples to ensure improvements aren't just due to seeing more data.

---

## 📊 Key Findings

We evaluated the framework on two distinct datasets to study how dataset complexity influences the effectiveness of reasoning-based learning. 

### Experiment 1: Fashion-MNIST
* **The Reasoning Layer Works:** AKRM successfully identified meaningful gap types (e.g., ~30% of uncertain samples were diagnosed with boundary confusion, aligning with known difficult classes like Shirt/Coat).
* **Significant Entropy Reduction:** AKRM retraining reduced mean predictive entropy by **23.2%** (more than any other pipeline), proving the model changes how it processes uncertain inputs.
* **Dataset Ceiling:** The accuracy on Fashion-MNIST was saturated (~91.7%). While AKRM outperformed random augmentation (Pipeline B), neither could beat the baseline. Furthermore, AKRM suffered from calibration drift (increased Expected Calibration Error).

### Experiment 2: CIFAR-10 and Knowledge Distillation
* *The CIFAR-10 experiment validated the robustness of AKRM on complex datasets, but revealed a new challenge: **Catastrophic Forgetting**.*
* **The Catastrophic Forgetting Problem:** When retraining on the difficult, high-entropy samples identified by AKRM, standard Offline Retraining, Replay Buffers, and Curriculum Learning all resulted in the network forgetting its core dataset distribution, leading to accuracy drops.
* **The Solution (Knowledge Distillation):** By freezing the original network to act as a "Teacher", and training a brand new "Student" model from scratch on the combined dataset using soft-target probabilities, we completely bypassed catastrophic forgetting. 
* **Findings (H1 Supported):** AKRM + Knowledge Distillation (`0.7350` Accuracy) comprehensively outperformed the Baseline (`0.7258`), Random Augmentation (`0.7180`), and Least-Uncertain Augmentation (`0.7221`). Expected Calibration Error (ECE) also dropped significantly from 0.1613 to 0.1475.

---

## ⚙️ Architecture

The codebase is strictly modular, avoiding unnecessary engineering abstractions to preserve scientific explainability.

```text
├── akrm.py          # Core Adaptive Knowledge Reasoning Module (5-stage pipeline)
├── augment.py       # Standard augmentation strategies for baseline pipelines
├── data.py          # Deterministic data loading (FashionMNIST, CIFAR-10)
├── evaluate.py      # Metric computation (Accuracy, F1, ECE, Entropy) & Plotting
├── model.py         # Simple CNN Architecture (greyscale and RGB support)
├── retrain.py       # Retraining loop supporting both Tensor and Subset data
└── main.py          # Main execution script running the 4-pipeline comparison
```

---

## 🚀 How to Run

### Local Execution (CPU/GPU)
```bash
# 1. Clone the repository
git clone https://github.com/Captdumbledore/akrm-learning-framework.git
cd akrm-learning-framework

# 2. Run the Fashion-MNIST experiment
python main.py --dataset FashionMNIST

# 3. Run the CIFAR-10 experiment
python main.py --dataset CIFAR10
```

### Google Colab (Recommended for Speed)
Because Monte Carlo Dropout and sequential retraining can be computationally heavy, running this on a Colab GPU is highly recommended:
1. Upload the project files to Colab.
2. Ensure you select a **T4 GPU** runtime (`Runtime > Change runtime type`).
3. Run `!python main.py --dataset CIFAR10`.

---

## 📝 Conclusion

The AKRM framework successfully demonstrates that uncertainty can be systematically mapped to human-interpretable knowledge gaps. We proved that targeted experience retrieval reduces overall model uncertainty and corrects classification errors more effectively than standard augmentation techniques.

Crucially, we discovered that **Knowledge Distillation is the mathematically correct mechanism for injecting targeted reasoning into pre-trained networks**. While standard retraining methods suffer from catastrophic forgetting, distillation preserves foundational knowledge while absorbing complex edge cases, yielding strict accuracy and calibration improvements. Dataset complexity plays a critical role: reasoning-based learning provides the most value when the decision boundaries are complex enough that simple augmentation fails to provide meaningful new information.

<div align="center">
  <i>Built as a B.Tech Final Year Research Prototype.</i>
</div>
