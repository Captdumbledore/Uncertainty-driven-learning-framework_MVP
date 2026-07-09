"""
uncertainty.py
--------------
Monte Carlo Dropout uncertainty estimation — Phases 2 & 3.

All functions here operate exclusively on the TRAINING dataset.
The validation and test sets are never touched.

Theory
------
  Standard MC Dropout (Gal & Ghahramani, 2016):
    1. Keep dropout active at inference time.
    2. Run T stochastic forward passes on each sample.
    3. Average the T softmax probability vectors → mean distribution p̄.
    4. Predictive entropy:  H = -Σ_c  p̄_c · log(p̄_c + ε)

  High entropy  → model is uncertain → candidate for targeted retraining.
  Low entropy   → model is confident → used as control (Pipeline D).

Outputs
-------
  uncertainty_analysis.csv   — per-sample entropy / confidence / correctness
  uncertainty_report.txt     — pre-retraining diagnostic analysis (4 questions)
"""

import csv
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import enable_dropout


# ─────────────────────────────────────────────────────────────────────────────
# Core MC Dropout inference
# ─────────────────────────────────────────────────────────────────────────────

def mc_dropout_predict(
    model,
    data_loader,
    n_samples: int = 30,
    device         = "cpu",
) -> dict:
    """
    Run Monte Carlo Dropout on data_loader (TRAINING data only).

    Performs n_samples stochastic forward passes with dropout enabled,
    then averages the softmax probability vectors per sample.

    Parameters
    ----------
    model       : Trained SimpleCNN
    data_loader : DataLoader with shuffle=False (preserves index alignment)
    n_samples   : Number of stochastic forward passes (T = 30)
    device      : torch.device or string

    Returns
    -------
    dict with numpy arrays of shape (N,) or (N, C):
      "mean_probs"        : averaged softmax distributions  (N, C)
      "predicted_classes" : argmax of mean_probs            (N,)
      "confidence"        : max of mean_probs               (N,)
      "entropy"           : predictive entropy H            (N,)
      "true_labels"       : ground-truth labels             (N,)
    """
    # ── Collect ground-truth labels ────────────────────────────────────────
    true_labels = []
    for _, labels in data_loader:
        true_labels.append(labels.numpy())
    true_labels = np.concatenate(true_labels)   # (N,)

    # ── T stochastic forward passes ────────────────────────────────────────
    stochastic_probs = []

    for _ in tqdm(range(n_samples), desc="    MC Dropout passes", ncols=65, leave=False):
        enable_dropout(model)   # keep dropout ON, BatchNorm stays eval
        batch_probs = []

        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(device)
                logits = model(images)
                probs  = F.softmax(logits, dim=1).cpu().numpy()
                batch_probs.append(probs)

        stochastic_probs.append(np.concatenate(batch_probs, axis=0))  # (N, C)

    model.eval()   # reset to full eval mode after MC passes

    # ── Aggregate ──────────────────────────────────────────────────────────
    stochastic_probs = np.stack(stochastic_probs, axis=0)  # (T, N, C)
    mean_probs       = stochastic_probs.mean(axis=0)        # (N, C)

    predicted_classes = mean_probs.argmax(axis=1)           # (N,)
    confidence        = mean_probs.max(axis=1)              # (N,)
    entropy           = -np.sum(
        mean_probs * np.log(mean_probs + 1e-8), axis=1
    )                                                       # (N,)

    print(
        f"    MC Dropout complete — "
        f"N={len(true_labels):,}  "
        f"mean entropy={entropy.mean():.4f}  "
        f"mean confidence={confidence.mean():.4f}"
    )

    return {
        "mean_probs":        mean_probs,
        "predicted_classes": predicted_classes,
        "confidence":        confidence,
        "entropy":           entropy,
        "true_labels":       true_labels,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sample selection
# ─────────────────────────────────────────────────────────────────────────────

def get_selection_indices(
    entropy_scores: np.ndarray,
    top_fraction:   float = 0.10,
    mode:           str   = "highest",
    seed:           int   = 42,
) -> np.ndarray:
    """
    Select a fraction of training sample indices based on entropy ranking.

    Parameters
    ----------
    entropy_scores : Predictive entropy per training sample  (N,)
    top_fraction   : Fraction to select, e.g. 0.10 for 10%
    mode           : Selection strategy —
                       "highest" → top-10% most uncertain  (Pipeline C)
                       "lowest"  → top-10% least uncertain (Pipeline D)
                       "random"  → random 10%              (Pipeline B)
    seed           : Random seed (only used when mode="random")

    Returns
    -------
    indices : ndarray of integer indices into the training dataset
    """
    n_total  = len(entropy_scores)
    n_select = max(1, int(n_total * top_fraction))

    if mode == "highest":
        indices = np.argsort(entropy_scores)[-n_select:]
    elif mode == "lowest":
        indices = np.argsort(entropy_scores)[:n_select]
    elif mode == "random":
        rng     = np.random.default_rng(seed)
        indices = rng.choice(n_total, size=n_select, replace=False)
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Use 'highest', 'lowest', or 'random'."
        )

    return indices


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_uncertainty_csv(results: dict, save_path: str) -> None:
    """
    Save per-sample uncertainty analysis to a CSV file.

    Columns
    -------
      sample_id          : Index within the training dataset
      true_label         : Ground-truth class
      predicted_label    : MC-Dropout predicted class (argmax of mean_probs)
      confidence         : Maximum entry in mean_probs
      predictive_entropy : H = -Σ p̄_c log(p̄_c + ε)
      correct_prediction : True if predicted == true
    """
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    correct = (results["true_labels"] == results["predicted_classes"])

    with open(save_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "sample_id", "true_label", "predicted_label",
            "confidence", "predictive_entropy", "correct_prediction",
        ])
        n = len(results["true_labels"])
        for i in range(n):
            writer.writerow([
                i,
                int(results["true_labels"][i]),
                int(results["predicted_classes"][i]),
                f"{results['confidence'][i]:.6f}",
                f"{results['entropy'][i]:.6f}",
                bool(correct[i]),
            ])

    print(f"    Saved uncertainty CSV ({n:,} rows) → {save_path}")


def generate_uncertainty_report(
    results:      dict,
    class_names:  list,
    save_path:    str,
    top_fraction: float = 0.10,
) -> str:
    """
    Generate a plain-text pre-retraining analysis report.

    Diagnostic questions answered
    ------------------------------
      Q1  What % of the top-uncertain samples are actually misclassified?
      Q2  Which digit/class is most frequent among uncertain samples?
      Q3  Pearson correlation between predictive entropy and prediction error?
      Q4  Mean entropy for correct vs. incorrect predictions?

    Returns the report as a string and writes it to save_path.
    """
    entropy      = results["entropy"]
    true_labels  = results["true_labels"]
    predicted    = results["predicted_classes"]
    correct_mask = (true_labels == predicted)
    errors       = (~correct_mask).astype(int)

    n_total  = len(entropy)
    n_select = max(1, int(n_total * top_fraction))
    top_idx  = np.argsort(entropy)[-n_select:]   # indices of most uncertain

    # Q1
    top_correct    = correct_mask[top_idx]
    pct_misclf     = (1 - top_correct.mean()) * 100
    pct_already_ok = top_correct.mean() * 100

    # Q2
    top_labels   = true_labels[top_idx]
    class_counts = Counter(int(lbl) for lbl in top_labels)

    # Q3  Pearson r(entropy, error)
    corr = float(np.corrcoef(entropy, errors)[0, 1])

    # Q4  Entropy split by correctness
    ent_correct   = entropy[correct_mask]
    ent_incorrect = entropy[~correct_mask]

    lines = [
        "=" * 64,
        "  PRE-RETRAINING UNCERTAINTY ANALYSIS REPORT",
        "=" * 64,
        "",
        f"  Total training samples              : {n_total:,}",
        f"  MC-Dropout accuracy on train set    : {correct_mask.mean()*100:.2f}%",
        f"  Top-{top_fraction*100:.0f}% uncertain samples selected : {n_select:,}",
        "",
        "  " + "─" * 60,
        "  Q1  Are the most uncertain samples actually misclassified?",
        "  " + "─" * 60,
        f"      % misclassified (among top uncertain)     : {pct_misclf:.2f}%",
        f"      % correctly classified (yet still uncertain): {pct_already_ok:.2f}%",
        "",
        "  " + "─" * 60,
        "  Q2  Which classes appear most among uncertain samples?",
        "  " + "─" * 60,
    ]

    for c in sorted(class_counts, key=lambda x: class_counts[x], reverse=True):
        name  = class_names[c] if c < len(class_names) else str(c)
        count = class_counts[c]
        pct   = count / n_select * 100
        lines.append(f"      {name:<24} {count:>5}  ({pct:5.1f}%)")

    if corr > 0.15:
        interp = "Positive — entropy is a meaningful signal for prediction errors."
    elif corr > 0.0:
        interp = "Weak positive — entropy imperfectly predicts errors."
    else:
        interp = "Near-zero or negative — entropy does not reliably predict errors."

    lines += [
        "",
        "  " + "─" * 60,
        "  Q3  Correlation: predictive entropy ↔ prediction error",
        "  " + "─" * 60,
        f"      Pearson r(entropy, error) = {corr:.4f}",
        f"      Interpretation: {interp}",
        "",
        "  " + "─" * 60,
        "  Q4  Entropy statistics by prediction correctness",
        "  " + "─" * 60,
    ]

    if correct_mask.sum() > 0:
        lines.append(
            f"      Mean entropy | correct predictions   : {ent_correct.mean():.4f}"
            f"  (n={correct_mask.sum():,})"
        )
    if (~correct_mask).sum() > 0:
        lines.append(
            f"      Mean entropy | incorrect predictions : {ent_incorrect.mean():.4f}"
            f"  (n={(~correct_mask).sum():,})"
        )

    lines += ["", "=" * 64]
    report = "\n".join(lines)

    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(f"    Saved uncertainty report → {save_path}")
    return report
