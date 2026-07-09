"""
main.py
-------
Orchestrator for the Uncertainty-Driven Learning experiment.

Research Question
-----------------
  Can prediction uncertainty be interpreted as knowledge gaps and converted
  into targeted learning experiences that improve neural network learning
  better than conventional uncertainty-guided augmentation?

Pipeline overview
-----------------
  A  — Baseline: standard supervised training, no retraining
  B  — Random control: augmented random training samples
  C  — AKRM (proposed): uncertainty → knowledge gap diagnosis
          → learning objective → strategy → experience pool
  D  — Lowest-uncertainty control: augmented low-uncertainty samples

Experiment is repeated for 3 random seeds (42, 123, 999).
Final comparison reports mean ± std across seeds.

IMPORTANT: val and test sets are NEVER used for augmentation or retraining.
MC Dropout uncertainty is estimated only on the TRAINING dataset.

Usage
-----
  python main.py [options]

Options
-------
  --dataset        FashionMNIST | MNIST          (default: FashionMNIST)
  --epochs         initial training epochs       (default: 15)
  --retrain_epochs retraining epochs             (default: 8)
  --mc_samples     MC Dropout forward passes     (default: 30)
  --top_fraction   fraction of samples selected  (default: 0.10)
  --lr             learning rate (Adam)          (default: 0.001)
  --batch_size     mini-batch size               (default: 64)
  --seeds          random seeds (space-sep.)     (default: 42 123 999)
  --cpu            force CPU even if CUDA is available
  --output_dir     directory for all outputs     (default: outputs)
"""

import argparse
import copy
import os
import random
import sys

# Force UTF-8 output so Unicode characters in print statements work correctly
# on Windows consoles that default to cp1252 / cp850.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
from torch.utils.data import DataLoader

from akrm.orchestrator import AdaptiveKnowledgeReasoningModule
from akrm.diagnosis import AKRMConfig
from augment import augment_samples
from data import DATASET_META, get_dataloaders
from evaluate import (
    compute_ece,
    compute_mean_entropy,
    count_corrected_errors,
    evaluate_model,
    plot_confusion_matrices,
    plot_metrics_bar,
    plot_training_curves,
    plot_uncertainty_before_after,
    plot_uncertainty_distribution,
    save_cross_dataset_comparison,
    save_results_summary,
)
from model import get_model
from retrain import retrain_model
from train import train_model
from uncertainty import (
    generate_uncertainty_report,
    get_selection_indices,
    mc_dropout_predict,
    save_uncertainty_csv,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser",  "Pullover", "Dress",     "Coat",
    "Sandal",      "Shirt",    "Sneaker",  "Bag",        "Ankle boot",
]
MNIST_CLASSES  = [str(i) for i in range(10)]
CIFAR10_CLASSES = [
    "Airplane", "Automobile", "Bird",  "Cat",  "Deer",
    "Dog",      "Frog",       "Horse", "Ship", "Truck",
]

CLASS_NAMES = {
    "FashionMNIST": FASHION_MNIST_CLASSES,
    "MNIST":        MNIST_CLASSES,
    "CIFAR10":      CIFAR10_CLASSES,
}

PIPELINES      = ["A", "B", "C", "D"]
# Pipeline C uses AKRM (no simple selection mode); B and D use augmentation
PIPELINE_MODES = {"B": "random", "D": "lowest"}

# Human-readable labels used in plots and summary reports
PIPELINE_LABELS = {
    "A": "Pipeline A  Baseline",
    "B": "Pipeline B  Random Aug",
    "C": "Pipeline C  AKRM",
    "D": "Pipeline D  Least Uncertain Aug",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Global seed for reproducibility (Python, NumPy, PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


def _print_eval(tag: str, res: dict) -> None:
    print(
        f"    [{tag}]  Acc {res['accuracy']:.4f}  "
        f"Prec {res['precision']:.4f}  "
        f"Rec {res['recall']:.4f}  "
        f"F1 {res['f1']:.4f}  "
        f"ECE {res['ece']:.4f}  "
        f"Entropy {res['mean_entropy']:.4f}"
    )


def _banner(text: str, width: int = 64) -> None:
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}")


def _section(text: str, width: int = 64) -> None:
    print(f"\n  {'-' * (width - 2)}")
    print(f"  {text}")
    print(f"  {'-' * (width - 2)}")


# ─────────────────────────────────────────────────────────────────────────────
# Single-seed experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_single_seed(seed: int, args, class_names: list, device) -> tuple:
    """
    Execute the full 6-phase experiment for one random seed.

    Parameters
    ----------
    seed        : Random seed
    args        : Parsed CLI arguments
    class_names : List of class label strings
    device      : torch.device

    Returns
    -------
    results   : dict[pipeline → metrics dict]
    histories : dict[pipeline → training history dict]
    unc       : MC Dropout results dict (training set, before retraining)
    """
    _banner(f"SEED: {seed}", width=64)
    set_seed(seed)

    seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    # ── Phase 1: Load data and train baseline ──────────────────────────────
    _section("Phase 1 — Loading data and training baseline CNN")
    train_loader, val_loader, test_loader, train_dataset = get_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        val_split=0.10,
        seed=seed,
    )
    print()

    model = get_model(
        num_classes = 10,
        dropout_p   = 0.3,
        in_channels = DATASET_META[args.dataset]["in_channels"],
        image_size  = DATASET_META[args.dataset]["image_size"],
    )
    model, history_A = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, device=device,
    )
    torch.save(model.state_dict(), os.path.join(seed_dir, "baseline_model.pth"))
    print(f"\n    Baseline model saved → {seed_dir}/baseline_model.pth")

    # ── Phase 2: MC Dropout on TRAINING data ──────────────────────────────
    _section(f"Phase 2 — MC Dropout on training set  (T={args.mc_samples} passes)")
    # Inference has no backward pass, so a larger batch size is safe and
    # dramatically faster (fewer DataLoader iterations per pass).
    INFER_BS = 256
    train_loader_mc = DataLoader(
        train_dataset, batch_size=INFER_BS,
        shuffle=False, num_workers=0,
    )
    # Separate high-throughput test loader used for entropy / ECE evaluation
    test_loader_mc = DataLoader(
        test_loader.dataset, batch_size=INFER_BS,
        shuffle=False, num_workers=0,
    )
    unc = mc_dropout_predict(
        model, train_loader_mc,
        n_samples=args.mc_samples, device=device,
    )

    # ── Phase 3: Save uncertainty analysis ────────────────────────────────
    _section("Phase 3 — Saving uncertainty analysis")
    save_uncertainty_csv(
        unc, os.path.join(seed_dir, "uncertainty_analysis.csv")
    )
    report = generate_uncertainty_report(
        unc, class_names,
        os.path.join(seed_dir, "uncertainty_report.txt"),
        top_fraction=args.top_fraction,
    )
    print("\n" + report)

    # ── Evaluate Pipeline A ───────────────────────────────────────────────
    _section("Evaluating Pipeline A (baseline) on test set")
    results_A = evaluate_model(model, test_loader, device)
    results_A["ece"]          = compute_ece(model, test_loader_mc, device=device)
    results_A["mean_entropy"] = compute_mean_entropy(
        model, test_loader_mc, n_mc_samples=args.mc_samples, device=device
    )
    _print_eval("A", results_A)

    results   = {"A": results_A}
    histories = {"A": history_A}

    # ── Phases 4 & 5: Retrain Pipelines B, C, D ──────────────────────────
    for pipeline in ["B", "C", "D"]:

        if pipeline == "C":
            # ── Pipeline C: AKRM ─────────────────────────────────────────
            _section("Pipeline C  (AKRM — Adaptive Knowledge Reasoning)")

            uncertain_indices = get_selection_indices(
                unc["entropy"],
                top_fraction=args.top_fraction,
                mode="highest",
                seed=seed,
            )
            print(f"    Selected {len(uncertain_indices):,} most uncertain "
                  f"training samples for AKRM analysis")

            akrm = AdaptiveKnowledgeReasoningModule(
                model        = model,
                train_dataset= train_dataset,
                unc_results  = unc,
                device       = device,
                class_names  = class_names,
                config       = AKRMConfig(policy_type=args.akrm_policy)
            )
            experience_pool = akrm.run(uncertain_indices)
            akrm.print_summary()

            akrm.save_reasoning_report(
                os.path.join(seed_dir, "akrm_reasoning.csv")
            )
            akrm.save_diagnosis_validation(
                os.path.join(seed_dir, "diagnosis_validation.txt")
            )

            pool_or_aug = experience_pool
            pool_size   = len(experience_pool)

        else:
            # ── Pipelines B and D: traditional augmentation ───────────────
            mode = PIPELINE_MODES[pipeline]
            _section(f"Pipeline {pipeline}  (selection: {mode}, strategy: augmentation)")

            indices = get_selection_indices(
                unc["entropy"],
                top_fraction=args.top_fraction,
                mode=mode,
                seed=seed,
            )
            print(f"    Selected {len(indices):,} training samples")

            pool_or_aug = augment_samples(
                train_dataset, indices,
                n_aug_per_image=3, seed=seed,
            )
            pool_size   = len(pool_or_aug)

        print(f"\n    Retraining for {args.retrain_epochs} epochs...")
        retrained, history = retrain_model(
            model, train_dataset, pool_or_aug, val_loader,
            epochs=args.retrain_epochs, lr=args.lr,
            batch_size=args.batch_size, device=device,
        )

        print(f"\n    Evaluating Pipeline {pipeline} on test set...")
        res = evaluate_model(retrained, test_loader, device)
        res["ece"]          = compute_ece(retrained, test_loader_mc, device=device)
        res["mean_entropy"] = compute_mean_entropy(
            retrained, test_loader_mc, n_mc_samples=args.mc_samples, device=device
        )
        _print_eval(pipeline, res)

        results[pipeline]   = res
        histories[pipeline] = history

        torch.save(
            retrained.state_dict(),
            os.path.join(seed_dir, f"pipeline_{pipeline}_model.pth"),
        )

    return results, histories, unc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uncertainty-Guided Retraining Research Prototype"
    )
    parser.add_argument("--dataset",        type=str,   default="FashionMNIST",
                        choices=["FashionMNIST", "MNIST", "CIFAR10"],
                        help="Dataset to use (FashionMNIST | MNIST | CIFAR10)")
    parser.add_argument("--epochs",         type=int,   default=15,
                        help="Initial training epochs")
    parser.add_argument("--retrain_epochs", type=int,   default=8,
                        help="Retraining epochs for pipelines B/C/D")
    parser.add_argument("--mc_samples",     type=int,   default=30,
                        help="MC Dropout forward passes for uncertainty estimation")
    parser.add_argument("--top_fraction",   type=float, default=0.10,
                        help="Fraction of training samples to select/augment")
    parser.add_argument("--lr",             type=float, default=0.001,
                        help="Adam learning rate (same for training and retraining)")
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--akrm_policy",    type=str,   default="heuristic", choices=["heuristic", "random"], help="Policy for AKRM v2 strategy selection")
    parser.add_argument("--seeds",          type=int,   nargs="+",
                        default=[42, 123, 999],
                        help="List of random seeds for the repeated experiment")
    parser.add_argument("--cpu",            action="store_true",
                        help="Force CPU even when CUDA is available")
    parser.add_argument("--output_dir",     type=str,   default=None,
                        help="Output directory (default: outputs_{dataset})")
    args = parser.parse_args()

    # Auto-set output directory based on dataset if not explicitly provided
    if args.output_dir is None:
        args.output_dir = f"outputs_{args.dataset.lower()}"

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    class_names = CLASS_NAMES.get(args.dataset, MNIST_CLASSES)

    _banner("UNCERTAINTY-GUIDED RETRAINING — RESEARCH PROTOTYPE", width=68)
    print(f"  Dataset          : {args.dataset}")
    print(f"  Initial epochs   : {args.epochs}")
    print(f"  Retraining epochs: {args.retrain_epochs}")
    print(f"  MC samples       : {args.mc_samples}")
    print(f"  Top fraction     : {args.top_fraction * 100:.0f}%")
    print(f"  Seeds            : {args.seeds}")
    print(f"  Device           : {device}")
    print(f"  Output dir       : {args.output_dir}/")

    # ── Run experiment for each seed ───────────────────────────────────────
    all_results   = {p: [] for p in PIPELINES}
    all_histories = {p: [] for p in PIPELINES}
    rep_results   = None   # representative seed (first) for per-sample plots
    rep_unc       = None

    for seed in args.seeds:
        seed_results, seed_histories, unc = run_single_seed(
            seed, args, class_names, device
        )
        for p in PIPELINES:
            all_results[p].append(seed_results[p])
            all_histories[p].append(seed_histories[p])

        if rep_results is None:
            rep_results = seed_results
            rep_unc     = unc

    # ── Aggregate across seeds ─────────────────────────────────────────────
    _banner("AGGREGATING RESULTS ACROSS SEEDS", width=64)

    agg_metrics = ["accuracy", "precision", "recall", "f1", "ece", "mean_entropy"]
    summary = {}
    for p in PIPELINES:
        summary[p] = {}
        for m in agg_metrics:
            vals = [r[m] for r in all_results[p]]
            summary[p][m] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
            }

    print(f"\n  {'Pipeline':<8}  {'Accuracy':>18}  {'F1':>18}  {'ECE':>18}  {'Entropy':>18}")
    print("  " + "-" * 80)
    for p in PIPELINES:
        print(
            f"  {p:<8}  "
            f"{summary[p]['accuracy']['mean']:.4f}±{summary[p]['accuracy']['std']:.4f}  "
            f"{summary[p]['f1']['mean']:.4f}±{summary[p]['f1']['std']:.4f}  "
            f"{summary[p]['ece']['mean']:.4f}±{summary[p]['ece']['std']:.4f}  "
            f"{summary[p]['mean_entropy']['mean']:.4f}±{summary[p]['mean_entropy']['std']:.4f}"
        )

    # ── Corrected errors (representative seed) ────────────────────────────
    preds_A     = rep_results["A"]["predictions"]
    true_labels = rep_results["A"]["true_labels"]
    corrected_errors = {}
    for p in ["B", "C", "D"]:
        corrected_errors[p] = count_corrected_errors(
            preds_A, rep_results[p]["predictions"], true_labels
        )

    # ── Generate plots ─────────────────────────────────────────────────────
    _banner("GENERATING PLOTS", width=64)

    plot_training_curves(
        all_histories, args.seeds,
        os.path.join(args.output_dir, "training_curves.png"),
    )
    plot_metrics_bar(
        summary,
        os.path.join(args.output_dir, "metrics_bar.png"),
    )
    plot_confusion_matrices(
        rep_results, class_names,
        os.path.join(args.output_dir, "confusion_matrices.png"),
    )
    plot_uncertainty_distribution(
        rep_unc["entropy"], args.top_fraction,
        os.path.join(args.output_dir, "uncertainty_distribution.png"),
    )
    plot_uncertainty_before_after(
        summary,
        os.path.join(args.output_dir, "uncertainty_before_after.png"),
    )

    # ── Save results summary ───────────────────────────────────────────────
    _banner("SAVING RESULTS SUMMARY", width=64)
    report = save_results_summary(
        summary, corrected_errors, args,
        os.path.join(args.output_dir, "results_summary.txt"),
    )
    print("\n" + report)

    # ── Save aggregated JSON (enables cross-dataset comparison) ───────────
    import json
    agg_json_path = os.path.join(args.output_dir, "aggregated_results.json")
    agg_data = {
        "dataset":         args.dataset,
        "epochs":          args.epochs,
        "retrain_epochs":  args.retrain_epochs,
        "mc_samples":      args.mc_samples,
        "top_fraction":    args.top_fraction,
        "seeds":           args.seeds,
        "summary":         summary,
    }
    with open(agg_json_path, "w", encoding="utf-8") as fh:
        json.dump(agg_data, fh, indent=2)
    print(f"  Aggregated JSON saved → {agg_json_path}")

    # ── Cross-dataset comparison (if both datasets' JSONs exist) ──────────
    other_dataset = "FashionMNIST" if args.dataset == "CIFAR10" else "CIFAR10"
    other_dir     = f"outputs_{other_dataset.lower()}"
    # Also check the legacy 'outputs' directory for Fashion-MNIST results
    other_json_paths = [
        os.path.join(other_dir, "aggregated_results.json"),
        os.path.join("outputs", "aggregated_results.json"),   # legacy path
    ]
    other_json = next((p for p in other_json_paths if os.path.exists(p)), None)
    if other_json:
        with open(other_json, "r", encoding="utf-8") as fh:
            other_data = json.load(fh)
        cross_path = os.path.join(args.output_dir, "cross_dataset_comparison.txt")
        save_cross_dataset_comparison(
            agg_data, other_data,
            cross_path,
        )
        print(f"  Cross-dataset comparison → {cross_path}")

    _banner("EXPERIMENT COMPLETE", width=64)
    print(f"  All outputs saved to  →  {args.output_dir}/")
    print(f"  Per-seed outputs      →  {args.output_dir}/seed_<N>/")
    print()


if __name__ == "__main__":
    main()
