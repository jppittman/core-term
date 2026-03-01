#!/usr/bin/env python3
"""
Simple Bayesian Optimization for Guide hyperparameters.

Uses Thompson Sampling with a simple surrogate model.
"""

import subprocess
import re
import random
import math
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Trial:
    lr: float
    epochs: int
    seed: int
    accuracy: float
    fp_rate: float
    fn_rate: float

def run_training(lr: float, epochs: int, seed: int) -> Tuple[float, float, float]:
    """Run training and return (accuracy, fp_rate, fn_rate)."""
    cmd = [
        "cargo", "run", "-p", "pixelflow-pipeline",
        "--bin", "train_guide", "--release", "--",
        "--epochs", str(epochs),
        "--learning-rate", str(lr),
        "--seed", str(seed),
        "--print-every", str(epochs + 1)  # Don't print intermediate
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/Users/jppittman/Documents/Projects/core-term")
    output = result.stdout + result.stderr

    # Parse accuracy
    acc_match = re.search(r"Best validation accuracy: ([\d.]+)%", output)
    accuracy = float(acc_match.group(1)) if acc_match else 0.0

    # Parse FP/FN from final evaluation
    fp_match = re.search(r"False positive rate: ([\d.]+)%", output)
    fn_match = re.search(r"False negative rate: ([\d.]+)%", output)
    fp_rate = float(fp_match.group(1)) if fp_match else 100.0
    fn_rate = float(fn_match.group(1)) if fn_match else 100.0

    return accuracy, fp_rate, fn_rate

def sample_hyperparams(trials: List[Trial], iteration: int) -> Tuple[float, int, int]:
    """Sample next hyperparameters using Thompson Sampling."""

    # Learning rate: log-uniform in [0.0001, 0.05]
    # Epochs: uniform in [50, 500]
    # Seed: random

    if iteration < 5:
        # Initial exploration phase - Latin hypercube-ish sampling
        lr = 10 ** random.uniform(-4, -1.3)  # 0.0001 to 0.05
        epochs = random.randint(50, 500)
        seed = random.randint(1, 10000)
    else:
        # Exploitation: sample near best performers with some noise
        best_trials = sorted(trials, key=lambda t: -t.accuracy)[:3]

        # Pick one of the best and perturb
        base = random.choice(best_trials)

        # Add noise (Thompson sampling style)
        lr_log = math.log10(base.lr) + random.gauss(0, 0.3)
        lr = 10 ** max(-4, min(-1.3, lr_log))

        epochs = base.epochs + random.randint(-50, 50)
        epochs = max(50, min(500, epochs))

        seed = random.randint(1, 10000)

    return lr, epochs, seed

def main():
    print("=" * 60)
    print("Bayesian Optimization for Guide Hyperparameters")
    print("=" * 60)
    print()

    trials: List[Trial] = []
    best_accuracy = 0.0
    best_trial = None

    n_iterations = 20

    for i in range(n_iterations):
        lr, epochs, seed = sample_hyperparams(trials, i)

        print(f"[{i+1}/{n_iterations}] lr={lr:.6f}, epochs={epochs}, seed={seed}")
        print("  Running training...", end=" ", flush=True)

        accuracy, fp_rate, fn_rate = run_training(lr, epochs, seed)

        trial = Trial(lr=lr, epochs=epochs, seed=seed,
                     accuracy=accuracy, fp_rate=fp_rate, fn_rate=fn_rate)
        trials.append(trial)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_trial = trial
            print(f"acc={accuracy:.1f}% (NEW BEST!) FP={fp_rate:.1f}% FN={fn_rate:.1f}%")
        else:
            print(f"acc={accuracy:.1f}% FP={fp_rate:.1f}% FN={fn_rate:.1f}%")

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print(f"Best accuracy: {best_accuracy:.1f}%")
    if best_trial:
        print(f"  lr={best_trial.lr:.6f}")
        print(f"  epochs={best_trial.epochs}")
        print(f"  seed={best_trial.seed}")
        print(f"  FP rate={best_trial.fp_rate:.1f}%")
        print(f"  FN rate={best_trial.fn_rate:.1f}%")

    print()
    print("Top 5 trials:")
    for i, t in enumerate(sorted(trials, key=lambda x: -x.accuracy)[:5]):
        print(f"  {i+1}. acc={t.accuracy:.1f}% lr={t.lr:.6f} epochs={t.epochs} seed={t.seed}")

if __name__ == "__main__":
    random.seed(42)
    main()
