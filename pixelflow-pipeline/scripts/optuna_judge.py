#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for the Judge (Value Head).

Tunes: learning_rate, momentum, weight_decay, batch_size, epochs.

Usage:
    python optuna_judge.py --n-trials 50
    python optuna_judge.py --n-trials 100 --study-name my_study --storage sqlite:///optuna.db

Requirements:
    pip install optuna
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import optuna
except ImportError:
    print("ERROR: optuna not installed. Run: pip install optuna")
    sys.exit(1)


def find_workspace_root() -> Path:
    """Find the workspace root by looking for Cargo.toml with [workspace]."""
    current = Path.cwd()
    while current != current.parent:
        cargo_toml = current / "Cargo.toml"
        if cargo_toml.exists():
            content = cargo_toml.read_text()
            if "[workspace]" in content:
                return current
        current = current.parent
    return Path.cwd()


def run_training(
    workspace_root: Path,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    seed: int,
) -> float | None:
    """
    Run train_judge with the given hyperparameters.
    Returns the best validation loss, or None if training failed.
    """
    cmd = [
        "cargo", "run", "-p", "pixelflow-pipeline",
        "--bin", "train_judge", "--release",
        "--features", "training",
        "--",
        "--learning-rate", str(learning_rate),
        "--momentum", str(momentum),
        "--weight-decay", str(weight_decay),
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--print-every", "20",
        "--output", f"pixelflow-pipeline/data/judge_trial_{seed}.bin",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=workspace_root,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
    except subprocess.TimeoutExpired:
        print("  Training timed out")
        return None

    if result.returncode != 0:
        print(f"  Training failed: {result.stderr[:500]}")
        return None

    # Parse the best validation loss from output
    # Looking for: "Best validation loss: 0.123456 at epoch 42"
    match = re.search(r"Best validation loss: ([\d.]+)", result.stdout)
    if match:
        return float(match.group(1))

    # Fallback: look in metadata file
    meta_path = workspace_root / f"pixelflow-pipeline/data/judge_trial_{seed}.meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return meta.get("best_val_loss")

    print("  Could not parse validation loss from output")
    return None


def objective(trial: optuna.Trial, workspace_root: Path, base_seed: int) -> float:
    """Optuna objective function."""
    # Sample hyperparameters
    # Narrowed LR range based on initial results: high LRs (>0.01) gave poor results
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    momentum = trial.suggest_float("momentum", 0.3, 0.9)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_int("epochs", 100, 400, step=50)

    # Use trial number as seed offset for reproducibility
    seed = base_seed + trial.number

    print(f"\nTrial {trial.number}:")
    print(f"  lr={learning_rate:.6f}, momentum={momentum:.3f}, wd={weight_decay:.6f}")
    print(f"  batch_size={batch_size}, epochs={epochs}")

    val_loss = run_training(
        workspace_root,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
    )

    if val_loss is None:
        # Return a large value to indicate failure
        return float("inf")

    print(f"  val_loss={val_loss:.6f}")
    return val_loss


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for Judge")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--study-name", type=str, default="judge_tuning", help="Study name")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    args = parser.parse_args()

    workspace_root = find_workspace_root()
    print(f"Workspace root: {workspace_root}")

    # Check that training data exists
    training_data = workspace_root / "pixelflow-pipeline/data/judge_training.jsonl"
    if not training_data.exists():
        print(f"ERROR: Training data not found: {training_data}")
        print("Run: cargo run -p pixelflow-pipeline --bin collect_judge_data --release")
        sys.exit(1)

    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
    )

    # Enqueue known good params as starting point (if no trials yet)
    if len(study.trials) == 0:
        print("Enqueuing known good hyperparameters as first trial...")
        study.enqueue_trial({
            "learning_rate": 0.0005,
            "momentum": 0.42,
            "weight_decay": 5e-5,
            "batch_size": 16,
            "epochs": 300,
        })

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, workspace_root, args.seed),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_trial.value:.6f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    # Save best params
    best_params_path = workspace_root / "pixelflow-pipeline/data/best_judge_params.json"
    best_params_path.write_text(json.dumps(study.best_trial.params, indent=2))
    print(f"\nSaved best params to: {best_params_path}")

    # Train final model with best params
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL WITH BEST PARAMS")
    print("=" * 60)

    best = study.best_trial.params
    final_loss = run_training(
        workspace_root,
        learning_rate=best["learning_rate"],
        momentum=best["momentum"],
        weight_decay=best["weight_decay"],
        batch_size=best["batch_size"],
        epochs=best["epochs"],
        seed=args.seed,
    )

    # Copy to final location
    trial_model = workspace_root / f"pixelflow-pipeline/data/judge_trial_{args.seed}.bin"
    final_model = workspace_root / "pixelflow-pipeline/data/judge.bin"
    if trial_model.exists():
        import shutil
        shutil.copy(trial_model, final_model)
        print(f"\nFinal model saved to: {final_model}")
        print(f"Final validation loss: {final_loss}")

    # Cleanup trial files
    for trial_file in (workspace_root / "pixelflow-pipeline/data").glob("judge_trial_*.bin"):
        trial_file.unlink()
    for trial_file in (workspace_root / "pixelflow-pipeline/data").glob("judge_trial_*.meta.json"):
        trial_file.unlink()


if __name__ == "__main__":
    main()
