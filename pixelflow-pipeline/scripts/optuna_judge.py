#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["optuna"]
# ///
"""
Optuna hyperparameter tuning for the Judge (Value Head).

Tunes: learning_rate, momentum, weight_decay, batch_size, epochs.

The judge now uses Gaussian NLL loss (trains both mean accuracy and variance
calibration). This requires lower learning rates than MSE — the variance head
creates a 1/exp(log_var) term that amplifies gradients when the model is
confident but wrong.

Usage:
    uv run pixelflow-pipeline/scripts/optuna_judge.py --n-trials 20
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import optuna


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
    output_bin = f"pixelflow-pipeline/data/judge_trial_{seed}.bin"
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
        "--print-every", str(max(1, epochs // 5)),
        "--patience", "30",
        "--output", output_bin,
    ]

    try:
        # Stream stderr (cargo warnings) to /dev/null, capture stdout for parsing
        result = subprocess.run(
            cmd,
            cwd=workspace_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        print("    TIMEOUT", flush=True)
        return None

    if result.returncode != 0:
        print(f"    FAILED (exit {result.returncode})", flush=True)
        return None

    # Print key lines from output
    for line in result.stdout.splitlines():
        if any(k in line for k in ["Epoch", "Best validation", "Correlation"]):
            print(f"    {line.strip()}", flush=True)

    # Detect NaN divergence early
    if "NaN" in result.stdout or "nan" in result.stdout:
        print("    DIVERGED (NaN in output)", flush=True)
        return None

    # Parse best validation loss
    match = re.search(r"Best validation loss: ([\d.]+)", result.stdout)
    if match:
        val = float(match.group(1))
        # Divergence guard — loss should be reasonable
        if val > 1e6:
            print(f"    DIVERGED (val_loss={val})", flush=True)
            return None
        return val

    # Fallback: metadata file
    meta_path = workspace_root / f"pixelflow-pipeline/data/judge_trial_{seed}.meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return meta.get("best_val_loss")

    print("    Could not parse val_loss", flush=True)
    return None


def objective(trial: optuna.Trial, workspace_root: Path, base_seed: int) -> float:
    """Optuna objective function.

    Gaussian NLL needs lower learning rates than MSE because the variance
    head creates 1/exp(log_var) scaling in gradients. Overconfident wrong
    predictions cause gradient explosion without low lr + clipping.
    """
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 5e-4, log=True)
    momentum = trial.suggest_float("momentum", 0.3, 0.85)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    epochs = trial.suggest_int("epochs", 100, 400, step=25)

    seed = base_seed + trial.number

    print(
        f"\n[Trial {trial.number}] lr={learning_rate:.6f} mom={momentum:.2f} "
        f"wd={weight_decay:.6f} bs={batch_size} ep={epochs}",
        flush=True,
    )

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
        return float("inf")

    print(f"    => val_loss={val_loss:.6f}", flush=True)
    return val_loss


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for Judge")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--study-name", type=str, default="judge_gnll_v1")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=None)
    args = parser.parse_args()

    workspace_root = find_workspace_root()
    print(f"Workspace: {workspace_root}", flush=True)

    training_data = workspace_root / "pixelflow-pipeline/data/judge_training.jsonl"
    if not training_data.exists():
        print(f"ERROR: No training data at {training_data}", file=sys.stderr)
        sys.exit(1)

    line_count = sum(1 for _ in training_data.open())
    print(f"Training data: {line_count} samples", flush=True)

    # Pre-build the binary so trial 0 doesn't pay compile cost
    print("Pre-building train_judge...", flush=True)
    subprocess.run(
        ["cargo", "build", "-p", "pixelflow-pipeline", "--bin", "train_judge",
         "--release", "--features", "training"],
        cwd=workspace_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("Build complete.", flush=True)

    # Suppress optuna's internal logging noise
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
    )

    if len(study.trials) == 0:
        # Seed with conservative defaults for Gaussian NLL.
        # NLL needs lower lr than MSE — the 1/exp(log_var) term amplifies gradients.
        study.enqueue_trial({
            "learning_rate": 1e-4,
            "momentum": 0.7,
            "weight_decay": 1e-4,
            "batch_size": 128,
            "epochs": 200,
        })
        study.enqueue_trial({
            "learning_rate": 5e-5,
            "momentum": 0.5,
            "weight_decay": 5e-5,
            "batch_size": 64,
            "epochs": 300,
        })
        study.enqueue_trial({
            "learning_rate": 2e-4,
            "momentum": 0.6,
            "weight_decay": 1e-4,
            "batch_size": 128,
            "epochs": 250,
        })

    study.optimize(
        lambda trial: objective(trial, workspace_root, args.seed),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=False,
    )

    # Results
    print("\n" + "=" * 60, flush=True)
    print("OPTIMIZATION COMPLETE", flush=True)
    print("=" * 60, flush=True)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best val_loss: {study.best_trial.value:.6f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    best_params_path = workspace_root / "pixelflow-pipeline/data/best_judge_params.json"
    best_params_path.write_text(json.dumps(study.best_trial.params, indent=2))
    print(f"\nSaved best params to: {best_params_path}")

    # Final training with best params
    print("\n" + "=" * 60, flush=True)
    print("TRAINING FINAL MODEL WITH BEST PARAMS", flush=True)
    print("=" * 60, flush=True)

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

    trial_model = workspace_root / f"pixelflow-pipeline/data/judge_trial_{args.seed}.bin"
    final_model = workspace_root / "pixelflow-pipeline/data/judge.bin"
    if trial_model.exists():
        import shutil
        shutil.copy(trial_model, final_model)
        print(f"\nFinal model: {final_model}")
        print(f"Final val_loss: {final_loss}")

    # Cleanup
    for f in (workspace_root / "pixelflow-pipeline/data").glob("judge_trial_*.bin"):
        f.unlink()
    for f in (workspace_root / "pixelflow-pipeline/data").glob("judge_trial_*.meta.json"):
        f.unlink()


if __name__ == "__main__":
    main()
