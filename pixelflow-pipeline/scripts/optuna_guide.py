#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.1",
#     "torch-geometric>=2.4",
#     "optuna>=3.5",
# ]
# ///
"""Optuna hyperparameter search for the Guide online training loop.

Imports graph_teacher.py as a module and runs the online training loop
with Optuna-suggested hyperparameters. HyperbandPruner prunes bad trials
after each round.

Usage:
    uv run optuna_guide.py --n-trials 50 --max-rounds 10
    uv run optuna_guide.py --n-trials 100 --max-rounds 20 --study-name guide_v2
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import optuna
import torch
from torch.optim import AdamW


# ---------------------------------------------------------------------------
# Import graph_teacher.py as a module
# ---------------------------------------------------------------------------
def _import_graph_teacher():
    gt_path = Path(__file__).resolve().parent / "graph_teacher.py"
    if not gt_path.exists():
        raise FileNotFoundError(f"graph_teacher.py not found at {gt_path}")
    spec = importlib.util.spec_from_file_location("graph_teacher", str(gt_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec for {gt_path}")
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so dataclass decorator can find the module
    sys.modules["graph_teacher"] = mod
    spec.loader.exec_module(mod)
    return mod


gt = _import_graph_teacher()


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Run online training with trial-suggested hyperparams.

    Reports val_actor_loss after each round for Hyperband pruning.
    Returns best val_actor_loss across all rounds.
    """
    # --- Sample hyperparameters ---
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    ema_decay = 1.0 - trial.suggest_float("one_minus_ema", 1e-4, 1e-1, log=True)
    hidden = trial.suggest_categorical("hidden", [64, 128, 256])
    heads = trial.suggest_categorical("heads", [2, 4, 8])
    layers = trial.suggest_categorical("layers", [2, 3, 4, 6])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    epochs_per_round = trial.suggest_categorical("epochs_per_round", [2, 3, 5])
    collect_perturbations = trial.suggest_categorical("collect_perturbations", [3, 5, 8])
    collect_sigma = trial.suggest_float("collect_sigma", 0.05, 0.3)
    collect_threshold = trial.suggest_float("collect_threshold", 0.1, 0.5)

    # --- Setup ---
    trial_dir = Path(args.output_dir) / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    replay_path = trial_dir / "replay_buffer.jsonl"
    student_path = trial_dir / "guide_student.bin"

    device = gt._select_device(args.device)
    seed = args.seed + trial.number * 1000
    torch.manual_seed(seed)

    # Initialize models
    num_rules = 64
    critic = gt.EGraphNet(
        hidden=hidden, heads=heads, layers=layers, dropout=dropout,
    ).to(device)
    critic_ema = copy.deepcopy(critic)
    policy_head = gt.PolicyHead(
        graph_dim=hidden, num_rules=num_rules, rule_dim=64,
    ).to(device)
    policy_head_ema = copy.deepcopy(policy_head)
    actor = gt.NNUEStudent().to(device)

    critic_opt = AdamW(
        list(critic.parameters()) + list(policy_head.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    actor_opt = AdamW(actor.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")

    for round_idx in range(args.max_rounds):
        t0 = time.monotonic()

        # === COLLECT ===
        if round_idx > 0:
            student_path.write_bytes(actor.export_bytes())
            guide_w = student_path
        else:
            guide_w = None

        try:
            gt._run_collect(
                guide_weights=guide_w,
                judge_weights=args.judge_weights,
                output=replay_path,
                count=args.collect_count,
                perturbations=collect_perturbations,
                sigma=collect_sigma,
                threshold=collect_threshold,
                max_epochs=args.collect_max_epochs,
                seed=seed + round_idx,
            )
        except RuntimeError as e:
            print(f"  Trial {trial.number} round {round_idx}: collect failed: {e}",
                  file=sys.stderr)
            raise optuna.TrialPruned()

        # === LOAD ===
        samples, graphs = gt.load_policy_dataset(replay_path)
        if not samples:
            print(f"  Trial {trial.number}: no samples in round {round_idx}",
                  file=sys.stderr)
            raise optuna.TrialPruned()

        data_num_rules = max(s.rule_idx for s in samples) + 1
        if data_num_rules != num_rules:
            num_rules = data_num_rules
            policy_head = gt.PolicyHead(
                graph_dim=hidden, num_rules=num_rules, rule_dim=64,
            ).to(device)
            policy_head_ema = copy.deepcopy(policy_head)
            critic_opt = AdamW(
                list(critic.parameters()) + list(policy_head.parameters()),
                lr=lr, weight_decay=weight_decay,
            )

        n = len(samples)
        n_train = int(0.8 * n)
        gen = torch.Generator().manual_seed(seed + round_idx)
        perm = torch.randperm(n, generator=gen).tolist()
        train_samples = [samples[i] for i in perm[:n_train]]
        val_samples = [samples[i] for i in perm[n_train:]]

        # === TRAIN ===
        round_best_val = float("inf")
        for epoch in range(1, epochs_per_round + 1):
            critic_loss, actor_loss = gt.train_opd_epoch(
                critic, critic_ema, policy_head, policy_head_ema, actor,
                train_samples, graphs, critic_opt, actor_opt, device,
                batch_size=batch_size, ema_decay=ema_decay,
            )
            val_actor = gt._validate_actor(
                actor, critic_ema, policy_head_ema,
                val_samples, graphs, device, batch_size,
            )
            round_best_val = min(round_best_val, val_actor)

        dt = time.monotonic() - t0
        if round_best_val < best_val_loss:
            best_val_loss = round_best_val

        # Report to Optuna for pruning
        trial.report(best_val_loss, step=round_idx)

        print(
            f"  Trial {trial.number} round {round_idx + 1}/{args.max_rounds}: "
            f"val={round_best_val:.6f} best={best_val_loss:.6f} ({dt:.1f}s)",
            file=sys.stderr,
        )

        if trial.should_prune():
            # Save what we have before pruning
            _save_trial_result(trial_dir, trial, best_val_loss, round_idx)
            raise optuna.TrialPruned()

    # === FINAL CHECKPOINT ===
    student_path.write_bytes(actor.export_bytes())
    _save_trial_result(trial_dir, trial, best_val_loss, args.max_rounds - 1)

    return best_val_loss


def _save_trial_result(
    trial_dir: Path, trial: optuna.Trial, best_val: float, last_round: int,
) -> None:
    """Save trial metadata for later analysis."""
    result = {
        "trial_number": trial.number,
        "best_val_actor_loss": best_val,
        "last_round": last_round,
        "params": trial.params,
    }
    with open(trial_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Optuna Guide hyperparameter search")
    p.add_argument("--judge-weights", default=None,
                   help="Path to judge.bin (default: auto-detect)")
    p.add_argument("--n-trials", type=int, default=50,
                   help="Number of Optuna trials")
    p.add_argument("--max-rounds", type=int, default=10,
                   help="Max online-train rounds per trial")
    p.add_argument("--collect-count", type=int, default=50,
                   help="Expressions per collection (lower=faster trials)")
    p.add_argument("--collect-max-epochs", type=int, default=50)
    p.add_argument("--output-dir", default="/tmp/guide_optuna",
                   help="Root output directory")
    p.add_argument("--study-name", default="guide_hpo",
                   help="Optuna study name (for DB)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    # Auto-detect judge weights
    if args.judge_weights is None:
        default_judge = Path(__file__).resolve().parent.parent / "data" / "judge.bin"
        if default_judge.exists():
            args.judge_weights = str(default_judge)
        else:
            raise FileNotFoundError(
                "judge.bin not found. Pass --judge-weights explicitly."
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "study.db"

    # Build Rust binary once upfront
    print("Building collect_guide_data (release)...", file=sys.stderr)
    gt._cargo_build_collect()
    print("Build complete.", file=sys.stderr)

    # Create or resume study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{db_path}",
        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=args.max_rounds,
            reduction_factor=3,
        ),
        load_if_exists=True,
    )

    print(
        f"\nOptuna study '{args.study_name}' ({args.n_trials} trials, "
        f"{args.max_rounds} rounds/trial)\n"
        f"DB: {db_path}\n"
        f"Output: {output_dir}\n",
        file=sys.stderr,
    )

    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # === RESULTS ===
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Best trial: #{study.best_trial.number}", file=sys.stderr)
    print(f"Best val_actor_loss: {study.best_value:.6f}", file=sys.stderr)
    print(f"Best params:", file=sys.stderr)
    for k, v in study.best_params.items():
        print(f"  {k}: {v}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    # Save best params to file
    best_path = output_dir / "best_params.json"
    with open(best_path, "w") as f:
        json.dump(
            {
                "best_trial": study.best_trial.number,
                "best_value": study.best_value,
                "best_params": study.best_params,
            },
            f,
            indent=2,
        )
    print(f"Best params saved to {best_path}", file=sys.stderr)

    # === FINAL TRAINING with best params ===
    print(f"\n{'=' * 60}", file=sys.stderr)
    print("Running final training with best hyperparameters...", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    best = study.best_params
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Build argparse namespace for cmd_online_train
    # Scale LR down for longer final run: Optuna tuned for 10 rounds × 50 exprs,
    # final run does 20 rounds × 100 exprs (~4× gradient steps).
    # Linear scaling rule: lr_final = lr_search * sqrt(search_steps / final_steps)
    final_lr = best["lr"] * (args.max_rounds * args.collect_count
                              / (20 * 100)) ** 0.5
    print(f"LR scaling: {best['lr']:.6f} (search) → {final_lr:.6f} (final)",
          file=sys.stderr)

    final_args = argparse.Namespace(
        judge_weights=args.judge_weights,
        rounds=20,  # Full 20 rounds with best params
        collect_count=100,  # Full collection
        collect_perturbations=best["collect_perturbations"],
        collect_sigma=best["collect_sigma"],
        collect_threshold=best["collect_threshold"],
        collect_max_epochs=args.collect_max_epochs,
        train_epochs_per_round=best["epochs_per_round"],
        batch_size=best["batch_size"],
        lr=final_lr,
        weight_decay=best["weight_decay"],
        ema_decay=1.0 - best["one_minus_ema"],
        hidden=best["hidden"],
        heads=best["heads"],
        layers=best["layers"],
        dropout=best["dropout"],
        output_dir=str(final_dir),
        warm_start=None,
        seed=args.seed,
        device=args.device,
    )
    gt.cmd_online_train(final_args)

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"DONE. Final model at: {final_dir}/guide_student.bin", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)


if __name__ == "__main__":
    main()
