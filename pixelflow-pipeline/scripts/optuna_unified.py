#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["optuna"]
# ///
"""
Optuna hyperparameter tuning for the unified training loop.

Two modes:
  --offline   Joint GPU+CPU sweep on existing trajectory data (no JIT/self-play).
              Each trial samples BOTH critic hyperparams (GPU) AND SGD hyperparams (CPU).
  (default)   Full online trials (self-play + JIT bench + critic per round)

Usage:
    # Offline sweep (fast — minutes per trial)
    uv run pixelflow-pipeline/scripts/optuna_unified.py --offline \
        --trajectory-dir pixelflow-pipeline/data/unified \
        --n-trials 100 --max-rounds 50

    # Online sweep (slow — self-play + JIT per trial)
    uv run pixelflow-pipeline/scripts/optuna_unified.py --n-trials 50 --max-rounds 30

    # Smoke test
    uv run pixelflow-pipeline/scripts/optuna_unified.py --offline --n-trials 2 --max-rounds 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
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


def _score_metrics_online(metrics: dict, mae_weight: float) -> float | None:
    """Composite score for online mode: speedup + judge quality."""
    speedup_raw = metrics.get("speedup_median")
    mae_raw = metrics.get("judge_mae")
    if speedup_raw is None or mae_raw is None:
        return None
    speedup = float(speedup_raw)
    mae = float(mae_raw)
    if not math.isfinite(speedup) or not math.isfinite(mae) or speedup <= 0:
        return None
    return -speedup + mae_weight * mae


def _score_metrics_offline(metrics: dict) -> float | None:
    """Composite score for offline mode: value_loss + policy_loss."""
    val = metrics.get("avg_value_loss")
    pol = metrics.get("avg_policy_loss")
    if val is None or pol is None:
        return None
    val, pol = float(val), float(pol)
    if not math.isfinite(val) or not math.isfinite(pol):
        return None
    # Minimize value loss + policy loss (both positive)
    return val + pol


def _read_metrics_lines(metrics_path: Path) -> list[dict]:
    """Read all valid JSONL lines from metrics file."""
    if not metrics_path.exists():
        return []
    lines = []
    for raw in metrics_path.read_text().strip().splitlines():
        try:
            lines.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return lines


# ============================================================================
# Offline Objective
# ============================================================================

def objective_offline(
    trial: optuna.Trial,
    workspace_root: Path,
    args: argparse.Namespace,
) -> float:
    """Run train_unified --offline with jointly sampled GPU (critic) + CPU (SGD) hyperparams."""
    # ── GPU: Critic hyperparams ──
    critic_epochs = trial.suggest_int("critic_epochs", 10, 100, step=10)
    critic_lr = trial.suggest_float("critic_lr", 1e-5, 1e-3, log=True)
    critic_dropout = trial.suggest_float("critic_dropout", 0.0, 0.4)

    # ── CPU: SGD hyperparams ──
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    momentum = trial.suggest_float("momentum", 0.0, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    grad_clip = trial.suggest_float("grad_clip", 0.05, 10.0, log=True)
    entropy_coeff = trial.suggest_float("entropy_coeff", 0.001, 0.2, log=True)
    value_coeff = trial.suggest_float("value_coeff", 0.05, 5.0, log=True)
    miss_penalty = trial.suggest_float("miss_penalty", 0.0, 1.0)

    # ── Replay sampling ──
    mini_batch_size = trial.suggest_categorical("mini_batch_size", [256, 512, 1024, 2048])
    updates_per_round = trial.suggest_int("updates_per_round", 1, 20)

    seed = args.seed + trial.number * 1000
    trial_dir = f"offline_trial_{trial.number}"
    output_dir = workspace_root / "pixelflow-pipeline" / "data" / trial_dir
    model_path = str(workspace_root / "pixelflow-pipeline" / "data" / "judge.bin")
    metrics_path = output_dir / "metrics.jsonl"

    print(
        f"\n[Trial {trial.number}] "
        f"CRITIC: ep={critic_epochs} lr={critic_lr:.2e} do={critic_dropout:.2f} | "
        f"SGD: lr={lr:.6f} mom={momentum:.2f} wd={weight_decay:.2e} "
        f"gc={grad_clip:.2f} ent={entropy_coeff:.4f} val={value_coeff:.2f} "
        f"miss={miss_penalty:.2f} bs={mini_batch_size} upd={updates_per_round}",
        flush=True,
    )

    traj_dir = str(args.trajectory_dir or (
        workspace_root / "pixelflow-pipeline" / "data" / "unified"
    ))

    cmd = [
        "cargo", "run", "-p", "pixelflow-pipeline",
        "--bin", "train_unified", "--release",
        "--features", "training",
        "--",
        "--offline",
        "--trajectory-dir", traj_dir,
        "--rounds", str(args.max_rounds),
        "--replay-capacity", "200000",
        "--seed", str(seed),
        "--model", model_path,
        "--output-dir", str(output_dir),
        "--critic-checkpoint", str(output_dir / "critic.pt"),
        "--critic-epochs", str(critic_epochs),
        "--critic-lr", str(critic_lr),
        "--critic-dropout", str(critic_dropout),
        "--lr", str(lr),
        "--momentum", str(momentum),
        "--weight-decay", str(weight_decay),
        "--grad-clip", str(grad_clip),
        "--entropy-coeff", str(entropy_coeff),
        "--value-coeff", str(value_coeff),
        "--mini-batch-size", str(mini_batch_size),
        "--updates-per-round", str(updates_per_round),
        "--miss-penalty", str(miss_penalty),
    ]

    stderr_path = output_dir / "stderr.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    stderr_file = open(stderr_path, "w")
    proc = subprocess.Popen(
        cmd,
        cwd=workspace_root,
        stdout=subprocess.DEVNULL,
        stderr=stderr_file,
    )

    best_score = float("inf")
    last_seen_rounds = 0
    # Offline is fast: ~1s per round + critic overhead
    timeout_s = 5 * args.max_rounds + 300
    t0 = time.monotonic()

    try:
        while proc.poll() is None:
            time.sleep(1)

            if time.monotonic() - t0 > timeout_s:
                print(f"    TIMEOUT after {timeout_s}s", flush=True)
                proc.kill()
                proc.wait()
                stderr_file.close()
                _cleanup_trial(output_dir)
                return float("inf")

            all_metrics = _read_metrics_lines(metrics_path)
            if len(all_metrics) <= last_seen_rounds:
                continue

            for round_idx in range(last_seen_rounds, len(all_metrics)):
                m = all_metrics[round_idx]
                score = _score_metrics_offline(m)

                if score is None:
                    continue

                if score < best_score:
                    best_score = score

                trial.report(best_score, step=round_idx)

                if round_idx % 10 == 0 or round_idx == len(all_metrics) - 1:
                    print(
                        f"    round {round_idx}: val_loss={m.get('avg_value_loss', '?'):.4f} "
                        f"pol_loss={m.get('avg_policy_loss', '?'):.4f} "
                        f"grad={m.get('grad_norm', '?'):.4f} "
                        f"score={score:.4f}",
                        flush=True,
                    )

                if trial.should_prune():
                    print(f"    PRUNED at round {round_idx}", flush=True)
                    proc.kill()
                    proc.wait()
                    stderr_file.close()
                    _cleanup_trial(output_dir)
                    raise optuna.TrialPruned()

            last_seen_rounds = len(all_metrics)

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"    ERROR: {e}", flush=True)
        proc.kill()
        proc.wait()
        stderr_file.close()
        _cleanup_trial(output_dir)
        return float("inf")

    stderr_file.close()

    if proc.returncode != 0:
        if stderr_path.exists():
            for line in stderr_path.read_text().strip().splitlines()[-5:]:
                print(f"    stderr: {line}", flush=True)
        print(f"    FAILED (exit {proc.returncode})", flush=True)
        _cleanup_trial(output_dir)
        return float("inf")

    # Final read
    all_metrics = _read_metrics_lines(metrics_path)
    for m in all_metrics[last_seen_rounds:]:
        score = _score_metrics_offline(m)
        if score is not None and score < best_score:
            best_score = score

    if best_score == float("inf"):
        print("    NO VALID ROUNDS", flush=True)
        _cleanup_trial(output_dir)
        return float("inf")

    final_m = all_metrics[-1] if all_metrics else {}
    print(
        f"    => FINAL val_loss={final_m.get('avg_value_loss', '?'):.4f} "
        f"pol_loss={final_m.get('avg_policy_loss', '?'):.4f} "
        f"score={best_score:.4f}",
        flush=True,
    )
    _cleanup_trial(output_dir)
    return best_score


# ============================================================================
# Online Objective (original)
# ============================================================================

def objective_online(
    trial: optuna.Trial,
    workspace_root: Path,
    args: argparse.Namespace,
) -> float:
    """Run train_unified, stream metrics, prune bad trials early."""
    # ── Policy optimizer ──
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    momentum = trial.suggest_float("momentum", 0.0, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    grad_clip = trial.suggest_float("grad_clip", 0.1, 10.0, log=True)
    entropy_coeff = trial.suggest_float("entropy_coeff", 0.001, 0.1, log=True)
    value_coeff = trial.suggest_float("value_coeff", 0.1, 5.0, log=True)
    miss_penalty = trial.suggest_float("miss_penalty", 0.0, 1.0)

    # ── Replay buffer ──
    mini_batch_size = trial.suggest_categorical("mini_batch_size", [256, 512, 1024, 2048])
    updates_per_round = trial.suggest_int("updates_per_round", 1, 20)

    # ── Critic ──
    critic_epochs = trial.suggest_int("critic_epochs", 10, 100, step=10)
    critic_lr = trial.suggest_float("critic_lr", 1e-5, 1e-3, log=True)
    critic_dropout = trial.suggest_float("critic_dropout", 0.0, 0.4)

    # ── Corpus mix ──
    corpus_fraction = trial.suggest_float("corpus_fraction", 0.0, 1.0)

    # ── Mask ──
    threshold = trial.suggest_float("threshold", 0.1, 0.7)

    seed = args.seed + trial.number * 1000
    trial_dir = f"unified_trial_{trial.number}"
    output_dir = workspace_root / "pixelflow-pipeline" / "data" / trial_dir
    model_path = str(workspace_root / "pixelflow-pipeline" / "data" / "judge.bin")
    metrics_path = output_dir / "metrics.jsonl"

    print(
        f"\n[Trial {trial.number}] lr={lr:.6f} mom={momentum:.2f} wd={weight_decay:.2e} "
        f"gc={grad_clip:.2f} ent={entropy_coeff:.4f} val={value_coeff:.2f} "
        f"bs={mini_batch_size} upd={updates_per_round} "
        f"crit_ep={critic_epochs} crit_lr={critic_lr:.2e} crit_do={critic_dropout:.2f} "
        f"miss={miss_penalty:.2f} thresh={threshold:.2f} corpus={corpus_fraction:.2f}",
        flush=True,
    )

    cmd = [
        "cargo", "run", "-p", "pixelflow-pipeline",
        "--bin", "train_unified", "--release",
        "--features", "training",
        "--",
        "--rounds", str(args.max_rounds),
        "--trajectories-per-round", "50",
        "--max-steps", "50",
        "--corpus-fraction", str(corpus_fraction),
        "--replay-capacity", "200000",
        "--seed", str(seed),
        "--model", model_path,
        "--output-dir", str(output_dir),
        "--critic-checkpoint", str(output_dir / "critic.pt"),
        "--lr", str(lr),
        "--momentum", str(momentum),
        "--weight-decay", str(weight_decay),
        "--grad-clip", str(grad_clip),
        "--entropy-coeff", str(entropy_coeff),
        "--value-coeff", str(value_coeff),
        "--mini-batch-size", str(mini_batch_size),
        "--updates-per-round", str(updates_per_round),
        "--critic-epochs", str(critic_epochs),
        "--critic-lr", str(critic_lr),
        "--critic-dropout", str(critic_dropout),
        "--threshold", str(threshold),
        "--miss-penalty", str(miss_penalty),
    ]

    # Redirect stderr to file (PIPE blocks when 64KB buffer fills, hanging the process)
    stderr_path = output_dir / "stderr.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    stderr_file = open(stderr_path, "w")
    proc = subprocess.Popen(
        cmd,
        cwd=workspace_root,
        stdout=subprocess.DEVNULL,
        stderr=stderr_file,
    )

    best_score = float("inf")
    last_seen_rounds = 0
    # ~3 min per round (JIT bench + critic training), plus 5 min headroom
    timeout_s = 180 * args.max_rounds + 300
    t0 = time.monotonic()

    try:
        while proc.poll() is None:
            time.sleep(2)  # poll every 2s

            # Timeout guard
            if time.monotonic() - t0 > timeout_s:
                print(f"    TIMEOUT after {timeout_s}s", flush=True)
                proc.kill()
                proc.wait()
                stderr_file.close()
                _cleanup_trial(output_dir)
                return float("inf")

            # Read new metrics lines
            all_metrics = _read_metrics_lines(metrics_path)
            if len(all_metrics) <= last_seen_rounds:
                continue

            # Process new rounds
            for round_idx in range(last_seen_rounds, len(all_metrics)):
                m = all_metrics[round_idx]
                score = _score_metrics_online(m, args.mae_weight)

                if score is None:
                    print(
                        f"    round {round_idx}: INVALID "
                        f"(speedup={m.get('speedup_median')}, mae={m.get('judge_mae')})",
                        flush=True,
                    )
                    continue

                if score < best_score:
                    best_score = score

                # Report to Optuna for Hyperband pruning
                trial.report(best_score, step=round_idx)

                speedup = m.get("speedup_median", 0)
                mae = m.get("judge_mae", 0)
                print(
                    f"    round {round_idx}: speedup={speedup:.2f}x mae={mae:.3f} "
                    f"best_score={best_score:.3f}",
                    flush=True,
                )

                if trial.should_prune():
                    print(f"    PRUNED at round {round_idx}", flush=True)
                    proc.kill()
                    proc.wait()
                    stderr_file.close()
                    _cleanup_trial(output_dir)
                    raise optuna.TrialPruned()

            last_seen_rounds = len(all_metrics)

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"    ERROR: {e}", flush=True)
        proc.kill()
        proc.wait()
        stderr_file.close()
        _cleanup_trial(output_dir)
        return float("inf")

    stderr_file.close()

    # Process exited — check return code
    if proc.returncode != 0:
        if stderr_path.exists():
            stderr_lines = stderr_path.read_text().strip().splitlines()
            for line in stderr_lines[-5:]:
                print(f"    stderr: {line}", flush=True)
        print(f"    FAILED (exit {proc.returncode})", flush=True)
        _cleanup_trial(output_dir)
        return float("inf")

    # Final read of all metrics
    all_metrics = _read_metrics_lines(metrics_path)
    for m in all_metrics[last_seen_rounds:]:
        score = _score_metrics_online(m, args.mae_weight)
        if score is not None and score < best_score:
            best_score = score

    if best_score == float("inf"):
        print("    NO VALID ROUNDS", flush=True)
        _cleanup_trial(output_dir)
        return float("inf")

    final_m = all_metrics[-1] if all_metrics else {}
    print(
        f"    => FINAL speedup={final_m.get('speedup_median', '?'):.3f}x "
        f"mae={final_m.get('judge_mae', '?'):.3f} best_score={best_score:.3f}",
        flush=True,
    )
    _cleanup_trial(output_dir)
    return best_score


def _cleanup_trial(trial_path: Path) -> None:
    """Remove trial output directory to save disk."""
    if trial_path.exists():
        shutil.rmtree(trial_path, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Optuna tuning for unified self-play training loop"
    )
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--max-rounds", type=int, default=30,
                        help="Training rounds per trial (Hyperband max_resource)")
    parser.add_argument("--final-rounds", type=int, default=0,
                        help="Rounds for final training with best params (0 to skip)")
    parser.add_argument("--mae-weight", type=float, default=0.5,
                        help="Weight on judge_mae in composite score (online mode)")
    parser.add_argument("--study-name", type=str, default="unified_v2")
    parser.add_argument("--output-dir", type=str, default="/tmp/optuna_unified")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=None,
                        help="Total optimization timeout in seconds")
    parser.add_argument("--resume", action="store_true",
                        help="Resume existing study instead of clearing stale results")

    # Offline mode
    parser.add_argument("--offline", action="store_true",
                        help="Offline mode: joint GPU+CPU sweep on existing trajectory data")
    parser.add_argument("--trajectory-dir", type=str, default=None,
                        help="Directory with trajectory JSONL files (offline mode)")
    args = parser.parse_args()

    workspace_root = find_workspace_root()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "study.db"

    print(f"Workspace: {workspace_root}", flush=True)
    print(f"Study DB: {db_path}", flush=True)
    print(f"Mode: {'OFFLINE' if args.offline else 'ONLINE'}", flush=True)

    if args.offline:
        traj_dir = Path(args.trajectory_dir or (
            workspace_root / "pixelflow-pipeline" / "data" / "unified"
        ))
        traj_count = len(list(traj_dir.glob("trajectories_r*.jsonl")))
        if traj_count == 0:
            print(f"ERROR: No trajectory files in {traj_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Trajectory dir: {traj_dir} ({traj_count} files)", flush=True)
        print("Joint GPU+CPU optimization: critic hyperparams in search space", flush=True)

    # Check prerequisites
    model = workspace_root / "pixelflow-pipeline" / "data" / "judge.bin"
    if not model.exists():
        print(f"INFO: No model at {model}, trials will init fresh", flush=True)

    # Pre-build so trial 0 doesn't pay compile cost
    print("Pre-building train_unified...", flush=True)
    result = subprocess.run(
        ["cargo", "build", "-p", "pixelflow-pipeline", "--bin", "train_unified",
         "--release", "--features", "training"],
        cwd=workspace_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: Build failed:\n{result.stderr[-500:]}", file=sys.stderr)
        sys.exit(1)
    print("Build complete.", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study_name = f"{args.study_name}_offline" if args.offline else args.study_name
    storage = f"sqlite:///{db_path}"

    # Delete stale study if it exists with all-infinity results
    if db_path.exists() and not args.resume:
        try:
            old = optuna.load_study(study_name=study_name, storage=storage)
            if old.trials and all(
                t.value is None or not math.isfinite(t.value)
                for t in old.trials if t.state == optuna.trial.TrialState.COMPLETE
            ):
                print(f"Deleting stale study with all-infinity results...", flush=True)
                optuna.delete_study(study_name=study_name, storage=storage)
        except KeyError:
            pass  # study doesn't exist yet

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=3,
            max_resource=args.max_rounds,
            reduction_factor=3,
        ),
        load_if_exists=True,
    )

    if len(study.trials) == 0:
        # Seed with current defaults
        study.enqueue_trial({
            "lr": 8.17e-3, "momentum": 0.891, "weight_decay": 5.34e-6,
            "grad_clip": 8.74, "entropy_coeff": 0.098, "value_coeff": 1.175,
            "mini_batch_size": 2048, "updates_per_round": 19,
            "critic_epochs": 80, "critic_lr": 1.66e-4, "critic_dropout": 0.124,
            "miss_penalty": 0.1,
            **({}  if args.offline else {
                "threshold": 0.508, "corpus_fraction": 0.9,
            }),
        })

    if args.offline:
        objective_fn = lambda trial: objective_offline(trial, workspace_root, args)
    else:
        objective_fn = lambda trial: objective_online(trial, workspace_root, args)

    print(
        f"\nOptuna study '{study_name}' "
        f"({args.n_trials} trials, {args.max_rounds} rounds/trial, "
        f"Hyperband pruning)\n",
        flush=True,
    )

    study.optimize(
        objective_fn,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    # ── Results ──
    print("\n" + "=" * 60, flush=True)
    print("OPTIMIZATION COMPLETE", flush=True)
    print("=" * 60, flush=True)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best score: {study.best_trial.value:.6f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    best_params_path = workspace_root / "pixelflow-pipeline" / "data" / "best_unified_params.json"
    best_params_path.write_text(json.dumps(
        {"best_trial": study.best_trial.number,
         "best_value": study.best_value,
         "best_params": study.best_params,
         "mode": "offline" if args.offline else "online"},
        indent=2,
    ))
    print(f"\nSaved best params to: {best_params_path}")

    # ── Final training with best params ──
    if args.final_rounds > 0 and not args.offline:
        print("\n" + "=" * 60, flush=True)
        print(f"TRAINING FINAL MODEL ({args.final_rounds} rounds)", flush=True)
        print("=" * 60, flush=True)

        best = study.best_trial.params
        final_dir = f"unified_final"
        final_output = workspace_root / "pixelflow-pipeline" / "data" / final_dir
        final_cmd = [
            "cargo", "run", "-p", "pixelflow-pipeline",
            "--bin", "train_unified", "--release",
            "--features", "training",
            "--",
            "--rounds", str(args.final_rounds),
            "--trajectories-per-round", "50",
            "--corpus-fraction", str(best["corpus_fraction"]),
            "--max-steps", "50",
            "--replay-capacity", "200000",
            "--seed", str(args.seed),
            "--model", str(workspace_root / "pixelflow-pipeline" / "data" / "judge.bin"),
            "--output-dir", str(final_output),
            "--critic-checkpoint", str(final_output / "critic.pt"),
            "--lr", str(best["lr"]),
            "--momentum", str(best["momentum"]),
            "--weight-decay", str(best["weight_decay"]),
            "--grad-clip", str(best["grad_clip"]),
            "--entropy-coeff", str(best["entropy_coeff"]),
            "--value-coeff", str(best["value_coeff"]),
            "--mini-batch-size", str(best["mini_batch_size"]),
            "--updates-per-round", str(best["updates_per_round"]),
            "--critic-epochs", str(best["critic_epochs"]),
            "--critic-lr", str(best["critic_lr"]),
            "--critic-dropout", str(best["critic_dropout"]),
            "--threshold", str(best["threshold"]),
            "--miss-penalty", str(best["miss_penalty"]),
        ]

        print(f"Running: {' '.join(final_cmd[-30:])}", flush=True)
        result = subprocess.run(
            final_cmd, cwd=workspace_root,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )

        if result.returncode == 0:
            final_metrics = _read_metrics_lines(final_output / "metrics.jsonl")
            if final_metrics:
                last = final_metrics[-1]
                print(f"\nFinal speedup: {last.get('speedup_median', '?')}")
                print(f"Final judge_mae: {last.get('judge_mae', '?')}")
        else:
            print(f"\nFinal training failed (exit {result.returncode})", file=sys.stderr)

    # ── Cleanup ──
    data_dir = workspace_root / "pixelflow-pipeline" / "data"
    for pattern in ["unified_trial_*", "offline_trial_*"]:
        for d in data_dir.glob(pattern):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)



if __name__ == "__main__":
    main()
