#!/usr/bin/env python3
"""Optuna sweep over the bilinear rules x nodes filter's training hyperparameters.

Every learned result in this tree so far was trained at one hand-picked
configuration, so every null it reported was an untuned null.  This script
answers the narrow question that follows: does the filter's registered
null (docs/plans/2026-09-08-rules-filter-bilinear-registration.md) survive
a hyperparameter search on the *intrinsic* metric?

The extrinsic protocol is not swept and is not touched here -- it is opened
once, at the end, at whichever configuration this study picks.

usage:
    # terminal 1: load the 640k samples once and serve trials
    rules_filter serve --samples samples.jsonl --socket /tmp/rf.sock

    # terminal 2
    optuna_rules_filter.py --socket /tmp/rf.sock --study sqlite:///rf.db \\
        --trials 200 --timeout 7200 --train-cap 100000

`optuna` is a dependency of this script alone -- install it into a venv,
never into the cargo workspace.

Objective (registered before the study ran, see the results doc): the mean
over the three family-held-out folds of PR-AUC *lift* on TIGHT labels,
`PR-AUC / positive_rate`.  Plain mean PR-AUC would be the same statistic
with the folds weighted by their base rates, which differ 3.5x (glyph
0.406, scene 0.166, shader 0.115) -- maximising it is mostly maximising
glyph.  Lift is the same metric made comparable across folds.  Both are
recorded on every trial; only lift is optimised.
"""

import argparse
import json
import socket
import sys
import time

import optuna

FOLDS = ["glyph", "shader", "scene"]

# The configuration the registered run used -- `TrainArgs::default` in
# pixelflow-pipeline/src/bin/rules_filter.rs.  Enqueued as trial 0 so the
# untuned result is a point in the study rather than a number quoted from
# another run.
REGISTERED = {
    "epochs": 3,
    "lr": 0.01,
    "lr_decay": 0.7,
    "l2": 1e-4,
    "max_grad_norm": 1.0,
    "label": "tight",
    "pos_weight_power": 1.0,
    "batch_size": 1,
    "neg_keep": 1.0,
    "init_scale": 1.0,
    "relu_warm_bias": 0.5,
}


def run_trial(sock_path, args, folds=FOLDS, timeout_s=1800):
    """One trial: a JSON line to the server, the manifest back."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout_s)
    s.connect(sock_path)
    payload = json.dumps({"folds": folds, "args": args}) + "\n"
    s.sendall(payload.encode())
    s.shutdown(socket.SHUT_WR)
    chunks = []
    while True:
        b = s.recv(1 << 20)
        if not b:
            break
        chunks.append(b)
    s.close()
    raw = b"".join(chunks).decode()
    if not raw.strip():
        raise RuntimeError("server closed without replying")
    reply = json.loads(raw)
    if "error" in reply:
        raise RuntimeError(reply["error"])
    return reply


def metrics(manifest):
    """Per-fold intrinsic numbers, and the two aggregates."""
    per_fold = {}
    for fold in FOLDS:
        model = manifest["models"][fold]
        h = model["holdout"]
        if h is None:
            raise RuntimeError(f"fold {fold}: no held-out intrinsic")
        pr = h["bilinear_pr_auc_tight"]
        base = h["positive_rate_tight"]
        if pr is None or not base:
            raise RuntimeError(f"fold {fold}: PR-AUC is undefined")
        per_fold[fold] = {
            "pr_auc_tight": pr,
            "pr_auc_tight_lift": pr / base,
            "auc_tight": h["bilinear_auc_tight"],
            "pr_auc_strict": h["bilinear_pr_auc_strict"],
            "auc_strict": h["bilinear_auc_strict"],
            "per_rule_pr_auc_tight": h["per_rule_pr_auc_tight"],
            "positive_rate_tight": base,
        }
    lifts = [per_fold[f]["pr_auc_tight_lift"] for f in FOLDS]
    prs = [per_fold[f]["pr_auc_tight"] for f in FOLDS]
    return per_fold, sum(lifts) / len(lifts), sum(prs) / len(prs)


def suggest(trial):
    return {
        "epochs": trial.suggest_int("epochs", 1, 10),
        "lr": trial.suggest_float("lr", 1e-4, 0.5, log=True),
        "lr_decay": trial.suggest_float("lr_decay", 0.3, 1.0),
        "l2": trial.suggest_float("l2", 1e-8, 1e-2, log=True),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.05, 20.0, log=True),
        "label": trial.suggest_categorical("label", ["tight", "strict"]),
        "pos_weight_power": trial.suggest_float("pos_weight_power", 0.0, 1.5),
        "batch_size": trial.suggest_categorical("batch_size", [1, 4, 16, 64, 256]),
        "neg_keep": trial.suggest_float("neg_keep", 0.05, 1.0),
        "init_scale": trial.suggest_float("init_scale", 0.02, 3.0, log=True),
        "relu_warm_bias": trial.suggest_float("relu_warm_bias", 0.0, 2.0),
    }


def record(trial, per_fold, mean_lift, mean_pr, wall):
    trial.set_user_attr("mean_pr_auc_tight", mean_pr)
    trial.set_user_attr("wall_s", wall)
    for fold, m in per_fold.items():
        for k, v in m.items():
            trial.set_user_attr(f"{fold}.{k}", v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True)
    ap.add_argument("--study", required=True, help="sqlite:///path storage URL")
    ap.add_argument("--name", default="rules-filter-bilinear")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--timeout", type=float, default=7200.0, help="seconds")
    ap.add_argument(
        "--train-cap",
        type=int,
        default=100000,
        help="stride cap on training samples per fold (0 = all). The sweep "
        "runs at reduced fidelity; --refit re-runs the top trials uncapped.",
    )
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument(
        "--refit",
        type=int,
        default=0,
        help="after the study, re-run the top N trials at full fidelity "
        "(train_cap 0) and print the ranking that produces",
    )
    ap.add_argument("--out", default=None, help="write the summary JSON here")
    a = ap.parse_args()

    fixed = {"seed": a.seed, "train_cap": a.train_cap}

    def objective(trial):
        args = suggest(trial) | fixed
        t0 = time.time()
        manifest = run_trial(a.socket, args)
        per_fold, mean_lift, mean_pr = metrics(manifest)
        record(trial, per_fold, mean_lift, mean_pr, time.time() - t0)
        return mean_lift

    study = optuna.create_study(
        study_name=a.name,
        storage=a.study,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=a.seed),
    )
    if not study.trials:
        study.enqueue_trial(REGISTERED, user_attrs={"registered": True})
    study.optimize(objective, n_trials=a.trials, timeout=a.timeout)

    complete = [t for t in study.trials if t.value is not None]
    complete.sort(key=lambda t: t.value, reverse=True)
    trial0 = study.trials[0]

    summary = {
        "n_trials": len(study.trials),
        "n_complete": len(complete),
        "train_cap": a.train_cap,
        "seed": a.seed,
        "objective": "mean over folds of PR-AUC(tight) / positive_rate(tight)",
        "trial_0": {
            "params": dict(trial0.params) | fixed,
            "value": trial0.value,
            "attrs": dict(trial0.user_attrs),
        },
        "best": {
            "number": study.best_trial.number,
            "params": dict(study.best_trial.params) | fixed,
            "value": study.best_trial.value,
            "attrs": dict(study.best_trial.user_attrs),
        },
        "top10": [
            {
                "number": t.number,
                "value": t.value,
                "params": dict(t.params),
                "mean_pr_auc_tight": t.user_attrs.get("mean_pr_auc_tight"),
                "wall_s": t.user_attrs.get("wall_s"),
            }
            for t in complete[:10]
        ],
    }
    try:
        summary["importances"] = optuna.importance.get_param_importances(study)
    except Exception as e:  # noqa: BLE001 -- reported, never swallowed
        summary["importances_error"] = repr(e)

    if a.refit:
        refits = []
        for t in complete[: a.refit]:
            args = dict(t.params) | {"seed": a.seed, "train_cap": 0}
            t0 = time.time()
            manifest = run_trial(a.socket, args)
            per_fold, mean_lift, mean_pr = metrics(manifest)
            refits.append(
                {
                    "number": t.number,
                    "capped_value": t.value,
                    "full_value": mean_lift,
                    "full_mean_pr_auc_tight": mean_pr,
                    "per_fold": per_fold,
                    "params": args,
                    "wall_s": time.time() - t0,
                }
            )
            print(
                f"refit trial {t.number}: capped {t.value:.4f} -> full {mean_lift:.4f} "
                f"(mean PR-AUC {mean_pr:.4f}, {time.time() - t0:.0f}s)",
                flush=True,
            )
        # Trial 0's own full-fidelity number is the comparison the whole
        # study exists to make, so it is refitted whether or not it ranked.
        if all(r["number"] != trial0.number for r in refits):
            args = dict(trial0.params) | {"seed": a.seed, "train_cap": 0}
            manifest = run_trial(a.socket, args)
            per_fold, mean_lift, mean_pr = metrics(manifest)
            refits.append(
                {
                    "number": trial0.number,
                    "capped_value": trial0.value,
                    "full_value": mean_lift,
                    "full_mean_pr_auc_tight": mean_pr,
                    "per_fold": per_fold,
                    "params": args,
                }
            )
        refits.sort(key=lambda r: r["full_value"], reverse=True)
        summary["refits"] = refits
        summary["winner"] = refits[0]

    text = json.dumps(summary, indent=2, sort_keys=True)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    sys.exit(main())
