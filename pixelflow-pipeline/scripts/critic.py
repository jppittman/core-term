#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.1",
#     "numpy>=1.26.0",
# ]
# ///
"""Causal Sequence Transformer Critic for temporal credit assignment.

Reads self-play trajectories (JSONL), trains a value model V_t for each
step, exports per-step advantages A_t = R_T - V_t.

Variable-length trajectories are batched with pad_sequence + padding masks
for efficient GPU matrix multiplication. Device auto-selected: CUDA > MPS > CPU.

Usage:
    uv run critic.py train --input trajectories.jsonl --output advantages.jsonl
    uv run critic.py train --input trajectories.jsonl --output advantages.jsonl --checkpoint critic.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW

# =============================================================================
# Constants — must match pixelflow-pipeline/src/training/unified.rs
# =============================================================================

ACC_DIM = 130        # EdgeAccumulator size (4*K dual sums + edge_count + node_count)
EXPR_DIM = 32        # EMBED_DIM for expression embeddings (expr_proj output)
RULE_DIM = 32        # EMBED_DIM for rule embeddings
RESOURCE_DIM = 2     # budget_remaining, epochs_remaining
STEP_DIM = ACC_DIM + EXPR_DIM + RULE_DIM + RESOURCE_DIM  # 196 floats per step


# =============================================================================
# SinusoidalPE — standard sinusoidal positional encoding
# =============================================================================

class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        pe = self._build_pe(d_model, max_len)
        # (1, max_len, d_model) — registered as buffer, not parameter
        self.register_buffer("pe", pe.unsqueeze(0))

    @staticmethod
    def _build_pe(d_model: int, max_len: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x: (batch, seq_len, d_model)."""
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # Grow PE buffer on demand — no silent truncation
            new_pe = self._build_pe(self.d_model, seq_len).unsqueeze(0).to(self.pe.device)
            self.pe = new_pe
        return x + self.pe[:, :seq_len]


# =============================================================================
# CriticTransformer — causal transformer value network
# =============================================================================

class CriticTransformer(nn.Module):
    """Causal Transformer that predicts per-step value V_t.

    Each step sees only itself and prior steps (causal mask).
    Padding positions are masked via src_key_padding_mask so batches of
    variable-length trajectories can be processed in a single forward pass.
    The value head outputs a scalar V_t per step.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(STEP_DIM, d_model)
        self.pe = SinusoidalPE(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False,  # MPS doesn't support nested tensor ops
        )
        self.value_head = nn.Linear(d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, STEP_DIM) step features (may include padding).
            src_key_padding_mask: (batch, seq_len) bool tensor where True
                marks padding positions that should be ignored by attention.

        Returns:
            (batch, seq_len, 1) predicted values V_t.
        """
        h = self.input_proj(x)
        h = self.pe(h)
        T = h.size(1)
        # Causal mask: position t can attend to positions <= t.
        # TransformerEncoder expects True = "block this position" for bool masks.
        causal = torch.triu(
            torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1
        )
        # causal mask (T, T) blocks future; src_key_padding_mask (B, T)
        # blocks padding keys. PyTorch combines them correctly — a position
        # is blocked if EITHER mask says so.
        h = self.transformer(
            h, mask=causal, src_key_padding_mask=src_key_padding_mask
        )
        return self.value_head(h)  # (batch, seq_len, 1)


# =============================================================================
# Data loading — fail fast, fail loudly
# =============================================================================

def load_trajectories(path: Path) -> list[dict]:
    """Load trajectory JSONL. Crash with a clear error if empty or malformed."""
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    trajectories: list[dict] = []
    with open(path) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Malformed JSON at {path}:{line_num}: {e}"
                ) from e

            # Validate required fields
            for field in ("trajectory_id", "steps", "final_cost_ns"):
                if field not in obj:
                    raise KeyError(
                        f"Missing required field '{field}' at {path}:{line_num}"
                    )
            if not isinstance(obj["steps"], list) or len(obj["steps"]) == 0:
                raise ValueError(
                    f"Trajectory at {path}:{line_num} has no steps "
                    f"(trajectory_id={obj.get('trajectory_id', '???')})"
                )
            for step_idx, step in enumerate(obj["steps"]):
                for sf in ("accumulator_state", "expression_embedding", "rule_embedding"):
                    if sf not in step:
                        raise KeyError(
                            f"Missing '{sf}' in step {step_idx} at "
                            f"{path}:{line_num}"
                        )
                acc_len = len(step["accumulator_state"])
                if acc_len != ACC_DIM:
                    raise ValueError(
                        f"accumulator_state has {acc_len} floats, expected "
                        f"{ACC_DIM} at {path}:{line_num} step {step_idx}"
                    )
                expr_len = len(step["expression_embedding"])
                if expr_len != EXPR_DIM:
                    raise ValueError(
                        f"expression_embedding has {expr_len} floats, expected "
                        f"{EXPR_DIM} at {path}:{line_num} step {step_idx}"
                    )
                rule_len = len(step["rule_embedding"])
                if rule_len != RULE_DIM:
                    raise ValueError(
                        f"rule_embedding has {rule_len} floats, expected "
                        f"{RULE_DIM} at {path}:{line_num} step {step_idx}"
                    )

            trajectories.append(obj)

    if not trajectories:
        raise ValueError(f"No trajectories found in {path} (file is empty)")

    print(
        f"Loaded {len(trajectories)} trajectories from {path}",
        file=sys.stderr,
    )
    return trajectories


def trajectories_to_tensors(
    trajectories: list[dict],
) -> tuple[list[torch.Tensor], list[float], list[torch.Tensor]]:
    """Convert trajectories to per-trajectory tensors.

    Returns:
        sequences: list of (seq_len_i, STEP_DIM) tensors (variable length).
        rewards: list of terminal rewards (log-ns domain).
        matched: list of (seq_len_i) boolean tensors indicating if rule applied.
    """
    sequences: list[torch.Tensor] = []
    rewards: list[float] = []
    matched_masks: list[torch.Tensor] = []

    for traj in trajectories:
        steps = traj["steps"]
        step_features = []
        step_matches = []
        for step in steps:
            # Scale features to match Rust's forward_shared scaling
            acc_state = step["accumulator_state"].copy()
            edge_count = acc_state[128]
            node_count = acc_state[129]
            
            scale = 1.0 / math.sqrt(max(1.0, node_count))
            for i in range(128):
                acc_state[i] *= scale
            
            acc_state[128] = math.log2(1.0 + edge_count)
            acc_state[129] = math.log2(1.0 + node_count)
            
            resource_features = [
                float(step["budget_remaining"]),
                float(step["epochs_remaining"]),
            ]
            features = acc_state + step["expression_embedding"] + step["rule_embedding"] + resource_features
            if len(features) != STEP_DIM:
                raise ValueError(
                    f"Step feature length {len(features)} != {STEP_DIM} "
                    f"(trajectory_id={traj['trajectory_id']})"
                )
            step_features.append(features)
            step_matches.append(step["matched"])

        seq = torch.tensor(step_features, dtype=torch.float32)
        sequences.append(seq)
        
        matches = torch.tensor(step_matches, dtype=torch.bool)
        matched_masks.append(matches)

        # Terminal reward in log-ns domain: relative improvements matter
        # equally at 1ns and 100ns.  Floor at 0.5ns prevents -inf.
        # Matches the convention in train_judge.rs: ln(max(ns, 0.5)).
        reward = -math.log(max(traj["final_cost_ns"], 0.5))
        rewards.append(reward)

    return sequences, rewards, matched_masks


def build_padded_batch(
    sequences: list[torch.Tensor],
    rewards: list[float],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Pad variable-length sequences into a single batch.

    Returns:
        padded: (B, T_max, STEP_DIM) padded input features.
        targets: (B, T_max, 1) per-step targets (reward broadcast, 0 at padding).
        padding_mask: (B, T_max) bool — True at padding positions.
        lengths: list of original sequence lengths per trajectory.
    """
    lengths = [seq.size(0) for seq in sequences]
    if not lengths:
        raise ValueError("Cannot build batch from empty sequence list")

    # pad_sequence: pads shorter sequences with 0.0 to match longest
    # Input: list of (T_i, STEP_DIM), Output: (B, T_max, STEP_DIM)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0).to(device)
    B, T_max, _ = padded.shape

    # Padding mask: True where position >= original length (padding)
    # Shape: (B, T_max)
    arange = torch.arange(T_max, device=device).unsqueeze(0)  # (1, T_max)
    len_tensor = torch.tensor(lengths, device=device).unsqueeze(1)  # (B, 1)
    padding_mask = arange >= len_tensor  # (B, T_max)

    # Targets: each real position gets the trajectory's terminal reward.
    # Padding positions get 0.0 (masked out of loss anyway).
    targets = torch.zeros(B, T_max, 1, device=device, dtype=torch.float32)
    for i, (length, reward) in enumerate(zip(lengths, rewards)):
        targets[i, :length, 0] = reward

    return padded, targets, padding_mask, lengths


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss that ignores padding positions.

    Args:
        pred: (B, T, 1) predicted values.
        target: (B, T, 1) target values.
        padding_mask: (B, T) bool — True at padding positions.

    Returns:
        Scalar mean MSE over non-padding positions. Raises if all positions
        are padding (should never happen with validated data).
    """
    # real_mask: (B, T, 1) — True at real (non-padding) positions
    real_mask = ~padding_mask.unsqueeze(-1)  # (B, T, 1)
    n_real = real_mask.sum()
    if n_real == 0:
        raise RuntimeError(
            "masked_mse_loss: all positions are padding — "
            "this should never happen with validated trajectories"
        )
    sq_err = (pred - target) ** 2
    # Zero out padding contributions, sum, divide by count of real positions
    return (sq_err * real_mask).sum() / n_real


# =============================================================================
# Training + advantage export
# =============================================================================

def train_and_export(args: argparse.Namespace) -> None:
    """Train the Critic and export per-step advantages."""
    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}", file=sys.stderr)

    # Load data
    trajectories = load_trajectories(Path(args.input))
    sequences, rewards, matched_masks = trajectories_to_tensors(trajectories)

    # Build padded batch (all trajectories in one tensor)
    padded, targets, padding_mask, lengths = build_padded_batch(
        sequences, rewards, device
    )
    B = padded.size(0)
    T_max = padded.size(1)
    n_real_steps = sum(lengths)
    print(
        f"Batch: {B} trajectories, T_max={T_max}, "
        f"{n_real_steps} real steps, "
        f"{B * T_max - n_real_steps} padding positions",
        file=sys.stderr,
    )

    # Build model
    model = CriticTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # Optionally load checkpoint — strict=True so dimension mismatches crash immediately
    if args.checkpoint and Path(args.checkpoint).exists():
        print(
            f"Loading checkpoint from {args.checkpoint}", file=sys.stderr
        )
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}", file=sys.stderr)

    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ---- Training loop (batched) ----
    model.train()
    best_loss = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        v_pred = model(padded, src_key_padding_mask=padding_mask)
        loss = masked_mse_loss(v_pred, targets, padding_mask)

        if not torch.isfinite(loss):
            print(
                f"Epoch {epoch + 1}/{args.epochs}  loss=NaN — stopping early, reverting to best checkpoint",
                file=sys.stderr,
            )
            if best_state is not None:
                model.load_state_dict(best_state)
            break

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{args.epochs}  loss={loss_val:.6f}",
                file=sys.stderr,
            )

    # ---- Export advantages ----
    output_path = Path(args.output)
    model.eval()
    with torch.no_grad():
        v_pred = model(padded, src_key_padding_mask=padding_mask)  # (B, T_max, 1)
        v_pred = v_pred.squeeze(-1).cpu()  # (B, T_max)

    with open(output_path, "w") as out_f:
        for i, length in enumerate(lengths):
            v_i = v_pred[i, :length]  # (T_i,) — only real positions
            adv = rewards[i] - v_i    # A_t = R_T - V_t
            
            # EXPLICIT PENALTY: Time-wasting rules (fired but didn't match) get a hard negative advantage.
            # This overrides the temporal credit assignment for that specific step, teaching the policy
            # to stop predicting high scores for rules that fail structural matching.
            step_matched = matched_masks[i][:length].to(device=adv.device)
            adv = torch.where(step_matched, adv, torch.full_like(adv, -0.01))
            
            # Replace any NaN/Inf with 0.0 — model diverged, neutral advantage
            if not torch.all(torch.isfinite(adv)):
                n_bad = (~torch.isfinite(adv)).sum().item()
                print(
                    f"WARNING: trajectory {i} has {n_bad}/{len(adv)} non-finite advantages, zeroing them",
                    file=sys.stderr,
                )
                adv = torch.where(torch.isfinite(adv), adv, torch.zeros_like(adv))
            record = {
                "trajectory_idx": i,
                "advantages": adv.tolist(),
            }
            out_f.write(json.dumps(record) + "\n")

    print(
        f"Wrote {len(sequences)} advantage records to {output_path}",
        file=sys.stderr,
    )

    # ---- Save checkpoint ----
    if args.checkpoint:
        torch.save(model.state_dict(), args.checkpoint)
        print(f"Saved checkpoint to {args.checkpoint}", file=sys.stderr)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Causal Sequence Transformer Critic for temporal credit assignment.",
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser(
        "train", help="Train critic and export advantages"
    )
    train_parser.add_argument(
        "--input", required=True, help="Path to trajectory JSONL"
    )
    train_parser.add_argument(
        "--output", required=True, help="Path to write advantages JSONL"
    )
    train_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to save/load model checkpoint (.pt)",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=50, help="Training epochs (default: 50)"
    )
    train_parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay (default: 1e-5)",
    )
    train_parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Transformer model dimension (default: 128)",
    )
    train_parser.add_argument(
        "--nhead",
        type=int,
        default=4,
        help="Number of attention heads (default: 4)",
    )
    train_parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of transformer layers (default: 3)",
    )
    train_parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (default: 0.1)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help(sys.stderr)
        raise SystemExit(1)

    if args.command == "train":
        train_and_export(args)
    else:
        raise ValueError(f"Unknown command: {args.command!r}")


if __name__ == "__main__":
    main()
