# Unified Training Flow Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the fragmented training pipeline (train_online, collect_guide_data, train_mask_reinforce, graph_teacher online-train) with a single self-play loop that jointly trains both heads through a shared backbone, using a Python Causal Transformer Critic for temporal credit assignment.

**Architecture:** Rust Actor (ExprNnue) generates self-play trajectories via hill-climbing, writes flat step payloads to JSONL. Python Critic (Causal Sequence Transformer) reads trajectories, produces per-step advantage scores A_t. Rust reads advantages, applies joint policy+value gradient update with hand-derived analytical backprop through the full network. Disk is the IPC boundary.

**Tech Stack:** Rust (pixelflow-search, pixelflow-pipeline), Python (PyTorch, no torch-geometric), JSONL serialization, JIT benchmarking for terminal cost.

**Design doc:** `docs/plans/2026-02-25-unified-training-design.md`

---

## Task 1: Trajectory Payload Structs

Define the JSONL schema that Rust writes and Python reads. These live in the pipeline training module so both the generator and the updater can share them.

**Files:**
- Create: `pixelflow-pipeline/src/training/unified.rs`
- Modify: `pixelflow-pipeline/src/training/mod.rs` (add `pub mod unified;`)

**Step 1: Write the data structs**

```rust
// pixelflow-pipeline/src/training/unified.rs

use serde::{Serialize, Deserialize};

/// A single step in a self-play trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryStep {
    /// EdgeAccumulator state at time t: 4*K dual sums + edge_count + node_count = 130 floats.
    pub accumulator_state: Vec<f32>,
    /// Rule embedding for the selected rule (EMBED_DIM = 24 floats).
    pub rule_embedding: Vec<f32>,
    /// Rule index in the global rule table.
    pub rule_idx: usize,
    /// sigmoid(bilinear_score) — the Actor's confidence in this rule.
    pub action_probability: f32,
    /// Whether the rule actually matched and was applied.
    pub matched: bool,
}

/// A complete self-play trajectory with terminal cost.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    /// Unique identifier for this trajectory.
    pub trajectory_id: String,
    /// Human-readable seed expression.
    pub seed_expr: String,
    /// Sequence of steps taken during hill-climbing.
    pub steps: Vec<TrajectoryStep>,
    /// JIT-benchmarked execution time of the final compiled AST (nanoseconds).
    pub final_cost_ns: f64,
    /// Judge-estimated initial cost before any rewrites.
    pub initial_cost: f32,
    /// Judge-estimated final cost after all rewrites.
    pub final_cost: f32,
}

/// Per-trajectory advantage scores produced by the Python Critic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryAdvantages {
    /// Index into the trajectory batch (matches order in trajectories JSONL).
    pub trajectory_idx: usize,
    /// Per-step advantage A_t = R_T - V_t.
    pub advantages: Vec<f32>,
}
```

**Step 2: Write round-trip serialization test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trajectory_round_trip_jsonl() {
        let traj = Trajectory {
            trajectory_id: "test_0".into(),
            seed_expr: "(X + Y)".into(),
            steps: vec![
                TrajectoryStep {
                    accumulator_state: vec![0.1; 130],
                    rule_embedding: vec![0.5; 24],
                    rule_idx: 3,
                    action_probability: 0.87,
                    matched: true,
                },
                TrajectoryStep {
                    accumulator_state: vec![0.2; 130],
                    rule_embedding: vec![0.6; 24],
                    rule_idx: 7,
                    action_probability: 0.42,
                    matched: false,
                },
            ],
            final_cost_ns: 3.42,
            initial_cost: 10.5,
            final_cost: 3.2,
        };
        let json = serde_json::to_string(&traj).expect("serialize");
        let back: Trajectory = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.steps.len(), 2);
        assert_eq!(back.trajectory_id, "test_0");
        assert!((back.final_cost_ns - 3.42).abs() < 1e-6);
    }

    #[test]
    fn advantages_round_trip_jsonl() {
        let adv = TrajectoryAdvantages {
            trajectory_idx: 0,
            advantages: vec![0.12, -0.34, 0.56],
        };
        let json = serde_json::to_string(&adv).expect("serialize");
        let back: TrajectoryAdvantages = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.advantages.len(), 3);
        assert!((back.advantages[1] - (-0.34)).abs() < 1e-6);
    }
}
```

**Step 3: Run tests**

```bash
cargo test -p pixelflow-pipeline --features training --lib training::unified
```

**Step 4: Wire up module export**

Add `pub mod unified;` to `pixelflow-pipeline/src/training/mod.rs`.

**Step 5: Commit**

```bash
git add pixelflow-pipeline/src/training/unified.rs pixelflow-pipeline/src/training/mod.rs
git commit -m "feat(training): add trajectory payload structs for unified training"
```

---

## Task 2: Full Backward Pass Infrastructure

The existing REINFORCE code (`factored.rs:2938`) only backprops through `interaction + mask_rule_bias`. We need gradients through the full path: `interaction → mask_mlp → expr_proj → backbone W1`, plus value loss through `value_mlp → expr_proj → backbone W1`.

**Files:**
- Create: `pixelflow-pipeline/src/training/unified_backward.rs`
- Modify: `pixelflow-pipeline/src/training/mod.rs` (add export)
- Reference: `pixelflow-search/src/nnue/factored.rs` (forward pass at lines 1545-1936)
- Reference: `pixelflow-pipeline/src/training/dual_head.rs` (existing backward pattern at lines 402-473)

### Step 1: Forward cache struct

The forward cache stores all intermediate activations needed for backprop. We need to cache everything from the forward pass.

```rust
// pixelflow-pipeline/src/training/unified_backward.rs

use pixelflow_search::nnue::{
    ExprNnue, EdgeAccumulator, Expr, EMBED_DIM, HIDDEN_DIM, MLP_HIDDEN,
    MASK_INPUT_DIM, INPUT_DIM,
};

/// Cached forward pass activations for analytical backprop.
pub struct UnifiedForwardCache {
    /// EdgeAccumulator input (130 floats).
    pub acc_values: [f32; INPUT_DIM],
    /// Pre-activation backbone (before ReLU).
    pub pre_relu: [f32; HIDDEN_DIM],
    /// Post-ReLU backbone hidden.
    pub hidden: [f32; HIDDEN_DIM],
    /// Expression embedding (24 dims).
    pub expr_embed: [f32; EMBED_DIM],
    /// Value MLP hidden (pre-ReLU, 16 dims).
    pub value_mlp_pre_relu: [f32; MLP_HIDDEN],
    /// Value MLP hidden (post-ReLU).
    pub value_mlp_hidden: [f32; MLP_HIDDEN],
    /// Value prediction (scalar).
    pub value_pred: f32,
    /// Mask MLP input: [expr_embed; value_pred] (25 dims).
    pub mask_input: [f32; MASK_INPUT_DIM],
    /// Mask MLP hidden (pre-ReLU, 16 dims).
    pub mask_mlp_pre_relu: [f32; MLP_HIDDEN],
    /// Mask MLP hidden (post-ReLU).
    pub mask_mlp_hidden: [f32; MLP_HIDDEN],
    /// Mask features (24 dims).
    pub mask_features: [f32; EMBED_DIM],
    /// Transformed = mask_features @ interaction (24 dims).
    pub transformed: [f32; EMBED_DIM],
    /// Raw bilinear score (scalar).
    pub score: f32,
    /// Action probability = sigmoid(score).
    pub prob: f32,
}
```

### Step 2: Caching forward pass

```rust
/// Run full forward pass and cache all intermediates for backprop.
pub fn forward_cached(
    net: &ExprNnue,
    acc: &EdgeAccumulator,
    rule_embed: &[f32; EMBED_DIM],
    rule_idx: usize,
) -> UnifiedForwardCache {
    // ... (replicates forward_shared, compute_expr_embed, value_mlp_forward,
    //  compute_mask_features, bilinear_score — but caches every intermediate)
}
```

### Step 3: Gradient accumulator struct

```rust
/// Accumulated gradients for all trainable parameters.
/// Mirrors ExprNnue structure but stores gradients.
pub struct UnifiedGradients {
    // Backbone
    pub d_w1: [[f32; HIDDEN_DIM]; INPUT_DIM],
    pub d_b1: [f32; HIDDEN_DIM],
    // Expr projection
    pub d_expr_proj_w: [[f32; EMBED_DIM]; HIDDEN_DIM],
    pub d_expr_proj_b: [f32; EMBED_DIM],
    // Value MLP
    pub d_value_mlp_w1: [[f32; MLP_HIDDEN]; EMBED_DIM],
    pub d_value_mlp_b1: [f32; MLP_HIDDEN],
    pub d_value_mlp_w2: [f32; MLP_HIDDEN],
    pub d_value_mlp_b2: f32,
    // Mask MLP
    pub d_mask_mlp_w1: [[f32; MLP_HIDDEN]; MASK_INPUT_DIM],
    pub d_mask_mlp_b1: [f32; MLP_HIDDEN],
    pub d_mask_mlp_w2: [[f32; EMBED_DIM]; MLP_HIDDEN],
    pub d_mask_mlp_b2: [f32; EMBED_DIM],
    // Interaction matrix
    pub d_interaction: [[f32; EMBED_DIM]; EMBED_DIM],
    // Mask rule bias
    pub d_mask_rule_bias: [f32; 1024], // MASK_MAX_RULES
}
```

### Step 4: Policy backward pass

Derive gradients for `L_policy = -log(pi(a_t|s_t)) * A_t`:

```rust
/// Backprop policy loss through full path.
/// d_score = ((1 - prob) if approved, else -prob) * advantage
/// Then chain rule through: bilinear → mask_mlp → mask_input split →
///   (value_mlp path for d_value_pred) + (expr_proj path) → backbone
pub fn backward_policy(
    net: &ExprNnue,
    cache: &UnifiedForwardCache,
    rule_embed: &[f32; EMBED_DIM],
    rule_idx: usize,
    approved: bool,
    advantage: f32,
    grads: &mut UnifiedGradients,
) {
    // 1. d_score
    let d_log_prob = if approved { 1.0 - cache.prob } else { -cache.prob };
    let d_score = d_log_prob * advantage;

    // 2. d_interaction, d_mask_features, d_mask_rule_bias
    grads.d_mask_rule_bias[rule_idx] += d_score;
    // d_transformed = d_score * rule_embed
    // d_mask_features[i] += sum_j(d_transformed[j] * interaction[i][j])
    // d_interaction[i][j] += d_score * mask_features[i] * rule_embed[j]

    // 3. Backprop through mask_mlp (ReLU gate from cache)
    // d_mask_mlp_w2, d_mask_mlp_b2, d_mask_mlp_hidden
    // d_mask_mlp_w1, d_mask_mlp_b1, d_mask_input

    // 4. Split d_mask_input into d_expr_embed (first 24) and d_value_pred (25th)

    // 5. If d_value_pred != 0, backprop through value_mlp to get additional d_expr_embed
    // d_value_mlp_w2, d_value_mlp_b2, d_value_mlp_hidden
    // d_value_mlp_w1, d_value_mlp_b1 → d_expr_embed (accumulated)

    // 6. Backprop through expr_proj: d_expr_proj_w, d_expr_proj_b, d_hidden

    // 7. Backprop through ReLU gate and backbone W1
    // d_b1, d_w1
}
```

### Step 5: Value backward pass

```rust
/// Backprop value loss: L_value = (value_pred - final_cost)^2
pub fn backward_value(
    net: &ExprNnue,
    cache: &UnifiedForwardCache,
    final_cost: f32,
    value_coeff: f32,
    grads: &mut UnifiedGradients,
) {
    let d_value = 2.0 * (cache.value_pred - final_cost) * value_coeff;
    // Backprop through value_mlp → expr_embed → expr_proj → backbone
    // (Same chain as steps 5-7 above, but starting from d_value)
}
```

### Step 6: SGD apply

```rust
/// Apply accumulated gradients via momentum SGD with gradient clipping.
pub fn apply_unified_sgd(
    net: &mut ExprNnue,
    grads: &UnifiedGradients,
    momentum_buf: &mut UnifiedGradients,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
) {
    // Same pattern as dual_head.rs:728 (apply_momentum_update)
    // For each parameter: clip grad, momentum update, weight decay, step
}
```

### Step 7: Numerical gradient test

This is the critical correctness test. Compute analytical gradient, compare against finite-difference numerical gradient.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Numerical gradient checking: perturb each parameter by epsilon,
    /// measure loss change, compare against analytical gradient.
    #[test]
    fn numerical_gradient_check_policy() {
        let mut net = ExprNnue::new();
        net.randomize_mask_only(42);
        // ... set up small test case
        // For a subset of parameters (e.g., interaction[0][0]):
        //   analytical = grads.d_interaction[0][0]
        //   numerical = (loss(param + eps) - loss(param - eps)) / (2 * eps)
        //   assert relative error < 1e-3
    }

    #[test]
    fn numerical_gradient_check_value() {
        // Same pattern for value loss path
    }

    #[test]
    fn joint_gradient_accumulates() {
        // Verify backward_policy + backward_value accumulate correctly
        // d_w1 should have contributions from both losses
    }
}
```

### Step 8: Run tests

```bash
cargo test -p pixelflow-pipeline --features training --lib training::unified_backward
```

### Step 9: Commit

```bash
git add pixelflow-pipeline/src/training/unified_backward.rs pixelflow-pipeline/src/training/mod.rs
git commit -m "feat(training): full analytical backward pass through shared backbone"
```

---

## Task 3: Python Critic (Causal Sequence Transformer)

A standalone Python script that reads trajectory JSONL, trains a Causal Transformer, and writes per-step advantages back to disk.

**Files:**
- Create: `pixelflow-pipeline/scripts/critic.py`

**No torch-geometric dependency.** Pure PyTorch. The Critic sees flat vectors, not graphs.

### Step 1: Write the Critic model

```python
#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.1"]
# ///
"""Causal Sequence Transformer Critic for temporal credit assignment.

Reads self-play trajectories (JSONL), trains a value model V_t for each
step, exports per-step advantages A_t = R_T - V_t.

Usage:
    uv run critic.py train --input trajectories.jsonl --output advantages.jsonl
    uv run critic.py train --input trajectories.jsonl --output advantages.jsonl --checkpoint critic.pt
"""

import argparse, json, math, sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants matching Rust side
ACC_DIM = 130        # EdgeAccumulator size
RULE_DIM = 24        # EMBED_DIM
STEP_DIM = ACC_DIM + RULE_DIM  # 154 floats per step


class SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class CriticTransformer(nn.Module):
    """Causal Sequence Transformer that predicts V_t at each step."""
    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(STEP_DIM, d_model)
        self.pe = SinusoidalPE(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (batch, seq_len, STEP_DIM) -> (batch, seq_len, 1)"""
        h = self.input_proj(x)           # (B, T, d_model)
        h = self.pe(h)                   # + positional encoding
        # Causal mask: step t attends only to steps <= t
        T = h.size(1)
        causal = torch.triu(torch.ones(T, T, device=h.device), diagonal=1).bool()
        h = self.transformer(h, mask=causal)
        return self.value_head(h)        # (B, T, 1)
```

### Step 2: Data loading

```python
def load_trajectories(path: Path) -> list[dict]:
    """Load trajectory JSONL, return list of trajectory dicts."""
    trajectories = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trajectories.append(json.loads(line))
    if not trajectories:
        raise ValueError(f"No trajectories found in {path}")
    return trajectories


def trajectories_to_tensors(trajectories: list[dict]) -> tuple[list[torch.Tensor], list[float]]:
    """Convert trajectory dicts to (step_sequences, terminal_rewards)."""
    sequences = []
    rewards = []
    for traj in trajectories:
        steps = traj["steps"]
        if not steps:
            continue
        seq = []
        for step in steps:
            features = step["accumulator_state"] + step["rule_embedding"]
            assert len(features) == STEP_DIM, f"Expected {STEP_DIM}, got {len(features)}"
            seq.append(features)
        sequences.append(torch.tensor(seq, dtype=torch.float32))
        rewards.append(-traj["final_cost_ns"])  # R_T = -cost (lower cost = higher reward)
    return sequences, rewards
```

### Step 3: Training loop + advantage export

```python
def train_and_export(args):
    """Train Critic on trajectories, export advantages."""
    trajectories = load_trajectories(Path(args.input))
    sequences, rewards = trajectories_to_tensors(trajectories)

    if not sequences:
        raise ValueError("No valid trajectories to train on")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CriticTransformer(
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dropout=args.dropout,
    ).to(device)

    if args.checkpoint and Path(args.checkpoint).exists():
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for seq, reward in zip(sequences, rewards):
            seq = seq.unsqueeze(0).to(device)           # (1, T, STEP_DIM)
            target = torch.full((1, seq.size(1), 1), reward, device=device)

            v_pred = model(seq)                          # (1, T, 1)
            loss = F.mse_loss(v_pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}  loss={total_loss/len(sequences):.6f}", file=sys.stderr)

    # Export advantages
    model.eval()
    advantages_out = []
    with torch.no_grad():
        for i, (seq, reward) in enumerate(zip(sequences, rewards)):
            seq = seq.unsqueeze(0).to(device)
            v_pred = model(seq).squeeze(0).squeeze(-1)   # (T,)
            adv = reward - v_pred                        # A_t = R_T - V_t
            advantages_out.append({
                "trajectory_idx": i,
                "advantages": adv.cpu().tolist(),
            })

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        for a in advantages_out:
            f.write(json.dumps(a) + "\n")

    if args.checkpoint:
        torch.save(model.state_dict(), args.checkpoint)

    print(f"Wrote {len(advantages_out)} advantage records to {output_path}", file=sys.stderr)
```

### Step 4: CLI

```python
def main():
    parser = argparse.ArgumentParser(description="Causal Transformer Critic")
    parser.add_argument("--input", required=True, help="Trajectory JSONL path")
    parser.add_argument("--output", required=True, help="Advantages JSONL output path")
    parser.add_argument("--checkpoint", default=None, help="Critic model checkpoint path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()
    train_and_export(args)


if __name__ == "__main__":
    main()
```

### Step 5: Smoke test

```bash
# Create a tiny test trajectory file
echo '{"trajectory_id":"t0","seed_expr":"X+Y","steps":[{"accumulator_state":[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],"rule_embedding":[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],"rule_idx":3,"action_probability":0.87,"matched":true}],"final_cost_ns":3.42,"initial_cost":10.5,"final_cost":3.2}' > /tmp/test_traj.jsonl

uv run pixelflow-pipeline/scripts/critic.py \
  --input /tmp/test_traj.jsonl \
  --output /tmp/test_adv.jsonl \
  --epochs 5

# Verify output
cat /tmp/test_adv.jsonl
```

### Step 6: Commit

```bash
git add pixelflow-pipeline/scripts/critic.py
git commit -m "feat(training): causal sequence transformer critic for temporal credit assignment"
```

---

## Task 4: Self-Play Trajectory Generator

Adapt the trajectory runner from `collect_guide_data.rs` (lines 296-470) to record per-step payload for the Critic. This runs hill-climbing with the current mask weights (no perturbation — self-play means the policy improves, not random noise).

**Files:**
- Create: `pixelflow-pipeline/src/training/self_play.rs`
- Modify: `pixelflow-pipeline/src/training/mod.rs` (add export)
- Reference: `pixelflow-pipeline/src/bin/collect_guide_data.rs:296-470` (trajectory runner)
- Reference: `pixelflow-search/src/nnue/factored.rs:1943-1988` (mask_score_all_rules)
- Reference: `pixelflow-search/src/nnue/factored.rs:566-586` (EdgeAccumulator)
- Reference: `pixelflow-pipeline/src/jit_bench.rs:66` (benchmark_jit)

### Step 1: Write the self-play trajectory runner

Core function that runs one trajectory and records the Critic payload:

```rust
// pixelflow-pipeline/src/training/self_play.rs

use pixelflow_search::nnue::{ExprNnue, Expr, EdgeAccumulator, RuleTemplates, EMBED_DIM, INPUT_DIM};
use pixelflow_search::egraph::{
    EGraph, all_rules, extract_neural, expr_tree_to_nnue, expr_to_egraph, Rewrite,
};
use crate::training::unified::{Trajectory, TrajectoryStep};
use crate::jit_bench::benchmark_jit;

/// Run a single self-play trajectory and record per-step Critic payload.
///
/// Returns the trajectory with all steps recorded, or None if the
/// expression failed to produce a valid trajectory.
pub fn run_self_play_trajectory(
    seed_expr: &Expr,
    seed_name: &str,
    model: &ExprNnue,
    rule_embeds: &[[f32; EMBED_DIM]],
    rules: &[Rewrite],
    threshold: f32,
    max_epochs: usize,
    trajectory_id: String,
) -> Option<Trajectory> {
    // 1. Build e-graph from seed, extract initial cost
    // 2. For each epoch:
    //    a. Extract best expression via extract_neural
    //    b. Build EdgeAccumulator
    //    c. Score all rules via mask_score_all_rules_with_hidden
    //    d. For each approved rule (sigmoid > threshold):
    //       - Record TrajectoryStep { accumulator_state, rule_embedding, action_probability, rule_idx, matched }
    //       - Apply rule to e-graph
    //    e. Update best cost
    // 3. JIT benchmark final expression for terminal cost
    // 4. Return Trajectory
}

/// Generate N trajectories from random seeds using BwdGenerator.
pub fn generate_trajectory_batch(
    model: &ExprNnue,
    templates: &RuleTemplates,
    rules: &[Rewrite],
    count: usize,
    seed: u64,
    threshold: f32,
    max_epochs: usize,
) -> Vec<Trajectory> {
    // Use BwdGenerator to create seed expressions
    // Run run_self_play_trajectory for each
    // Return batch
}
```

### Step 2: Write trajectory recording test

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_play_trajectory_records_steps() {
        let model = ExprNnue::new();
        // ... create simple seed expr (X + Y)
        // ... run trajectory
        // Assert: steps is non-empty
        // Assert: each step has 130-float accumulator_state
        // Assert: each step has 24-float rule_embedding
        // Assert: action_probability in [0, 1]
        // Assert: final_cost_ns > 0
    }
}
```

### Step 3: Run tests

```bash
cargo test -p pixelflow-pipeline --features training --lib training::self_play
```

### Step 4: Commit

```bash
git add pixelflow-pipeline/src/training/self_play.rs pixelflow-pipeline/src/training/mod.rs
git commit -m "feat(training): self-play trajectory generator with Critic payload recording"
```

---

## Task 5: Unified Training Binary

The main binary that orchestrates the outer loop: GENERATE → EXPORT → CRITIQUE → UPDATE → CHECKPOINT.

**Files:**
- Create: `pixelflow-pipeline/src/bin/train_unified.rs`
- Reference: `pixelflow-pipeline/src/bin/train_online.rs` (CLI pattern, checkpoint pattern)
- Reference: `pixelflow-pipeline/src/bin/train_mask_reinforce.rs` (REINFORCE decision loop)

### Step 1: CLI args

```rust
// pixelflow-pipeline/src/bin/train_unified.rs

use clap::Parser;

/// Unified self-play training: joint value + policy through shared backbone.
#[derive(Parser, Debug)]
#[command(name = "train_unified")]
struct Args {
    /// Number of outer training rounds.
    #[arg(long, default_value_t = 30)]
    rounds: usize,

    /// Trajectories per round.
    #[arg(long, default_value_t = 50)]
    trajectories_per_round: usize,

    /// Max hill-climbing steps per trajectory.
    #[arg(long, default_value_t = 50)]
    max_steps: usize,

    /// Mask threshold (sigmoid > threshold to apply rule).
    #[arg(long, default_value_t = 0.3)]
    threshold: f32,

    /// Learning rate for SGD.
    #[arg(long, default_value_t = 0.001)]
    lr: f32,

    /// Momentum for SGD.
    #[arg(long, default_value_t = 0.9)]
    momentum: f32,

    /// Weight decay.
    #[arg(long, default_value_t = 1e-5)]
    weight_decay: f32,

    /// Value loss coefficient (AlphaZero uses 1.0).
    #[arg(long, default_value_t = 1.0)]
    value_coeff: f32,

    /// Gradient clipping threshold.
    #[arg(long, default_value_t = 1.0)]
    grad_clip: f32,

    /// Path to model weights (loaded at start, saved at checkpoints).
    #[arg(long, default_value = "pixelflow-pipeline/data/judge.bin")]
    model: std::path::PathBuf,

    /// Output directory for checkpoints, trajectories, advantages.
    #[arg(long, default_value = "pixelflow-pipeline/data/unified")]
    output_dir: std::path::PathBuf,

    /// Path to Python critic script.
    #[arg(long, default_value = "pixelflow-pipeline/scripts/critic.py")]
    critic_script: std::path::PathBuf,

    /// Critic checkpoint path (reused across rounds).
    #[arg(long, default_value = "pixelflow-pipeline/data/unified/critic.pt")]
    critic_checkpoint: std::path::PathBuf,

    /// Critic training epochs per round.
    #[arg(long, default_value_t = 50)]
    critic_epochs: usize,

    /// Random seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}
```

### Step 2: Outer loop

```rust
fn main() {
    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir).expect("create output dir");

    // Load model
    let mut model = ExprNnue::load(&args.model).unwrap_or_else(|e| {
        panic!("Failed to load model from {:?}: {e}", args.model);
    });

    // Build rules + templates + rule embeddings
    let rules = all_rules();
    let templates = build_rule_templates(&rules);
    let rule_embeds = model.encode_all_rules_from_templates(&templates);

    // Initialize momentum buffer
    let mut momentum_buf = UnifiedGradients::zero();

    for round in 0..args.rounds {
        let round_seed = args.seed.wrapping_add(round as u64 * 1000);

        // ── PHASE 1: GENERATE ──
        let trajectories = generate_trajectory_batch(
            &model, &templates, &rules,
            args.trajectories_per_round, round_seed,
            args.threshold, args.max_steps,
        );

        // ── PHASE 2: EXPORT ──
        let traj_path = args.output_dir.join(format!("trajectories_r{round}.jsonl"));
        write_trajectories_jsonl(&trajectories, &traj_path);

        // ── PHASE 3: CRITIQUE (shell out to Python) ──
        let adv_path = args.output_dir.join(format!("advantages_r{round}.jsonl"));
        let status = std::process::Command::new("uv")
            .args(["run", args.critic_script.to_str().unwrap()])
            .args(["--input", traj_path.to_str().unwrap()])
            .args(["--output", adv_path.to_str().unwrap()])
            .args(["--checkpoint", args.critic_checkpoint.to_str().unwrap()])
            .args(["--epochs", &args.critic_epochs.to_string()])
            .status()
            .expect("Failed to run critic");
        if !status.success() {
            panic!("Critic failed with exit code: {status}");
        }

        // ── PHASE 4: UPDATE ──
        let advantages = read_advantages_jsonl(&adv_path);
        let mut grads = UnifiedGradients::zero();

        for (traj, adv) in trajectories.iter().zip(advantages.iter()) {
            assert_eq!(traj.steps.len(), adv.advantages.len(),
                "Step count mismatch: {} steps vs {} advantages", traj.steps.len(), adv.advantages.len());

            for (step, &advantage) in traj.steps.iter().zip(adv.advantages.iter()) {
                // Reconstruct EdgeAccumulator from step.accumulator_state
                let acc = acc_from_slice(&step.accumulator_state);
                let rule_embed = embed_from_slice(&step.rule_embedding);

                // Forward with caching
                let cache = forward_cached(&model, &acc, &rule_embed, step.rule_idx);

                // Policy gradient
                backward_policy(&model, &cache, &rule_embed, step.rule_idx,
                    step.matched, advantage, &mut grads);
            }

            // Value gradient (one per trajectory, using final step's state)
            if let Some(last_step) = traj.steps.last() {
                let acc = acc_from_slice(&last_step.accumulator_state);
                let rule_embed = embed_from_slice(&last_step.rule_embedding);
                let cache = forward_cached(&model, &acc, &rule_embed, last_step.rule_idx);
                backward_value(&model, &cache, traj.final_cost_ns as f32,
                    args.value_coeff, &mut grads);
            }
        }

        // Normalize gradients by batch size
        let batch_size = trajectories.iter().map(|t| t.steps.len()).sum::<usize>().max(1) as f32;
        grads.scale(1.0 / batch_size);

        // Apply momentum SGD
        apply_unified_sgd(&mut model, &grads, &mut momentum_buf,
            args.lr, args.momentum, args.weight_decay);

        // Re-encode rule embeddings (backbone changed)
        let rule_embeds = model.encode_all_rules_from_templates(&templates);

        // ── PHASE 5: CHECKPOINT ──
        let ckpt_path = args.output_dir.join(format!("model_r{round}.bin"));
        model.save(&ckpt_path).expect("save checkpoint");

        // Log metrics
        let avg_improvement: f32 = trajectories.iter()
            .map(|t| (t.initial_cost - t.final_cost) / t.initial_cost.max(1e-6))
            .sum::<f32>() / trajectories.len().max(1) as f32;
        eprintln!("Round {round}/{}: trajectories={}, avg_improvement={avg_improvement:.4}, grad_norm={:.6}",
            args.rounds, trajectories.len(), grads.norm());
    }

    // Save final model
    model.save(&args.model).expect("save final model");
    eprintln!("Training complete. Final model saved to {:?}", args.model);
}
```

### Step 3: Integration smoke test

```bash
# Build
cargo build --release -p pixelflow-pipeline --features training --bin train_unified

# Run 1 round with 3 trajectories (smoke test)
cargo run --release -p pixelflow-pipeline --features training --bin train_unified -- \
  --rounds 1 --trajectories-per-round 3 --max-steps 10 --critic-epochs 5

# Verify output files exist
ls pixelflow-pipeline/data/unified/
# Should see: trajectories_r0.jsonl, advantages_r0.jsonl, model_r0.bin, critic.pt
```

### Step 4: Commit

```bash
git add pixelflow-pipeline/src/bin/train_unified.rs
git commit -m "feat(training): unified self-play training binary with joint value+policy loss"
```

---

## Task 6: Delete Dead Code

Remove superseded files. Do this after Tasks 1-5 are green.

**Files to delete:**
- `pixelflow-pipeline/src/bin/train_online.rs`
- `pixelflow-pipeline/src/bin/collect_guide_data.rs`
- `pixelflow-pipeline/src/bin/train_mask.rs`
- `pixelflow-pipeline/src/bin/train_mask_reinforce.rs`
- `pixelflow-pipeline/src/bin/gen_mask_data.rs`
- `pixelflow-pipeline/src/bin/collect_judge_data.rs`
- `pixelflow-search/src/nnue/guide.rs`

**Files to modify:**
- `pixelflow-search/src/nnue/mod.rs` — remove `pub mod guide;` if present
- `pixelflow-pipeline/Cargo.toml` — remove `[[bin]]` entries for deleted binaries
- `pixelflow-pipeline/scripts/graph_teacher.py` — remove `online-train` command (keep `train-policy` for now as reference, mark deprecated)

### Step 1: Delete files

```bash
git rm pixelflow-pipeline/src/bin/train_online.rs
git rm pixelflow-pipeline/src/bin/collect_guide_data.rs
git rm pixelflow-pipeline/src/bin/train_mask.rs
git rm pixelflow-pipeline/src/bin/train_mask_reinforce.rs
git rm pixelflow-pipeline/src/bin/gen_mask_data.rs
git rm pixelflow-pipeline/src/bin/collect_judge_data.rs
git rm pixelflow-search/src/nnue/guide.rs
```

### Step 2: Fix imports and module declarations

Remove any `pub mod guide;` from `pixelflow-search/src/nnue/mod.rs`. Remove `[[bin]]` entries from `Cargo.toml` if they're explicitly listed (Cargo auto-discovers from `src/bin/` so this may be a no-op). Fix any compilation errors from broken imports.

### Step 3: Verify everything compiles

```bash
cargo check --workspace
cargo test --workspace
```

### Step 4: Commit

```bash
git commit -m "chore: delete superseded training binaries and GuideNnue

Kill list:
- train_online.rs (value head now in unified loop)
- collect_guide_data.rs (replaced by self-play)
- train_mask.rs, train_mask_reinforce.rs, gen_mask_data.rs (superseded)
- collect_judge_data.rs (no longer needed)
- guide.rs (GuideNnue replaced by ExprNnue mask)"
```

---

## Task 7: Update Memory + Docs

Update project memory and training architecture docs to reflect the new unified flow.

**Files:**
- Modify: `~/.claude/projects/-Users-jppittman-Documents-Projects-core-term/memory/nnue-training.md`
- Modify: `~/.claude/projects/-Users-jppittman-Documents-Projects-core-term/memory/MEMORY.md`

### Step 1: Update nnue-training.md

Replace Loop A / Loop B sections with unified flow description. Update key files table. Remove references to deleted binaries.

### Step 2: Update MEMORY.md

Update architecture status, next major task section.

### Step 3: Commit design doc

```bash
git add docs/plans/2026-02-25-unified-training-design.md docs/plans/2026-02-25-unified-training-plan.md
git commit -m "docs: unified training design and implementation plan"
```

---

## Dependency Graph

```
Task 1 (payload structs) ─────────────┐
                                       ├──> Task 5 (unified binary)
Task 2 (backward pass) ───────────────┤
                                       │
Task 3 (Python critic) ───────────────┤
                                       │
Task 4 (trajectory generator) ────────┘
                                       │
                                       v
                                Task 6 (delete dead code)
                                       │
                                       v
                                Task 7 (update docs)
```

Tasks 1-4 are independent and can be dispatched in parallel. Task 5 depends on all of them. Tasks 6-7 are cleanup after Task 5 is green.
