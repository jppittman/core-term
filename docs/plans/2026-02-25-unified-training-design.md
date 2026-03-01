# Unified Training Flow: Temporal Credit Assignment via Sequence Transformer

## Problem

The current training pipeline has two separate loops that don't share representations:

- **Loop A (Judge):** `train_online.rs` — adversarial self-play for the value head only
- **Loop B (Mask/Policy):** `collect_guide_data.rs` → `graph_teacher.py train-policy` — offline random perturbation, no self-play

The mask loop generates trajectories by randomly perturbing weights (Gaussian noise), which is good for bootstrapping but doesn't improve during collection. The judge loop trains the value head in isolation, so shared backbone representations diverge.

AlphaZero showed that joint training through a shared backbone produces stronger representations than training heads independently. We need one loop with one loss.

## Architecture

### Participants

**Actor (Rust, CPU):** ExprNnue with bilinear mask architecture. Executes guided hill-climbing over e-graphs. All inference and weight updates happen in Rust with hand-derived analytical gradients.

**Critic (Python, GPU):** Causal Sequence Transformer. Reads flat trajectory data from disk. Its sole purpose is temporal credit assignment — determining which rule at step 5 caused the cost drop at step 20. Produces per-step advantage scores A_t. No graph information; operates strictly on the Actor's flat mathematical state.

**Disk:** JSONL files serve as the IPC boundary between Rust and Python. No sockets, no orchestration framework.

### ExprNnue Data Flow (Reference)

```
EXPRESSION (AST)
    |
    v
EdgeAccumulator::from_expr_dedup()      130 floats
    |                                    [4x32 dual sums + edge_count + node_count]
    v
SHARED BACKBONE (W1 + ReLU)             130 -> 64 (HIDDEN_DIM)
    |
    v
compute_expr_embed()                     64 -> 24 (EMBED_DIM)
    |
    +------+------------------+
    v                         v
VALUE MLP                  MASK MLP
24 -> 16 -> 1              [24, value_pred] = 25 -> 16 -> 24
(predicted cost)           compute_mask_features()
                                |
                                v
                           bilinear_score()
                           mask_features(24) @ interaction(24x24) @ rule_embed(24) + bias
                                |
                                v
                           raw logit -> sigmoid -> action probability
```

Both heads share the backbone. Joint training updates all parameters through both losses simultaneously.

## Outer Loop

```
for round in 0..rounds:
    1. GENERATE   -- Rust: run N trajectories via hill-climbing with current mask weights
    2. EXPORT     -- Rust: write trajectory payloads to trajectories.jsonl
    3. CRITIQUE   -- Python: read JSONL, run Causal Transformer, write advantages.jsonl
    4. UPDATE     -- Rust: read advantages, joint backward pass through both heads
    5. CHECKPOINT -- Rust: save weights + metrics
```

All hyperparameters (N, rounds, learning rate, etc.) exposed as CLI args with small defaults for fast iteration.

## Phase 1: Replay Buffer Payload (Rust -> Disk)

For every completed extraction pass, Rust writes a sequence of T steps followed by the terminal cost.

### Per-Step Record (step t)

| Field | Type | Size | Description |
|-------|------|------|-------------|
| `accumulator_state` | `[f32; 130]` | 520 bytes | EdgeAccumulator at time t: 4x32 dual sums + edge_count + node_count |
| `rule_embedding` | `[f32; 24]` | 96 bytes | Rule encoder output for the selected rule |
| `action_probability` | `f32` | 4 bytes | sigmoid(bilinear_score) — the Actor's confidence in this rule |

### Terminal Record

| Field | Type | Description |
|-------|------|-------------|
| `final_cost` | `f32` | JIT-benchmarked execution time of the final compiled AST |

### Serialization

One JSON line per trajectory:

```json
{
  "seed_expr": "...",
  "steps": [
    {
      "accumulator_state": [0.1, 0.2, ...],
      "rule_embedding": [0.3, 0.4, ...],
      "action_probability": 0.87
    },
    ...
  ],
  "final_cost": 3.42
}
```

## Phase 2: Critic Forward Pass (Python)

The Critic is a standard Causal Sequence Transformer. It outputs a value prediction V_t for each state, not probabilities.

### Architecture

1. **Input Projection:** Concatenate `accumulator_state` (130) and `rule_embedding` (24) = 154 floats per step. Linear projection to Transformer hidden dimension d_model.

2. **Positional Encoding:** Standard 1D sinusoidal encoding injected additively. Encodes time-step order.

3. **Causal Self-Attention:** Standard Transformer encoder layers with causal mask. Step t attends only to steps <= t.

4. **Value Head:** Linear projection from hidden state at each step t to scalar V_t. This predicts the final cost from that point in the trajectory.

### Critic Training

Loss = MSE(V_t, R_T) where R_T = -final_cost for all steps.

The Critic trains on accumulated replay data. It improves its temporal credit assignment as it sees more trajectories.

## Phase 3: Advantage Calculation (Python -> Disk)

### Reward Signal

The environment provides only a terminal reward:
- R_T = -final_cost (at final step T)
- R_t = 0 for all t < T

### Advantage

A_t = R_T - V_t

- A_t > 0: this rule sequence led to faster execution than expected
- A_t < 0: this rule was a detrimental detour

### Export

Write advantages back to disk:

```json
{
  "trajectory_idx": 0,
  "advantages": [0.12, -0.34, 0.56, ...]
}
```

## Phase 4: Policy Gradient Update (Rust)

Rust reads A_t from disk and performs a joint backward pass through the full network.

### Policy Loss (Mask Head)

For each step t:

```
L_policy = -log(pi(a_t | s_t)) * A_t
```

Gradients flow through: `interaction` -> `mask_mlp` -> `expr_proj` -> `shared backbone (W1)`

### Value Loss (Value Head)

For the terminal cost:

```
L_value = (value_pred - final_cost)^2
```

Gradients flow through: `value_mlp` -> `expr_proj` -> `shared backbone (W1)`

### Joint Loss

```
L = L_policy + c_value * L_value
```

Where c_value is a hyperparameter balancing the two heads (AlphaZero uses 1.0).

### Trainable Parameters

All parameters in the forward path receive gradients:

| Component | Parameters | Gradient Source |
|-----------|-----------|-----------------|
| Shared backbone W1, b1 | 130x64 + 64 | Both losses |
| Expr projection w, b | 64x24 + 24 | Both losses |
| Value MLP w1, b1, w2, b2 | 24x16 + 16 + 16 + 1 | Value loss only |
| Mask MLP w1, b1, w2, b2 | 25x16 + 16 + 16x24 + 24 | Policy loss only |
| Interaction matrix | 24x24 | Policy loss only |
| Mask rule bias | per-rule | Policy loss only |

### Implementation

Hand-derived analytical gradients in Rust. No autograd dependency. The existing `train_mask_reinforce.rs` has the pattern for interaction matrix gradients; this extends it to the full path.

## Expression Source

`BwdGenerator` with kernel seeds generates expressions. No pre-collected corpus needed. The self-play loop creates its own distribution of expressions as training progresses.

## Kill List

These files are superseded by the unified binary and should be deleted:

| File | Reason |
|------|--------|
| `src/bin/train_online.rs` | Value head now trained in unified loop |
| `src/bin/collect_guide_data.rs` | Replaced by self-play trajectory generation |
| `src/bin/train_mask.rs` | Superseded |
| `src/bin/train_mask_reinforce.rs` | REINFORCE logic moves into unified binary |
| `src/bin/gen_mask_data.rs` | No longer needed |
| `src/bin/collect_judge_data.rs` | No longer needed |
| `pixelflow-search/src/nnue/guide.rs` | GuideNnue replaced by ExprNnue mask |

## CLI Interface

```
cargo run --release -p pixelflow-pipeline --features training --bin train_unified -- \
  --rounds 30 \
  --trajectories-per-round 50 \
  --max-steps 50 \
  --lr 0.001 \
  --value-coeff 1.0 \
  --model pixelflow-pipeline/data/judge.bin \
  --output-dir pixelflow-pipeline/data/unified \
  --critic-script pixelflow-pipeline/scripts/critic.py
```

Default values tuned for fast iteration on a single machine. Scale up by increasing `--rounds` and `--trajectories-per-round`.

## Verification

```bash
# 1. Rust compiles
cargo check -p pixelflow-pipeline --features training

# 2. Generate one batch of trajectories
cargo run --release -p pixelflow-pipeline --features training --bin train_unified -- \
  --rounds 1 --trajectories-per-round 5

# 3. Python critic produces advantages
uv run pixelflow-pipeline/scripts/critic.py \
  --input pixelflow-pipeline/data/unified/trajectories_r0.jsonl \
  --output pixelflow-pipeline/data/unified/advantages_r0.jsonl

# 4. Full loop smoke test (3 rounds, 10 trajectories)
cargo run --release -p pixelflow-pipeline --features training --bin train_unified -- \
  --rounds 3 --trajectories-per-round 10

# 5. Weights actually change (loss decreases over rounds)
# Check unified/metrics.jsonl for round-over-round improvement
```
