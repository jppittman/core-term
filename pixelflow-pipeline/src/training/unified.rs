//! # Unified Self-Play Trajectory Payload Structs
//!
//! These structs define the IPC boundary between Rust (Actor) and Python (Critic).
//!
//! - Rust writes [`Trajectory`] JSONL during self-play hill-climbing.
//! - Python reads trajectories, trains a Causal Transformer Critic, and writes
//!   [`TrajectoryAdvantages`] JSONL back.
//! - Rust reads advantages to apply policy gradient updates to the mask head.

use serde::{Deserialize, Serialize};

/// A single step in a self-play trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryStep {
    /// EdgeAccumulator state at time t: 4*K dual sums + edge_count + node_count = 130 floats.
    pub accumulator_state: Vec<f32>,
    /// Expression embedding at decision time (EMBED_DIM floats).
    /// This is one side of the bilinear product: score = expr_embed^T @ M @ rule_embed.
    pub expression_embedding: Vec<f32>,
    /// Rule embedding for the selected rule (EMBED_DIM floats).
    pub rule_embedding: Vec<f32>,
    /// Remaining node budget: node_budget - egraph.node_count(). Markov state.
    pub budget_remaining: i32,
    /// Remaining epoch budget: epoch_budget - current_epoch. Markov state.
    pub epochs_remaining: i32,
    /// sigmoid(bilinear_score) — the Actor's confidence in this rule.
    pub action_probability: f32,
    /// Whether the rule actually matched and was applied.
    pub matched: bool,
    /// JIT-benchmarked cost of the expression at this epoch (nanoseconds).
    /// All steps within the same epoch share this value.
    pub jit_cost_ns: f64,
    /// Edge list for embedding gradient flow: (parent_op, child_op, depth).
    /// Compact representation of the expression structure at decision time.
    /// Each tuple is (parent OpKind index as u8, child OpKind index as u8, effective depth as u16).
    #[serde(default)]
    pub edges: Vec<(u8, u8, u16)>,
    /// Graph accumulator state (VSA encoding) at decision time (GRAPH_INPUT_DIM floats).
    /// Used by the graph backbone for mask scoring (separate from expression EdgeAccumulator).
    #[serde(default)]
    pub graph_accumulator_state: Vec<f32>,
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
    /// JIT-benchmarked initial cost before any rewrites (nanoseconds).
    pub initial_cost_ns: f64,
    /// JIT-benchmarked execution time of the final compiled AST (nanoseconds).
    pub final_cost_ns: f64,
    /// Judge-estimated initial cost before any rewrites.
    #[serde(default)]
    pub initial_cost: Option<f32>,
    /// Judge-estimated final cost after all rewrites.
    #[serde(default)]
    pub final_cost: Option<f32>,
}

/// Per-trajectory advantage scores produced by the Python Critic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryAdvantages {
    /// Index into the trajectory batch (matches order in trajectories JSONL).
    pub trajectory_idx: usize,
    /// Per-step advantage A_t = R_T - V_t.
    pub advantages: Vec<f32>,
}

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
                    expression_embedding: vec![0.3; pixelflow_search::nnue::factored::EMBED_DIM],
                    rule_embedding: vec![0.5; pixelflow_search::nnue::factored::EMBED_DIM],
                    budget_remaining: 1500,
                    epochs_remaining: 10,
                    action_probability: 0.87,
                    matched: true,
                    jit_cost_ns: 8.75,
                    edges: vec![(2, 3, 0), (2, 4, 1)],
                    graph_accumulator_state: vec![0.0; pixelflow_search::nnue::factored::GRAPH_INPUT_DIM],
                },
                TrajectoryStep {
                    accumulator_state: vec![0.2; 130],
                    expression_embedding: vec![0.4; pixelflow_search::nnue::factored::EMBED_DIM],
                    rule_embedding: vec![0.6; pixelflow_search::nnue::factored::EMBED_DIM],
                    budget_remaining: 1200,
                    epochs_remaining: 9,
                    action_probability: 0.42,
                    matched: false,
                    jit_cost_ns: 8.75,
                    edges: vec![(5, 6, 3)],
                    graph_accumulator_state: vec![0.0; pixelflow_search::nnue::factored::GRAPH_INPUT_DIM],
                },
            ],
            initial_cost_ns: 10.25,
            final_cost_ns: 3.42,
            initial_cost: Some(10.5),
            final_cost: Some(3.2),
        };
        let json = serde_json::to_string(&traj).expect("serialize failed");
        let back: Trajectory = serde_json::from_str(&json).expect("deserialize failed");
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
        let json = serde_json::to_string(&adv).expect("serialize failed");
        let back: TrajectoryAdvantages = serde_json::from_str(&json).expect("deserialize failed");
        assert_eq!(back.advantages.len(), 3);
        assert!((back.advantages[1] - (-0.34)).abs() < 1e-6);
    }
}
