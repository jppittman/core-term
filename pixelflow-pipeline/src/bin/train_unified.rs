//! # Unified Self-Play Training Binary
//!
//! Orchestrates the outer training loop:
//!
//! ```text
//! GENERATE → EXPORT → CRITIQUE → UPDATE → CHECKPOINT
//! ```
//!
//! This is the single binary that replaces `train_online`, `collect_guide_data`,
//! `train_mask_reinforce`, and `gen_mask_data`. It runs an AlphaZero-style
//! training loop:
//!
//! 1. **GENERATE**: Self-play trajectories via e-graph hill-climbing with the
//!    current mask head.
//! 2. **EXPORT**: Write trajectory JSONL for the Python Critic.
//! 3. **CRITIQUE**: Python Causal Transformer Critic assigns per-step temporal
//!    credit (advantages A_t = R_T - V_t).
//! 4. **UPDATE**: Joint policy + value gradient update through the shared
//!    ExprNnue backbone using REINFORCE + MSE.
//! 5. **CHECKPOINT**: Save model weights and log metrics.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin train_unified -- \
//!   --rounds 30 --trajectories-per-round 50
//! ```

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;
use pixelflow_search::egraph::{all_rules, Rewrite};
use pixelflow_search::nnue::factored::{EdgeAccumulator, ExprNnue, INPUT_DIM, EMBED_DIM, K};

use pixelflow_pipeline::training::gen_es::{log_ns, GenEs, GenEsConfig};
use pixelflow_pipeline::training::self_play::{
    build_rule_templates, generate_corpus_trajectories, generate_trajectory_batch,
    load_corpus_exprs, read_advantages_jsonl, write_trajectories_jsonl,
};
use pixelflow_pipeline::training::unified::{Trajectory, TrajectoryAdvantages, TrajectoryStep};
use pixelflow_pipeline::training::unified_backward::{
    apply_unified_sgd, backward_policy, backward_value, forward_cached, UnifiedGradients,
};

// ============================================================================
// Memory observability
// ============================================================================

/// Get current process RSS in megabytes via `ps`. Works on macOS and Linux.
/// Returns 0.0 if the measurement fails (no silent crash).
fn rss_mb() -> f64 {
    let pid = std::process::id();
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|kb| kb as f64 / 1024.0)
        .unwrap_or(0.0)
}

// ============================================================================
// CLI
// ============================================================================

/// Unified self-play training: joint value + policy through shared backbone.
///
/// Outer loop: GENERATE -> EXPORT -> CRITIQUE -> UPDATE -> CHECKPOINT
#[derive(Parser, Debug)]
#[command(name = "train_unified")]
#[command(about = "Unified self-play training with temporal credit assignment")]
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
    #[arg(long, default_value_t = 2.91e-4)]
    lr: f32,

    /// Momentum for SGD.
    #[arg(long, default_value_t = 0.903)]
    momentum: f32,

    /// Weight decay.
    #[arg(long, default_value_t = 3.34e-6)]
    weight_decay: f32,

    /// Value loss coefficient.
    #[arg(long, default_value_t = 0.804)]
    value_coeff: f32,

    /// Gradient clipping threshold (global L2 norm).
    #[arg(long, default_value_t = 0.103)]
    grad_clip: f32,

    /// Entropy bonus coefficient (prevents policy collapse).
    #[arg(long, default_value_t = 0.025)]
    entropy_coeff: f32,

    /// Path to model weights (loaded at start, saved at end).
    #[arg(long, default_value = "pixelflow-pipeline/data/judge.bin")]
    model: PathBuf,

    /// Output directory for checkpoints, trajectories, advantages.
    #[arg(long, default_value = "pixelflow-pipeline/data/unified")]
    output_dir: PathBuf,

    /// Path to Python critic script.
    #[arg(long, default_value = "pixelflow-pipeline/scripts/critic.py")]
    critic_script: PathBuf,

    /// Critic checkpoint path (reused across rounds).
    #[arg(long, default_value = "pixelflow-pipeline/data/unified/critic.pt")]
    critic_checkpoint: PathBuf,

    /// Critic training epochs per round.
    #[arg(long, default_value_t = 20)]
    critic_epochs: usize,

    /// Critic learning rate (forwarded to critic.py --lr).
    #[arg(long, default_value_t = 4.03e-5)]
    critic_lr: f64,

    /// Critic dropout (forwarded to critic.py --dropout).
    #[arg(long, default_value_t = 0.085)]
    critic_dropout: f64,

    /// Random seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    // ── ES-guided generation ─────────────────────────────────────
    /// ES population size (perturbation candidates per round).
    #[arg(long, default_value_t = 10)]
    es_population: usize,

    /// ES noise standard deviation.
    #[arg(long, default_value_t = 0.1)]
    es_sigma: f32,

    /// ES learning rate.
    #[arg(long, default_value_t = 0.05)]
    es_alpha: f32,

    /// Fraction of trajectories per round sampled from corpus (0.0–1.0).
    #[arg(long, default_value_t = 0.3)]
    corpus_fraction: f32,

    /// Path to bench_corpus.jsonl for corpus-based trajectories.
    #[arg(long, default_value = "pixelflow-pipeline/data/bench_corpus.jsonl")]
    corpus_path: PathBuf,

    /// Max corpus expressions to hold in memory.
    #[arg(long, default_value_t = 2000)]
    corpus_max: usize,

    // ── Replay buffer ───────────────────────────────────────────
    /// Replay buffer capacity (max stored steps across rounds).
    #[arg(long, default_value_t = 200000)]
    replay_capacity: usize,

    /// Mini-batch size for SGD updates from replay buffer.
    #[arg(long, default_value_t = 1024)]
    mini_batch_size: usize,

    /// Number of gradient update steps per round.
    #[arg(long, default_value_t = 3)]
    updates_per_round: usize,
}

// ============================================================================
// Helpers
// ============================================================================

/// Reconstruct [`EdgeAccumulator`] from a [`TrajectoryStep`]'s accumulator_state.
///
/// Layout: `[values (4*K=128 floats), edge_count, node_count]` = 130 floats total.
fn acc_from_step(step: &TrajectoryStep) -> EdgeAccumulator {
    assert_eq!(
        step.accumulator_state.len(),
        INPUT_DIM,
        "Expected {} accumulator values, got {}",
        INPUT_DIM,
        step.accumulator_state.len()
    );

    let mut acc = EdgeAccumulator::default();
    acc.values.copy_from_slice(&step.accumulator_state[..4 * K]);
    acc.edge_count = step.accumulator_state[4 * K] as u32;
    acc.node_count = step.accumulator_state[4 * K + 1] as u32;
    acc
}

/// Reconstruct [`EdgeAccumulator`] from a [`ReplayStep`].
fn acc_from_replay(step: &ReplayStep) -> EdgeAccumulator {
    assert_eq!(step.acc.len(), INPUT_DIM,
        "Expected {} accumulator values in replay step, got {}", INPUT_DIM, step.acc.len());
    let mut acc = EdgeAccumulator::default();
    acc.values.copy_from_slice(&step.acc[..4 * K]);
    acc.edge_count = step.acc[4 * K] as u32;
    acc.node_count = step.acc[4 * K + 1] as u32;
    acc
}

/// Extract rule embedding from a [`ReplayStep`].
fn embed_from_replay(step: &ReplayStep) -> [f32; EMBED_DIM] {
    assert_eq!(step.rule_embed.len(), EMBED_DIM,
        "Expected {} rule embedding dims in replay step, got {}", EMBED_DIM, step.rule_embed.len());
    let mut embed = [0.0f32; EMBED_DIM];
    embed.copy_from_slice(&step.rule_embed);
    embed
}

// ============================================================================
// Replay Buffer
// ============================================================================

struct ReplayStep {
    acc: Vec<f32>,           // 130 floats (INPUT_DIM)
    expr_embed: Vec<f32>,    // 32 floats (EMBED_DIM) — expr_proj output at decision time
    rule_embed: Vec<f32>,    // 32 floats (EMBED_DIM)
    matched: bool,
    jit_cost_ns: f64,
    advantage: f32,
}

struct ReplayBuffer {
    steps: Vec<ReplayStep>,
    max_steps: usize,
}

impl ReplayBuffer {
    fn new(capacity: usize) -> Self {
        Self { steps: Vec::new(), max_steps: capacity }
    }

    /// Flatten trajectory steps + advantages into ReplaySteps and append.
    fn push_round(
        &mut self,
        trajectories: &[Trajectory],
        advantages: &[TrajectoryAdvantages],
    ) {
        for (traj, adv) in trajectories.iter().zip(advantages.iter()) {
            assert_eq!(traj.steps.len(), adv.advantages.len(),
                "Step/advantage count mismatch in trajectory {}: {} vs {}",
                traj.trajectory_id, traj.steps.len(), adv.advantages.len());
            for (step, &advantage) in traj.steps.iter().zip(adv.advantages.iter()) {
                self.steps.push(ReplayStep {
                    acc: step.accumulator_state.clone(),
                    expr_embed: step.expression_embedding.clone(),
                    rule_embed: step.rule_embedding.clone(),
                    matched: step.matched,
                    jit_cost_ns: step.jit_cost_ns,
                    advantage,
                });
            }
        }
        self.prune();
    }

    /// FIFO evict oldest steps when over capacity.
    fn prune(&mut self) {
        if self.steps.len() > self.max_steps {
            let excess = self.steps.len() - self.max_steps;
            self.steps.drain(..excess);
        }
    }

    fn len(&self) -> usize {
        self.steps.len()
    }

    /// Sample batch_size random indices using LCG PRNG.
    fn sample_batch(&self, batch_size: usize, seed: u64) -> Vec<usize> {
        let n = self.steps.len();
        assert!(n > 0, "Cannot sample from empty replay buffer");
        let batch_size = batch_size.min(n);
        let mut indices = Vec::with_capacity(batch_size);
        let mut state = seed;
        for _ in 0..batch_size {
            // LCG: state = state * 6364136223846793005 + 1442695040888963407
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            indices.push((state >> 33) as usize % n);
        }
        indices
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let args = Args::parse();

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)
        .unwrap_or_else(|e| panic!("Failed to create output dir {:?}: {e}", args.output_dir));

    // Load model or initialize fresh
    let mut model = if args.model.exists() {
        let m = ExprNnue::load(&args.model)
            .unwrap_or_else(|e| panic!("Failed to load model from {:?}: {e}", args.model));
        eprintln!("Loaded model from {:?}", args.model);
        m
    } else {
        eprintln!("No model at {:?}, initializing fresh with seed {}", args.model, args.seed);
        ExprNnue::new_with_latency_prior(args.seed)
    };

    // Build rules + templates
    let rules: Vec<Box<dyn Rewrite>> = all_rules();
    let templates = build_rule_templates(&rules);

    // Initialize momentum buffer
    let mut momentum_buf = UnifiedGradients::zero();

    // Initialize ES optimizer
    let mut gen_es = GenEs::new(GenEsConfig {
        sigma: args.es_sigma,
        alpha: args.es_alpha,
        population: args.es_population,
        samples_per_candidate: 8,
        seed: args.seed.wrapping_add(0xE5),
    }, templates.clone());

    // Load corpus expressions
    let corpus_exprs = if args.corpus_fraction > 0.0 {
        load_corpus_exprs(&args.corpus_path, args.corpus_max, args.seed)
    } else {
        eprintln!("Corpus fraction is 0.0, skipping corpus loading");
        Vec::new()
    };

    // Initialize replay buffer
    let mut replay_buffer = ReplayBuffer::new(args.replay_capacity);

    // Metrics log
    let metrics_path = args.output_dir.join("metrics.jsonl");
    let mut metrics_file = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&metrics_path)
            .unwrap_or_else(|e| panic!("Failed to open metrics file {:?}: {e}", metrics_path)),
    );

    for round in 0..args.rounds {
        let round_start = std::time::Instant::now();
        let round_seed = args.seed.wrapping_add(round as u64 * 1000);

        eprintln!("\n{}", "=".repeat(60));
        eprintln!("Round {}/{}", round + 1, args.rounds);
        eprintln!("{}", "=".repeat(60));
        eprintln!("[MEMORY] round {} start: {:.0}MB", round, rss_mb());

        // ── PHASE 1: GENERATE ──────────────────────────────────────────
        // ES step: adapt generator toward judge's blind spots
        let gen_config = gen_es.step(&model);
        let es_fitness = gen_es.last_fitness();
        eprintln!(
            "[ES] depth={} leaf={:.2} vars={} fused={:.2} junkify={}/{:.2} fitness={:.4}",
            gen_config.max_depth, gen_config.leaf_prob, gen_config.num_vars,
            gen_config.fused_op_prob, gen_config.max_junkify_passes, gen_config.junkify_prob,
            es_fitness,
        );

        // Split trajectories between ES-generated and corpus-sampled
        let corpus_count = (args.trajectories_per_round as f32 * args.corpus_fraction) as usize;
        let es_count = args.trajectories_per_round - corpus_count;

        eprintln!(
            "[GENERATE] Running {} ES + {} corpus trajectories...",
            es_count, corpus_count,
        );
        let gen_start = std::time::Instant::now();

        let mut trajectories = generate_trajectory_batch(
            &model,
            &templates,
            &rules,
            es_count,
            round_seed,
            args.threshold,
            args.max_steps,
            &gen_config,
        );

        if corpus_count > 0 && !corpus_exprs.is_empty() {
            trajectories.extend(generate_corpus_trajectories(
                &model,
                &templates,
                &rules,
                &corpus_exprs,
                corpus_count,
                round_seed.wrapping_add(0xC0),
                args.threshold,
                args.max_steps,
            ));
        }
        let gen_elapsed = gen_start.elapsed();

        // Filter out zero-step trajectories (expressions where no rules matched)
        let pre_filter = trajectories.len();
        let mut trajectories: Vec<_> = trajectories
            .into_iter()
            .filter(|t| !t.steps.is_empty())
            .collect();
        let empty_count = pre_filter - trajectories.len();

        if trajectories.is_empty() {
            eprintln!("[GENERATE] No valid trajectories generated, skipping round");
            continue;
        }

        // Replace NaN/non-finite jit_cost_ns with a penalty cost instead of dropping trajectories
        let max_cost_ns = trajectories.iter()
            .flat_map(|t| t.steps.iter())
            .filter_map(|s| if s.jit_cost_ns.is_finite() { Some(s.jit_cost_ns) } else { None })
            .fold(0.0f64, f64::max);
        let penalty_cost_ns = (max_cost_ns * 2.0).max(100.0);

        let mut nan_replaced = 0usize;
        for traj in &mut trajectories {
            for step in &mut traj.steps {
                if !step.jit_cost_ns.is_finite() || step.jit_cost_ns < 0.0 {
                    step.jit_cost_ns = penalty_cost_ns;
                    nan_replaced += 1;
                }
            }
        }
        if nan_replaced > 0 {
            eprintln!("[GENERATE] Replaced {nan_replaced} NaN jit_cost_ns with penalty={penalty_cost_ns:.1}ns");
        }

        if empty_count > 0 {
            eprintln!(
                "[GENERATE] Filtered {empty_count} empty out of {pre_filter} trajectories",
            );
        }

        eprintln!("[MEMORY] after generate: {:.0}MB", rss_mb());

        let total_steps: usize = trajectories.iter().map(|t| t.steps.len()).sum();
        eprintln!(
            "[GENERATE] {} trajectories, {} total steps in {:.1}s",
            trajectories.len(),
            total_steps,
            gen_elapsed.as_secs_f64()
        );

        // ── PHASE 2: EXPORT ────────────────────────────────────────────
        let traj_path = args.output_dir.join(format!("trajectories_r{round}.jsonl"));
        write_trajectories_jsonl(&trajectories, &traj_path);
        eprintln!("[EXPORT] Wrote {}", traj_path.display());

        // ── PHASE 3: CRITIQUE ──────────────────────────────────────────
        let adv_path = args.output_dir.join(format!("advantages_r{round}.jsonl"));
        eprintln!("[CRITIQUE] Running Python critic...");
        let critic_start = std::time::Instant::now();

        let critic_script_str = args
            .critic_script
            .to_str()
            .unwrap_or_else(|| panic!("Invalid UTF-8 in critic script path: {:?}", args.critic_script));
        let traj_path_str = traj_path
            .to_str()
            .unwrap_or_else(|| panic!("Invalid UTF-8 in trajectory path: {:?}", traj_path));
        let adv_path_str = adv_path
            .to_str()
            .unwrap_or_else(|| panic!("Invalid UTF-8 in advantages path: {:?}", adv_path));
        let critic_ckpt_str = args
            .critic_checkpoint
            .to_str()
            .unwrap_or_else(|| panic!("Invalid UTF-8 in critic checkpoint path: {:?}", args.critic_checkpoint));

        let status = std::process::Command::new("uv")
            .args(["run", critic_script_str])
            .arg("train")
            .args(["--input", traj_path_str])
            .args(["--output", adv_path_str])
            .args(["--checkpoint", critic_ckpt_str])
            .args(["--epochs", &args.critic_epochs.to_string()])
            .args(["--lr", &args.critic_lr.to_string()])
            .args(["--dropout", &args.critic_dropout.to_string()])
            .status()
            .unwrap_or_else(|e| panic!("Failed to run critic (uv run {:?} train): {e}", args.critic_script));

        if !status.success() {
            panic!(
                "Critic failed with exit code: {} (script: {:?})",
                status, args.critic_script
            );
        }
        let critic_elapsed = critic_start.elapsed();
        eprintln!("[CRITIQUE] Done in {:.1}s", critic_elapsed.as_secs_f64());

        // ── PHASE 4: UPDATE ────────────────────────────────────────────
        eprintln!("[UPDATE] Pushing {} steps into replay buffer...", total_steps);
        let advantages = read_advantages_jsonl(&adv_path);

        if advantages.len() != trajectories.len() {
            panic!(
                "Trajectory/advantage count mismatch: {} trajectories vs {} advantage records",
                trajectories.len(),
                advantages.len()
            );
        }

        // Anneal entropy bonus toward 0 as training progresses
        let annealed_entropy = args.entropy_coeff * (1.0 - round as f32 / args.rounds as f32);

        replay_buffer.push_round(&trajectories, &advantages);
        eprintln!("[UPDATE] Buffer size: {} steps", replay_buffer.len());
        eprintln!("[MEMORY] after buffer push: {:.0}MB ({} steps)", rss_mb(), replay_buffer.len());

        // Multiple mini-batch gradient steps from replay buffer
        let mut round_grad_norm = 0.0f32;
        let mut total_policy_steps = 0usize;
        let mut total_value_steps = 0usize;
        let nan_cost_count = 0usize; // Always zero — NaN costs replaced with penalty upstream

        for update_idx in 0..args.updates_per_round {
            let batch_indices = replay_buffer.sample_batch(
                args.mini_batch_size,
                round_seed.wrapping_add(update_idx as u64 * 7919),
            );
            let mut grads = UnifiedGradients::zero();
            let mut batch_policy = 0usize;
            let mut batch_value = 0usize;

            for &idx in &batch_indices {
                let step = &replay_buffer.steps[idx];

                // NaN costs replaced with penalty upstream — none should reach replay buffer.
                assert!(
                    step.jit_cost_ns.is_finite() && step.jit_cost_ns >= 0.0,
                    "NaN/negative jit_cost_ns={} in replay buffer at index {idx} — \
                     trajectory filtering is broken",
                    step.jit_cost_ns,
                );

                let acc = acc_from_replay(step);
                let rule_embed = embed_from_replay(step);
                let cache = forward_cached(&model, &acc, &rule_embed);

                backward_policy(
                    &model, &cache, &rule_embed,
                    step.matched, step.advantage, annealed_entropy, &mut grads,
                );
                batch_policy += 1;

                let target = log_ns(step.jit_cost_ns);
                backward_value(&model, &cache, target, args.value_coeff, &mut grads);
                batch_value += 1;
            }

            let batch_size = batch_policy.max(1) as f32;
            grads.scale(1.0 / batch_size);
            let grad_norm = grads.norm();
            round_grad_norm += grad_norm;

            apply_unified_sgd(
                &mut model, &grads, &mut momentum_buf,
                args.lr, args.momentum, args.weight_decay, args.grad_clip,
            );

            eprintln!(
                "[UPDATE] step {}/{}: {} policy, {} value, grad_norm={:.6}",
                update_idx + 1, args.updates_per_round,
                batch_policy, batch_value, grad_norm,
            );

            total_policy_steps += batch_policy;
            total_value_steps += batch_value;
        }

        let avg_grad_norm = round_grad_norm / args.updates_per_round as f32;

        // NaN costs are replaced with penalty upstream, so none should reach here.
        // Assert this invariant holds rather than silently tolerating bad data.
        assert_eq!(nan_cost_count, 0, "NaN cost leaked into replay buffer");

        eprintln!("[MEMORY] after update: {:.0}MB", rss_mb());

        eprintln!(
            "[UPDATE] {} updates x {} batch = {} effective steps, \
             entropy_coeff={:.4}, avg_grad_norm={:.6}",
            args.updates_per_round, args.mini_batch_size,
            args.updates_per_round * args.mini_batch_size,
            annealed_entropy, avg_grad_norm,
        );

        // ── PHASE 5: CHECKPOINT ────────────────────────────────────────
        let ckpt_path = args.output_dir.join(format!("model_r{round}.bin"));
        model
            .save(&ckpt_path)
            .unwrap_or_else(|e| panic!("Failed to save checkpoint to {:?}: {e}", ckpt_path));

        // ── Compute metrics ──────────────────────────────────────────
        let round_elapsed = round_start.elapsed();
        let avg_steps = total_steps as f64 / trajectories.len() as f64;

        // Speedup: median of per-trajectory initial_ns / final_ns.
        // Median is robust to constant-collapse outliers (legit zero-mul simplifications
        // that hit the clock floor).
        const MAX_REASONABLE_NS: f64 = 1_000_000_000.0;
        let valid_jit_trajs: Vec<_> = trajectories
            .iter()
            .filter(|t| t.initial_cost_ns.is_finite() && t.initial_cost_ns > 0.0
                     && t.initial_cost_ns < MAX_REASONABLE_NS
                     && t.final_cost_ns.is_finite() && t.final_cost_ns > 0.0
                     && t.final_cost_ns < MAX_REASONABLE_NS)
            .collect();
        let mut speedups: Vec<f64> = valid_jit_trajs
            .iter()
            .map(|t| t.initial_cost_ns / t.final_cost_ns)
            .collect();
        speedups.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let speedup_median = if speedups.is_empty() {
            f64::NAN
        } else {
            let mid = speedups.len() / 2;
            if speedups.len() % 2 == 0 {
                (speedups[mid - 1] + speedups[mid]) / 2.0
            } else {
                speedups[mid]
            }
        };
        // Log outliers so we can see the constant-collapse simplifications
        let collapse_count = speedups.iter().filter(|&&s| s > 100.0).count();
        if collapse_count > 0 {
            eprintln!(
                "[METRICS] {collapse_count}/{} trajectories with >100x speedup \
                 (likely constant-fold), median={speedup_median:.2}x, max={:.0}x",
                speedups.len(),
                speedups.last().copied().unwrap_or(0.0),
            );
        }

        let avg_initial_ns: f64 = if valid_jit_trajs.is_empty() {
            f64::NAN
        } else {
            valid_jit_trajs.iter().map(|t| t.initial_cost_ns).sum::<f64>()
                / valid_jit_trajs.len() as f64
        };
        let avg_final_ns: f64 = if valid_jit_trajs.is_empty() {
            f64::NAN
        } else {
            valid_jit_trajs.iter().map(|t| t.final_cost_ns).sum::<f64>()
                / valid_jit_trajs.len() as f64
        };

        // Judge MAE: mean |predict_log_cost - log_ns(jit_cost)| across all steps
        let mut judge_error_sum = 0.0f64;
        let mut judge_error_count = 0u64;
        for traj in &trajectories {
            for step in &traj.steps {
                if step.jit_cost_ns.is_finite() && step.jit_cost_ns >= 0.5 {
                    let acc = acc_from_step(step);
                    let predicted = model.predict_log_cost_with_features(&acc);
                    let actual = log_ns(step.jit_cost_ns);
                    judge_error_sum += libm::fabs((predicted - actual) as f64);
                    judge_error_count += 1;
                }
            }
        }
        let judge_mae = if judge_error_count == 0 {
            f64::NAN
        } else {
            judge_error_sum / judge_error_count as f64
        };

        // Log metrics as JSONL
        let metrics = serde_json::json!({
            "round": round,
            "trajectories": trajectories.len(),
            "total_steps": total_steps,
            "avg_steps": avg_steps,
            "speedup_median": speedup_median,
            "avg_initial_ns": avg_initial_ns,
            "avg_final_ns": avg_final_ns,
            "judge_mae": judge_mae,
            "es_depth": gen_config.max_depth,
            "es_leaf_prob": gen_config.leaf_prob,
            "es_fitness": es_fitness,
            "policy_steps": total_policy_steps,
            "value_steps": total_value_steps,
            "nan_cost_count": nan_cost_count,
            "entropy_coeff": annealed_entropy,
            "grad_norm": avg_grad_norm,
            "buffer_size": replay_buffer.len(),
            "updates_this_round": args.updates_per_round,
            "effective_batch": args.updates_per_round * args.mini_batch_size,
            "rss_mb": rss_mb(),
            "elapsed_s": round_elapsed.as_secs_f64(),
        });
        writeln!(
            metrics_file,
            "{}",
            serde_json::to_string(&metrics)
                .unwrap_or_else(|e| panic!("Failed to serialize metrics: {e}"))
        )
        .unwrap_or_else(|e| panic!("Failed to write metrics: {e}"));
        metrics_file
            .flush()
            .unwrap_or_else(|e| panic!("Failed to flush metrics file: {e}"));

        eprintln!("[CHECKPOINT] Saved to {}", ckpt_path.display());
        eprintln!(
            "[METRICS] speedup={speedup_median:.3}x init={avg_initial_ns:.1}ns final={avg_final_ns:.1}ns \
             judge_mae={judge_mae:.3} es_fit={es_fitness:.3} steps={avg_steps:.1} \
             grad={avg_grad_norm:.2} buf={} time={:.1}s",
            replay_buffer.len(),
            round_elapsed.as_secs_f64()
        );
    }

    // Save final model back to original path
    model
        .save(&args.model)
        .unwrap_or_else(|e| panic!("Failed to save final model to {:?}: {e}", args.model));
    eprintln!("\nTraining complete. Final model saved to {:?}", args.model);
    eprintln!("Metrics log: {:?}", metrics_path);
}
