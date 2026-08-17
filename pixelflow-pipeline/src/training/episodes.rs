//! # Episode Generation
//!
//! Runs budget-bounded e-graph saturation episodes and JIT-benchmarks the
//! initial and final (post-saturation) forms of a seed expression.
//!
//! This is the surviving core of what was `self_play.rs`: the RL policy
//! (mask-head rule scoring, threshold-gated rule approval, per-step
//! Trajectory/PFTJ export for the transformer critic) has been deleted per
//! docs/plans/2026-07-07-guided-saturation-redesign.md — that estimator was
//! methodologically unsound and its policy was never consumed by the
//! compiler. What remains — expression building, budget-bounded saturation,
//! JIT benchmarking of initial/final forms, semantic-equivalence checking —
//! is deliberately kept simple and synchronous: **the seed of the future
//! episode collector, not the collector itself** (mirroring
//! `pixelflow_search::egraph::labeler::run_episode`, which plays the same
//! role one layer down — at the e-graph/provenance level, without JIT
//! measurement).

use std::path::Path;

use pixelflow_ir::{ExprArena, ExprId};
use pixelflow_search::egraph::saturate::saturate_with_full_budget;
use pixelflow_search::egraph::{
    EGraph, IncrementalExtractor, Rewrite, all_rules, extract_neural_to_arena,
};
use pixelflow_search::nnue::ExprNnue;
use pixelflow_search::nnue::RuleTemplates;
use serde::Serialize;

use crate::jit_bench::{BenchMode, BenchSession};
use crate::training::corpus;
use crate::training::factored::arena_to_kernel_code;
use crate::training::mint::normalized_label_ns;

/// Build [`RuleTemplates`] from rule definitions.
///
/// Each rule provides LHS/RHS expression templates via the [`Rewrite`] trait.
pub fn build_rule_templates(rules: &[Box<dyn Rewrite>]) -> RuleTemplates {
    let mut templates = RuleTemplates::with_capacity(rules.len());

    for (idx, rule) in rules.iter().enumerate() {
        templates.build(idx, rule.as_ref());
    }

    templates
}

// ============================================================================
// Episode outcome
// ============================================================================

/// Outcome of one budget-bounded saturation episode: the seed expression's
/// JIT-benchmarked cost before and after saturation.
///
/// No policy, no critic, no per-step export payload — just the measured
/// facts an episode collector needs: what the e-graph achieved within
/// budget, ground-truth JIT cost at both ends, and a proof the rewrite
/// preserved semantics.
pub struct Episode {
    /// Human-readable identifier for this episode (propagated from the caller).
    pub episode_id: String,
    /// Name of the seed expression (for logging).
    pub seed_name: String,
    /// The seed expression, unmodified.
    pub initial_arena: ExprArena,
    pub initial_root: ExprId,
    /// Overhead-adjusted, sentinel-normalized single-eval latency of the seed
    /// expression (nanoseconds, [`BenchMode::Latency`] — chain-serialized per
    /// audit H3, call overhead subtracted per audit M1, expressed in the
    /// session's opening clock per
    /// [`normalized_label_ns`](crate::training::mint::normalized_label_ns)).
    pub initial_cost_ns: f64,
    /// The expression extracted after budget-bounded saturation.
    pub final_arena: ExprArena,
    pub final_root: ExprId,
    /// Overhead-adjusted, sentinel-normalized single-eval latency of the
    /// extracted expression (nanoseconds, same measurement contract as
    /// `initial_cost_ns` — same clock, so the ratio of the two is a speedup
    /// rather than a speedup plus whatever the machine did in between).
    pub final_cost_ns: f64,
    /// The e-class budget saturation was run under.
    pub node_budget: usize,
    /// The iteration budget saturation was run under.
    pub epoch_budget: usize,
}

/// A structured record of why an episode minted no label (audit M2).
///
/// Emitted as one JSON line on stderr per excluded episode so failures are
/// machine-countable, never a silent `None`. When the episode collector is
/// built, these records are what its exclusion-rate alarm will consume.
#[derive(Serialize)]
struct EpisodeExclusion<'a> {
    episode_id: &'a str,
    seed_name: &'a str,
    /// Pipeline stage that failed (e.g. `initial_bench_failed`,
    /// `equivalence_failed`).
    stage: &'a str,
    /// Human-readable detail: the underlying error, or the divergence report.
    detail: &'a str,
}

/// Emit one structured exclusion line to stderr. Panics if serialization
/// itself fails — a failure path that cannot report is worse than a crash.
fn log_exclusion(episode_id: &str, seed_name: &str, stage: &str, detail: &str) {
    let record = EpisodeExclusion {
        episode_id,
        seed_name,
        stage,
        detail,
    };
    let json = serde_json::to_string(&record).unwrap_or_else(|e| {
        panic!("failed to serialize EpisodeExclusion for episode {episode_id}: {e}")
    });
    eprintln!("EPISODE_EXCLUDED {json}");
}

/// The seed expression an episode starts from: the arena, its root, and the
/// name used in logs and exclusion records.
pub struct EpisodeSeed<'a> {
    pub arena: &'a ExprArena,
    pub root: ExprId,
    pub name: &'a str,
}

/// Identity and saturation budget for an episode run.
///
/// The seed's name lives on [`EpisodeSeed`], not here: it names the expression,
/// and one field cannot be the authority for two structs.
pub struct EpisodeSpec {
    pub max_epochs: usize,
    pub episode_id: String,
}

/// Run a single budget-bounded saturation episode.
///
/// Benchmarks run through the caller's [`BenchSession`] so episode labels get
/// the same sentinel/QoS/tick-floor protection as every other minted label
/// (audit H4/H5); costs are overhead-adjusted [`BenchMode::Latency`]
/// measurements (audit H3/M1 — chain-serialized latency is what production
/// per-pixel evaluation pays).
///
/// Returns `None` if the seed or extracted expression cannot be JIT-benchmarked,
/// is degenerate (NaN/Inf constants), or if the rewrite fails the
/// semantic-equivalence check. Every `None` is preceded by a structured
/// `EPISODE_EXCLUDED` stderr line (audit M2 — exclusions are never silent).
///
/// # Algorithm
///
/// 1. Insert the seed expression into an e-graph with all rewrite rules.
/// 2. JIT-benchmark the seed for ground-truth initial cost.
/// 3. Saturate within a size/iteration budget (randomized per episode, scaled
///    by expression size — see [`saturate_with_full_budget`]). No rule
///    filtering: every rule fires whenever it matches, same as production
///    `saturate()`, just budget-capped for training-corpus diversity.
/// 4. Extract the best expression the budget-bounded e-graph achieved.
/// 5. JIT-benchmark the extraction for ground-truth final cost.
/// 6. Verify the rewrite preserved semantics (SIMD-lane equivalence at a
///    fixed test point).
pub fn run_episode(
    session: &mut BenchSession,
    seed: EpisodeSeed<'_>,
    model: &ExprNnue,
    spec: EpisodeSpec,
) -> Option<Episode> {
    let EpisodeSeed {
        arena: seed_arena,
        root: seed_root,
        name: seed_name,
    } = seed;
    let EpisodeSpec {
        max_epochs,
        episode_id,
    } = spec;
    // Hard wall-clock deadline per episode: safety net against runaway extraction.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let episode_start = std::time::Instant::now();

    // 1. Convert seed expression to e-graph
    let mut egraph = EGraph::with_rules(all_rules());
    let root = egraph.add_arena(seed_arena, seed_root);

    // 2. Extract + JIT-benchmark the initial expression
    let (initial_arena, initial_root, _initial_cost) =
        extract_neural_to_arena(&egraph, root, model);

    if initial_arena.has_degenerate(initial_root) {
        log_exclusion(
            &episode_id,
            seed_name,
            "degenerate_seed",
            "seed expression contains NaN/Inf constants",
        );
        return None;
    }
    let initial_bench =
        match session.benchmark_arena(&initial_arena, initial_root, BenchMode::Latency) {
            Ok(b) => b,
            Err(e) => {
                log_exclusion(
                    &episode_id,
                    seed_name,
                    "initial_bench_failed",
                    &e.to_string(),
                );
                return None;
            }
        };

    // Domain check: if any SIMD lane is NaN or Inf at the test point, the
    // expression is undefined there. Rewrite equivalence checks would always
    // fire false positives (REWRITE BUG) because IEEE 754 NaN/Inf behavior
    // diverges across mathematically-equivalent forms. Skip early.
    if initial_bench.output.iter().any(|x| !x.is_finite()) {
        log_exclusion(
            &episode_id,
            seed_name,
            "initial_output_nonfinite",
            &format!("output at test point: {:?}", initial_bench.output),
        );
        return None;
    }

    // Sentinel-normalized: an episode's headline number is the RATIO of two
    // measurements taken at different points in the session, so uncorrected
    // drift between them shows up as a fake speedup or a fake regression.
    let initial_cost_ns = normalized_label_ns(&initial_bench, "episode initial cost");

    // 3. Randomize resource constraints per episode, scaled by expression size.
    // Larger expressions need more budget to explore meaningful rewrites.
    // Budget = base + multiplier * node_count, with random multiplier.
    let episode_hash: usize = episode_id
        .bytes()
        .fold(0usize, |h, b| h.wrapping_mul(31).wrapping_add(b as usize));
    let initial_nodes = initial_arena.node_count_subtree(initial_root);
    // multiplier in [3, 10]: small exprs get 50-200, large (100-node) get 300-1000
    let budget_mult = 3 + (episode_hash % 8);
    let node_budget = (50 + budget_mult * initial_nodes).min(2000); // floor 50, cap 2000
    let epoch_budget = 10 + (episode_hash.wrapping_mul(7) % 51); // [10, 60]
    let effective_epochs = max_epochs.min(epoch_budget);

    // 4. Saturate within budget. No rule filtering — every rule fires
    // whenever it matches (same rewrite semantics as production `saturate()`),
    // just capped by node/iteration budget so episodes stay bounded and diverse.
    let remaining = deadline.saturating_duration_since(std::time::Instant::now());
    let sat_result =
        saturate_with_full_budget(&mut egraph, effective_epochs, node_budget, remaining);

    // 5. Extract + JIT-benchmark the final expression
    let post_sat_nodes = egraph.node_count();
    if post_sat_nodes > 5000 {
        log_exclusion(
            &episode_id,
            seed_name,
            "egraph_too_large",
            &format!("{post_sat_nodes} nodes after saturation (budget={node_budget})"),
        );
        return None;
    }
    if std::time::Instant::now() > deadline {
        log_exclusion(
            &episode_id,
            seed_name,
            "deadline_before_extraction",
            &format!("egraph at {post_sat_nodes} nodes when the 5s episode deadline hit"),
        );
        return None;
    }
    let extractor = IncrementalExtractor::new(model, 8);
    let (_final_cost, final_choices) = extractor.extract_choices_only(&egraph, root);
    let (final_arena, final_arena_root) =
        pixelflow_search::egraph::choices_to_arena(&final_choices);

    if final_arena.has_degenerate(final_arena_root) {
        log_exclusion(
            &episode_id,
            seed_name,
            "degenerate_extraction",
            "extracted expression contains NaN/Inf constants",
        );
        return None;
    }
    let final_bench =
        match session.benchmark_arena(&final_arena, final_arena_root, BenchMode::Latency) {
            Ok(b) => b,
            Err(e) => {
                log_exclusion(&episode_id, seed_name, "final_bench_failed", &e.to_string());
                return None;
            }
        };
    let final_cost_ns = normalized_label_ns(&final_bench, "episode final cost");

    if episode_start.elapsed().as_millis() > 100 {
        eprintln!(
            "[SLOW_EPISODE] {episode_id} (seed={seed_name}) total={:.1}ms: \
             iterations={} unions={} saturated={} nodes={post_sat_nodes}",
            episode_start.elapsed().as_secs_f64() * 1000.0,
            sat_result.iterations,
            sat_result.total_unions,
            sat_result.saturated,
        );
    }

    // 6. Correctness check: rewrites must preserve semantics.
    //
    // The initial output was checked finite above; the final form needs no
    // separate finiteness check because check_equivalence treats any
    // non-finite lane pair that is not both-NaN as divergence (mirroring
    // Tolerance::Numeric) — an extraction that goes NaN where the seed was
    // finite lands in the REWRITE BUG branch below instead of minting a fake
    // speedup label through a NaN-blind `diff > max_diff` comparison.
    //
    // WHY return None instead of a label: an equivalence failure here means
    // saturation+extraction drove the expression into a semantically broken
    // form — exactly the outcome a policy must be penalized for, and dropping
    // it is the censored-failure pathology the July post-mortem killed the RL
    // loop over (audit M2). When the episode collector is built, these
    // episodes must become MAXIMAL-COST labels (not exclusions) so the policy
    // sees the failure region. Returning None is acceptable only while
    // episodes are unconsumed; the structured EPISODE_EXCLUDED line below is
    // what the collector's exclusion-rate alarm will count in the meantime.
    const EQUIV_EPSILON: f32 = 1e-3;
    if let Err(max_diff) = initial_bench.check_equivalence(&final_bench, EQUIV_EPSILON) {
        log_exclusion(
            &episode_id,
            seed_name,
            "equivalence_failed",
            &format!(
                "REWRITE BUG at x=0.5 y=0.7 z=1.3 w=-0.2: initial output {:?}, final output {:?}, \
                 max_diff={max_diff:.6} epsilon={EQUIV_EPSILON}; initial expr: {}; final expr: {}",
                initial_bench.output,
                final_bench.output,
                arena_to_kernel_code(&initial_arena, initial_root),
                arena_to_kernel_code(&final_arena, final_arena_root),
            ),
        );
        return None;
    }

    let speedup = if final_cost_ns > 0.0 {
        initial_cost_ns / final_cost_ns
    } else {
        0.0
    };
    eprintln!(
        "[REWRITE] {seed_name}: {speedup:.2}x ({initial_cost_ns:.1}ns -> {final_cost_ns:.1}ns)"
    );

    Some(Episode {
        episode_id,
        seed_name: seed_name.to_string(),
        initial_arena,
        initial_root,
        initial_cost_ns,
        final_arena,
        final_root: final_arena_root,
        final_cost_ns,
        node_budget,
        epoch_budget,
    })
}

// ============================================================================
// Corpus loading
// ============================================================================

/// The largest expression this loader will hand back, measured the way
/// `gen_bench_corpus --max-nodes` measures: [`ExprArena::node_count_subtree`]
/// from the root.
pub const MAX_CORPUS_NODES: usize = 1000;

/// Load expressions from a binary corpus tier.
///
/// Returns up to `max_count` `(name, arena, root)` triples, sampled uniformly
/// via LCG shuffle. Expressions larger than [`MAX_CORPUS_NODES`] are filtered
/// out.
///
/// # The size a filter must measure (bug B3)
///
/// This filter tested `arena.len()` — every node in the stored arena,
/// including ones unreachable from the root — while `gen_bench_corpus
/// --max-nodes` measures `node_count_subtree(root)`. Generator arenas are
/// append-only scratch space, so the two numbers differ by however much
/// rewriting the generator did, and the gap is *not* random: junkify passes
/// leave dead nodes behind, so the filter preferentially discarded exactly
/// the rewritten shapes. On the first end-to-end run it silently removed 110
/// of 380 DEV entries (29%) from held-out evaluation.
///
/// Corpus format v3 also stores only the reachable subtree
/// (`corpus::reachable_subtree`), which makes the two measures agree for
/// freshly written files — but this filter measures the subtree regardless,
/// because a filter that is only correct when its input happens to be
/// pre-compacted is a filter waiting to be wrong again.
///
/// # Panics
///
/// Panics if zero expressions load successfully from the file.
pub fn load_corpus_exprs(
    path: &Path,
    max_count: usize,
    seed: u64,
) -> Vec<(String, ExprArena, ExprId)> {
    let raw = corpus::read_corpus(path)
        .unwrap_or_else(|e| panic!("Failed to read binary corpus {}: {e}", path.display()));
    let total_entries = raw.len();

    let mut parsed: Vec<(String, ExprArena, ExprId)> = Vec::new();
    let mut skipped_large = 0u64;
    let mut dead_node_entries = 0u64;

    for (name, arena, root) in raw {
        let expression_nodes = arena.node_count_subtree(root);
        if arena.len() > expression_nodes {
            dead_node_entries += 1;
        }
        if expression_nodes > MAX_CORPUS_NODES {
            skipped_large += 1;
            continue;
        }
        parsed.push((name, arena, root));
    }

    assert!(
        !parsed.is_empty(),
        "Zero expressions loaded from {} ({} entries, {skipped_large} over {MAX_CORPUS_NODES} \
         reachable nodes)",
        path.display(),
        total_entries
    );

    // A v3 corpus is compacted at write time, so any entry whose arena is
    // larger than its expression came from a writer that did not compact.
    // Not fatal — the filter above measures the subtree either way — but the
    // discrepancy is the B3 signature and must never be invisible again.
    if dead_node_entries > 0 {
        eprintln!(
            "NOTE: {dead_node_entries}/{total_entries} entries in {} carry nodes unreachable \
             from their root. Sizes here are `node_count_subtree(root)` (what --max-nodes \
             measures), NOT `arena.len()`.",
            path.display()
        );
    }

    // LCG-based Fisher-Yates shuffle
    let mut state = seed;
    let len = parsed.len();
    for i in (1..len).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        parsed.swap(i, j);
    }

    // Truncate to max_count
    parsed.truncate(max_count);

    eprintln!(
        "Loaded {} corpus expressions from {} ({} entries, {skipped_large} over \
         {MAX_CORPUS_NODES} reachable nodes)",
        parsed.len(),
        path.display(),
        total_entries
    );

    parsed
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::OpKind;

    /// End-to-end smoke test: a tiny seed expression survives a full episode
    /// (extract -> JIT bench -> saturate under budget -> extract -> JIT bench
    /// -> equivalence check) without panicking, and returns sane costs.
    #[test]
    fn run_episode_smoke() {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let one = arena.push_const(1.0);
        let mul = arena.push_binary(OpKind::Mul, x, one); // x * 1.0 -> simplifies to x
        let root = arena.push_binary(OpKind::Add, mul, x); // (x * 1.0) + x

        let model = ExprNnue::new();
        let mut session = BenchSession::new();
        let episode = run_episode(
            &mut session,
            EpisodeSeed {
                arena: &arena,
                root,
                name: "smoke_seed",
            },
            &model,
            EpisodeSpec {
                max_epochs: 10,
                episode_id: "smoke_0".to_string(),
            },
        )
        .expect("episode should complete for a small, well-defined seed expression");

        // Both costs are `BenchResult::adjusted_ns` — raw time MINUS the
        // session's identity-kernel call overhead — rescaled by the sentinel's
        // (strictly positive) drift factor, which preserves sign. For a kernel
        // this small
        // (`(x * 1.0) + x` saturates to one op) the marginal cost over the
        // identity kernel is ~0 ± noise, so a NEGATIVE adjusted value is a
        // documented measurement condition, not a failure: see
        // `BenchResult::adjusted_ns` and `jit_bench`'s
        // `nonpositive_adjusted_is_recorded_not_fatal`, and note `run_episode`
        // itself branches on `final_cost_ns > 0.0` rather than assuming it.
        // Asserting `>= 0.0` here contradicted that contract and failed under
        // parallel-suite contention. Finiteness is the real invariant.
        assert!(
            episode.initial_cost_ns.is_finite(),
            "initial_cost_ns should be a finite measurement, got {}",
            episode.initial_cost_ns
        );
        assert!(
            episode.final_cost_ns.is_finite(),
            "final_cost_ns should be a finite measurement, got {}",
            episode.final_cost_ns
        );
        assert!(
            episode.node_budget >= 50,
            "node_budget should respect the floor"
        );
        assert!(
            episode.epoch_budget >= 10,
            "epoch_budget should respect the floor"
        );
    }

    // ── B3: the size filter must measure the expression, not the arena ──────

    fn scratch_corpus(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is before the unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "episodes_corpus_{tag}_{}_{nanos}.bin",
            std::process::id()
        ))
    }

    /// A small expression (`X * 2.0`, 3 nodes) buried in an arena carrying
    /// `dead` abandoned nodes — the shape a junkify pass leaves behind.
    fn small_expr_in_junky_arena(dead: usize) -> (ExprArena, ExprId) {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        for i in 0..dead {
            let c = arena.push_const(i as f32 + 100.0);
            let _abandoned = arena.push_binary(OpKind::Add, x, c);
        }
        let two = arena.push_const(2.0);
        let root = arena.push_binary(OpKind::Mul, x, two);
        (arena, root)
    }

    #[test]
    fn size_filter_keeps_small_expressions_from_junk_heavy_arenas() {
        // Straight from the smoke run: 110/380 DEV entries vanished because
        // the filter counted dead nodes. The corpus writer compacts now, but
        // the filter is pinned independently — a filter that is only correct
        // on pre-compacted input is one refactor from being wrong again.
        let (arena, root) = small_expr_in_junky_arena(2 * MAX_CORPUS_NODES);
        assert!(
            arena.len() > MAX_CORPUS_NODES,
            "fixture must exceed the limit by TOTAL nodes"
        );
        assert!(
            arena.node_count_subtree(root) <= MAX_CORPUS_NODES,
            "fixture's EXPRESSION must be small"
        );

        let path = scratch_corpus("junky");
        corpus::write_corpus(&path, &[("junky".to_string(), arena, root)]).expect("write");
        let loaded = load_corpus_exprs(&path, 10, 7);
        let _ = std::fs::remove_file(&path);

        assert_eq!(
            loaded.len(),
            1,
            "a 3-node expression must survive the size filter no matter how much dead \
             scratch its generator arena carried"
        );
        assert_eq!(loaded[0].1.node_count_subtree(loaded[0].2), 3);
    }

    #[test]
    fn size_filter_still_rejects_genuinely_large_expressions() {
        // The complement: the filter must not have become a no-op. A left
        // spine of MAX+2 reachable nodes is over the limit by the measure
        // that counts.
        let mut arena = ExprArena::new();
        let mut node = arena.push_var(0);
        while arena.node_count_subtree(node) <= MAX_CORPUS_NODES {
            let c = arena.push_const(1.0);
            node = arena.push_binary(OpKind::Add, node, c);
        }
        let big = node;
        let (small, small_root) = small_expr_in_junky_arena(0);

        let path = scratch_corpus("mixed");
        corpus::write_corpus(
            &path,
            &[
                ("too_big".to_string(), arena, big),
                ("ok".to_string(), small, small_root),
            ],
        )
        .expect("write");
        let loaded = load_corpus_exprs(&path, 10, 7);
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.len(), 1, "the oversized expression must be dropped");
        assert_eq!(loaded[0].0, "ok");
    }

    #[test]
    fn build_rule_templates_covers_all_rules() {
        let rules = all_rules();
        let templates = build_rule_templates(&rules);
        assert_eq!(
            templates.len(),
            rules.len(),
            "every rule should get a template entry"
        );
    }
}
