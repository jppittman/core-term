//! Budget-limited saturation with instrumentation.
//!
//! This module provides depth-limited saturation for training data generation.
//! The key insight from Stockfish-style training: we want NNUE to predict
//! "what's achievable within budget", not the theoretical optimum.
//!
//! # Usage
//!
//! ```ignore
//! let mut eg = EGraph::new();
//! let root = insert_tree(&mut eg, &expr_tree);
//! let result = saturate_with_budget(&mut eg, 100);
//!
//! // result contains stats about what happened during saturation
//! println!("Unions: {}, Saturated: {}", result.total_unions, result.saturated);
//! ```

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use super::candidate::{
    CandidateFeatures, CandidateKey, Firing, REGISTERED_PRIMARY_BUDGET_APPLICATIONS,
};
use super::graph::{AppBudgetSaturationStats, EGraph, RewriteTarget, SaturationStop};
use super::node::EClassId;
use crate::nnue::factored::EMBED_DIM;
use crate::nnue::guide::{CandidateSummary, SaturationGuide};

/// Result of a budget-limited saturation run.
///
/// This captures everything needed for training data generation:
/// - How much work was done (iterations, unions)
/// - Whether saturation completed or was cut off
/// - E-graph size before and after
#[derive(Clone, Debug)]
pub struct SaturationResult {
    /// Number of iterations completed.
    pub iterations: usize,

    /// Total unions performed across all iterations.
    pub total_unions: usize,

    /// Whether saturation completed (no more changes) before budget exhausted.
    pub saturated: bool,

    /// Number of e-classes before saturation.
    pub classes_before: usize,

    /// Number of e-classes after saturation.
    pub classes_after: usize,

    /// Rule match counts by rule name.
    pub rule_matches: HashMap<String, usize>,

    /// The rewrite budget that was used.
    pub budget: usize,

    /// Which condition ended the run, read from
    /// [`EGraph::saturate_with_limits`]'s own loop — the production caller
    /// (`crate::runtime::optimize_runtime_arena`) discards this, but an
    /// offline harness reproducing the production call must be able to
    /// report "quiesced / iteration cap / class cap / timeout" without
    /// inferring it from `saturated`.
    pub stop: SaturationStop,
}

impl SaturationResult {
    /// Calculate the improvement ratio (how much the e-graph grew).
    pub fn growth_ratio(&self) -> f64 {
        if self.classes_before == 0 {
            1.0
        } else {
            self.classes_after as f64 / self.classes_before as f64
        }
    }

    /// Whether the budget was exhausted (saturation was cut off).
    pub fn budget_exhausted(&self) -> bool {
        !self.saturated && self.iterations >= self.budget
    }
}

/// Run saturation with a budget limit, returning detailed statistics.
///
/// This is the teacher for Stockfish-style training: it runs full saturation
/// (up to the budget) and records what cost was achievable.
///
/// # Arguments
///
/// * `egraph` - The e-graph to saturate (mutated in place)
/// * `max_iterations` - Maximum number of saturation iterations (rewrite budget)
///
/// # Returns
///
/// A `SaturationResult` containing statistics about the saturation run.
///
/// # Example
///
/// ```ignore
/// let mut eg = EGraph::new();
/// let root = eg.add(ENode::Var(0));
/// let result = saturate_with_budget(&mut eg, 100);
/// assert!(result.saturated || result.iterations <= 100);
/// ```
pub fn saturate_with_budget(egraph: &mut EGraph, max_iterations: usize) -> SaturationResult {
    saturate_with_full_budget(
        egraph,
        max_iterations,
        10_000,
        std::time::Duration::from_secs(5),
    )
}

/// Run saturation with budget, class, and time limits.
///
/// Unlike `saturate_with_budget`, this gives full control over safety limits.
/// The e-graph stops growing when ANY limit is reached.
pub fn saturate_with_full_budget(
    egraph: &mut EGraph,
    max_iterations: usize,
    max_classes: usize,
    timeout: std::time::Duration,
) -> SaturationResult {
    let classes_before = egraph.classes.len();
    egraph.match_counts.clear();

    // One call drives the entire multi-round run — the loop that decides
    // when to stop (timeout, class limit, or convergence) lives exactly
    // once, in `EGraph::saturate_with_limits`
    // (docs/plans/2026-08-17-cost-model-domain.md, J11). This module
    // previously re-decided the same stopping conditions in a second,
    // hand-rolled outer loop that drove the e-graph one round at a time —
    // the duplicate-loop drift the domain model doc calls out by name.
    let stats = egraph.saturate_with_limits(max_iterations, max_classes, timeout);

    let saturated = stats.iterations < max_iterations || stats.total_unions == 0;
    let classes_after = egraph.classes.len();
    let rule_matches = egraph.match_counts.clone();

    SaturationResult {
        iterations: stats.iterations,
        total_unions: stats.total_unions,
        saturated,
        classes_before,
        classes_after,
        rule_matches,
        budget: max_iterations,
        stop: stats.stop,
    }
}

// ============================================================================
// Saturation budget presets — the compile-time policy shared by every caller
// ============================================================================

/// Safety limits for one saturation run, chess-clock style: several
/// independent limits, any one tripping ends the run. This is the policy
/// knob; [`saturate_with_full_budget`] is the mechanism both the AOT macro
/// tier (`pixelflow-compiler`) and the runtime tier
/// ([`crate::runtime::optimize_runtime_arena`]) drive with it.
#[derive(Clone, Copy, Debug)]
pub struct SaturationConfig {
    /// Rewrite-round budget (each round applies every rule once).
    pub max_iterations: usize,
    /// Wall-clock budget for the whole run.
    pub hard_timeout: std::time::Duration,
    /// E-class count budget — caps pathological e-graph blowup.
    pub max_classes: usize,
}

impl SaturationConfig {
    /// Trivial expressions (≤10 nodes): minimal budget.
    pub fn blitz() -> Self {
        Self {
            max_iterations: 20,
            hard_timeout: std::time::Duration::from_millis(10),
            max_classes: 500,
        }
    }

    /// Normal complexity (11-50 nodes): balanced.
    pub fn rapid() -> Self {
        Self {
            max_iterations: 50,
            hard_timeout: std::time::Duration::from_millis(50),
            max_classes: 2000,
        }
    }

    /// Complex expressions (51+ nodes): thorough search.
    pub fn classical() -> Self {
        Self {
            max_iterations: 100,
            hard_timeout: std::time::Duration::from_millis(200),
            max_classes: 5000,
        }
    }
}

/// Pick a [`SaturationConfig`] preset from a rough expression-size measure.
///
/// | Nodes | Config | Rationale |
/// |-------|--------|-----------|
/// | 0-10 | blitz | Trivial expressions need minimal optimization |
/// | 11-50 | rapid | Normal complexity, balanced approach |
/// | 51+ | classical | Complex expressions need thorough search |
///
/// `node_count` is a proxy, not a precise measure — AST node count for the
/// macro tier, reachable-arena-node count for the runtime tier both serve
/// equally well as "how big is this expression".
pub fn config_for_node_count(node_count: usize) -> SaturationConfig {
    match node_count {
        0..=10 => SaturationConfig::blitz(),
        11..=50 => SaturationConfig::rapid(),
        _ => SaturationConfig::classical(),
    }
}

/// Configuration for multi-budget training data generation.
///
/// Generate training data at multiple budget levels for curriculum learning.
#[derive(Clone, Debug)]
pub struct MultiBudgetConfig {
    /// Budget levels to generate data at (e.g., [50, 100, 200, 500]).
    pub budgets: Vec<usize>,

    /// Number of samples to generate at each budget level.
    pub samples_per_budget: usize,
}

impl Default for MultiBudgetConfig {
    fn default() -> Self {
        Self {
            budgets: vec![50, 100, 200, 500],
            samples_per_budget: 2500,
        }
    }
}

/// Extract the best achievable cost within budget.
///
/// This is the ground truth label for training: given an expression,
/// what's the lowest cost we can achieve with `budget` rewrite iterations?
pub fn achievable_cost_within_budget(
    egraph: &mut EGraph,
    root: EClassId,
    budget: usize,
    costs: &super::cost::CostModel,
) -> (usize, SaturationResult) {
    // Run budget-limited saturation
    let result = saturate_with_budget(egraph, budget);

    // Extract best cost
    let (_arena, _arena_root, cost) = egraph.extract_best(root, costs);

    (cost, result)
}

// ============================================================================
// Guided saturation (docs/plans/2026-08-31-guide-design-revision.md §4/§5)
// ============================================================================

/// Guided-saturation ordering loop: score every surviving (post-dedup)
/// candidate rewrite with `guide`, apply them in descending-score order, and
/// repeat until a budget/quiescence/cap/timeout condition stops the run —
/// mirroring [`EGraph::saturate_until_applications`]'s stop-reason contract
/// exactly (same [`AppBudgetSaturationStats`] return shape), so a caller can
/// compare guided and unguided runs directly.
///
/// # What this changes relative to unguided saturation
///
/// Unguided saturation (`saturate_until_applications`) applies every rule
/// against every matching node it finds, in a fixed rule-then-class scan
/// order, once per sweep — with no notion of "have I already resolved this
/// exact (rule, class content) pair". This function instead, once per round:
///
/// 1. Enumerates every current match ([`EGraph::find_rewrite_matches`]).
/// 2. Builds each match's [`CandidateKey`] (rule identity + canonical class
///    content, via [`CandidateFeatures::observe`]) and skips any key already
///    seen in an EARLIER round of this same call, without ever handing it to
///    `guide` — the "dedup before scoring" step design doc §2.2/§4 measured
///    as the dominant per-round cost (90.4% of raw matches are exactly this:
///    an idempotent re-fire of an already-installed rewrite, from rules like
///    `commutative`/`associative` that have no "already applied" check).
/// 3. Scores the survivors in ONE batched [`SaturationGuide::score_candidates`]
///    call (binding rule: batch per iteration, no incrementality work).
/// 4. Applies them in descending-score order via
///    [`EGraph::apply_single_rule`], checking the application budget, the
///    wall-clock deadline, and the class cap after every single application
///    — finer-grained than the unguided per-sweep check, since one guided
///    round can apply many more candidates before its next full rescan.
///
/// A round in which every match is already a seen key (survivors empty, but
/// the raw match set was not) is also treated as quiescence: without this,
/// guided mode would loop, re-enumerating the same fully-deduped match set,
/// until `max_iters` or `timeout` — a `SaturationStop` that misreports "ran
/// out of budget/time" for a graph that had, in truth, nothing left to try.
///
/// # Approximation, stated plainly (mirrors `egraph::candidate`'s own note)
///
/// A match enumerated at the start of a round can be invalidated mid-round:
/// applying an earlier, higher-scored candidate may union or rebuild the
/// e-class a later candidate matched against. [`RewriteTarget`] therefore
/// names the matched node by its stable [`super::provenance::ENodeId`], not
/// by a position in the class's node vector, and `apply_single_rule`
/// re-resolves that identity before re-checking the rule — so a
/// mid-round rebuild either leaves the exact matched node in place (the rule
/// is re-checked against it, as before) or removes it, in which case nothing
/// fires and nothing is recorded. It can never apply a *different* node that
/// happened to inherit the old index while the stale candidate's score and
/// dedup key were credited to it. The missed opportunity, if any, is picked
/// up on the next round's rescan. `budget_fraction`'s `application_ordinal` is
/// similarly a per-round approximation (the cumulative application count at
/// the START of scoring, shared by every candidate scored that round, not
/// each candidate's own eventual firing order) — exact per-candidate ordinals
/// would require scoring one candidate at a time, which the batch-per-
/// iteration binding rule above rules out for v1.
///
/// # Unguided mode is untouched
///
/// This function is purely additive: it calls only [`EGraph`]'s existing
/// public API (`find_rewrite_matches`, `apply_single_rule`, `provenance`,
/// `num_classes`) and does not modify `saturate_until_applications`, `batch`,
/// or any other unguided path. See `tests::guided::unguided_mode_is_bit_
/// identical_to_before_this_change` for the determinism regression this
/// claim is pinned by.
///
/// # Panics
///
/// - If `guide.score_candidates` returns a different number of scores than
///   candidates given, or any non-finite score (NO SILENT FAILURES: a guide
///   bug must not silently corrupt move ordering).
/// - If a match fires a `rule_idx` for which `rule_embeds` has no entry —
///   the caller is responsible for supplying one embedding per registered
///   rule.
///
/// # `budget_fraction`'s denominator is fixed, not `max_total_applications`
///
/// Every [`Firing`] built here uses
/// [`REGISTERED_PRIMARY_BUDGET_APPLICATIONS`] as `registered_budget`, never
/// this call's own `max_total_applications` argument. A Guide is trained
/// once against a fixed feature space (`gen_strict_labels` mints
/// `budget_fraction` against the same constant); calling this function with
/// a different `max_total_applications` — e.g. to evaluate the registered
/// secondary tier, B=200 — must change how far a curve runs, never what
/// `budget_fraction` *means* to an already-trained Guide. `max_total_
/// applications` remains exactly the stopping condition (unchanged); only
/// the feature denominator is decoupled from it.
pub fn saturate_guided_until_applications<G: SaturationGuide>(
    egraph: &mut EGraph,
    guide: &G,
    rule_embeds: &[[f32; EMBED_DIM]],
    max_total_applications: usize,
    max_iters: usize,
    max_classes: usize,
    timeout: Duration,
) -> AppBudgetSaturationStats {
    GuidedSaturation::new(guide, rule_embeds).until_applications(
        egraph,
        max_total_applications,
        max_iters,
        max_classes,
        timeout,
    )
}

/// One guided-saturation *episode*: the Guide, its per-rule embeddings, and
/// the candidate-key dedup set, kept together so a caller that advances the
/// same e-graph through several successive application budgets (an anytime
/// curve sampled at 25, 50, 100, ... applications —
/// [`super::anytime::run_anytime_curve_with`]) sees ONE continuous guided
/// run, not a fresh episode per checkpoint.
///
/// Why this must be a struct and not a per-call local: with the dedup set
/// re-created on every call, the second checkpoint's first round would
/// re-score and re-fire every already-resolved `(rule, class content)` key
/// from the first — each an idempotent re-fire that still *records* an
/// application, i.e. spends the budget on exactly the 90% no-op traffic
/// dedup exists to eliminate. Unguided saturation has no per-call state, so
/// its curve was never exposed to this; the guided curve would have been
/// silently handicapped at every checkpoint after the first.
///
/// `expr_node_count` (the episode-level feature constant, see
/// [`CandidateSummary::new`]) is captured on the FIRST `until_applications`
/// call — the graph's node count before any guided round has fired — and
/// reused by later calls, matching the single-call semantics exactly.
pub struct GuidedSaturation<'a, G: SaturationGuide> {
    guide: &'a G,
    rule_embeds: &'a [[f32; EMBED_DIM]],
    seen_keys: HashSet<CandidateKey>,
    expr_node_count: Option<usize>,
}

impl<'a, G: SaturationGuide> GuidedSaturation<'a, G> {
    /// Start an episode: empty dedup set, feature constant not yet captured.
    #[must_use]
    pub fn new(guide: &'a G, rule_embeds: &'a [[f32; EMBED_DIM]]) -> Self {
        Self {
            guide,
            rule_embeds,
            seen_keys: HashSet::new(),
            expr_node_count: None,
        }
    }

    /// Number of distinct candidate keys this episode has scored so far
    /// (diagnostic: dedup coverage).
    #[must_use]
    pub fn seen_key_count(&self) -> usize {
        self.seen_keys.len()
    }

    /// Advance `egraph` until the cumulative recorded application count
    /// reaches `max_total_applications` (or quiescence / `max_iters` rounds
    /// for THIS call / class cap / timeout) — the body documented on
    /// [`saturate_guided_until_applications`], with the dedup set and
    /// feature constant carried across calls.
    pub fn until_applications(
        &mut self,
        egraph: &mut EGraph,
        max_total_applications: usize,
        max_iters: usize,
        max_classes: usize,
        timeout: Duration,
    ) -> AppBudgetSaturationStats {
        let guide = self.guide;
        let rule_embeds = self.rule_embeds;
        let seen_keys = &mut self.seen_keys;
        let start = Instant::now();
        let mut iterations = 0usize;
        let mut total_unions = 0usize;
        // Episode-level constant for `CandidateSummary::new`'s `expr_node_count`
        // (see that constructor's doc): the graph's node count at the moment
        // the episode started, before any guided round has fired — the
        // live-loop analogue of the offline label-minting replay's
        // `arena.nodes_raw().len()` snapshot taken before
        // `saturate_with_limits` runs.
        let expr_node_count = *self
            .expr_node_count
            .get_or_insert_with(|| egraph.node_count());

        let stop = 'outer: loop {
            if egraph.provenance().application_count() >= max_total_applications {
                break SaturationStop::ApplicationBudget;
            }
            if iterations >= max_iters {
                break SaturationStop::IterationCeiling;
            }
            if start.elapsed() >= timeout {
                break SaturationStop::Timeout;
            }
            if egraph.num_classes() > max_classes {
                break SaturationStop::ClassCap;
            }
            iterations += 1;

            // `find_rewrite_matches` scans every rule against every node of
            // every class and is NOT deadline-aware, so on a large graph it
            // can consume the whole remaining budget by itself. Declaring
            // quiescence off an empty result without re-reading the clock
            // would hand `run_anytime_curve_with` a `Quiesced` it accepts as
            // a completed run when the safety ceiling had in fact expired
            // mid-scan (that runner only rejects `Timeout`).
            let matches = egraph.find_rewrite_matches();
            if matches.is_empty() {
                break if start.elapsed() >= timeout {
                    SaturationStop::Timeout
                } else {
                    SaturationStop::Quiesced
                };
            }

            // Dedup BEFORE scoring (§2.2/§4): the ordinal recorded is a
            // per-round approximation (see doc above), used only for the
            // budget_fraction feature, never for correctness.
            //
            // A key becomes "seen" only once an application has actually
            // been RECORDED for it (below), not here at enumeration time.
            // Marking at enumeration burned every scored survivor the
            // round never got to — the ones after a mid-round budget /
            // deadline / class-cap stop, and the ones whose matched node an
            // earlier application in the same round removed so
            // `apply_single_rule` recorded nothing — and the next round's
            // rescan then deduped them away for good. Measured on
            // 2026-09-01 (DEV classical, 334 expressions): a median 39–44%
            // of scored keys were burned that way, and sampling a guided
            // anytime curve at 25/50 before 100 cost the guided arm a
            // median ~50% of its applications-to-quiescence versus sampling
            // at 100 directly — the curve definition was silently
            // handicapping exactly the arm it exists to measure.
            // `round_keys` groups this round's match list by key (two nodes
            // of one class matching the same rule share a score but stay
            // two separate applications).
            let ordinal = egraph.provenance().application_count() as u64;
            let mut survivors: Vec<(Vec<RewriteTarget>, CandidateFeatures)> = Vec::new();
            let mut round_keys: HashMap<CandidateKey, usize> = HashMap::new();
            for target in matches {
                let firing = Firing {
                    rule_idx: target.rule_idx,
                    match_root: target.class_id,
                    application_ordinal: ordinal,
                    registered_budget: REGISTERED_PRIMARY_BUDGET_APPLICATIONS,
                };
                let features = CandidateFeatures::observe(egraph, &firing);
                if seen_keys.contains(&features.key) {
                    continue;
                }
                // One SCORED candidate per key — two nodes of one class
                // matching the same rule are, by the key's construction,
                // feature-identical and would receive the same score, so
                // scoring both would be wasted work. But they are two
                // distinct ACTIONS: dropping the second and then burning the
                // key once the first fires (an idempotent first firing keeps
                // the class content, hence the key, unchanged) would skip it
                // permanently and let the loop report quiescence with a real
                // rewrite unexplored. So every distinct matched node is kept
                // and applied under its key's single score.
                match round_keys.get(&features.key) {
                    Some(&i) => survivors[i].0.push(target),
                    None => {
                        round_keys.insert(features.key.clone(), survivors.len());
                        survivors.push((vec![target], features));
                    }
                }
            }

            if survivors.is_empty() {
                break if start.elapsed() >= timeout {
                    SaturationStop::Timeout
                } else {
                    SaturationStop::Quiesced
                };
            }

            // Batch-score (never one candidate at a time — binding rule).
            let summaries: Vec<CandidateSummary> = survivors
                .iter()
                .map(|(targets, features)| {
                    let rule_idx = targets[0].rule_idx;
                    let rule_embed = *rule_embeds.get(rule_idx).unwrap_or_else(|| {
                        panic!(
                            "saturate_guided_until_applications: rule_embeds has {} entries, \
                         but a match fired rule_idx {} — the caller must supply one \
                         embedding per registered rule",
                            rule_embeds.len(),
                            rule_idx
                        )
                    });
                    CandidateSummary::new(features, rule_embed, expr_node_count)
                })
                .collect();
            let scores = guide.score_candidates(&summaries);
            assert_eq!(
                scores.len(),
                survivors.len(),
                "SaturationGuide::score_candidates must return exactly one score per \
             candidate — got {} scores for {} candidates",
                scores.len(),
                survivors.len()
            );
            for &s in &scores {
                assert!(
                    s.is_finite(),
                    "SaturationGuide produced a non-finite score ({s}) — fail loud rather \
                 than let NaN/inf silently corrupt move ordering"
                );
            }

            // Order descending by score; ties keep match-enumeration order
            // (`sort_by` is stable) for determinism given a fixed guide.
            let mut order: Vec<usize> = (0..survivors.len()).collect();
            order.sort_by(|&a, &b| {
                scores[b]
                    .partial_cmp(&scores[a])
                    .expect("scores were just asserted finite above")
            });

            // Apply in that order, budget/deadline/cap-checked per application —
            // see the doc's "Approximation, stated plainly" section for why a
            // stale `node_idx` mid-round is safe rather than silently wrong.
            for idx in order {
                if egraph.provenance().application_count() >= max_total_applications {
                    break 'outer SaturationStop::ApplicationBudget;
                }
                if start.elapsed() >= timeout {
                    break 'outer SaturationStop::Timeout;
                }
                let (targets, features) = &survivors[idx];
                let before = egraph.provenance().application_count();
                for target in targets {
                    if egraph.provenance().application_count() >= max_total_applications {
                        break;
                    }
                    if egraph.apply_single_rule(target.rule_idx, target.class_id, target.tag) {
                        total_unions += 1;
                    }
                }
                // Recorded (whether or not it changed anything): this key
                // is resolved. Not recorded (no matched node survived, so
                // nothing fired): leave it unseen so the next rescan can
                // retry it against the rebuilt graph — see the dedup
                // comment above.
                if egraph.provenance().application_count() > before {
                    seen_keys.insert(features.key.clone());
                }
                if egraph.num_classes() > max_classes {
                    break 'outer SaturationStop::ClassCap;
                }
            }
        };

        AppBudgetSaturationStats {
            iterations,
            total_unions,
            applications: egraph.provenance().application_count(),
            stop,
        }
    }
}

impl<G: SaturationGuide> super::anytime::AnytimeStepper for GuidedSaturation<'_, G> {
    fn advance(
        &mut self,
        egraph: &mut EGraph,
        step: super::anytime::AnytimeStep,
    ) -> AppBudgetSaturationStats {
        self.until_applications(
            egraph,
            step.app_target,
            step.sweeps_left,
            step.max_classes,
            step.remaining,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::{CostModel, ENode, Rewrite, ops};
    use crate::math::algebra::{
        AddNeg, Annihilator, Cancellation, Canonicalize, Commutative, Identity,
        InverseAnnihilation, Involution, MulRecip,
    };

    /// Create an e-graph with standard algebraic rules for testing.
    fn egraph_with_rules() -> EGraph {
        let rules: Vec<Box<dyn Rewrite>> = vec![
            Canonicalize::<AddNeg>::new(),
            Involution::<AddNeg>::new(),
            Cancellation::<AddNeg>::new(),
            InverseAnnihilation::<AddNeg>::new(),
            Canonicalize::<MulRecip>::new(),
            Involution::<MulRecip>::new(),
            Cancellation::<MulRecip>::new(),
            InverseAnnihilation::<MulRecip>::new(),
            Commutative::new(&ops::Add),
            Commutative::new(&ops::Mul),
            Identity::new(&ops::Add),
            Identity::new(&ops::Mul),
            Annihilator::new(&ops::Mul),
        ];
        EGraph::with_rules(rules)
    }

    #[test]
    fn saturate_with_budget_simple() {
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let zero = eg.add(ENode::constant(0.0));
        let _sum = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, zero],
        });

        let result = saturate_with_budget(&mut eg, 10);

        // Should saturate quickly for simple expression
        assert!(result.iterations <= 10);
        assert!(result.classes_after >= result.classes_before);
    }

    #[test]
    fn saturate_with_budget_exhausted() {
        let mut eg = egraph_with_rules();
        // Create a moderately complex expression
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        let mul = eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![x, y],
        });
        let add = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![mul, x],
        });
        let _sub = eg.add(ENode::Op {
            op: &ops::Sub,
            children: vec![add, y],
        });

        // Very small budget - may not saturate
        let result = saturate_with_budget(&mut eg, 1);

        assert_eq!(result.budget, 1);
        assert!(result.iterations <= 1);
    }

    #[test]
    fn achievable_cost() {
        let mut eg = egraph_with_rules();
        let x = eg.add(ENode::Var(0));
        let zero = eg.add(ENode::constant(0.0));
        let sum = eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, zero],
        });

        let costs = CostModel::new();
        let (cost, result) = achievable_cost_within_budget(&mut eg, sum, 10, &costs);

        // x + 0 should simplify to x (cost 0)
        assert_eq!(cost, 0);
        assert!(result.saturated);
    }

    #[test]
    fn saturation_result_growth_ratio() {
        let result = SaturationResult {
            iterations: 5,
            total_unions: 10,
            saturated: true,
            classes_before: 10,
            classes_after: 15,
            rule_matches: HashMap::new(),
            budget: 100,
            stop: SaturationStop::Quiesced,
        };

        assert!((result.growth_ratio() - 1.5).abs() < 0.01);
    }
}

// ============================================================================
// Guided-saturation tests (docs/plans/2026-08-31-guide-design-revision.md
// §4/§5 — task 2/4 of the phase3-guide wiring round)
// ============================================================================

#[cfg(test)]
mod guided_tests {
    use super::*;
    use crate::egraph::{ENode, Rewrite, ops};
    use crate::math::algebra::Commutative;
    use crate::nnue::guide::{CandidateSummary, SaturationGuide};

    /// A one-rule e-graph whose single match will grow the matched class's
    /// own content on the very first firing (`x+y` gains a `y+x` sibling) —
    /// exactly the idempotent-refire shape §2.2 measured: `find_rewrite_
    /// matches` keeps reporting a non-empty match set forever (both `x+y`
    /// and `y+x` always match `Commutative`), so nothing but candidate-key
    /// dedup can ever stop this loop short of `max_iters`/`timeout`.
    fn commutative_only_egraph() -> EGraph {
        let rules: Vec<Box<dyn Rewrite>> = vec![Commutative::new(&ops::Add)];
        let mut eg = EGraph::with_rules(rules);
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, y],
        });
        eg
    }

    /// Records how many candidates each `score_candidates` call received;
    /// scores everything `0.0` (ordering is not under test here).
    struct SpyGuide {
        calls: std::cell::RefCell<Vec<usize>>,
    }

    impl SaturationGuide for SpyGuide {
        fn score_candidates(&self, candidates: &[CandidateSummary]) -> Vec<f32> {
            self.calls.borrow_mut().push(candidates.len());
            vec![0.0; candidates.len()]
        }
    }

    /// Two nodes of ONE e-class matching the same rule share a
    /// [`CandidateKey`] — they are feature-identical, so the guide is asked
    /// to score them once. They are still two distinct actions, and both
    /// must fire: keeping only the first and then burning the key (an
    /// idempotent first firing leaves the class content, hence the key,
    /// unchanged) skipped the second permanently and let the loop report
    /// quiescence with a real rewrite unexplored.
    #[test]
    fn every_node_level_match_sharing_one_key_is_applied_not_just_the_first() {
        let mut eg = commutative_only_egraph();

        // Fire commutative once so the sum's class holds BOTH Add(x, y) and
        // Add(y, x) — two nodes, one class, both matching `commutative`.
        let target = eg
            .find_rewrite_matches()
            .into_iter()
            .next()
            .expect("the seeded x + y has a commutative match");
        assert!(eg.apply_single_rule(target.rule_idx, target.class_id, target.tag));
        let class = eg.find(target.class_id);
        assert_eq!(
            eg.nodes(class).len(),
            2,
            "setup: the class should hold both orderings"
        );
        let matches: Vec<_> = eg
            .find_rewrite_matches()
            .into_iter()
            .filter(|t| eg.find(t.class_id) == class)
            .collect();
        assert_eq!(
            matches.len(),
            2,
            "setup: both nodes of the class match commutative"
        );

        let before = eg.provenance().application_count();
        let rule_embeds = [[0.0f32; EMBED_DIM]];
        let guide = SpyGuide {
            calls: std::cell::RefCell::new(Vec::new()),
        };
        let stats = saturate_guided_until_applications(
            &mut eg,
            &guide,
            &rule_embeds,
            before + 2,
            1,
            2_000,
            Duration::from_secs(5),
        );

        assert_eq!(
            guide.calls.into_inner(),
            vec![1],
            "the two matches are one candidate KEY, so exactly one score is asked for"
        );
        assert_eq!(
            stats.applications - before,
            2,
            "both matched nodes must fire under that single score, not just the first"
        );
    }

    #[test]
    fn dedup_should_keep_the_guide_from_ever_rescoring_an_already_seen_candidate_key() {
        let mut eg = commutative_only_egraph();
        let rule_embeds = [[0.0f32; EMBED_DIM]];
        let guide = SpyGuide {
            calls: std::cell::RefCell::new(Vec::new()),
        };

        // A generous budget/iteration/time ceiling: if dedup were broken (or
        // absent), `find_rewrite_matches` reporting a permanently non-empty
        // match set would drive this loop to `max_iters` scoring the same
        // candidate every round, not to quiescence in a handful of rounds.
        let stats = saturate_guided_until_applications(
            &mut eg,
            &guide,
            &rule_embeds,
            1_000,
            50,
            2_000,
            Duration::from_secs(5),
        );

        assert_eq!(
            stats.stop,
            SaturationStop::Quiesced,
            "a single-rule commutative e-graph has nothing left to try once its one \
             candidate key has fired and settled — dedup-driven quiescence, not an \
             exhausted iteration/time budget: {stats:?}"
        );
        assert!(
            stats.iterations <= 3,
            "expected quiescence within a couple of rounds once every candidate key \
             has been seen, got {} iterations",
            stats.iterations
        );

        let calls = guide.calls.into_inner();
        assert!(
            !calls.is_empty(),
            "the guide should have been asked to score at least once"
        );
        let total_scored: usize = calls.iter().sum();
        assert!(
            total_scored <= 2,
            "dedup must keep the total number of scored candidates tiny even though \
             find_rewrite_matches keeps finding a non-empty (but already-resolved) \
             match set every round forever without it — calls were {calls:?}"
        );
    }

    /// Two independent rules, each with exactly one match available in the
    /// same round, scored by a guide that reads the rule embedding directly
    /// (so the test controls which candidate scores higher without needing
    /// the real NNUE encoding).
    fn two_rule_egraph() -> EGraph {
        let rules: Vec<Box<dyn Rewrite>> =
            vec![Commutative::new(&ops::Add), Commutative::new(&ops::Mul)];
        let mut eg = EGraph::with_rules(rules);
        let x = eg.add(ENode::Var(0));
        let y = eg.add(ENode::Var(1));
        eg.add(ENode::Op {
            op: &ops::Add,
            children: vec![x, y],
        });
        let a = eg.add(ENode::Var(2));
        let b = eg.add(ENode::Var(3));
        eg.add(ENode::Op {
            op: &ops::Mul,
            children: vec![a, b],
        });
        eg
    }

    struct ScoreByRuleEmbedFirstLane;

    impl SaturationGuide for ScoreByRuleEmbedFirstLane {
        fn score_candidates(&self, candidates: &[CandidateSummary]) -> Vec<f32> {
            candidates.iter().map(|c| c.rule_embed[0]).collect()
        }
    }

    #[test]
    fn a_budget_stop_mid_round_must_not_burn_the_candidates_the_round_never_reached() {
        // Two independent candidates, one round, budget 1: the round scores
        // both, applies the higher-scored one, and hits the budget. The
        // lower-scored candidate was scored but never applied — it must
        // still fire when the SAME episode is resumed with a larger budget
        // (an anytime curve sampling at 1 then at 1000), not be deduped
        // away as "already seen". Before 2026-09-01 the second call
        // reported Quiesced with a single application.
        let mut eg = two_rule_egraph();
        let rule_embeds = [[1.0f32; EMBED_DIM], [10.0f32; EMBED_DIM]];
        let mut episode = GuidedSaturation::new(&ScoreByRuleEmbedFirstLane, &rule_embeds);

        let first = episode.until_applications(&mut eg, 1, 50, 2_000, Duration::from_secs(5));
        assert_eq!(first.stop, SaturationStop::ApplicationBudget);
        assert_eq!(first.applications, 1);

        let second = episode.until_applications(&mut eg, 1_000, 50, 2_000, Duration::from_secs(5));
        assert_eq!(second.stop, SaturationStop::Quiesced);
        let mut fired: Vec<usize> = eg
            .provenance()
            .applications()
            .map(|(_, r)| r.rule_idx)
            .collect();
        fired.sort_unstable();
        fired.dedup();
        assert_eq!(
            fired,
            vec![0, 1],
            "the candidate cut off by the budget stop must fire on resume, not be burned"
        );
    }

    #[test]
    fn ordering_should_apply_the_higher_scored_candidate_before_the_lower_scored_one() {
        let mut eg = two_rule_egraph();
        // rule 0 (Add) scores low, rule 1 (Mul) scores high.
        let rule_embeds = [[1.0f32; EMBED_DIM], [10.0f32; EMBED_DIM]];

        let stats = saturate_guided_until_applications(
            &mut eg,
            &ScoreByRuleEmbedFirstLane,
            &rule_embeds,
            1_000,
            50,
            2_000,
            Duration::from_secs(5),
        );
        assert!(
            stats.applications >= 2,
            "both the Add and Mul candidates should have fired: {stats:?}"
        );

        let mut records: Vec<_> = eg.provenance().applications().collect();
        records.sort_by_key(|(id, _)| id.as_u64());
        assert!(
            records.len() >= 2,
            "expected at least 2 recorded applications, got {}",
            records.len()
        );
        assert_eq!(
            records[0].1.rule_idx, 1,
            "the higher-scored (Mul, rule_idx=1) candidate must be applied first"
        );
        assert_eq!(
            records[1].1.rule_idx, 0,
            "the lower-scored (Add, rule_idx=0) candidate must be applied second"
        );
    }

    /// `saturate_guided_until_applications` is purely additive: it calls only
    /// `EGraph`'s existing public API and does not touch
    /// `saturate_until_applications` or any other unguided path. This test
    /// pins that the unguided path stays deterministic (same cost, same
    /// extracted-arena size) across two independent runs over identical
    /// input — the regression this change must not disturb.
    #[test]
    fn unguided_saturate_until_applications_stays_deterministic() {
        use crate::egraph::CostModel;
        use crate::math::all_rules;
        use pixelflow_ir::{ExprArena, ExprId, OpKind};

        fn build() -> (ExprArena, ExprId) {
            let mut a = ExprArena::new();
            let x = a.push_var(0);
            let y = a.push_var(1);
            let s = a.push_binary(OpKind::Add, x, y);
            let s2 = a.push_binary(OpKind::Mul, s, s);
            let two = a.push_const(2.0);
            let ts = a.push_binary(OpKind::Mul, two, s);
            let out = a.push_binary(OpKind::Add, s2, ts);
            (a, out)
        }

        fn digest() -> (usize, usize, SaturationStop) {
            let (arena, root) = build();
            let mut eg = EGraph::with_rules(all_rules());
            let root_class = eg.add_arena(&arena, root);
            let stats = eg.saturate_until_applications(2_000, 200, 5_000, Duration::from_secs(30));
            let costs = CostModel::latency_prior();
            let (out_arena, _out_root, cost) = eg.extract_best(root_class, &costs);
            (cost, out_arena.len(), stats.stop)
        }

        let (cost1, len1, stop1) = digest();
        let (cost2, len2, stop2) = digest();
        assert_eq!(
            cost1, cost2,
            "unguided extraction cost must be deterministic"
        );
        assert_eq!(
            len1, len2,
            "unguided extracted-arena size must be deterministic"
        );
        assert_eq!(stop1, stop2, "unguided stop reason must be deterministic");
    }
}
