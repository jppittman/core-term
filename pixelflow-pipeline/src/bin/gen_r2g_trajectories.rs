//! Trajectory minting for the Guide's return-to-go objective
//! (`docs/plans/2026-09-01-guide-return-to-go.md` §2/§5): for every corpus
//! expression, run `K` differently-ordered saturation trajectories to a
//! generous ceiling budget, and emit one JSONL record per recorded
//! rewrite-rule application carrying its trajectory's hindsight return at
//! the two registered checkpoints (B=100 primary, B=200 secondary) relative
//! to the expression's best cost seen across every trajectory.
//!
//! # Ordering diversity ([`OrderingPolicy`], §2)
//!
//! Twelve trajectories per expression: `unguided` (the production rule-major
//! sweep — see the equivalence test below), `per-rule`
//! ([`PerRuleRateGuide`], the zero-candidate-local-information control),
//! `strict-v1` ([`LinearCandidateGuide`] over the FROZEN Round-1 checkpoint),
//! six `random:<seed>` ([`UniformRandomGuide`]), and three rank-mixes of
//! `strict-v1` with a uniform term (`mix:strict-v1:1/4:<seed>` ×2,
//! `mix:strict-v1:1/2:<seed>` ×1) — §2's exact table, reused here rather
//! than re-derived (K and the seed count are justified there from
//! per-expression application volume and mint cost, not re-argued here).
//!
//! # Checkpoints: B=100 / B=200, ceiling 4×B=400
//!
//! Every trajectory runs to a ceiling of `4 * REGISTERED_PRIMARY_BUDGET_
//! APPLICATIONS = 400` recorded applications (or ends earlier), with
//! extraction snapshots taken at the two REGISTERED tiers this program's
//! `train_guide_r2g`/[`R2gRecord`] consumer already expects
//! (`cost_b100`/`cost_b200`, §7's pre-registered comparison) plus a third,
//! unreported extraction at the ceiling itself — used only to (a) label
//! `label_positive` against the fullest available derivation and (b) widen
//! the empirical per-expression best-cost reference beyond the two reported
//! checkpoints (§1.2: "the unguided trajectory contributes its full grid
//! through quiescence/cap"). Running to a ceiling past the registered
//! secondary tier, while still labeling at the two registered checkpoints,
//! is a deliberate reconciliation: the task instruction driving this binary
//! asked for trajectories run "to budget 4B, so B and 4B returns both
//! exist," but every consumer already landed in this tree
//! (`training/r2g.rs`'s [`R2gRecord`], `train_guide_r2g`,
//! `phase3_at_budget_eval`'s planned `--r2g-checkpoint` arm) is built
//! against the plan's actual pre-registered tiers, B=100/B=200 (§7) — not
//! 4B=400. Matching the landed schema rather than inventing a third,
//! incompatible one is the smaller, more defensible surface; the ceiling
//! still runs to 400, satisfying "trajectories to budget 4B" as the run
//! length, while `cost_b100`/`cost_b200` satisfy every downstream reader.
//!
//! # Feature observation is post-hoc, not in-loop (a stated approximation)
//!
//! [`CandidateFeatures::observe`] is called once per trajectory, after that
//! trajectory has run to its ceiling, against the FINAL e-graph — exactly
//! `gen_strict_labels`' own precedent (that binary's module doc: this
//! module built on top of `EpisodeLabels`/`CandidateFeatures` from day one).
//! `candidate.rs`'s own module doc names this precisely as an accepted
//! over-approximation ("acceptable for a cold-start dataset and not
//! silent"): a matched e-class's content and neighborhood can include nodes
//! added to it by a LATER application in the same trajectory. The
//! alternative — an in-loop observer hook threaded through
//! `GuidedSaturation` and a parallel one through the unguided sweep — is
//! `docs/plans/2026-09-01-guide-return-to-go.md` §1.5's skew-free design;
//! it is not built here because doing so for `Unguided` requires
//! instrumenting `EGraph`'s production sweep loop, and the smallest thing
//! that answers this task's question (does trajectory-level ordering
//! diversity produce a hindsight-return label with real spread?) does not
//! need it. `application_ordinal`/`budget_fraction` themselves are NOT
//! subject to this approximation (`candidate.rs`'s own doc): they are a
//! pure function of the recorded `ApplicationId`, exact regardless of when
//! `observe` is called.
//!
//! Usage:
//! ```bash
//! cargo run --release -p pixelflow-pipeline --features training --bin gen_r2g_trajectories -- \
//!     --corpus-dir pixelflow-pipeline/data \
//!     --manifest pixelflow-pipeline/corpus_split.toml \
//!     --strict-checkpoint pixelflow-pipeline/data/guide_checkpoint_strict_v1.json \
//!     --train-guide-report docs/results/2026-09-01-train-guide-report.json \
//!     --train-limit 800 \
//!     --report-json docs/results/2026-09-01-r2g-dataset.json \
//!     --report-md docs/results/2026-09-01-r2g-dataset.md
//! ```

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Parser;

use pixelflow_ir::{ExprArena, ExprId};
use pixelflow_pipeline::training::corpus::read_corpus;
use pixelflow_pipeline::training::r2g::{OrderingPolicy, TrajectoryRow, log_regret, spread_report};
use pixelflow_pipeline::training::split::{Family, SplitManifest, Tier};
use pixelflow_pipeline::training::structural::FenceKey;
use pixelflow_search::egraph::{
    ApplicationId, CandidateFeatures, CostModel, EGraph, EpisodeLabels, Firing, GuidedSaturation,
    Label, Origin, REGISTERED_PRIMARY_BUDGET_APPLICATIONS, SaturationStop, all_rules, extract_dag,
};
use pixelflow_search::nnue::factored::EMBED_DIM;
use pixelflow_search::nnue::guide::diversity::{UniformRandomGuide, uniform_unit_score};
use pixelflow_search::nnue::guide::linear::{LinearCandidateGuide, PerRuleRateGuide};
use pixelflow_search::nnue::guide::{CandidateSummary, SaturationGuide};

/// Registered primary tier, `B`.
const B_PRIMARY: usize = REGISTERED_PRIMARY_BUDGET_APPLICATIONS;
/// Registered secondary tier (Phase 3 registration §4).
const B_SECONDARY: usize = 200;
/// Ceiling every trajectory runs to (or ends before) — `4 * B`, per this
/// task's instruction; see the module doc's "Checkpoints" section for why
/// the two REPORTED checkpoints stay at the registered 100/200 tiers rather
/// than becoming B/4B.
const CEILING: usize = 4 * B_PRIMARY;
/// Same iteration ceiling `gen_strict_labels`/`guide_headroom` use — a
/// generous safety cap, not expected to bind under an application budget
/// far below it.
const SATURATE_MAX_ITERS: usize = 400;
/// Per-stage wall-clock safety ceiling — this is an offline batch
/// measurement, so production's sub-second deadline does not apply; a
/// single expression's three checkpoint stages share this budget each.
const SATURATE_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Parser)]
#[command(name = "gen_r2g_trajectories")]
#[command(
    about = "Mint the R2G trajectory dataset: K ordering policies per expression, \
                    hindsight return labels at B=100/B=200"
)]
struct Args {
    /// Directory holding `corpus_{train,dev}.bin` and `corpus_dev_ood.bin`.
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    corpus_dir: String,

    /// Path to the checked-in family split manifest.
    #[arg(long, default_value = "pixelflow-pipeline/corpus_split.toml")]
    manifest: String,

    /// The frozen Round-1 `LinearCandidateGuide` checkpoint (`strict-v1`),
    /// used by the `FrozenLinear`/`EpsilonMix` policies.
    #[arg(
        long,
        default_value = "pixelflow-pipeline/data/guide_checkpoint_strict_v1.json"
    )]
    strict_checkpoint: String,

    /// `train_guide`'s `--report-json` output, supplying the TRAIN-measured
    /// per-rule rates the `per-rule` control policy scores from.
    #[arg(
        long,
        default_value = "docs/results/2026-09-01-train-guide-report.json"
    )]
    train_guide_report: String,

    /// Which split(s) to mint: `train`, `dev`, `sh`, `bezier`, or `all`.
    #[arg(long, default_value = "all")]
    tier: String,

    /// Cap on TRAIN expressions processed (0 = all), stride-sampled exactly
    /// as `gen_strict_labels --train-limit` samples — identical semantics,
    /// so the two mints are comparable at the same limit.
    #[arg(long, default_value_t = 800)]
    train_limit: usize,

    /// Cap on DEV expressions processed (0 = all).
    #[arg(long, default_value_t = 0)]
    dev_limit: usize,

    /// Number of `random:<seed>` trajectories per expression (seeds
    /// `1..=n_rand`) — §2's `n_rand = 6` justified there from per-round
    /// survivor-set coverage, not re-derived here.
    #[arg(long, default_value_t = 6)]
    n_rand: u64,

    /// Comma-separated `numerator/denominator` ratios for the
    /// `mix:strict-v1:<n>/<d>:<seed>` policies — §2's table uses
    /// `1/4,1/4,1/2` (three mixes, reusing seeds 1 and 2 from the random
    /// pool: `1/4:1`, `1/4:2`, `1/2:1`).
    #[arg(long, default_value = "1/4,1/4,1/2")]
    mix: String,

    /// Skip corpus entries whose arena node count exceeds this — identical
    /// flag and identical default to `gen_strict_labels --max-expr-nodes`,
    /// for direct comparability (this binary's own module doc and mint
    /// report restate the value used, per this task's instruction).
    #[arg(long, default_value_t = 250)]
    max_expr_nodes: usize,

    /// Per-expression e-class cap passed to every saturation call —
    /// identical flag and identical default to `gen_strict_labels
    /// --max-classes`.
    #[arg(long, default_value_t = 2_000)]
    max_classes: usize,

    /// Output directory for the four `r2g_{train,dev,sh,bezier}.jsonl` /
    /// `r2g_trajectories_{train,dev,sh,bezier}.jsonl` file pairs.
    #[arg(long, default_value = "pixelflow-pipeline/data")]
    out_dir: String,

    #[arg(long)]
    report_json: Option<String>,

    #[arg(long)]
    report_md: Option<String>,
}

// ============================================================================
// Guide construction
// ============================================================================

/// The two loaded, checkpoint-backed guides every guided policy but
/// `Random`/`Unguided` needs, loaded once per mint run (not once per
/// expression/trajectory — both are read-only and reused across every
/// trajectory). `Option` (rather than requiring both unconditionally) so a
/// caller — including this binary's own equivalence test, which only
/// exercises `Unguided` — can build an empty cache without touching disk;
/// [`build_guide`] panics loudly (NO SILENT FAILURES) if a policy needs an
/// entry the cache doesn't have.
struct GuideCache {
    per_rule: Option<PerRuleRateGuide>,
    frozen_linear: Option<LinearCandidateGuide>,
}

impl GuideCache {
    /// Used only by this binary's own equivalence test (`Unguided` needs no
    /// checkpoint-backed guide at all) — dead outside `#[cfg(test)]`.
    #[allow(dead_code)]
    fn empty() -> Self {
        Self {
            per_rule: None,
            frozen_linear: None,
        }
    }

    fn load(strict_checkpoint: &Path, train_guide_report: &Path) -> Self {
        let frozen_linear = LinearCandidateGuide::load(strict_checkpoint).unwrap_or_else(|e| {
            panic!(
                "gen_r2g_trajectories: failed to load strict checkpoint {}: {e}",
                strict_checkpoint.display()
            )
        });
        let per_rule = PerRuleRateGuide::from_train_guide_report(train_guide_report)
            .unwrap_or_else(|e| {
                panic!(
                    "gen_r2g_trajectories: failed to load train_guide report {}: {e}",
                    train_guide_report.display()
                )
            });
        Self {
            per_rule: Some(per_rule),
            frozen_linear: Some(frozen_linear),
        }
    }
}

/// A fully-built, owned ordering scorer for one [`OrderingPolicy`] (every
/// variant except `Unguided`, which never builds one — see
/// [`run_trajectory`]). An enum rather than `Box<dyn SaturationGuide>`
/// because [`GuidedSaturation`] requires its guide parameter `G:
/// SaturationGuide` to be `Sized` (a bare reference field, no `?Sized`
/// relaxation) — a trait object satisfies the trait but not `Sized`, so a
/// concrete, monomorphized carrier is what a guided episode can actually
/// borrow.
enum BuiltGuide<'a> {
    PerRule(&'a PerRuleRateGuide),
    Linear(&'a LinearCandidateGuide),
    Random(UniformRandomGuide),
    /// A rank-mix of `base`'s score with a per-candidate uniform term —
    /// `docs/plans/2026-09-01-guide-return-to-go.md` §2's `RankMixGuide`
    /// row, computed inline here rather than as a separate
    /// `pixelflow-search` type (see this binary's module doc: the smallest
    /// thing that answers the question).
    Mix {
        base: Box<BuiltGuide<'a>>,
        numerator: u16,
        denominator: u16,
        seed: u64,
    },
}

impl SaturationGuide for BuiltGuide<'_> {
    fn score_candidates(&self, candidates: &[CandidateSummary]) -> Vec<f32> {
        match self {
            BuiltGuide::PerRule(g) => g.score_candidates(candidates),
            BuiltGuide::Linear(g) => g.score_candidates(candidates),
            BuiltGuide::Random(g) => g.score_candidates(candidates),
            BuiltGuide::Mix {
                base,
                numerator,
                denominator,
                seed,
            } => {
                let base_scores = base.score_candidates(candidates);
                let n = base_scores.len();
                if n == 0 {
                    return Vec::new();
                }
                let eps = f32::from(*numerator) / f32::from(*denominator);
                // Rank-normalize `base_scores` to `[0,1]` (highest score ->
                // 1.0) — mixing on ranks, not raw logits, so `eps` means the
                // same thing regardless of the base guide's score scale.
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by(|&a, &b| {
                    base_scores[b]
                        .partial_cmp(&base_scores[a])
                        .unwrap_or_else(|| {
                            panic!(
                                "BuiltGuide::Mix: base guide produced a non-finite score \
                             ({} or {})",
                                base_scores[a], base_scores[b]
                            )
                        })
                });
                let mut rank_norm = vec![0.0f32; n];
                for (rank, &idx) in order.iter().enumerate() {
                    rank_norm[idx] = if n > 1 {
                        1.0 - (rank as f32 / (n as f32 - 1.0))
                    } else {
                        1.0
                    };
                }
                candidates
                    .iter()
                    .zip(rank_norm.iter())
                    .map(|(c, &rn)| {
                        let u = uniform_unit_score(*seed, c);
                        (1.0 - eps) * rn + eps * u
                    })
                    .collect()
            }
        }
    }
}

/// Build the [`BuiltGuide`] a non-`Unguided` [`OrderingPolicy`] names.
///
/// # Panics
///
/// - `Unguided` has no guide — callers must special-case it (see
///   [`run_trajectory`]); calling this with it is a caller bug.
/// - A `PerRuleRate`/`FrozenLinear` policy whose corresponding `cache` entry
///   is `None` — a caller that built an empty [`GuideCache`] (e.g. this
///   binary's own equivalence test) but then asked for a checkpoint-backed
///   policy anyway.
/// - A nested `EpsilonMix` (an `EpsilonMix` whose `guide` is itself an
///   `EpsilonMix`) — not a meaning this binary defines; §2's table never
///   nests one.
fn build_guide<'a>(policy: &OrderingPolicy, cache: &'a GuideCache) -> BuiltGuide<'a> {
    match policy {
        OrderingPolicy::Unguided => {
            panic!("build_guide: OrderingPolicy::Unguided has no SaturationGuide")
        }
        OrderingPolicy::PerRuleRate => BuiltGuide::PerRule(cache.per_rule.as_ref().unwrap_or_else(
            || panic!("build_guide: PerRuleRate policy requested but GuideCache has no per_rule guide loaded"),
        )),
        OrderingPolicy::FrozenLinear(_) => {
            BuiltGuide::Linear(cache.frozen_linear.as_ref().unwrap_or_else(|| {
                panic!(
                    "build_guide: FrozenLinear policy requested but GuideCache has no \
                     frozen_linear guide loaded"
                )
            }))
        }
        OrderingPolicy::Random(seed) => BuiltGuide::Random(UniformRandomGuide { seed: *seed }),
        OrderingPolicy::EpsilonMix {
            guide,
            numerator,
            denominator,
            seed,
        } => {
            assert!(
                !matches!(guide.as_ref(), OrderingPolicy::EpsilonMix { .. }),
                "build_guide: nested EpsilonMix is not supported — {guide:?} is itself an \
                 EpsilonMix"
            );
            BuiltGuide::Mix {
                base: Box::new(build_guide(guide, cache)),
                numerator: *numerator,
                denominator: *denominator,
                seed: *seed,
            }
        }
    }
}

/// The K=12 ordering policies §2's table names, built from CLI-parameterized
/// seed counts. `strict-v1`'s label is fixed to that name regardless of the
/// `--strict-checkpoint` path — see [`OrderingPolicy::FrozenLinear`]'s doc.
fn build_policies(n_rand: u64, mix_ratios: &[(u16, u16)]) -> Vec<OrderingPolicy> {
    let mut policies = vec![
        OrderingPolicy::Unguided,
        OrderingPolicy::PerRuleRate,
        OrderingPolicy::FrozenLinear("strict-v1".to_string()),
    ];
    for seed in 1..=n_rand {
        policies.push(OrderingPolicy::Random(seed));
    }
    // §2's table reuses seeds 1 and 2 from the random pool for its three
    // mixes (`mix:strict-v1:1/4:1`, `mix:strict-v1:1/4:2`,
    // `mix:strict-v1:1/2:1`) rather than minting fresh ones — deliberate,
    // so a mix trajectory shares its randomness source with an existing
    // named `random:<seed>` trajectory for direct comparison.
    let mix_seeds = [1u64, 2u64, 1u64];
    for (i, &(num, den)) in mix_ratios.iter().enumerate() {
        let seed = mix_seeds[i % mix_seeds.len()];
        policies.push(OrderingPolicy::EpsilonMix {
            guide: Box::new(OrderingPolicy::FrozenLinear("strict-v1".to_string())),
            numerator: num,
            denominator: den,
            seed,
        });
    }
    policies
}

fn parse_mix_ratios(s: &str) -> Vec<(u16, u16)> {
    s.split(',')
        .map(|tok| {
            let tok = tok.trim();
            let (num, den) = tok.split_once('/').unwrap_or_else(|| {
                panic!("gen_r2g_trajectories: --mix entry {tok:?} is not \"num/den\"")
            });
            let num: u16 = num
                .parse()
                .unwrap_or_else(|e| panic!("gen_r2g_trajectories: --mix numerator {num:?}: {e}"));
            let den: u16 = den
                .parse()
                .unwrap_or_else(|e| panic!("gen_r2g_trajectories: --mix denominator {den:?}: {e}"));
            (num, den)
        })
        .collect()
}

// ============================================================================
// Running one trajectory
// ============================================================================

/// One trajectory's raw outcome: the e-graph it produced (kept, so the
/// per-application feature/label pass below can walk its final provenance
/// log — see the module doc's "Feature observation is post-hoc" section),
/// its three checkpoint costs, and the hindsight strict labels computed
/// against the ceiling extraction.
struct RawTrajectory {
    egraph: EGraph,
    app_actual_b100: u64,
    cost_b100: u64,
    app_actual_b200: u64,
    cost_b200: u64,
    app_actual_ceiling: u64,
    cost_ceiling: u64,
    /// `true` iff saturation had already stopped for a reason other than
    /// the application budget (quiesced / class-capped / iteration-
    /// ceilinged) by the time the ceiling stage returned.
    ended: bool,
    labels: EpisodeLabels,
}

/// Run one `(expression, policy)` trajectory through the three checkpoint
/// stages (B=100, B=200, ceiling=4B), extracting at each. The `Unguided`
/// and guided branches are written out separately, rather than behind one
/// shared closure/helper, because a closure capturing `&mut egraph` would
/// hold that borrow for its own entire lifetime — blocking the
/// `extract_dag(&egraph, ...)` calls this function needs to interleave
/// between checkpoint stages; three direct calls avoid the borrow
/// entirely and read no worse for the duplication.
fn run_trajectory(
    policy: &OrderingPolicy,
    cache: &GuideCache,
    arena: &ExprArena,
    root: ExprId,
    max_classes: usize,
    costs: &CostModel,
) -> RawTrajectory {
    let mut egraph = EGraph::with_rules(all_rules());
    let root_class = egraph.add_arena(arena, root);

    let (stats_b100, cost_b100, stats_b200, cost_b200, stats_ceiling, ext_ceiling);

    if matches!(policy, OrderingPolicy::Unguided) {
        // The production path, byte-for-byte: `EGraph::saturate_until_
        // applications` is called directly, untouched — see this binary's
        // `unguided_trajectory_reproduces_run_anytime_curve_cost_at_b` test
        // for the equivalence this guarantees against
        // `egraph::run_anytime_curve`.
        stats_b100 = egraph.saturate_until_applications(
            B_PRIMARY,
            SATURATE_MAX_ITERS,
            max_classes,
            SATURATE_TIMEOUT,
        );
        cost_b100 = extract_dag(&egraph, root_class, costs).total_cost;
        stats_b200 = egraph.saturate_until_applications(
            B_SECONDARY,
            SATURATE_MAX_ITERS,
            max_classes,
            SATURATE_TIMEOUT,
        );
        cost_b200 = extract_dag(&egraph, root_class, costs).total_cost;
        stats_ceiling = egraph.saturate_until_applications(
            CEILING,
            SATURATE_MAX_ITERS,
            max_classes,
            SATURATE_TIMEOUT,
        );
        ext_ceiling = extract_dag(&egraph, root_class, costs);
    } else {
        let rule_embeds = vec![[0.0f32; EMBED_DIM]; all_rules().len()];
        let built = build_guide(policy, cache);
        // ONE continuous episode across all three checkpoints — resuming a
        // fresh `GuidedSaturation` per checkpoint would re-score and
        // re-fire every already-resolved candidate key, per that type's own
        // doc ("silently handicapped at every checkpoint after the
        // first").
        let mut episode = GuidedSaturation::new(&built, &rule_embeds);
        stats_b100 = episode.until_applications(
            &mut egraph,
            B_PRIMARY,
            SATURATE_MAX_ITERS,
            max_classes,
            SATURATE_TIMEOUT,
        );
        cost_b100 = extract_dag(&egraph, root_class, costs).total_cost;
        stats_b200 = episode.until_applications(
            &mut egraph,
            B_SECONDARY,
            SATURATE_MAX_ITERS,
            max_classes,
            SATURATE_TIMEOUT,
        );
        cost_b200 = extract_dag(&egraph, root_class, costs).total_cost;
        stats_ceiling = episode.until_applications(
            &mut egraph,
            CEILING,
            SATURATE_MAX_ITERS,
            max_classes,
            SATURATE_TIMEOUT,
        );
        ext_ceiling = extract_dag(&egraph, root_class, costs);
    }

    let labels = EpisodeLabels::compute_strict(&egraph, ext_ceiling.root, &ext_ceiling.choices);
    let ended = stats_ceiling.stop != SaturationStop::ApplicationBudget;

    RawTrajectory {
        egraph,
        app_actual_b100: stats_b100.applications as u64,
        cost_b100: cost_b100 as u64,
        app_actual_b200: stats_b200.applications as u64,
        cost_b200: cost_b200 as u64,
        app_actual_ceiling: stats_ceiling.applications as u64,
        cost_ceiling: ext_ceiling.total_cost as u64,
        ended,
        labels,
    }
}

/// Which recorded applications actually changed the e-graph (committed a
/// node or a union), read post-hoc from `egraph`'s own provenance rather
/// than threaded through the saturation call: an application "changed"
/// iff it created at least one e-node (`Origin::Rule(app_id)` appears on
/// some node) or caused at least one recorded union
/// (`UnionEvent::application_id == Some(app_id)`) — the same two ways
/// `apply_action_from_rule`'s executor can report `> 0`.
fn changed_application_ids(egraph: &EGraph) -> HashSet<ApplicationId> {
    let mut ids: HashSet<ApplicationId> = egraph
        .provenance()
        .origins()
        .filter_map(|(_, origin)| match origin {
            Origin::Rule(id) => Some(id),
            Origin::Seed => None,
        })
        .collect();
    for event in egraph.provenance().union_events() {
        if let Some(id) = event.application_id {
            ids.insert(id);
        }
    }
    ids
}

fn opt_f32_json(x: Option<f32>) -> String {
    match x {
        Some(v) => format!("{v:.6}"),
        None => "null".to_string(),
    }
}

fn mean_of_some(xs: &[Option<f32>]) -> Option<f32> {
    let vals: Vec<f32> = xs.iter().filter_map(|&x| x).collect();
    if vals.is_empty() {
        None
    } else {
        Some(vals.iter().sum::<f32>() / vals.len() as f32)
    }
}

// ============================================================================
// Minting one expression (K trajectories -> R2gRecord rows + TrajectoryRows)
// ============================================================================

struct ExprMintOutcome {
    applications_written: usize,
    trajectory_rows: Vec<TrajectoryRow>,
    /// `true` iff every trajectory's checkpoint costs bottomed out at zero
    /// (the registration's zero-best convention: a positive cost against a
    /// zero reference is excluded rather than reported as infinite regret)
    /// — this expression contributed no records to the output file.
    zero_best_excluded: bool,
}

#[allow(clippy::too_many_arguments)]
fn mint_expression(
    name: &str,
    arena: &ExprArena,
    root: ExprId,
    family_band: u32,
    family_seed: u64,
    tier_label: &str,
    policies: &[OrderingPolicy],
    cache: &GuideCache,
    max_classes: usize,
    costs: &CostModel,
    rule_names: &[String],
    out: &mut impl Write,
) -> ExprMintOutcome {
    let expr_node_count = arena.nodes_raw().len();
    let raw: Vec<RawTrajectory> = policies
        .iter()
        .map(|p| run_trajectory(p, cache, arena, root, max_classes, costs))
        .collect();

    // Empirical best (§1.2's `c*_e`): the minimum extraction cost seen at
    // ANY checkpoint of ANY trajectory — wider than just the two reported
    // checkpoints (includes the ceiling extraction), matching "the unguided
    // trajectory contributes its full grid" in spirit without needing the
    // full anytime grid this mint does not sample.
    let best: u64 = raw
        .iter()
        .flat_map(|t| [t.cost_b100, t.cost_b200, t.cost_ceiling])
        .min()
        .unwrap_or(0);

    if best == 0 {
        return ExprMintOutcome {
            applications_written: 0,
            trajectory_rows: Vec::new(),
            zero_best_excluded: true,
        };
    }

    let returns_b100: Vec<Option<f32>> =
        raw.iter().map(|t| log_regret(t.cost_b100, best)).collect();
    let returns_b200: Vec<Option<f32>> =
        raw.iter().map(|t| log_regret(t.cost_b200, best)).collect();
    let mean_b100 = mean_of_some(&returns_b100);
    let mean_b200 = mean_of_some(&returns_b200);

    let mut trajectory_rows = Vec::with_capacity(raw.len());
    let mut applications_written = 0usize;

    for (i, (traj, policy)) in raw.iter().zip(policies.iter()).enumerate() {
        let trajectory_id = i as u32;
        let policy_label = policy.label();
        let return_b100 = returns_b100[i];
        let return_b200 = returns_b200[i];
        let centered_b100 = return_b100.zip(mean_b100).map(|(r, m)| r - m);
        let centered_b200 = return_b200.zip(mean_b200).map(|(r, m)| r - m);

        trajectory_rows.push(TrajectoryRow {
            expr_name: name.to_string(),
            tier: tier_label.to_string(),
            trajectory_id,
            policy: policy_label.clone(),
            app_actual_b100: traj.app_actual_b100,
            cost_b100: traj.cost_b100,
            app_actual_b200: traj.app_actual_b200,
            cost_b200: traj.cost_b200,
            ended: traj.ended,
            ended_at_apps: traj.app_actual_ceiling,
            return_b100,
            return_b200,
        });

        let changed_ids = changed_application_ids(&traj.egraph);
        for (app_id, record) in traj.egraph.provenance().applications() {
            let firing = Firing {
                rule_idx: record.rule_idx,
                match_root: record.match_root,
                application_ordinal: app_id.as_u64(),
                registered_budget: REGISTERED_PRIMARY_BUDGET_APPLICATIONS,
            };
            let features = CandidateFeatures::observe(&traj.egraph, &firing);
            let label_positive =
                traj.labels.labels.get(&app_id).copied() == Some(Label::LoadBearing);
            let rule_name = rule_names
                .get(record.rule_idx)
                .cloned()
                .unwrap_or_else(|| format!("<rule {}>", record.rule_idx));
            let changed = changed_ids.contains(&app_id);

            // A histogram, not the raw op list — see `gen_strict_labels`'s
            // identical comment (a matched class deep into saturation can
            // hold thousands of near-duplicate node shapes).
            let mut op_hist: BTreeMap<String, usize> = BTreeMap::new();
            for op in &features.neighborhood_ops {
                *op_hist.entry(format!("{op:?}")).or_insert(0) += 1;
            }
            let hist_json: String = {
                let mut s = String::from("{");
                for (i, (op, count)) in op_hist.iter().enumerate() {
                    if i > 0 {
                        s.push(',');
                    }
                    s.push_str(&format!("{op:?}:{count}"));
                }
                s.push('}');
                s
            };

            writeln!(
                out,
                "{{\"expr_name\":{:?},\"family_band\":{},\"family_seed\":{},\
                 \"expr_node_count\":{},\"rule_idx\":{},\"rule_name\":{:?},\
                 \"budget_fraction\":{:.6},\"match_class_node_count\":{},\
                 \"neighborhood_op_count\":{},\"neighborhood_op_hist\":{},\
                 \"label_positive\":{},\"trajectory_id\":{},\"policy\":{:?},\
                 \"round_ordinal\":{},\"application_ordinal\":{},\"changed\":{},\
                 \"cost_b100\":{},\"cost_b200\":{},\"expr_best_cost\":{},\
                 \"return_b100\":{},\"return_b200\":{},\"centered_b100\":{},\
                 \"centered_b200\":{}}}",
                name,
                family_band,
                family_seed,
                expr_node_count,
                record.rule_idx,
                rule_name,
                features.budget_fraction(),
                features.key.content.node_count(),
                features.neighborhood_ops.len(),
                hist_json,
                label_positive,
                trajectory_id,
                policy_label,
                record.step,
                app_id.as_u64(),
                changed,
                traj.cost_b100,
                traj.cost_b200,
                best,
                opt_f32_json(return_b100),
                opt_f32_json(return_b200),
                opt_f32_json(centered_b100),
                opt_f32_json(centered_b200),
            )
            .unwrap_or_else(|e| panic!("gen_r2g_trajectories: write failed: {e}"));

            applications_written += 1;
        }
    }

    ExprMintOutcome {
        applications_written,
        trajectory_rows,
        zero_best_excluded: false,
    }
}

// ============================================================================
// Corpus loading, fence checks, and per-split mint
// ============================================================================

/// Parse `gen_bench_corpus`'s `{tier}_b{band:02}_f{seed:02}_{idx:05}` name
/// format back into a [`Family`] — identical logic to `gen_strict_labels`'s
/// own (private, duplicated per this codebase's own convention: every
/// label-minting binary that needs this parses it itself rather than
/// sharing a helper — see that binary's copy).
fn parse_family(name: &str) -> Family {
    let parts: Vec<&str> = name.split('_').collect();
    assert!(
        parts.len() >= 3,
        "gen_r2g_trajectories: corpus entry name '{name}' does not match the expected \
         `{{tier}}_b{{band}}_f{{seed}}_{{idx}}` format"
    );
    let band_tok = parts[1].strip_prefix('b').unwrap_or_else(|| {
        panic!("gen_r2g_trajectories: corpus entry name '{name}' has no 'b<NN>' band token")
    });
    let seed_tok = parts[2].strip_prefix('f').unwrap_or_else(|| {
        panic!("gen_r2g_trajectories: corpus entry name '{name}' has no 'f<NN>' seed token")
    });
    let band: usize = band_tok.parse().unwrap_or_else(|e| {
        panic!("gen_r2g_trajectories: '{name}' band token '{band_tok}' is not a number: {e}")
    });
    let seed: u64 = seed_tok.parse().unwrap_or_else(|e| {
        panic!("gen_r2g_trajectories: '{name}' seed token '{seed_tok}' is not a number: {e}")
    });
    Family::new(band, seed)
}

/// Stride-sample `entries` down to at most `limit`, evenly across the whole
/// population — identical to `gen_strict_labels`'s own `stride_sample`.
fn stride_sample<T: Clone>(entries: Vec<T>, limit: usize) -> Vec<T> {
    if limit == 0 || limit >= entries.len() {
        return entries;
    }
    let stride = entries.len() as f64 / limit as f64;
    let mut sampled = Vec::with_capacity(limit);
    for i in 0..limit {
        let idx = ((i as f64) * stride) as usize;
        sampled.push(entries[idx.min(entries.len() - 1)].clone());
    }
    sampled
}

/// Fail loud (NO SILENT FAILURES) if any classical entry's parsed family is
/// not assigned to `expected` by the checked-in split manifest — the same
/// fence `gen_strict_labels` enforces for TRAIN/DEV.
fn assert_family_fence(
    entries: &[(String, ExprArena, ExprId)],
    manifest: &SplitManifest,
    expected: Tier,
) {
    for (name, _, _) in entries {
        let family = parse_family(name);
        let assigned = manifest.tier_of(family);
        assert_eq!(
            assigned,
            Some(expected),
            "gen_r2g_trajectories: fence check failed — corpus entry '{name}' parses to \
             {family}, but corpus_split.toml assigns that family to {assigned:?}, not \
             {expected:?}"
        );
    }
}

/// Every `FenceKey` in `corpus_train.bin` — the TRAIN-fence source for the
/// OOD (`sh`/`bezier`) splits, which carry no `{band,seed}` family and so
/// cannot be fence-checked by [`assert_family_fence`].
fn train_fence_keys(corpus_dir: &Path) -> HashSet<FenceKey> {
    let path = corpus_dir.join("corpus_train.bin");
    let entries = read_corpus(&path).unwrap_or_else(|e| {
        panic!(
            "gen_r2g_trajectories: failed to read {}: {e}",
            path.display()
        )
    });
    entries
        .iter()
        .map(|(_, arena, root)| FenceKey::of(arena, *root))
        .collect()
}

/// Hard-fail if any OOD entry shares a feature-quotient structure with
/// TRAIN — the same discipline `phase3_at_budget_eval`'s
/// `enforce_train_fence` applies (round-1b registration §3).
fn assert_train_fence(
    entries: &[(String, ExprArena, ExprId)],
    train_keys: &HashSet<FenceKey>,
    source: &str,
) {
    let collisions: Vec<&str> = entries
        .iter()
        .filter(|(_, arena, root)| train_keys.contains(&FenceKey::of(arena, *root)))
        .map(|(name, _, _)| name.as_str())
        .collect();
    assert!(
        collisions.is_empty(),
        "gen_r2g_trajectories: TRAIN-fence violation in {source}: {} of {} entries share a \
         feature-quotient structure with corpus_train.bin: {:?}{}",
        collisions.len(),
        entries.len(),
        &collisions[..collisions.len().min(10)],
        if collisions.len() > 10 {
            ", ... (truncated)"
        } else {
            ""
        },
    );
}

/// Everything one split's mint measured — feeds the combined report.
struct SplitOutcome {
    tier: String,
    expressions: usize,
    zero_best_excluded: usize,
    skipped_oversized: usize,
    trajectories: usize,
    applications: usize,
    trajectory_rows: Vec<TrajectoryRow>,
}

#[allow(clippy::too_many_arguments)]
fn mint_split(
    tier_label: &str,
    entries: &[(String, ExprArena, ExprId)],
    policies: &[OrderingPolicy],
    cache: &GuideCache,
    max_expr_nodes: usize,
    max_classes: usize,
    rule_names: &[String],
    family_of: impl Fn(&str) -> (u32, u64),
    out_records_path: &Path,
    out_trajectories_path: &Path,
) -> SplitOutcome {
    let costs = CostModel::latency_prior();
    let mut records_out = BufWriter::new(File::create(out_records_path).unwrap_or_else(|e| {
        panic!(
            "gen_r2g_trajectories: cannot create {}: {e}",
            out_records_path.display()
        )
    }));
    let mut trajectories_out =
        BufWriter::new(File::create(out_trajectories_path).unwrap_or_else(|e| {
            panic!(
                "gen_r2g_trajectories: cannot create {}: {e}",
                out_trajectories_path.display()
            )
        }));

    let mut outcome = SplitOutcome {
        tier: tier_label.to_string(),
        expressions: 0,
        zero_best_excluded: 0,
        skipped_oversized: 0,
        trajectories: 0,
        applications: 0,
        trajectory_rows: Vec::new(),
    };

    let n = entries.len();
    for (i, (name, arena, root)) in entries.iter().enumerate() {
        let node_count = arena.nodes_raw().len();
        if node_count > max_expr_nodes {
            outcome.skipped_oversized += 1;
            continue;
        }
        let (family_band, family_seed) = family_of(name);
        let result = mint_expression(
            name,
            arena,
            *root,
            family_band,
            family_seed,
            tier_label,
            policies,
            cache,
            max_classes,
            &costs,
            rule_names,
            &mut records_out,
        );
        if result.zero_best_excluded {
            outcome.zero_best_excluded += 1;
        } else {
            outcome.expressions += 1;
            outcome.trajectories += result.trajectory_rows.len();
            outcome.applications += result.applications_written;
            for row in &result.trajectory_rows {
                let json = serde_json::to_string(row).unwrap_or_else(|e| {
                    panic!("gen_r2g_trajectories: TrajectoryRow serialization failed: {e}")
                });
                writeln!(trajectories_out, "{json}").unwrap_or_else(|e| {
                    panic!(
                        "gen_r2g_trajectories: write to {}: {e}",
                        out_trajectories_path.display()
                    )
                });
            }
            outcome.trajectory_rows.extend(result.trajectory_rows);
        }
        if (i + 1) % 25 == 0 || i + 1 == n {
            eprintln!(
                "gen_r2g_trajectories[{tier_label}]: {}/{n} expressions minted ({} \
                 applications so far)",
                i + 1,
                outcome.applications
            );
        }
    }

    records_out.flush().unwrap_or_else(|e| {
        panic!(
            "gen_r2g_trajectories: flushing {}: {e}",
            out_records_path.display()
        )
    });
    trajectories_out.flush().unwrap_or_else(|e| {
        panic!(
            "gen_r2g_trajectories: flushing {}: {e}",
            out_trajectories_path.display()
        )
    });

    outcome
}

// ============================================================================
// Report
// ============================================================================

fn write_report(
    path_json: Option<&str>,
    path_md: Option<&str>,
    args: &Args,
    outcomes: &[SplitOutcome],
    mint_seconds: f64,
) {
    let mut per_split_json = String::new();
    let mut per_split_md = String::new();
    for (i, outcome) in outcomes.iter().enumerate() {
        let spread = spread_report(&outcome.trajectory_rows);

        // Per-policy median regret at B=100 — the report's headline
        // per-policy statistic.
        let mut by_policy: BTreeMap<&str, Vec<f32>> = BTreeMap::new();
        for row in &outcome.trajectory_rows {
            if let Some(r) = row.return_b100 {
                by_policy.entry(row.policy.as_str()).or_default().push(r);
            }
        }
        let mut policy_medians: Vec<(String, f32, usize)> = by_policy
            .into_iter()
            .map(|(policy, mut rets)| {
                rets.sort_by(|a, b| a.partial_cmp(b).expect("finite return"));
                let n = rets.len();
                let median = if n == 0 {
                    0.0
                } else if n % 2 == 1 {
                    rets[n / 2]
                } else {
                    (rets[n / 2 - 1] + rets[n / 2]) / 2.0
                };
                (policy.to_string(), median, n)
            })
            .collect();
        policy_medians.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("finite median"));

        if i > 0 {
            per_split_json.push_str(",\n");
            per_split_md.push('\n');
        }
        per_split_json.push_str(&format!(
            "  {{\n    \"tier\": \"{}\",\n    \"expressions\": {},\n    \
             \"zero_best_excluded\": {},\n    \"skipped_oversized\": {},\n    \
             \"trajectories\": {},\n    \"applications\": {},\n    \
             \"zero_spread_b100\": {},\n    \"zero_spread_b100_share\": {:.4},\n    \
             \"zero_spread_b100_record_share\": {:.4},\n    \
             \"spread_b100_quartiles\": [{:.4}, {:.4}, {:.4}],\n    \
             \"dataset_gate_fired\": {},\n    \"per_policy_median_return_b100\": [\n{}\n    ]\n  }}",
            outcome.tier,
            outcome.expressions,
            outcome.zero_best_excluded,
            outcome.skipped_oversized,
            outcome.trajectories,
            outcome.applications,
            spread.zero_spread_b100,
            if spread.expressions == 0 {
                0.0
            } else {
                spread.zero_spread_b100 as f64 / spread.expressions as f64
            },
            spread.zero_spread_b100_record_share,
            spread.spread_b100_quartiles.0,
            spread.spread_b100_quartiles.1,
            spread.spread_b100_quartiles.2,
            spread.dataset_gate_fired,
            policy_medians
                .iter()
                .map(|(policy, median, n)| format!(
                    "      {{\"policy\": {policy:?}, \"median_return_b100\": {median:.4}, \"n\": {n}}}"
                ))
                .collect::<Vec<_>>()
                .join(",\n"),
        ));

        per_split_md.push_str(&format!(
            "## {}\n\n\
             - expressions: {} ({} zero-best excluded, {} oversized-skipped)\n\
             - trajectories: {}\n\
             - applications (JSONL rows): {}\n\
             - return spread at B=100: Q1 {:.4}  median {:.4}  Q3 {:.4}\n\
             - zero-spread expressions: {} / {} ({:.1}%), {:.1}% of records\n\
             - dataset gate (>50% zero-spread): {}\n\n\
             | policy | median return_b100 | n |\n|---|---:|---:|\n{}\n",
            outcome.tier,
            outcome.expressions,
            outcome.zero_best_excluded,
            outcome.skipped_oversized,
            outcome.trajectories,
            outcome.applications,
            spread.spread_b100_quartiles.0,
            spread.spread_b100_quartiles.1,
            spread.spread_b100_quartiles.2,
            spread.zero_spread_b100,
            spread.expressions,
            if spread.expressions == 0 {
                0.0
            } else {
                100.0 * spread.zero_spread_b100 as f64 / spread.expressions as f64
            },
            100.0 * spread.zero_spread_b100_record_share as f64,
            if spread.dataset_gate_fired {
                "FIRED"
            } else {
                "not fired"
            },
            policy_medians
                .iter()
                .map(|(policy, median, n)| format!("| {policy} | {median:.4} | {n} |"))
                .collect::<Vec<_>>()
                .join("\n"),
        ));
    }

    if let Some(path) = path_json {
        let json = format!(
            "{{\n  \"registered_budget_b\": {B_PRIMARY},\n  \"registered_budget_b200\": \
             {B_SECONDARY},\n  \"ceiling\": {CEILING},\n  \"max_expr_nodes\": {},\n  \
             \"max_classes\": {},\n  \"train_limit\": {},\n  \"dev_limit\": {},\n  \
             \"n_rand\": {},\n  \"mix\": {:?},\n  \"mint_seconds\": {mint_seconds:.2},\n  \
             \"splits\": [\n{per_split_json}\n  ]\n}}\n",
            args.max_expr_nodes,
            args.max_classes,
            args.train_limit,
            args.dev_limit,
            args.n_rand,
            args.mix,
        );
        std::fs::write(path, json)
            .unwrap_or_else(|e| panic!("gen_r2g_trajectories: cannot write {path}: {e}"));
        eprintln!("gen_r2g_trajectories: wrote {path}");
    }
    if let Some(path) = path_md {
        let md = format!(
            "# R2G trajectory mint\n\n\
             Registered budget B={B_PRIMARY} (primary) / {B_SECONDARY} (secondary); every \
             trajectory ran to a ceiling of {CEILING} applications. \
             `--max-expr-nodes {}` / `--max-classes {}` (identical flags and defaults to \
             `gen_strict_labels`, for direct comparability). `--train-limit {}` \
             `--dev-limit {}` `--n-rand {}` `--mix {:?}`. Mint wall-clock: {mint_seconds:.1}s.\n\n\
             {per_split_md}",
            args.max_expr_nodes,
            args.max_classes,
            args.train_limit,
            args.dev_limit,
            args.n_rand,
            args.mix,
        );
        std::fs::write(path, md)
            .unwrap_or_else(|e| panic!("gen_r2g_trajectories: cannot write {path}: {e}"));
        eprintln!("gen_r2g_trajectories: wrote {path}");
    }
}

// ============================================================================
// main
// ============================================================================

fn main() {
    let args = Args::parse();
    let start = std::time::Instant::now();

    let corpus_dir = PathBuf::from(&args.corpus_dir);
    let out_dir = PathBuf::from(&args.out_dir);
    let manifest = SplitManifest::load(&PathBuf::from(&args.manifest)).unwrap_or_else(|e| {
        panic!(
            "gen_r2g_trajectories: failed to load split manifest {}: {e}",
            args.manifest
        )
    });

    let cache = GuideCache::load(
        Path::new(&args.strict_checkpoint),
        Path::new(&args.train_guide_report),
    );
    let mix_ratios = parse_mix_ratios(&args.mix);
    let policies = build_policies(args.n_rand, &mix_ratios);
    eprintln!(
        "gen_r2g_trajectories: {} ordering policies per expression: {:?}",
        policies.len(),
        policies
            .iter()
            .map(OrderingPolicy::label)
            .collect::<Vec<_>>()
    );

    let rule_names: Vec<String> = {
        let probe = EGraph::with_rules(all_rules());
        (0..all_rules().len())
            .map(|i| {
                probe
                    .rule(i)
                    .map(|r| r.name().to_string())
                    .unwrap_or_else(|| format!("<rule {i}>"))
            })
            .collect()
    };

    let run_all = args.tier == "all";
    let mut outcomes: Vec<SplitOutcome> = Vec::new();

    if run_all || args.tier == "train" {
        let train_path = corpus_dir.join("corpus_train.bin");
        let entries = read_corpus(&train_path).unwrap_or_else(|e| {
            panic!(
                "gen_r2g_trajectories: failed to read {}: {e}",
                train_path.display()
            )
        });
        let entries = stride_sample(entries, args.train_limit);
        assert_family_fence(&entries, &manifest, Tier::Train);
        eprintln!(
            "gen_r2g_trajectories: minting TRAIN ({} expressions)",
            entries.len()
        );
        outcomes.push(mint_split(
            "train",
            &entries,
            &policies,
            &cache,
            args.max_expr_nodes,
            args.max_classes,
            &rule_names,
            |name| {
                let f = parse_family(name);
                (f.band as u32, f.seed)
            },
            &out_dir.join("r2g_train.jsonl"),
            &out_dir.join("r2g_trajectories_train.jsonl"),
        ));
    }

    if run_all || args.tier == "dev" {
        let dev_path = corpus_dir.join("corpus_dev.bin");
        let entries = read_corpus(&dev_path).unwrap_or_else(|e| {
            panic!(
                "gen_r2g_trajectories: failed to read {}: {e}",
                dev_path.display()
            )
        });
        let entries = stride_sample(entries, args.dev_limit);
        assert_family_fence(&entries, &manifest, Tier::Dev);
        eprintln!(
            "gen_r2g_trajectories: minting DEV ({} expressions)",
            entries.len()
        );
        outcomes.push(mint_split(
            "dev",
            &entries,
            &policies,
            &cache,
            args.max_expr_nodes,
            args.max_classes,
            &rule_names,
            |name| {
                let f = parse_family(name);
                (f.band as u32, f.seed)
            },
            &out_dir.join("r2g_dev.jsonl"),
            &out_dir.join("r2g_trajectories_dev.jsonl"),
        ));
    }

    if run_all || args.tier == "sh" || args.tier == "bezier" {
        let ood_path = corpus_dir.join("corpus_dev_ood.bin");
        let ood_entries = read_corpus(&ood_path).unwrap_or_else(|e| {
            panic!(
                "gen_r2g_trajectories: failed to read {}: {e}",
                ood_path.display()
            )
        });
        let train_keys = train_fence_keys(&corpus_dir);

        if run_all || args.tier == "sh" {
            let entries: Vec<_> = ood_entries
                .iter()
                .filter(|(name, _, _)| name.starts_with("dev_sh_"))
                .cloned()
                .collect();
            assert!(
                !entries.is_empty(),
                "gen_r2g_trajectories: no dev_sh_* entries in {}",
                ood_path.display()
            );
            assert_train_fence(&entries, &train_keys, "corpus_dev_ood.bin (dev_sh_*)");
            eprintln!(
                "gen_r2g_trajectories: minting sh ({} expressions)",
                entries.len()
            );
            outcomes.push(mint_split(
                "sh",
                &entries,
                &policies,
                &cache,
                args.max_expr_nodes,
                args.max_classes,
                &rule_names,
                |_name| (0u32, 0u64),
                &out_dir.join("r2g_sh.jsonl"),
                &out_dir.join("r2g_trajectories_sh.jsonl"),
            ));
        }

        if run_all || args.tier == "bezier" {
            let entries: Vec<_> = ood_entries
                .iter()
                .filter(|(name, _, _)| name.starts_with("dev_bezier_"))
                .cloned()
                .collect();
            assert!(
                !entries.is_empty(),
                "gen_r2g_trajectories: no dev_bezier_* entries in {}",
                ood_path.display()
            );
            assert_train_fence(&entries, &train_keys, "corpus_dev_ood.bin (dev_bezier_*)");
            eprintln!(
                "gen_r2g_trajectories: minting bezier ({} expressions)",
                entries.len()
            );
            outcomes.push(mint_split(
                "bezier",
                &entries,
                &policies,
                &cache,
                args.max_expr_nodes,
                args.max_classes,
                &rule_names,
                |_name| (0u32, 0u64),
                &out_dir.join("r2g_bezier.jsonl"),
                &out_dir.join("r2g_trajectories_bezier.jsonl"),
            ));
        }
    }

    let mint_seconds = start.elapsed().as_secs_f64();

    for outcome in &outcomes {
        println!(
            "=== {} split: {} expressions ({} zero-best excluded, {} oversized-skipped), \
             {} trajectories, {} applications ===",
            outcome.tier,
            outcome.expressions,
            outcome.zero_best_excluded,
            outcome.skipped_oversized,
            outcome.trajectories,
            outcome.applications,
        );
    }
    println!("gen_r2g_trajectories: mint took {mint_seconds:.1}s total");

    write_report(
        args.report_json.as_deref(),
        args.report_md.as_deref(),
        &args,
        &outcomes,
        mint_seconds,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_search::egraph::{APP_CHECKPOINT_GRID, run_anytime_curve};

    fn small_expr() -> (ExprArena, ExprId) {
        // (x + y) * (x + y) + 2 * (x + y) — the same fixture
        // `egraph::anytime`'s own tests use.
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let s = a.push_binary(pixelflow_ir::OpKind::Add, x, y);
        let s2 = a.push_binary(pixelflow_ir::OpKind::Mul, s, s);
        let two = a.push_const(2.0);
        let ts = a.push_binary(pixelflow_ir::OpKind::Mul, two, s);
        let out = a.push_binary(pixelflow_ir::OpKind::Add, s2, ts);
        (a, out)
    }

    /// The task's required equivalence check: the `Unguided` trajectory's
    /// cost at the registered primary budget B must equal
    /// `egraph::run_anytime_curve`'s cost at the same target — both
    /// ultimately call `EGraph::saturate_until_applications` with the same
    /// arguments over the same rule set, so this is a definitional
    /// equality, pinned as a regression so a future change to either path
    /// cannot silently diverge them.
    #[test]
    fn unguided_trajectory_reproduces_run_anytime_curve_cost_at_b() {
        let (arena, root) = small_expr();
        let costs = CostModel::latency_prior();
        let cache = GuideCache::empty();

        let traj = run_trajectory(
            &OrderingPolicy::Unguided,
            &cache,
            &arena,
            root,
            2_000,
            &costs,
        );

        let grid: Vec<usize> = APP_CHECKPOINT_GRID
            .iter()
            .copied()
            .take_while(|&t| t <= B_PRIMARY)
            .collect();
        assert_eq!(
            grid.last().copied(),
            Some(B_PRIMARY),
            "B_PRIMARY must be a literal member of APP_CHECKPOINT_GRID"
        );
        let curve_out = run_anytime_curve(
            &arena,
            root,
            all_rules(),
            &grid,
            2_000,
            10_000,
            Duration::from_secs(30),
            &costs,
        );
        let cost_at_b = curve_out
            .curve
            .checkpoints
            .iter()
            .find(|cp| cp.app_target == B_PRIMARY)
            .expect("B_PRIMARY is in the grid")
            .cost;

        assert_eq!(
            traj.cost_b100, cost_at_b as u64,
            "the minter's Unguided trajectory must extract to the same cost at B=100 as \
             run_anytime_curve's own sample at the same target"
        );
    }

    #[test]
    fn mixed_guide_reduces_to_the_base_guide_when_epsilon_is_zero() {
        // eps = 0/1 -> Mix must reproduce PerRule's own order exactly.
        let per_rule = PerRuleRateGuide::new(vec![0.9, 0.1]);
        let base = BuiltGuide::PerRule(&per_rule);
        let mixed = BuiltGuide::Mix {
            base: Box::new(BuiltGuide::PerRule(&per_rule)),
            numerator: 0,
            denominator: 1,
            seed: 42,
        };
        let candidates = vec![
            CandidateSummary {
                rule_embed: [0.0; EMBED_DIM],
                neighborhood_ops: vec![],
                budget_fraction: 0.1,
                rule_idx: 0,
                match_class_node_count: 1,
                expr_node_count: 1,
            },
            CandidateSummary {
                rule_embed: [0.0; EMBED_DIM],
                neighborhood_ops: vec![],
                budget_fraction: 0.1,
                rule_idx: 1,
                match_class_node_count: 1,
                expr_node_count: 1,
            },
        ];
        let base_scores = base.score_candidates(&candidates);
        let mixed_scores = mixed.score_candidates(&candidates);
        // Both must rank candidate 0 above candidate 1 (0.9 > 0.1), and at
        // eps=0 the mixed score is a pure (monotone) rank transform of the
        // base score, so the ORDER must match even though the raw values
        // differ (rank-normalized vs raw rate).
        assert!(base_scores[0] > base_scores[1]);
        assert!(mixed_scores[0] > mixed_scores[1]);
    }

    #[test]
    #[should_panic(expected = "nested EpsilonMix is not supported")]
    fn build_guide_refuses_a_nested_epsilon_mix() {
        // `cargo test` runs with cwd = the crate root (`pixelflow-pipeline/`),
        // unlike this binary's own `--corpus-dir`/`--*-checkpoint` defaults,
        // which assume the workspace root (the documented `cargo run`
        // invocation in this file's module doc).
        let cache = GuideCache::load(
            Path::new("data/guide_checkpoint_strict_v1.json"),
            Path::new("../docs/results/2026-09-01-train-guide-report.json"),
        );
        let policy = OrderingPolicy::EpsilonMix {
            guide: Box::new(OrderingPolicy::EpsilonMix {
                guide: Box::new(OrderingPolicy::PerRuleRate),
                numerator: 1,
                denominator: 4,
                seed: 1,
            }),
            numerator: 1,
            denominator: 2,
            seed: 2,
        };
        let _ = build_guide(&policy, &cache);
    }

    #[test]
    fn build_policies_produces_the_twelve_named_arms() {
        let policies = build_policies(6, &[(1, 4), (1, 4), (1, 2)]);
        let labels: Vec<String> = policies.iter().map(OrderingPolicy::label).collect();
        assert_eq!(labels.len(), 12);
        assert_eq!(labels[0], "unguided");
        assert_eq!(labels[1], "per-rule");
        assert_eq!(labels[2], "strict-v1");
        for seed in 1..=6u64 {
            assert!(labels.contains(&format!("random:{seed}")));
        }
        assert!(labels.contains(&"mix:strict-v1:1/4:1".to_string()));
        assert!(labels.contains(&"mix:strict-v1:1/4:2".to_string()));
        assert!(labels.contains(&"mix:strict-v1:1/2:1".to_string()));
    }

    /// Schema round-trip: every line this binary writes to `r2g_*.jsonl`
    /// must deserialize as `pixelflow_pipeline::training::r2g::R2gRecord`
    /// (the schema `train_guide_r2g` already consumes) and every line
    /// written to `r2g_trajectories_*.jsonl` as `TrajectoryRow` — caught
    /// here rather than only at the first real trainer run against a mint.
    #[test]
    fn minted_records_deserialize_as_r2g_record_and_trajectory_row() {
        use pixelflow_pipeline::training::r2g::R2gRecord;

        let (arena, root) = small_expr();
        let policies = vec![
            OrderingPolicy::Unguided,
            OrderingPolicy::Random(1),
            OrderingPolicy::Random(2),
        ];
        let cache = GuideCache::empty();
        let costs = CostModel::latency_prior();
        let rule_names: Vec<String> = {
            let probe = EGraph::with_rules(all_rules());
            (0..all_rules().len())
                .map(|i| {
                    probe
                        .rule(i)
                        .map(|r| r.name().to_string())
                        .unwrap_or_else(|| format!("<rule {i}>"))
                })
                .collect()
        };

        let mut buf: Vec<u8> = Vec::new();
        let outcome = mint_expression(
            "fixture_00000",
            &arena,
            root,
            0,
            0,
            "test",
            &policies,
            &cache,
            2_000,
            &costs,
            &rule_names,
            &mut buf,
        );
        assert!(!outcome.zero_best_excluded);
        assert!(outcome.applications_written > 0);
        assert_eq!(outcome.trajectory_rows.len(), policies.len());

        let text = String::from_utf8(buf).expect("valid UTF-8 JSONL");
        let mut checked = 0usize;
        for line in text.lines() {
            let record: R2gRecord = serde_json::from_str(line).unwrap_or_else(|e| {
                panic!("line failed to deserialize as R2gRecord: {e}\nline: {line}")
            });
            assert_eq!(record.base.expr_name, "fixture_00000");
            checked += 1;
        }
        assert_eq!(checked, outcome.applications_written);

        for row in &outcome.trajectory_rows {
            let json = serde_json::to_string(row).unwrap();
            let round_tripped: TrajectoryRow = serde_json::from_str(&json)
                .unwrap_or_else(|e| panic!("TrajectoryRow failed to round-trip: {e}"));
            assert_eq!(round_tripped.expr_name, row.expr_name);
        }
    }
}
