//! The optimizer: one entry point from a term to its optimized form.
//!
//! Saturating an e-graph and extracting from it is four steps — build the
//! graph with a rule set, saturate under a budget, extract under a cost
//! model, materialise the choice. Three production call sites did those four
//! steps independently and had already drifted: two agreed byte-for-byte, and
//! the third (the `Dwrt` expansion tier in `pixelflow-compiler`'s
//! `ir_bridge`) used a different budget than the other two. Nothing detected
//! that, because there was no place where the four steps were written down
//! together.
//!
//! [`Optimizer`] is that place. Everything the research program wants to vary
//! is a field on it that defaults to what production does today: the rule set
//! and its [`Fingerprint`](super::Fingerprint), the [`Budget`], the cost
//! model, an optional [`Reranker`](super::extract::Reranker), and an optional
//! [`Observer`].
//!
//! # Why an optional lever is safe without a correctness review
//!
//! The laws are audited in `docs/plans/2026-09-02-optimizer-api.md`. The one
//! that makes this API worth having is **L4, guide neutrality**:
//!
//! 1. Every rewrite rule preserves denotation, so every equality in the graph
//!    is a semantic equality and an e-class *is* an equivalence class (L1).
//! 2. Saturation only ever adds equalities, so any policy that orders or
//!    truncates can only make the graph hold a *subset* of the equalities an
//!    exhaustive run would hold — never a different one (L2).
//! 3. Extraction picks one node from the root's class, and by (1) every node
//!    in that class denotes the root's function.
//!
//! Therefore the extracted term denotes the same function under **any**
//! ordering policy and **any** budget. A policy changes cost and compile
//! time, never meaning — so a policy PR owes a *quality* measurement, not a
//! numeric-equivalence suite. `laws.rs` pins this rather than leaving it as
//! an argument.
//!
//! What that argument does **not** cover, and the doc comment on
//! [`Optimizer::cost`] repeats: `extract_dag` is *not* argmin under the cost
//! model. It prices a tree rather than a DAG (sharing is never charged), it
//! is a single DFS over graphs that saturation makes cyclic, and the cost it
//! reports is read before a well-foundedness repair may change the term. The
//! cost model is the objective extraction *aims at*, not a minimum it
//! attains. [`Reranker`](super::extract::Reranker) is the seam for any
//! objective that needs a guarantee — non-additive or otherwise — because it
//! scores whole extractions.

use alloc::boxed::Box;
use alloc::vec::Vec;

use pixelflow_ir::{ExprArena, ExprId, LatticeShape};

use super::cost::CostModel;
use super::extract::{ChoiceCost, Extraction, IncrementalExtractor, Reranker, choices_to_arena};
use super::graph::{EGraph, SaturationStop};
use super::node::EClassId;
#[cfg(feature = "provenance-journal")]
use super::provenance::ApplicationRecord;
use super::rules::{Fingerprint, RuleSet};

/// A sink for what saturation did.
///
/// Optional: production passes nothing and the graph records nothing, which
/// is the point — provenance recording was unconditional and had no
/// production consumer, and a production compile builds and discards a log of
/// a median 8 446 applications per kernel.
///
/// Every *credit definition* — strict, tight, output-class, return-to-go,
/// leave-one-out counterfactual — is a function of these records and belongs
/// to the crate that consumes them. A library that shipped a menu of credit
/// bounds would have to review each one; this one ships the records.
///
/// Records arrive in firing order once the run ends, not as each application
/// commits. The log is append-only and an application's record is complete
/// only after its action has run, so the two deliveries carry identical
/// content; delivering at the end keeps the observer out of the rewrite
/// dispatch path entirely.
#[cfg(feature = "provenance-journal")]
pub trait Observer {
    /// One rewrite application: which rule fired, in which round, what it
    /// minted, and whether anything actually changed.
    fn on_application(&mut self, record: &ApplicationRecord);
}

/// What stops saturation.
///
/// Every variant is deterministic: the same arena under the same budget
/// produces the same graph on any machine, at any load. Wall clock is
/// deliberately **not** a variant — see [`Optimizer::hard_ceiling`].
///
/// That is a change from what the production path used to do. The
/// [`SaturationConfig`](super::saturate::SaturationConfig) presets carry a
/// 10/50/200 ms `hard_timeout` that `saturate_with_limits` breaks on, and
/// `saturate_with_full_budget` then reports the truncated run as
/// `saturated: true`. Compiler output was therefore a function of machine
/// load, and "the same kernel compiles to the same code" was not a claim
/// anyone could make. Under [`Budget`] it is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Budget {
    /// What production does: iteration and class caps chosen from the
    /// input's node count, exactly as
    /// [`config_for_node_count`](super::saturate::config_for_node_count)
    /// picks them (≤10 nodes → 20 rounds / 500 classes; 11–50 → 50 / 2 000;
    /// 51+ → 100 / 5 000).
    Production,
    /// A fixed number of rule applications, with production's round and
    /// class caps as backstops. The budget the research arms compare under,
    /// because an application count is the one currency that means the same
    /// thing to every ordering policy.
    Applications(u64),
    /// Every limit named outright.
    Explicit {
        /// Rewrite rounds.
        iterations: usize,
        /// E-class ceiling.
        classes: usize,
        /// Rule applications, if capped.
        applications: Option<u64>,
    },
}

/// The limits [`Budget`] resolves to for a given input size.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Limits {
    /// Rewrite rounds.
    pub iterations: usize,
    /// E-class ceiling.
    pub classes: usize,
    /// Rule applications, if capped.
    pub applications: Option<u64>,
}

impl Budget {
    /// Resolve to concrete limits.
    ///
    /// `node_count` is the same rough size measure the presets have always
    /// keyed on — AST node count for the macro tier, reachable-arena node
    /// count for the runtime tier — and is ignored by the variants that name
    /// their limits.
    #[must_use]
    pub fn limits(self, node_count: usize) -> Limits {
        let preset = super::saturate::config_for_node_count(node_count);
        match self {
            Self::Production => Limits {
                iterations: preset.max_iterations,
                classes: preset.max_classes,
                // Deterministic replacement for the wall clock this used to
                // carry (docs/plans/2026-09-01-production-budget-determinism.md):
                // calibrated per tier so it binds strictly after the class
                // cap or quiescence on every kernel measured 2026-09-01.
                applications: Some(preset.max_applications),
            },
            Self::Applications(n) => Limits {
                iterations: preset.max_iterations,
                classes: preset.max_classes,
                applications: Some(n),
            },
            Self::Explicit {
                iterations,
                classes,
                applications,
            } => Limits {
                iterations,
                classes,
                applications,
            },
        }
    }
}

/// What one [`Optimizer::run`] did.
#[derive(Clone, Debug)]
pub struct OptimizerStats {
    /// Why saturation stopped — typed, so "ran out of rounds" and "converged"
    /// are not the same answer wearing the same clothes.
    pub stop: SaturationStop,
    /// Rewrite rounds completed.
    pub iterations: usize,
    /// Rule applications that fired.
    pub applications: u64,
    /// Class merges saturation performed.
    pub unions: usize,
    /// E-classes when the run ended.
    pub classes: usize,
    /// The limits this run was held to.
    pub limits: Limits,
}

/// The result of [`Optimizer::run`]: the per-e-class extraction choices, and
/// what the run did to get them.
///
/// Choices rather than an arena because the three tiers materialise
/// differently — the macro tier needs `ExtractedDAG` ref counts to place
/// let-bindings, the arena tiers need an [`ExprArena`]. [`Optimized::to_arena`]
/// is the arena half.
#[derive(Clone, Debug)]
pub struct Optimized {
    /// One node index per canonical e-class, well-founded from the root.
    pub choices: Vec<Option<usize>>,
    /// What [`Self::choices`] costs, in both the shape the extraction DP
    /// minimizes ([`ChoiceCost::tree`]) and the shape the emitted kernel pays
    /// ([`ChoiceCost::dag`]). Read from the settled choices, so it describes
    /// the term [`Self::to_arena`] materializes — including under a
    /// [`Reranker`], whose search has its own scale and never produced this
    /// number before.
    pub cost: ChoiceCost,
    /// What the run did.
    pub stats: OptimizerStats,
}

impl Optimized {
    /// Materialise the extracted DAG as an arena.
    ///
    /// `egraph` and `root` must be the ones [`Optimizer::run`] was given;
    /// the choices index that graph's classes.
    #[must_use]
    pub fn to_arena(&self, egraph: &EGraph, root: EClassId) -> (ExprArena, ExprId) {
        choices_to_arena(&Extraction::from_dp(egraph, root, self.choices.clone()))
    }
}

/// One entry point from a saturated-and-extracted term to its optimized form,
/// with every research lever an optional field that defaults to production.
///
/// ```ignore
/// let mut egraph = optimizer.egraph();          // carries the rule set
/// let root = /* insert your term */;
/// let out = optimizer.run(&mut egraph, root, node_count);
/// let (arena, arena_root) = out.to_arena(&egraph, root);
/// ```
///
/// The two-step — build the graph, then run — is not ceremony: the three
/// tiers insert their terms differently (an AST through `EGraphContext`, an
/// arena through `add_arena`), and that boundary is genuinely theirs. What
/// they must not each decide for themselves is the rule set, the budget, the
/// cost model, and the extractor, which is exactly what this type owns.
pub struct Optimizer {
    rules: RuleSet,
    budget: Budget,
    cost: CostModel,
    shape: LatticeShape,
    rerank: Option<Box<dyn Reranker>>,
    #[cfg(feature = "provenance-journal")]
    observer: Option<Box<dyn Observer>>,
    hard_ceiling: HardCeiling,
}

/// [`Optimizer::hard_ceiling`]'s resolved policy.
///
/// A fixed [`core::time::Duration`] isn't enough on its own:
/// [`Optimizer::production`]'s default ceiling must scale with the tier
/// [`Budget::Production`] resolves to (blitz/rapid/classical carry different
/// `safety_ceiling`s), which is only known once `node_count` reaches
/// [`Optimizer::run`] — not at construction time, when
/// `.hard_ceiling(d)`'s fixed override *is* already known. `None` stays a
/// third state (rather than `Fixed` with a sentinel) for non-production
/// configurations that want no assertion at all.
#[derive(Clone, Copy, Debug)]
enum HardCeiling {
    /// No wall-clock assertion.
    None,
    /// A caller-chosen duration, regardless of tier. Set by
    /// [`Optimizer::hard_ceiling`]; exempt from
    /// `PIXELFLOW_SATURATION_CEILING_MS` — a measurement harness that asks
    /// for a specific ceiling gets exactly that ceiling.
    Fixed(core::time::Duration),
    /// [`SaturationConfig::safety_ceiling`](super::saturate::SaturationConfig::safety_ceiling)
    /// for the tier `node_count` resolves to, subject to
    /// `PIXELFLOW_SATURATION_CEILING_MS`. [`Optimizer::production`]'s
    /// default.
    Tiered,
}

/// The three things `PIXELFLOW_SATURATION_CEILING_MS` can ask for. See
/// [`tiered_ceiling_override`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CeilingOverride {
    /// The variable is unset: use the tier's own `safety_ceiling`.
    UseTierDefault,
    /// A positive millisecond count: this ceiling, at every tier.
    Fixed(core::time::Duration),
    /// `0` or `off`: no ceiling at all.
    Disabled,
}

/// `PIXELFLOW_SATURATION_CEILING_MS`, read fresh on every call (cheap — one
/// env lookup) rather than cached once: a diagnostic override a test can
/// flip between runs in-process would otherwise need the process restarted
/// to take effect. Read at the same two places `PIXELFLOW_NNUE_WEIGHTS` is:
/// proc-macro expansion time for the macro tier, process start (in practice,
/// per-call — this crate has no init hook) for the runtime tier.
///
/// | value | effect |
/// |---|---|
/// | unset | the tier default |
/// | a positive integer | that many milliseconds, at every tier |
/// | `0` or `off` (case-insensitive) | ceiling disabled |
/// | anything else | panic, quoting the offending value — no silent failures |
///
/// This variable can only change *whether [`Optimizer::run`] panics*, never
/// what it computes: it is read after saturation and extraction have already
/// produced their result, purely to decide whether to assert on the elapsed
/// time. That is the property that makes it safe to leave permanently
/// available rather than a debug-only cfg.
fn tiered_ceiling_override() -> CeilingOverride {
    const VAR: &str = "PIXELFLOW_SATURATION_CEILING_MS";
    let Ok(raw) = std::env::var(VAR) else {
        return CeilingOverride::UseTierDefault;
    };
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("off") {
        return CeilingOverride::Disabled;
    }
    match trimmed.parse::<u64>() {
        Ok(0) => CeilingOverride::Disabled,
        Ok(ms) => CeilingOverride::Fixed(core::time::Duration::from_millis(ms)),
        Err(_) => panic!(
            "{VAR}={raw:?} is not unset, a positive integer, `0`, or `off`. This variable can \
             only change whether a saturation run's safety ceiling panics, never which kernel \
             it produces — so an unparsable value fails loudly rather than picking a default."
        ),
    }
}

/// How many alternatives per e-class the reranking search evaluates. Matches
/// the value the swap-refinement search shipped with.
const RERANK_TOP_K: usize = 4;

impl Optimizer {
    /// The production configuration: the production rule set,
    /// [`Budget::Production`], [`CostModel::latency_prior`], no lattice
    /// (everything weighted by one), no reranker, no observer.
    #[must_use]
    pub fn production() -> Self {
        Self {
            rules: RuleSet::production(),
            budget: Budget::Production,
            cost: CostModel::latency_prior(),
            shape: LatticeShape::POINT,
            rerank: None,
            #[cfg(feature = "provenance-journal")]
            observer: None,
            // The tier's own `SaturationConfig::safety_ceiling`, resolved
            // from `node_count` at `run()` time — every production call site
            // gets a fail-loud ceiling without having to remember to ask for
            // one. See `HardCeiling::Tiered`.
            hard_ceiling: HardCeiling::Tiered,
        }
    }

    /// Price extraction against the lattice this kernel will run over: a
    /// node's cost is multiplied by how many times that lattice evaluates it,
    /// so extraction chooses the factorization rather than leaving it to a
    /// later hoisting pass.
    ///
    /// The default, [`LatticeShape::POINT`], weights everything by one — what
    /// a caller with no lattice (the `kernel!` macros, expanding before any
    /// bake) should use.
    #[must_use]
    pub fn for_lattice(mut self, shape: LatticeShape) -> Self {
        self.shape = shape;
        self
    }

    /// Hold saturation to a different budget.
    #[must_use]
    pub fn budget(mut self, budget: Budget) -> Self {
        self.budget = budget;
        self
    }

    /// Use a different rule set. Research harnesses that compare orderings;
    /// production names only [`RuleSet::production`].
    #[must_use]
    pub fn rules(mut self, rules: RuleSet) -> Self {
        self.rules = rules;
        self
    }

    /// Refine the extraction with a [`Reranker`] over whole extractions.
    ///
    /// This is the seam for a cost that additive DP cannot express — the
    /// schedule-cost residual — and, per the module docs, for any objective
    /// that wants an argmin guarantee, since `extract_dag` does not provide
    /// one.
    #[must_use]
    pub fn rerank(mut self, rerank: Option<Box<dyn Reranker>>) -> Self {
        self.rerank = rerank;
        self
    }

    /// Record what saturation did, and hand it to `observer` when the run
    /// ends. Production passes `None` and nothing is recorded.
    #[cfg(feature = "provenance-journal")]
    #[must_use]
    pub fn observe(mut self, observer: Option<Box<dyn Observer>>) -> Self {
        self.observer = observer;
        self
    }

    /// A fail-loud wall-clock ceiling, fixed regardless of tier. **Not** a
    /// budget dimension: exceeding it is a bug in the budget, so it panics
    /// rather than silently truncating and reporting success, which is what
    /// the old `hard_timeout` did.
    ///
    /// [`Optimizer::production`]'s default is *not* "none" — it is
    /// [`config_for_node_count`](super::saturate::config_for_node_count)'s
    /// tier-appropriate `safety_ceiling`, resolved once `node_count` is
    /// known. Call this only to override that with a fixed value regardless
    /// of tier (measurement harnesses that vary the budget independently of
    /// input size); the override is exempt from
    /// `PIXELFLOW_SATURATION_CEILING_MS`, which affects only the tiered
    /// default.
    #[must_use]
    pub fn hard_ceiling(mut self, d: core::time::Duration) -> Self {
        self.hard_ceiling = HardCeiling::Fixed(d);
        self
    }

    /// Disable the wall-clock ceiling assertion entirely. Exists for
    /// configurations (research budgets sweeping deliberately extreme
    /// inputs) that want no ceiling at all rather than a very generous one.
    #[must_use]
    pub fn no_ceiling(mut self) -> Self {
        self.hard_ceiling = HardCeiling::None;
        self
    }

    /// The cost model extraction aims at.
    ///
    /// Read "aims at" literally: `extract_dag` is a heuristic, not an argmin.
    /// It sums children's costs, which prices a tree — a shared
    /// subexpression is charged once per use in the objective and once in
    /// total in the emitted code. It is a single DFS, so on the cyclic graphs
    /// saturation always produces (commutativity alone makes cycles) a class
    /// whose child is still on the stack is scored at a cycle sentinel and
    /// never revisited. Use [`Optimizer::rerank`] for an objective that needs
    /// a guarantee.
    #[must_use]
    pub fn cost(mut self, cost: CostModel) -> Self {
        self.cost = cost;
        self
    }

    /// The digest of this configuration: today the rule set's content and
    /// order.
    ///
    /// A cache that keys on the input expression alone would serve one
    /// configuration's code to another the moment two configurations coexist
    /// in a process — a warm-up compiled at one budget and steady state at
    /// another, or some kernels reranked and some not.
    #[must_use]
    pub fn fingerprint(&self) -> Fingerprint {
        self.rules.fingerprint()
    }

    /// The rule set, for a caller that has to name a rule this run applied.
    #[must_use]
    pub fn rule_set(&self) -> &RuleSet {
        &self.rules
    }

    /// An empty e-graph carrying this optimizer's rule set. Insert your term
    /// into it and hand it to [`Optimizer::run`].
    #[must_use]
    pub fn egraph(&self) -> EGraph {
        let (rules, ids) = self.rules.shared();
        EGraph::with_shared_rules(rules, ids)
    }

    /// Saturate `egraph` from `root` under this configuration, and extract.
    ///
    /// `node_count` is the rough size measure [`Budget::Production`] picks
    /// its preset from — the same one the presets have always keyed on.
    ///
    /// # Panics
    ///
    /// Panics if a [`hard_ceiling`](Optimizer::hard_ceiling) was set and the
    /// run exceeded it. That is the intended behavior: a ceiling is an
    /// assertion about the budget, and a silently truncated optimization that
    /// reports success is the failure mode this API exists to remove.
    pub fn run(&mut self, egraph: &mut EGraph, root: EClassId, node_count: usize) -> Optimized {
        let limits = self.budget.limits(node_count);
        let started = std::time::Instant::now();

        #[cfg(feature = "provenance-journal")]
        egraph.set_provenance_recording(self.observer.is_some());
        let saturation =
            egraph.saturate_budgeted(limits.iterations, limits.classes, limits.applications);

        let ceiling = match self.hard_ceiling {
            HardCeiling::None => None,
            HardCeiling::Fixed(d) => Some(d),
            HardCeiling::Tiered => match tiered_ceiling_override() {
                CeilingOverride::UseTierDefault => {
                    Some(super::saturate::config_for_node_count(node_count).safety_ceiling)
                }
                CeilingOverride::Fixed(d) => Some(d),
                CeilingOverride::Disabled => None,
            },
        };
        if let Some(ceiling) = ceiling {
            let elapsed = started.elapsed();
            let applications = egraph.application_count();
            assert!(
                elapsed <= ceiling,
                "Optimizer::run exceeded its safety ceiling: {elapsed:?} > {ceiling:?} \
                 (tier {tier}, {node_count} nodes, {applications} applications reached, \
                 stop: {stop:?}). A ceiling is an assertion about the budget, not a budget \
                 dimension — either the budget is wrong for this input or the machine is \
                 unusually slow. Override with PIXELFLOW_SATURATION_CEILING_MS (milliseconds; \
                 `0` or `off` disables it) only for diagnosis — it can change whether this \
                 panics, never which kernel is emitted.",
                tier = tier_name(node_count),
                stop = saturation.stop,
            );
        }

        #[cfg(feature = "provenance-journal")]
        if let Some(observer) = self.observer.as_mut() {
            for (_id, record) in egraph.provenance().applications() {
                observer.on_application(record);
            }
        }

        // Both arms report the cost of the choices they RETURN — the DP's
        // own table is read before `repair_choices_well_founded` rewrites
        // picks and so can name a different term (#1111), and the reranker's
        // search score is on its own scale entirely.
        let (choices, cost) = match self.rerank.as_ref() {
            Some(reranker) => {
                let choices = IncrementalExtractor::new(reranker.as_ref(), RERANK_TOP_K)
                    .extract_choices_only(egraph, root)
                    .1
                    .into_choices();
                let cost =
                    super::extract::cost_of_choices(egraph, root, &choices, &self.cost, self.shape);
                (choices, cost)
            }
            None => {
                let dag = super::extract::extract_dag_scoped(egraph, root, &self.cost, self.shape);
                let cost = dag.cost();
                (dag.choices, cost)
            }
        };

        Optimized {
            choices,
            cost,
            stats: OptimizerStats {
                stop: saturation.stop,
                iterations: saturation.iterations,
                applications: egraph.application_count(),
                unions: saturation.total_unions,
                classes: egraph.num_classes(),
                limits,
            },
        }
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::production()
    }
}

/// The tier name matching [`config_for_node_count`](super::saturate::config_for_node_count)'s
/// ranges, for the safety-ceiling panic message — named rather than left as
/// a bare node count so the message reads the way the calibration doc does.
fn tier_name(node_count: usize) -> &'static str {
    match node_count {
        0..=10 => "blitz",
        11..=50 => "rapid",
        _ => "classical",
    }
}
