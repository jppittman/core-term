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
//! and its [`Fingerprint`], the [`Budget`], the cost
//! model, an optional `Reranker`, and an optional
//! [`Observer`].
//!
//! # Why an optional lever is safe without a correctness review
//!
//! The laws are audited in `docs/archive/plans/2026-09-02-optimizer-api.md`. The one
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
//! is a heuristic over graphs that saturation makes cyclic, and the cost it
//! reports is that of the returned choices. The cost model is the objective
//! extraction *aims at*, not a minimum it attains. `Reranker` is the seam for any
//! objective that needs a guarantee — non-additive or otherwise — because it
//! scores whole extractions.

use alloc::boxed::Box;
use alloc::vec::Vec;

use pixelflow_ir::{ExprArena, ExprId, LatticeShape};

use super::and_or::{Frozen, Problem};
use super::cost::CostModel;
use super::extract::{
    ChoiceCost, ExtractionReport, IncrementalExtractor, Pricing, Reranker, Selection,
};
use super::graph::{ApplicationMask, EGraph, Limits, Saturation, SaturationStats, SaturationStop};
use super::guided::GuidedEpisode;
use super::node::EClassId;
#[cfg(feature = "provenance-journal")]
use super::provenance::ApplicationRecord;
use super::rules::{Fingerprint, RuleSet};
use super::saturate::InputSize;
use crate::nnue::guide::SaturationGuide;

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

/// The observer for a caller that wants the journal **kept on the graph**
/// rather than streamed: it consumes nothing, and its presence is what turns
/// recording on.
///
/// The hindsight-label harnesses (`EpisodeLabels::compute_strict`,
/// `derivation_ancestors`, the return-to-go minter) read the whole journal
/// off [`EGraph::provenance`] after the run — as a graph, in an order the
/// stream cannot give them. Each writing its own do-nothing `Observer` would
/// be several copies of the same four lines, and the thing they actually
/// mean — "record; I will read it later" — deserves a name.
///
/// Note that [`Optimizer::run`] replays the journal to its observer at the
/// end of **every** call, so a resumed sequence of budgeted runs delivers
/// earlier records again. That is harmless here (this observer consumes
/// nothing) and is exactly why a harness reading the journal wants this
/// rather than a counting observer.
#[cfg(feature = "provenance-journal")]
#[derive(Debug, Default, Clone, Copy)]
pub struct KeepJournal;

#[cfg(feature = "provenance-journal")]
impl Observer for KeepJournal {
    fn on_application(&mut self, _record: &ApplicationRecord) {}
}

/// A saturation configuration, as a value a cache can key on: the rule set's
/// [`Fingerprint`] and the [`Budget`], and nothing else — see
/// [`Optimizer::saturation_key`] for what is in and why.
///
/// A second type beside [`Fingerprint`], not a wider one: a `Fingerprint`
/// names a rule vocabulary and is written into checkpoints, where a guide
/// records which vocabulary it was trained against, so it must keep meaning
/// that. Built only by [`Optimizer::saturation_key`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SaturationKey(Vec<u8>);

impl SaturationKey {
    const PRODUCTION: u8 = 0;
    const APPLICATIONS: u8 = 1;
    const EXPLICIT: u8 = 2;
    const UNCAPPED: u8 = 0;
    const CAPPED: u8 = 1;

    fn of(rules: Fingerprint, budget: Budget) -> Self {
        fn push_count(bytes: &mut Vec<u8>, count: u64) {
            bytes.extend_from_slice(&count.to_le_bytes());
        }
        let mut bytes = rules.to_bytes().to_vec();
        match budget {
            Budget::Production => bytes.push(Self::PRODUCTION),
            Budget::Applications(n) => {
                bytes.push(Self::APPLICATIONS);
                push_count(&mut bytes, n);
            }
            Budget::Explicit {
                iterations,
                minted,
                applications,
            } => {
                bytes.push(Self::EXPLICIT);
                push_count(&mut bytes, iterations as u64);
                push_count(&mut bytes, minted as u64);
                match applications {
                    None => bytes.push(Self::UNCAPPED),
                    Some(n) => {
                        bytes.push(Self::CAPPED);
                        push_count(&mut bytes, n);
                    }
                }
            }
        }
        Self(bytes)
    }
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
    /// input's sizes, exactly as
    /// [`config_for_input`](super::saturate::config_for_input) picks them
    /// (≤10 classes → 20 rounds / 500 ids; 11–50 → 50 / 2 000; 51+ →
    /// 100 rounds and a class cap of
    /// [`CLASSICAL_CLASSES_PER_INSERTED_CLASS`](super::saturate::CLASSICAL_CLASSES_PER_INSERTED_CLASS)
    /// per inserted class, clamped to 5 000..=50 000).
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
        /// The most e-class ids the graph may mint.
        minted: usize,
        /// Rule applications, if capped.
        applications: Option<u64>,
    },
}

impl Budget {
    /// Resolve to concrete limits.
    ///
    /// `input` is the class count the term inserted to, which picks the tier
    /// and sizes the classical cap; [`Budget::Explicit`] ignores it.
    #[must_use]
    pub fn limits(self, input: InputSize) -> Limits {
        let preset = super::saturate::config_for_input(input);
        match self {
            Self::Production => Limits {
                iterations: preset.max_iterations,
                minted: preset.max_minted,
                // Deterministic replacement for the wall clock this used to
                // carry (docs/archive/plans/2026-09-01-production-budget-determinism.md):
                // calibrated per tier so it binds strictly after the class
                // cap or quiescence on every kernel measured 2026-09-01.
                applications: Some(preset.max_applications),
            },
            Self::Applications(n) => Limits {
                iterations: preset.max_iterations,
                minted: preset.max_minted,
                applications: Some(n),
            },
            Self::Explicit {
                iterations,
                minted,
                applications,
            } => Limits {
                iterations,
                minted,
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

/// The result of [`Optimizer::run`]: the term the extraction selected, and
/// what the run did to get it.
///
/// The term rather than an arena because the saturated graph is what a tier
/// materializes it against ([`Optimized::to_arena`]), and the macro tier and
/// the runtime tier link what comes out differently.
#[derive(Clone, Debug)]
pub struct Optimized {
    /// The node index chosen for each canonical e-class, by class id. Under
    /// the DP it is the settlement's record, a superset of the term (every
    /// class the settlement drained, not only those the root reaches);
    /// under a `Reranker`, where no settlement ran, it is the term's own
    /// reach. [`Self::to_arena`] materializes the term, not this field;
    /// editing it does not change what comes out.
    pub choices: Vec<Option<usize>>,
    /// The term: the classes the root reaches through [`Self::choices`],
    /// well-founded and in the order it materializes in.
    selection: Selection,
    /// What the term costs, in both the shape the extraction DP minimizes
    /// ([`ChoiceCost::tree`]) and the shape the emitted kernel pays
    /// ([`ChoiceCost::dag`]): the price of the selected term, so it holds
    /// under a `Reranker` too, whose search has its own scale and never
    /// produced this number before. It describes the term [`Self::to_arena`]
    /// emits: nothing is repaired after extraction (a shift's count is a
    /// parameter of its node, never a class a choice could get wrong).
    pub cost: ChoiceCost,
    /// Which objective chose [`Self::choices`], and what the sharing-aware
    /// pass cost — `ExtractionObjective::External` under a `Reranker`.
    /// A cost quoted without this is a cost from an unknown extractor.
    pub extraction: ExtractionReport,
    /// What the run did.
    pub stats: OptimizerStats,
}

impl Optimized {
    /// Materialise the extracted DAG as an arena.
    ///
    /// `egraph` and `root` must be the ones [`Optimizer::run`] was given: the
    /// term indexes that graph's classes and nodes.
    ///
    /// # Panics
    ///
    /// If `root` is not the class the term was extracted from, or `egraph`
    /// no longer holds the nodes it chose.
    #[must_use]
    pub fn to_arena(&self, egraph: &EGraph, root: EClassId) -> (ExprArena, ExprId) {
        assert_eq!(
            egraph.find(root),
            self.selection.root(),
            "Optimized::to_arena: the term was extracted from another class than `root`"
        );
        self.selection.to_arena(egraph)
    }
}

/// One entry point from a saturated-and-extracted term to its optimized form,
/// with every research lever an optional field that defaults to production.
///
/// ```ignore
/// let mut egraph = optimizer.egraph();          // carries the rule set
/// let root = /* insert your term */;
/// let out = optimizer.run(&mut egraph, root);
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
    guide: Option<Box<dyn SaturationGuide>>,
    mask: Option<ApplicationMask>,
    /// The guided episode's carried state — dedup set, feature constant,
    /// rule-embedding cache — created on the first guided [`Self::run`] and
    /// reused by later ones, so a caller stepping an anytime curve through
    /// several application budgets on ONE e-graph sees one continuous guided
    /// run. `None` whenever `guide` is `None`.
    episode: Option<GuidedEpisode>,
    #[cfg(feature = "provenance-journal")]
    observer: Option<Box<dyn Observer>>,
    hard_ceiling: HardCeiling,
}

/// [`Optimizer::hard_ceiling`]'s resolved policy.
///
/// A fixed [`core::time::Duration`] isn't enough on its own:
/// [`Optimizer::production`]'s default ceiling must scale with the tier
/// [`Budget::Production`] resolves to (blitz/rapid/classical carry different
/// `safety_ceiling`s), which is only known once the input has been inserted and
/// [`Optimizer::run`] sizes it — not at construction time, when
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
    /// for the tier the input's size resolves to, subject to
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
            guide: None,
            mask: None,
            episode: None,
            #[cfg(feature = "provenance-journal")]
            observer: None,
            // The tier's own `SaturationConfig::safety_ceiling`, resolved
            // from the inserted size at `run()` time — every production call site
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

    /// Refine the extraction with a `Reranker` over whole extractions.
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

    /// Order the rewrite candidates saturation applies.
    ///
    /// A [`SaturationGuide`] supplies
    /// **order only** — it never constructs a rewrite action, never unions,
    /// never sees a mutable e-graph. That is what makes law **L4** (see this
    /// module's docs) hold without a numeric-equivalence suite: every
    /// ordering leaves the graph holding a subset of the equalities an
    /// exhaustive run would hold, and every node of the root's class denotes
    /// the root's function, so the extracted term means the same thing under
    /// any guide. A guide owes a *quality* measurement, not a correctness
    /// argument.
    ///
    /// [`Self::production`] leaves this `None` and takes the unguided sweep,
    /// byte for byte. `Budget::Applications` is the budget a guided arm
    /// should be compared under: an application count is the one currency
    /// that means the same thing to every ordering policy.
    ///
    /// Setting a guide resets the carried episode state, so one optimizer
    /// can be re-pointed at a fresh guide without inheriting the previous
    /// one's dedup set.
    #[must_use]
    pub fn guide(mut self, guide: Option<Box<dyn SaturationGuide>>) -> Self {
        self.guide = guide;
        self.episode = None;
        self
    }

    /// Withhold one named rewrite application (and, under
    /// [`MaskScope::AllMatchingCandidate`](super::graph::MaskScope::AllMatchingCandidate),
    /// every later application that would re-derive the same thing), so a
    /// harness can measure what that application was worth:
    /// `Δ_a = R(τ\a,B) − R(τ,B)`
    /// (docs/archive/plans/2026-09-01-guide-return-to-go.md §4.1).
    ///
    /// A field, not a second saturation entry point, for the same reason
    /// [`Self::rerank`] and [`Self::guide`] are fields: withholding an
    /// application is a *policy* over the same one loop, and #1085/#1108
    /// exist to keep there being one loop. Like every other lever it is
    /// covered by L4 — a withheld application can only leave the graph
    /// holding fewer equalities, never different ones.
    ///
    /// Read [`Saturation::last_replay_mask_skips`] afterwards rather than
    /// assuming the mask fired: under the confluence-aware scope it can fire
    /// many times, and if its ordinal was never reached it fires not at all.
    #[must_use]
    pub fn mask(mut self, mask: Option<ApplicationMask>) -> Self {
        self.mask = mask;
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
    /// [`config_for_input`](super::saturate::config_for_input)'s
    /// tier-appropriate `safety_ceiling`, resolved once the input's size is
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
    /// total in the emitted code. Extraction settles classes in cost order,
    /// so a class is admitted only after the children of the candidate it
    /// settles on, but the sum is still taken over a tree. Use
    /// [`Optimizer::rerank`] for an objective that needs a guarantee.
    #[must_use]
    pub fn cost(mut self, cost: CostModel) -> Self {
        self.cost = cost;
        self
    }

    /// Everything about this configuration that decides the graph a
    /// saturation leaves behind, or `None` when the configuration cannot be
    /// named.
    ///
    /// A cache that keys on the input expression alone would serve one
    /// configuration's graph to another the moment two configurations coexist
    /// in a process — a warm-up at one budget and steady state at another.
    /// The key covers the rule set (content and order: the sweep is
    /// order-sensitive under a budget) and the [`Budget`]; the budget's
    /// limits then follow from the class count the input inserts to. That
    /// count is a function of the term as inserted, which the cache's
    /// canonical key fixes only up to commutative spelling: the key is
    /// order-free and the e-graph memo is not, so `(X + Y) * (Y + X)` keys
    /// one subterm and inserts two classes. It leaves out what only the extraction reads — the cost model,
    /// the lattice, the reranker — and what changes no result: the observer
    /// and the wall-clock ceiling.
    ///
    /// It does not cover the [`Vocabulary`](super::Vocabulary): which terms
    /// `insert` declines is a property of the call site's tier, not of the
    /// optimizer, and the one cache keyed on this (the runtime tier's) always
    /// inserts under `Vocabulary::Runtime`. A second tier sharing a cache
    /// would have to add it.
    ///
    /// `None` for a guided or masked run. A [`SaturationGuide`] has no
    /// identity to key on and reorders the sweep, and a replay mask withholds
    /// an application: neither run is a function of the input and the rules,
    /// so no graph it leaves may be served to another caller.
    #[must_use]
    pub fn saturation_key(&self) -> Option<SaturationKey> {
        (self.guide.is_none() && self.mask.is_none())
            .then(|| SaturationKey::of(self.rules.fingerprint(), self.budget))
    }

    /// The limits this optimizer's budget resolves to for an input of these
    /// sizes — the *environment* an [`anytime`](super::anytime) curve holds
    /// fixed while it varies only the application dimension.
    #[must_use]
    pub fn limits_for(&self, input: InputSize) -> Limits {
        self.budget.limits(input)
    }

    /// How many distinct candidate keys the carried guided episode has
    /// resolved so far, or `None` when no guide is set.
    ///
    /// The guided loop's own unit of work: a key is scored once per episode
    /// and never again, so this is "candidates this Guide was actually asked
    /// to rank" — the denominator a Guide-overhead measurement needs, and
    /// not derivable from the application count (a key can be scored and
    /// then fail to fire).
    #[must_use]
    pub fn guided_keys_seen(&self) -> Option<usize> {
        self.episode.as_ref().map(GuidedEpisode::seen_key_count)
    }

    /// The rule set, for a caller that has to name a rule this run applied.
    #[must_use]
    pub fn rule_set(&self) -> &RuleSet {
        &self.rules
    }

    /// An empty e-graph under this optimizer's rule set. Insert your term
    /// into it and hand it to [`Optimizer::run`].
    ///
    /// The graph keeps a mutation journal exactly when an
    /// [`Observer`] is attached: the journal is chosen here, when the graph
    /// is built, because a log cannot be turned on halfway through a run
    /// without misattributing what happened before it. Production attaches
    /// none, so a production graph records nothing.
    #[must_use]
    pub fn egraph(&self) -> Saturation {
        let saturation = Saturation::over(self.rules.clone());
        if self.has_observer() {
            return saturation;
        }
        saturation.without_journal()
    }

    /// Whether anything is attached to hear what saturation did.
    fn has_observer(&self) -> bool {
        #[cfg(feature = "provenance-journal")]
        return self.observer.is_some();
        #[cfg(not(feature = "provenance-journal"))]
        return false;
    }

    /// Saturate `egraph` from `root` under this configuration, and extract.
    ///
    /// [`Budget::Production`] picks its tier, and sizes the classical cap,
    /// from `egraph`'s class count on entry: the input has been inserted and
    /// nothing rewritten yet, so that count is the hash-consed input
    /// ([`InputSize::classes`]). To resume a graph already rewritten, name the
    /// limits outright with [`Budget::Explicit`].
    ///
    /// # Panics
    ///
    /// Panics if a budget keyed on the input's size meets a graph that has
    /// already been rewritten (its class count is then the grown graph's, not
    /// the input's), or if a [`hard_ceiling`](Optimizer::hard_ceiling) was set
    /// and the run exceeded it. That is the intended behavior: a ceiling is an
    /// assertion about the budget, and a silently truncated optimization that
    /// reports success is the failure mode this API exists to remove.
    pub fn run(&mut self, egraph: &mut Saturation, root: EClassId) -> Optimized {
        let input = self.inserted_size(egraph);
        self.run_bounded(egraph, root, self.budget.limits(input), input)
    }

    /// [`Self::run`] with the limits named outright — the seam
    /// [`super::anytime`] steps a curve on, where each step's budget is the
    /// *gap* to the next checkpoint and `input` is only still needed to
    /// name the tier the ceiling belongs to.
    pub(crate) fn run_bounded(
        &mut self,
        egraph: &mut Saturation,
        root: EClassId,
        limits: Limits,
        input: InputSize,
    ) -> Optimized {
        let stats = self.saturate_bounded(egraph, limits, input);
        self.extract(egraph, root, stats)
    }

    /// Saturate `egraph` under this configuration and stop there: the half of
    /// [`Self::run`] that does not depend on the lattice. The rules, the
    /// budget (a function of the input's size) and the graph they produce are
    /// the same at every extent, so what this leaves in `egraph` is a graph
    /// every extent can be extracted from ([`Self::extract`]) — which is what
    /// a cache holding one saturation per structure keeps.
    ///
    /// # Panics
    ///
    /// As [`Self::run`]: if the graph was already rewritten under a budget
    /// keyed on the input's size, or a [`hard_ceiling`](Optimizer::hard_ceiling)
    /// was set and the saturation exceeded it.
    pub fn saturate_term(&mut self, egraph: &mut Saturation) -> OptimizerStats {
        let input = self.inserted_size(egraph);
        self.saturate_bounded(egraph, self.budget.limits(input), input)
    }

    /// Extract from a saturated `egraph` under this configuration's cost model
    /// and lattice: the half of [`Self::run`] that depends on the shape.
    /// [`Self::for_lattice`] prices a node by how many times the loop nest
    /// evaluates it, so the term that is cheapest at one extent need not be
    /// at another — which is why this runs once per shape over a graph that
    /// was saturated once. `stats` is what that saturation reported, carried
    /// through so an [`Optimized`] reads the same whichever way it was made.
    #[must_use]
    pub fn extract(&self, egraph: &EGraph, root: EClassId, stats: OptimizerStats) -> Optimized {
        // A reranked extraction reads the e-graph and no extraction problem,
        // so none is built for it.
        match self.rerank.as_ref() {
            Some(reranker) => self.reranked(reranker.as_ref(), egraph, root, stats),
            None => self.extract_over(Frozen::new(egraph, root).problem(), stats),
        }
    }

    /// [`Self::extract`] over an extraction problem already read from its
    /// e-graph: a cache that holds one saturated graph for many shapes holds
    /// the problem beside it ([`Frozen`]), so each shape adds its weights and
    /// its settlement and nothing else.
    #[must_use]
    pub(crate) fn extract_over(&self, problem: Problem<'_>, stats: OptimizerStats) -> Optimized {
        if let Some(reranker) = self.rerank.as_ref() {
            return self.reranked(reranker.as_ref(), problem.egraph(), problem.root(), stats);
        }
        // Both the cost and the term are those of the settlement — the DP's
        // table is priced after settlement rather than read off the DP
        // (#1111) — and the term is well-founded by construction (settled in
        // cost order).
        let dag = super::extract::extract_dag_over(
            problem,
            &Pricing {
                costs: &self.cost,
                shape: self.shape,
            },
        );
        let cost = dag.cost();
        Optimized {
            choices: dag.choices,
            selection: dag.selection,
            cost,
            extraction: dag.report,
            stats,
        }
    }

    /// The reranker's extraction. Its search has its own scale entirely, so
    /// its cost is read from the term it returns, as the DP's is (#1111); the
    /// term is its [`Extraction`](super::extract::Extraction)'s, well-founded
    /// by construction.
    fn reranked(
        &self,
        reranker: &dyn Reranker,
        egraph: &EGraph,
        root: EClassId,
        stats: OptimizerStats,
    ) -> Optimized {
        let (_, extraction) =
            IncrementalExtractor::new(reranker, RERANK_TOP_K).extract_choices_only(egraph, root);
        let selection = extraction.to_selection();
        let cost = selection.price(
            egraph,
            &Pricing {
                costs: &self.cost,
                shape: self.shape,
            },
        );
        Optimized {
            choices: selection.to_map(egraph.num_classes()),
            selection,
            cost,
            extraction: ExtractionReport::external(),
            stats,
        }
    }

    /// The size of the input `egraph` was handed, read off its class count.
    ///
    /// Only the first call on a graph can read it: once rules have fired the
    /// count includes what they minted. A budget that keys on the size
    /// therefore refuses a graph already rewritten, rather than drift to a
    /// larger tier on each call.
    fn inserted_size(&self, egraph: &Saturation) -> InputSize {
        let keyed_on_size = matches!(self.budget, Budget::Production | Budget::Applications(_));
        assert!(
            !(keyed_on_size && egraph.application_count() > 0),
            "this budget is keyed on the inserted class count, and this graph has already been \
             rewritten ({} applications): resume it under `Budget::Explicit`",
            egraph.application_count(),
        );
        InputSize {
            classes: egraph.num_classes(),
        }
    }

    fn saturate_bounded(
        &mut self,
        egraph: &mut Saturation,
        limits: Limits,
        input: InputSize,
    ) -> OptimizerStats {
        // Before the run, so a long research run does not discover at its end
        // that its graph never kept the journal the observer was to hear.
        #[cfg(feature = "provenance-journal")]
        assert!(
            self.observer.is_none() || egraph.provenance().is_some(),
            "an observer needs a graph that keeps a journal: build it with \
             `Optimizer::egraph` on the optimizer that observes",
        );
        let started = std::time::Instant::now();

        // A fresh clone per run: the confluence-aware scope mutates its
        // captured key as the run goes, and one optimizer replaying two
        // expressions must not carry the first one's capture into the
        // second.
        egraph.set_replay_mask(self.mask.clone());
        let saturation = self.saturate(egraph, limits);

        // Name and budget from one lookup, so the panic below cannot name a
        // tier other than the one whose ceiling it is asserting.
        let (tier, tier_config) = super::saturate::tier_for_input(input);
        let ceiling = match self.hard_ceiling {
            HardCeiling::None => None,
            HardCeiling::Fixed(d) => Some(d),
            HardCeiling::Tiered => match tiered_ceiling_override() {
                CeilingOverride::UseTierDefault => Some(tier_config.safety_ceiling),
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
                 (tier {tier}, {classes} inserted classes, {applications} \
                 applications reached, \
                 stop: {stop:?}). A ceiling is an assertion about the budget, not a budget \
                 dimension — either the budget is wrong for this input or the machine is \
                 unusually slow. Override with PIXELFLOW_SATURATION_CEILING_MS (milliseconds; \
                 `0` or `off` disables it) only for diagnosis — it can change whether this \
                 panics, never which kernel is emitted.",
                stop = saturation.stop,
                classes = input.classes,
            );
        }

        #[cfg(feature = "provenance-journal")]
        if let Some(observer) = self.observer.as_mut() {
            let journal = egraph
                .provenance()
                .expect("checked on entry: the graph keeps a journal");
            for (_id, record) in journal.applications() {
                observer.on_application(record);
            }
        }

        OptimizerStats {
            stop: saturation.stop,
            iterations: saturation.iterations,
            applications: egraph.application_count(),
            unions: saturation.total_unions,
            classes: egraph.num_classes(),
            limits,
        }
    }
}

impl Optimizer {
    /// The one place this type decides *how* saturation runs: the unguided
    /// sweep, or the guided ordering when a
    /// [`SaturationGuide`](crate::nnue::guide::SaturationGuide) is set.
    ///
    /// Both arms are held to the same [`Limits`] and report the same
    /// [`SaturationStats`], and neither can report
    /// [`SaturationStop::Timeout`] — the guided loop takes no clock either.
    /// Every mutation the guided arm makes goes through
    /// `Saturation::apply_single_rule`, i.e. through the same
    /// `commit` the sweep uses, which is what keeps the
    /// application counter (the budget's denominator) meaning one thing.
    fn saturate(&mut self, egraph: &mut Saturation, limits: Limits) -> SaturationStats {
        let Some(guide) = self.guide.as_ref() else {
            return egraph.saturate_budgeted(limits);
        };
        self.episode
            .get_or_insert_with(GuidedEpisode::default)
            .advance(egraph, guide.as_ref(), limits)
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::production()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue::guide::CandidateSummary;

    struct Flat;

    impl SaturationGuide for Flat {
        fn score_candidates(&self, candidates: &[CandidateSummary]) -> Vec<f32> {
            vec![0.0; candidates.len()]
        }
    }

    fn key_of(optimizer: &Optimizer) -> SaturationKey {
        optimizer
            .saturation_key()
            .expect("an unguided, unmasked optimizer is cacheable")
    }

    /// The key distinguishes exactly what decides the saturated graph, and
    /// nothing that only the extraction reads.
    #[test]
    fn the_saturation_key_covers_the_rules_and_the_budget() {
        let production = key_of(&Optimizer::production());
        assert_eq!(production, key_of(&Optimizer::production()));

        let reordered = {
            let mut rules = crate::egraph::all_rules();
            rules.reverse();
            Optimizer::production().rules(RuleSet::new(rules))
        };
        assert_ne!(production, key_of(&reordered), "rule order is part of it");

        let budgets = [
            Budget::Production,
            Budget::Applications(100),
            Budget::Applications(101),
            Budget::Explicit {
                iterations: 5,
                minted: 500,
                applications: None,
            },
            Budget::Explicit {
                iterations: 5,
                minted: 500,
                applications: Some(100),
            },
            Budget::Explicit {
                iterations: 5,
                minted: 501,
                applications: Some(100),
            },
        ];
        let keys: Vec<_> = budgets
            .iter()
            .map(|&budget| key_of(&Optimizer::production().budget(budget)))
            .collect();
        for (i, a) in keys.iter().enumerate() {
            for b in &keys[i + 1..] {
                assert_ne!(a, b, "two budgets must not share a key");
            }
        }

        let priced_elsewhere = Optimizer::production()
            .for_lattice(LatticeShape::new([16, 16]))
            .cost(CostModel::default())
            .no_ceiling();
        assert_eq!(
            production,
            key_of(&priced_elsewhere),
            "extraction's pricing is not saturation's"
        );
    }

    /// A guided run reorders the sweep and a masked one withholds an
    /// application: neither is a function of the input and the rules, so
    /// neither has a key.
    #[test]
    fn a_guided_or_masked_run_has_no_key() {
        let guided = Optimizer::production().guide(Some(Box::new(Flat)));
        assert!(guided.saturation_key().is_none());
        let masked = Optimizer::production().mask(Some(ApplicationMask::leave_one_out(0)));
        assert!(masked.saturation_key().is_none());
    }

    /// The journal is chosen when the graph is built, so an observer cannot be
    /// attached to a graph built without one — and is refused before the run,
    /// not after it.
    #[cfg(feature = "provenance-journal")]
    #[test]
    #[should_panic(expected = "needs a graph that keeps a journal")]
    fn an_observer_needs_a_graph_that_keeps_a_journal() {
        let mut egraph = Optimizer::production().egraph();
        let root = egraph.add(crate::egraph::ENode::Var(0));
        let mut observed = Optimizer::production()
            .observe(Some(Box::new(KeepJournal)))
            .no_ceiling();
        let _optimized = observed.run(&mut egraph, root);
    }
}
