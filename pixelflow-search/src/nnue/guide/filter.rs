//! The bilinear head as an [`ApplicationFilter`], and the two controls it
//! is registered against
//! (docs/plans/2026-09-08-rules-filter-bilinear-registration.md).
//!
//! `F : M → M'` over the rules × e-classes match matrix
//! (`egraph::filter`): each row is one rule's matches, and each arm here
//! keeps a cell iff its score clears a threshold. Only the score differs:
//!
//! | arm | score of `(r, c)` |
//! |---|---|
//! | [`BilinearFilter`] | `m(x_c)ᵀ W e_r + bᵀ e_r` — `SaturationHead::score_candidate` |
//! | [`PerRuleRateFilter`] | a per-rule constant: the rule's positive rate on the training samples |
//! | [`UniformRandomFilter`] | a hash of `(seed, r, c, applications so far)` |
//!
//! # One featurization, minted and deployed
//!
//! [`CellContext::observe`] is the only function that turns a cell into the
//! context the tower reads, and it reads the graph **as the filter sees
//! it** — after the row was matched, before it is committed. The training
//! samples are minted by a recording filter that calls the same function
//! at the same point, so the model is trained on the features it will be
//! scored on. `egraph::candidate::CandidateFeatures::observe` describes a
//! class as the graph it is handed holds it, and the earlier label pipelines
//! handed it the fully-saturated graph — a class as it *ended*, not as it
//! was matched. That gap is why this module does not reuse it.
//!
//! # Cost is counted
//!
//! Every filter reports what it saw, kept and computed through a
//! [`FilterStats`] handle ([`Reporting`]): the point of a CPU-resident
//! filter is that it is cheap, and "cheap" is a number the evaluation
//! prints beside the extraction it bought, not an adjective.

extern crate alloc;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::Cell;

use pixelflow_ir::OpKind;

use super::bilinear::{BilinearWeights, load_parts, rule_concats};
use super::diversity::{fnv1a64, splitmix64};
use super::scoring::SaturationHead;
use super::{CandidateSummary, EMBED_DIM};
use crate::egraph::EClassId;
use crate::egraph::EGraph;
use crate::egraph::candidate::neighborhood_ops;
use crate::egraph::filter::{ApplicationFilter, MatchRow};
use crate::egraph::rules::{Fingerprint, RuleId, RuleSet};
use crate::nnue::factored::OpEmbeddings;

/// The two constants of one saturation episode a cell's context reads:
/// the kernel's size, and the budget `budget_fraction` is denominated in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Episode {
    /// The kernel's reachable input node count — the `node_count`
    /// [`Optimizer::run`](crate::egraph::Optimizer::run) is given.
    pub expr_node_count: usize,
    /// The denominator of `budget_fraction`: the production tier's
    /// application cap (`SaturationConfig::max_applications`), so a run at
    /// half the budget reads the same units the labels were minted in.
    pub budget: u64,
}

impl Episode {
    /// # Panics
    ///
    /// If `budget` is zero — a fraction of nothing is not a feature.
    #[must_use]
    pub fn new(expr_node_count: usize, budget: u64) -> Self {
        assert!(
            budget > 0,
            "Episode: budget_fraction needs a positive budget denominator, got 0"
        );
        Self {
            expr_node_count,
            budget,
        }
    }
}

/// One cell's context — everything the candidate tower reads except the
/// rule — observed on the live graph at the moment the filter sees the row.
#[derive(Clone, Debug, PartialEq)]
pub struct CellContext {
    /// One-hop child ops of every node in the matched class, with
    /// multiplicity, sorted.
    pub neighborhood_ops: Vec<OpKind>,
    /// Nodes in the matched class's canonical content.
    pub match_class_node_count: usize,
    /// Applications committed so far ÷ [`Episode::budget`].
    pub budget_fraction: f32,
    /// [`Episode::expr_node_count`].
    pub expr_node_count: usize,
}

impl CellContext {
    /// Observe `class` on `graph` as it stands now.
    #[must_use]
    pub fn observe(graph: &EGraph, class: EClassId, episode: &Episode) -> Self {
        let canonical = graph.find(class);
        Self {
            neighborhood_ops: neighborhood_ops(graph, canonical),
            match_class_node_count: graph.nodes(canonical).len(),
            budget_fraction: graph.application_count() as f32 / episode.budget as f32,
            expr_node_count: episode.expr_node_count,
        }
    }

    /// The [`CandidateSummary`] the head scores, with the rule supplied.
    #[must_use]
    pub fn summary(&self, rule: RuleId, rule_embed: [f32; EMBED_DIM]) -> CandidateSummary {
        CandidateSummary {
            rule_embed,
            neighborhood_ops: self.neighborhood_ops.clone(),
            budget_fraction: self.budget_fraction,
            rule,
            match_class_node_count: self.match_class_node_count,
            expr_node_count: self.expr_node_count,
        }
    }
}

/// What a filter did over one run: rows offered, cells seen, cells kept,
/// distinct cells scored, and the multiply-adds those scores cost.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FilterStats {
    pub rows: u64,
    pub cells: u64,
    pub kept: u64,
    /// Cells whose score was computed. Consecutive cells of one row at the
    /// same class share a context and a rule, so they share a score; this
    /// counts the computations, `cells` counts the decisions.
    pub scored: u64,
    /// Multiply-adds spent scoring, counted from the head's dimensions.
    pub macs: u64,
    /// Rows whose rule had no entry in the arm's table (per-rule arm only):
    /// scored as zero and reported, never silently.
    pub unknown_rule_rows: u64,
    /// Wall clock spent inside the bilinear filter's `filter` calls, in
    /// nanoseconds — the filter's own share of saturation. A sign, not a
    /// claim: read it beside the load it was taken at.
    pub filter_ns: u64,
}

/// A shared, copyable stats slot: the [`Optimizer`](crate::egraph::Optimizer)
/// owns its filter as `Box<dyn ApplicationFilter>` and hands nothing back,
/// so a harness that wants the numbers keeps a clone of this handle.
pub type Reporting = Rc<Cell<FilterStats>>;

fn bump(reporting: &Reporting, f: impl FnOnce(&mut FilterStats)) {
    let mut s = reporting.get();
    f(&mut s);
    reporting.set(s);
}

/// Content digest of an `f32` slice, for a filter fingerprint.
fn digest_f32(bytes: &mut Vec<u8>, values: &[f32]) {
    for v in values {
        bytes.extend_from_slice(&v.to_bits().to_le_bytes());
    }
}

// ── Bilinear ────────────────────────────────────────────────────────────────

/// The bilinear head, deployed as a row filter: keep a cell iff
/// `score(r, c) ≥ threshold`.
pub struct BilinearFilter {
    embeddings: OpEmbeddings,
    head: SaturationHead,
    rule_embeds: BTreeMap<RuleId, [f32; EMBED_DIM]>,
    threshold: f32,
    episode: Episode,
    fingerprint: Fingerprint,
    reporting: Reporting,
}

impl BilinearFilter {
    /// Deploy `weights` against `rules` for one episode, keeping cells whose
    /// raw bilinear score is at least `threshold`.
    ///
    /// # Errors
    ///
    /// The same refusals as
    /// [`BilinearCandidateGuide::new`](super::bilinear::BilinearCandidateGuide::new):
    /// a vocabulary fingerprint or a parameter block that is not this
    /// build's.
    pub fn new(
        weights: &BilinearWeights,
        rules: &RuleSet,
        threshold: f32,
        episode: Episode,
    ) -> Result<Self, super::linear::GuideError> {
        assert!(
            !threshold.is_nan(),
            "BilinearFilter: threshold is NaN — no score clears it and nothing is applied, \
             silently; a keep-nothing arm spells its threshold +inf"
        );
        let (embeddings, head) = load_parts(weights, rules)?;
        let (concats, _) = rule_concats(&embeddings, rules);
        let rule_embeds = concats
            .iter()
            .map(|(&id, c)| (id, head.project_rule(c)))
            .collect();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"bilinear-filter");
        bytes.extend_from_slice(&weights.fingerprint.to_bytes());
        digest_f32(&mut bytes, &weights.parameters);
        digest_f32(&mut bytes, &weights.op_embeddings);
        bytes.extend_from_slice(&threshold.to_bits().to_le_bytes());
        Ok(Self {
            embeddings,
            head,
            rule_embeds,
            threshold,
            episode,
            fingerprint: Fingerprint::from_raw(fnv1a64(&bytes)),
            reporting: Rc::default(),
        })
    }

    /// Report this filter's [`FilterStats`] into `reporting`.
    #[must_use]
    pub fn reporting(mut self, reporting: Reporting) -> Self {
        self.reporting = reporting;
        self
    }

    /// The raw score of one cell — the deployed computation, for a harness
    /// that calibrates the threshold on the same function the filter runs.
    #[must_use]
    pub fn score(&self, rule: RuleId, context: &CellContext) -> f32 {
        let rule_embed = self.rule_embed(rule);
        self.head
            .score_candidate(&self.embeddings, &context.summary(rule, rule_embed))
    }

    fn rule_embed(&self, rule: RuleId) -> [f32; EMBED_DIM] {
        *self.rule_embeds.get(&rule).unwrap_or_else(|| {
            panic!(
                "BilinearFilter: rule {rule} has no derived embedding — the row came from a \
                 rule set this filter was not built against, which the fingerprint check in \
                 `BilinearFilter::new` should have refused"
            )
        })
    }
}

impl ApplicationFilter for BilinearFilter {
    fn filter(&mut self, graph: &EGraph, row: &mut MatchRow) {
        let started = std::time::Instant::now();
        let rule = row.rule();
        let rule_embed = self.rule_embed(rule);
        let cells = row.matches().len() as u64;
        let (head, embeddings, episode, threshold) =
            (&self.head, &self.embeddings, &self.episode, self.threshold);
        let mut scored = 0u64;
        let mut macs = 0u64;
        let mut kept = 0u64;
        // The scan visits a class, then each of its nodes: a class's cells
        // are consecutive, share a context and a rule, and so share a
        // score. One computation per run of a class, not per cell.
        let mut last: Option<(EClassId, bool)> = None;
        row.retain(|(class, _)| {
            let keep = match last {
                Some((c, keep)) if c == *class => keep,
                _ => {
                    let context = CellContext::observe(graph, *class, episode);
                    let summary = context.summary(rule, rule_embed);
                    scored += 1;
                    macs += SaturationHead::score_candidate_macs(summary.neighborhood_ops.len());
                    head.score_candidate(embeddings, &summary) >= threshold
                }
            };
            last = Some((*class, keep));
            kept += u64::from(keep);
            keep
        });
        let filter_ns = started.elapsed().as_nanos() as u64;
        bump(&self.reporting, |s| {
            s.rows += 1;
            s.cells += cells;
            s.kept += kept;
            s.scored += scored;
            s.macs += macs;
            s.filter_ns += filter_ns;
        });
    }

    fn fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }
}

// ── Per-rule prior ──────────────────────────────────────────────────────────

/// The per-rule control: every cell of rule `r` scores `rate[r]`, so a rule
/// is kept in every context or in none. What a per-rule prior can express,
/// and nothing more.
pub struct PerRuleRateFilter {
    rates: BTreeMap<RuleId, f32>,
    threshold: f32,
    fingerprint: Fingerprint,
    reporting: Reporting,
}

impl PerRuleRateFilter {
    /// `rates` is each rule's positive rate on the training samples; a rule
    /// absent from it scores zero and is counted in
    /// [`FilterStats::unknown_rule_rows`].
    #[must_use]
    pub fn new(rates: BTreeMap<RuleId, f32>, threshold: f32) -> Self {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"per-rule-rate-filter");
        for (rule, rate) in &rates {
            bytes.extend_from_slice(&rule.get().to_le_bytes());
            bytes.extend_from_slice(&rate.to_bits().to_le_bytes());
        }
        bytes.extend_from_slice(&threshold.to_bits().to_le_bytes());
        Self {
            rates,
            threshold,
            fingerprint: Fingerprint::from_raw(fnv1a64(&bytes)),
            reporting: Rc::default(),
        }
    }

    /// Report this filter's [`FilterStats`] into `reporting`.
    #[must_use]
    pub fn reporting(mut self, reporting: Reporting) -> Self {
        self.reporting = reporting;
        self
    }
}

impl ApplicationFilter for PerRuleRateFilter {
    fn filter(&mut self, _graph: &EGraph, row: &mut MatchRow) {
        let cells = row.matches().len() as u64;
        let (rate, unknown) = match self.rates.get(&row.rule()) {
            Some(&r) => (r, 0),
            None => (0.0, 1),
        };
        let keep = rate >= self.threshold;
        if !keep {
            row.retain(|_| false);
        }
        bump(&self.reporting, |s| {
            s.rows += 1;
            s.cells += cells;
            s.kept += if keep { cells } else { 0 };
            s.unknown_rule_rows += unknown;
        });
    }

    fn fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }
}

// ── Uniform random ──────────────────────────────────────────────────────────

/// The no-information control: keep each cell with probability `keep_rate`,
/// decided by a hash of `(seed, rule, class, applications so far)` so the
/// same kernel under the same seed filters the same way every time.
pub struct UniformRandomFilter {
    seed: u64,
    keep_rate: f32,
    fingerprint: Fingerprint,
    reporting: Reporting,
}

impl UniformRandomFilter {
    /// # Panics
    ///
    /// If `keep_rate` is not in `[0, 1]`.
    #[must_use]
    pub fn new(seed: u64, keep_rate: f32) -> Self {
        assert!(
            (0.0..=1.0).contains(&keep_rate),
            "UniformRandomFilter: keep_rate must be in [0, 1], got {keep_rate}"
        );
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"uniform-random-filter");
        bytes.extend_from_slice(&seed.to_le_bytes());
        bytes.extend_from_slice(&keep_rate.to_bits().to_le_bytes());
        Self {
            seed,
            keep_rate,
            fingerprint: Fingerprint::from_raw(fnv1a64(&bytes)),
            reporting: Rc::default(),
        }
    }

    /// Report this filter's [`FilterStats`] into `reporting`.
    #[must_use]
    pub fn reporting(mut self, reporting: Reporting) -> Self {
        self.reporting = reporting;
        self
    }
}

impl ApplicationFilter for UniformRandomFilter {
    fn filter(&mut self, graph: &EGraph, row: &mut MatchRow) {
        let cells = row.matches().len() as u64;
        let applications = graph.application_count();
        let rule = row.rule().get();
        let (seed, keep_rate) = (self.seed, self.keep_rate);
        let mut kept = 0u64;
        row.retain(|(class, _)| {
            let mut bytes = [0u8; 32];
            bytes[..8].copy_from_slice(&rule.to_le_bytes());
            bytes[8..16].copy_from_slice(&(class.index() as u64).to_le_bytes());
            bytes[16..24].copy_from_slice(&applications.to_le_bytes());
            bytes[24..].copy_from_slice(&seed.to_le_bytes());
            let mixed = splitmix64(seed ^ fnv1a64(&bytes));
            let unit = (mixed as f64 / u64::MAX as f64) as f32;
            let keep = unit < keep_rate;
            kept += u64::from(keep);
            keep
        });
        bump(&self.reporting, |s| {
            s.rows += 1;
            s.cells += cells;
            s.kept += kept;
        });
    }

    fn fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }
}

/// Box a filter for [`Optimizer::filter`](crate::egraph::Optimizer::filter).
#[must_use]
pub fn boxed<F: ApplicationFilter + 'static>(filter: F) -> Box<dyn ApplicationFilter> {
    Box::new(filter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egraph::{Budget, Optimizer, Vocabulary, insert, reachable_count};
    use crate::nnue::guide::bilinear::BilinearTrainer;
    use pixelflow_ir::ExprArena;

    fn a_kernel() -> (ExprArena, pixelflow_ir::ExprId) {
        let mut arena = ExprArena::new();
        let x = arena.push_var(0);
        let y = arena.push_var(1);
        let s = arena.push_binary(OpKind::Add, x, y);
        let p = arena.push_binary(OpKind::Mul, s, s);
        let q = arena.push_binary(OpKind::Mul, x, y);
        let root = arena.push_binary(OpKind::Sub, p, q);
        (arena, root)
    }

    fn run_with(filter: Box<dyn ApplicationFilter>) -> (usize, u64, usize) {
        let (arena, root) = a_kernel();
        let mut optimizer = Optimizer::production()
            .budget(Budget::Applications(400))
            .filter(filter);
        let mut egraph = optimizer.egraph();
        let root_class = insert(&arena, root, &mut egraph, Vocabulary::Templates).expect("insert");
        let n = reachable_count(&arena, root);
        let out = optimizer.run(&mut egraph, root_class, n);
        (out.cost.dag, out.stats.applications, out.stats.classes)
    }

    #[test]
    fn a_threshold_of_minus_infinity_keeps_everything_and_is_production() {
        let rules = RuleSet::production();
        let weights = BilinearTrainer::new_cold(&rules, 7).weights();
        let stats: Reporting = Rc::default();
        let filter =
            BilinearFilter::new(&weights, &rules, f32::NEG_INFINITY, Episode::new(6, 20_000))
                .expect("same vocabulary")
                .reporting(Rc::clone(&stats));
        let filtered = run_with(boxed(filter));
        let production = run_with(boxed(crate::egraph::KeepAll));
        assert_eq!(filtered, production, "keep-all bilinear must be production");
        let s = stats.get();
        assert!(s.rows > 0 && s.cells > 0);
        assert_eq!(s.kept, s.cells);
        assert!(s.scored <= s.cells && s.scored > 0);
        assert!(s.macs > 0);
    }

    #[test]
    fn a_threshold_of_plus_infinity_applies_nothing() {
        let rules = RuleSet::production();
        let weights = BilinearTrainer::new_cold(&rules, 7).weights();
        let stats: Reporting = Rc::default();
        let filter = BilinearFilter::new(&weights, &rules, f32::INFINITY, Episode::new(6, 20_000))
            .expect("same vocabulary")
            .reporting(Rc::clone(&stats));
        let (_, applications, _) = run_with(boxed(filter));
        assert_eq!(applications, 0);
        assert_eq!(stats.get().kept, 0);
        assert!(stats.get().cells > 0);
    }

    #[test]
    fn uniform_random_is_deterministic_in_its_seed() {
        let a = run_with(boxed(UniformRandomFilter::new(3, 0.5)));
        let b = run_with(boxed(UniformRandomFilter::new(3, 0.5)));
        assert_eq!(a, b, "same seed, same kernel, same run");
        let production = run_with(boxed(crate::egraph::KeepAll));
        assert!(
            a.1 < production.1,
            "half the cells kept must fire fewer applications"
        );
    }

    #[test]
    fn uniform_random_keeps_about_its_rate_over_a_wide_row() {
        use crate::egraph::{ENode, RewriteAction};
        let rules = RuleSet::production();
        let mut graph = Optimizer::production().egraph();
        let cells: Vec<(EClassId, RewriteAction)> = (0..4000u32)
            .map(|i| {
                let class = graph.add(ENode::Const(i));
                (class, RewriteAction::Union(class))
            })
            .collect();
        let stats: Reporting = Rc::default();
        let mut filter = UniformRandomFilter::new(11, 0.25).reporting(Rc::clone(&stats));
        let mut row = MatchRow::new(0, rules.id_of(0).expect("a rule"), cells);
        filter.filter(&graph, &mut row);
        let s = stats.get();
        let rate = s.kept as f64 / s.cells as f64;
        assert_eq!(s.cells, 4000);
        assert!(
            (0.22..0.28).contains(&rate),
            "keep rate {rate} far from 0.25 over {} cells",
            s.cells
        );
    }

    #[test]
    fn per_rule_rate_keeps_a_rule_everywhere_or_nowhere() {
        let rules = RuleSet::production();
        let mut rates = BTreeMap::new();
        for id in rules.ids() {
            rates.insert(*id, 1.0);
        }
        let all = run_with(boxed(PerRuleRateFilter::new(rates.clone(), 0.5)));
        let production = run_with(boxed(crate::egraph::KeepAll));
        assert_eq!(all, production);
        let stats: Reporting = Rc::default();
        let (_, applications, _) = run_with(boxed(
            PerRuleRateFilter::new(BTreeMap::new(), 0.5).reporting(Rc::clone(&stats)),
        ));
        assert_eq!(applications, 0);
        assert!(stats.get().unknown_rule_rows > 0);
    }

    #[test]
    fn filters_fingerprint_differently_from_keep_all_and_from_each_other() {
        let rules = RuleSet::production();
        let weights = BilinearTrainer::new_cold(&rules, 7).weights();
        let bilinear = BilinearFilter::new(&weights, &rules, 0.0, Episode::new(6, 20_000))
            .expect("same vocabulary");
        let bilinear2 = BilinearFilter::new(&weights, &rules, 0.5, Episode::new(6, 20_000))
            .expect("same vocabulary");
        let keep_all = crate::egraph::KeepAll;
        let uniform = UniformRandomFilter::new(1, 0.25);
        let fps = [
            keep_all.fingerprint(),
            bilinear.fingerprint(),
            bilinear2.fingerprint(),
            uniform.fingerprint(),
        ];
        for i in 0..fps.len() {
            for j in (i + 1)..fps.len() {
                assert_ne!(fps[i], fps[j], "fingerprints {i} and {j} collide");
            }
        }
        assert_eq!(
            Fingerprint::from_raw(7).combine(keep_all.fingerprint()),
            Fingerprint::from_raw(7),
            "KeepAll is the identity of Fingerprint::combine"
        );
    }
}
