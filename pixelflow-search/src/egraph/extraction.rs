//! Extraction policy: which cost model picks the cheapest representative of
//! each e-class after saturation.
//!
//! One policy, two tiers: the `kernel!`/`kernel_jit!` macros (AOT,
//! `pixelflow-compiler`) and runtime-built kernels
//! ([`crate::runtime::optimize_runtime_arena`]) both select a policy through
//! [`env_extraction_policy`] and drive extraction through
//! [`ExtractionPolicy::choices`] — one implementation, one place the policy
//! is chosen.
//!
//! There is exactly one policy today: the static latency prior
//! ([`CostModel::latency_prior`]). A learned alternative — the NNUE
//! extraction head, opt-in through a weights-file env var — lived behind this
//! seam from 2026-07 to 2026-09 and was closed as an honest negative: the
//! static table tied it and every lever made it worse
//! (docs/paper/2026-08-egraph-nnue-parity.md). It is deleted, not disabled;
//! this seam is where a successor would be chosen, and until one earns its
//! place the type below refuses to represent a second choice.

use super::cost::CostModel;
use super::extract::{Extraction, extract_dag};
use super::graph::EGraph;
use super::node::EClassId;
use alloc::vec::Vec;

/// The cost model that drives e-graph extraction: a static per-op cost table.
pub struct ExtractionPolicy {
    costs: CostModel,
}

impl ExtractionPolicy {
    /// The production policy: [`CostModel::latency_prior`].
    #[must_use]
    pub fn latency_prior() -> Self {
        Self::with_costs(CostModel::latency_prior())
    }

    /// Extraction under an explicit cost table — tests and research bins
    /// that want to compare tables go through here; production goes through
    /// [`env_extraction_policy`].
    #[must_use]
    pub fn with_costs(costs: CostModel) -> Self {
        Self { costs }
    }

    /// Per-e-class extraction choices under this policy, as a validated
    /// [`Extraction`] — the well-founded (egraph, root, choices) triple
    /// [`super::extract::choices_to_arena`] requires.
    pub fn extraction<'g>(&self, egraph: &'g EGraph, root: EClassId) -> Extraction<'g> {
        let dag = extract_dag(egraph, root, &self.costs);
        // `extract_dag` already repairs `dag.choices` into a well-founded set
        // internally; `from_dp`'s repair pass is then a verified no-op
        // (drain phase only).
        Extraction::from_dp(egraph, root, dag.choices)
    }

    /// Per-e-class extraction choices under this policy, as a raw vector —
    /// kept for `pixelflow-compiler`'s ref-counting / DAG-codegen consumers
    /// (`compute_ref_counts`, `build_extracted_dag_from_choices`) that
    /// predate the [`Extraction`] type.
    pub fn choices(&self, egraph: &EGraph, root: EClassId) -> Vec<Option<usize>> {
        self.extraction(egraph, root).into_choices()
    }
}

/// The one place the extraction policy is chosen for both tiers.
///
/// Returns the static latency prior. The name records the seam's history:
/// this function used to consult the process environment for an opt-in
/// learned model, and both tiers still route through it so a future policy
/// change is one edit, not two. Nothing is read from the environment now.
#[must_use]
pub fn env_extraction_policy() -> ExtractionPolicy {
    ExtractionPolicy::latency_prior()
}
