//! Extraction policy: which cost model picks the cheapest representative of
//! each e-class after saturation.
//!
//! One policy, two tiers: the `kernel!`/`kernel_jit!` macros (AOT,
//! `pixelflow-compiler`) and runtime-built kernels
//! ([`crate::runtime::optimize_runtime_arena`]) both select a policy through
//! [`env_extraction_policy`] and drive extraction through
//! [`ExtractionPolicy::choices`] — one implementation, not two copies of the
//! `PIXELFLOW_NNUE_WEIGHTS` opt-in.

use super::cost::CostModel;
use super::graph::EGraph;
use super::extract::extract_dag;
use super::node::EClassId;
use crate::nnue::ExprNnue;
use std::sync::OnceLock;

/// Which cost model drives e-graph extraction.
///
/// ## Default: static latency-prior extraction
///
/// Per the Phase 2 gate in docs/plans/2026-07-07-guided-saturation-redesign.md,
/// the static [`CostModel::latency_prior`] is the default and the learned
/// model is opt-in only.
///
/// ## Opt-in: `PIXELFLOW_NNUE_WEIGHTS`
///
/// Set this env var to the path of a trained weights file to enable learned
/// extraction. Any failure to load — missing file, wrong magic, wrong length
/// — is a hard failure with a precise diagnostic: if you asked for learned
/// weights, you get them or the caller panics. There is no silent fallback.
pub enum ExtractionPolicy<'a> {
    /// Opt-in learned extraction (`PIXELFLOW_NNUE_WEIGHTS` set).
    Nnue(&'a ExprNnue),
    /// Default: static per-op latency-prior costs. Boxed — `CostModel` is far
    /// larger than the `Nnue` reference, so an unboxed variant bloats the enum.
    Static(Box<CostModel>),
}

impl ExtractionPolicy<'_> {
    /// Per-e-class extraction choices under this policy.
    pub fn choices(&self, egraph: &EGraph, root: EClassId) -> Vec<Option<usize>> {
        match self {
            ExtractionPolicy::Nnue(model) => {
                let extractor = super::extract::IncrementalExtractor::new(model, 8);
                extractor.extract_choices_only(egraph, root).1
            }
            ExtractionPolicy::Static(costs) => extract_dag(egraph, root, costs.as_ref()).choices,
        }
    }
}

/// The opt-in NNUE weights, loaded once and cached for the process lifetime.
static OPTIMIZATION_MODEL: OnceLock<ExprNnue> = OnceLock::new();

/// Select the extraction policy from the environment: `PIXELFLOW_NNUE_WEIGHTS`
/// if set (hard-failing on a bad file), else the static latency-prior model.
///
/// Read at the call site's own time — macro expansion time for the AOT tier,
/// first-bake time for the runtime tier — so both tiers honor the same env
/// var the same way.
pub fn env_extraction_policy() -> ExtractionPolicy<'static> {
    match std::env::var("PIXELFLOW_NNUE_WEIGHTS") {
        Ok(path) => {
            ExtractionPolicy::Nnue(OPTIMIZATION_MODEL.get_or_init(|| load_opt_in_weights(&path)))
        }
        Err(_) => ExtractionPolicy::Static(Box::new(CostModel::latency_prior())),
    }
}

/// Load NNUE weights from an opt-in path set via `PIXELFLOW_NNUE_WEIGHTS`.
///
/// Hard-fails on any error, per the repo's no-silent-failures rule: the
/// caller explicitly asked for learned weights, so a failure to honor that
/// request must not be swallowed.
fn load_opt_in_weights(path: &str) -> ExprNnue {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        panic!(
            "pixelflow: PIXELFLOW_NNUE_WEIGHTS={path:?} could not be read: {e}. \
             The env var must point to a valid NNUE weights file produced by \
             pixelflow-pipeline training."
        )
    });

    const EXPECTED_MAGIC: &[u8; 4] = b"TRID";
    let found_magic = bytes.get(0..4);
    match ExprNnue::from_bytes(&bytes) {
        Ok(model) => model,
        Err(e) => {
            let magic_desc = match found_magic {
                Some(m) => format!("{:?} ({})", m, String::from_utf8_lossy(m)),
                None => format!("<file too short: {} bytes>", bytes.len()),
            };
            panic!(
                "pixelflow: PIXELFLOW_NNUE_WEIGHTS={path:?} failed to load: {e}. \
                 Expected magic {:?} ({}), found magic {}. File length: {} bytes.",
                EXPECTED_MAGIC,
                String::from_utf8_lossy(EXPECTED_MAGIC),
                magic_desc,
                bytes.len()
            )
        }
    }
}
