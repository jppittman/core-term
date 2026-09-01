//! Extraction policy: which cost model picks the cheapest representative of
//! each e-class after saturation.
//!
//! One policy, two tiers: the `kernel!`/`kernel_jit!` macros (AOT,
//! `pixelflow-compiler`) and runtime-built kernels
//! ([`crate::runtime::optimize_runtime_arena`]) both select a policy through
//! [`env_extraction_policy`] and drive extraction through
//! [`ExtractionPolicy::choices`] — one implementation, not two copies of the
//! `PIXELFLOW_NNUE_WEIGHTS` opt-in.

// The `PIXELFLOW_NNUE_WEIGHTS` opt-in reads a process env var and a weights
// file, so that branch — and only that branch — is gated on the crate's
// `std` feature, matching the gate on `ExprNnue::from_bytes` it calls. With
// `std` off, the static latency prior is the only policy compiled in; the
// `no-std` job in `.github/workflows/rust.yaml` enforces that this crate
// keeps building in that configuration.
use super::cost::CostModel;
use super::extract::{Extraction, extract_dag};
use super::graph::EGraph;
use super::node::EClassId;
use crate::nnue::ExprNnue;
#[cfg(feature = "std")]
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
/// The mechanism is an env var plus a weights file, so it exists only with
/// the `std` feature (default-on); without it, [`env_extraction_policy`]
/// compiles down to the static branch alone.
pub enum ExtractionPolicy<'a> {
    /// Opt-in learned extraction (`PIXELFLOW_NNUE_WEIGHTS` set).
    Nnue(&'a ExprNnue),
    /// Default: static per-op latency-prior costs. Boxed — `CostModel` is far
    /// larger than the `Nnue` reference, so an unboxed variant bloats the enum.
    Static(Box<CostModel>),
}

impl ExtractionPolicy<'_> {
    /// Per-e-class extraction choices under this policy, as a validated
    /// [`Extraction`] — the well-founded (egraph, root, choices) triple
    /// [`super::extract::choices_to_arena`] and the NNUE accumulator
    /// builders require.
    pub fn extraction<'g>(&self, egraph: &'g EGraph, root: EClassId) -> Extraction<'g> {
        match self {
            ExtractionPolicy::Nnue(model) => {
                let extractor = super::extract::IncrementalExtractor::new(model, 8);
                extractor.extract_choices_only(egraph, root).1
            }
            ExtractionPolicy::Static(costs) => {
                let dag = extract_dag(egraph, root, costs.as_ref());
                // `extract_dag` already repairs `dag.choices` into a
                // well-founded set internally; `from_dp`'s repair pass is
                // then a verified no-op (drain phase only).
                Extraction::from_dp(egraph, root, dag.choices)
            }
        }
    }

    /// Per-e-class extraction choices under this policy, as a raw vector —
    /// kept for `pixelflow-compiler`'s ref-counting / DAG-codegen consumers
    /// (`compute_ref_counts`, `build_extracted_dag_from_choices`) that
    /// predate the [`Extraction`] type.
    pub fn choices(&self, egraph: &EGraph, root: EClassId) -> Vec<Option<usize>> {
        self.extraction(egraph, root).into_choices()
    }
}

/// The opt-in NNUE weights, loaded once and cached for the process lifetime.
#[cfg(feature = "std")]
static OPTIMIZATION_MODEL: OnceLock<ExprNnue> = OnceLock::new();

/// Select the extraction policy from the environment: `PIXELFLOW_NNUE_WEIGHTS`
/// if set (hard-failing on a bad file), else the static latency-prior model.
///
/// Read at the call site's own time — macro expansion time for the AOT tier,
/// first-bake time for the runtime tier — so both tiers honor the same env
/// var the same way.
#[cfg(feature = "std")]
pub fn env_extraction_policy() -> ExtractionPolicy<'static> {
    match std::env::var("PIXELFLOW_NNUE_WEIGHTS") {
        Ok(path) => {
            ExtractionPolicy::Nnue(OPTIMIZATION_MODEL.get_or_init(|| load_opt_in_weights(&path)))
        }
        Err(_) => ExtractionPolicy::Static(Box::new(CostModel::latency_prior())),
    }
}

/// With `std` off there is no process environment to consult, so the
/// `PIXELFLOW_NNUE_WEIGHTS` opt-in mechanism is not compiled in at all — the
/// static latency prior is the only policy, by contract rather than as a
/// fallback. Nothing is silently ignored: an env var cannot be "set" for a
/// build configuration that has no environment.
#[cfg(not(feature = "std"))]
pub fn env_extraction_policy() -> ExtractionPolicy<'static> {
    ExtractionPolicy::Static(Box::new(CostModel::latency_prior()))
}

/// Load NNUE weights from an opt-in path set via `PIXELFLOW_NNUE_WEIGHTS`.
///
/// Hard-fails on any error, per the repo's no-silent-failures rule: the
/// caller explicitly asked for learned weights, so a failure to honor that
/// request must not be swallowed.
#[cfg(feature = "std")]
fn load_opt_in_weights(path: &str) -> ExprNnue {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        panic!(
            "pixelflow: PIXELFLOW_NNUE_WEIGHTS={path:?} could not be read: {e}. \
             The env var must point to a valid NNUE weights file produced by \
             pixelflow-pipeline training."
        )
    });

    const EXPECTED_MAGIC: &[u8; 4] = b"TRIF";
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
