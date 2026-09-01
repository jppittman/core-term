//! PixelFlow Pipeline: Training and Data Generation.
//!
//! This crate provides the training pipeline for the PixelFlow compiler's
//! cost model: JIT bench harness, corpus generation, and extraction-head
//! (Judge value head) training. See
//! docs/plans/2026-07-07-guided-saturation-redesign.md for the architecture
//! and the (2026-07) removal of the RL self-play/critic loop this crate used
//! to also host.

pub mod jit_bench;
pub mod journal;
pub mod oracle_lowering;
pub mod schema;
pub mod shader_bench;

// Training infrastructure (requires std feature)
#[cfg(feature = "training")]
pub mod training;

/// Canonical on-disk location of the extraction-head weights:
/// `<this crate>/data/extraction_head.bin`.
///
/// The single source of truth shared by `bootstrap_extraction_head` (default
/// input and output) and `bench_extraction_3way` (default input), so a fresh
/// `gen_bench_corpus → bootstrap_extraction_head → bench_extraction_3way` run
/// feeds the bench with no manual rename step
/// (docs/plans/2026-08-05-egraph-nnue-research-workflow.md, Phase 0.1).
#[must_use]
pub fn extraction_head_weights_path() -> std::path::PathBuf {
    std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/extraction_head.bin"
    ))
}
