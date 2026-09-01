//! Training infrastructure for the extraction (Judge) head.
//!
//! This is the live training surface:
//! - binary corpus I/O
//! - the tiered split manifest (`split`, plan 0.2)
//! - expression serialization helpers
//! - value-head backward pass (`forward_cached` / `backward_value`, which
//!   differentiates through the recorded `EdgeTrace` into the op embeddings
//!   via `backward_through_accumulator`)
//! - budget-bounded episode generation (`episodes`)
//! - the numeric quarantine every label source must pass (`quarantine`,
//!   plan 0.4) — shared so the corpus build and the trainer's live synthetic
//!   stream apply one predicate, not two
//! - label minting: sentinel drift normalization and the weights-objective
//!   sidecar (`mint`)
//!
//! The RL apparatus this module used to host (self-play mask policy,
//! REINFORCE `backward_policy`, PFTJ/PFAD trajectory export, the ES-guided
//! corpus-growth optimizer) was removed per
//! docs/plans/2026-07-07-guided-saturation-redesign.md — see that doc for
//! the full post-mortem and the supervised, hindsight-labeled replacement
//! architecture.

#[cfg(feature = "training")]
pub mod factored;

#[cfg(feature = "training")]
pub mod corpus;

#[cfg(feature = "training")]
pub mod split;

#[cfg(feature = "training")]
pub mod unified_backward;

#[cfg(feature = "training")]
pub mod episodes;

#[cfg(feature = "training")]
pub mod mint;

#[cfg(feature = "training")]
pub mod structural;

#[cfg(feature = "training")]
pub mod quarantine;

#[cfg(feature = "training")]
pub mod guide_linear;
