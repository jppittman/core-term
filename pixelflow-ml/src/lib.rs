//! # PixelFlow Machine Learning Integration
//!
//! This crate provides machine learning primitives for graphics, specifically
//! harmonic attention and spherical harmonic features.
//!
//! (The compiler optimization logic has moved to `pixelflow-compiler`.)

extern crate alloc;

#[cfg(feature = "graphics")]
pub mod graphics;

pub mod layer;
pub mod nnue;
pub mod nnue_trainer;
pub mod evaluator;
pub mod nonlinear_eval;
pub mod benchmark;
pub mod train;
pub mod hce_extractor;

#[cfg(feature = "egraph-training")]
pub mod training;