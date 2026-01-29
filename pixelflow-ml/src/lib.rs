//! # PixelFlow ML: Neural Networks for Graphics and Compilers
//!
//! This crate provides neural network primitives for two domains:
//!
//! 1. **Linear Attention as Harmonic Global Illumination** - Unifying ML and graphics
//! 2. **NNUE for Instruction Selection** - Neural-guided compiler optimization
//!
//! ## Features
//!
//! - `graphics` (default): Enables harmonic attention and SH feature maps (requires pixelflow-core)
//! - `training`: Enables training data generation
//! - `std`: Enables standard library features

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

extern crate alloc;

// Core NNUE module - no pixelflow-core dependency
pub mod nnue;
pub mod evaluator;
pub mod hce_extractor;
pub mod nonlinear_eval;

#[cfg(any(feature = "training", feature = "egraph-training"))]
pub mod training;

#[cfg(feature = "training")]
pub mod nnue_trainer;

// Graphics module - requires pixelflow-core
#[cfg(feature = "graphics")]
mod graphics;

#[cfg(feature = "graphics")]
pub use graphics::*;
