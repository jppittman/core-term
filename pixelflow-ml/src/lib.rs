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
//!
//! ## NNUE
//!
//! The core NNUE implementation lives in the `pixelflow-nnue` crate.
//! This crate re-exports it for convenience.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

extern crate alloc;

// Re-export pixelflow-nnue as nnue module for backwards compatibility
pub use pixelflow_nnue as nnue;

pub mod evaluator;
pub mod nonlinear_eval;

// Training module requires SIMD evaluation (std + graphics)
#[cfg(all(feature = "std", feature = "graphics"))]
pub mod training;

#[cfg(all(feature = "std", feature = "graphics"))]
pub mod nnue_trainer;

// Benchmark module requires std for timing and thread pinning
#[cfg(feature = "std")]
pub mod benchmark;

// Graphics module - requires pixelflow-core
#[cfg(feature = "graphics")]
mod graphics;

#[cfg(feature = "graphics")]
pub use graphics::*;

// SIMD evaluation module - requires pixelflow-core for Field type

