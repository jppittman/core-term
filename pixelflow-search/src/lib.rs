#![allow(clippy::all)]
#![allow(warnings)]
#![allow(unused)]
extern crate alloc;

#[cfg(test)]
mod arena_corpus;
pub mod egraph;
#[cfg(test)]
mod extraction_gap;
pub mod math;
pub mod nnue;
pub mod runtime;
#[cfg(feature = "saturation-telemetry")]
pub mod telemetry;
