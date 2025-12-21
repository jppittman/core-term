//! # Operations Module
//!
//! All manifold operations: arithmetic, comparisons, min/max, and operator overloads.

mod base;
mod binary;
mod chained;
mod compare;
mod unary;

pub use binary::{Add, Div, Mul, Sub};
pub use compare::{Ge, Gt, Le, Lt};
pub use unary::{Abs, Max, Min, Sqrt};

// base and chained are purely side-effect modules (they define impls)
// No re-exports needed from them.
