/// Arithmetic surfaces (Add, Sub, Mul, Div).
pub mod arithmetic;
/// Comparison surfaces (Lt, Gt, Eq, etc).
pub mod comparison;
/// Control flow surfaces (Select, Partition).
pub mod control;
/// Coordinate projection surfaces (X, Y, Z, W).
pub mod coords;
/// Grade (contrast/brightness) adjustment.
pub mod grade;
/// Implicit surface adapters.
pub mod implicit;
/// Linear interpolation (Lerp).
pub mod lerp;
/// Mathematical functions (Abs, Sqrt, etc).
pub mod math;
/// Maximum value of two surfaces.
pub mod max;
/// Spatial partitioning for tiled rendering.
pub mod partition;
/// Functional and pre-baked sources (Compute, Baked).
pub mod source;
/// Coordinate transformations (Offset, Scale, Skew).
pub mod transform;
/// Coordinate warping (UV mapping).
pub mod warp;

pub use arithmetic::{Add, Div, Mul, Sub};
pub use comparison::{Eq, Ge, Gt, Le, Lt, Ne};
pub use control::{Fix, Select};
pub use coords::{W, X, Y, Z};
pub use grade::Grade;
pub use implicit::{CubicImplicit, Implicit, ImplicitEvaluator, QuadraticImplicit};
pub use lerp::Lerp;
pub use math::{Abs, Negate, Sqrt};
pub use max::Max;
pub use partition::Partition;
pub use source::{Baked, Compute};

#[deprecated(note = "Use Compute instead")]
pub type FnSurface<F, T> = Compute<F, T>;
pub use transform::{Map, Offset, Scale, Skew};
pub use warp::Warp;
// Alias FnSurface to Compute for backward compatibility if needed, or just remove it.
// Let's keep existing code happy if possible, but the instruction is to refactor.
// I'll add an alias for now to avoid breaking everything immediately, or fix references.
// The user asked to "rename", so I will prefer replacing usages.
// But for this file, just export Compute.
