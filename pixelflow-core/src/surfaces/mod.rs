pub mod blend;
pub mod control;
pub mod grade;
pub mod implicit;
pub mod lerp;
pub mod partition;
pub mod raster;
pub mod source;
pub mod transform;
pub mod warp;

pub use blend::{Max, Mul, Over};
pub use control::{Fix, Select};
pub use grade::Grade;
pub use implicit::{CubicImplicit, Implicit, ImplicitEvaluator, QuadraticImplicit};
pub use lerp::Lerp;
pub use partition::Partition;
pub use raster::Rasterize;
pub use source::{Baked, Compute, SampleAtlas};

#[deprecated(note = "Use Compute instead")]
pub type FnSurface<F, T> = Compute<F, T>;
pub use transform::{Map, Offset, Scale, Skew};
pub use warp::Warp;
// Alias FnSurface to Compute for backward compatibility if needed, or just remove it.
// Let's keep existing code happy if possible, but the instruction is to refactor.
// I'll add an alias for now to avoid breaking everything immediately, or fix references.
// The user asked to "rename", so I will prefer replacing usages.
// But for this file, just export Compute.
