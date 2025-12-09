//! 3D volumetric evaluation primitives.
//!
//! Volumes implement the Volume<T> trait for 3D coordinate evaluation.
//! Many Volume types also implement Surface<T> for 2D projection at z=0.

pub mod source;
pub mod transform;

pub use source::FnVolume;
pub use transform::Translate;

