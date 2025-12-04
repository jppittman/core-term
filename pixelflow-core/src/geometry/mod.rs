/// Affine transformations.
pub mod affine;
/// Polygon and curve primitives.
pub mod poly;
/// Basic geometric primitives like Rect.
pub mod primitive;

pub use affine::Mat3;
pub use poly::{Curve2D, Poly};
pub use primitive::Rect;
