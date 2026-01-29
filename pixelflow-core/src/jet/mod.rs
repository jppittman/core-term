//! # Jet Types for Automatic Differentiation
//!
//! - Jet2: value + 2D gradient (∂f/∂x, ∂f/∂y)
//! - Jet2H: value + 2D gradient + Hessian (∂²f/∂x², ∂²f/∂x∂y, ∂²f/∂y²)
//! - Jet3: value + 3D gradient (∂f/∂x, ∂f/∂y, ∂f/∂z) for surface normals
//! - PathJet: origin + direction (for ray space coordinates)

pub mod jet2;
pub mod jet2h;
pub mod jet3;
pub mod path_jet;

// Re-export for backward compatibility
pub use jet2::{Jet2, Jet2Sqrt};
pub use jet2h::{Jet2H, Jet2HSqrt};
pub use jet3::{Jet3, Jet3Sqrt};
pub use path_jet::PathJet;
