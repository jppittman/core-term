//! # Jet Types for Automatic Differentiation
//!
//! Jets are SIMD batches of dual numbers for automatic differentiation:
//!
//! - `Jet2 = Field<Dual<2>>`: value + 2D gradient (∂f/∂x, ∂f/∂y)
//! - `Jet3 = Field<Dual<3>>`: value + 3D gradient (∂f/∂x, ∂f/∂y, ∂f/∂z)
//! - `Jet2H`: value + 2D gradient + Hessian (second derivatives)
//! - `PathJet`: origin + direction (for ray space coordinates)
//!
//! ## Unified Type System
//!
//! `Jet2` and `Jet3` are now type aliases for `Field<Dual<N>>`, part of the
//! unified `Field<A: Algebra>` type system. This means:
//!
//! - Same SIMD storage layout (SoA for efficiency)
//! - Automatic chain rule propagation
//! - Seamless interop with generic manifold code
//!
//! ## Usage
//!
//! ```ignore
//! use pixelflow_core::jet::{Jet2, Jet3};
//!
//! // Create variables seeded for differentiation
//! let x = Jet2::x(Field::from(3.0));  // ∂x/∂x = 1, ∂x/∂y = 0
//! let y = Jet2::y(Field::from(4.0));  // ∂y/∂x = 0, ∂y/∂y = 1
//!
//! // Compute with automatic derivatives
//! let r = (x * x + y * y).sqrt();
//! // r.val() ≈ 5.0
//! // r.dx() ≈ 0.6  (∂r/∂x = x/r)
//! // r.dy() ≈ 0.8  (∂r/∂y = y/r)
//! ```

// Jet2H still uses its own struct (Hessian = 6 components)
pub mod jet2h;

// PathJet is a different abstraction (ray origin + direction)
pub mod path_jet;

// Re-export types
pub use jet2h::{Jet2H, Jet2HSqrt};
pub use path_jet::PathJet;

// ============================================================================
// Type Aliases for Unified Field<Dual<N>> System
// ============================================================================

use crate::dual::Dual;
use crate::numeric::Numeric;
use crate::Field;

/// 2D Jet: SIMD batch of dual numbers with 2 partial derivatives.
///
/// `Jet2` is a type alias for `Field<Dual<2>>`. When manifolds are evaluated
/// with Jet2 inputs, derivatives propagate automatically via the chain rule.
///
/// ## Methods
///
/// - `Jet2::x(val)` - Create seeded for X differentiation
/// - `Jet2::y(val)` - Create seeded for Y differentiation
/// - `Jet2::constant(val)` - Create constant (zero derivatives)
/// - `jet.val()` - Get the function value
/// - `jet.dx()` - Get ∂f/∂x
/// - `jet.dy()` - Get ∂f/∂y
///
/// ## Example
///
/// ```ignore
/// let x = Jet2::x(Field::from(3.0));
/// let y = Jet2::y(Field::from(4.0));
/// let circle_sdf = (x * x + y * y).sqrt() - Jet2::constant(Field::from(1.0));
/// ```
pub type Jet2 = Field<Dual<2>>;

/// 3D Jet: SIMD batch of dual numbers with 3 partial derivatives.
///
/// `Jet3` is a type alias for `Field<Dual<3>>`. Essential for computing
/// surface normals from SDF gradients in 3D rendering.
///
/// ## Methods
///
/// - `Jet3::x(val)` - Create seeded for X differentiation
/// - `Jet3::y(val)` - Create seeded for Y differentiation
/// - `Jet3::z(val)` - Create seeded for Z differentiation
/// - `Jet3::constant(val)` - Create constant (zero derivatives)
/// - `jet.val()` - Get the function value
/// - `jet.dx()` - Get ∂f/∂x
/// - `jet.dy()` - Get ∂f/∂y
/// - `jet.dz()` - Get ∂f/∂z
pub type Jet3 = Field<Dual<3>>;

/// Wrapper for sqrt(Jet2) that enables automatic rsqrt fusion.
///
/// When `Jet2 / Jet2Sqrt` is computed, this automatically uses the faster
/// `rsqrt` path: `a / sqrt(b)` becomes `a * rsqrt(b)`.
#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct Jet2Sqrt(pub Jet2);

impl Jet2Sqrt {
    /// Evaluate to get the actual sqrt result as Jet2.
    #[inline(always)]
    pub fn eval(self) -> Jet2 {
        self.0.sqrt()
    }
}

impl From<Jet2Sqrt> for Jet2 {
    #[inline(always)]
    fn from(s: Jet2Sqrt) -> Jet2 {
        s.eval()
    }
}

/// Rsqrt fusion: `Jet2 / Jet2Sqrt` computes `a * rsqrt(b)` directly.
impl core::ops::Div<Jet2Sqrt> for Jet2 {
    type Output = Jet2;
    #[inline(always)]
    fn div(self, rhs: Jet2Sqrt) -> Jet2 {
        self * rhs.0.rsqrt()
    }
}

impl core::ops::Add<Jet2Sqrt> for Jet2 {
    type Output = Jet2;
    #[inline(always)]
    fn add(self, rhs: Jet2Sqrt) -> Jet2 {
        self + rhs.eval()
    }
}

impl core::ops::Sub<Jet2Sqrt> for Jet2 {
    type Output = Jet2;
    #[inline(always)]
    fn sub(self, rhs: Jet2Sqrt) -> Jet2 {
        self - rhs.eval()
    }
}

impl core::ops::Mul<Jet2Sqrt> for Jet2 {
    type Output = Jet2;
    #[inline(always)]
    fn mul(self, rhs: Jet2Sqrt) -> Jet2 {
        self * rhs.eval()
    }
}

impl core::ops::Add<Jet2> for Jet2Sqrt {
    type Output = Jet2;
    #[inline(always)]
    fn add(self, rhs: Jet2) -> Jet2 {
        self.eval() + rhs
    }
}

impl core::ops::Sub<Jet2> for Jet2Sqrt {
    type Output = Jet2;
    #[inline(always)]
    fn sub(self, rhs: Jet2) -> Jet2 {
        self.eval() - rhs
    }
}

impl core::ops::Mul<Jet2> for Jet2Sqrt {
    type Output = Jet2;
    #[inline(always)]
    fn mul(self, rhs: Jet2) -> Jet2 {
        self.eval() * rhs
    }
}

impl core::ops::Div<Jet2> for Jet2Sqrt {
    type Output = Jet2;
    #[inline(always)]
    fn div(self, rhs: Jet2) -> Jet2 {
        self.eval() / rhs
    }
}

/// Wrapper for sqrt(Jet3) that enables automatic rsqrt fusion.
#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct Jet3Sqrt(pub Jet3);

impl Jet3Sqrt {
    /// Evaluate to get the actual sqrt result as Jet3.
    #[inline(always)]
    pub fn eval(self) -> Jet3 {
        self.0.sqrt()
    }
}

impl From<Jet3Sqrt> for Jet3 {
    #[inline(always)]
    fn from(s: Jet3Sqrt) -> Jet3 {
        s.eval()
    }
}

impl core::ops::Div<Jet3Sqrt> for Jet3 {
    type Output = Jet3;
    #[inline(always)]
    fn div(self, rhs: Jet3Sqrt) -> Jet3 {
        self * rhs.0.rsqrt()
    }
}

impl core::ops::Add<Jet3Sqrt> for Jet3 {
    type Output = Jet3;
    #[inline(always)]
    fn add(self, rhs: Jet3Sqrt) -> Jet3 {
        self + rhs.eval()
    }
}

impl core::ops::Sub<Jet3Sqrt> for Jet3 {
    type Output = Jet3;
    #[inline(always)]
    fn sub(self, rhs: Jet3Sqrt) -> Jet3 {
        self - rhs.eval()
    }
}

impl core::ops::Mul<Jet3Sqrt> for Jet3 {
    type Output = Jet3;
    #[inline(always)]
    fn mul(self, rhs: Jet3Sqrt) -> Jet3 {
        self * rhs.eval()
    }
}

impl core::ops::Add<Jet3> for Jet3Sqrt {
    type Output = Jet3;
    #[inline(always)]
    fn add(self, rhs: Jet3) -> Jet3 {
        self.eval() + rhs
    }
}

impl core::ops::Sub<Jet3> for Jet3Sqrt {
    type Output = Jet3;
    #[inline(always)]
    fn sub(self, rhs: Jet3) -> Jet3 {
        self.eval() - rhs
    }
}

impl core::ops::Mul<Jet3> for Jet3Sqrt {
    type Output = Jet3;
    #[inline(always)]
    fn mul(self, rhs: Jet3) -> Jet3 {
        self.eval() * rhs
    }
}

impl core::ops::Div<Jet3> for Jet3Sqrt {
    type Output = Jet3;
    #[inline(always)]
    fn div(self, rhs: Jet3) -> Jet3 {
        self.eval() / rhs
    }
}
