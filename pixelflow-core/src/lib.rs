//! # PixelFlow Core
//!
//! A minimal lambda calculus EDSL over SIMD fields.
//!
//! The type system IS the AST. `Field` is the computational substrate.
//! `Manifold` is the core abstraction: a function from coordinates to values.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

extern crate alloc;

// ============================================================================
// Modules
// ============================================================================

/// SIMD backend abstractions (completely private).
mod backend;

/// Numeric trait for computational substrate (private).
mod numeric;

/// Jet types for automatic differentiation.
pub mod jet;

/// The core Manifold trait.
pub mod manifold;

/// Coordinate variables (X, Y, Z, W).
pub mod variables;

/// Arithmetic and logic operations.
pub mod ops;

/// Combinators (Select, Fix).
pub mod combinators;

/// Fluent API extensions.
pub mod ext;

// ============================================================================
// Re-exports (The "Prelude")
// ============================================================================

pub use combinators::*;
pub use ext::*;
pub use jet::Jet2;
pub use manifold::*;
pub use ops::*;
pub use variables::*;

// ============================================================================
// Field: The ONLY User-Facing SIMD Type
// ============================================================================

use backend::{Backend, SimdOps};

#[cfg(target_arch = "x86_64")]
type NativeSimd = <backend::x86::Avx512 as Backend>::F32;

#[cfg(target_arch = "aarch64")]
type NativeSimd = <backend::arm::Neon as Backend>::F32;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
type NativeSimd = <backend::scalar::Scalar as Backend>::F32;

/// The computational substrate.
///
/// `Field` represents a SIMD batch of floating-point values.
/// This is the concrete type that manifolds evaluate to.
///
/// Users never see the internal SIMD representation.
/// Create Fields via `From<f32>` or `From<i32>`.
#[derive(Copy, Clone, Debug, Default)]
#[repr(transparent)]
pub struct Field(NativeSimd);

impl Field {
    /// Create sequential values [start, start+1, start+2, ...].
    #[inline(always)]
    pub(crate) fn sequential(start: f32) -> Self {
        Self(NativeSimd::sequential(start))
    }

    /// Store values to a slice.
    #[inline(always)]
    pub(crate) fn store(&self, out: &mut [f32]) {
        self.0.store(out)
    }

    /// Check if any lane is non-zero.
    #[inline(always)]
    pub(crate) fn any(&self) -> bool {
        self.0.any()
    }

    /// Check if all lanes are non-zero.
    #[inline(always)]
    pub(crate) fn all(&self) -> bool {
        self.0.all()
    }

    /// Less than comparison (returns mask).
    #[inline(always)]
    pub(crate) fn lt(self, rhs: Self) -> Self {
        Self(self.0.cmp_lt(rhs.0))
    }

    /// Less than or equal (returns mask).
    #[inline(always)]
    pub(crate) fn le(self, rhs: Self) -> Self {
        Self(self.0.cmp_le(rhs.0))
    }

    /// Greater than comparison (returns mask).
    #[inline(always)]
    pub(crate) fn gt(self, rhs: Self) -> Self {
        Self(self.0.cmp_gt(rhs.0))
    }

    /// Greater than or equal (returns mask).
    #[inline(always)]
    pub(crate) fn ge(self, rhs: Self) -> Self {
        Self(self.0.cmp_ge(rhs.0))
    }

    /// Square root.
    #[inline(always)]
    pub(crate) fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }

    /// Absolute value.
    #[inline(always)]
    pub(crate) fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// Element-wise minimum.
    #[inline(always)]
    pub(crate) fn min(self, rhs: Self) -> Self {
        Self(self.0.min(rhs.0))
    }

    /// Element-wise maximum.
    #[inline(always)]
    pub(crate) fn max(self, rhs: Self) -> Self {
        Self(self.0.max(rhs.0))
    }

    /// Conditional select.
    #[inline(always)]
    pub(crate) fn select(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(NativeSimd::select(mask.0, if_true.0, if_false.0))
    }
}

// ============================================================================
// Numeric Implementation for Field
// ============================================================================

impl numeric::Numeric for Field {
    #[inline(always)]
    fn sqrt(self) -> Self {
        Self::sqrt(self)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self::abs(self)
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        Self::min(self, rhs)
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        Self::max(self, rhs)
    }

    #[inline(always)]
    fn lt(self, rhs: Self) -> Self {
        Self::lt(self, rhs)
    }

    #[inline(always)]
    fn le(self, rhs: Self) -> Self {
        Self::le(self, rhs)
    }

    #[inline(always)]
    fn gt(self, rhs: Self) -> Self {
        Self::gt(self, rhs)
    }

    #[inline(always)]
    fn ge(self, rhs: Self) -> Self {
        Self::ge(self, rhs)
    }

    #[inline(always)]
    fn select(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self::select(mask, if_true, if_false)
    }

    #[inline(always)]
    fn any(&self) -> bool {
        Self::any(self)
    }

    #[inline(always)]
    fn all(&self) -> bool {
        Self::all(self)
    }

    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        Self::from(val)
    }

    #[inline(always)]
    fn from_i32(val: i32) -> Self {
        Self::from(val)
    }
}

// ============================================================================
// From Implementations (the ONLY way to create Field from scalars)
// ============================================================================

impl From<f32> for Field {
    #[inline(always)]
    fn from(val: f32) -> Self {
        Self(NativeSimd::splat(val))
    }
}

impl From<i32> for Field {
    #[inline(always)]
    fn from(val: i32) -> Self {
        Self(NativeSimd::splat(val as f32))
    }
}

// ============================================================================
// Operator Implementations
// ============================================================================

impl core::ops::Add for Field {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl core::ops::Sub for Field {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl core::ops::Mul for Field {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl core::ops::Div for Field {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(self.0 / rhs.0)
    }
}

impl core::ops::BitAnd for Field {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl core::ops::BitOr for Field {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl core::ops::Not for Field {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(!self.0)
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Materialize a scalar manifold into a buffer.
///
/// Evaluates at sequential x coordinates starting from (x, y).
#[inline(always)]
pub fn materialize<M>(m: &M, x: f32, y: f32, out: &mut [f32])
where
    M: Manifold<Output = Field>,
{
    let xs = Field::sequential(x);
    let val = m.eval(xs, Field::from(y), Field::from(0.0), Field::from(0.0));
    val.store(out);
}

/// Materialize a vector manifold into interleaved output.
///
/// Evaluates at sequential x coordinates starting from (x, y), then transposes
/// from SoA (structure of arrays) to AoS (array of structures) for storage.
///
/// Output is interleaved: [x0,y0,z0,w0, x1,y1,z1,w1, ...]
#[inline(always)]
pub fn materialize_vector<M, V>(m: &M, x: f32, y: f32, out: &mut [f32])
where
    M: Manifold<Output = V>,
    V: ops::Vector<Component = Field>,
{
    let xs = Field::sequential(x);
    let val = m.eval_raw(xs, Field::from(y), Field::from(0.0), Field::from(0.0));

    // Store each component to temporary buffers
    let mut buf_x = [0.0f32; PARALLELISM];
    let mut buf_y = [0.0f32; PARALLELISM];
    let mut buf_z = [0.0f32; PARALLELISM];
    let mut buf_w = [0.0f32; PARALLELISM];

    val.get(variables::Axis::X).store(&mut buf_x);
    val.get(variables::Axis::Y).store(&mut buf_y);
    val.get(variables::Axis::Z).store(&mut buf_z);
    val.get(variables::Axis::W).store(&mut buf_w);

    // Transpose: SoA → AoS (interleaved)
    for i in 0..PARALLELISM {
        let base = i * 4;
        if base + 3 < out.len() {
            out[base] = buf_x[i];
            out[base + 1] = buf_y[i];
            out[base + 2] = buf_z[i];
            out[base + 3] = buf_w[i];
        }
    }
}

/// Parallelism width (number of lanes).
pub const PARALLELISM: usize = NativeSimd::LANES;
