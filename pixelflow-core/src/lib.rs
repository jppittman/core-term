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

/// SIMD backend abstractions.
pub mod backend;

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
pub use numeric::Numeric;
pub use ops::binary::*;
pub use ops::compare::{Ge, Gt, Le, Lt, SoftGt, SoftLt, SoftSelect};
pub use ops::logic::*;
pub use ops::unary::*;
pub use variables::*;

// ============================================================================
// Field: The ONLY User-Facing SIMD Type
// ============================================================================

use backend::{Backend, SimdOps, SimdU32Ops};

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
type NativeSimd = <backend::x86::Avx512 as Backend>::F32;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
type NativeU32Simd = <backend::x86::Avx512 as Backend>::U32;

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
type NativeSimd = <backend::x86::Sse2 as Backend>::F32;
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
type NativeU32Simd = <backend::x86::Sse2 as Backend>::U32;

#[cfg(target_arch = "aarch64")]
type NativeSimd = <backend::arm::Neon as Backend>::F32;
#[cfg(target_arch = "aarch64")]
type NativeU32Simd = <backend::arm::Neon as Backend>::U32;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
type NativeSimd = <backend::scalar::Scalar as Backend>::F32;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
type NativeU32Simd = <backend::scalar::Scalar as Backend>::U32;

/// The computational substrate for continuous values.
///
/// `Field` represents a SIMD batch of floating-point values.
/// This is the concrete type that manifolds evaluate to.
///
/// Users never see the internal SIMD representation.
/// Create Fields via `From<f32>` or `From<i32>`.
#[derive(Copy, Clone, Debug, Default)]
#[repr(transparent)]
pub struct Field(NativeSimd);

/// SIMD batch of packed RGBA pixels.
///
/// `Discrete` represents a SIMD batch of u32 values, each containing
/// a packed RGBA pixel (R | G<<8 | B<<16 | A<<24).
///
/// This is the output type for color manifolds, ready for framebuffer writes.
#[derive(Copy, Clone, Debug, Default)]
#[repr(transparent)]
pub struct Discrete(NativeU32Simd);

impl Field {
    /// Create sequential values [start, start+1, start+2, ...].
    ///
    /// Creates a SIMD vector with lane values:
    /// `[start, start+1, start+2, ..., start+(PARALLELISM-1)]`
    ///
    /// This is the efficient way to create sequential x-coordinates
    /// for rasterization loops.
    #[inline(always)]
    pub fn sequential(start: f32) -> Self {
        Self(NativeSimd::sequential(start))
    }

    /// Store values to a slice.
    ///
    /// # Internal Use Only
    ///
    /// **If you're reading this, you're trying to use the library wrong.**
    ///
    /// This function is intentionally `pub(crate)` and should remain so.
    /// Users should not directly extract values from `Field` - the library
    /// is designed around declarative manifold composition, not imperative
    /// value extraction.
    ///
    /// **The function you're looking for is [`materialize`] in lib.rs.**
    ///
    /// `materialize` properly evaluates a manifold at coordinates and handles
    /// the SoA-to-AoS transpose required for interleaved output buffers.
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
    pub fn field_min(self, rhs: Self) -> Self {
        Self(self.0.min(rhs.0))
    }

    /// Element-wise maximum.
    #[inline(always)]
    pub fn field_max(self, rhs: Self) -> Self {
        Self(self.0.max(rhs.0))
    }

    /// Conditional select.
    #[inline(always)]
    pub(crate) fn select(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self(NativeSimd::select(mask.0, if_true.0, if_false.0))
    }

    // ========================================================================
    // Trigonometric Operations (for Spherical Harmonics)
    // ========================================================================

    /// Sine (per-lane via libm).
    #[inline(always)]
    pub(crate) fn sin(self) -> Self {
        self.map_lanes(libm::sinf)
    }

    /// Cosine (per-lane via libm).
    #[inline(always)]
    pub(crate) fn cos(self) -> Self {
        self.map_lanes(libm::cosf)
    }

    /// Two-argument arctangent (per-lane via libm).
    #[inline(always)]
    pub(crate) fn atan2(self, x: Self) -> Self {
        self.zip_lanes(x, libm::atan2f)
    }

    /// Power function (per-lane via libm).
    #[inline(always)]
    pub(crate) fn pow(self, exp: Self) -> Self {
        self.zip_lanes(exp, libm::powf)
    }

    /// Exponential function (per-lane via libm).
    #[inline(always)]
    pub(crate) fn exp(self) -> Self {
        self.map_lanes(libm::expf)
    }

    /// Apply a unary function to each lane.
    #[inline(always)]
    pub(crate) fn map_lanes(self, f: fn(f32) -> f32) -> Self {
        let mut buf = [0.0f32; PARALLELISM];
        self.store(&mut buf);
        for v in buf.iter_mut() {
            *v = f(*v);
        }
        // Reconstruct from buffer
        Self::from_slice(&buf)
    }

    /// Apply a binary function to each pair of lanes.
    #[inline(always)]
    fn zip_lanes(self, other: Self, f: fn(f32, f32) -> f32) -> Self {
        let mut buf_a = [0.0f32; PARALLELISM];
        let mut buf_b = [0.0f32; PARALLELISM];
        self.store(&mut buf_a);
        other.store(&mut buf_b);
        for i in 0..PARALLELISM {
            buf_a[i] = f(buf_a[i], buf_b[i]);
        }
        Self::from_slice(&buf_a)
    }

    /// Load from a slice.
    #[inline(always)]
    fn from_slice(slice: &[f32]) -> Self {
        Self(NativeSimd::from_slice(slice))
    }
}

impl Discrete {
    /// Store packed pixels to a slice.
    ///
    /// Stores the packed RGBA u32 pixel values to the output buffer.
    /// This is the final output stage for color manifolds after packing.
    #[inline(always)]
    pub fn store(&self, out: &mut [u32]) {
        self.0.store(out)
    }

    /// Pack 4 Fields (RGBA, 0.0-1.0) into packed u32 pixels.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub fn pack(r: Field, g: Field, b: Field, a: Field) -> Self {
        Self(backend::arm::U32x4::pack_rgba(
            unsafe { core::mem::transmute(r.0) },
            unsafe { core::mem::transmute(g.0) },
            unsafe { core::mem::transmute(b.0) },
            unsafe { core::mem::transmute(a.0) },
        ))
    }

    /// Pack 4 Fields (RGBA, 0.0-1.0) into packed u32 pixels.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub fn pack(r: Field, g: Field, b: Field, a: Field) -> Self {
        Self(backend::x86::U32x4::pack_rgba(
            // SAFETY: F32x4 and Field have the same repr(transparent) layout
            unsafe { core::mem::transmute(r.0) },
            unsafe { core::mem::transmute(g.0) },
            unsafe { core::mem::transmute(b.0) },
            unsafe { core::mem::transmute(a.0) },
        ))
    }

    /// Pack 4 Fields (RGBA, 0.0-1.0) into packed u32 pixels.
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    #[inline(always)]
    pub fn pack(r: Field, g: Field, b: Field, _a: Field) -> Self {
        // Scalar fallback - only packs first element
        let mut r_buf = [0.0f32; 1];
        let mut g_buf = [0.0f32; 1];
        let mut b_buf = [0.0f32; 1];
        let mut a_buf = [0.0f32; 1];
        r.store(&mut r_buf);
        g.store(&mut g_buf);
        b.store(&mut b_buf);
        _a.store(&mut a_buf);

        let r_u8 = (r_buf[0].clamp(0.0, 1.0) * 255.0) as u32;
        let g_u8 = (g_buf[0].clamp(0.0, 1.0) * 255.0) as u32;
        let b_u8 = (b_buf[0].clamp(0.0, 1.0) * 255.0) as u32;
        let a_u8 = (a_buf[0].clamp(0.0, 1.0) * 255.0) as u32;
        let packed = r_u8 | (g_u8 << 8) | (b_u8 << 16) | (a_u8 << 24);

        Self(backend::scalar::ScalarU32::splat(packed))
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
        Self::field_min(self, rhs)
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        Self::field_max(self, rhs)
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

    #[inline(always)]
    fn from_field(field: Field) -> Self {
        field
    }

    #[inline(always)]
    fn sin(self) -> Self {
        Self::sin(self)
    }

    #[inline(always)]
    fn cos(self) -> Self {
        Self::cos(self)
    }

    #[inline(always)]
    fn atan2(self, x: Self) -> Self {
        Self::atan2(self, x)
    }

    #[inline(always)]
    fn pow(self, exp: Self) -> Self {
        Self::pow(self, exp)
    }

    #[inline(always)]
    fn exp(self) -> Self {
        Self::exp(self)
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

/// Materialize a color manifold into an interleaved buffer.
///
/// Evaluates at sequential x coordinates starting from (x, y), then transposes
/// from SoA (structure of arrays) to AoS (array of structures) for storage.
///
/// Output is interleaved RGBA: [r0,g0,b0,a0, r1,g1,b1,a1, ...]
///
/// For scalar manifolds, use `Lift` to convert them to a uniform color first.
#[inline(always)]
pub fn materialize<M, V>(m: &M, x: f32, y: f32, out: &mut [f32])
where
    M: Manifold<Output = V> + ?Sized,
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
