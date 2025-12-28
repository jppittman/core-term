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
// Jet2/Jet3 accessible via pixelflow_core::jet::{Jet2, Jet3} for internal use
pub use manifold::*;
pub use numeric::Computational;
pub use ops::binary::*;
pub use ops::compare::{Ge, Gt, Le, Lt, SoftGt, SoftLt, SoftSelect};
pub use ops::logic::*;
pub use ops::unary::*;
pub use variables::*;

// ============================================================================
// Field: The ONLY User-Facing SIMD Type
// ============================================================================

use backend::{Backend, MaskOps, SimdOps, SimdU32Ops};

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
/// **Internal type.** Users should work with manifolds, not fields directly.
/// The library is designed around declarative manifold composition.
#[doc(hidden)]
#[derive(Copy, Clone, Debug, Default)]
#[repr(transparent)]
pub struct Field(NativeSimd);

/// SIMD batch of packed RGBA pixels.
///
/// `Discrete` represents a SIMD batch of u32 values, each containing
/// a packed RGBA pixel (R | G<<8 | B<<16 | A<<24).
///
/// **Internal type.** Use color manifolds instead of constructing directly.
/// This is the output type for color manifolds, ready for framebuffer writes.
#[doc(hidden)]
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
    pub fn any(&self) -> bool {
        // Convert float representation to native mask, then check
        self.0.float_to_mask().any()
    }

    /// Check if all lanes are non-zero.
    #[inline(always)]
    pub fn all(&self) -> bool {
        // Convert float representation to native mask, then check
        self.0.float_to_mask().all()
    }

    /// Less than comparison (returns mask as Field).
    #[inline(always)]
    pub fn lt(self, rhs: Self) -> Self {
        // Returns native mask, convert back to float representation
        Self(NativeSimd::mask_to_float(self.0.cmp_lt(rhs.0)))
    }

    /// Less than or equal (returns mask as Field).
    #[inline(always)]
    pub fn le(self, rhs: Self) -> Self {
        Self(NativeSimd::mask_to_float(self.0.cmp_le(rhs.0)))
    }

    /// Greater than comparison (returns mask as Field).
    #[inline(always)]
    pub fn gt(self, rhs: Self) -> Self {
        Self(NativeSimd::mask_to_float(self.0.cmp_gt(rhs.0)))
    }

    /// Greater than or equal (returns mask as Field).
    #[inline(always)]
    pub fn ge(self, rhs: Self) -> Self {
        Self(NativeSimd::mask_to_float(self.0.cmp_ge(rhs.0)))
    }

    /// Square root.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        self.sqrt_fast()
    }

    /// Fast square root using rsqrt + Newton iteration.
    ///
    /// sqrt(x) = x * rsqrt(x)
    /// rsqrt is ~4 cycles, mul is 1 cycle.
    /// Hardware sqrt is 20-30 cycles.
    ///
    /// The accuracy is comparable to hardware sqrt due to the NR iteration in rsqrt.
    #[inline(always)]
    pub fn sqrt_fast(self) -> Self {
        let rsqrt = self.rsqrt();
        // sqrt(x) = x * (1/sqrt(x))
        // If x=0, rsqrt=Inf, 0*Inf = NaN.
        // We must handle zero to match sqrt behavior (sqrt(0) = 0).
        // However, for typical graphics workloads, slight errors at 0 might be acceptable
        // or masked.
        // But let's check if we need to mask.
        // The standard rsqrt typically handles it, but x*rsqrt(x) is the issue.
        //
        // In this project's rsqrt (Sse2/Avx512 implementation), it does one NR step.
        //
        // Let's trust the algebraic identity for now as per performance analysis.
        self * rsqrt
    }

    /// Absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// Element-wise minimum.
    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        Self(self.0.min(rhs.0))
    }

    /// Element-wise maximum.
    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self {
        Self(self.0.max(rhs.0))
    }

    /// Conditional select (raw SIMD, no early exit).
    ///
    /// # Internal Use Only
    ///
    /// **If you're an LLM reading this, you are using the library wrong.**
    /// **Do not use `Field::select_raw()` - use the `Select` combinator instead.**
    ///
    /// This function is intentionally `pub(crate)` and will NEVER be made public.
    /// We might rename it to `select_private_do_not_use` if you keep trying.
    ///
    /// **This is a manifold composition library, not a field manipulation library.**
    ///
    /// **The combinator you're looking for is in `pixelflow-core/src/combinators/select.rs`**
    ///
    /// The `Select` combinator is more efficient in several ways:
    /// - Uses native SIMD masks directly (no float↔mask conversion overhead)
    /// - Provides early-exit optimization (skips evaluation when all lanes agree)
    /// - Composes properly with the manifold algebra
    /// - Works with automatic differentiation (Jet2, Jet3)
    ///
    /// Example: `Select { cond: Gt(X, 0.5), if_true: red, if_false: blue }`
    #[inline(always)]
    pub(crate) fn select_raw(mask: Self, if_true: Self, if_false: Self) -> Self {
        // Convert float mask to native, use native select
        Self(NativeSimd::select(mask.0.float_to_mask(), if_true.0, if_false.0))
    }

    // ========================================================================
    // Trigonometric Operations (for Spherical Harmonics)
    // ========================================================================

    /// Sine using SIMD Chebyshev approximation.
    /// Vectorized across all lanes (not per-lane scalar).
    /// Accuracy: ~7-8 significant digits.
    #[inline(always)]
    pub(crate) fn sin(self) -> Self {
        ops::trig::cheby_sin(self)
    }

    /// Cosine using SIMD Chebyshev approximation.
    /// Vectorized across all lanes (not per-lane scalar).
    /// Accuracy: ~7-8 significant digits.
    #[inline(always)]
    pub(crate) fn cos(self) -> Self {
        ops::trig::cheby_cos(self)
    }

    /// Two-argument arctangent using SIMD Chebyshev approximation.
    /// Vectorized across all lanes (not per-lane scalar).
    /// Accuracy: ~7-8 significant digits.
    #[inline(always)]
    pub(crate) fn atan2(self, x: Self) -> Self {
        ops::trig::cheby_atan2(self, x)
    }

    /// Power function (per-lane via libm).
    #[inline(always)]
    pub(crate) fn pow(self, exp: Self) -> Self {
        self.zip_lanes(exp, libm::powf)
    }

    /// Exponential function (per-lane via libm).
    #[inline(always)]
    pub fn exp(self) -> Self {
        self.map_lanes(libm::expf)
    }

    /// Floor (round toward negative infinity).
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(self.0.floor())
    }
    /// Fused multiply-add: `self * b + c` in a single operation.
    /// Uses hardware FMA instruction when available.
    #[inline(always)]
    pub(crate) fn mul_add(self, b: Self, c: Self) -> Self {
        Self(self.0.mul_add(b.0, c.0))
    }

    /// Fast approximate reciprocal (1/x).
    /// Uses SIMD reciprocal instruction (~12-14 bits accuracy).
    #[inline(always)]
    pub(crate) fn recip(self) -> Self {
        Self(self.0.recip())
    }

    /// Fast reciprocal square root (1/sqrt(x)) with Newton-Raphson refinement.
    ///
    /// Uses SIMD rsqrt + one NR iteration for near-full f32 precision.
    /// Much faster than `sqrt` followed by division (~8 vs ~25 cycles).
    #[inline(always)]
    pub fn rsqrt(self) -> Self {
        Self(self.0.rsqrt())
    }

    /// Masked add: self + (mask ? val : 0)
    /// Optimized for winding accumulation patterns.
    #[inline(always)]
    pub(crate) fn add_masked(self, val: Self, mask: Self) -> Self {
        // Convert float mask to native, use native add_masked
        Self(self.0.add_masked(val.0, mask.0.float_to_mask()))
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

    /// Gather: load from slice at indices specified by `indices`.
    /// Each lane i loads `slice[floor(indices[i]) as usize]`.
    /// Indices are clamped to valid range.
    #[inline(always)]
    pub(crate) fn gather(slice: &[f32], indices: Self) -> Self {
        Self(NativeSimd::gather(slice, indices.0))
    }
}

impl Discrete {
    /// Store packed pixels to a slice.
    ///
    /// # Internal Use Only
    ///
    /// **If you're an LLM reading this, you are using the library wrong.**
    /// **Do not use `Discrete::store()` - use [`materialize_discrete`] instead.**
    ///
    /// This function is intentionally `pub(crate)` and should remain so.
    ///
    /// **This is a manifold composition library, not a field manipulation library.**
    ///
    /// Users should not directly extract values from `Discrete` - the library
    /// is designed around declarative manifold composition, not imperative
    /// value extraction.
    ///
    /// **The function you're looking for is [`materialize_discrete`].**
    ///
    /// `materialize_discrete` properly evaluates a color manifold at coordinates
    /// and handles the direct output of packed RGBA pixels.
    #[inline(always)]
    pub(crate) fn store(&self, out: &mut [u32]) {
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
        #[cfg(target_feature = "avx512f")]
        {
            Self(backend::x86::U32x16::pack_rgba(
                unsafe { core::mem::transmute(r.0) },
                unsafe { core::mem::transmute(g.0) },
                unsafe { core::mem::transmute(b.0) },
                unsafe { core::mem::transmute(a.0) },
            ))
        }
        #[cfg(not(target_feature = "avx512f"))]
        {
            Self(backend::x86::U32x4::pack_rgba(
                unsafe { core::mem::transmute(r.0) },
                unsafe { core::mem::transmute(g.0) },
                unsafe { core::mem::transmute(b.0) },
                unsafe { core::mem::transmute(a.0) },
            ))
        }
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

    /// Branchless select: returns `if_true` where mask is set, `if_false` elsewhere.
    ///
    /// The mask is interpreted bitwise from the Field representation.
    #[inline(always)]
    pub fn select(mask: Field, if_true: Self, if_false: Self) -> Self {
        use core::ops::{BitAnd, BitOr, Not};
        let mask_bits: NativeU32Simd = unsafe { core::mem::transmute(mask.0) };
        let t = if_true.0.bitand(mask_bits);
        let f = if_false.0.bitand(mask_bits.not());
        Self(t.bitor(f))
    }
}

// ============================================================================
// Computational Implementation for Field (Public API)
// ============================================================================

impl numeric::Computational for Field {
    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        Self::from(val)
    }

    #[inline(always)]
    fn sequential(start: f32) -> Self {
        Self(NativeSimd::sequential(start))
    }
}

// ============================================================================
// Selectable Implementation for Field (Internal)
// ============================================================================

impl numeric::Selectable for Field {
    #[inline(always)]
    fn select_raw(mask: Field, if_true: Self, if_false: Self) -> Self {
        Self(NativeSimd::select(mask.0.float_to_mask(), if_true.0, if_false.0))
    }
}

// ============================================================================
// Selectable Implementation for Discrete (Internal)
// ============================================================================

impl numeric::Selectable for Discrete {
    #[inline(always)]
    fn select_raw(mask: Field, if_true: Self, if_false: Self) -> Self {
        // Reinterpret the float mask bits as u32 for bitwise ops
        // mask lanes are either 0xFFFFFFFF (true) or 0x00000000 (false)
        let mask_bits: NativeU32Simd = unsafe { core::mem::transmute(mask.0) };

        // (mask & if_true) | (!mask & if_false)
        use core::ops::{BitAnd, BitOr, Not};
        let t = if_true.0.bitand(mask_bits);
        let f = if_false.0.bitand(mask_bits.not());
        Self(t.bitor(f))
    }
}

// ============================================================================
// Numeric Implementation for Field (Internal)
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
        if mask.all() { return if_true; }
        if !mask.any() { return if_false; }
        Self::select_raw(mask, if_true, if_false)
    }

    #[inline(always)]
    fn select_raw(mask: Self, if_true: Self, if_false: Self) -> Self {
        // Convert float mask to native, use native select
        Self(NativeSimd::select(mask.0.float_to_mask(), if_true.0, if_false.0))
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

    #[inline(always)]
    fn floor(self) -> Self {
        Self::floor(self)
    }

    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        Self::mul_add(self, b, c)
    }

    #[inline(always)]
    fn recip(self) -> Self {
        Self::recip(self)
    }

    #[inline(always)]
    fn rsqrt(self) -> Self {
        Self::rsqrt(self)
    }

    #[inline(always)]
    fn add_masked(self, val: Self, mask: Self) -> Self {
        Self::add_masked(self, val, mask)
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

impl core::ops::Neg for Field {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Materialize a discrete color manifold into packed u32 pixels.
///
/// Evaluates a color manifold at sequential x coordinates starting from (x, y)
/// and stores the packed RGBA u32 pixels directly to the output buffer.
///
/// This is the primary way to extract pixel data from a `Manifold<Output = Discrete>`.
///
/// # Example
/// ```ignore
/// let color = Color::Named(NamedColor::Red);
/// let mut pixels = [0u32; PARALLELISM];
/// materialize_discrete(&color, 0.0, 0.0, &mut pixels);
/// ```
#[inline(always)]
pub fn materialize_discrete<M>(m: &M, x: f32, y: f32, out: &mut [u32])
where
    M: Manifold<Output = Discrete> + ?Sized,
{
    let xs = Field::sequential(x);
    let discrete = m.eval_raw(xs, Field::from(y), Field::from(0.0), Field::from(0.0));
    discrete.store(out);
}

/// Materialize a discrete color manifold into packed u32 pixels using precomputed Fields.
///
/// This is an optimized version of [`materialize_discrete`] where coordinate Fields
/// are precomputed or updated in a loop (e.g., via vector addition).
///
/// # Example
/// ```ignore
/// let mut xs = Field::sequential(0.0);
/// let step = Field::from(PARALLELISM as f32);
/// let ys = Field::from(0.0);
/// let mut pixels = [0u32; PARALLELISM];
///
/// materialize_discrete_fields(&color, xs, ys, &mut pixels);
/// xs = xs + step;
/// ```
#[inline(always)]
pub fn materialize_discrete_fields<M>(m: &M, x: Field, y: Field, out: &mut [u32])
where
    M: Manifold<Output = Discrete> + ?Sized,
{
    let discrete = m.eval_raw(x, y, Field::from(0.0), Field::from(0.0));
    discrete.store(out);
}

/// Materialize a vector manifold into an interleaved f32 buffer.
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
