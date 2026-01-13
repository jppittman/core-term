//! # Select Combinator
//!
//! Branchless conditional with short-circuit evaluation.
//!
//! ## Automatic Optimization for Comparisons
//!
//! When the condition is a comparison (`Lt`, `Gt`, `Le`, `Ge`, or combinations
//! with `And`/`Or`), the implementation automatically uses native masks:
//!
//! ```text
//! // Old path (wasteful on AVX-512):
//! cmp_lt → __mmask16 → mask_to_float → Field → float_to_mask → __mmask16 → select
//!
//! // Automatic optimization:
//! cmp_lt → __mmask16 → select
//! ```
//!
//! This happens transparently - just use `Select { cond: Lt(...), ... }`.

use crate::Field;
use crate::Manifold;
use crate::backend::{MaskOps, SimdOps};
use crate::numeric::Numeric;
use crate::ops::compare::{Ge, Gt, Le, Lt};
use crate::ops::logic::{And, BNot, Or};

/// Branchless conditional with short-circuit.
///
/// Evaluates `if_true` only when some lanes are true,
/// `if_false` only when some are false.
#[derive(Clone, Debug)]
pub struct Select<C, T, F> {
    /// The condition (produces a mask).
    pub cond: C,
    /// Value when condition is true.
    pub if_true: T,
    /// Value when condition is false.
    pub if_false: F,
}

// ============================================================================
// Native SIMD types for Field
// ============================================================================

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
type NativeSimd = crate::backend::x86::F32x16;
#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    target_feature = "avx2"
))]
type NativeSimd = crate::backend::x86::F32x8;
#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    not(target_feature = "avx2")
))]
type NativeSimd = crate::backend::x86::F32x4;
#[cfg(target_arch = "aarch64")]
type NativeSimd = crate::backend::arm::F32x4;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
type NativeSimd = crate::backend::scalar::ScalarF32;

type NativeMask = <NativeSimd as SimdOps>::Mask;

// ============================================================================
// FieldCondition trait - returns native masks for optimal performance
// ============================================================================

/// Trait for conditions that can be evaluated to native masks for Field.
///
/// This is automatically implemented for comparison types (Lt, Gt, Le, Ge)
/// and their boolean combinations (And, Or, Not).
pub trait FieldCondition: Send + Sync {
    /// Evaluate the condition, returning a native mask.
    fn eval_mask(&self, x: Field, y: Field, z: Field, w: Field) -> NativeMask;
}

// ============================================================================
// FieldCondition implementations for comparison types
// ============================================================================

impl<L, R> FieldCondition for Lt<L, R>
where
    L: Manifold<Field, Output = Field>,
    R: Manifold<Field, Output = Field>,
{
    #[inline(always)]
    fn eval_mask(&self, x: Field, y: Field, z: Field, w: Field) -> NativeMask {
        let left = self.0.eval_raw(x, y, z, w);
        let right = self.1.eval_raw(x, y, z, w);
        left.0.cmp_lt(right.0)
    }
}

impl<L, R> FieldCondition for Gt<L, R>
where
    L: Manifold<Field, Output = Field>,
    R: Manifold<Field, Output = Field>,
{
    #[inline(always)]
    fn eval_mask(&self, x: Field, y: Field, z: Field, w: Field) -> NativeMask {
        let left = self.0.eval_raw(x, y, z, w);
        let right = self.1.eval_raw(x, y, z, w);
        left.0.cmp_gt(right.0)
    }
}

impl<L, R> FieldCondition for Le<L, R>
where
    L: Manifold<Field, Output = Field>,
    R: Manifold<Field, Output = Field>,
{
    #[inline(always)]
    fn eval_mask(&self, x: Field, y: Field, z: Field, w: Field) -> NativeMask {
        let left = self.0.eval_raw(x, y, z, w);
        let right = self.1.eval_raw(x, y, z, w);
        left.0.cmp_le(right.0)
    }
}

impl<L, R> FieldCondition for Ge<L, R>
where
    L: Manifold<Field, Output = Field>,
    R: Manifold<Field, Output = Field>,
{
    #[inline(always)]
    fn eval_mask(&self, x: Field, y: Field, z: Field, w: Field) -> NativeMask {
        let left = self.0.eval_raw(x, y, z, w);
        let right = self.1.eval_raw(x, y, z, w);
        left.0.cmp_ge(right.0)
    }
}

// ============================================================================
// FieldCondition for boolean combinations
// ============================================================================

impl<L, R> FieldCondition for And<L, R>
where
    L: FieldCondition,
    R: FieldCondition,
{
    #[inline(always)]
    fn eval_mask(&self, x: Field, y: Field, z: Field, w: Field) -> NativeMask {
        self.0.eval_mask(x, y, z, w) & self.1.eval_mask(x, y, z, w)
    }
}

impl<L, R> FieldCondition for Or<L, R>
where
    L: FieldCondition,
    R: FieldCondition,
{
    #[inline(always)]
    fn eval_mask(&self, x: Field, y: Field, z: Field, w: Field) -> NativeMask {
        self.0.eval_mask(x, y, z, w) | self.1.eval_mask(x, y, z, w)
    }
}

impl<T> FieldCondition for BNot<T>
where
    T: FieldCondition,
{
    #[inline(always)]
    fn eval_mask(&self, x: Field, y: Field, z: Field, w: Field) -> NativeMask {
        !self.0.eval_mask(x, y, z, w)
    }
}

// ============================================================================
// FieldCondition for Abs<T> (common in threshold patterns like |winding| >= 0.5)
// ============================================================================

impl<T> FieldCondition for crate::ops::unary::Abs<T>
where
    T: Manifold<Field, Output = Field>,
{
    #[inline(always)]
    fn eval_mask(&self, x: Field, y: Field, z: Field, w: Field) -> NativeMask {
        // Abs doesn't produce a mask directly, so this shouldn't be called.
        // But we need it for nested expressions like Ge(Abs(m), 0.5)
        // In that case, Ge's eval_mask handles it properly.
        // This impl is for cases where Abs is used as a bare condition (unusual).
        let val = self.0.eval_raw(x, y, z, w);
        val.0.float_to_mask()
    }
}

// Field itself can be a condition (mask stored as float bits).
// Used when Field::gt/lt/etc. returns a Field mask directly.
impl FieldCondition for Field {
    #[inline(always)]
    fn eval_mask(&self, _x: Field, _y: Field, _z: Field, _w: Field) -> NativeMask {
        self.0.float_to_mask()
    }
}

// ============================================================================
// Select implementation for Field with FieldCondition (OPTIMIZED PATH)
// ============================================================================

impl<C, T, F, O> Manifold<Field> for Select<C, T, F>
where
    O: crate::numeric::Selectable,
    C: FieldCondition,
    T: Manifold<Field, Output = O>,
    F: Manifold<Field, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> O {
        // Get native mask directly - no float conversion!
        let mask = self.cond.eval_mask(x, y, z, w);

        // Early exit using native mask ops (free on AVX-512 k-registers)
        if mask.all() {
            return self.if_true.eval_raw(x, y, z, w);
        }
        if !mask.any() {
            return self.if_false.eval_raw(x, y, z, w);
        }

        // Select using Selectable trait
        // Convert native mask back to Field (float mask) for the generic interface
        let mask_field = Field(NativeSimd::mask_to_float(mask));
        let true_val = self.if_true.eval_raw(x, y, z, w);
        let false_val = self.if_false.eval_raw(x, y, z, w);

        O::select_raw(mask_field, true_val, false_val)
    }
}

// ============================================================================
// Generic implementation for non-Field types (Jet2, etc.)
// ============================================================================

impl<C, T, F, O> Manifold<crate::jet::Jet2> for Select<C, T, F>
where
    O: Numeric,
    C: Manifold<crate::jet::Jet2, Output = O>,
    T: Manifold<crate::jet::Jet2, Output = O>,
    F: Manifold<crate::jet::Jet2, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval_raw(
        &self,
        x: crate::jet::Jet2,
        y: crate::jet::Jet2,
        z: crate::jet::Jet2,
        w: crate::jet::Jet2,
    ) -> O {
        let mask = self.cond.eval_raw(x, y, z, w);
        if mask.all() {
            return self.if_true.eval_raw(x, y, z, w);
        }
        if !mask.any() {
            return self.if_false.eval_raw(x, y, z, w);
        }

        let true_val = self.if_true.eval_raw(x, y, z, w);
        let false_val = self.if_false.eval_raw(x, y, z, w);
        O::select_raw(mask, true_val, false_val)
    }
}

// ============================================================================
// Jet3 with Selectable output (for Discrete in the mullet architecture)
// ============================================================================

impl<C, T, F, O> Manifold<crate::jet::Jet3> for Select<C, T, F>
where
    O: crate::numeric::Selectable,
    C: Manifold<crate::jet::Jet3, Output = crate::jet::Jet3>,
    T: Manifold<crate::jet::Jet3, Output = O>,
    F: Manifold<crate::jet::Jet3, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval_raw(
        &self,
        x: crate::jet::Jet3,
        y: crate::jet::Jet3,
        z: crate::jet::Jet3,
        w: crate::jet::Jet3,
    ) -> O {
        let cond_jet = self.cond.eval_raw(x, y, z, w);
        let mask = cond_jet.val; // Extract Field mask from Jet3

        // Early exit using Field's any/all
        if mask.all() {
            return self.if_true.eval_raw(x, y, z, w);
        }
        if !mask.any() {
            return self.if_false.eval_raw(x, y, z, w);
        }

        let true_val = self.if_true.eval_raw(x, y, z, w);
        let false_val = self.if_false.eval_raw(x, y, z, w);
        O::select_raw(mask, true_val, false_val)
    }
}
