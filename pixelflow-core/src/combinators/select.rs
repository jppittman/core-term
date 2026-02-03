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
use pixelflow_compiler::Element;

/// Branchless conditional with short-circuit.
///
/// Evaluates `if_true` only when some lanes are true,
/// `if_false` only when some are false.
#[derive(Clone, Debug, Element)]
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
pub trait FieldCondition<P>: Send + Sync {
    /// Evaluate the condition, returning a native mask.
    fn eval_mask(&self, p: P) -> NativeMask;
}

// ============================================================================
// FieldCondition implementations for comparison types (4D Field domains)
// ============================================================================

type Field4 = (Field, Field, Field, Field);

impl<L, R> FieldCondition<Field4> for Lt<L, R>
where
    L: Manifold<Field4, Output = Field>,
    R: Manifold<Field4, Output = Field>,
{
    #[inline(always)]
    fn eval_mask(&self, p: Field4) -> NativeMask {
        let left = self.0.eval(p);
        let right = self.1.eval(p);
        left.0.cmp_lt(right.0)
    }
}

impl<L, R> FieldCondition<Field4> for Gt<L, R>
where
    L: Manifold<Field4, Output = Field>,
    R: Manifold<Field4, Output = Field>,
{
    #[inline(always)]
    fn eval_mask(&self, p: Field4) -> NativeMask {
        let left = self.0.eval(p);
        let right = self.1.eval(p);
        left.0.cmp_gt(right.0)
    }
}

impl<L, R> FieldCondition<Field4> for Le<L, R>
where
    L: Manifold<Field4, Output = Field>,
    R: Manifold<Field4, Output = Field>,
{
    #[inline(always)]
    fn eval_mask(&self, p: Field4) -> NativeMask {
        let left = self.0.eval(p);
        let right = self.1.eval(p);
        left.0.cmp_le(right.0)
    }
}

impl<L, R> FieldCondition<Field4> for Ge<L, R>
where
    L: Manifold<Field4, Output = Field>,
    R: Manifold<Field4, Output = Field>,
{
    #[inline(always)]
    fn eval_mask(&self, p: Field4) -> NativeMask {
        let left = self.0.eval(p);
        let right = self.1.eval(p);
        left.0.cmp_ge(right.0)
    }
}

// ============================================================================
// FieldCondition for boolean combinations
// ============================================================================

impl<L, R, P> FieldCondition<P> for And<L, R>
where
    P: Copy,
    L: FieldCondition<P>,
    R: FieldCondition<P>,
{
    #[inline(always)]
    fn eval_mask(&self, p: P) -> NativeMask {
        self.0.eval_mask(p) & self.1.eval_mask(p)
    }
}

impl<L, R, P> FieldCondition<P> for Or<L, R>
where
    P: Copy,
    L: FieldCondition<P>,
    R: FieldCondition<P>,
{
    #[inline(always)]
    fn eval_mask(&self, p: P) -> NativeMask {
        self.0.eval_mask(p) | self.1.eval_mask(p)
    }
}

impl<T, P> FieldCondition<P> for BNot<T>
where
    P: Copy,
    T: FieldCondition<P>,
{
    #[inline(always)]
    fn eval_mask(&self, p: P) -> NativeMask {
        !self.0.eval_mask(p)
    }
}

// ============================================================================
// FieldCondition for Abs<T> (common in threshold patterns like |winding| >= 0.5)
// ============================================================================

impl<T> FieldCondition<Field4> for crate::ops::unary::Abs<T>
where
    T: Manifold<Field4, Output = Field>,
{
    #[inline(always)]
    fn eval_mask(&self, p: Field4) -> NativeMask {
        // Abs doesn't produce a mask directly, so this shouldn't be called.
        // But we need it for nested expressions like Ge(Abs(m), 0.5)
        // In that case, Ge's eval_mask handles it properly.
        // This impl is for cases where Abs is used as a bare condition (unusual).
        let val = self.0.eval(p);
        val.0.float_to_mask()
    }
}

// Field itself can be a condition (mask stored as float bits).
// Used when Field::gt/lt/etc. returns a Field mask directly.
impl FieldCondition<Field4> for Field {
    #[inline(always)]
    fn eval_mask(&self, _p: Field4) -> NativeMask {
        self.0.float_to_mask()
    }
}

// ============================================================================
// Select implementation for 4D Field domain with FieldCondition (OPTIMIZED PATH)
// ============================================================================

impl<C, T, F, O> Manifold<Field4> for Select<C, T, F>
where
    O: crate::numeric::Selectable,
    C: FieldCondition<Field4>,
    T: Manifold<Field4, Output = O>,
    F: Manifold<Field4, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: Field4) -> O {
        // Get native mask directly - no float conversion!
        let mask = self.cond.eval_mask(p);

        // Early exit using native mask ops (free on AVX-512 k-registers)
        if mask.all() {
            return self.if_true.eval(p);
        }
        if !mask.any() {
            return self.if_false.eval(p);
        }

        // Select using Selectable trait
        // Convert native mask back to Field (float mask) for the generic interface
        let mask_field = Field(NativeSimd::mask_to_float(mask));
        let true_val = self.if_true.eval(p);
        let false_val = self.if_false.eval(p);

        O::select_raw(mask_field, true_val, false_val)
    }
}

// ============================================================================
// Generic implementation for non-Field types (Jet2, etc.) on 4D domains
// ============================================================================

type Jet4 = (
    crate::jet::Jet2,
    crate::jet::Jet2,
    crate::jet::Jet2,
    crate::jet::Jet2,
);

impl<C, T, F, O> Manifold<Jet4> for Select<C, T, F>
where
    O: crate::numeric::Selectable,
    C: Manifold<Jet4>,
    <C as Manifold<Jet4>>::Output: Into<Field> + Copy,
    T: Manifold<Jet4, Output = O>,
    F: Manifold<Jet4, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: Jet4) -> O {
        let mask: Field = self.cond.eval(p).into();
        if mask.all() {
            return self.if_true.eval(p);
        }
        if !mask.any() {
            return self.if_false.eval(p);
        }

        let true_val = self.if_true.eval(p);
        let false_val = self.if_false.eval(p);
        O::select_raw(mask, true_val, false_val)
    }
}

// ============================================================================
// Jet3 with Selectable output (for Discrete in the mullet architecture)
// ============================================================================

type Jet3_4 = (
    crate::jet::Jet3,
    crate::jet::Jet3,
    crate::jet::Jet3,
    crate::jet::Jet3,
);

impl<C, T, F, O> Manifold<Jet3_4> for Select<C, T, F>
where
    O: crate::numeric::Selectable,
    C: Manifold<Jet3_4>,
    <C as Manifold<Jet3_4>>::Output: Into<Field> + Copy,
    T: Manifold<Jet3_4, Output = O>,
    F: Manifold<Jet3_4, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: Jet3_4) -> O {
        let mask: Field = self.cond.eval(p).into();

        // Early exit using Field's any/all
        if mask.all() {
            return self.if_true.eval(p);
        }
        if !mask.any() {
            return self.if_false.eval(p);
        }

        let true_val = self.if_true.eval(p);
        let false_val = self.if_false.eval(p);
        O::select_raw(mask, true_val, false_val)
    }
}
// ============================================================================
// Generic implementation for tuple domains ((V0, V1, ...), P)
// ============================================================================
//
// This enables Select to work with WithContext-extended domains.
// The condition, true branch, and false branch all evaluate on the extended domain,
// but we extract the base spatial domain P for coordinate access.
//
// NOTE: 1-4 element tuple impls removed - covered by array-based impls below

/// Generic Select impl for 5-element tuple domains (the critical font rendering case)
impl<V0, V1, V2, V3, V4, P, C, T, F, O> Manifold<((V0, V1, V2, V3, V4), P)> for Select<C, T, F>
where
    V0: Copy + Send + Sync,
    V1: Copy + Send + Sync,
    V2: Copy + Send + Sync,
    V3: Copy + Send + Sync,
    V4: Copy + Send + Sync,
    P: Copy + Send + Sync,
    O: crate::numeric::Selectable,
    C: Manifold<((V0, V1, V2, V3, V4), P)>,
    <C as Manifold<((V0, V1, V2, V3, V4), P)>>::Output: Into<Field> + Copy,
    T: Manifold<((V0, V1, V2, V3, V4), P), Output = O>,
    F: Manifold<((V0, V1, V2, V3, V4), P), Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4), P)) -> O {
        let mask: Field = self.cond.eval(p).into();
        if mask.all() {
            return self.if_true.eval(p);
        }
        if !mask.any() {
            return self.if_false.eval(p);
        }
        let true_val = self.if_true.eval(p);
        let false_val = self.if_false.eval(p);
        O::select_raw(mask, true_val, false_val)
    }
}

/// Generic Select impl for 6-element tuple domains (needed for antialiased line rendering)
impl<V0, V1, V2, V3, V4, V5, P, C, T, F, O> Manifold<((V0, V1, V2, V3, V4, V5), P)> for Select<C, T, F>
where
    V0: Copy + Send + Sync,
    V1: Copy + Send + Sync,
    V2: Copy + Send + Sync,
    V3: Copy + Send + Sync,
    V4: Copy + Send + Sync,
    V5: Copy + Send + Sync,
    P: Copy + Send + Sync,
    O: crate::numeric::Selectable,
    C: Manifold<((V0, V1, V2, V3, V4, V5), P)>,
    <C as Manifold<((V0, V1, V2, V3, V4, V5), P)>>::Output: Into<Field> + Copy,
    T: Manifold<((V0, V1, V2, V3, V4, V5), P), Output = O>,
    F: Manifold<((V0, V1, V2, V3, V4, V5), P), Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5), P)) -> O {
        let mask: Field = self.cond.eval(p).into();
        if mask.all() {
            return self.if_true.eval(p);
        }
        if !mask.any() {
            return self.if_false.eval(p);
        }
        let true_val = self.if_true.eval(p);
        let false_val = self.if_false.eval(p);
        O::select_raw(mask, true_val, false_val)
    }
}

/// Generic Select impl for 8-element tuple domains
impl<V0, V1, V2, V3, V4, V5, V6, V7, P, C, T, F, O> Manifold<((V0, V1, V2, V3, V4, V5, V6, V7), P)> for Select<C, T, F>
where
    V0: Copy + Send + Sync,
    V1: Copy + Send + Sync,
    V2: Copy + Send + Sync,
    V3: Copy + Send + Sync,
    V4: Copy + Send + Sync,
    V5: Copy + Send + Sync,
    V6: Copy + Send + Sync,
    V7: Copy + Send + Sync,
    P: Copy + Send + Sync,
    O: crate::numeric::Selectable,
    C: Manifold<((V0, V1, V2, V3, V4, V5, V6, V7), P)>,
    <C as Manifold<((V0, V1, V2, V3, V4, V5, V6, V7), P)>>::Output: Into<Field> + Copy,
    T: Manifold<((V0, V1, V2, V3, V4, V5, V6, V7), P), Output = O>,
    F: Manifold<((V0, V1, V2, V3, V4, V5, V6, V7), P), Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7), P)) -> O {
        let mask: Field = self.cond.eval(p).into();
        if mask.all() {
            return self.if_true.eval(p);
        }
        if !mask.any() {
            return self.if_false.eval(p);
        }
        let true_val = self.if_true.eval(p);
        let false_val = self.if_false.eval(p);
        O::select_raw(mask, true_val, false_val)
    }
}

/// Generic Select impl for 9-element tuple domains (for font quadratic curves with dy/dt)
impl<V0, V1, V2, V3, V4, V5, V6, V7, V8, P, C, T, F, O> Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8), P)> for Select<C, T, F>
where
    V0: Copy + Send + Sync,
    V1: Copy + Send + Sync,
    V2: Copy + Send + Sync,
    V3: Copy + Send + Sync,
    V4: Copy + Send + Sync,
    V5: Copy + Send + Sync,
    V6: Copy + Send + Sync,
    V7: Copy + Send + Sync,
    V8: Copy + Send + Sync,
    P: Copy + Send + Sync,
    O: crate::numeric::Selectable,
    C: Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8), P)>,
    <C as Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8), P)>>::Output: Into<Field> + Copy,
    T: Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8), P), Output = O>,
    F: Manifold<((V0, V1, V2, V3, V4, V5, V6, V7, V8), P), Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: ((V0, V1, V2, V3, V4, V5, V6, V7, V8), P)) -> O {
        let mask: Field = self.cond.eval(p).into();
        if mask.all() {
            return self.if_true.eval(p);
        }
        if !mask.any() {
            return self.if_false.eval(p);
        }
        let true_val = self.if_true.eval(p);
        let false_val = self.if_false.eval(p);
        O::select_raw(mask, true_val, false_val)
    }
}

// ============================================================================
// Macro-generated Select impls for context tuple domains (10-16 elements)
// ============================================================================

macro_rules! impl_select_for_ctx {
    ([$($V:ident),+]) => {
        impl<$($V,)+ P, C, T, F, O> Manifold<(($($V,)+), P)> for Select<C, T, F>
        where
            $($V: Copy + Send + Sync,)+
            P: Copy + Send + Sync,
            O: crate::numeric::Selectable,
            C: Manifold<(($($V,)+), P)>,
            <C as Manifold<(($($V,)+), P)>>::Output: Into<Field> + Copy,
            T: Manifold<(($($V,)+), P), Output = O>,
            F: Manifold<(($($V,)+), P), Output = O>,
        {
            type Output = O;
            #[inline(always)]
            fn eval(&self, p: (($($V,)+), P)) -> O {
                let mask: Field = self.cond.eval(p).into();
                if mask.all() {
                    return self.if_true.eval(p);
                }
                if !mask.any() {
                    return self.if_false.eval(p);
                }
                let true_val = self.if_true.eval(p);
                let false_val = self.if_false.eval(p);
                O::select_raw(mask, true_val, false_val)
            }
        }
    };
}

impl_select_for_ctx!([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9]);
impl_select_for_ctx!([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10]);
impl_select_for_ctx!([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11]);
impl_select_for_ctx!([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12]);
impl_select_for_ctx!([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13]);
impl_select_for_ctx!([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14]);
impl_select_for_ctx!([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15]);

// ============================================================================
// Select Implementations for Array-Based Context Domains
// ============================================================================
//
// These support the new array-based WithContext approach where values are
// grouped by type into arrays: ([T; N],) or ([T0; N], [T1; M]), etc.

/// Select impl for single array context domains
impl<T, const N: usize, P, C, Tr, F, O> Manifold<(([T; N],), P)> for Select<C, Tr, F>
where
    T: Copy + Send + Sync,
    P: Copy + Send + Sync,
    O: crate::numeric::Selectable,
    C: Manifold<(([T; N],), P)>,
    <C as Manifold<(([T; N],), P)>>::Output: Into<Field> + Copy,
    Tr: Manifold<(([T; N],), P), Output = O>,
    F: Manifold<(([T; N],), P), Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: (([T; N],), P)) -> O {
        let mask: Field = self.cond.eval(p).into();
        if mask.all() {
            return self.if_true.eval(p);
        }
        if !mask.any() {
            return self.if_false.eval(p);
        }
        let true_val = self.if_true.eval(p);
        let false_val = self.if_false.eval(p);
        O::select_raw(mask, true_val, false_val)
    }
}

/// Select impl for two array context domains
impl<T0, T1, const N: usize, const M: usize, P, C, Tr, F, O> Manifold<(([T0; N], [T1; M]), P)>
    for Select<C, Tr, F>
where
    T0: Copy + Send + Sync,
    T1: Copy + Send + Sync,
    P: Copy + Send + Sync,
    O: crate::numeric::Selectable,
    C: Manifold<(([T0; N], [T1; M]), P)>,
    <C as Manifold<(([T0; N], [T1; M]), P)>>::Output: Into<Field> + Copy,
    Tr: Manifold<(([T0; N], [T1; M]), P), Output = O>,
    F: Manifold<(([T0; N], [T1; M]), P), Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: (([T0; N], [T1; M]), P)) -> O {
        let mask: Field = self.cond.eval(p).into();
        if mask.all() {
            return self.if_true.eval(p);
        }
        if !mask.any() {
            return self.if_false.eval(p);
        }
        let true_val = self.if_true.eval(p);
        let false_val = self.if_false.eval(p);
        O::select_raw(mask, true_val, false_val)
    }
}

/// Select impl for three array context domains
impl<T0, T1, T2, const N: usize, const M: usize, const K: usize, P, C, Tr, F, O>
    Manifold<(([T0; N], [T1; M], [T2; K]), P)> for Select<C, Tr, F>
where
    T0: Copy + Send + Sync,
    T1: Copy + Send + Sync,
    T2: Copy + Send + Sync,
    P: Copy + Send + Sync,
    O: crate::numeric::Selectable,
    C: Manifold<(([T0; N], [T1; M], [T2; K]), P)>,
    <C as Manifold<(([T0; N], [T1; M], [T2; K]), P)>>::Output: Into<Field> + Copy,
    Tr: Manifold<(([T0; N], [T1; M], [T2; K]), P), Output = O>,
    F: Manifold<(([T0; N], [T1; M], [T2; K]), P), Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: (([T0; N], [T1; M], [T2; K]), P)) -> O {
        let mask: Field = self.cond.eval(p).into();
        if mask.all() {
            return self.if_true.eval(p);
        }
        if !mask.any() {
            return self.if_false.eval(p);
        }
        let true_val = self.if_true.eval(p);
        let false_val = self.if_false.eval(p);
        O::select_raw(mask, true_val, false_val)
    }
}

/// Select impl for four array context domains
impl<
        T0,
        T1,
        T2,
        T3,
        const N: usize,
        const M: usize,
        const K: usize,
        const L: usize,
        P,
        C,
        Tr,
        F,
        O,
    > Manifold<(([T0; N], [T1; M], [T2; K], [T3; L]), P)> for Select<C, Tr, F>
where
    T0: Copy + Send + Sync,
    T1: Copy + Send + Sync,
    T2: Copy + Send + Sync,
    T3: Copy + Send + Sync,
    P: Copy + Send + Sync,
    O: crate::numeric::Selectable,
    C: Manifold<(([T0; N], [T1; M], [T2; K], [T3; L]), P)>,
    <C as Manifold<(([T0; N], [T1; M], [T2; K], [T3; L]), P)>>::Output: Into<Field> + Copy,
    Tr: Manifold<(([T0; N], [T1; M], [T2; K], [T3; L]), P), Output = O>,
    F: Manifold<(([T0; N], [T1; M], [T2; K], [T3; L]), P), Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: (([T0; N], [T1; M], [T2; K], [T3; L]), P)) -> O {
        let mask: Field = self.cond.eval(p).into();
        if mask.all() {
            return self.if_true.eval(p);
        }
        if !mask.any() {
            return self.if_false.eval(p);
        }
        let true_val = self.if_true.eval(p);
        let false_val = self.if_false.eval(p);
        O::select_raw(mask, true_val, false_val)
    }
}
