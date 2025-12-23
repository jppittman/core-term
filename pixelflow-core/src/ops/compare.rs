//! # Comparison Operations
//!
//! AST nodes for comparisons: Lt, Gt, Le, Ge (hard thresholds)
//! and SoftLt, SoftGt, SoftSelect (sigmoid-smooth for Jet2 gradients).

use crate::{Field, Jet2, Manifold};

// ============================================================================
// Hard Comparisons (generic over Numeric)
// ============================================================================

/// Less than: L < R
#[derive(Clone, Copy, Debug)]
pub struct Lt<L, R>(pub L, pub R);

/// Greater than: L > R
#[derive(Clone, Copy, Debug)]
pub struct Gt<L, R>(pub L, pub R);

/// Less than or equal: L <= R
#[derive(Clone, Copy, Debug)]
pub struct Le<L, R>(pub L, pub R);

/// Greater than or equal: L >= R
#[derive(Clone, Copy, Debug)]
pub struct Ge<L, R>(pub L, pub R);

/// Hard select: returns if_true where mask != 0, else if_false
/// Always returns Field (for final pixel evaluation)
#[derive(Clone, Copy, Debug)]
pub struct Select<Mask, IfTrue, IfFalse> {
    pub mask: Mask,
    pub if_true: IfTrue,
    pub if_false: IfFalse,
}

impl<L, R, I> Manifold<I> for Lt<L, R>
where
    I: crate::numeric::Numeric,
    L: Manifold<I, Output = I>,
    R: Manifold<I, Output = I>,
{
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> I {
        self.0.eval_raw(x, y, z, w).lt(self.1.eval_raw(x, y, z, w))
    }
}

impl<L, R, I> Manifold<I> for Gt<L, R>
where
    I: crate::numeric::Numeric,
    L: Manifold<I, Output = I>,
    R: Manifold<I, Output = I>,
{
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> I {
        self.0.eval_raw(x, y, z, w).gt(self.1.eval_raw(x, y, z, w))
    }
}

impl<L, R, I> Manifold<I> for Le<L, R>
where
    I: crate::numeric::Numeric,
    L: Manifold<I, Output = I>,
    R: Manifold<I, Output = I>,
{
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> I {
        self.0.eval_raw(x, y, z, w).le(self.1.eval_raw(x, y, z, w))
    }
}

impl<L, R, I> Manifold<I> for Ge<L, R>
where
    I: crate::numeric::Numeric,
    L: Manifold<I, Output = I>,
    R: Manifold<I, Output = I>,
{
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> I {
        self.0.eval_raw(x, y, z, w).ge(self.1.eval_raw(x, y, z, w))
    }
}

// Select always returns Field
impl<Mask, IfTrue, IfFalse, I> Manifold<I> for Select<Mask, IfTrue, IfFalse>
where
    I: crate::numeric::Numeric,
    Mask: Manifold<I, Output = I>,
    IfTrue: Manifold<I, Output = I>,
    IfFalse: Manifold<I, Output = I>,
{
    type Output = I;
    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> I {
        let mask_val = self.mask.eval_raw(x, y, z, w);
        let true_val = self.if_true.eval_raw(x, y, z, w);
        let false_val = self.if_false.eval_raw(x, y, z, w);
        I::select(mask_val, true_val, false_val)
    }
}

// ============================================================================
// Bitwise ops for chaining comparisons: X.ge(0) & X.le(1)
// ============================================================================

use crate::ops::logic::{And, Or};

macro_rules! impl_logic_ops {
    ($ty:ident) => {
        impl<L, R, Rhs> core::ops::BitAnd<Rhs> for $ty<L, R>
        where
            Self: Manifold,
            Rhs: Manifold,
        {
            type Output = And<Self, Rhs>;
            fn bitand(self, rhs: Rhs) -> Self::Output {
                And(self, rhs)
            }
        }

        impl<L, R, Rhs> core::ops::BitOr<Rhs> for $ty<L, R>
        where
            Self: Manifold,
            Rhs: Manifold,
        {
            type Output = Or<Self, Rhs>;
            fn bitor(self, rhs: Rhs) -> Self::Output {
                Or(self, rhs)
            }
        }
    };
}

impl_logic_ops!(Lt);
impl_logic_ops!(Gt);
impl_logic_ops!(Le);
impl_logic_ops!(Ge);
impl_logic_ops!(And);
impl_logic_ops!(Or);

// ============================================================================
// Smooth/Sigmoid Comparisons (Jet2-specific for gradients)
// ============================================================================

/// Smooth greater-than using sigmoid: sigmoid((L - R) / k)
/// Returns ~0 when L << R, ~1 when L >> R, smooth transition in between.
/// Smaller k = sharper transition.
///
/// **Jet2-specific**: Only works with Jet2 to provide smooth derivatives.
/// For Field evaluation, use hard Gt.
#[derive(Clone, Copy, Debug)]
pub struct SoftGt<L, R> {
    pub left: L,
    pub right: R,
    pub sharpness: f32,
}

/// Smooth less-than: sigmoid((R - L) / k)
/// **Jet2-specific** for smooth derivatives.
#[derive(Clone, Copy, Debug)]
pub struct SoftLt<L, R> {
    pub left: L,
    pub right: R,
    pub sharpness: f32,
}

/// Smooth select: blend between if_false and if_true based on smooth mask.
/// result = if_false + mask * (if_true - if_false)
///
/// **Always returns Jet2** and **only takes Manifold<Jet2> inputs**.
/// For Field select, use hard Select.
#[derive(Clone, Copy, Debug)]
pub struct SoftSelect<Mask, IfTrue, IfFalse> {
    pub mask: Mask,
    pub if_true: IfTrue,
    pub if_false: IfFalse,
}

impl<L, R> Manifold<Jet2> for SoftGt<L, R>
where
    L: Manifold<Jet2, Output = Jet2>,
    R: Manifold<Jet2, Output = Jet2>,
{
    type Output = Jet2;
    #[inline(always)]
    fn eval_raw(&self, x: Jet2, y: Jet2, z: Jet2, w: Jet2) -> Jet2 {
        use crate::numeric::Numeric;

        let left_val = self.left.eval_raw(x, y, z, w);
        let right_val = self.right.eval_raw(x, y, z, w);
        let diff = left_val - right_val;

        // Smooth sigmoid via Hermite polynomial
        // t = clamp((diff/k + 1)/2, 0, 1), then smoothstep: 3t² - 2t³
        let k = Jet2::from_f32(self.sharpness);
        let t = ((diff / k) + Jet2::from_f32(1.0)) / Jet2::from_f32(2.0);
        let t = t.max(Jet2::from_f32(0.0)).min(Jet2::from_f32(1.0));

        let t2 = t * t;
        let t3 = t2 * t;
        t3 * Jet2::from_f32(-2.0) + t2 * Jet2::from_f32(3.0)
    }
}

impl<L, R> Manifold<Jet2> for SoftLt<L, R>
where
    L: Manifold<Jet2, Output = Jet2>,
    R: Manifold<Jet2, Output = Jet2>,
{
    type Output = Jet2;
    #[inline(always)]
    fn eval_raw(&self, x: Jet2, y: Jet2, z: Jet2, w: Jet2) -> Jet2 {
        use crate::numeric::Numeric;

        let left_val = self.left.eval_raw(x, y, z, w);
        let right_val = self.right.eval_raw(x, y, z, w);
        let diff = right_val - left_val; // Reversed for Lt

        let k = Jet2::from_f32(self.sharpness);
        let t = ((diff / k) + Jet2::from_f32(1.0)) / Jet2::from_f32(2.0);
        let t = t.max(Jet2::from_f32(0.0)).min(Jet2::from_f32(1.0));

        let t2 = t * t;
        let t3 = t2 * t;
        t3 * Jet2::from_f32(-2.0) + t2 * Jet2::from_f32(3.0)
    }
}

/// SoftSelect always returns Jet2, only takes Manifold<Jet2> inputs
impl<Mask, IfTrue, IfFalse> Manifold<Jet2> for SoftSelect<Mask, IfTrue, IfFalse>
where
    Mask: Manifold<Jet2, Output = Jet2>,
    IfTrue: Manifold<Jet2, Output = Jet2>,
    IfFalse: Manifold<Jet2, Output = Jet2>,
{
    type Output = Jet2;
    #[inline(always)]
    fn eval_raw(&self, x: Jet2, y: Jet2, z: Jet2, w: Jet2) -> Jet2 {
        let mask_val = self.mask.eval_raw(x, y, z, w);
        let true_val = self.if_true.eval_raw(x, y, z, w);
        let false_val = self.if_false.eval_raw(x, y, z, w);

        // Linear blend with smooth mask
        false_val + mask_val * (true_val - false_val)
    }
}
