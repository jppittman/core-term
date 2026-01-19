//! # Comparison Operations
//!
//! AST nodes for comparisons: Lt, Gt, Le, Ge (hard thresholds)
//! and SoftLt, SoftGt, SoftSelect (sigmoid-smooth for Jet2 gradients).
//!
//! ## Automatic Optimization
//!
//! When used with `Select`, comparison operations automatically use native
//! mask registers (k-registers on AVX-512) without any extra work:
//!
//! ```ignore
//! // This automatically uses native masks - no manual optimization needed!
//! Select { cond: Lt(a, b), if_true, if_false }
//! ```

use crate::Manifold;
use crate::jet::Jet2;
use crate::numeric::Computational;

// ============================================================================
// Hard Comparisons (generic over Numeric)
// ============================================================================

/// Less than: L < R
#[derive(Clone, Debug)]
pub struct Lt<L, R>(pub L, pub R);

/// Greater than: L > R
#[derive(Clone, Debug)]
pub struct Gt<L, R>(pub L, pub R);

/// Less than or equal: L <= R
#[derive(Clone, Debug)]
pub struct Le<L, R>(pub L, pub R);

/// Greater than or equal: L >= R
#[derive(Clone, Debug)]
pub struct Ge<L, R>(pub L, pub R);

// Select is defined in combinators/select.rs with early-exit optimization.
// Use `pixelflow_core::Select` from there.

// ============================================================================
// Domain-Generic Manifold Implementations for Comparisons
// ============================================================================

impl<P, L, R, O> Manifold<P> for Lt<L, R>
where
    P: Copy + Send + Sync,
    O: crate::numeric::Numeric,
    L: Manifold<P, Output = O>,
    R: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).lt(self.1.eval(p))
    }
}

impl<P, L, R, O> Manifold<P> for Gt<L, R>
where
    P: Copy + Send + Sync,
    O: crate::numeric::Numeric,
    L: Manifold<P, Output = O>,
    R: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).gt(self.1.eval(p))
    }
}

impl<P, L, R, O> Manifold<P> for Le<L, R>
where
    P: Copy + Send + Sync,
    O: crate::numeric::Numeric,
    L: Manifold<P, Output = O>,
    R: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).le(self.1.eval(p))
    }
}

impl<P, L, R, O> Manifold<P> for Ge<L, R>
where
    P: Copy + Send + Sync,
    O: crate::numeric::Numeric,
    L: Manifold<P, Output = O>,
    R: Manifold<P, Output = O>,
{
    type Output = O;
    #[inline(always)]
    fn eval(&self, p: P) -> O {
        self.0.eval(p).ge(self.1.eval(p))
    }
}

// ============================================================================
// Bitwise ops for chaining comparisons: X.ge(0) & X.le(1)
// ============================================================================

use crate::ops::logic::{And, Or};

macro_rules! impl_logic_ops {
    ($ty:ident) => {
        impl<L, R, Rhs> core::ops::BitAnd<Rhs> for $ty<L, R> {
            type Output = And<Self, Rhs>;
            fn bitand(self, rhs: Rhs) -> Self::Output {
                And(self, rhs)
            }
        }

        impl<L, R, Rhs> core::ops::BitOr<Rhs> for $ty<L, R> {
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

/// Smooth greater-than using sigmoid: sigmoid((L - R) / k).
/// Returns ~0 when L << R, ~1 when L >> R, smooth transition in between.
/// Smaller k = sharper transition.
///
/// **Jet2-specific**: Only works with Jet2 to provide smooth derivatives.
/// For Field evaluation, use hard Gt.
#[derive(Clone, Debug)]
pub struct SoftGt<L, R> {
    /// Left operand.
    pub left: L,
    /// Right operand.
    pub right: R,
    /// Transition sharpness (smaller = sharper).
    pub sharpness: f32,
}

/// Smooth less-than: sigmoid((R - L) / k).
/// **Jet2-specific** for smooth derivatives.
#[derive(Clone, Debug)]
pub struct SoftLt<L, R> {
    /// Left operand.
    pub left: L,
    /// Right operand.
    pub right: R,
    /// Transition sharpness (smaller = sharper).
    pub sharpness: f32,
}

/// Smooth select: blend between if_false and if_true based on smooth mask.
/// result = if_false + mask * (if_true - if_false)
///
/// **Always returns Jet2** and **only takes Manifold<Jet2> inputs**.
/// For Field select, use hard Select.
#[derive(Clone, Debug)]
pub struct SoftSelect<Mask, IfTrue, IfFalse> {
    /// The smooth mask (0.0 to 1.0).
    pub mask: Mask,
    /// Value when mask is 1.0.
    pub if_true: IfTrue,
    /// Value when mask is 0.0.
    pub if_false: IfFalse,
}

// Hermite interpolation coefficients for smoothstep: 3t² - 2t³
const HERMITE_CUBIC: f32 = -2.0;
const HERMITE_QUAD: f32 = 3.0;

/// Smooth sigmoid via Hermite polynomial (smoothstep).
/// t = clamp((diff/k + 1)/2, 0, 1)
/// result = 3t² - 2t³
#[inline(always)]
fn smoothstep_sigmoid(diff: Jet2, sharpness: f32) -> Jet2 {
    let k = Jet2::from_f32(sharpness);
    let t = ((diff / k) + Jet2::from_f32(1.0)) / Jet2::from_f32(2.0);
    let t = t.max(Jet2::from_f32(0.0)).min(Jet2::from_f32(1.0));

    let t2 = t * t;
    let t3 = t2 * t;
    t3 * Jet2::from_f32(HERMITE_CUBIC) + t2 * Jet2::from_f32(HERMITE_QUAD)
}

// SoftGt, SoftLt, SoftSelect are Jet2-specific and work on 4D Jet2 domains
impl<L, R> Manifold<(Jet2, Jet2, Jet2, Jet2)> for SoftGt<L, R>
where
    L: Manifold<(Jet2, Jet2, Jet2, Jet2), Output = Jet2>,
    R: Manifold<(Jet2, Jet2, Jet2, Jet2), Output = Jet2>,
{
    type Output = Jet2;
    #[inline(always)]
    fn eval(&self, p: (Jet2, Jet2, Jet2, Jet2)) -> Jet2 {
        let left_val = self.left.eval(p);
        let right_val = self.right.eval(p);
        let diff = left_val - right_val;

        smoothstep_sigmoid(diff, self.sharpness)
    }
}

impl<L, R> Manifold<(Jet2, Jet2, Jet2, Jet2)> for SoftLt<L, R>
where
    L: Manifold<(Jet2, Jet2, Jet2, Jet2), Output = Jet2>,
    R: Manifold<(Jet2, Jet2, Jet2, Jet2), Output = Jet2>,
{
    type Output = Jet2;
    #[inline(always)]
    fn eval(&self, p: (Jet2, Jet2, Jet2, Jet2)) -> Jet2 {
        let left_val = self.left.eval(p);
        let right_val = self.right.eval(p);
        let diff = right_val - left_val; // Reversed for Lt

        smoothstep_sigmoid(diff, self.sharpness)
    }
}

/// SoftSelect always returns Jet2, only takes Manifold inputs on 4D Jet2 domain
impl<Mask, IfTrue, IfFalse> Manifold<(Jet2, Jet2, Jet2, Jet2)> for SoftSelect<Mask, IfTrue, IfFalse>
where
    Mask: Manifold<(Jet2, Jet2, Jet2, Jet2), Output = Jet2>,
    IfTrue: Manifold<(Jet2, Jet2, Jet2, Jet2), Output = Jet2>,
    IfFalse: Manifold<(Jet2, Jet2, Jet2, Jet2), Output = Jet2>,
{
    type Output = Jet2;
    #[inline(always)]
    fn eval(&self, p: (Jet2, Jet2, Jet2, Jet2)) -> Jet2 {
        let mask_val = self.mask.eval(p);
        let true_val = self.if_true.eval(p);
        let false_val = self.if_false.eval(p);

        // Linear blend with smooth mask
        false_val + mask_val * (true_val - false_val)
    }
}
