//! # Comparison Operations
//!
//! AST nodes for comparisons: Lt, Gt, Le, Ge.
//! These produce masks that can be used with Select.

use crate::{Field, Manifold};

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

impl<L: Manifold<Output = Field>, R: Manifold<Output = Field>> Manifold for Lt<L, R> {
    type Output = Field;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval_raw(x, y, z, w).lt(self.1.eval_raw(x, y, z, w))
    }
}

impl<L: Manifold<Output = Field>, R: Manifold<Output = Field>> Manifold for Gt<L, R> {
    type Output = Field;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval_raw(x, y, z, w).gt(self.1.eval_raw(x, y, z, w))
    }
}

impl<L: Manifold<Output = Field>, R: Manifold<Output = Field>> Manifold for Le<L, R> {
    type Output = Field;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval_raw(x, y, z, w).le(self.1.eval_raw(x, y, z, w))
    }
}

impl<L: Manifold<Output = Field>, R: Manifold<Output = Field>> Manifold for Ge<L, R> {
    type Output = Field;
    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        self.0.eval_raw(x, y, z, w).ge(self.1.eval_raw(x, y, z, w))
    }
}

// Bitwise ops for chaining comparisons: X.ge(0) & X.le(1)
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
