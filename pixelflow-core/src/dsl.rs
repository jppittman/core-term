use crate::batch::Batch;
use crate::surfaces::{
    self, Abs, Add, Div, Eq, Ge, Gt, Le, Lt, Map, Max, Mul, Ne, Negate, Offset, Select, Skew, Sqrt,
    Sub, W, X, Y, Z,
};
use crate::traits::Surface;
use core::fmt::Debug;
use core::marker::PhantomData;
use core::ops;

/// Extensions for any surface (Algebraic and Coordinate transforms).
pub trait SurfaceExt<T>: Surface<T> + Sized
where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
{
    /// Applies a translation offset to the surface.
    fn offset(self, dx: i32, dy: i32) -> Offset<Self> {
        Offset {
            source: self,
            dx,
            dy,
        }
    }

    /// Applies a skew (shear) transformation to the surface.
    fn skew(self, shear: i32) -> Skew<Self> {
        Skew {
            source: self,
            shear,
        }
    }

    /// Warps the coordinate space of the surface.
    ///
    /// This is the `Warp` eigenshader.
    fn warp<F, C>(self, mapping: F) -> surfaces::Warp<Self, F, C>
    where
        F: Fn(Batch<C>, Batch<C>, Batch<C>, Batch<C>) -> (Batch<C>, Batch<C>, Batch<C>, Batch<C>)
            + Send
            + Sync,
        C: Copy + Debug + Default + PartialEq + Send + Sync + 'static,
    {
        surfaces::Warp::new(self, mapping)
    }

    /// Applies a linear transform to the values: `val * slope + bias`.
    ///
    /// This is the `Grade` eigenshader: `val * slope + bias`.
    fn grade<M, B>(self, slope: M, bias: B) -> surfaces::Grade<Self, M, B> {
        surfaces::Grade::new(self, slope, bias)
    }

    /// Linearly interpolates to another surface: `self + t * (other - self)`.
    ///
    /// This is the `Lerp` eigenshader: `self + t * (other - self)`.
    /// Note: `self` is the 'start' (a), `other` is the 'end' (b).
    fn lerp<Param, Other>(self, t: Param, other: Other) -> surfaces::Lerp<Param, Self, Other> {
        surfaces::Lerp::new(t, self, other)
    }

    /// Absolute value of the surface.
    fn abs(self) -> Abs<Self> {
        Abs(self)
    }

    /// Square root of the surface.
    fn sqrt(self) -> Sqrt<Self> {
        Sqrt(self)
    }

    /// Negation of the surface.
    fn neg(self) -> Negate<Self> {
        Negate(self)
    }

    /// Less than: `self < other`. Returns a mask (u32).
    fn lt<O>(self, other: O) -> Lt<Self, O, T> {
        Lt(self, other, PhantomData)
    }

    /// Greater than: `self > other`. Returns a mask (u32).
    fn gt<O>(self, other: O) -> Gt<Self, O, T> {
        Gt(self, other, PhantomData)
    }

    /// Equal to: `self == other`. Returns a mask (u32).
    fn eq<O>(self, other: O) -> Eq<Self, O, T> {
        Eq(self, other, PhantomData)
    }

    /// Less than or equal: `self <= other`. Returns a mask (u32).
    fn le<O>(self, other: O) -> Le<Self, O, T> {
        Le(self, other, PhantomData)
    }

    /// Greater than or equal: `self >= other`. Returns a mask (u32).
    fn ge<O>(self, other: O) -> Ge<Self, O, T> {
        Ge(self, other, PhantomData)
    }

    /// Not equal to: `self != other`. Returns a mask (u32).
    fn ne<O>(self, other: O) -> Ne<Self, O, T> {
        Ne(self, other, PhantomData)
    }

    /// Branching: `if mask { if_true } else { if_false }`.
    fn select<I, F>(self, if_true: I, if_false: F) -> Select<Self, I, F> {
        Select {
            mask: self,
            if_true,
            if_false,
        }
    }

    /// Computes the maximum of this surface and another.
    fn max<O>(self, other: O) -> Max<Self, O>
    where
        O: Surface<T>,
    {
        Max(self, other)
    }

    /// Applies a transformation function to surface output.
    ///
    /// Use for gamma correction, color adjustments, or any per-pixel transform.
    fn map<F>(self, transform: F) -> Map<Self, F>
    where
        F: Fn(Batch<T>) -> Batch<T> + Send + Sync,
    {
        Map::new(self, transform)
    }
}

/// Extensions for mask surfaces (u32/bool).
pub trait MaskExt: Surface<u32> + Sized {
    /// Composites one surface over another using this surface as a mask.
    ///
    /// This is a convenience for `self.select(fg, bg)`.
    fn over<F, B>(self, fg: F, bg: B) -> Select<Self, F, B> {
        Select {
            mask: self,
            if_true: fg,
            if_false: bg,
        }
    }
}

impl<S: Surface<u32>> MaskExt for S {}

// Blanket implementations
impl<T, S: Surface<T>> SurfaceExt<T> for S where
    T: Copy + Debug + Default + PartialEq + Send + Sync + 'static
{
}

// ============================================================================
// Operator Overloading
// ============================================================================

macro_rules! impl_surface_ops {
    ($($t:ty),*) => {
        $(
            impl<RHS> ops::Add<RHS> for $t {
                type Output = Add<Self, RHS>;
                fn add(self, rhs: RHS) -> Self::Output { Add(self, rhs) }
            }
            impl<RHS> ops::Sub<RHS> for $t {
                type Output = Sub<Self, RHS>;
                fn sub(self, rhs: RHS) -> Self::Output { Sub(self, rhs) }
            }
            impl<RHS> ops::Mul<RHS> for $t {
                type Output = Mul<Self, RHS>;
                fn mul(self, rhs: RHS) -> Self::Output { Mul(self, rhs) }
            }
            impl<RHS> ops::Div<RHS> for $t {
                type Output = Div<Self, RHS>;
                fn div(self, rhs: RHS) -> Self::Output { Div(self, rhs) }
            }
            impl ops::Neg for $t {
                type Output = Negate<Self>;
                fn neg(self) -> Self::Output { Negate(self) }
            }
        )*
    };
}

// Correctly implement operators for binary combinators too, to support chaining.
macro_rules! impl_bin_op_chaining {
    ($name:ident) => {
        impl<A, B, RHS> ops::Add<RHS> for $name<A, B> {
            type Output = Add<Self, RHS>;
            fn add(self, rhs: RHS) -> Self::Output {
                Add(self, rhs)
            }
        }
        impl<A, B, RHS> ops::Sub<RHS> for $name<A, B> {
            type Output = Sub<Self, RHS>;
            fn sub(self, rhs: RHS) -> Self::Output {
                Sub(self, rhs)
            }
        }
        impl<A, B, RHS> ops::Mul<RHS> for $name<A, B> {
            type Output = Mul<Self, RHS>;
            fn mul(self, rhs: RHS) -> Self::Output {
                Mul(self, rhs)
            }
        }
        impl<A, B, RHS> ops::Div<RHS> for $name<A, B> {
            type Output = Div<Self, RHS>;
            fn div(self, rhs: RHS) -> Self::Output {
                Div(self, rhs)
            }
        }
        impl<A, B> ops::Neg for $name<A, B> {
            type Output = Negate<Self>;
            fn neg(self) -> Self::Output {
                Negate(self)
            }
        }
    };
}

// Special case for comparison combinators (they need T)
macro_rules! impl_cmp_op_chaining {
    ($name:ident) => {
        impl<A, B, T, RHS> ops::Add<RHS> for $name<A, B, T> {
            type Output = Add<Self, RHS>;
            fn add(self, rhs: RHS) -> Self::Output {
                Add(self, rhs)
            }
        }
        impl<A, B, T, RHS> ops::Sub<RHS> for $name<A, B, T> {
            type Output = Sub<Self, RHS>;
            fn sub(self, rhs: RHS) -> Self::Output {
                Sub(self, rhs)
            }
        }
        impl<A, B, T, RHS> ops::Mul<RHS> for $name<A, B, T> {
            type Output = Mul<Self, RHS>;
            fn mul(self, rhs: RHS) -> Self::Output {
                Mul(self, rhs)
            }
        }
        impl<A, B, T, RHS> ops::Div<RHS> for $name<A, B, T> {
            type Output = Div<Self, RHS>;
            fn div(self, rhs: RHS) -> Self::Output {
                Div(self, rhs)
            }
        }
        impl<A, B, T> ops::Neg for $name<A, B, T> {
            type Output = Negate<Self>;
            fn neg(self) -> Self::Output {
                Negate(self)
            }
        }
    };
}

// Implement for base projections
impl_surface_ops!(X, Y, Z, W);

// Implement for combinators to allow nesting
impl_bin_op_chaining!(Add);
impl_bin_op_chaining!(Sub);
impl_bin_op_chaining!(Mul);
impl_bin_op_chaining!(Div);

impl_cmp_op_chaining!(Lt);
impl_cmp_op_chaining!(Gt);
impl_cmp_op_chaining!(Eq);
impl_cmp_op_chaining!(Le);
impl_cmp_op_chaining!(Ge);
impl_cmp_op_chaining!(Ne);

// Implement for math combinators
macro_rules! impl_unary_op_chaining {
    ($name:ident) => {
        impl<S, RHS> ops::Add<RHS> for $name<S> {
            type Output = Add<Self, RHS>;
            fn add(self, rhs: RHS) -> Self::Output {
                Add(self, rhs)
            }
        }
        impl<S, RHS> ops::Sub<RHS> for $name<S> {
            type Output = Sub<Self, RHS>;
            fn sub(self, rhs: RHS) -> Self::Output {
                Sub(self, rhs)
            }
        }
        impl<S, RHS> ops::Mul<RHS> for $name<S> {
            type Output = Mul<Self, RHS>;
            fn mul(self, rhs: RHS) -> Self::Output {
                Mul(self, rhs)
            }
        }
        impl<S, RHS> ops::Div<RHS> for $name<S> {
            type Output = Div<Self, RHS>;
            fn div(self, rhs: RHS) -> Self::Output {
                Div(self, rhs)
            }
        }
        impl<S> ops::Neg for $name<S> {
            type Output = Negate<Self>;
            fn neg(self) -> Self::Output {
                Negate(self)
            }
        }
    };
}

impl_unary_op_chaining!(Abs);
impl_unary_op_chaining!(Sqrt);
impl_unary_op_chaining!(Negate);
