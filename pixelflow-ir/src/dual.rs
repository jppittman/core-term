//! Dual numbers for automatic differentiation.
//!
//! A dual number `Dual<N, A>` carries a value and N partial derivatives.
//! When you compose operations on dual numbers, the chain rule is applied
//! automatically, giving you exact derivatives.
//!
//! # Structure
//!
//! ```text
//! Dual<2, f32> = { val: f32, partials: [dx, dy] }
//! Dual<3, f32> = { val: f32, partials: [dx, dy, dz] }
//! ```
//!
//! # Usage
//!
//! - `Dual<2, f32>` for 2D gradients (antialiasing)
//! - `Dual<3, f32>` for 3D gradients (surface normals)
//! - `Dual<1, T>` for directional derivatives (ray marching)

use crate::algebra::{Algebra, Transcendental};

/// Dual number with N partials over base algebra A.
///
/// This is the scalar version used to define the algebra.
/// For SIMD, see `Field<Dual<N>>` which uses SoA layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual<const N: usize, A: Algebra = f32> {
    /// The primal value.
    pub val: A,
    /// Partial derivatives.
    pub partials: [A; N],
}

impl<const N: usize, A: Algebra> Dual<N, A> {
    /// Create a constant (zero derivatives).
    #[inline(always)]
    pub fn constant(val: A) -> Self {
        Self {
            val,
            partials: [A::zero(); N],
        }
    }
}

// ============================================================================
// Macro for generating Algebra implementations without closures
// ============================================================================

macro_rules! impl_dual_algebra {
    ($n:literal, [$($i:literal),*]) => {
        impl<A: Algebra> Algebra for Dual<$n, A> {
            // Mask comes from base algebra - hard select, AA handled at higher level
            type Mask = A::Mask;

            #[inline(always)]
            fn zero() -> Self {
                Self {
                    val: A::zero(),
                    partials: [$({ let _ = $i; A::zero() }),*],
                }
            }

            #[inline(always)]
            fn one() -> Self {
                Self {
                    val: A::one(),
                    partials: [$({ let _ = $i; A::zero() }),*],
                }
            }

            #[inline(always)]
            fn from_f32(val: f32) -> Self {
                Self::constant(A::from_f32(val))
            }

            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self {
                    val: self.val.add(rhs.val),
                    partials: [$(self.partials[$i].add(rhs.partials[$i])),*],
                }
            }

            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self {
                    val: self.val.sub(rhs.val),
                    partials: [$(self.partials[$i].sub(rhs.partials[$i])),*],
                }
            }

            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                // Product rule: (f·g)' = f'·g + f·g'
                Self {
                    val: self.val.mul(rhs.val),
                    partials: [$(
                        self.partials[$i].mul(rhs.val).add(self.val.mul(rhs.partials[$i]))
                    ),*],
                }
            }

            #[inline(always)]
            fn neg(self) -> Self {
                Self {
                    val: self.val.neg(),
                    partials: [$(self.partials[$i].neg()),*],
                }
            }

            // Comparisons delegate to base - return bool for hard select
            #[inline(always)]
            fn lt(self, rhs: Self) -> Self::Mask {
                self.val.lt(rhs.val)
            }

            #[inline(always)]
            fn le(self, rhs: Self) -> Self::Mask {
                self.val.le(rhs.val)
            }

            #[inline(always)]
            fn gt(self, rhs: Self) -> Self::Mask {
                self.val.gt(rhs.val)
            }

            #[inline(always)]
            fn ge(self, rhs: Self) -> Self::Mask {
                self.val.ge(rhs.val)
            }

            #[inline(always)]
            fn eq(self, rhs: Self) -> Self::Mask {
                self.val.eq(rhs.val)
            }

            #[inline(always)]
            fn ne(self, rhs: Self) -> Self::Mask {
                self.val.ne(rhs.val)
            }

            // Hard select - AA handled at rendering layer
            #[inline(always)]
            fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self {
                Self {
                    val: A::select(mask, if_true.val, if_false.val),
                    partials: [$(A::select(mask, if_true.partials[$i], if_false.partials[$i])),*],
                }
            }
        }
    };
}

// Generate implementations for common N values
impl_dual_algebra!(1, [0]);
impl_dual_algebra!(2, [0, 1]);
impl_dual_algebra!(3, [0, 1, 2]);
impl_dual_algebra!(4, [0, 1, 2, 3]);

// ============================================================================
// Transcendental for Dual (chain rule)
// ============================================================================

macro_rules! impl_dual_transcendental {
    ($n:literal, [$($i:literal),*]) => {
        impl<A: Transcendental> Transcendental for Dual<$n, A> {
            #[inline(always)]
            fn div(self, rhs: Self) -> Self {
                // Quotient rule: (f/g)' = (f'·g - f·g') / g²
                let g_sq = rhs.val.mul(rhs.val);
                Self {
                    val: self.val.div(rhs.val),
                    partials: [$(
                        self.partials[$i].mul(rhs.val)
                            .sub(self.val.mul(rhs.partials[$i]))
                            .div(g_sq)
                    ),*],
                }
            }

            #[inline(always)]
            fn recip(self) -> Self {
                // (1/f)' = -f'/f²
                let inv = self.val.recip();
                let neg_inv_sq = inv.mul(inv).neg();
                Self {
                    val: inv,
                    partials: [$(self.partials[$i].mul(neg_inv_sq)),*],
                }
            }

            #[inline(always)]
            fn sqrt(self) -> Self {
                // (√f)' = f'/(2√f)
                let sqrt_val = self.val.sqrt();
                let half_rsqrt = self.val.rsqrt().mul(A::one().div(A::one().add(A::one())));
                Self {
                    val: sqrt_val,
                    partials: [$(self.partials[$i].mul(half_rsqrt)),*],
                }
            }

            #[inline(always)]
            fn rsqrt(self) -> Self {
                // (1/√f)' = -f'/(2·f^(3/2))
                let rsqrt_val = self.val.rsqrt();
                let rsqrt_cubed = rsqrt_val.mul(rsqrt_val).mul(rsqrt_val);
                let half = A::one().div(A::one().add(A::one()));
                let scale = half.neg().mul(rsqrt_cubed);
                Self {
                    val: rsqrt_val,
                    partials: [$(self.partials[$i].mul(scale)),*],
                }
            }

            #[inline(always)]
            fn sin(self) -> Self {
                // (sin f)' = f'·cos f
                let cos_val = self.val.cos();
                Self {
                    val: self.val.sin(),
                    partials: [$(self.partials[$i].mul(cos_val)),*],
                }
            }

            #[inline(always)]
            fn cos(self) -> Self {
                // (cos f)' = -f'·sin f
                let neg_sin = self.val.sin().neg();
                Self {
                    val: self.val.cos(),
                    partials: [$(self.partials[$i].mul(neg_sin)),*],
                }
            }

            #[inline(always)]
            fn exp(self) -> Self {
                // (e^f)' = f'·e^f
                let exp_val = self.val.exp();
                Self {
                    val: exp_val,
                    partials: [$(self.partials[$i].mul(exp_val)),*],
                }
            }

            #[inline(always)]
            fn ln(self) -> Self {
                // (ln f)' = f'/f
                let inv = self.val.recip();
                Self {
                    val: self.val.ln(),
                    partials: [$(self.partials[$i].mul(inv)),*],
                }
            }

            #[inline(always)]
            fn exp2(self) -> Self {
                // (2^f)' = f'·2^f·ln(2)
                let exp2_val = self.val.exp2();
                let ln_2 = A::one().add(A::one()).ln(); // ln(2)
                let coeff = exp2_val.mul(ln_2);
                Self {
                    val: exp2_val,
                    partials: [$(self.partials[$i].mul(coeff)),*],
                }
            }

            #[inline(always)]
            fn log2(self) -> Self {
                // (log₂ f)' = f'/(f·ln(2))
                let ln_2 = A::one().add(A::one()).ln();
                let coeff = self.val.mul(ln_2).recip();
                Self {
                    val: self.val.log2(),
                    partials: [$(self.partials[$i].mul(coeff)),*],
                }
            }

            #[inline(always)]
            fn pow(self, exp: Self) -> Self {
                // (f^g)' = f^g · (g'·ln(f) + g·f'/f)
                let val = self.val.pow(exp.val);
                let ln_base = self.val.ln();
                let inv_base = self.val.recip();
                Self {
                    val,
                    partials: [$(
                        val.mul(
                            exp.partials[$i].mul(ln_base)
                                .add(exp.val.mul(self.partials[$i]).mul(inv_base))
                        )
                    ),*],
                }
            }

            #[inline(always)]
            fn atan2(self, x: Self) -> Self {
                // atan2(y,x)' = (x·dy - y·dx) / (x² + y²)
                let r_sq = self.val.mul(self.val).add(x.val.mul(x.val));
                let inv_r_sq = r_sq.recip();
                Self {
                    val: self.val.atan2(x.val),
                    partials: [$(
                        x.val.mul(self.partials[$i])
                            .sub(self.val.mul(x.partials[$i]))
                            .mul(inv_r_sq)
                    ),*],
                }
            }

            #[inline(always)]
            fn abs(self) -> Self {
                // |f|' = f' · sign(f)
                let abs_val = self.val.abs();
                let sign = self.val.div(abs_val);
                Self {
                    val: abs_val,
                    partials: [$(self.partials[$i].mul(sign)),*],
                }
            }

            #[inline(always)]
            fn floor(self) -> Self {
                // floor has zero derivative (piecewise constant)
                Self {
                    val: self.val.floor(),
                    partials: [$({ let _ = $i; A::zero() }),*],
                }
            }

            #[inline(always)]
            fn min(self, rhs: Self) -> Self {
                let mask = self.val.lt(rhs.val);
                Self::select(mask, self, rhs)
            }

            #[inline(always)]
            fn max(self, rhs: Self) -> Self {
                let mask = self.val.gt(rhs.val);
                Self::select(mask, self, rhs)
            }

            #[inline(always)]
            fn mul_add(self, b: Self, c: Self) -> Self {
                // (a·b + c)' = a'·b + a·b' + c'
                Self {
                    val: self.val.mul_add(b.val, c.val),
                    partials: [$(
                        self.partials[$i].mul(b.val)
                            .add(self.val.mul(b.partials[$i]))
                            .add(c.partials[$i])
                    ),*],
                }
            }
        }
    };
}

impl_dual_transcendental!(1, [0]);
impl_dual_transcendental!(2, [0, 1]);
impl_dual_transcendental!(3, [0, 1, 2]);
impl_dual_transcendental!(4, [0, 1, 2, 3]);

// ============================================================================
// Convenience constructors for common cases
// ============================================================================

impl<A: Algebra> Dual<2, A> {
    /// Create a dual number representing the X variable (∂/∂x = 1, ∂/∂y = 0).
    #[inline(always)]
    pub fn var_x(val: A) -> Self {
        Self {
            val,
            partials: [A::one(), A::zero()],
        }
    }

    /// Create a dual number representing the Y variable (∂/∂x = 0, ∂/∂y = 1).
    #[inline(always)]
    pub fn var_y(val: A) -> Self {
        Self {
            val,
            partials: [A::zero(), A::one()],
        }
    }
}

impl<A: Algebra> Dual<3, A> {
    /// Create a dual number representing the X variable.
    #[inline(always)]
    pub fn var_x(val: A) -> Self {
        Self {
            val,
            partials: [A::one(), A::zero(), A::zero()],
        }
    }

    /// Create a dual number representing the Y variable.
    #[inline(always)]
    pub fn var_y(val: A) -> Self {
        Self {
            val,
            partials: [A::zero(), A::one(), A::zero()],
        }
    }

    /// Create a dual number representing the Z variable.
    #[inline(always)]
    pub fn var_z(val: A) -> Self {
        Self {
            val,
            partials: [A::zero(), A::zero(), A::one()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual2_addition() {
        let x = Dual::<2, f32>::var_x(3.0);
        let y = Dual::<2, f32>::var_y(4.0);
        let sum = x.add(y);

        assert_eq!(sum.val, 7.0);
        assert_eq!(sum.partials[0], 1.0); // dx
        assert_eq!(sum.partials[1], 1.0); // dy
    }

    #[test]
    fn test_dual2_multiplication() {
        let x = Dual::<2, f32>::var_x(3.0);
        let y = Dual::<2, f32>::var_y(4.0);
        let prod = x.mul(y);

        // x * y at (3, 4) = 12
        // d(xy)/dx = y = 4
        // d(xy)/dy = x = 3
        assert_eq!(prod.val, 12.0);
        assert_eq!(prod.partials[0], 4.0); // dy
        assert_eq!(prod.partials[1], 3.0); // dx
    }

    #[test]
    fn test_dual2_chain_rule() {
        // f(x,y) = x² + y² at (3, 4)
        let x = Dual::<2, f32>::var_x(3.0);
        let y = Dual::<2, f32>::var_y(4.0);
        let f = x.mul(x).add(y.mul(y));

        // f = 9 + 16 = 25
        // df/dx = 2x = 6
        // df/dy = 2y = 8
        assert_eq!(f.val, 25.0);
        assert_eq!(f.partials[0], 6.0);
        assert_eq!(f.partials[1], 8.0);
    }
}
