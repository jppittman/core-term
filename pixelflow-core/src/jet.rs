//! # Jet Types for Automatic Differentiation
//!
//! 2-jets: value + gradient (∂f/∂x, ∂f/∂y)

use crate::Field;
use crate::numeric::Numeric;

/// A 2-jet: value and first derivatives.
///
/// Represents f(x,y) along with ∂f/∂x and ∂f/∂y.
/// When manifolds are evaluated with Jet2 inputs, derivatives
/// propagate automatically via the chain rule.
#[derive(Copy, Clone, Debug)]
pub struct Jet2 {
    /// The function value f(x,y)
    pub val: Field,
    /// Partial derivative ∂f/∂x
    pub dx: Field,
    /// Partial derivative ∂f/∂y
    pub dy: Field,
}

impl Jet2 {
    /// Create a jet seeded for the X variable (∂x/∂x = 1, ∂x/∂y = 0)
    #[inline(always)]
    pub fn x(val: Field) -> Self {
        Self {
            val,
            dx: Field::from(1.0),
            dy: Field::from(0.0),
        }
    }

    /// Create a jet seeded for the Y variable (∂y/∂x = 0, ∂y/∂y = 1)
    #[inline(always)]
    pub fn y(val: Field) -> Self {
        Self {
            val,
            dx: Field::from(0.0),
            dy: Field::from(1.0),
        }
    }

    /// Create a constant jet (no derivatives)
    #[inline(always)]
    pub fn constant(val: Field) -> Self {
        Self {
            val,
            dx: Field::from(0.0),
            dy: Field::from(0.0),
        }
    }
}

// ============================================================================
// Arithmetic via chain rule
// ============================================================================

impl core::ops::Add for Jet2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // (f + g)' = f' + g'
        Self {
            val: self.val + rhs.val,
            dx: self.dx + rhs.dx,
            dy: self.dy + rhs.dy,
        }
    }
}

impl core::ops::Sub for Jet2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        // (f - g)' = f' - g'
        Self {
            val: self.val - rhs.val,
            dx: self.dx - rhs.dx,
            dy: self.dy - rhs.dy,
        }
    }
}

impl core::ops::Mul for Jet2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // Product rule: (f * g)' = f' * g + f * g'
        Self {
            val: self.val * rhs.val,
            dx: self.dx * rhs.val + self.val * rhs.dx,
            dy: self.dy * rhs.val + self.val * rhs.dy,
        }
    }
}

impl core::ops::Div for Jet2 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        // Quotient rule: (f / g)' = (f' * g - f * g') / g²
        let g_sq = rhs.val * rhs.val;
        Self {
            val: self.val / rhs.val,
            dx: (self.dx * rhs.val - self.val * rhs.dx) / g_sq,
            dy: (self.dy * rhs.val - self.val * rhs.dy) / g_sq,
        }
    }
}

// ============================================================================
// Numeric trait implementation
// ============================================================================

impl Numeric for Jet2 {
    #[inline(always)]
    fn sqrt(self) -> Self {
        // Chain rule: (√f)' = f' / (2√f)
        let sqrt_val = self.val.sqrt();
        let two_sqrt = Field::from(2.0) * sqrt_val;
        Self {
            val: sqrt_val,
            dx: self.dx / two_sqrt,
            dy: self.dy / two_sqrt,
        }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // |f|' = f' * sign(f)
        // Note: derivative undefined at f=0, we use sign
        let sign = self.val / self.val.abs(); // NaN at zero, but close enough
        Self {
            val: self.val.abs(),
            dx: self.dx * sign,
            dy: self.dy * sign,
        }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        // min(f,g)' = f' if f < g, g' otherwise
        // The mask determines which derivative to use
        let mask = self.val.lt(rhs.val);
        Self {
            val: self.val.min(rhs.val),
            dx: Field::select(mask, self.dx, rhs.dx),
            dy: Field::select(mask, self.dy, rhs.dy),
        }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        // max(f,g)' = f' if f > g, g' otherwise
        let mask = self.val.gt(rhs.val);
        Self {
            val: self.val.max(rhs.val),
            dx: Field::select(mask, self.dx, rhs.dx),
            dy: Field::select(mask, self.dy, rhs.dy),
        }
    }

    #[inline(always)]
    fn lt(self, rhs: Self) -> Self {
        // Comparison only looks at values, derivatives are zero
        // (derivative of a step function is 0 almost everywhere)
        Self::constant(self.val.lt(rhs.val))
    }

    #[inline(always)]
    fn le(self, rhs: Self) -> Self {
        Self::constant(self.val.le(rhs.val))
    }

    #[inline(always)]
    fn gt(self, rhs: Self) -> Self {
        Self::constant(self.val.gt(rhs.val))
    }

    #[inline(always)]
    fn ge(self, rhs: Self) -> Self {
        Self::constant(self.val.ge(rhs.val))
    }

    #[inline(always)]
    fn select(mask: Self, if_true: Self, if_false: Self) -> Self {
        // Use the mask's value to select, blend derivatives
        Self {
            val: Field::select(mask.val, if_true.val, if_false.val),
            dx: Field::select(mask.val, if_true.dx, if_false.dx),
            dy: Field::select(mask.val, if_true.dy, if_false.dy),
        }
    }

    #[inline(always)]
    fn any(&self) -> bool {
        // Check if any lane of the VALUE is true
        // (derivatives don't matter for control flow)
        self.val.any()
    }

    #[inline(always)]
    fn all(&self) -> bool {
        // Check if all lanes of the VALUE are true
        self.val.all()
    }

    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        Self::constant(Field::from(val))
    }

    #[inline(always)]
    fn from_i32(val: i32) -> Self {
        Self::constant(Field::from(val))
    }
}
