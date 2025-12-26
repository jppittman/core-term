//! # Jet Types for Automatic Differentiation
//!
//! 2-jets: value + gradient (∂f/∂x, ∂f/∂y)

use crate::Field;
use crate::numeric::{Computational, Numeric};

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

    /// Raw select without early exit (pub(crate) only).
    #[inline(always)]
    pub(crate) fn select_raw(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self {
            val: Field::select_raw(mask.val, if_true.val, if_false.val),
            dx: Field::select_raw(mask.val, if_true.dx, if_false.dx),
            dy: Field::select_raw(mask.val, if_true.dy, if_false.dy),
        }
    }

    // ========================================================================
    // Public methods for comparison and math operations
    // ========================================================================

    /// Less than comparison (returns mask jet).
    #[inline(always)]
    pub fn lt(self, rhs: Self) -> Self {
        Self::constant(self.val.lt(rhs.val))
    }

    /// Less than or equal (returns mask jet).
    #[inline(always)]
    pub fn le(self, rhs: Self) -> Self {
        Self::constant(self.val.le(rhs.val))
    }

    /// Greater than comparison (returns mask jet).
    #[inline(always)]
    pub fn gt(self, rhs: Self) -> Self {
        Self::constant(self.val.gt(rhs.val))
    }

    /// Greater than or equal (returns mask jet).
    #[inline(always)]
    pub fn ge(self, rhs: Self) -> Self {
        Self::constant(self.val.ge(rhs.val))
    }

    /// Square root with derivative.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        // Chain rule: (√f)' = f' / (2√f)
        let sqrt_val = self.val.sqrt();
        let two_sqrt = Field::from(2.0) * sqrt_val;
        Self {
            val: sqrt_val,
            dx: self.dx / two_sqrt,
            dy: self.dy / two_sqrt,
        }
    }

    /// Absolute value with derivative.
    #[inline(always)]
    pub fn abs(self) -> Self {
        // |f|' = f' * sign(f)
        let sign = self.val / self.val.abs();
        Self {
            val: self.val.abs(),
            dx: self.dx * sign,
            dy: self.dy * sign,
        }
    }

    /// Element-wise minimum with derivative.
    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        let mask = self.val.lt(rhs.val);
        Self {
            val: self.val.min(rhs.val),
            dx: Field::select_raw(mask, self.dx, rhs.dx),
            dy: Field::select_raw(mask, self.dy, rhs.dy),
        }
    }

    /// Element-wise maximum with derivative.
    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self {
        let mask = self.val.gt(rhs.val);
        Self {
            val: self.val.max(rhs.val),
            dx: Field::select_raw(mask, self.dx, rhs.dx),
            dy: Field::select_raw(mask, self.dy, rhs.dy),
        }
    }

    /// Check if any lane of the value is non-zero.
    #[inline(always)]
    pub fn any(&self) -> bool {
        self.val.any()
    }

    /// Check if all lanes of the value are non-zero.
    #[inline(always)]
    pub fn all(&self) -> bool {
        self.val.all()
    }

    /// Conditional select with early-exit optimization.
    /// Returns if_true where mask is set, if_false elsewhere.
    #[inline(always)]
    pub fn select(mask: Self, if_true: Self, if_false: Self) -> Self {
        if mask.all() { return if_true; }
        if !mask.any() { return if_false; }
        Self::select_raw(mask, if_true, if_false)
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

impl core::ops::BitAnd for Jet2 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        // Bitwise AND on masks - derivatives are zero (step function)
        Self::constant(self.val & rhs.val)
    }
}

impl core::ops::BitOr for Jet2 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        // Bitwise OR on masks - derivatives are zero (step function)
        Self::constant(self.val | rhs.val)
    }
}

impl core::ops::Not for Jet2 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self {
            val: !self.val,
            dx: Field::from(0.0),
            dy: Field::from(0.0),
        }
    }
}

// ============================================================================
// Computational trait implementation (Public API)
// ============================================================================

impl Computational for Jet2 {
    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        Self::constant(Field::from(val))
    }

    #[inline(always)]
    fn sequential(start: f32) -> Self {
        // Zero derivatives - users wrap with Jet2::x() to seed X-differentiation
        Self::constant(Field::sequential(start))
    }
}

// ============================================================================
// Numeric trait implementation (Internal)
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
        // This is a true blend - both derivatives already computed
        let mask = self.val.lt(rhs.val);
        Self {
            val: self.val.min(rhs.val),
            dx: Field::select_raw(mask, self.dx, rhs.dx),
            dy: Field::select_raw(mask, self.dy, rhs.dy),
        }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        // max(f,g)' = f' if f > g, g' otherwise
        let mask = self.val.gt(rhs.val);
        Self {
            val: self.val.max(rhs.val),
            dx: Field::select_raw(mask, self.dx, rhs.dx),
            dy: Field::select_raw(mask, self.dy, rhs.dy),
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
        if mask.all() { return if_true; }
        if !mask.any() { return if_false; }
        Self::select_raw(mask, if_true, if_false)
    }

    #[inline(always)]
    fn select_raw(mask: Self, if_true: Self, if_false: Self) -> Self {
        // Blend values and derivatives
        Self {
            val: Field::select_raw(mask.val, if_true.val, if_false.val),
            dx: Field::select_raw(mask.val, if_true.dx, if_false.dx),
            dy: Field::select_raw(mask.val, if_true.dy, if_false.dy),
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
    fn from_i32(val: i32) -> Self {
        Self::constant(Field::from(val))
    }

    #[inline(always)]
    fn from_field(field: Field) -> Self {
        Self::constant(field)
    }

    // ========================================================================
    // Trigonometric Operations with Automatic Differentiation
    // ========================================================================

    #[inline(always)]
    fn sin(self) -> Self {
        // Chain rule: (sin f)' = cos(f) * f'
        let sin_val = self.val.sin();
        let cos_deriv = self.val.cos();
        Self {
            val: sin_val,
            dx: self.dx * cos_deriv,
            dy: self.dy * cos_deriv,
        }
    }

    #[inline(always)]
    fn cos(self) -> Self {
        // Chain rule: (cos f)' = -sin(f) * f'
        let cos_val = self.val.cos();
        let neg_sin = Field::from(0.0) - self.val.sin();
        Self {
            val: cos_val,
            dx: self.dx * neg_sin,
            dy: self.dy * neg_sin,
        }
    }

    #[inline(always)]
    fn atan2(self, x: Self) -> Self {
        // atan2(y, x) derivatives:
        // ∂/∂y = x / (x² + y²)
        // ∂/∂x = -y / (x² + y²)
        let r_sq = self.val * self.val + x.val * x.val;
        let dy_darg = x.val / r_sq;
        let dx_darg = (Field::from(0.0) - self.val) / r_sq;
        Self {
            val: self.val.atan2(x.val),
            dx: self.dx * dy_darg + x.dx * dx_darg,
            dy: self.dy * dy_darg + x.dy * dx_darg,
        }
    }

    #[inline(always)]
    fn pow(self, exp: Self) -> Self {
        // For f^g: (f^g)' = f^g * (g' * ln(f) + g * f'/f)
        // Simplified case when g is constant: (f^c)' = c * f^(c-1) * f'
        let val = self.val.pow(exp.val);
        // Use general formula
        let ln_base = self.val.map_lanes(libm::logf);
        let scale = val * (exp.dx * ln_base + exp.val * self.dx / self.val);
        let scale_y = val * (exp.dy * ln_base + exp.val * self.dy / self.val);
        Self {
            val,
            dx: scale,
            dy: scale_y,
        }
    }

    #[inline(always)]
    fn exp(self) -> Self {
        // Chain rule: (exp f)' = exp(f) * f'
        let exp_val = self.val.exp();
        Self {
            val: exp_val,
            dx: self.dx * exp_val,
            dy: self.dy * exp_val,
        }
    }

    #[inline(always)]
    fn floor(self) -> Self {
        // Floor is a step function - derivative is 0 almost everywhere
        Self::constant(self.val.floor())
    }

    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        // (a * b + c)' where a, b, c are jets
        // val = a.val * b.val + c.val
        // d/dx = a.dx * b.val + a.val * b.dx + c.dx  (product rule + chain rule)
        // d/dy = a.dy * b.val + a.val * b.dy + c.dy
        Self {
            val: self.val.mul_add(b.val, c.val),
            dx: self.dx * b.val + self.val * b.dx + c.dx,
            dy: self.dy * b.val + self.val * b.dy + c.dy,
        }
    }

    #[inline(always)]
    fn recip(self) -> Self {
        // (1/f)' = -f'/f²
        let inv = self.val.recip();
        let neg_inv_sq = Field::from(0.0) - inv * inv;
        Self {
            val: inv,
            dx: self.dx * neg_inv_sq,
            dy: self.dy * neg_inv_sq,
        }
    }

    #[inline(always)]
    fn rsqrt(self) -> Self {
        // d/dx[1/√f] = -f' / (2 * f * √f) = -f' * rsqrt(f) / (2f)
        // = -f' / (2 * f^(3/2)) = -f' * rsqrt(f)³ / 2
        let rsqrt_val = self.val.rsqrt();
        let rsqrt_cubed = rsqrt_val * rsqrt_val * rsqrt_val;
        let scale = Field::from(-0.5) * rsqrt_cubed;
        Self {
            val: rsqrt_val,
            dx: self.dx * scale,
            dy: self.dy * scale,
        }
    }

    #[inline(always)]
    fn add_masked(self, val: Self, mask: Self) -> Self {
        // For jets, mask.val is the actual mask
        Self {
            val: self.val.add_masked(val.val, mask.val),
            dx: self.dx.add_masked(val.dx, mask.val),
            dy: self.dy.add_masked(val.dy, mask.val),
        }
    }
}
