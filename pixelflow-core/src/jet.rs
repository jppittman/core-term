//! # Jet Types for Automatic Differentiation
//!
//! - Jet2: value + 2D gradient (∂f/∂x, ∂f/∂y)
//! - Jet3: value + 3D gradient (∂f/∂x, ∂f/∂y, ∂f/∂z) for surface normals

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
        // Use FMA for both terms: fma(a, b, c*d) = a*b + c*d in one SIMD FMA instruction
        Self {
            val: self.val * rhs.val,
            dx: self.dx.mul_add(rhs.val, self.val * rhs.dx),
            dy: self.dy.mul_add(rhs.val, self.val * rhs.dy),
        }
    }
}

impl core::ops::Div for Jet2 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        // Quotient rule: (f / g)' = (f' * g - f * g') / g²
        // Restructure to use FMA: f'*(g/g²) - f*(g'/g²) = f'*(1/g) - f*g'*(1/g²)
        let g_sq = rhs.val * rhs.val;
        let inv_g_sq = Field::from(1.0) / g_sq;  // One division
        let scale = rhs.val * inv_g_sq;          // Pre-compute rhs.val/g_sq
        Self {
            val: self.val / rhs.val,
            dx: self.dx.mul_add(scale, (Field::from(0.0) - self.val) * rhs.dx * inv_g_sq),  // FMA: f'*(g/g²) - f*(g'/g²)
            dy: self.dy.mul_add(scale, (Field::from(0.0) - self.val) * rhs.dy * inv_g_sq),
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
        // Use rsqrt (4 cycles) instead of sqrt (20-30 cycles)
        // sqrt(x) = x * rsqrt(x), derivative = rsqrt(x) / 2
        let rsqrt_val = self.val.rsqrt();
        let sqrt_val = self.val * rsqrt_val;
        let half_rsqrt = rsqrt_val * Field::from(0.5);
        Self {
            val: sqrt_val,
            dx: self.dx * half_rsqrt,
            dy: self.dy * half_rsqrt,
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
        // Compute 1/(x² + y²) once, then multiply instead of 2 sequential divs
        let r_sq = self.val * self.val + x.val * x.val;
        let inv_r_sq = Field::from(1.0) / r_sq;  // One division
        let dy_darg = x.val * inv_r_sq;
        let dx_darg = (Field::from(0.0) - self.val) * inv_r_sq;
        Self {
            val: self.val.atan2(x.val),
            dx: self.dx * dy_darg + x.dx * dx_darg,
            dy: self.dy * dy_darg + x.dy * dx_darg,
        }
    }

    #[inline(always)]
    fn pow(self, exp: Self) -> Self {
        // For f^g: (f^g)' = f^g * (g' * ln(f) + g * f'/f)
        // Compute 1/f once, then use FMA for each derivative
        let val = self.val.pow(exp.val);
        let ln_base = self.val.map_lanes(libm::logf);
        let inv_self = Field::from(1.0) / self.val;  // One division
        let coeff = exp.val * inv_self;               // Common coefficient g/f
        Self {
            val,
            dx: val * exp.dx.mul_add(ln_base, coeff * self.dx),  // FMA: g*ln(f) + g/f*f'
            dy: val * exp.dy.mul_add(ln_base, coeff * self.dy),
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
        // val = a.val * b.val + c.val (already FMA'd above)
        // d/dx = a.dx * b.val + a.val * b.dx + c.dx (triple FMA)
        // Use nested FMA: fma(fma(a.dx, b.val, a.val * b.dx), 1, c.dx)
        Self {
            val: self.val.mul_add(b.val, c.val),
            dx: self.dx.mul_add(b.val, self.val.mul_add(b.dx, c.dx)),
            dy: self.dy.mul_add(b.val, self.val.mul_add(b.dy, c.dy)),
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


// ============================================================================
// Jet3: 3D Automatic Differentiation (for surface normals)
// ============================================================================

/// A 3-jet: value and first derivatives in 3D.
///
/// Represents f(x,y,z) along with ∂f/∂x, ∂f/∂y, and ∂f/∂z.
/// Essential for computing surface normals from SDF gradients.
///
/// # Example: Getting normal from SDF
/// ```ignore
/// // Evaluate SDF with seeded jets at hit point
/// let jx = Jet3::x(hit_x);
/// let jy = Jet3::y(hit_y);
/// let jz = Jet3::z(hit_z);
/// let result = sdf.eval_raw(jx, jy, jz, w);
///
/// // Normal is the normalized gradient
/// let (nx, ny, nz) = result.normal();
/// ```
#[derive(Copy, Clone, Debug)]
pub struct Jet3 {
    /// The function value f(x,y,z)
    pub val: Field,
    /// Partial derivative ∂f/∂x
    pub dx: Field,
    /// Partial derivative ∂f/∂y
    pub dy: Field,
    /// Partial derivative ∂f/∂z
    pub dz: Field,
}

impl Jet3 {
    /// Create a jet seeded for the X variable (∂x/∂x = 1, others = 0)
    #[inline(always)]
    pub fn x(val: Field) -> Self {
        Self {
            val,
            dx: Field::from(1.0),
            dy: Field::from(0.0),
            dz: Field::from(0.0),
        }
    }

    /// Create a jet seeded for the Y variable (∂y/∂y = 1, others = 0)
    #[inline(always)]
    pub fn y(val: Field) -> Self {
        Self {
            val,
            dx: Field::from(0.0),
            dy: Field::from(1.0),
            dz: Field::from(0.0),
        }
    }

    /// Create a jet seeded for the Z variable (∂z/∂z = 1, others = 0)
    #[inline(always)]
    pub fn z(val: Field) -> Self {
        Self {
            val,
            dx: Field::from(0.0),
            dy: Field::from(0.0),
            dz: Field::from(1.0),
        }
    }

    /// Create a constant jet (no derivatives)
    #[inline(always)]
    pub fn constant(val: Field) -> Self {
        Self {
            val,
            dx: Field::from(0.0),
            dy: Field::from(0.0),
            dz: Field::from(0.0),
        }
    }

    /// Get the normalized gradient as a surface normal.
    ///
    /// For an SDF f(p) = 0, the gradient ∇f points outward from the surface.
    /// This function returns the unit normal: ∇f / |∇f|
    #[inline(always)]
    pub fn normal(&self) -> (Field, Field, Field) {
        let len_sq = self.dx * self.dx + self.dy * self.dy + self.dz * self.dz;
        let inv_len = len_sq.rsqrt();
        (self.dx * inv_len, self.dy * inv_len, self.dz * inv_len)
    }

    /// Get the raw gradient without normalization.
    #[inline(always)]
    pub fn gradient(&self) -> (Field, Field, Field) {
        (self.dx, self.dy, self.dz)
    }

    /// Raw select without early exit.
    #[inline(always)]
    pub(crate) fn select_raw(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self {
            val: Field::select_raw(mask.val, if_true.val, if_false.val),
            dx: Field::select_raw(mask.val, if_true.dx, if_false.dx),
            dy: Field::select_raw(mask.val, if_true.dy, if_false.dy),
            dz: Field::select_raw(mask.val, if_true.dz, if_false.dz),
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
            dz: self.dz / two_sqrt,
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
            dz: self.dz * sign,
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
            dz: Field::select_raw(mask, self.dz, rhs.dz),
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
            dz: Field::select_raw(mask, self.dz, rhs.dz),
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
    #[inline(always)]
    pub fn select(mask: Self, if_true: Self, if_false: Self) -> Self {
        if mask.all() { return if_true; }
        if !mask.any() { return if_false; }
        Self::select_raw(mask, if_true, if_false)
    }
}

// ============================================================================
// Arithmetic via chain rule (Jet3)
// ============================================================================

impl core::ops::Add for Jet3 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            val: self.val + rhs.val,
            dx: self.dx + rhs.dx,
            dy: self.dy + rhs.dy,
            dz: self.dz + rhs.dz,
        }
    }
}

impl core::ops::Sub for Jet3 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            val: self.val - rhs.val,
            dx: self.dx - rhs.dx,
            dy: self.dy - rhs.dy,
            dz: self.dz - rhs.dz,
        }
    }
}

impl core::ops::Mul for Jet3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // Product rule: (f * g)' = f' * g + f * g'
        // Use FMA for both terms: fma(a, b, c*d) = a*b + c*d in one SIMD FMA instruction
        Self {
            val: self.val * rhs.val,
            dx: self.dx.mul_add(rhs.val, self.val * rhs.dx),
            dy: self.dy.mul_add(rhs.val, self.val * rhs.dy),
            dz: self.dz.mul_add(rhs.val, self.val * rhs.dz),
        }
    }
}

impl core::ops::Div for Jet3 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        // Quotient rule: (f / g)' = (f' * g - f * g') / g²
        // Restructure to use FMA: f'*(g/g²) - f*(g'/g²) = f'*(1/g) - f*g'*(1/g²)
        let g_sq = rhs.val * rhs.val;
        let inv_g_sq = Field::from(1.0) / g_sq;  // One division
        let scale = rhs.val * inv_g_sq;          // Pre-compute rhs.val/g_sq
        Self {
            val: self.val / rhs.val,
            dx: self.dx.mul_add(scale, (Field::from(0.0) - self.val) * rhs.dx * inv_g_sq),  // FMA: f'*(g/g²) - f*(g'/g²)
            dy: self.dy.mul_add(scale, (Field::from(0.0) - self.val) * rhs.dy * inv_g_sq),
            dz: self.dz.mul_add(scale, (Field::from(0.0) - self.val) * rhs.dz * inv_g_sq),
        }
    }
}

impl core::ops::BitAnd for Jet3 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self::constant(self.val & rhs.val)
    }
}

impl core::ops::BitOr for Jet3 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self::constant(self.val | rhs.val)
    }
}

impl core::ops::Not for Jet3 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self {
            val: !self.val,
            dx: Field::from(0.0),
            dy: Field::from(0.0),
            dz: Field::from(0.0),
        }
    }
}

// ============================================================================
// Computational trait implementation (Jet3)
// ============================================================================

impl Computational for Jet3 {
    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        Self::constant(Field::from(val))
    }

    #[inline(always)]
    fn sequential(start: f32) -> Self {
        Self::constant(Field::sequential(start))
    }
}

// ============================================================================
// Numeric trait implementation (Jet3)
// ============================================================================

impl Numeric for Jet3 {
    #[inline(always)]
    fn sqrt(self) -> Self {
        // Use rsqrt (4 cycles) instead of sqrt (20-30 cycles)
        // sqrt(x) = x * rsqrt(x)
        // d(sqrt(x))/dx = rsqrt(x) / 2
        let rsqrt_val = self.val.rsqrt();
        let sqrt_val = self.val * rsqrt_val;
        let half_rsqrt = rsqrt_val * Field::from(0.5);
        Self {
            val: sqrt_val,
            dx: self.dx * half_rsqrt,
            dy: self.dy * half_rsqrt,
            dz: self.dz * half_rsqrt,
        }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        let sign = self.val / self.val.abs();
        Self {
            val: self.val.abs(),
            dx: self.dx * sign,
            dy: self.dy * sign,
            dz: self.dz * sign,
        }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        let mask = self.val.lt(rhs.val);
        Self {
            val: self.val.min(rhs.val),
            dx: Field::select_raw(mask, self.dx, rhs.dx),
            dy: Field::select_raw(mask, self.dy, rhs.dy),
            dz: Field::select_raw(mask, self.dz, rhs.dz),
        }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        let mask = self.val.gt(rhs.val);
        Self {
            val: self.val.max(rhs.val),
            dx: Field::select_raw(mask, self.dx, rhs.dx),
            dy: Field::select_raw(mask, self.dy, rhs.dy),
            dz: Field::select_raw(mask, self.dz, rhs.dz),
        }
    }

    #[inline(always)]
    fn lt(self, rhs: Self) -> Self {
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
        Self {
            val: Field::select_raw(mask.val, if_true.val, if_false.val),
            dx: Field::select_raw(mask.val, if_true.dx, if_false.dx),
            dy: Field::select_raw(mask.val, if_true.dy, if_false.dy),
            dz: Field::select_raw(mask.val, if_true.dz, if_false.dz),
        }
    }

    #[inline(always)]
    fn any(&self) -> bool {
        self.val.any()
    }

    #[inline(always)]
    fn all(&self) -> bool {
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

    #[inline(always)]
    fn sin(self) -> Self {
        let sin_val = self.val.sin();
        let cos_deriv = self.val.cos();
        Self {
            val: sin_val,
            dx: self.dx * cos_deriv,
            dy: self.dy * cos_deriv,
            dz: self.dz * cos_deriv,
        }
    }

    #[inline(always)]
    fn cos(self) -> Self {
        let cos_val = self.val.cos();
        let neg_sin = Field::from(0.0) - self.val.sin();
        Self {
            val: cos_val,
            dx: self.dx * neg_sin,
            dy: self.dy * neg_sin,
            dz: self.dz * neg_sin,
        }
    }

    #[inline(always)]
    fn atan2(self, x: Self) -> Self {
        // atan2(y, x) derivatives:
        // ∂/∂y = x / (x² + y²)
        // ∂/∂x = -y / (x² + y²)
        // Compute 1/(x² + y²) once, then multiply instead of 2 sequential divs
        let r_sq = self.val * self.val + x.val * x.val;
        let inv_r_sq = Field::from(1.0) / r_sq;  // One division
        let dy_darg = x.val * inv_r_sq;
        let dx_darg = (Field::from(0.0) - self.val) * inv_r_sq;
        Self {
            val: self.val.atan2(x.val),
            dx: self.dx * dy_darg + x.dx * dx_darg,
            dy: self.dy * dy_darg + x.dy * dx_darg,
            dz: self.dz * dy_darg + x.dz * dx_darg,
        }
    }

    #[inline(always)]
    fn pow(self, exp: Self) -> Self {
        // For f^g: (f^g)' = f^g * (g' * ln(f) + g * f'/f)
        // Compute 1/f once, then use FMA for each derivative
        let val = self.val.pow(exp.val);
        let ln_base = self.val.map_lanes(libm::logf);
        let inv_self = Field::from(1.0) / self.val;  // One division
        let coeff = exp.val * inv_self;               // Common coefficient g/f
        Self {
            val,
            dx: val * exp.dx.mul_add(ln_base, coeff * self.dx),  // FMA: g*ln(f) + g/f*f'
            dy: val * exp.dy.mul_add(ln_base, coeff * self.dy),
            dz: val * exp.dz.mul_add(ln_base, coeff * self.dz),
        }
    }

    #[inline(always)]
    fn exp(self) -> Self {
        let exp_val = self.val.exp();
        Self {
            val: exp_val,
            dx: self.dx * exp_val,
            dy: self.dy * exp_val,
            dz: self.dz * exp_val,
        }
    }

    #[inline(always)]
    fn floor(self) -> Self {
        Self::constant(self.val.floor())
    }

    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        // (a * b + c)' where a, b, c are jets
        // val = a.val * b.val + c.val (already FMA'd above)
        // d/dx = a.dx * b.val + a.val * b.dx + c.dx (triple FMA)
        // Use nested FMA: fma(fma(a.dx, b.val, a.val * b.dx), 1, c.dx)
        Self {
            val: self.val.mul_add(b.val, c.val),
            dx: self.dx.mul_add(b.val, self.val.mul_add(b.dx, c.dx)),
            dy: self.dy.mul_add(b.val, self.val.mul_add(b.dy, c.dy)),
            dz: self.dz.mul_add(b.val, self.val.mul_add(b.dz, c.dz)),
        }
    }

    #[inline(always)]
    fn recip(self) -> Self {
        let inv = self.val.recip();
        let neg_inv_sq = Field::from(0.0) - inv * inv;
        Self {
            val: inv,
            dx: self.dx * neg_inv_sq,
            dy: self.dy * neg_inv_sq,
            dz: self.dz * neg_inv_sq,
        }
    }

    #[inline(always)]
    fn rsqrt(self) -> Self {
        let rsqrt_val = self.val.rsqrt();
        let rsqrt_cubed = rsqrt_val * rsqrt_val * rsqrt_val;
        let scale = Field::from(-0.5) * rsqrt_cubed;
        Self {
            val: rsqrt_val,
            dx: self.dx * scale,
            dy: self.dy * scale,
            dz: self.dz * scale,
        }
    }

    #[inline(always)]
    fn add_masked(self, val: Self, mask: Self) -> Self {
        Self {
            val: self.val.add_masked(val.val, mask.val),
            dx: self.dx.add_masked(val.dx, mask.val),
            dy: self.dy.add_masked(val.dy, mask.val),
            dz: self.dz.add_masked(val.dz, mask.val),
        }
    }
}
