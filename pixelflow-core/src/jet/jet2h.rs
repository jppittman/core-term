//! # Jet2H: 2D Automatic Differentiation with Hessian (second derivatives)

use crate::Field;
use crate::Manifold;
use crate::ManifoldExt;
use crate::numeric::{Computational, Numeric, Selectable};

/// A 2-jet with Hessian: value, first derivatives, and second derivatives.
///
/// Represents f(x,y) along with ∂f/∂x, ∂f/∂y (gradient),
/// and ∂²f/∂x², ∂²f/∂x∂y, ∂²f/∂y² (Hessian matrix).
///
/// The Hessian matrix is:
/// ```text
/// H = [dxx  dxy]
///     [dxy  dyy]
/// ```
///
/// When manifolds are evaluated with Jet2H inputs, both first and second
/// derivatives propagate automatically via the chain rule.
///
/// **Internal type.** Used for second-order optimization and curvature analysis.
#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct Jet2H {
    /// The function value f(x,y)
    pub val: Field,
    /// Partial derivative ∂f/∂x
    pub dx: Field,
    /// Partial derivative ∂f/∂y
    pub dy: Field,
    /// Second partial derivative ∂²f/∂x²
    pub dxx: Field,
    /// Mixed partial derivative ∂²f/∂x∂y
    pub dxy: Field,
    /// Second partial derivative ∂²f/∂y²
    pub dyy: Field,
}

impl Jet2H {
    /// Create a jet seeded for the X variable (∂x/∂x = 1, others = 0)
    #[inline(always)]
    pub fn x(val: Field) -> Self {
        Self {
            val,
            dx: Field::from(1.0),
            dy: Field::from(0.0),
            dxx: Field::from(0.0),
            dxy: Field::from(0.0),
            dyy: Field::from(0.0),
        }
    }

    /// Create a jet seeded for the Y variable (∂y/∂y = 1, others = 0)
    #[inline(always)]
    pub fn y(val: Field) -> Self {
        Self {
            val,
            dx: Field::from(0.0),
            dy: Field::from(1.0),
            dxx: Field::from(0.0),
            dxy: Field::from(0.0),
            dyy: Field::from(0.0),
        }
    }

    /// Create a constant jet (no derivatives)
    #[inline(always)]
    pub fn constant(val: Field) -> Self {
        Self {
            val,
            dx: Field::from(0.0),
            dy: Field::from(0.0),
            dxx: Field::from(0.0),
            dxy: Field::from(0.0),
            dyy: Field::from(0.0),
        }
    }

    /// Collapse manifold expressions into a Jet2H.
    ///
    /// Evaluates each component at origin to get concrete Field values.
    /// Use sparingly - prefer keeping expressions as manifolds.
    #[inline(always)]
    pub fn new<V, Dx, Dy, Dxx, Dxy, Dyy>(val: V, dx: Dx, dy: Dy, dxx: Dxx, dxy: Dxy, dyy: Dyy) -> Self
    where
        V: Manifold<Field, Output = Field>,
        Dx: Manifold<Field, Output = Field>,
        Dy: Manifold<Field, Output = Field>,
        Dxx: Manifold<Field, Output = Field>,
        Dxy: Manifold<Field, Output = Field>,
        Dyy: Manifold<Field, Output = Field>,
    {
        Self {
            val: val.constant(),
            dx: dx.constant(),
            dy: dy.constant(),
            dxx: dxx.constant(),
            dxy: dxy.constant(),
            dyy: dyy.constant(),
        }
    }

    /// Raw select without early exit (pub(crate) only).
    #[inline(always)]
    pub(crate) fn select_raw(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self {
            val: Field::select_raw(mask.val, if_true.val, if_false.val),
            dx: Field::select_raw(mask.val, if_true.dx, if_false.dx),
            dy: Field::select_raw(mask.val, if_true.dy, if_false.dy),
            dxx: Field::select_raw(mask.val, if_true.dxx, if_false.dxx),
            dxy: Field::select_raw(mask.val, if_true.dxy, if_false.dxy),
            dyy: Field::select_raw(mask.val, if_true.dyy, if_false.dyy),
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

    /// Square root with first and second derivatives.
    ///
    /// Returns `Jet2HSqrt` which enables automatic rsqrt fusion when divided.
    #[inline(always)]
    pub fn sqrt(self) -> Jet2HSqrt {
        Jet2HSqrt(self)
    }

    /// Absolute value with derivatives.
    #[inline(always)]
    pub fn abs(self) -> Self {
        // |f|' = f' * sign(f)
        // |f|'' = f'' * sign(f) + (f'/|f|) * (f' * sign(f) - f')'
        // Simplified: |f|'' = f'' * sign(f)  (derivative of sign is 0 away from 0)
        let sign = self.val / self.val.abs();
        Self::new(
            self.val.abs(),
            self.dx * sign,
            self.dy * sign,
            self.dxx * sign,
            self.dxy * sign,
            self.dyy * sign,
        )
    }

    /// Element-wise minimum with derivatives.
    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        let mask = self.val.lt(rhs.val);
        Self {
            val: self.val.min(rhs.val),
            dx: Field::select_raw(mask, self.dx, rhs.dx),
            dy: Field::select_raw(mask, self.dy, rhs.dy),
            dxx: Field::select_raw(mask, self.dxx, rhs.dxx),
            dxy: Field::select_raw(mask, self.dxy, rhs.dxy),
            dyy: Field::select_raw(mask, self.dyy, rhs.dyy),
        }
    }

    /// Element-wise maximum with derivatives.
    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self {
        let mask = self.val.gt(rhs.val);
        Self {
            val: self.val.max(rhs.val),
            dx: Field::select_raw(mask, self.dx, rhs.dx),
            dy: Field::select_raw(mask, self.dy, rhs.dy),
            dxx: Field::select_raw(mask, self.dxx, rhs.dxx),
            dxy: Field::select_raw(mask, self.dxy, rhs.dxy),
            dyy: Field::select_raw(mask, self.dyy, rhs.dyy),
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
// Jet2HSqrt: Enables rsqrt fusion for Jet2H
// ============================================================================

/// Wrapper for sqrt(Jet2H) that enables automatic rsqrt fusion.
///
/// When `Jet2H / Jet2HSqrt` is computed, this automatically uses the faster
/// `rsqrt` path: `a / sqrt(b)` becomes `a * rsqrt(b)`.
#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct Jet2HSqrt(Jet2H);

impl Jet2HSqrt {
    /// Evaluate to get the actual sqrt result as Jet2H.
    #[inline(always)]
    pub fn eval(self) -> Jet2H {
        let rsqrt_val = self.0.val.rsqrt();
        let sqrt_val = self.0.val * rsqrt_val;
        let half_rsqrt = rsqrt_val * Field::from(0.5);

        // sqrt(x)' = rsqrt(x) / 2
        let sqrt_dx = self.0.dx * half_rsqrt;
        let sqrt_dy = self.0.dy * half_rsqrt;

        // sqrt(x)'' = ∂/∂x[x' * rsqrt(x) / 2]
        //           = x'' * rsqrt(x) / 2 + x' * (rsqrt(x) / 2)'
        // where (rsqrt(x))' = -x' * rsqrt(x)³ / 2
        let rsqrt_cubed = rsqrt_val * rsqrt_val * rsqrt_val;
        let quarter_rsqrt_cubed = rsqrt_cubed * Field::from(0.25);
        let sqrt_dxx = self.0.dxx * half_rsqrt
            - self.0.dx * self.0.dx * quarter_rsqrt_cubed;
        let sqrt_dxy = self.0.dxy * half_rsqrt
            - self.0.dx * self.0.dy * quarter_rsqrt_cubed;
        let sqrt_dyy = self.0.dyy * half_rsqrt
            - self.0.dy * self.0.dy * quarter_rsqrt_cubed;

        Jet2H::new(sqrt_val, sqrt_dx, sqrt_dy, sqrt_dxx, sqrt_dxy, sqrt_dyy)
    }
}

impl From<Jet2HSqrt> for Jet2H {
    #[inline(always)]
    fn from(s: Jet2HSqrt) -> Jet2H {
        s.eval()
    }
}

/// Rsqrt fusion: `Jet2H / Jet2HSqrt` computes `a * rsqrt(b)` directly.
impl core::ops::Div<Jet2HSqrt> for Jet2H {
    type Output = Jet2H;
    #[inline(always)]
    fn div(self, rhs: Jet2HSqrt) -> Jet2H {
        let b = rhs.0;
        let rsqrt_b = b.val.rsqrt();
        let result_val = self.val * rsqrt_b;

        // d/dx[a * rsqrt(b)] = a' * rsqrt(b) - a * b' * rsqrt(b)³ / 2
        let rsqrt_cubed = rsqrt_b * rsqrt_b * rsqrt_b;
        let half_rsqrt_cubed = rsqrt_cubed * Field::from(0.5);

        // First derivatives
        let result_dx = self.dx * rsqrt_b + self.val * b.dx * half_rsqrt_cubed * Field::from(-1.0);
        let result_dy = self.dy * rsqrt_b + self.val * b.dy * half_rsqrt_cubed * Field::from(-1.0);

        // Second derivatives: d²/dx²[a * rsqrt(b)]
        // = a'' * rsqrt(b) + 2 * a' * (rsqrt(b))'
        //   - a * b'' * rsqrt(b)³ / 2
        //   - a * b' * d/dx[rsqrt(b)³ / 2]
        // where d/dx[rsqrt(b)³] = 3 * rsqrt(b)² * (rsqrt(b))'
        //                        = -3 * b' * rsqrt(b)⁵ / 2
        let rsqrt_fifth = rsqrt_cubed * rsqrt_b * rsqrt_b;
        let term = rsqrt_fifth * Field::from(0.75);  // 3/2 / 2
        let two = Field::from(2.0);

        let result_dxx = self.dxx * rsqrt_b
            + two * self.dx * b.dx * term * Field::from(-1.0)
            + self.val * b.dxx * half_rsqrt_cubed * Field::from(-1.0)
            + self.val * b.dx * b.dx * term;

        let result_dxy = self.dxy * rsqrt_b
            + self.dx * b.dy * term * Field::from(-1.0)
            + self.dy * b.dx * term * Field::from(-1.0)
            + self.val * b.dxy * half_rsqrt_cubed * Field::from(-1.0)
            + self.val * b.dx * b.dy * term;

        let result_dyy = self.dyy * rsqrt_b
            + two * self.dy * b.dy * term * Field::from(-1.0)
            + self.val * b.dyy * half_rsqrt_cubed * Field::from(-1.0)
            + self.val * b.dy * b.dy * term;

        Jet2H::new(
            result_val,
            result_dx,
            result_dy,
            result_dxx,
            result_dxy,
            result_dyy,
        )
    }
}

impl core::ops::Add<Jet2H> for Jet2HSqrt {
    type Output = Jet2H;
    #[inline(always)]
    fn add(self, rhs: Jet2H) -> Jet2H { self.eval() + rhs }
}

impl core::ops::Sub<Jet2H> for Jet2HSqrt {
    type Output = Jet2H;
    #[inline(always)]
    fn sub(self, rhs: Jet2H) -> Jet2H { self.eval() - rhs }
}

impl core::ops::Mul<Jet2H> for Jet2HSqrt {
    type Output = Jet2H;
    #[inline(always)]
    fn mul(self, rhs: Jet2H) -> Jet2H { self.eval() * rhs }
}

impl core::ops::Div<Jet2H> for Jet2HSqrt {
    type Output = Jet2H;
    #[inline(always)]
    fn div(self, rhs: Jet2H) -> Jet2H { self.eval() / rhs }
}

impl core::ops::Add<Jet2HSqrt> for Jet2H {
    type Output = Jet2H;
    #[inline(always)]
    fn add(self, rhs: Jet2HSqrt) -> Jet2H { self + rhs.eval() }
}

impl core::ops::Sub<Jet2HSqrt> for Jet2H {
    type Output = Jet2H;
    #[inline(always)]
    fn sub(self, rhs: Jet2HSqrt) -> Jet2H { self - rhs.eval() }
}

impl core::ops::Mul<Jet2HSqrt> for Jet2H {
    type Output = Jet2H;
    #[inline(always)]
    fn mul(self, rhs: Jet2HSqrt) -> Jet2H { self * rhs.eval() }
}

// ============================================================================
// Arithmetic via chain rule (Jet2H)
// ============================================================================

impl core::ops::Add for Jet2H {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // (f + g)'' = f'' + g''
        Self::new(
            self.val + rhs.val,
            self.dx + rhs.dx,
            self.dy + rhs.dy,
            self.dxx + rhs.dxx,
            self.dxy + rhs.dxy,
            self.dyy + rhs.dyy,
        )
    }
}

impl core::ops::Sub for Jet2H {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        // (f - g)'' = f'' - g''
        Self::new(
            self.val - rhs.val,
            self.dx - rhs.dx,
            self.dy - rhs.dy,
            self.dxx - rhs.dxx,
            self.dxy - rhs.dxy,
            self.dyy - rhs.dyy,
        )
    }
}

impl core::ops::Mul for Jet2H {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // Product rule: (f * g)' = f' * g + f * g'
        // Second derivative: (f * g)'' = f'' * g + 2 * f' * g' + f * g''
        let two = Field::from(2.0);
        Self::new(
            self.val * rhs.val,
            self.dx * rhs.val + self.val * rhs.dx,
            self.dy * rhs.val + self.val * rhs.dy,
            self.dxx * rhs.val + two * self.dx * rhs.dx + self.val * rhs.dxx,
            self.dxy * rhs.val + self.dx * rhs.dy + self.dy * rhs.dx + self.val * rhs.dxy,
            self.dyy * rhs.val + two * self.dy * rhs.dy + self.val * rhs.dyy,
        )
    }
}

impl core::ops::Div for Jet2H {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        // Quotient rule: (f / g)' = (f' * g - f * g') / g²
        // For second derivatives, use (f/g) = f * (1/g):
        // d²/dx²[f * g^(-1)] = f'' * g^(-1) + 2 * f' * (g^(-1))'
        //                      + f * (g^(-1))''
        // where (g^(-1))' = -g' * g^(-2)
        //       (g^(-1))'' = -g'' * g^(-2) + 2 * g'² * g^(-3)

        let inv_g = rhs.val.recip();
        let inv_g_sq = inv_g * inv_g;
        let inv_g_cube = inv_g_sq * inv_g;
        let two = Field::from(2.0);

        // First derivatives
        let dx = self.dx * inv_g + self.val * rhs.dx * inv_g_sq * Field::from(-1.0);
        let dy = self.dy * inv_g + self.val * rhs.dy * inv_g_sq * Field::from(-1.0);

        // Second derivatives
        // d/dx[self.dx * inv_g + self.val * rhs.dx * inv_g_sq * (-1)]
        // = self.dxx * inv_g + self.dx * (-rhs.dx * inv_g_sq)
        //   + self.dx * rhs.dx * (-inv_g_sq) + self.val * rhs.dxx * (-inv_g_sq)
        //   + self.val * rhs.dx * (-2 * rhs.dx * inv_g_cube)
        let dxx = self.dxx * inv_g
            + two * self.dx * rhs.dx * inv_g_sq * Field::from(-1.0)
            + self.val * rhs.dxx * inv_g_sq * Field::from(-1.0)
            + two * self.val * rhs.dx * rhs.dx * inv_g_cube;

        let dxy = self.dxy * inv_g
            + self.dx * rhs.dy * inv_g_sq * Field::from(-1.0)
            + self.dy * rhs.dx * inv_g_sq * Field::from(-1.0)
            + self.val * rhs.dxy * inv_g_sq * Field::from(-1.0)
            + two * self.val * rhs.dx * rhs.dy * inv_g_cube;

        let dyy = self.dyy * inv_g
            + two * self.dy * rhs.dy * inv_g_sq * Field::from(-1.0)
            + self.val * rhs.dyy * inv_g_sq * Field::from(-1.0)
            + two * self.val * rhs.dy * rhs.dy * inv_g_cube;

        Self::new(
            self.val * inv_g,
            dx,
            dy,
            dxx,
            dxy,
            dyy,
        )
    }
}

impl core::ops::BitAnd for Jet2H {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        // Bitwise AND on masks - derivatives are zero (step function)
        Self::constant(self.val & rhs.val)
    }
}

impl core::ops::BitOr for Jet2H {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        // Bitwise OR on masks - derivatives are zero (step function)
        Self::constant(self.val | rhs.val)
    }
}

impl core::ops::Not for Jet2H {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self {
            val: !self.val,
            dx: Field::from(0.0),
            dy: Field::from(0.0),
            dxx: Field::from(0.0),
            dxy: Field::from(0.0),
            dyy: Field::from(0.0),
        }
    }
}

// ============================================================================
// Computational trait implementation (Jet2H)
// ============================================================================

impl Computational for Jet2H {
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
// Selectable trait implementation (Jet2H)
// ============================================================================

impl Selectable for Jet2H {
    #[inline(always)]
    fn select_raw(mask: Field, if_true: Self, if_false: Self) -> Self {
        Self {
            val: <Field as Selectable>::select_raw(mask, if_true.val, if_false.val),
            dx: <Field as Selectable>::select_raw(mask, if_true.dx, if_false.dx),
            dy: <Field as Selectable>::select_raw(mask, if_true.dy, if_false.dy),
            dxx: <Field as Selectable>::select_raw(mask, if_true.dxx, if_false.dxx),
            dxy: <Field as Selectable>::select_raw(mask, if_true.dxy, if_false.dxy),
            dyy: <Field as Selectable>::select_raw(mask, if_true.dyy, if_false.dyy),
        }
    }
}

// ============================================================================
// Numeric trait implementation (Jet2H)
// ============================================================================

impl Numeric for Jet2H {
    #[inline(always)]
    fn sqrt(self) -> Self {
        // Chain rule: (√f)' = f' / (2√f)
        // Using rsqrt: (√f)' = f' * rsqrt(f) / 2
        // Second derivative via product rule
        let rsqrt_val = self.val.rsqrt();
        let sqrt_val = self.val * rsqrt_val;
        let half_rsqrt = rsqrt_val * Field::from(0.5);

        // d/dx[f' * rsqrt(f) / 2] = (f'' * rsqrt(f) + f' * rsqrt(f)') / 2
        // where rsqrt(f)' = -f' * rsqrt(f)³ / 2
        let rsqrt_cubed = rsqrt_val * rsqrt_val * rsqrt_val;
        let quarter_rsqrt_cubed = rsqrt_cubed * Field::from(0.25);
        let sqrt_dxx = self.dxx * half_rsqrt - self.dx * self.dx * quarter_rsqrt_cubed;
        let sqrt_dyy = self.dyy * half_rsqrt - self.dy * self.dy * quarter_rsqrt_cubed;
        let sqrt_dxy = self.dxy * half_rsqrt - self.dx * self.dy * quarter_rsqrt_cubed;

        Self::new(sqrt_val, self.dx * half_rsqrt, self.dy * half_rsqrt, sqrt_dxx, sqrt_dxy, sqrt_dyy)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        let sign = self.val / self.val.abs();
        Self::new(
            self.val.abs(),
            self.dx * sign,
            self.dy * sign,
            self.dxx * sign,
            self.dxy * sign,
            self.dyy * sign,
        )
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        let mask = self.val.lt(rhs.val);
        Self {
            val: self.val.min(rhs.val),
            dx: Field::select_raw(mask, self.dx, rhs.dx),
            dy: Field::select_raw(mask, self.dy, rhs.dy),
            dxx: Field::select_raw(mask, self.dxx, rhs.dxx),
            dxy: Field::select_raw(mask, self.dxy, rhs.dxy),
            dyy: Field::select_raw(mask, self.dyy, rhs.dyy),
        }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        let mask = self.val.gt(rhs.val);
        Self {
            val: self.val.max(rhs.val),
            dx: Field::select_raw(mask, self.dx, rhs.dx),
            dy: Field::select_raw(mask, self.dy, rhs.dy),
            dxx: Field::select_raw(mask, self.dxx, rhs.dxx),
            dxy: Field::select_raw(mask, self.dxy, rhs.dxy),
            dyy: Field::select_raw(mask, self.dyy, rhs.dyy),
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
            dxx: Field::select_raw(mask.val, if_true.dxx, if_false.dxx),
            dxy: Field::select_raw(mask.val, if_true.dxy, if_false.dxy),
            dyy: Field::select_raw(mask.val, if_true.dyy, if_false.dyy),
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
        // sin(f)' = cos(f) * f'
        // sin(f)'' = -sin(f) * (f')² + cos(f) * f''
        let sin_val = self.val.sin();
        let cos_val = self.val.cos();
        let neg_sin_val = -sin_val;

        Self::new(
            sin_val,
            self.dx * cos_val,
            self.dy * cos_val,
            neg_sin_val * self.dx * self.dx + self.dxx * cos_val,
            neg_sin_val * self.dx * self.dy + self.dxy * cos_val,
            neg_sin_val * self.dy * self.dy + self.dyy * cos_val,
        )
    }

    #[inline(always)]
    fn cos(self) -> Self {
        // cos(f)' = -sin(f) * f'
        // cos(f)'' = -cos(f) * (f')² - sin(f) * f''
        let cos_val = self.val.cos();
        let sin_val = self.val.sin();
        let neg_sin_val = sin_val * Field::from(-1.0);
        let neg_cos_val = cos_val * Field::from(-1.0);

        Self::new(
            cos_val,
            self.dx * neg_sin_val,
            self.dy * neg_sin_val,
            neg_cos_val * self.dx * self.dx + self.dxx * neg_sin_val,
            neg_cos_val * self.dx * self.dy + self.dxy * neg_sin_val,
            neg_cos_val * self.dy * self.dy + self.dyy * neg_sin_val,
        )
    }

    #[inline(always)]
    fn atan2(self, x: Self) -> Self {
        // atan2(y, x) = atan(y/x)
        // ∂/∂y = x / (x² + y²)
        // ∂/∂x = -y / (x² + y²)
        // Second derivatives more complex; see below
        let r_sq = self.val * self.val + x.val * x.val;
        let inv_r_sq = Field::from(1.0) / r_sq;
        let dy_darg = x.val * inv_r_sq;
        let dx_darg = self.val * inv_r_sq * Field::from(-1.0);

        // Second derivatives: complex but doable
        // d²atan2 = derivatives of (x * inv_r_sq, -y * inv_r_sq)
        let inv_r_fourth = inv_r_sq * inv_r_sq;
        let two = Field::from(2.0);
        let term = two * inv_r_fourth;

        let d_dy_darg_y = self.val * dy_darg * term * Field::from(-1.0);
        let d_dy_darg_x = inv_r_sq + x.val * x.val * term * Field::from(-1.0);
        let d_dx_darg_y = inv_r_sq * Field::from(-1.0) + self.val * self.val * term;
        let d_dx_darg_x = self.val * dx_darg * term;

        Self::new(
            self.val.atan2(x.val),
            self.dx * dy_darg + x.dx * dx_darg,
            self.dy * dy_darg + x.dy * dx_darg,
            self.dxx * dy_darg + self.dx * d_dy_darg_y * self.dx + x.dxx * dx_darg + x.dx * d_dx_darg_x * x.dx
                + self.dx * x.dx * (d_dy_darg_x + d_dx_darg_y),
            self.dxy * dy_darg + self.dx * d_dy_darg_y * self.dy + x.dxy * dx_darg + x.dx * d_dx_darg_x * x.dy
                + self.dy * x.dx * (d_dy_darg_x + d_dx_darg_y) + self.dx * x.dy * (d_dy_darg_x + d_dx_darg_y),
            self.dyy * dy_darg + self.dy * d_dy_darg_y * self.dy + x.dyy * dx_darg + x.dy * d_dx_darg_x * x.dy
                + self.dy * x.dy * (d_dy_darg_x + d_dx_darg_y),
        )
    }

    #[inline(always)]
    fn pow(self, exp: Self) -> Self {
        // For f^g: (f^g)' = f^g * (g' * ln(f) + g * f'/f)
        // Second derivative is complex; simplified version:
        let val = self.val.pow(exp.val);
        let ln_base = self.val.map_lanes(libm::logf);
        let inv_self = Field::from(1.0) / self.val;
        let coeff = exp.val * inv_self;
        let two = Field::from(2.0);

        Self::new(
            val,
            val * (exp.dx * ln_base + coeff * self.dx),
            val * (exp.dy * ln_base + coeff * self.dy),
            // Simplified Hessian (full version would be very complex)
            self.dxx * val * coeff + two * self.dx * val * inv_self * (exp.dx * ln_base + coeff * self.dx)
                + val * exp.dxx * ln_base - self.dx * self.dx * val * inv_self * inv_self,
            self.dxy * val * coeff + self.dx * val * inv_self * (exp.dy * ln_base + coeff * self.dy)
                + self.dy * val * inv_self * (exp.dx * ln_base + coeff * self.dx) + val * exp.dxy * ln_base,
            self.dyy * val * coeff + two * self.dy * val * inv_self * (exp.dy * ln_base + coeff * self.dy)
                + val * exp.dyy * ln_base - self.dy * self.dy * val * inv_self * inv_self,
        )
    }

    #[inline(always)]
    fn exp(self) -> Self {
        // exp(f)' = exp(f) * f'
        // exp(f)'' = exp(f) * (f')² + exp(f) * f''
        let exp_val = self.val.exp();

        Self::new(
            exp_val,
            self.dx * exp_val,
            self.dy * exp_val,
            exp_val * self.dx * self.dx + exp_val * self.dxx,
            exp_val * self.dx * self.dy + exp_val * self.dxy,
            exp_val * self.dy * self.dy + exp_val * self.dyy,
        )
    }

    #[inline(always)]
    fn floor(self) -> Self {
        // Floor is a step function - derivative is 0 almost everywhere
        Self::constant(self.val.floor())
    }

    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        // (a * b + c)' = a' * b + a * b' + c'
        // (a * b + c)'' = a'' * b + 2 * a' * b' + a * b'' + c''
        let two = Field::from(2.0);
        Self::new(
            self.val.mul_add(b.val, c.val),
            self.dx * b.val + self.val * b.dx + c.dx,
            self.dy * b.val + self.val * b.dy + c.dy,
            self.dxx * b.val + two * self.dx * b.dx + self.val * b.dxx + c.dxx,
            self.dxy * b.val + self.dx * b.dy + self.dy * b.dx + self.val * b.dxy + c.dxy,
            self.dyy * b.val + two * self.dy * b.dy + self.val * b.dyy + c.dyy,
        )
    }

    #[inline(always)]
    fn recip(self) -> Self {
        // (1/f)' = -f'/f²
        // (1/f)'' = -f''/f² + 2*(f')²/f³
        let inv = self.val.recip();
        let inv_sq = inv * inv;
        let inv_cube = inv_sq * inv;
        let neg_inv_sq = inv_sq * Field::from(-1.0);
        let two = Field::from(2.0);

        Self::new(
            inv,
            self.dx * neg_inv_sq,
            self.dy * neg_inv_sq,
            self.dxx * neg_inv_sq + two * self.dx * self.dx * inv_cube,
            self.dxy * neg_inv_sq + two * self.dx * self.dy * inv_cube,
            self.dyy * neg_inv_sq + two * self.dy * self.dy * inv_cube,
        )
    }

    #[inline(always)]
    fn rsqrt(self) -> Self {
        // rsqrt(f)' = -f' * rsqrt(f)³ / 2
        // rsqrt(f)'' = -f'' * rsqrt(f)³ / 2 - 3/2 * f' * rsqrt(f)⁵ * f'
        let rsqrt_val = self.val.rsqrt();
        let rsqrt_cubed = rsqrt_val * rsqrt_val * rsqrt_val;
        let rsqrt_fifth = rsqrt_cubed * rsqrt_val * rsqrt_val;
        let scale = rsqrt_cubed * Field::from(-0.5);
        let scale_second = rsqrt_fifth * Field::from(-1.5);

        Self::new(
            rsqrt_val,
            self.dx * scale,
            self.dy * scale,
            self.dxx * scale + self.dx * self.dx * scale_second,
            self.dxy * scale + self.dx * self.dy * scale_second,
            self.dyy * scale + self.dy * self.dy * scale_second,
        )
    }

    #[inline(always)]
    fn add_masked(self, val: Self, mask: Self) -> Self {
        Self {
            val: self.val.add_masked(val.val, mask.val),
            dx: self.dx.add_masked(val.dx, mask.val),
            dy: self.dy.add_masked(val.dy, mask.val),
            dxx: self.dxx.add_masked(val.dxx, mask.val),
            dxy: self.dxy.add_masked(val.dxy, mask.val),
            dyy: self.dyy.add_masked(val.dyy, mask.val),
        }
    }

    #[inline(always)]
    fn raw_add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline(always)]
    fn raw_sub(self, rhs: Self) -> Self {
        self - rhs
    }

    #[inline(always)]
    fn raw_mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline(always)]
    fn raw_div(self, rhs: Self) -> Self {
        self / rhs
    }
}
