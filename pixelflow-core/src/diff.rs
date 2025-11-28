use crate::batch::Batch;
use core::ops::{Add, Div, Mul, Sub};

/// A differential coordinate: value plus screen-space derivatives.
///
/// This represents a coordinate and how it changes as we move across
/// the screen. Enables automatic mipmapping, anti-aliasing, and
/// resolution-independent rendering.
#[derive(Copy, Clone)]
pub struct DiffCoord {
    /// The coordinate value.
    pub val: Batch<f32>,
    /// Rate of change moving one pixel right (∂/∂screen_x).
    pub dx: Batch<f32>,
    /// Rate of change moving one pixel down (∂/∂screen_y).
    pub dy: Batch<f32>,
}

impl DiffCoord {
    /// Create a differential coordinate with no variation (constant).
    #[inline(always)]
    pub fn constant(val: Batch<f32>) -> Self {
        Self {
            val,
            dx: Batch::splat(0.0),
            dy: Batch::splat(0.0),
        }
    }

    /// The screen-space X coordinate (identity: dx=1, dy=0).
    #[inline(always)]
    pub fn screen_x(x: Batch<f32>) -> Self {
        Self {
            val: x,
            dx: Batch::splat(1.0),
            dy: Batch::splat(0.0),
        }
    }

    /// The screen-space Y coordinate (identity: dx=0, dy=1).
    #[inline(always)]
    pub fn screen_y(y: Batch<f32>) -> Self {
        Self {
            val: y,
            dx: Batch::splat(0.0),
            dy: Batch::splat(1.0),
        }
    }

    /// Compute the magnitude of the gradient (for LOD selection).
    #[inline(always)]
    pub fn gradient_magnitude(&self) -> Batch<f32> {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }

    /// Square root with derivative propagation.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        let sqrt_val = self.val.sqrt();
        // d(sqrt(u)) = du / (2 * sqrt(u))
        let two = Batch::splat(2.0);
        let denom = two * sqrt_val;

        // Avoid division by zero?
        // If val is 0, sqrt is 0, denom is 0.
        // For now assume valid input (non-negative, non-zero for derivative).

        Self {
            val: sqrt_val,
            dx: self.dx / denom,
            dy: self.dy / denom,
        }
    }
}

impl Add for DiffCoord {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            val: self.val + rhs.val,
            dx: self.dx + rhs.dx,   // d(a+b)/dx = da/dx + db/dx
            dy: self.dy + rhs.dy,
        }
    }
}

impl Sub for DiffCoord {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            val: self.val - rhs.val,
            dx: self.dx - rhs.dx,
            dy: self.dy - rhs.dy,
        }
    }
}

impl Mul for DiffCoord {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            val: self.val * rhs.val,
            // Product rule: d(ab)/dx = a*db/dx + b*da/dx
            dx: self.val * rhs.dx + rhs.val * self.dx,
            dy: self.val * rhs.dy + rhs.val * self.dy,
        }
    }
}

impl Div for DiffCoord {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let v_sq = rhs.val * rhs.val;
        Self {
            val: self.val / rhs.val,
            // Quotient rule: d(u/v) = (v*du - u*dv) / v^2
            dx: (rhs.val * self.dx - self.val * rhs.dx) / v_sq,
            dy: (rhs.val * self.dy - self.val * rhs.dy) / v_sq,
        }
    }
}

/// A surface that accepts differential coordinates.
///
/// This enables automatic computation of texture LOD, anti-aliased edges,
/// and resolution-independent rendering. Surfaces implementing this trait
/// receive screen-space derivative information and can use it for filtering.
pub trait DiffSurface: Send + Sync {
    /// The output type of this surface.
    type Output: Copy;

    /// Sample the surface with differential coordinates.
    fn sample_diff(&self, u: DiffCoord, v: DiffCoord) -> Self::Output;
}
