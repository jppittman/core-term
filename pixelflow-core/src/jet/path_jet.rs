//! # PathJet: Ray Space Coordinates
//!
//! A dual number for ray tracing that carries both origin (val) and direction (dir).
//!
//! ## Motivation
//!
//! Standard manifolds assume rays originate from the camera at origin. But for:
//! - Reflected rays bouncing off surfaces
//! - Refracted rays entering/exiting materials
//! - Secondary bounces
//!
//! We need to track WHERE the ray starts, not just its direction.
//!
//! ## Design
//!
//! `PathJet<T>` is generic over the inner scalar type:
//! - `PathJet<Field>` - basic ray without derivatives
//! - `PathJet<Jet3>` - ray with screen-space derivatives for AA
//!
//! Each coordinate (x, y, z, w) is a `PathJet`, where:
//! - `val` = that component of the ray origin
//! - `dir` = that component of the ray direction

use crate::Field;
use crate::numeric::{Computational, Coordinate, Numeric, Selectable};
use core::ops::{Add, BitAnd, BitOr, Div, Mul, Not, Sub};

/// A coordinate in ray space, carrying both origin and direction.
///
/// For a ray `P(t) = origin + t * direction`, each spatial coordinate
/// is represented as a PathJet where:
/// - `val` = origin component (where the ray starts)
/// - `dir` = direction component (where it's going)
#[derive(Clone, Copy, Debug)]
pub struct PathJet<T = Field> {
    /// Origin component (intercept)
    pub val: T,
    /// Direction component (slope)
    pub dir: T,
}

impl<T> PathJet<T> {
    /// Create a new PathJet with explicit origin and direction.
    #[inline(always)]
    pub fn new(val: T, dir: T) -> Self {
        Self { val, dir }
    }
}

impl<T: Computational> PathJet<T> {
    /// Create a PathJet for a ray from origin with given direction.
    ///
    /// `val = 0, dir = slope`
    ///
    /// Use this to convert screen coordinates to rays emanating from camera.
    #[inline(always)]
    pub fn from_slope(slope: T) -> Self {
        Self {
            val: T::from_f32(0.0),
            dir: slope,
        }
    }

    /// Create a PathJet for a point (no direction).
    ///
    /// `val = point, dir = 0`
    ///
    /// Use this for fixed positions that don't vary with the ray parameter.
    #[inline(always)]
    pub fn from_point(point: T) -> Self {
        Self {
            val: point,
            dir: T::from_f32(0.0),
        }
    }

    /// Create a constant PathJet (both origin and direction are the same constant).
    #[inline(always)]
    pub fn constant(val: T) -> Self {
        Self {
            val,
            dir: T::from_f32(0.0),
        }
    }
}

// ============================================================================
// Arithmetic: Component-wise operations
// ============================================================================

impl<T: Add<Output = T>> Add for PathJet<T> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            val: self.val + rhs.val,
            dir: self.dir + rhs.dir,
        }
    }
}

impl<T: Sub<Output = T>> Sub for PathJet<T> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            val: self.val - rhs.val,
            dir: self.dir - rhs.dir,
        }
    }
}

impl<T: Mul<Output = T> + Copy> Mul for PathJet<T> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // Component-wise multiplication
        // For the algebra to work in manifold composition
        Self {
            val: self.val * rhs.val,
            dir: self.dir * rhs.dir,
        }
    }
}

impl<T: Div<Output = T> + Copy> Div for PathJet<T> {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self {
            val: self.val / rhs.val,
            dir: self.dir / rhs.dir,
        }
    }
}

impl<T: BitAnd<Output = T>> BitAnd for PathJet<T> {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self {
            val: self.val & rhs.val,
            dir: self.dir & rhs.dir,
        }
    }
}

impl<T: BitOr<Output = T>> BitOr for PathJet<T> {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self {
            val: self.val | rhs.val,
            dir: self.dir | rhs.dir,
        }
    }
}

impl<T: Not<Output = T>> Not for PathJet<T> {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self {
        Self {
            val: !self.val,
            dir: !self.dir,
        }
    }
}

// ============================================================================
// Computational trait implementation
// ============================================================================

impl<T: Computational> Computational for PathJet<T> {
    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        Self {
            val: T::from_f32(val),
            dir: T::from_f32(0.0), // Constant has no direction
        }
    }

    #[inline(always)]
    fn sequential(start: f32) -> Self {
        Self {
            val: T::sequential(start),
            dir: T::from_f32(0.0),
        }
    }
}

// PathJet<T> is a coordinate type when T is a coordinate type
impl<T: Coordinate> Coordinate for PathJet<T> {}

// ============================================================================
// Selectable trait implementation
// ============================================================================

impl<T: Selectable> Selectable for PathJet<T> {
    #[inline(always)]
    fn select_raw(mask: Field, if_true: Self, if_false: Self) -> Self {
        Self {
            val: T::select_raw(mask, if_true.val, if_false.val),
            dir: T::select_raw(mask, if_true.dir, if_false.dir),
        }
    }
}

// ============================================================================
// Numeric trait implementation
// ============================================================================

impl<T: Numeric> Numeric for PathJet<T>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    #[inline(always)]
    fn sqrt(self) -> Self {
        Self {
            val: self.val.sqrt(),
            dir: self.dir.sqrt(),
        }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        Self {
            val: self.val.abs(),
            dir: self.dir.abs(),
        }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        Self {
            val: self.val.min(rhs.val),
            dir: self.dir.min(rhs.dir),
        }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        Self {
            val: self.val.max(rhs.val),
            dir: self.dir.max(rhs.dir),
        }
    }

    #[inline(always)]
    fn lt(self, rhs: Self) -> Self {
        Self {
            val: self.val.lt(rhs.val),
            dir: self.dir.lt(rhs.dir),
        }
    }

    #[inline(always)]
    fn le(self, rhs: Self) -> Self {
        Self {
            val: self.val.le(rhs.val),
            dir: self.dir.le(rhs.dir),
        }
    }

    #[inline(always)]
    fn gt(self, rhs: Self) -> Self {
        Self {
            val: self.val.gt(rhs.val),
            dir: self.dir.gt(rhs.dir),
        }
    }

    #[inline(always)]
    fn ge(self, rhs: Self) -> Self {
        Self {
            val: self.val.ge(rhs.val),
            dir: self.dir.ge(rhs.dir),
        }
    }

    #[inline(always)]
    fn select(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self {
            val: T::select(mask.val, if_true.val, if_false.val),
            dir: T::select(mask.dir, if_true.dir, if_false.dir),
        }
    }

    #[inline(always)]
    fn select_raw(mask: Self, if_true: Self, if_false: Self) -> Self {
        Self {
            val: T::select_raw(mask.val, if_true.val, if_false.val),
            dir: T::select_raw(mask.dir, if_true.dir, if_false.dir),
        }
    }

    #[inline(always)]
    fn any(&self) -> bool {
        self.val.any() || self.dir.any()
    }

    #[inline(always)]
    fn all(&self) -> bool {
        self.val.all() && self.dir.all()
    }

    #[inline(always)]
    fn from_i32(val: i32) -> Self {
        Self {
            val: T::from_i32(val),
            dir: T::from_f32(0.0),
        }
    }

    #[inline(always)]
    fn from_field(field: Field) -> Self {
        Self {
            val: T::from_field(field),
            dir: T::from_f32(0.0),
        }
    }

    #[inline(always)]
    fn sin(self) -> Self {
        Self {
            val: self.val.sin(),
            dir: self.dir.sin(),
        }
    }

    #[inline(always)]
    fn cos(self) -> Self {
        Self {
            val: self.val.cos(),
            dir: self.dir.cos(),
        }
    }

    #[inline(always)]
    fn atan2(self, x: Self) -> Self {
        Self {
            val: self.val.atan2(x.val),
            dir: self.dir.atan2(x.dir),
        }
    }

    #[inline(always)]
    fn pow(self, exp: Self) -> Self {
        Self {
            val: self.val.pow(exp.val),
            dir: self.dir.pow(exp.dir),
        }
    }

    #[inline(always)]
    fn exp(self) -> Self {
        Self {
            val: self.val.exp(),
            dir: self.dir.exp(),
        }
    }

    #[inline(always)]
    fn log2(self) -> Self {
        Self {
            val: self.val.log2(),
            dir: self.dir.log2(),
        }
    }

    #[inline(always)]
    fn exp2(self) -> Self {
        Self {
            val: self.val.exp2(),
            dir: self.dir.exp2(),
        }
    }

    #[inline(always)]
    fn floor(self) -> Self {
        Self {
            val: self.val.floor(),
            dir: self.dir.floor(),
        }
    }

    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        Self {
            val: self.val.mul_add(b.val, c.val),
            dir: self.dir.mul_add(b.dir, c.dir),
        }
    }

    #[inline(always)]
    fn recip(self) -> Self {
        Self {
            val: self.val.recip(),
            dir: self.dir.recip(),
        }
    }

    #[inline(always)]
    fn rsqrt(self) -> Self {
        Self {
            val: self.val.rsqrt(),
            dir: self.dir.rsqrt(),
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

// ============================================================================
// Conversions
// ============================================================================

/// Field → PathJet<Field>: constant ray at origin
impl From<Field> for PathJet<Field> {
    #[inline(always)]
    fn from(val: Field) -> Self {
        Self::constant(val)
    }
}

/// PathJet<Field> → Field: extract origin (discard direction)
impl From<PathJet<Field>> for Field {
    #[inline(always)]
    fn from(pj: PathJet<Field>) -> Self {
        pj.val
    }
}
