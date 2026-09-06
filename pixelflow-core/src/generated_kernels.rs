//! Auto-generated kernels by xtask codegen.

use crate::combinator::Manifold;
use crate::{Field};

/// Discriminant: d^2 - (c^2 - r^2)
/// Optimized form: d.mul_add(d, -c.mul_add(c, -(r * r)))
pub fn discriminant(d: Field, c: Field, r: Field) -> Field {
    d.mul_add(d, -c.mul_add(c, -(r * r)))
}
