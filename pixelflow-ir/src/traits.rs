//! Core traits for the Intermediate Representation.

use core::fmt::Debug;

/// How an operation should be emitted in generated code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmitStyle {
    /// Not directly emittable (Var, Const handled specially)
    Special,
    /// Unary prefix: `(-a)`
    UnaryPrefix,
    /// Unary method: `(a).sqrt()`
    UnaryMethod,
    /// Binary infix: `(a + b)`
    BinaryInfix(&'static str),
    /// Binary method: `(a).min(b)`
    BinaryMethod,
    /// Binary method with different Rust name: `(a).powf(b)` for pow
    BinaryMethodNamed(&'static str),
    /// Ternary method: `(a).clamp(b, c)`
    TernaryMethod,
}
