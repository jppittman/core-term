//! Expression AST.
//!
//! This module defines the recursive Expression tree.
//! It requires the `alloc` feature.
//!
//! This is the **canonical** Expr type used across the compiler pipeline.
//! Other crates (pixelflow-nnue, pixelflow-search) should re-export this type.

#[cfg(feature = "alloc")]
use alloc::boxed::Box;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use crate::kind::OpKind;

/// A recursive expression tree.
///
/// This is the canonical IR representation used throughout the compiler.
/// The type is designed for both optimization passes (e-graph) and
/// evaluation (NNUE cost models).
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
pub enum Expr {
    /// Variable reference (index: 0=X, 1=Y, 2=Z, 3=W).
    Var(u8),
    /// Constant floating-point value.
    Const(f32),
    /// Unary operation (e.g., Neg, Sqrt, Abs).
    Unary(OpKind, Box<Expr>),
    /// Binary operation (e.g., Add, Sub, Mul, Div).
    Binary(OpKind, Box<Expr>, Box<Expr>),
    /// Ternary operation (e.g., MulAdd, Select).
    Ternary(OpKind, Box<Expr>, Box<Expr>, Box<Expr>),
    /// N-ary operation (e.g., Tuple).
    Nary(OpKind, Vec<Expr>),
}

#[cfg(feature = "alloc")]
impl Expr {
    /// Get the operation kind.
    ///
    /// For leaf nodes, returns `OpKind::Var` or `OpKind::Const`.
    #[inline]
    pub fn kind(&self) -> OpKind {
        match self {
            Self::Var(_) => OpKind::Var,
            Self::Const(_) => OpKind::Const,
            Self::Unary(op, _) => *op,
            Self::Binary(op, _, _) => *op,
            Self::Ternary(op, _, _, _) => *op,
            Self::Nary(op, _) => *op,
        }
    }

    /// Alias for `kind()` - used by NNUE feature extraction.
    #[inline]
    pub fn op_type(&self) -> OpKind {
        self.kind()
    }

    /// Compute the depth of this expression tree.
    #[must_use]
    pub fn depth(&self) -> usize {
        match self {
            Self::Var(_) | Self::Const(_) => 1,
            Self::Unary(_, a) => 1 + a.depth(),
            Self::Binary(_, a, b) => 1 + a.depth().max(b.depth()),
            Self::Ternary(_, a, b, c) => 1 + a.depth().max(b.depth()).max(c.depth()),
            Self::Nary(_, children) => {
                1 + children.iter().map(|c| c.depth()).max().unwrap_or(0)
            }
        }
    }

    /// Count total nodes in the expression.
    #[must_use]
    pub fn node_count(&self) -> usize {
        match self {
            Self::Var(_) | Self::Const(_) => 1,
            Self::Unary(_, a) => 1 + a.node_count(),
            Self::Binary(_, a, b) => 1 + a.node_count() + b.node_count(),
            Self::Ternary(_, a, b, c) => 1 + a.node_count() + b.node_count() + c.node_count(),
            Self::Nary(_, children) => {
                1 + children.iter().map(|c| c.node_count()).sum::<usize>()
            }
        }
    }

    /// Get the children as a slice (only works for Nary).
    ///
    /// # Panics
    ///
    /// Panics for Unary, Binary, and Ternary variants since their children
    /// are stored in `Box<Expr>` fields, not a `Vec`. Use pattern matching
    /// for full access to children of these variants.
    pub fn children(&self) -> &[Expr] {
        match self {
            Self::Var(_) | Self::Const(_) => &[],
            Self::Nary(_, children) => children,
            Self::Unary(op, _) => panic!(
                "Expr::children() cannot return slice for Unary({:?}) - use pattern matching",
                op
            ),
            Self::Binary(op, _, _) => panic!(
                "Expr::children() cannot return slice for Binary({:?}) - use pattern matching",
                op
            ),
            Self::Ternary(op, _, _, _) => panic!(
                "Expr::children() cannot return slice for Ternary({:?}) - use pattern matching",
                op
            ),
        }
    }
}
