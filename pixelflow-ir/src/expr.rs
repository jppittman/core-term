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
use core::fmt;

/// A recursive expression tree.
///
/// This is the canonical IR representation used throughout the compiler.
/// The type is designed for both optimization passes (e-graph) and
/// evaluation (NNUE cost models).
#[cfg(feature = "alloc")]
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// Variable reference (index: 0=X, 1=Y, 2=Z, 3=W).
    Var(u8),
    /// Constant floating-point value.
    Const(f32),
    /// Captured parameter by index. Ephemeral — must be substituted to [`Expr::Const`]
    /// via [`substitute_params`] before passing to the JIT emitter.
    /// Index is declaration order: first param = 0, second = 1, etc.
    Param(u8),
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
            Self::Const(_) | Self::Param(_) => OpKind::Const,
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

    /// Compute the depth of this expression tree (iterative).
    #[must_use]
    pub fn depth(&self) -> usize {
        // Stack holds (node, depth_at_this_node).
        let mut stack: Vec<(&Expr, usize)> = Vec::new();
        stack.push((self, 1));
        let mut max_depth: usize = 0;

        while let Some((node, d)) = stack.pop() {
            match node {
                Self::Var(_) | Self::Const(_) | Self::Param(_) => {
                    max_depth = max_depth.max(d);
                }
                Self::Unary(_, a) => {
                    stack.push((a, d + 1));
                }
                Self::Binary(_, a, b) => {
                    stack.push((a, d + 1));
                    stack.push((b, d + 1));
                }
                Self::Ternary(_, a, b, c) => {
                    stack.push((a, d + 1));
                    stack.push((b, d + 1));
                    stack.push((c, d + 1));
                }
                Self::Nary(_, children) => {
                    if children.is_empty() {
                        max_depth = max_depth.max(d);
                    } else {
                        for child in children {
                            stack.push((child, d + 1));
                        }
                    }
                }
            }
        }
        max_depth
    }

    /// Returns `true` if this expression contains at least one `Var` node (iterative).
    ///
    /// Expressions without variables are constant-foldable and benchmark
    /// at the timing floor — useless for training.
    #[must_use]
    pub fn has_var(&self) -> bool {
        let mut stack: Vec<&Expr> = Vec::new();
        stack.push(self);

        while let Some(node) = stack.pop() {
            match node {
                Self::Var(_) => return true,
                Self::Const(_) | Self::Param(_) => {}
                Self::Unary(_, a) => stack.push(a),
                Self::Binary(_, a, b) => {
                    stack.push(a);
                    stack.push(b);
                }
                Self::Ternary(_, a, b, c) => {
                    stack.push(a);
                    stack.push(b);
                    stack.push(c);
                }
                Self::Nary(_, children) => {
                    for child in children {
                        stack.push(child);
                    }
                }
            }
        }
        false
    }

    /// Count total nodes in the expression (iterative).
    #[must_use]
    pub fn node_count(&self) -> usize {
        let mut stack: Vec<&Expr> = Vec::new();
        stack.push(self);
        let mut count: usize = 0;

        while let Some(node) = stack.pop() {
            count += 1;
            match node {
                Self::Var(_) | Self::Const(_) | Self::Param(_) => {}
                Self::Unary(_, a) => stack.push(a),
                Self::Binary(_, a, b) => {
                    stack.push(a);
                    stack.push(b);
                }
                Self::Ternary(_, a, b, c) => {
                    stack.push(a);
                    stack.push(b);
                    stack.push(c);
                }
                Self::Nary(_, children) => {
                    for child in children {
                        stack.push(child);
                    }
                }
            }
        }
        count
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
            Self::Var(_) | Self::Const(_) | Self::Param(_) => &[],
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

/// S-expression format that round-trips through `parse_expr`.
///
/// Examples:
///   `Var(0)`, `Const(-3.14)`, `add(Var(0), Var(1))`, `sin(Var(0))`
///   `mul_add(Var(0), Var(1), Const(2.0))`
#[cfg(feature = "alloc")]
impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Var(i) => write!(f, "Var({})", i),
            Expr::Const(v) => write!(f, "Const({})", v),
            Expr::Param(i) => write!(f, "Param({})", i),
            Expr::Unary(op, a) => write!(f, "{}({})", op.name(), a),
            Expr::Binary(op, a, b) => write!(f, "{}({}, {})", op.name(), a, b),
            Expr::Ternary(op, a, b, c) => write!(f, "{}({}, {}, {})", op.name(), a, b, c),
            Expr::Nary(op, children) => {
                write!(f, "{}(", op.name())?;
                for (i, child) in children.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", child)?;
                }
                write!(f, ")")
            }
        }
    }
}
