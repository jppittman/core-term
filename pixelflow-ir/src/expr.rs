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
    Unary(crate::OpKind, Box<Expr>),
    /// Binary operation (e.g., Add, Sub, Mul, Div).
    Binary(crate::OpKind, Box<Expr>, Box<Expr>),
    /// Ternary operation (e.g., MulAdd, Select).
    Ternary(crate::OpKind, Box<Expr>, Box<Expr>, Box<Expr>),
    /// N-ary operation (e.g., Tuple).
    Nary(crate::OpKind, Vec<Expr>),
}

#[cfg(feature = "alloc")]
impl Expr {
    /// Get the operation kind.
    ///
    /// For leaf nodes, returns `crate::OpKind::Var` or `crate::OpKind::Const`.
    #[inline]
    #[must_use]
    pub fn kind(&self) -> crate::OpKind {
        match self {
            Self::Var(_) => crate::OpKind::Var,
            Self::Const(_) => crate::OpKind::Const,
            Self::Unary(op, _) => *op,
            Self::Binary(op, _, _) => *op,
            Self::Ternary(op, _, _, _) => *op,
            Self::Nary(op, _) => *op,
        }
    }

    /// Alias for `kind()` - used by NNUE feature extraction.
    #[inline]
    #[must_use]
    pub fn op_type(&self) -> crate::OpKind {
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
            Self::Nary(_, children) => 1 + children.iter().map(|c| c.depth()).max().unwrap_or(0),
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
            Self::Nary(_, children) => 1 + children.iter().map(|c| c.node_count()).sum::<usize>(),
        }
    }

    /// Evaluate the expression with given variable values.
    ///
    /// `vars[0]` = X, `vars[1]` = Y, `vars[2]` = Z, `vars[3]` = W
    #[cfg(feature = "std")]
    #[must_use]
    pub fn eval(&self, vars: &[f32; 4]) -> f32 {
        match self {
            Self::Var(i) => *vars.get(*i as usize).unwrap_or_else(|| {
                panic!("Expr::eval: Var({}) out of bounds (max 3 for X/Y/Z/W)", i)
            }),
            Self::Const(c) => *c,
            Self::Unary(op, a) => {
                let a = a.eval(vars);
                match op {
                    crate::OpKind::Neg => -a,
                    crate::OpKind::Sqrt => a.sqrt(),
                    crate::OpKind::Rsqrt => 1.0 / a.sqrt(),
                    crate::OpKind::Abs => a.abs(),
                    crate::OpKind::Recip => 1.0 / a,
                    crate::OpKind::Floor => a.floor(),
                    crate::OpKind::Ceil => a.ceil(),
                    crate::OpKind::Round => a.round(),
                    crate::OpKind::Fract => a.fract(),
                    crate::OpKind::Sin => a.sin(),
                    crate::OpKind::Cos => a.cos(),
                    crate::OpKind::Tan => a.tan(),
                    crate::OpKind::Asin => a.asin(),
                    crate::OpKind::Acos => a.acos(),
                    crate::OpKind::Atan => a.atan(),
                    crate::OpKind::Exp => a.exp(),
                    crate::OpKind::Exp2 => a.exp2(),
                    crate::OpKind::Ln => a.ln(),
                    crate::OpKind::Log2 => a.log2(),
                    crate::OpKind::Log10 => a.log10(),
                    other => panic!("Expr::eval: {:?} is not a unary operation", other),
                }
            }
            Self::Binary(op, a, b) => {
                let a = a.eval(vars);
                let b = b.eval(vars);
                match op {
                    crate::OpKind::Add => a + b,
                    crate::OpKind::Sub => a - b,
                    crate::OpKind::Mul => a * b,
                    crate::OpKind::Div => a / b,
                    crate::OpKind::Min => a.min(b),
                    crate::OpKind::Max => a.max(b),
                    crate::OpKind::MulRsqrt => a / b.sqrt(),
                    crate::OpKind::Pow => a.powf(b),
                    crate::OpKind::Hypot => a.hypot(b),
                    crate::OpKind::Atan2 => a.atan2(b),
                    crate::OpKind::Lt => {
                        if a < b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    crate::OpKind::Le => {
                        if a <= b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    crate::OpKind::Gt => {
                        if a > b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    crate::OpKind::Ge => {
                        if a >= b {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    crate::OpKind::Eq => {
                        if (a - b).abs() < 1e-10 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    crate::OpKind::Ne => {
                        if (a - b).abs() >= 1e-10 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    other => panic!("Expr::eval: {:?} is not a binary operation", other),
                }
            }
            Self::Ternary(op, a, b, c) => {
                let a = a.eval(vars);
                let b = b.eval(vars);
                let c = c.eval(vars);
                match op {
                    crate::OpKind::MulAdd => a.mul_add(b, c),
                    crate::OpKind::Select => {
                        if a != 0.0 {
                            b
                        } else {
                            c
                        }
                    }
                    crate::OpKind::Clamp => a.clamp(b, c),
                    other => panic!("Expr::eval: {:?} is not a ternary operation", other),
                }
            }
            Self::Nary(op, children) => {
                match op {
                    crate::OpKind::Tuple => {
                        // Tuple evaluates to its first element
                        children
                            .first()
                            .map(|c| c.eval(vars))
                            .expect("Expr::eval: Tuple with no children")
                    }
                    other => panic!("Expr::eval: {:?} is not an n-ary operation", other),
                }
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
    #[must_use]
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
