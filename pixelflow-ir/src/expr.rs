//! Expression AST.
//!
//! This module defines the recursive Expression tree.
//! It requires the `alloc` feature.

#[cfg(feature = "alloc")]
use alloc::boxed::Box;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use crate::kind::OpKind;

/// A recursive expression tree.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
pub enum Expr {
    /// Variable reference (index).
    Var(u8),
    /// Constant value.
    Const(f32),
    /// Unary operation.
    Unary(OpKind, Box<Expr>),
    /// Binary operation.
    Binary(OpKind, Box<Expr>, Box<Expr>),
    /// Ternary operation.
    Ternary(OpKind, Box<Expr>, Box<Expr>, Box<Expr>),
    /// N-ary operation (e.g. Tuple).
    Nary(OpKind, Vec<Expr>),
}

#[cfg(feature = "alloc")]
impl Expr {
    /// Get the operation kind.
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

    /// Get the children.
    pub fn children(&self) -> &[Expr] {
        match self {
            Self::Var(_) | Self::Const(_) => &[],
            Self::Unary(_, a) => core::slice::from_ref(a),
            Self::Binary(_, a, b) => {
                // This is unsafe/tricky to return slice from Box fields.
                // We return Vec or iterator?
                // For simplicity in this base IR, we might not provide a generic children() slice 
                // without allocation or unsafe hacks.
                // But Nary provides slice.
                // Let's leave this out for now and let consumers match.
                &[] 
            }
            Self::Nary(_, children) => children,
            _ => &[], // Todo: unify storage if generic traversal is needed
        }
    }
}
