//! Core traits for the Intermediate Representation.

use core::fmt::Debug;
use core::hash::Hash;
use crate::kind::OpKind;

/// Base trait for operation arity.
pub trait Arity {
    /// Number of operands.
    const ARITY: usize;
}

/// Marker trait for nullary operations (0 operands).
pub trait Nullary: Arity {}

/// Marker trait for unary operations (1 operand).
pub trait Unary: Arity {}

/// Marker trait for binary operations (2 operands).
pub trait Binary: Arity {}

/// Marker trait for ternary operations (3 operands).
pub trait Ternary: Arity {}

/// Marker trait for variadic/n-ary operations.
pub trait Nary: Arity {}

/// A static operation in the IR.
/// 
/// This trait is implemented by ZSTs (unit structs) representing individual
/// operations. It defines the ISA properties at the type level.
pub trait Op: 'static + Arity + Eq + Hash + Copy + Clone + Debug + Send + Sync {
    /// Display name of the operation.
    const NAME: &'static str;
    
    /// The unified enum kind for this operation.
    const KIND: OpKind;

    /// Instance method to get arity (delegates to Arity trait).
    #[inline(always)]
    fn arity(&self) -> usize { <Self as Arity>::ARITY }

    /// Instance method to get name (delegates to constant).
    #[inline(always)]
    fn name(&self) -> &'static str { Self::NAME }

    /// Instance method to get kind (delegates to constant).
    #[inline(always)]
    fn kind(&self) -> OpKind { Self::KIND }

    /// Canonical index for feature extraction.
    #[inline(always)]
    fn index(&self) -> usize { Self::KIND as usize }
}