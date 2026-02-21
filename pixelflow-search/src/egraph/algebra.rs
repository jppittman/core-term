//! Type-based algebraic relationships and derived rewrite rules.
//!
//! The algebraic structure is encoded in types, not runtime data.
//! One `InversePair` trait yields four rewrite rules automatically.

use std::marker::PhantomData;

use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::ops::{self, Op};
use super::rewrite::{Rewrite, RewriteAction};

// ============================================================================
// InversePair: The Core Algebraic Relationship
// ============================================================================

/// A complete inverse relationship between operations.
///
/// An inverse pair captures the full algebraic structure:
/// - BASE: The fundamental binary operation (Add, Mul)
/// - INVERSE: The unary inverse operation (Neg, Recip)
/// - DERIVED: Syntactic sugar for BASE(a, INVERSE(b)) (Sub, Div)
/// - IDENTITY: The identity element for BASE (0, 1)
///
/// From one InversePair, we derive four rewrite rules:
/// - Canonicalize: a ⊖ b → a ⊕ inv(b)
/// - Involution: inv(inv(x)) → x
/// - Cancellation: (x ⊕ a) ⊖ a → x
/// - InverseAnnihilation: x ⊕ inv(x) → identity
pub trait InversePair {
    /// The base operation (Add, Mul)
    fn base() -> &'static dyn Op;
    /// The inverse operation (Neg, Recip)
    fn inverse() -> &'static dyn Op;
    /// The derived operation (Sub, Div)
    fn derived() -> &'static dyn Op;
    /// The identity element (0.0 for Add, 1.0 for Mul)
    fn identity() -> f32;
}

/// Addition and Negation are inverses.
/// - x + neg(x) = 0
/// - neg(neg(x)) = x
/// - a - b = a + neg(b)
/// - (x + a) - a = x
pub struct AddNeg;
impl InversePair for AddNeg {
    fn base() -> &'static dyn Op { &ops::Add }
    fn inverse() -> &'static dyn Op { &ops::Neg }
    fn derived() -> &'static dyn Op { &ops::Sub }
    fn identity() -> f32 { 0.0 }
}

/// Multiplication and Reciprocal are inverses.
/// - x * recip(x) = 1
/// - recip(recip(x)) = x
/// - a / b = a * recip(b)
/// - (x * a) / a = x
pub struct MulRecip;
impl InversePair for MulRecip {
    fn base() -> &'static dyn Op { &ops::Mul }
    fn inverse() -> &'static dyn Op { &ops::Recip }
    fn derived() -> &'static dyn Op { &ops::Div }
    fn identity() -> f32 { 1.0 }
}

// ============================================================================
// Helper: Check if node matches an operation by name
// ============================================================================

fn node_matches_op(node: &ENode, op: &dyn Op) -> bool {
    match node {
        ENode::Op { op: node_op, .. } => node_op.name() == op.name(),
        _ => false,
    }
}

// ============================================================================
// Rules Derived from InversePair
// ============================================================================

/// Canonicalize: a ⊖ b → a ⊕ inv(b)
///
/// Reduces the operator set by expressing derived ops in terms of base + inverse.
/// - `Canonicalize::<AddNeg>`: a - b → a + neg(b)
/// - `Canonicalize::<MulRecip>`: a / b → a * recip(b)
pub struct Canonicalize<T: InversePair>(PhantomData<T>);

impl<T: InversePair> Canonicalize<T> {
    pub fn new() -> Box<Self> {
        Box::new(Self(PhantomData))
    }
}

impl<T: InversePair + Send + Sync> Rewrite for Canonicalize<T> {
    fn name(&self) -> &str { "canonicalize" }

    fn apply(&self, _egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if !node_matches_op(node, T::derived()) { return None; }
        let (a, b) = node.binary_operands()?;

        Some(RewriteAction::Canonicalize {
            target: T::base(),
            inverse: T::inverse(),
            a,
            b,
        })
    }
}

/// Involution: inv(inv(x)) → x
///
/// The unary inverse is its own inverse.
/// - `Involution::<AddNeg>`: neg(neg(x)) → x
/// - `Involution::<MulRecip>`: recip(recip(x)) → x
pub struct Involution<T: InversePair>(PhantomData<T>);

impl<T: InversePair> Involution<T> {
    pub fn new() -> Box<Self> {
        Box::new(Self(PhantomData))
    }
}

impl<T: InversePair + Send + Sync> Rewrite for Involution<T> {
    fn name(&self) -> &str { "involution" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if !node_matches_op(node, T::inverse()) { return None; }

        let children = node.children();
        if children.len() != 1 { return None; }
        let inner_id = children[0];

        for inner_node in egraph.nodes(inner_id) {
            if node_matches_op(inner_node, T::inverse()) {
                let inner_children = inner_node.children();
                if inner_children.len() == 1 {
                    return Some(RewriteAction::Union(inner_children[0]));
                }
            }
        }
        None
    }
}

/// Cancellation: (x ⊕ a) ⊖ a → x
///
/// The derived op cancels the base op when applied to the same operand.
/// - `Cancellation::<AddNeg>`: (x + a) - a → x
/// - `Cancellation::<MulRecip>`: (x * a) / a → x
pub struct Cancellation<T: InversePair>(PhantomData<T>);

impl<T: InversePair> Cancellation<T> {
    pub fn new() -> Box<Self> {
        Box::new(Self(PhantomData))
    }
}

impl<T: InversePair + Send + Sync> Rewrite for Cancellation<T> {
    fn name(&self) -> &str { "cancellation" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if !node_matches_op(node, T::derived()) { return None; }
        let (numerator, canceller) = node.binary_operands()?;

        for inner_node in egraph.nodes(numerator) {
            if node_matches_op(inner_node, T::base())
                && let Some((a, b)) = inner_node.binary_operands() {
                    // (a ⊕ b) ⊖ b → a
                    if egraph.find(b) == egraph.find(canceller) {
                        return Some(RewriteAction::Union(a));
                    }
                    // (b ⊕ a) ⊖ b → a (if BASE is commutative)
                    if T::base().is_commutative() && egraph.find(a) == egraph.find(canceller) {
                        return Some(RewriteAction::Union(b));
                    }
                }
        }
        None
    }
}

/// InverseAnnihilation: x ⊕ inv(x) → identity
///
/// An element combined with its inverse yields the identity.
/// - `InverseAnnihilation::<AddNeg>`: x + neg(x) → 0
/// - `InverseAnnihilation::<MulRecip>`: x * recip(x) → 1
pub struct InverseAnnihilation<T: InversePair>(PhantomData<T>);

impl<T: InversePair> InverseAnnihilation<T> {
    pub fn new() -> Box<Self> {
        Box::new(Self(PhantomData))
    }
}

impl<T: InversePair + Send + Sync> Rewrite for InverseAnnihilation<T> {
    fn name(&self) -> &str { "inverse-annihilation" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if !node_matches_op(node, T::base()) { return None; }
        let (a, b) = node.binary_operands()?;

        // x ⊕ inv(x) → identity
        for node_b in egraph.nodes(b) {
            if node_matches_op(node_b, T::inverse())
                && let Some(&inner) = node_b.children().first()
                    && egraph.find(inner) == egraph.find(a) {
                        return Some(RewriteAction::Create(ENode::constant(T::identity())));
                    }
        }

        // inv(x) ⊕ x → identity
        for node_a in egraph.nodes(a) {
            if node_matches_op(node_a, T::inverse())
                && let Some(&inner) = node_a.children().first()
                    && egraph.find(inner) == egraph.find(b) {
                        return Some(RewriteAction::Create(ENode::constant(T::identity())));
                    }
        }

        None
    }
}
