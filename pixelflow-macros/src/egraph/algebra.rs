//! Type-based algebraic relationships and derived rewrite rules.
//!
//! The algebraic structure is encoded in types, not runtime data.
//! One `InversePair` trait yields four rewrite rules automatically.

use std::marker::PhantomData;

use super::graph::EGraph;
use super::node::{EClassId, ENode, Op};
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
    const BASE: Op;      // Add, Mul
    const INVERSE: Op;   // Neg, Recip
    const DERIVED: Op;   // Sub, Div
    const IDENTITY: f32; // 0, 1
}

/// Addition and Negation are inverses.
/// - x + neg(x) = 0
/// - neg(neg(x)) = x
/// - a - b = a + neg(b)
/// - (x + a) - a = x
pub struct AddNeg;
impl InversePair for AddNeg {
    const BASE: Op = Op::Add;
    const INVERSE: Op = Op::Neg;
    const DERIVED: Op = Op::Sub;
    const IDENTITY: f32 = 0.0;
}

/// Multiplication and Reciprocal are inverses.
/// - x * recip(x) = 1
/// - recip(recip(x)) = x
/// - a / b = a * recip(b)
/// - (x * a) / a = x
pub struct MulRecip;
impl InversePair for MulRecip {
    const BASE: Op = Op::Mul;
    const INVERSE: Op = Op::Recip;
    const DERIVED: Op = Op::Div;
    const IDENTITY: f32 = 1.0;
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

impl<T: InversePair> Rewrite for Canonicalize<T> {
    fn name(&self) -> &str { "canonicalize" }

    fn apply(&self, _egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if node.op() != Some(T::DERIVED) { return None; }
        let (a, b) = node.binary_operands()?;

        Some(RewriteAction::Canonicalize {
            target: T::BASE,
            inverse: T::INVERSE,
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

impl<T: InversePair> Rewrite for Involution<T> {
    fn name(&self) -> &str { "involution" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if node.op() != Some(T::INVERSE) { return None; }

        let children = node.children();
        if children.len() != 1 { return None; }
        let inner_id = children[0];

        for inner_node in egraph.nodes(inner_id) {
            if inner_node.op() == Some(T::INVERSE) {
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

impl<T: InversePair> Rewrite for Cancellation<T> {
    fn name(&self) -> &str { "cancellation" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if node.op() != Some(T::DERIVED) { return None; }
        let (numerator, canceller) = node.binary_operands()?;

        for inner_node in egraph.nodes(numerator) {
            if inner_node.op() == Some(T::BASE) {
                if let Some((a, b)) = inner_node.binary_operands() {
                    // (a ⊕ b) ⊖ b → a
                    if egraph.find(b) == egraph.find(canceller) {
                        return Some(RewriteAction::Union(a));
                    }
                    // (b ⊕ a) ⊖ b → a (if BASE is commutative)
                    if T::BASE.is_commutative() && egraph.find(a) == egraph.find(canceller) {
                        return Some(RewriteAction::Union(b));
                    }
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

impl<T: InversePair> Rewrite for InverseAnnihilation<T> {
    fn name(&self) -> &str { "inverse-annihilation" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        if node.op() != Some(T::BASE) { return None; }
        let (a, b) = node.binary_operands()?;

        // x ⊕ inv(x) → identity
        for node_b in egraph.nodes(b) {
            if node_b.op() == Some(T::INVERSE) {
                if let Some(&inner) = node_b.children().first() {
                    if egraph.find(inner) == egraph.find(a) {
                        return Some(RewriteAction::Create(ENode::constant(T::IDENTITY)));
                    }
                }
            }
        }

        // inv(x) ⊕ x → identity
        for node_a in egraph.nodes(a) {
            if node_a.op() == Some(T::INVERSE) {
                if let Some(&inner) = node_a.children().first() {
                    if egraph.find(inner) == egraph.find(b) {
                        return Some(RewriteAction::Create(ENode::constant(T::IDENTITY)));
                    }
                }
            }
        }

        None
    }
}
