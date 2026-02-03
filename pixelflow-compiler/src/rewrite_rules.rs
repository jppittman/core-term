//! Compiler-specific rewrite rules for CPU instruction selection.
//!
//! These rules are NOT mathematical identities - they encode knowledge about
//! CPU instruction sets and performance. Mathematical rules live in
//! `pixelflow_search::math`.
//!
//! ## FMA (Fused Multiply-Add)
//! Most modern CPUs have FMA instructions:
//! - `a * b + c` → `fma(a, b, c)` (one instruction instead of two)
//! - `a * b - c` → `fms(a, b, c)` (fused multiply-subtract)
//!
//! ## Reciprocal Square Root (rsqrt)
//! `1 / sqrt(x)` → `rsqrt(x)` - common in vector normalization
//!
//! ## Multiplication by Rsqrt (Vector Normalization)
//! `x * rsqrt(y)` → `mulrsqrt(x, y)` - single fused operation

use pixelflow_ir::OpKind;
use pixelflow_search::egraph::{EGraph, EClassId, ENode, Rewrite, RewriteAction, ops};

// ============================================================================
// FMA Fusion
// ============================================================================

/// Fused Multiply-Add: a * b + c → muladd(a, b, c)
///
/// Modern CPUs (AVX2, ARM NEON) have single-instruction FMA.
/// This reduces latency and improves numerical precision (one rounding).
pub struct FmaFusion;

impl FmaFusion {
    pub fn new() -> Box<Self> {
        Box::new(Self)
    }
}

impl Default for FmaFusion {
    fn default() -> Self {
        Self
    }
}

impl Rewrite for FmaFusion {
    fn name(&self) -> &str { "fma-fusion" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: Add(Mul(a, b), c)
        let ENode::Op { op, children } = node else { return None };
        if op.kind() != OpKind::Add { return None; }
        if children.len() != 2 { return None; }

        let left = children[0];
        let right = children[1];

        // Try left = Mul(a, b), right = c
        if let Some((a, b)) = extract_mul(egraph, left) {
            return Some(RewriteAction::Create(ENode::Op {
                op: &ops::MulAdd,
                children: vec![a, b, right],
            }));
        }

        // Try left = c, right = Mul(a, b)
        if let Some((a, b)) = extract_mul(egraph, right) {
            return Some(RewriteAction::Create(ENode::Op {
                op: &ops::MulAdd,
                children: vec![a, b, left],
            }));
        }

        None
    }
}

// ============================================================================
// Reciprocal Square Root
// ============================================================================

/// Reciprocal square root: 1 / sqrt(x) → rsqrt(x)
///
/// Common pattern in vector normalization: v / |v| = v * rsqrt(dot(v,v))
/// CPUs have fast approximate rsqrt (rsqrtps/vrsqrtps on x86).
pub struct RecipSqrt;

impl RecipSqrt {
    pub fn new() -> Box<Self> {
        Box::new(Self)
    }
}

impl Default for RecipSqrt {
    fn default() -> Self {
        Self
    }
}

impl Rewrite for RecipSqrt {
    fn name(&self) -> &str { "recip-sqrt" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: Div(1, Sqrt(x)) or Recip(Sqrt(x))
        let ENode::Op { op, children } = node else { return None };

        match op.kind() {
            OpKind::Div if children.len() == 2 => {
                // Check if numerator is 1
                let num = children[0];
                let denom = children[1];

                if !is_one(egraph, num) {
                    return None;
                }

                // Check if denominator is sqrt(x)
                if let Some(x) = extract_sqrt(egraph, denom) {
                    return Some(RewriteAction::Create(ENode::Op {
                        op: &ops::Rsqrt,
                        children: vec![x],
                    }));
                }
            }
            OpKind::Recip if children.len() == 1 => {
                // Check if argument is sqrt(x)
                if let Some(x) = extract_sqrt(egraph, children[0]) {
                    return Some(RewriteAction::Create(ENode::Op {
                        op: &ops::Rsqrt,
                        children: vec![x],
                    }));
                }
            }
            _ => {}
        }

        None
    }
}

// ============================================================================
// Multiply by Rsqrt (Vector Normalization)
// ============================================================================

/// Multiply by reciprocal square root: x * rsqrt(y) → mulrsqrt(x, y)
///
/// This pattern appears in vector normalization:
/// normalize(v) = v * rsqrt(dot(v, v))
pub struct MulRsqrt;

impl MulRsqrt {
    pub fn new() -> Box<Self> {
        Box::new(Self)
    }
}

impl Default for MulRsqrt {
    fn default() -> Self {
        Self
    }
}

impl Rewrite for MulRsqrt {
    fn name(&self) -> &str { "mul-rsqrt" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match: Mul(x, Rsqrt(y))
        let ENode::Op { op, children } = node else { return None };
        if op.kind() != OpKind::Mul { return None; }
        if children.len() != 2 { return None; }

        let left = children[0];
        let right = children[1];

        // Try right = Rsqrt(y)
        if let Some(y) = extract_rsqrt(egraph, right) {
            return Some(RewriteAction::Create(ENode::Op {
                op: &ops::MulRsqrt,
                children: vec![left, y],
            }));
        }

        // Try left = Rsqrt(y)
        if let Some(y) = extract_rsqrt(egraph, left) {
            return Some(RewriteAction::Create(ENode::Op {
                op: &ops::MulRsqrt,
                children: vec![right, y],
            }));
        }

        None
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn extract_mul(egraph: &EGraph, class: EClassId) -> Option<(EClassId, EClassId)> {
    for node in egraph.nodes(class) {
        if let ENode::Op { op, children } = node {
            if op.kind() == OpKind::Mul && children.len() == 2 {
                return Some((children[0], children[1]));
            }
        }
    }
    None
}

fn is_one(egraph: &EGraph, class: EClassId) -> bool {
    for node in egraph.nodes(class) {
        if let ENode::Const(bits) = node {
            let v = f32::from_bits(*bits);
            if (v - 1.0).abs() < 1e-10 {
                return true;
            }
        }
    }
    false
}

fn extract_sqrt(egraph: &EGraph, class: EClassId) -> Option<EClassId> {
    for node in egraph.nodes(class) {
        if let ENode::Op { op, children } = node {
            if op.kind() == OpKind::Sqrt && children.len() == 1 {
                return Some(children[0]);
            }
        }
    }
    None
}

fn extract_rsqrt(egraph: &EGraph, class: EClassId) -> Option<EClassId> {
    for node in egraph.nodes(class) {
        if let ENode::Op { op, children } = node {
            if op.kind() == OpKind::Rsqrt && children.len() == 1 {
                return Some(children[0]);
            }
        }
    }
    None
}

// ============================================================================
// Rule Collection
// ============================================================================

/// All compiler-specific instruction fusion rules.
///
/// These are CPU optimization rules, not mathematical identities.
/// Combine with `pixelflow_search::math::all_math_rules()` for full optimization.
pub fn fusion_rules() -> Vec<Box<dyn Rewrite>> {
    vec![
        FmaFusion::new(),
        RecipSqrt::new(),
        MulRsqrt::new(),
    ]
}
