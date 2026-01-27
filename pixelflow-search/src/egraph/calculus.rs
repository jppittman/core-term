//! Calculus Rewrite Rules for Automatic Differentiation
//!
//! This module implements differentiation as egraph rewrite rules. Instead of
//! computing derivatives at runtime with dual numbers, the egraph symbolically
//! differentiates expressions at compile time.
//!
//! ## How It Works
//!
//! 1. User requests derivatives by wrapping expressions with D_X, D_Y operators
//! 2. The egraph applies calculus rules to derive symbolic expressions
//! 3. The optimizer simplifies and fuses the resulting expressions
//! 4. Final code generation produces efficient derivative code
//!
//! ## Supported Rules
//!
//! ### Linearity
//! - D[f + g] → D[f] + D[g]
//! - D[f - g] → D[f] - D[g]
//! - D[c * f] → c * D[f] (c constant)
//! - D[c] → 0
//!
//! ### Product & Quotient
//! - D[f * g] → D[f] * g + f * D[g]
//! - D[f / g] → (D[f] * g - f * D[g]) / (g * g)
//!
//! ### Chain Rule (Transcendentals)
//! - D[sqrt(f)] → D[f] / (2 * sqrt(f))
//! - D[sin(f)] → cos(f) * D[f]
//! - D[cos(f)] → -sin(f) * D[f]
//! - D[exp(f)] → exp(f) * D[f]
//! - D[ln(f)] → D[f] / f
//! - D[pow(f, g)] → pow(f, g) * (D[g] * ln(f) + g * D[f] / f)
//!
//! ### Coordinate Variables
//! - D_X[X] → 1
//! - D_X[Y] → 0
//! - D_X[Z] → 0
//! - (analogous for D_Y, D_Z)

use super::graph::EGraph;
use super::node::{EClassId, ENode};
use super::ops::{self, Op};
use super::rewrite::{Rewrite, RewriteAction};

// ============================================================================
// Derivative Operators
// ============================================================================

/// Derivative with respect to X coordinate
pub struct DiffX;
impl Op for DiffX {
    fn name(&self) -> &'static str { "D_X" }
    fn default_cost(&self) -> usize { 0 } // Free - gets eliminated
}

/// Derivative with respect to Y coordinate
pub struct DiffY;
impl Op for DiffY {
    fn name(&self) -> &'static str { "D_Y" }
    fn default_cost(&self) -> usize { 0 }
}

/// Derivative with respect to Z coordinate
pub struct DiffZ;
impl Op for DiffZ {
    fn name(&self) -> &'static str { "D_Z" }
    fn default_cost(&self) -> usize { 0 }
}

/// Coordinate variable X
pub struct VarX;
impl Op for VarX {
    fn name(&self) -> &'static str { "X" }
    fn default_cost(&self) -> usize { 0 }
}

/// Coordinate variable Y
pub struct VarY;
impl Op for VarY {
    fn name(&self) -> &'static str { "Y" }
    fn default_cost(&self) -> usize { 0 }
}

/// Coordinate variable Z
pub struct VarZ;
impl Op for VarZ {
    fn name(&self) -> &'static str { "Z" }
    fn default_cost(&self) -> usize { 0 }
}

// ============================================================================
// Linearity Rules
// ============================================================================

/// D[f + g] → D[f] + D[g]
pub struct DiffAdd {
    diff_op: &'static dyn Op,
}

impl DiffAdd {
    pub fn new(diff_op: &'static dyn Op) -> Box<Self> {
        Box::new(Self { diff_op })
    }
}

impl Rewrite for DiffAdd {
    fn name(&self) -> &str { "diff-add" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        // Match D[Add(f, g)]
        let node_op = node.op()?;
        if node_op.name() != self.diff_op.name() { return None; }

        let inner = node.unary_operand()?;

        for child in egraph.nodes(inner) {
            if let Some(child_op) = child.op() {
                if child_op.name() == "add" {
                    if let Some((f, g)) = child.binary_operands() {
                        // D[f + g] → D[f] + D[g]
                        let df = ENode::Op {
                            op: self.diff_op,
                            children: vec![f],
                        };
                        let dg = ENode::Op {
                            op: self.diff_op,
                            children: vec![g],
                        };
                        return Some(RewriteAction::BinaryOp {
                            op: &ops::Add,
                            lhs_node: df,
                            rhs_node: dg,
                        });
                    }
                }
            }
        }
        None
    }
}

/// D[f - g] → D[f] - D[g]
pub struct DiffSub {
    diff_op: &'static dyn Op,
}

impl DiffSub {
    pub fn new(diff_op: &'static dyn Op) -> Box<Self> {
        Box::new(Self { diff_op })
    }
}

impl Rewrite for DiffSub {
    fn name(&self) -> &str { "diff-sub" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.name() != self.diff_op.name() { return None; }

        let inner = node.unary_operand()?;

        for child in egraph.nodes(inner) {
            if let Some(child_op) = child.op() {
                if child_op.name() == "sub" {
                    if let Some((f, g)) = child.binary_operands() {
                        let df = ENode::Op {
                            op: self.diff_op,
                            children: vec![f],
                        };
                        let dg = ENode::Op {
                            op: self.diff_op,
                            children: vec![g],
                        };
                        return Some(RewriteAction::BinaryOp {
                            op: &ops::Sub,
                            lhs_node: df,
                            rhs_node: dg,
                        });
                    }
                }
            }
        }
        None
    }
}

/// D[neg(f)] → neg(D[f])
pub struct DiffNeg {
    diff_op: &'static dyn Op,
}

impl DiffNeg {
    pub fn new(diff_op: &'static dyn Op) -> Box<Self> {
        Box::new(Self { diff_op })
    }
}

impl Rewrite for DiffNeg {
    fn name(&self) -> &str { "diff-neg" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.name() != self.diff_op.name() { return None; }

        let inner = node.unary_operand()?;

        for child in egraph.nodes(inner) {
            if let Some(child_op) = child.op() {
                if child_op.name() == "neg" {
                    if let Some(f) = child.unary_operand() {
                        // D[-f] → -D[f]
                        let df = ENode::Op {
                            op: self.diff_op,
                            children: vec![f],
                        };
                        return Some(RewriteAction::UnaryOp {
                            op: &ops::Neg,
                            inner_node: df,
                        });
                    }
                }
            }
        }
        None
    }
}

// ============================================================================
// Product and Quotient Rules
// ============================================================================

/// D[f * g] → D[f] * g + f * D[g]
pub struct DiffMul {
    diff_op: &'static dyn Op,
}

impl DiffMul {
    pub fn new(diff_op: &'static dyn Op) -> Box<Self> {
        Box::new(Self { diff_op })
    }
}

impl Rewrite for DiffMul {
    fn name(&self) -> &str { "diff-mul" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.name() != self.diff_op.name() { return None; }

        let inner = node.unary_operand()?;

        for child in egraph.nodes(inner) {
            if let Some(child_op) = child.op() {
                if child_op.name() == "mul" {
                    if let Some((f, g)) = child.binary_operands() {
                        // D[f * g] → D[f] * g + f * D[g]
                        return Some(RewriteAction::ProductRule {
                            diff_op: self.diff_op,
                            f,
                            g,
                        });
                    }
                }
            }
        }
        None
    }
}

/// D[f / g] → (D[f] * g - f * D[g]) / (g * g)
pub struct DiffDiv {
    diff_op: &'static dyn Op,
}

impl DiffDiv {
    pub fn new(diff_op: &'static dyn Op) -> Box<Self> {
        Box::new(Self { diff_op })
    }
}

impl Rewrite for DiffDiv {
    fn name(&self) -> &str { "diff-div" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.name() != self.diff_op.name() { return None; }

        let inner = node.unary_operand()?;

        for child in egraph.nodes(inner) {
            if let Some(child_op) = child.op() {
                if child_op.name() == "div" {
                    if let Some((f, g)) = child.binary_operands() {
                        // D[f / g] → (D[f] * g - f * D[g]) / (g * g)
                        return Some(RewriteAction::QuotientRule {
                            diff_op: self.diff_op,
                            f,
                            g,
                        });
                    }
                }
            }
        }
        None
    }
}

// ============================================================================
// Chain Rule for Transcendentals
// ============================================================================

/// D[sqrt(f)] → D[f] / (2 * sqrt(f))
pub struct DiffSqrt {
    diff_op: &'static dyn Op,
}

impl DiffSqrt {
    pub fn new(diff_op: &'static dyn Op) -> Box<Self> {
        Box::new(Self { diff_op })
    }
}

impl Rewrite for DiffSqrt {
    fn name(&self) -> &str { "diff-sqrt" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.name() != self.diff_op.name() { return None; }

        let inner = node.unary_operand()?;

        for child in egraph.nodes(inner) {
            if let Some(child_op) = child.op() {
                if child_op.name() == "sqrt" {
                    if let Some(f) = child.unary_operand() {
                        // D[sqrt(f)] → D[f] / (2 * sqrt(f))
                        return Some(RewriteAction::ChainSqrt {
                            diff_op: self.diff_op,
                            f,
                        });
                    }
                }
            }
        }
        None
    }
}

/// D[sin(f)] → cos(f) * D[f]
pub struct DiffSin {
    diff_op: &'static dyn Op,
}

impl DiffSin {
    pub fn new(diff_op: &'static dyn Op) -> Box<Self> {
        Box::new(Self { diff_op })
    }
}

impl Rewrite for DiffSin {
    fn name(&self) -> &str { "diff-sin" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.name() != self.diff_op.name() { return None; }

        let inner = node.unary_operand()?;

        for child in egraph.nodes(inner) {
            if let Some(child_op) = child.op() {
                if child_op.name() == "sin" {
                    if let Some(f) = child.unary_operand() {
                        // D[sin(f)] → cos(f) * D[f]
                        return Some(RewriteAction::ChainSin {
                            diff_op: self.diff_op,
                            f,
                        });
                    }
                }
            }
        }
        None
    }
}

/// D[cos(f)] → -sin(f) * D[f]
pub struct DiffCos {
    diff_op: &'static dyn Op,
}

impl DiffCos {
    pub fn new(diff_op: &'static dyn Op) -> Box<Self> {
        Box::new(Self { diff_op })
    }
}

impl Rewrite for DiffCos {
    fn name(&self) -> &str { "diff-cos" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.name() != self.diff_op.name() { return None; }

        let inner = node.unary_operand()?;

        for child in egraph.nodes(inner) {
            if let Some(child_op) = child.op() {
                if child_op.name() == "cos" {
                    if let Some(f) = child.unary_operand() {
                        // D[cos(f)] → -sin(f) * D[f]
                        return Some(RewriteAction::ChainCos {
                            diff_op: self.diff_op,
                            f,
                        });
                    }
                }
            }
        }
        None
    }
}

/// D[exp(f)] → exp(f) * D[f]
pub struct DiffExp {
    diff_op: &'static dyn Op,
}

impl DiffExp {
    pub fn new(diff_op: &'static dyn Op) -> Box<Self> {
        Box::new(Self { diff_op })
    }
}

impl Rewrite for DiffExp {
    fn name(&self) -> &str { "diff-exp" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.name() != self.diff_op.name() { return None; }

        let inner = node.unary_operand()?;

        for child in egraph.nodes(inner) {
            if let Some(child_op) = child.op() {
                if child_op.name() == "exp" {
                    if let Some(f) = child.unary_operand() {
                        // D[exp(f)] → exp(f) * D[f]
                        return Some(RewriteAction::ChainExp {
                            diff_op: self.diff_op,
                            f,
                        });
                    }
                }
            }
        }
        None
    }
}

/// D[ln(f)] → D[f] / f
pub struct DiffLn {
    diff_op: &'static dyn Op,
}

impl DiffLn {
    pub fn new(diff_op: &'static dyn Op) -> Box<Self> {
        Box::new(Self { diff_op })
    }
}

impl Rewrite for DiffLn {
    fn name(&self) -> &str { "diff-ln" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.name() != self.diff_op.name() { return None; }

        let inner = node.unary_operand()?;

        for child in egraph.nodes(inner) {
            if let Some(child_op) = child.op() {
                if child_op.name() == "ln" {
                    if let Some(f) = child.unary_operand() {
                        // D[ln(f)] → D[f] / f
                        return Some(RewriteAction::ChainLn {
                            diff_op: self.diff_op,
                            f,
                        });
                    }
                }
            }
        }
        None
    }
}

// ============================================================================
// Coordinate Variable Rules
// ============================================================================

/// D_X[X] → 1, D_X[Y] → 0, D_X[Z] → 0
pub struct DiffCoordinate {
    diff_op: &'static dyn Op,
    target_var: &'static dyn Op,
}

impl DiffCoordinate {
    pub fn new(diff_op: &'static dyn Op, target_var: &'static dyn Op) -> Box<Self> {
        Box::new(Self { diff_op, target_var })
    }
}

impl Rewrite for DiffCoordinate {
    fn name(&self) -> &str { "diff-coord" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.name() != self.diff_op.name() { return None; }

        let inner = node.unary_operand()?;

        for child in egraph.nodes(inner) {
            if let Some(child_op) = child.op() {
                // Check if this is a coordinate variable
                let child_name = child_op.name();
                if child_name == "X" || child_name == "Y" || child_name == "Z" {
                    // D_X[X] → 1, D_X[Y] → 0, etc.
                    let diff_name = self.diff_op.name();
                    let is_same = match (diff_name, child_name) {
                        ("D_X", "X") => true,
                        ("D_Y", "Y") => true,
                        ("D_Z", "Z") => true,
                        _ => false,
                    };
                    let result = if is_same { 1.0 } else { 0.0 };
                    return Some(RewriteAction::Create(ENode::constant(result)));
                }
            }
        }
        None
    }
}

/// D[constant] → 0
pub struct DiffConstant {
    diff_op: &'static dyn Op,
}

impl DiffConstant {
    pub fn new(diff_op: &'static dyn Op) -> Box<Self> {
        Box::new(Self { diff_op })
    }
}

impl Rewrite for DiffConstant {
    fn name(&self) -> &str { "diff-const" }

    fn apply(&self, egraph: &EGraph, _id: EClassId, node: &ENode) -> Option<RewriteAction> {
        let node_op = node.op()?;
        if node_op.name() != self.diff_op.name() { return None; }

        let inner = node.unary_operand()?;

        // Check if inner is a constant
        if egraph.is_constant(inner) {
            return Some(RewriteAction::Create(ENode::constant(0.0)));
        }
        None
    }
}

// ============================================================================
// Rule Builder
// ============================================================================

/// Create all calculus rules for a given derivative operator.
pub fn calculus_rules(diff_op: &'static dyn Op) -> Vec<Box<dyn Rewrite>> {
    vec![
        // Linearity
        DiffAdd::new(diff_op),
        DiffSub::new(diff_op),
        DiffNeg::new(diff_op),

        // Product/Quotient
        DiffMul::new(diff_op),
        DiffDiv::new(diff_op),

        // Chain rule for transcendentals
        DiffSqrt::new(diff_op),
        DiffSin::new(diff_op),
        DiffCos::new(diff_op),
        DiffExp::new(diff_op),
        DiffLn::new(diff_op),

        // Coordinates
        DiffCoordinate::new(diff_op, &VarX),
        DiffCoordinate::new(diff_op, &VarY),
        DiffCoordinate::new(diff_op, &VarZ),
        DiffConstant::new(diff_op),
    ]
}

/// Create all calculus rules for X, Y, and Z differentiation.
pub fn all_calculus_rules() -> Vec<Box<dyn Rewrite>> {
    let mut rules = Vec::new();
    rules.extend(calculus_rules(&DiffX));
    rules.extend(calculus_rules(&DiffY));
    rules.extend(calculus_rules(&DiffZ));
    rules
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_operators_have_zero_cost() {
        assert_eq!(DiffX.default_cost(), 0);
        assert_eq!(DiffY.default_cost(), 0);
        assert_eq!(DiffZ.default_cost(), 0);
    }

    #[test]
    fn all_calculus_rules_creates_rules() {
        let rules = all_calculus_rules();
        // 14 rules per derivative × 3 derivatives = 42 rules
        // (Add, Sub, Neg, Mul, Div, Sqrt, Sin, Cos, Exp, Ln, CoordX, CoordY, CoordZ, Const)
        assert_eq!(rules.len(), 42);
    }
}
