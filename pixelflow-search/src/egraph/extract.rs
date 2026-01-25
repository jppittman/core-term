//! Expression tree for extracted expressions.

use std::hash::{Hash, Hasher};
use super::cost::CostModel;

/// A simplified expression tree extracted from the e-graph.
///
/// This is the output of e-graph extraction. It can be:
/// - Evaluated (for testing)
/// - Analyzed for depth/complexity (for training)
/// - Converted back to AST (for code generation)
#[derive(Clone, Debug, PartialEq)]
pub enum ExprTree {
    Var(u8),
    Const(f32),
    Add(Box<ExprTree>, Box<ExprTree>),
    Sub(Box<ExprTree>, Box<ExprTree>),
    Mul(Box<ExprTree>, Box<ExprTree>),
    Div(Box<ExprTree>, Box<ExprTree>),
    Neg(Box<ExprTree>),
    Recip(Box<ExprTree>),
    Sqrt(Box<ExprTree>),
    Rsqrt(Box<ExprTree>),
    Abs(Box<ExprTree>),
    Min(Box<ExprTree>, Box<ExprTree>),
    Max(Box<ExprTree>, Box<ExprTree>),
    MulAdd(Box<ExprTree>, Box<ExprTree>, Box<ExprTree>),
    Floor(Box<ExprTree>),
    Ceil(Box<ExprTree>),
    Round(Box<ExprTree>),
    Fract(Box<ExprTree>),
    Sin(Box<ExprTree>),
    Cos(Box<ExprTree>),
    Tan(Box<ExprTree>),
    Asin(Box<ExprTree>),
    Acos(Box<ExprTree>),
    Atan(Box<ExprTree>),
    Exp(Box<ExprTree>),
    Exp2(Box<ExprTree>),
    Ln(Box<ExprTree>),
    Log2(Box<ExprTree>),
    Log10(Box<ExprTree>),
    Atan2(Box<ExprTree>, Box<ExprTree>),
    Pow(Box<ExprTree>, Box<ExprTree>),
    Hypot(Box<ExprTree>, Box<ExprTree>),
    Lt(Box<ExprTree>, Box<ExprTree>),
    Le(Box<ExprTree>, Box<ExprTree>),
    Gt(Box<ExprTree>, Box<ExprTree>),
    Ge(Box<ExprTree>, Box<ExprTree>),
    Eq(Box<ExprTree>, Box<ExprTree>),
    Ne(Box<ExprTree>, Box<ExprTree>),
    Select(Box<ExprTree>, Box<ExprTree>, Box<ExprTree>),
    Clamp(Box<ExprTree>, Box<ExprTree>, Box<ExprTree>),
    Tuple(Vec<ExprTree>),
}

impl ExprTree {
    /// Calculate the depth of this expression tree.
    ///
    /// Depth is the longest path from root to any leaf.
    /// Var and Const have depth 0. Unary ops add 1. Binary ops take max of children + 1.
    pub fn depth(&self) -> usize {
        match self {
            ExprTree::Var(_) | ExprTree::Const(_) => 0,

            // Unary operations
            ExprTree::Neg(a) | ExprTree::Recip(a) | ExprTree::Sqrt(a) | ExprTree::Rsqrt(a)
            | ExprTree::Abs(a) | ExprTree::Floor(a) | ExprTree::Ceil(a) | ExprTree::Round(a)
            | ExprTree::Fract(a) | ExprTree::Sin(a) | ExprTree::Cos(a) | ExprTree::Tan(a)
            | ExprTree::Asin(a) | ExprTree::Acos(a) | ExprTree::Atan(a) | ExprTree::Exp(a)
            | ExprTree::Exp2(a) | ExprTree::Ln(a) | ExprTree::Log2(a) | ExprTree::Log10(a) => {
                1 + a.depth()
            }

            // Binary operations
            ExprTree::Add(a, b) | ExprTree::Sub(a, b) | ExprTree::Mul(a, b) | ExprTree::Div(a, b)
            | ExprTree::Min(a, b) | ExprTree::Max(a, b) | ExprTree::Atan2(a, b)
            | ExprTree::Pow(a, b) | ExprTree::Hypot(a, b) | ExprTree::Lt(a, b)
            | ExprTree::Le(a, b) | ExprTree::Gt(a, b) | ExprTree::Ge(a, b)
            | ExprTree::Eq(a, b) | ExprTree::Ne(a, b) => {
                1 + a.depth().max(b.depth())
            }

            // Ternary operations
            ExprTree::MulAdd(a, b, c) | ExprTree::Select(a, b, c) | ExprTree::Clamp(a, b, c) => {
                1 + a.depth().max(b.depth()).max(c.depth())
            }

            ExprTree::Tuple(elems) => {
                elems.iter().map(|e| e.depth()).max().unwrap_or(0)
            }
        }
    }

    /// Count the total number of nodes in this expression tree.
    ///
    /// Each operation, variable, and constant counts as one node.
    pub fn node_count(&self) -> usize {
        match self {
            ExprTree::Var(_) | ExprTree::Const(_) => 1,

            // Unary operations
            ExprTree::Neg(a) | ExprTree::Recip(a) | ExprTree::Sqrt(a) | ExprTree::Rsqrt(a)
            | ExprTree::Abs(a) | ExprTree::Floor(a) | ExprTree::Ceil(a) | ExprTree::Round(a)
            | ExprTree::Fract(a) | ExprTree::Sin(a) | ExprTree::Cos(a) | ExprTree::Tan(a)
            | ExprTree::Asin(a) | ExprTree::Acos(a) | ExprTree::Atan(a) | ExprTree::Exp(a)
            | ExprTree::Exp2(a) | ExprTree::Ln(a) | ExprTree::Log2(a) | ExprTree::Log10(a) => {
                1 + a.node_count()
            }

            // Binary operations
            ExprTree::Add(a, b) | ExprTree::Sub(a, b) | ExprTree::Mul(a, b) | ExprTree::Div(a, b)
            | ExprTree::Min(a, b) | ExprTree::Max(a, b) | ExprTree::Atan2(a, b)
            | ExprTree::Pow(a, b) | ExprTree::Hypot(a, b) | ExprTree::Lt(a, b)
            | ExprTree::Le(a, b) | ExprTree::Gt(a, b) | ExprTree::Ge(a, b)
            | ExprTree::Eq(a, b) | ExprTree::Ne(a, b) => {
                1 + a.node_count() + b.node_count()
            }

            // Ternary operations
            ExprTree::MulAdd(a, b, c) | ExprTree::Select(a, b, c) | ExprTree::Clamp(a, b, c) => {
                1 + a.node_count() + b.node_count() + c.node_count()
            }

            ExprTree::Tuple(elems) => {
                1 + elems.iter().map(|e| e.node_count()).sum::<usize>()
            }
        }
    }

    /// Calculate the cost of this expression tree using the given cost model.
    pub fn cost(&self, costs: &CostModel) -> usize {
        match self {
            ExprTree::Var(_) | ExprTree::Const(_) => 0,

            // Unary operations
            ExprTree::Neg(a) => costs.neg + a.cost(costs),
            ExprTree::Recip(a) => costs.recip + a.cost(costs),
            ExprTree::Sqrt(a) => costs.sqrt + a.cost(costs),
            ExprTree::Rsqrt(a) => costs.rsqrt + a.cost(costs),
            ExprTree::Abs(a) => costs.abs + a.cost(costs),
            ExprTree::Floor(a) | ExprTree::Ceil(a) | ExprTree::Round(a)
            | ExprTree::Fract(a) | ExprTree::Sin(a) | ExprTree::Cos(a) | ExprTree::Tan(a)
            | ExprTree::Asin(a) | ExprTree::Acos(a) | ExprTree::Atan(a) | ExprTree::Exp(a)
            | ExprTree::Exp2(a) | ExprTree::Ln(a) | ExprTree::Log2(a) | ExprTree::Log10(a) => {
                // Use sqrt cost as default for transcendentals
                costs.sqrt + a.cost(costs)
            }

            // Binary operations
            ExprTree::Add(a, b) => costs.add + a.cost(costs) + b.cost(costs),
            ExprTree::Sub(a, b) => costs.sub + a.cost(costs) + b.cost(costs),
            ExprTree::Mul(a, b) => costs.mul + a.cost(costs) + b.cost(costs),
            ExprTree::Div(a, b) => costs.div + a.cost(costs) + b.cost(costs),
            ExprTree::Min(a, b) => costs.min + a.cost(costs) + b.cost(costs),
            ExprTree::Max(a, b) => costs.max + a.cost(costs) + b.cost(costs),
            ExprTree::Atan2(a, b) | ExprTree::Pow(a, b) | ExprTree::Hypot(a, b) => {
                // Use div cost as default for expensive binary ops
                costs.div + a.cost(costs) + b.cost(costs)
            }
            ExprTree::Lt(a, b) | ExprTree::Le(a, b) | ExprTree::Gt(a, b)
            | ExprTree::Ge(a, b) | ExprTree::Eq(a, b) | ExprTree::Ne(a, b) => {
                // Comparisons are cheap
                costs.add + a.cost(costs) + b.cost(costs)
            }

            // Ternary operations
            ExprTree::MulAdd(a, b, c) => costs.mul_add + a.cost(costs) + b.cost(costs) + c.cost(costs),
            ExprTree::Select(a, b, c) | ExprTree::Clamp(a, b, c) => {
                costs.add + a.cost(costs) + b.cost(costs) + c.cost(costs)
            }

            ExprTree::Tuple(elems) => {
                elems.iter().map(|e| e.cost(costs)).sum()
            }
        }
    }
}

// Manual Hash implementation because f32 doesn't implement Hash
impl Hash for ExprTree {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Use discriminant for variant identification
        core::mem::discriminant(self).hash(state);

        match self {
            ExprTree::Var(i) => i.hash(state),
            ExprTree::Const(c) => c.to_bits().hash(state),

            // Unary operations
            ExprTree::Neg(a) | ExprTree::Recip(a) | ExprTree::Sqrt(a) | ExprTree::Rsqrt(a)
            | ExprTree::Abs(a) | ExprTree::Floor(a) | ExprTree::Ceil(a) | ExprTree::Round(a)
            | ExprTree::Fract(a) | ExprTree::Sin(a) | ExprTree::Cos(a) | ExprTree::Tan(a)
            | ExprTree::Asin(a) | ExprTree::Acos(a) | ExprTree::Atan(a) | ExprTree::Exp(a)
            | ExprTree::Exp2(a) | ExprTree::Ln(a) | ExprTree::Log2(a) | ExprTree::Log10(a) => {
                a.hash(state);
            }

            // Binary operations
            ExprTree::Add(a, b) | ExprTree::Sub(a, b) | ExprTree::Mul(a, b) | ExprTree::Div(a, b)
            | ExprTree::Min(a, b) | ExprTree::Max(a, b) | ExprTree::Atan2(a, b)
            | ExprTree::Pow(a, b) | ExprTree::Hypot(a, b) | ExprTree::Lt(a, b)
            | ExprTree::Le(a, b) | ExprTree::Gt(a, b) | ExprTree::Ge(a, b)
            | ExprTree::Eq(a, b) | ExprTree::Ne(a, b) => {
                a.hash(state);
                b.hash(state);
            }

            // Ternary operations
            ExprTree::MulAdd(a, b, c) | ExprTree::Select(a, b, c) | ExprTree::Clamp(a, b, c) => {
                a.hash(state);
                b.hash(state);
                c.hash(state);
            }

            ExprTree::Tuple(elems) => {
                elems.hash(state);
            }
        }
    }
}

#[cfg(test)]
impl ExprTree {
    pub fn eval(&self, vars: &[f32; 4]) -> f32 {
        match self {
            ExprTree::Var(i) => vars[*i as usize],
            ExprTree::Const(c) => *c,
            ExprTree::Add(a, b) => a.eval(vars) + b.eval(vars),
            ExprTree::Sub(a, b) => a.eval(vars) - b.eval(vars),
            ExprTree::Mul(a, b) => a.eval(vars) * b.eval(vars),
            ExprTree::Div(a, b) => a.eval(vars) / b.eval(vars),
            ExprTree::Neg(a) => -a.eval(vars),
            ExprTree::Recip(a) => 1.0 / a.eval(vars),
            ExprTree::Sqrt(a) => a.eval(vars).sqrt(),
            ExprTree::Rsqrt(a) => 1.0 / a.eval(vars).sqrt(),
            ExprTree::Abs(a) => a.eval(vars).abs(),
            ExprTree::Min(a, b) => a.eval(vars).min(b.eval(vars)),
            ExprTree::Max(a, b) => a.eval(vars).max(b.eval(vars)),
            ExprTree::MulAdd(a, b, c) => a.eval(vars) * b.eval(vars) + c.eval(vars),
            _ => todo!("Implement eval for more variants if needed"),
        }
    }
}
