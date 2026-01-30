//! Cost model builder using feature-based evaluation.
//!
//! This module generates better operation costs for the E-graph extractor
//! by evaluating canonical test expressions with feature-based cost estimation.
//!
//! ## Architecture
//!
//! - **pixelflow-nnue**: Core expression types (Expr, OpType)
//! - **pixelflow-search**: Generic E-graph with simple CostModel
//! - **This module**: Bridges the gap with feature-based cost estimation
//!
//! ## Why Not Use pixelflow-ml?
//!
//! pixelflow-ml depends on pixelflow-core (for graphics features), which creates
//! a dependency cycle: core -> macros -> ml -> core.
//!
//! Instead, we implement a lightweight Hand-Crafted Evaluator here using only
//! pixelflow-nnue types.

use pixelflow_nnue::{Expr, OpType};
use pixelflow_search::egraph::CostModel;
use std::collections::HashMap;

/// Hand-Crafted Evaluator weights (operation costs in approximate cycles).
///
/// Based on typical x86-64 SIMD latencies:
/// - Cheap: neg, abs (1 cycle)
/// - Medium: add, sub, mul, min, max (4-5 cycles)
/// - Expensive: div, sqrt (15-20 cycles)
/// - Fused: FMA (~5 cycles, same as mul alone!)
struct HceWeights {
    add: i32,
    sub: i32,
    mul: i32,
    div: i32,
    neg: i32,
    sqrt: i32,
    rsqrt: i32,
    abs: i32,
    min: i32,
    max: i32,
    fma: i32,
}

impl Default for HceWeights {
    fn default() -> Self {
        Self {
            add: 4,
            sub: 4,
            mul: 5,
            div: 15,
            neg: 1,
            sqrt: 15,
            rsqrt: 5,
            abs: 1,
            min: 4,
            max: 4,
            fma: 5, // Same cost as mul alone (modern CPUs)
        }
    }
}

/// Extract simplified features from an expression.
///
/// This is a lightweight version of pixelflow-ml's feature extractor,
/// avoiding the dependency cycle.
struct ExprFeatures {
    // Operation counts
    add_count: i32,
    sub_count: i32,
    mul_count: i32,
    div_count: i32,
    neg_count: i32,
    sqrt_count: i32,
    rsqrt_count: i32,
    abs_count: i32,
    min_count: i32,
    max_count: i32,
    fma_count: i32,

    // Structural
    critical_path: i32, // Longest dependency chain (ILP-aware!)
}

impl ExprFeatures {
    fn new() -> Self {
        Self {
            add_count: 0,
            sub_count: 0,
            mul_count: 0,
            div_count: 0,
            neg_count: 0,
            sqrt_count: 0,
            rsqrt_count: 0,
            abs_count: 0,
            min_count: 0,
            max_count: 0,
            fma_count: 0,
            critical_path: 0,
        }
    }

    /// Evaluate cost using HCE weights.
    fn evaluate(&self, weights: &HceWeights) -> i32 {
        // Linear combination of operation counts
        let op_cost =
            self.add_count * weights.add +
            self.sub_count * weights.sub +
            self.mul_count * weights.mul +
            self.div_count * weights.div +
            self.neg_count * weights.neg +
            self.sqrt_count * weights.sqrt +
            self.rsqrt_count * weights.rsqrt +
            self.abs_count * weights.abs +
            self.min_count * weights.min +
            self.max_count * weights.max +
            self.fma_count * weights.fma;

        // Critical path is the PRIMARY cost driver (captures ILP)
        // Operation counts are SECONDARY (captures instruction mix)
        self.critical_path + (op_cost / 10)
    }
}

/// Extract features from an expression.
///
/// Returns the critical path cost (longest dependency chain).
fn extract_features(expr: &Expr, features: &mut ExprFeatures) -> i32 {
    match expr {
        Expr::Var(_) | Expr::Const(_) => 0, // No latency

        Expr::Unary(op, a) => {
            let op_cost = match op {
                OpType::Neg => { features.neg_count += 1; 1 }
                OpType::Sqrt => { features.sqrt_count += 1; 15 }
                OpType::Rsqrt => { features.rsqrt_count += 1; 5 }
                OpType::Abs => { features.abs_count += 1; 1 }
                _ => 5,
            };
            let child_cost = extract_features(a, features);
            op_cost + child_cost
        }

        Expr::Binary(op, a, b) => {
            let op_cost = match op {
                OpType::Add => { features.add_count += 1; 4 }
                OpType::Sub => { features.sub_count += 1; 4 }
                OpType::Mul => { features.mul_count += 1; 5 }
                OpType::Div => { features.div_count += 1; 15 }
                OpType::Min => { features.min_count += 1; 4 }
                OpType::Max => { features.max_count += 1; 4 }
                OpType::MulRsqrt => { features.mul_count += 1; features.rsqrt_count += 1; 6 }
                _ => 5,
            };
            let cost_a = extract_features(a, features);
            let cost_b = extract_features(b, features);
            // Critical path = max of children (parallel execution) + this op
            op_cost + cost_a.max(cost_b)
        }

        Expr::Ternary(op, a, b, c) => {
            let op_cost = match op {
                OpType::MulAdd => { features.fma_count += 1; 5 }
                _ => 10,
            };
            let cost_a = extract_features(a, features);
            let cost_b = extract_features(b, features);
            let cost_c = extract_features(c, features);
            op_cost + cost_a.max(cost_b).max(cost_c)
        }
    }
}

/// Evaluate an expression with HCE, returning predicted cost.
fn evaluate_with_hce(expr: &Expr) -> i32 {
    let weights = HceWeights::default();
    let mut features = ExprFeatures::new();
    features.critical_path = extract_features(expr, &mut features);
    features.evaluate(&weights)
}

/// Build a cost model using Hand-Crafted Evaluator (HCE).
///
/// This evaluates canonical expressions for each operation and derives
/// relative costs.
pub fn build_cost_model_with_hce() -> CostModel {
    // Evaluate canonical expressions for each operation
    let mut costs = HashMap::new();

    // Unary operations: op(X)
    costs.insert("neg", evaluate_unary(OpType::Neg));
    costs.insert("sqrt", evaluate_unary(OpType::Sqrt));
    costs.insert("rsqrt", evaluate_unary(OpType::Rsqrt));
    costs.insert("abs", evaluate_unary(OpType::Abs));

    // Binary operations: X op Y
    costs.insert("add", evaluate_binary(OpType::Add));
    costs.insert("sub", evaluate_binary(OpType::Sub));
    costs.insert("mul", evaluate_binary(OpType::Mul));
    costs.insert("div", evaluate_binary(OpType::Div));
    costs.insert("min", evaluate_binary(OpType::Min));
    costs.insert("max", evaluate_binary(OpType::Max));

    // Ternary operations
    costs.insert("mul_add", evaluate_ternary(OpType::MulAdd));

    // Derived operations
    costs.insert("recip", costs["div"]);

    // Build CostModel from evaluated costs
    CostModel {
        add: costs["add"] as usize,
        sub: costs["sub"] as usize,
        mul: costs["mul"] as usize,
        div: costs["div"] as usize,
        neg: costs["neg"] as usize,
        sqrt: costs["sqrt"] as usize,
        recip: costs["recip"] as usize,
        rsqrt: costs["rsqrt"] as usize,
        abs: costs["abs"] as usize,
        min: costs["min"] as usize,
        max: costs["max"] as usize,
        mul_add: costs["mul_add"] as usize,
        depth_threshold: 32,
        depth_penalty: 100,
    }
}

/// Evaluate a unary operation: op(X)
fn evaluate_unary(op: OpType) -> i32 {
    let expr = Expr::Unary(
        op,
        Box::new(Expr::Var(0))
    );
    evaluate_with_hce(&expr)
}

/// Evaluate a binary operation: X op Y
fn evaluate_binary(op: OpType) -> i32 {
    let expr = Expr::Binary(
        op,
        Box::new(Expr::Var(0)),
        Box::new(Expr::Var(1))
    );
    evaluate_with_hce(&expr)
}

/// Evaluate a ternary operation: a * b + c
fn evaluate_ternary(op: OpType) -> i32 {
    let expr = Expr::Ternary(
        op,
        Box::new(Expr::Var(0)),
        Box::new(Expr::Var(1)),
        Box::new(Expr::Var(2))
    );
    evaluate_with_hce(&expr)
}

#[cfg(feature = "nnue-eval")]
/// Build cost model using full NNUE network (requires trained model).
///
/// This would use the full 401k-feature NNUE network from pixelflow-ml,
/// but that creates a dependency cycle. For now, fall back to HCE.
pub fn build_cost_model_with_nnue() -> CostModel {
    eprintln!("NNUE evaluation requires breaking dependency cycle");
    eprintln!("Falling back to HCE (ILP-aware, 11 features)");
    build_cost_model_with_hce()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hce_cost_model() {
        let costs = build_cost_model_with_hce();

        // Sanity checks: expensive ops should cost more
        assert!(costs.div > costs.add, "Division should be more expensive than addition");
        assert!(costs.sqrt > costs.add, "Sqrt should be more expensive than addition");
        assert!(costs.mul > costs.neg, "Multiply should be more expensive than negation");
    }

    #[test]
    fn test_critical_path_aware() {
        // Wide expression: (X + Y) + (Z + W)
        let wide = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Binary(
                OpType::Add,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Var(1))
            )),
            Box::new(Expr::Binary(
                OpType::Add,
                Box::new(Expr::Var(2)),
                Box::new(Expr::Var(3))
            ))
        );

        // Deep expression: (((X + Y) + Z) + W)
        let deep = Expr::Binary(
            OpType::Add,
            Box::new(Expr::Binary(
                OpType::Add,
                Box::new(Expr::Binary(
                    OpType::Add,
                    Box::new(Expr::Var(0)),
                    Box::new(Expr::Var(1))
                )),
                Box::new(Expr::Var(2))
            )),
            Box::new(Expr::Var(3))
        );

        let wide_cost = evaluate_with_hce(&wide);
        let deep_cost = evaluate_with_hce(&deep);

        // Wide should be cheaper (shorter critical path, more ILP)
        assert!(wide_cost < deep_cost,
            "Wide ({}) should be cheaper than deep ({}) due to ILP",
            wide_cost, deep_cost);
    }
}
