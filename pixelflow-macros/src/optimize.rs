//! # AST Optimization
//!
//! Performs algebraic simplification and constant folding on the AST.
//!
//! ## Optimizations
//!
//! 1. **Constant Folding**: `1.0 + 2.0` → `3.0`
//! 2. **Identity Removal**: `x + 0.0` → `x`, `x * 1.0` → `x`
//! 3. **Zero Propagation**: `x * 0.0` → `0.0`
//! 4. **E-Graph Saturation**: Discovers all equivalent forms via rewrite rules
//! 5. **FMA Fusion**: `a * b + c` → `MulAdd(a, b, c)` when profitable
//! 6. **Rsqrt Fusion**: `x / sqrt(y)` → `MulRsqrt(x, y)` when profitable
//!
//! These optimizations reduce the size of the generated expression tree,
//! which reduces compile time and runtime overhead (fewer function calls).

use crate::ast::{
    BinaryExpr, BinaryOp, BlockExpr, Expr, IdentExpr, LiteralExpr, MethodCallExpr, Stmt,
    UnaryExpr, UnaryOp,
};
use crate::egraph::{CostModel, EClassId, EGraph, ENode, ExprTree};
use crate::sema::AnalyzedKernel;
use proc_macro2::Span;
use std::collections::HashMap;
use syn::{Ident, Lit};

/// Optimize an analyzed kernel using tree rewriting.
///
/// This is the basic optimizer that applies local rewrites.
/// For more powerful optimization, use [`optimize_with_egraph`].
pub fn optimize(mut analyzed: AnalyzedKernel) -> AnalyzedKernel {
    analyzed.def.body = optimize_expr(analyzed.def.body);
    analyzed
}

/// Optimize an analyzed kernel using e-graph equality saturation.
///
/// This optimizer discovers all equivalent forms of an expression and
/// extracts the optimal one based on the provided cost model. It enables
/// optimizations like FMA fusion that span multiple operations.
///
/// # Cost Model
///
/// The cost model determines which equivalent form is "best". Use:
/// - `CostModel::default()` for CPUs without FMA
/// - `CostModel::with_fma()` for CPUs with FMA (Haswell+, M1+)
/// - `CostModel::fully_optimized()` for modern CPUs with FMA and fast rsqrt
/// - `CostModel::from_map()` for custom costs from build.rs detection
pub fn optimize_with_egraph(mut analyzed: AnalyzedKernel, costs: &CostModel) -> AnalyzedKernel {
    analyzed.def.body = optimize_expr_with_egraph(analyzed.def.body, costs);
    analyzed
}

/// Optimize a single expression using e-graph saturation.
fn optimize_expr_with_egraph(expr: Expr, costs: &CostModel) -> Expr {
    let mut ctx = EGraphContext::new();

    // Convert AST to e-graph, getting the root e-class
    let root = ctx.expr_to_egraph(&expr);

    // Run equality saturation to discover all equivalent forms
    ctx.egraph.saturate();

    // Extract optimal expression tree
    let tree = ctx.egraph.extract_tree_with_costs(root, costs);

    // Convert back to AST
    ctx.tree_to_expr(&tree)
}

// ============================================================================
// E-Graph Integration
// ============================================================================

/// Context for converting between AST and e-graph representations.
struct EGraphContext {
    /// The e-graph being built.
    egraph: EGraph,
    /// Map from variable name to e-class ID.
    var_to_eclass: HashMap<String, EClassId>,
    /// Map from variable index to name (for extraction).
    idx_to_name: Vec<String>,
}

impl EGraphContext {
    fn new() -> Self {
        Self {
            egraph: EGraph::new(),
            var_to_eclass: HashMap::new(),
            idx_to_name: Vec::new(),
        }
    }

    /// Get or create an e-class for a variable.
    fn get_or_create_var(&mut self, name: &str) -> EClassId {
        if let Some(&id) = self.var_to_eclass.get(name) {
            return id;
        }

        // Assign next index
        let idx = self.idx_to_name.len() as u8;
        self.idx_to_name.push(name.to_string());

        let id = self.egraph.add(ENode::Var(idx));
        self.var_to_eclass.insert(name.to_string(), id);
        id
    }

    /// Convert an AST expression to an e-graph, returning the root e-class.
    fn expr_to_egraph(&mut self, expr: &Expr) -> EClassId {
        match expr {
            Expr::Ident(ident) => self.get_or_create_var(&ident.name.to_string()),

            Expr::Literal(lit) => {
                if let Some(val) = extract_f64_from_lit(&lit.lit) {
                    self.egraph.add(ENode::constant(val as f32))
                } else {
                    // Non-numeric literal - treat as a unique variable
                    let name = format!("__lit_{:?}", lit.span);
                    self.get_or_create_var(&name)
                }
            }

            Expr::Binary(binary) => {
                let lhs = self.expr_to_egraph(&binary.lhs);
                let rhs = self.expr_to_egraph(&binary.rhs);

                let node = match binary.op {
                    BinaryOp::Add => ENode::Add(lhs, rhs),
                    BinaryOp::Sub => ENode::Sub(lhs, rhs),
                    BinaryOp::Mul => ENode::Mul(lhs, rhs),
                    BinaryOp::Div => ENode::Div(lhs, rhs),
                    // For other ops, we can't optimize them - treat as opaque
                    _ => {
                        // Create a unique "opaque" variable for unsupported ops
                        let name = format!("__binop_{:?}", binary.span);
                        return self.get_or_create_var(&name);
                    }
                };
                self.egraph.add(node)
            }

            Expr::Unary(unary) => {
                let operand = self.expr_to_egraph(&unary.operand);

                let node = match unary.op {
                    UnaryOp::Neg => ENode::Neg(operand),
                    UnaryOp::Not => {
                        // Boolean not - treat as opaque
                        let name = format!("__not_{:?}", unary.span);
                        return self.get_or_create_var(&name);
                    }
                };
                self.egraph.add(node)
            }

            Expr::MethodCall(call) => {
                let method = call.method.to_string();
                let receiver = self.expr_to_egraph(&call.receiver);

                match method.as_str() {
                    "sqrt" => self.egraph.add(ENode::Sqrt(receiver)),
                    "abs" => self.egraph.add(ENode::Abs(receiver)),
                    "min" if call.args.len() == 1 => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Min(receiver, arg))
                    }
                    "max" if call.args.len() == 1 => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Max(receiver, arg))
                    }
                    "mul_add" if call.args.len() == 2 => {
                        // receiver.mul_add(b, c) = receiver * b + c
                        let b = self.expr_to_egraph(&call.args[0]);
                        let c = self.expr_to_egraph(&call.args[1]);
                        self.egraph.add(ENode::MulAdd(receiver, b, c))
                    }
                    _ => {
                        // Unknown method - treat as opaque
                        let name = format!("__method_{}_{:?}", method, call.span);
                        self.get_or_create_var(&name)
                    }
                }
            }

            Expr::Paren(inner) => self.expr_to_egraph(inner),

            Expr::Block(block) => {
                // For blocks with let bindings, add bindings to var map
                for stmt in &block.stmts {
                    if let Stmt::Let(let_stmt) = stmt {
                        let init_id = self.expr_to_egraph(&let_stmt.init);
                        self.var_to_eclass
                            .insert(let_stmt.name.to_string(), init_id);
                    }
                }

                // Optimize the final expression
                if let Some(expr) = &block.expr {
                    self.expr_to_egraph(expr)
                } else {
                    // Empty block - return zero
                    self.egraph.add(ENode::constant(0.0))
                }
            }

            // For Call and Verbatim, treat as opaque
            Expr::Call(call) => {
                let name = format!("__call_{}_{:?}", call.func, call.span);
                self.get_or_create_var(&name)
            }

            Expr::Verbatim(_) => {
                let name = "__verbatim".to_string();
                self.get_or_create_var(&name)
            }
        }
    }

    /// Convert an extracted expression tree back to an AST expression.
    fn tree_to_expr(&self, tree: &ExprTree) -> Expr {
        let span = Span::call_site();

        match tree {
            ExprTree::Var(idx) => {
                let name = self
                    .idx_to_name
                    .get(*idx as usize)
                    .cloned()
                    .unwrap_or_else(|| format!("__var{}", idx));
                Expr::Ident(IdentExpr {
                    name: Ident::new(&name, span),
                    span,
                })
            }

            ExprTree::Const(val) => make_literal(*val as f64, span),

            ExprTree::Add(a, b) => Expr::Binary(BinaryExpr {
                op: BinaryOp::Add,
                lhs: Box::new(self.tree_to_expr(a)),
                rhs: Box::new(self.tree_to_expr(b)),
                span,
            }),

            ExprTree::Sub(a, b) => Expr::Binary(BinaryExpr {
                op: BinaryOp::Sub,
                lhs: Box::new(self.tree_to_expr(a)),
                rhs: Box::new(self.tree_to_expr(b)),
                span,
            }),

            ExprTree::Mul(a, b) => Expr::Binary(BinaryExpr {
                op: BinaryOp::Mul,
                lhs: Box::new(self.tree_to_expr(a)),
                rhs: Box::new(self.tree_to_expr(b)),
                span,
            }),

            ExprTree::Div(a, b) => Expr::Binary(BinaryExpr {
                op: BinaryOp::Div,
                lhs: Box::new(self.tree_to_expr(a)),
                rhs: Box::new(self.tree_to_expr(b)),
                span,
            }),

            ExprTree::Neg(a) => Expr::Unary(UnaryExpr {
                op: UnaryOp::Neg,
                operand: Box::new(self.tree_to_expr(a)),
                span,
            }),

            ExprTree::Sqrt(a) => Expr::MethodCall(MethodCallExpr {
                receiver: Box::new(self.tree_to_expr(a)),
                method: Ident::new("sqrt", span),
                args: vec![],
                span,
            }),

            ExprTree::Rsqrt(a) => {
                // rsqrt(x) = 1.0 / sqrt(x), but we emit it as a method call
                // since pixelflow-core has rsqrt support
                Expr::MethodCall(MethodCallExpr {
                    receiver: Box::new(self.tree_to_expr(a)),
                    method: Ident::new("rsqrt", span),
                    args: vec![],
                    span,
                })
            }

            ExprTree::Abs(a) => Expr::MethodCall(MethodCallExpr {
                receiver: Box::new(self.tree_to_expr(a)),
                method: Ident::new("abs", span),
                args: vec![],
                span,
            }),

            ExprTree::Min(a, b) => Expr::MethodCall(MethodCallExpr {
                receiver: Box::new(self.tree_to_expr(a)),
                method: Ident::new("min", span),
                args: vec![self.tree_to_expr(b)],
                span,
            }),

            ExprTree::Max(a, b) => Expr::MethodCall(MethodCallExpr {
                receiver: Box::new(self.tree_to_expr(a)),
                method: Ident::new("max", span),
                args: vec![self.tree_to_expr(b)],
                span,
            }),

            ExprTree::MulAdd(a, b, c) => {
                // a.mul_add(b, c) = a * b + c
                Expr::MethodCall(MethodCallExpr {
                    receiver: Box::new(self.tree_to_expr(a)),
                    method: Ident::new("mul_add", span),
                    args: vec![self.tree_to_expr(b), self.tree_to_expr(c)],
                    span,
                })
            }

            ExprTree::MulRsqrt(a, b) => {
                // a * rsqrt(b) - emit as method call for fused operation
                Expr::MethodCall(MethodCallExpr {
                    receiver: Box::new(self.tree_to_expr(a)),
                    method: Ident::new("mul_rsqrt", span),
                    args: vec![self.tree_to_expr(b)],
                    span,
                })
            }
        }
    }
}

/// Extract f64 from a syn::Lit.
fn extract_f64_from_lit(lit: &Lit) -> Option<f64> {
    match lit {
        Lit::Float(f) => f.base10_parse::<f64>().ok(),
        Lit::Int(i) => i.base10_parse::<f64>().ok(),
        _ => None,
    }
}

fn optimize_expr(expr: Expr) -> Expr {
    match expr {
        Expr::Binary(binary) => optimize_binary(binary),
        Expr::Unary(unary) => optimize_unary(unary),
        Expr::Paren(inner) => Expr::Paren(Box::new(optimize_expr(*inner))),
        Expr::Block(block) => optimize_block(block),
        // Recursively optimize method call arguments and receiver
        Expr::MethodCall(mut call) => {
            call.receiver = Box::new(optimize_expr(*call.receiver));
            call.args = call.args.into_iter().map(optimize_expr).collect();
            Expr::MethodCall(call)
        }
        _ => expr,
    }
}

fn optimize_binary(mut binary: BinaryExpr) -> Expr {
    // 1. Optimize operands first
    binary.lhs = Box::new(optimize_expr(*binary.lhs));
    binary.rhs = Box::new(optimize_expr(*binary.rhs));

    // 2. Try constant folding
    if let (Some(lhs_val), Some(rhs_val)) = (extract_f64(&binary.lhs), extract_f64(&binary.rhs)) {
        if let Some(result) = fold_constants(binary.op, lhs_val, rhs_val) {
            return make_literal(result, binary.span);
        }
    }

    // 3. Try algebraic simplification
    if let Some(simplified) = simplify_algebraic(&binary) {
        return simplified;
    }

    Expr::Binary(binary)
}

fn optimize_unary(mut unary: UnaryExpr) -> Expr {
    unary.operand = Box::new(optimize_expr(*unary.operand));

    if let Some(val) = extract_f64(&unary.operand) {
        if let Some(result) = fold_unary(unary.op, val) {
            return make_literal(result, unary.span);
        }
    }

    Expr::Unary(unary)
}

fn optimize_block(mut block: BlockExpr) -> Expr {
    // Optimize statements
    for stmt in &mut block.stmts {
        if let Stmt::Let(let_stmt) = stmt {
            let_stmt.init = optimize_expr(std::mem::replace(
                &mut let_stmt.init,
                make_literal(0.0, Span::call_site()), // Dummy placeholder
            ));
        } else if let Stmt::Expr(expr) = stmt {
            *expr = optimize_expr(std::mem::replace(
                expr,
                make_literal(0.0, Span::call_site()), // Dummy placeholder
            ));
        }
    }

    // Optimize final expression
    if let Some(expr) = block.expr {
        block.expr = Some(Box::new(optimize_expr(*expr)));
    }

    Expr::Block(block)
}

// --- Helpers ---

fn extract_f64(expr: &Expr) -> Option<f64> {
    if let Expr::Literal(lit_expr) = expr {
        match &lit_expr.lit {
            Lit::Float(f) => f.base10_parse::<f64>().ok(),
            Lit::Int(i) => i.base10_parse::<f64>().ok(),
            _ => None,
        }
    } else {
        None
    }
}

fn make_literal(val: f64, span: Span) -> Expr {
    let mut s = val.to_string();
    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        s.push_str(".0");
    }
    let lit = syn::LitFloat::new(&s, span);
    Expr::Literal(LiteralExpr {
        lit: Lit::Float(lit),
        span,
    })
}

fn fold_constants(op: BinaryOp, lhs: f64, rhs: f64) -> Option<f64> {
    match op {
        BinaryOp::Add => Some(lhs + rhs),
        BinaryOp::Sub => Some(lhs - rhs),
        BinaryOp::Mul => Some(lhs * rhs),
        BinaryOp::Div => Some(lhs / rhs),
        BinaryOp::Rem => Some(lhs % rhs),
        _ => None, // Comparisons etc. not folded to float (return bool)
    }
}

fn fold_unary(op: UnaryOp, val: f64) -> Option<f64> {
    match op {
        UnaryOp::Neg => Some(-val),
        _ => None,
    }
}

fn simplify_algebraic(binary: &BinaryExpr) -> Option<Expr> {
    let lhs_val = extract_f64(&binary.lhs);
    let rhs_val = extract_f64(&binary.rhs);

    match binary.op {
        BinaryOp::Add => {
            // x + 0 = x
            if is_zero(rhs_val) {
                return Some(*binary.lhs.clone());
            }
            // 0 + x = x
            if is_zero(lhs_val) {
                return Some(*binary.rhs.clone());
            }
        }
        BinaryOp::Sub => {
            // x - 0 = x
            if is_zero(rhs_val) {
                return Some(*binary.lhs.clone());
            }
        }
        BinaryOp::Mul => {
            // x * 1 = x
            if is_one(rhs_val) {
                return Some(*binary.lhs.clone());
            }
            // 1 * x = x
            if is_one(lhs_val) {
                return Some(*binary.rhs.clone());
            }
            // x * 0 = 0
            if is_zero(rhs_val) {
                return Some(make_literal(0.0, binary.span));
            }
            // 0 * x = 0
            if is_zero(lhs_val) {
                return Some(make_literal(0.0, binary.span));
            }
        }
        BinaryOp::Div => {
            // x / 1 = x
            if is_one(rhs_val) {
                return Some(*binary.lhs.clone());
            }
            // 0 / x = 0
            if is_zero(lhs_val) {
                return Some(make_literal(0.0, binary.span));
            }
        }
        _ => {}
    }

    None
}

fn is_zero(val: Option<f64>) -> bool {
    matches!(val, Some(v) if v.abs() < f64::EPSILON)
}

fn is_one(val: Option<f64>) -> bool {
    matches!(val, Some(v) if (v - 1.0).abs() < f64::EPSILON)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use crate::sema::analyze;
    use quote::quote;

    fn optimize_code(input: proc_macro2::TokenStream) -> String {
        let kernel = parse(input).unwrap();
        let analyzed = analyze(kernel).unwrap();
        let optimized = optimize(analyzed);
        format!("{:?}", optimized.def.body)
    }

    fn optimize_code_egraph(input: proc_macro2::TokenStream, costs: &CostModel) -> String {
        let kernel = parse(input).unwrap();
        let analyzed = analyze(kernel).unwrap();
        let optimized = optimize_with_egraph(analyzed, costs);
        format!("{:?}", optimized.def.body)
    }

    #[test]
    fn test_constant_folding() {
        let input = quote! { |x: f32| x + (1.0 + 2.0) };
        let debug = optimize_code(input);
        assert!(debug.contains("LiteralExpr"));
        assert!(debug.contains("3.0"));
        assert!(!debug.contains("1.0"));
        assert!(!debug.contains("2.0"));
    }

    #[test]
    fn test_identity_add() {
        let input = quote! { |x: f32| x + 0.0 };
        let debug = optimize_code(input);
        assert!(debug.contains("IdentExpr"));
        assert!(debug.contains("x"));
        assert!(!debug.contains("BinaryExpr"));
    }

    #[test]
    fn test_zero_mul() {
        let input = quote! { |x: f32| x * 0.0 };
        let debug = optimize_code(input);
        assert!(debug.contains("LiteralExpr"));
        assert!(debug.contains("0.0"));
        assert!(!debug.contains("IdentExpr"));
    }

    #[test]
    fn test_complex_folding() {
        let input = quote! { |x: f32| (1.0 + 2.0) * x + 0.0 };
        let debug = optimize_code(input);
        assert!(debug.contains("3.0"));
        assert!(debug.contains("x"));
    }

    // ========================================================================
    // E-Graph Integration Tests
    // ========================================================================

    #[test]
    fn test_egraph_identity_add() {
        let input = quote! { |x: f32| x + 0.0 };
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Should simplify to just x
        assert!(debug.contains("IdentExpr"));
        assert!(debug.contains("x"));
        assert!(!debug.contains("Add"));
    }

    #[test]
    fn test_egraph_identity_mul() {
        let input = quote! { |x: f32| x * 1.0 };
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Should simplify to just x
        assert!(debug.contains("IdentExpr"));
        assert!(debug.contains("x"));
        assert!(!debug.contains("Mul"));
    }

    #[test]
    fn test_egraph_zero_mul() {
        let input = quote! { |x: f32| x * 0.0 };
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Should simplify to 0.0
        assert!(debug.contains("LiteralExpr"));
        assert!(debug.contains("0.0") || debug.contains("0"));
    }

    #[test]
    fn test_egraph_sub_self() {
        let input = quote! { |x: f32| x - x };
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Should simplify to 0.0
        assert!(debug.contains("0"));
    }

    #[test]
    fn test_egraph_fma_fusion_with_fma_costs() {
        // a * b + c should become mul_add when FMA is cheap
        let input = quote! { |a: f32, b: f32, c: f32| a * b + c };
        let debug = optimize_code_egraph(input, &CostModel::with_fma());
        // With FMA costs, should extract mul_add
        assert!(debug.contains("mul_add"));
    }

    #[test]
    fn test_egraph_fma_unfused_without_fma_costs() {
        // a * b + c should stay as mul + add when FMA is expensive
        let input = quote! { |a: f32, b: f32, c: f32| a * b + c };
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Without FMA, should prefer unfused (mul(5) + add(4) = 9 < mul_add(10))
        // The expression should NOT contain mul_add
        assert!(!debug.contains("mul_add"));
    }

    #[test]
    fn test_egraph_complex_expression() {
        // ((x + 0) * 1 - x) should simplify to 0
        let input = quote! { |x: f32| (x + 0.0) * 1.0 - x };
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Should simplify to 0.0
        assert!(debug.contains("0"));
    }

    #[test]
    fn test_egraph_preserves_variables() {
        // Simple expression with named variables
        let input = quote! { |cx: f32, cy: f32| cx + cy };
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Should preserve variable names
        assert!(debug.contains("cx"));
        assert!(debug.contains("cy"));
    }

    #[test]
    fn test_egraph_handles_sqrt() {
        let input = quote! { |x: f32| x.sqrt() };
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Should preserve sqrt
        assert!(debug.contains("sqrt"));
    }

    #[test]
    fn test_egraph_div_sqrt_to_rsqrt() {
        // x / sqrt(y) should become x * rsqrt(y) then mul_rsqrt
        let input = quote! { |x: f32, y: f32| x / y.sqrt() };
        let debug = optimize_code_egraph(input, &CostModel::with_fast_rsqrt());
        // Should be converted to mul_rsqrt (fused form)
        assert!(debug.contains("mul_rsqrt") || debug.contains("rsqrt"));
    }

    #[test]
    fn test_egraph_double_negation() {
        let input = quote! { |x: f32| - -x };
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Should simplify to just x
        assert!(debug.contains("IdentExpr"));
        assert!(debug.contains("x"));
        assert!(!debug.contains("Neg"));
    }
}
