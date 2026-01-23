//! # AST Optimization
//!
//! Performs algebraic simplification and constant folding on the AST.
//!
//! ## Two-Pass Architecture
//!
//! **Pass 1 (Structural)**: Tree-based peephole optimization
//! - Constant folding: `1.0 + 2.0` → `3.0`
//! - Identity removal: `x + 0.0` → `x`, `x * 1.0` → `x`
//! - Zero propagation: `x * 0.0` → `0.0`
//!
//! **Pass 2 (Global)**: E-graph equality saturation
//! - Processes entire kernel expression globally (across let bindings)
//! - FMA fusion: `a * b + c` → `MulAdd(a, b, c)` when profitable
//! - Rsqrt: `1 / sqrt(y)` → `rsqrt(y)` (real instruction)
//! - Algebraic identities discovered via rewrite rules
//!
//! The global pass sees through let bindings, enabling optimizations like:
//! ```text
//! let a = X * X;
//! let b = Y * Y;
//! (a + b).sqrt()  // E-graph sees: sqrt(X*X + Y*Y)
//! ```

use crate::ast::{
    BinaryExpr, BinaryOp, BlockExpr, CallExpr, Expr, IdentExpr, LiteralExpr, MethodCallExpr, Stmt,
    UnaryExpr, UnaryOp,
};
use crate::sema::AnalyzedKernel;
use pixelflow_search::egraph::{CostModel, EClassId, EGraph, ENode, ExprTree};
use proc_macro2::Span;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use syn::{Ident, Lit};

/// Counter for generating unique opaque variable names.
static OPAQUE_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Generate a unique name for an opaque expression (unknown method call, etc.)
fn unique_opaque_name(prefix: &str) -> String {
    let id = OPAQUE_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("__{}{}", prefix, id)
}

/// Optimize an analyzed kernel using tree rewriting and e-graphs.
pub fn optimize(mut analyzed: AnalyzedKernel) -> AnalyzedKernel {
    // 1. Structural optimization (catches things inside opaque nodes)
    analyzed.def.body = optimize_expr(analyzed.def.body);

    // 2. E-Graph optimization (global rewriting & fusion)
    // We assume fully optimized costs for this project (AVX-512 target implied)
    optimize_with_egraph(analyzed, &CostModel::fully_optimized())
}

/// Optimize an analyzed kernel using e-graph equality saturation.
///
/// This optimizer discovers all equivalent forms of an expression and
/// extracts the optimal one based on the provided cost model. It enables
/// optimizations like FMA fusion that span multiple operations.
pub fn optimize_with_egraph(mut analyzed: AnalyzedKernel, costs: &CostModel) -> AnalyzedKernel {
    analyzed.def.body = optimize_expr_with_egraph(analyzed.def.body, costs);
    analyzed
}

/// Optimize a single expression using e-graph saturation.
///
/// This is the global optimization pass. It processes the entire expression
/// (including blocks with let bindings) as a single unit, enabling cross-binding
/// optimizations.
///
/// However, blocks containing opaque expressions that reference local variables
/// must preserve their structure, since the e-graph extraction would lose the
/// let bindings those locals depend on.
fn optimize_expr_with_egraph(expr: Expr, costs: &CostModel) -> Expr {
    match &expr {
        // Blocks with opaque expressions need special handling
        Expr::Block(block) => {
            // Check if the block has any opaque expressions (method calls on captured
            // manifolds, etc.) that might reference local let-bound variables.
            // If so, preserve block structure and optimize each part independently.
            if block_has_opaque_with_locals(block) {
                optimize_block_preserving_structure(block.clone(), costs)
            } else {
                // Pure arithmetic block - safe for global optimization
                optimize_via_egraph(&expr, costs)
            }
        }

        // Method calls: optimize receiver and args, but preserve method structure
        Expr::MethodCall(call) => {
            let mut optimized_call = call.clone();
            optimized_call.receiver = Box::new(optimize_expr_with_egraph(
                (*call.receiver).clone(),
                costs,
            ));
            optimized_call.args = call
                .args
                .iter()
                .map(|arg| optimize_expr_with_egraph(arg.clone(), costs))
                .collect();
            Expr::MethodCall(optimized_call)
        }

        // Pure expressions: use full egraph optimization
        _ => optimize_via_egraph(&expr, costs),
    }
}

/// Check if a block contains opaque expressions that reference local variables.
///
/// An "opaque expression" is one the e-graph can't optimize (like a method call
/// on a captured manifold). If such expressions reference let-bound locals,
/// we must preserve the block structure.
fn block_has_opaque_with_locals(block: &BlockExpr) -> bool {
    // Collect names of let-bound locals
    let local_names: std::collections::HashSet<String> = block
        .stmts
        .iter()
        .filter_map(|s| {
            if let Stmt::Let(let_stmt) = s {
                Some(let_stmt.name.to_string())
            } else {
                None
            }
        })
        .collect();

    if local_names.is_empty() {
        return false;
    }

    // Check if the final expression or any statement references locals in opaque contexts
    if let Some(ref final_expr) = block.expr {
        if expr_has_opaque_refs(final_expr, &local_names) {
            return true;
        }
    }

    // Also check statement expressions
    for stmt in &block.stmts {
        if let Stmt::Let(let_stmt) = stmt {
            if expr_has_opaque_refs(&let_stmt.init, &local_names) {
                return true;
            }
        }
    }

    false
}

/// Check if an expression has opaque sub-expressions that reference the given names.
fn expr_has_opaque_refs(expr: &Expr, local_names: &std::collections::HashSet<String>) -> bool {
    match expr {
        // Method calls on non-intrinsic receivers are opaque if they use locals
        Expr::MethodCall(call) => {
            // Check if the receiver is opaque (Verbatim) and args reference locals
            // This catches patterns like: ColorCube::default().at(red, green, blue, 1.0)
            // where ColorCube::default() is Verbatim and red/green/blue are locals
            if matches!(call.receiver.as_ref(), Expr::Verbatim(_)) {
                if call.args.iter().any(|arg| expr_references_any(arg, local_names)) {
                    return true;
                }
            }
            // Check if this is a method on a captured variable (not X, Y, Z, W)
            if let Expr::Ident(ident) = call.receiver.as_ref() {
                let name = ident.name.to_string();
                // If the receiver is a local or an external captured variable,
                // and args contain locals, this is problematic
                if !is_coordinate_intrinsic(&name) {
                    // Check if any arg references a local
                    if call.args.iter().any(|arg| expr_references_any(arg, local_names)) {
                        return true;
                    }
                }
            }
            // Recurse into receiver and args
            expr_has_opaque_refs(&call.receiver, local_names)
                || call.args.iter().any(|a| expr_has_opaque_refs(a, local_names))
        }

        // Function calls that aren't known intrinsics
        Expr::Call(call) => {
            let func_name = call.func.to_string();
            // V, DX, DY, DZ are known - others might be opaque
            if !["V", "DX", "DY", "DZ", "DXX", "DXY", "DYY"].contains(&func_name.as_str()) {
                if call.args.iter().any(|arg| expr_references_any(arg, local_names)) {
                    return true;
                }
            }
            call.args.iter().any(|a| expr_has_opaque_refs(a, local_names))
        }

        // Recurse into other expression types
        Expr::Binary(b) => {
            expr_has_opaque_refs(&b.lhs, local_names) || expr_has_opaque_refs(&b.rhs, local_names)
        }
        Expr::Unary(u) => expr_has_opaque_refs(&u.operand, local_names),
        Expr::Paren(p) => expr_has_opaque_refs(p, local_names),
        Expr::Tuple(t) => t.elems.iter().any(|e| expr_has_opaque_refs(e, local_names)),
        Expr::Block(b) => {
            b.stmts.iter().any(|s| {
                if let Stmt::Let(l) = s {
                    expr_has_opaque_refs(&l.init, local_names)
                } else {
                    false
                }
            }) || b.expr.as_ref().map_or(false, |e| expr_has_opaque_refs(e, local_names))
        }

        Expr::Ident(_) | Expr::Literal(_) => false,

        // Verbatim expressions wrap syn::Expr - check if they reference locals
        Expr::Verbatim(syn_expr) => syn_expr_references_any(syn_expr, local_names),
    }
}

/// Check if an expression references any of the given names.
fn expr_references_any(expr: &Expr, names: &std::collections::HashSet<String>) -> bool {
    match expr {
        Expr::Ident(i) => names.contains(&i.name.to_string()),
        Expr::Binary(b) => {
            expr_references_any(&b.lhs, names) || expr_references_any(&b.rhs, names)
        }
        Expr::Unary(u) => expr_references_any(&u.operand, names),
        Expr::MethodCall(c) => {
            expr_references_any(&c.receiver, names)
                || c.args.iter().any(|a| expr_references_any(a, names))
        }
        Expr::Call(c) => c.args.iter().any(|a| expr_references_any(a, names)),
        Expr::Paren(p) => expr_references_any(p, names),
        Expr::Tuple(t) => t.elems.iter().any(|e| expr_references_any(e, names)),
        Expr::Block(b) => {
            b.stmts.iter().any(|s| {
                if let Stmt::Let(l) = s {
                    expr_references_any(&l.init, names)
                } else {
                    false
                }
            }) || b.expr.as_ref().map_or(false, |e| expr_references_any(e, names))
        }
        Expr::Literal(_) => false,

        // Verbatim expressions wrap syn::Expr - check if they reference any names
        Expr::Verbatim(syn_expr) => syn_expr_references_any(syn_expr, names),
    }
}

/// Check if a syn::Expr references any of the given names.
///
/// This walks the syn::Expr tree looking for identifiers that match any of the names.
/// Used for checking Verbatim expressions that wrap raw syn::Expr values.
fn syn_expr_references_any(expr: &syn::Expr, names: &std::collections::HashSet<String>) -> bool {
    use syn::Expr as SynExpr;

    match expr {
        SynExpr::Path(path) => {
            // Simple identifier like `c_x`
            if let Some(ident) = path.path.get_ident() {
                names.contains(&ident.to_string())
            } else {
                // Qualified path like `Discrete::pack` - check segments
                path.path.segments.iter().any(|seg| names.contains(&seg.ident.to_string()))
            }
        }

        SynExpr::MethodCall(call) => {
            // Recursively check receiver and arguments
            syn_expr_references_any(&call.receiver, names)
                || call.args.iter().any(|arg| syn_expr_references_any(arg, names))
        }

        SynExpr::Call(call) => {
            // Check function and arguments
            syn_expr_references_any(&call.func, names)
                || call.args.iter().any(|arg| syn_expr_references_any(arg, names))
        }

        SynExpr::Binary(bin) => {
            syn_expr_references_any(&bin.left, names)
                || syn_expr_references_any(&bin.right, names)
        }

        SynExpr::Unary(un) => syn_expr_references_any(&un.expr, names),

        SynExpr::Paren(paren) => syn_expr_references_any(&paren.expr, names),

        SynExpr::Field(field) => syn_expr_references_any(&field.base, names),

        SynExpr::Index(index) => {
            syn_expr_references_any(&index.expr, names)
                || syn_expr_references_any(&index.index, names)
        }

        SynExpr::Cast(cast) => syn_expr_references_any(&cast.expr, names),

        SynExpr::Reference(reference) => syn_expr_references_any(&reference.expr, names),

        SynExpr::Tuple(tuple) => tuple.elems.iter().any(|e| syn_expr_references_any(e, names)),

        SynExpr::Array(array) => array.elems.iter().any(|e| syn_expr_references_any(e, names)),

        SynExpr::Block(block) => {
            block.block.stmts.iter().any(|stmt| {
                match stmt {
                    syn::Stmt::Local(local) => {
                        local.init.as_ref().map_or(false, |init| {
                            syn_expr_references_any(&init.expr, names)
                        })
                    }
                    syn::Stmt::Expr(expr, _) => syn_expr_references_any(expr, names),
                    _ => false,
                }
            })
        }

        SynExpr::If(if_expr) => {
            syn_expr_references_any(&if_expr.cond, names)
                || if_expr.then_branch.stmts.iter().any(|stmt| {
                    if let syn::Stmt::Expr(expr, _) = stmt {
                        syn_expr_references_any(expr, names)
                    } else {
                        false
                    }
                })
                || if_expr.else_branch.as_ref().map_or(false, |(_, else_expr)| {
                    syn_expr_references_any(else_expr, names)
                })
        }

        // Literals don't reference variables
        SynExpr::Lit(_) => false,

        // For other expression types, conservatively return true to preserve structure
        // (better to preserve than to accidentally lose bindings)
        _ => true,
    }
}

/// Check if a name is a coordinate intrinsic (X, Y, Z, W).
fn is_coordinate_intrinsic(name: &str) -> bool {
    matches!(name, "X" | "Y" | "Z" | "W")
}

/// Optimize a block while preserving its structure.
///
/// Each let binding and the final expression are optimized independently.
fn optimize_block_preserving_structure(mut block: BlockExpr, costs: &CostModel) -> Expr {
    for stmt in &mut block.stmts {
        if let Stmt::Let(let_stmt) = stmt {
            let init = std::mem::replace(
                &mut let_stmt.init,
                make_literal(0.0, Span::call_site()),
            );
            let_stmt.init = optimize_expr_with_egraph(init, costs);
        }
    }
    if let Some(final_expr) = block.expr.take() {
        block.expr = Some(Box::new(optimize_expr_with_egraph(*final_expr, costs)));
    }
    Expr::Block(block)
}

/// Optimize an expression via the e-graph.
fn optimize_via_egraph(expr: &Expr, costs: &CostModel) -> Expr {
    let mut ctx = EGraphContext::new();
    let root = ctx.expr_to_egraph(expr);
    ctx.egraph.saturate();
    let tree = ctx.egraph.extract_tree_with_costs(root, costs);
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
    /// Map from opaque variable names to their original expressions.
    /// Used to restore expressions that can't be represented in the e-graph.
    opaque_exprs: HashMap<String, Expr>,
}

impl EGraphContext {
    fn new() -> Self {
        Self {
            egraph: EGraph::new(),
            var_to_eclass: HashMap::new(),
            idx_to_name: Vec::new(),
            opaque_exprs: HashMap::new(),
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

    /// Create an opaque variable for an expression we can't optimize.
    /// The original expression is stored and will be restored during extraction.
    fn create_opaque_var(&mut self, prefix: &str, expr: &Expr) -> EClassId {
        let name = unique_opaque_name(prefix);
        self.opaque_exprs.insert(name.clone(), expr.clone());
        self.get_or_create_var(&name)
    }

    /// Check if a method is known and can be converted to ENode.
    fn is_known_method(method: &str, arg_count: usize) -> bool {
        match method {
            // Unary methods (0 args)
            "sqrt" | "rsqrt" | "recip" | "abs" | "neg"
            | "floor" | "ceil" | "round" | "fract"
            | "sin" | "cos" | "tan" | "asin" | "acos" | "atan"
            | "exp" | "exp2" | "ln" | "log2" | "log10" => arg_count == 0,

            // Binary methods (1 arg)
            "min" | "max" | "atan2" | "pow" | "hypot"
            | "lt" | "le" | "gt" | "ge" | "eq" | "ne" => arg_count == 1,

            // Ternary methods (2 args)
            "mul_add" | "select" | "clamp" => arg_count == 2,

            _ => false,
        }
    }

    /// Convert an AST expression to an e-graph, returning the root e-class.
    fn expr_to_egraph(&mut self, expr: &Expr) -> EClassId {
        match expr {
            Expr::Ident(ident) => self.get_or_create_var(&ident.name.to_string()),

            Expr::Literal(lit) => {
                if let Some(val) = extract_f64_from_lit(&lit.lit) {
                    self.egraph.add(ENode::constant(val as f32))
                } else {
                    // Non-numeric literal - preserve original
                    self.create_opaque_var("lit", expr)
                }
            }

            Expr::Binary(binary) => {
                // Check if this is a supported binary op BEFORE converting children
                // Unsupported ops are preserved as opaque expressions
                match binary.op {
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
                    | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge
                    | BinaryOp::Eq | BinaryOp::Ne => {
                        // Supported - convert children
                        let lhs = self.expr_to_egraph(&binary.lhs);
                        let rhs = self.expr_to_egraph(&binary.rhs);

                        let node = match binary.op {
                            BinaryOp::Add => ENode::Add(lhs, rhs),
                            BinaryOp::Sub => ENode::Sub(lhs, rhs),
                            BinaryOp::Mul => ENode::Mul(lhs, rhs),
                            BinaryOp::Div => ENode::Div(lhs, rhs),
                            BinaryOp::Lt => ENode::Lt(lhs, rhs),
                            BinaryOp::Le => ENode::Le(lhs, rhs),
                            BinaryOp::Gt => ENode::Gt(lhs, rhs),
                            BinaryOp::Ge => ENode::Ge(lhs, rhs),
                            BinaryOp::Eq => ENode::Eq(lhs, rhs),
                            BinaryOp::Ne => ENode::Ne(lhs, rhs),
                            _ => unreachable!(),
                        };
                        self.egraph.add(node)
                    }
                    // For other ops (Rem, And, Or, BitAnd, BitOr, BitXor, Shl, Shr)
                    // preserve as opaque expression with original structure
                    _ => self.create_opaque_var("binop", expr),
                }
            }

            Expr::Unary(unary) => {
                match unary.op {
                    UnaryOp::Neg => {
                        let operand = self.expr_to_egraph(&unary.operand);
                        self.egraph.add(ENode::Neg(operand))
                    }
                    UnaryOp::Not => {
                        // Boolean not - preserve original
                        self.create_opaque_var("not", expr)
                    }
                }
            }

            Expr::MethodCall(call) => {
                let method = call.method.to_string();

                // Check if this is a known method BEFORE converting children
                // Unknown methods preserve the original expression structure
                if !Self::is_known_method(&method, call.args.len()) {
                    return self.create_opaque_var("method", expr);
                }

                let receiver = self.expr_to_egraph(&call.receiver);

                match method.as_str() {
                    // === Unary methods ===
                    "sqrt" => self.egraph.add(ENode::Sqrt(receiver)),
                    "rsqrt" => self.egraph.add(ENode::Rsqrt(receiver)),
                    "recip" => self.egraph.add(ENode::Recip(receiver)),
                    "abs" => self.egraph.add(ENode::Abs(receiver)),
                    "neg" => self.egraph.add(ENode::Neg(receiver)),
                    "floor" => self.egraph.add(ENode::Floor(receiver)),
                    "ceil" => self.egraph.add(ENode::Ceil(receiver)),
                    "round" => self.egraph.add(ENode::Round(receiver)),
                    "fract" => self.egraph.add(ENode::Fract(receiver)),
                    "sin" => self.egraph.add(ENode::Sin(receiver)),
                    "cos" => self.egraph.add(ENode::Cos(receiver)),
                    "tan" => self.egraph.add(ENode::Tan(receiver)),
                    "asin" => self.egraph.add(ENode::Asin(receiver)),
                    "acos" => self.egraph.add(ENode::Acos(receiver)),
                    "atan" => self.egraph.add(ENode::Atan(receiver)),
                    "exp" => self.egraph.add(ENode::Exp(receiver)),
                    "exp2" => self.egraph.add(ENode::Exp2(receiver)),
                    "ln" => self.egraph.add(ENode::Ln(receiver)),
                    "log2" => self.egraph.add(ENode::Log2(receiver)),
                    "log10" => self.egraph.add(ENode::Log10(receiver)),

                    // === Binary methods ===
                    "min" => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Min(receiver, arg))
                    }
                    "max" => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Max(receiver, arg))
                    }
                    "atan2" => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Atan2(receiver, arg))
                    }
                    "pow" => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Pow(receiver, arg))
                    }
                    "hypot" => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Hypot(receiver, arg))
                    }

                    // === Comparison methods ===
                    "lt" => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Lt(receiver, arg))
                    }
                    "le" => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Le(receiver, arg))
                    }
                    "gt" => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Gt(receiver, arg))
                    }
                    "ge" => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Ge(receiver, arg))
                    }
                    "eq" => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Eq(receiver, arg))
                    }
                    "ne" => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        self.egraph.add(ENode::Ne(receiver, arg))
                    }

                    // === Ternary methods ===
                    "mul_add" => {
                        let b = self.expr_to_egraph(&call.args[0]);
                        let c = self.expr_to_egraph(&call.args[1]);
                        self.egraph.add(ENode::MulAdd(receiver, b, c))
                    }
                    "select" => {
                        let if_true = self.expr_to_egraph(&call.args[0]);
                        let if_false = self.expr_to_egraph(&call.args[1]);
                        self.egraph.add(ENode::Select(receiver, if_true, if_false))
                    }
                    "clamp" => {
                        let min_val = self.expr_to_egraph(&call.args[0]);
                        let max_val = self.expr_to_egraph(&call.args[1]);
                        self.egraph.add(ENode::Clamp(receiver, min_val, max_val))
                    }

                    // Should not reach here due to is_known_method check
                    _ => unreachable!("Unknown method {} should have been handled as opaque", method),
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
                let name = unique_opaque_name(&format!("call_{}_", call.func));
                self.get_or_create_var(&name)
            }

            Expr::Verbatim(_) => {
                // Store verbatim expressions as opaque so they can be restored during extraction
                self.create_opaque_var("verbatim_", expr)
            }

            Expr::Tuple(tuple) => {
                let elems = tuple.elems.iter().map(|e| self.expr_to_egraph(e)).collect();
                self.egraph.add(ENode::Tuple(elems))
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

                // Check if this is an opaque variable - restore original expression
                if let Some(original) = self.opaque_exprs.get(&name) {
                    return original.clone();
                }

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

            ExprTree::Recip(a) => {
                // recip(x) = 1.0 / x - emit as division since there's no Recip combinator
                Expr::Binary(BinaryExpr {
                    op: BinaryOp::Div,
                    lhs: Box::new(make_literal(1.0, span)),
                    rhs: Box::new(self.tree_to_expr(a)),
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

            // Unary
            ExprTree::Floor(a) => self.unary_method(a, "floor", span),
            ExprTree::Ceil(a) => self.unary_method(a, "ceil", span),
            ExprTree::Round(a) => self.unary_method(a, "round", span),
            ExprTree::Fract(a) => self.unary_method(a, "fract", span),
            ExprTree::Sin(a) => self.unary_method(a, "sin", span),
            ExprTree::Cos(a) => self.unary_method(a, "cos", span),
            ExprTree::Tan(a) => self.unary_method(a, "tan", span),
            ExprTree::Asin(a) => self.unary_method(a, "asin", span),
            ExprTree::Acos(a) => self.unary_method(a, "acos", span),
            ExprTree::Atan(a) => self.unary_method(a, "atan", span),
            ExprTree::Exp(a) => self.unary_method(a, "exp", span),
            ExprTree::Exp2(a) => self.unary_method(a, "exp2", span),
            ExprTree::Ln(a) => self.unary_method(a, "ln", span),
            ExprTree::Log2(a) => self.unary_method(a, "log2", span),
            ExprTree::Log10(a) => self.unary_method(a, "log10", span),

            // Binary
            ExprTree::Atan2(a, b) => self.binary_method(a, b, "atan2", span),
            ExprTree::Pow(a, b) => self.binary_method(a, b, "pow", span),
            ExprTree::Hypot(a, b) => self.binary_method(a, b, "hypot", span),

            // Comparisons
            ExprTree::Lt(a, b) => self.binary_op(a, b, BinaryOp::Lt, span),
            ExprTree::Le(a, b) => self.binary_op(a, b, BinaryOp::Le, span),
            ExprTree::Gt(a, b) => self.binary_op(a, b, BinaryOp::Gt, span),
            ExprTree::Ge(a, b) => self.binary_op(a, b, BinaryOp::Ge, span),
            ExprTree::Eq(a, b) => self.binary_op(a, b, BinaryOp::Eq, span),
            ExprTree::Ne(a, b) => self.binary_op(a, b, BinaryOp::Ne, span),

            // Ternary
            ExprTree::Select(a, b, c) => Expr::MethodCall(MethodCallExpr {
                receiver: Box::new(self.tree_to_expr(a)),
                method: Ident::new("select", span),
                args: vec![self.tree_to_expr(b), self.tree_to_expr(c)],
                span,
            }),
            ExprTree::Clamp(a, b, c) => Expr::MethodCall(MethodCallExpr {
                receiver: Box::new(self.tree_to_expr(a)),
                method: Ident::new("clamp", span),
                args: vec![self.tree_to_expr(b), self.tree_to_expr(c)],
                span,
            }),
            ExprTree::Tuple(elems) => Expr::Tuple(crate::ast::TupleExpr {
                elems: elems.iter().map(|e| self.tree_to_expr(e)).collect(),
                span,
            }),
        }
    }

    fn unary_method(&self, a: &ExprTree, name: &str, span: Span) -> Expr {
        Expr::MethodCall(MethodCallExpr {
            receiver: Box::new(self.tree_to_expr(a)),
            method: Ident::new(name, span),
            args: vec![],
            span,
        })
    }

    fn binary_method(&self, a: &ExprTree, b: &ExprTree, name: &str, span: Span) -> Expr {
        Expr::MethodCall(MethodCallExpr {
            receiver: Box::new(self.tree_to_expr(a)),
            method: Ident::new(name, span),
            args: vec![self.tree_to_expr(b)],
            span,
        })
    }

    fn binary_op(&self, a: &ExprTree, b: &ExprTree, op: BinaryOp, span: Span) -> Expr {
        Expr::Binary(BinaryExpr {
            op,
            lhs: Box::new(self.tree_to_expr(a)),
            rhs: Box::new(self.tree_to_expr(b)),
            span,
        })
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
        Expr::Tuple(mut tuple) => {
            tuple.elems = tuple.elems.into_iter().map(optimize_expr).collect();
            Expr::Tuple(tuple)
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
        // x / sqrt(y) should become x * rsqrt(y) via algebra:
        // x / sqrt(y) = x * (1/sqrt(y)) = x * rsqrt(y)
        let input = quote! { |x: f32, y: f32| x / y.sqrt() };
        let debug = optimize_code_egraph(input, &CostModel::with_fast_rsqrt());
        // Should use rsqrt (real instruction) instead of 1/sqrt
        assert!(debug.contains("rsqrt"));
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

    // ========================================================================
    // Cross-Binding Optimization Tests (Global Pass)
    // ========================================================================

    #[test]
    fn test_global_optimization_across_let_bindings() {
        // The global pass should see through let bindings:
        // let a = x; let b = 0.0; a + b → x
        let input = quote! { |x: f32| {
            let a = x;
            let b = 0.0;
            a + b
        }};
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Should simplify to just x (no Add, no 0.0 literal in result)
        assert!(debug.contains("x"), "Expected x in output: {}", debug);
        assert!(!debug.contains("Add"), "Should eliminate addition with zero: {}", debug);
    }

    #[test]
    fn test_global_optimization_zero_multiplication() {
        // let a = x * x; let b = 0.0; a * b → 0.0
        let input = quote! { |x: f32| {
            let a = x * x;
            let b = 0.0;
            a * b
        }};
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Should simplify to 0.0
        assert!(debug.contains("0"), "Expected 0 in output: {}", debug);
        assert!(!debug.contains("Mul"), "Should eliminate multiplication: {}", debug);
    }

    #[test]
    fn test_global_optimization_self_subtraction() {
        // let a = X * X + Y * Y; a - a → 0.0
        let input = quote! { || {
            let a = X * X + Y * Y;
            a - a
        }};
        let debug = optimize_code_egraph(input, &CostModel::default());
        // Should simplify to 0.0
        assert!(debug.contains("0"), "Expected 0 in output: {}", debug);
    }

    #[test]
    fn test_global_fma_across_bindings() {
        // let product = a * b; product + c → mul_add(a, b, c)
        let input = quote! { |a: f32, b: f32, c: f32| {
            let product = a * b;
            product + c
        }};
        let debug = optimize_code_egraph(input, &CostModel::with_fma());
        // Should fuse into mul_add
        assert!(debug.contains("mul_add"), "Expected FMA fusion: {}", debug);
    }

    #[test]
    fn test_discriminant_pattern() {
        // This is the problematic pattern:
        // d_dot_c² - (c_sq - r_sq) should use Neg to wrap (c_sq - r_sq)
        let input = quote! { |d: f32, c: f32, r: f32| {
            let d_sq = d * d;
            let c_sq = c * c;
            let r_sq = r * r;
            d_sq - (c_sq - r_sq)
        }};
        let debug = optimize_code_egraph(input, &CostModel::fully_optimized());
        eprintln!("Discriminant AST: {}", debug);

        // The AST should contain a Neg wrapping the inner subtraction
        // If FMA is used: mul_add(d, d, Neg(Sub(c_sq, r_sq)))
        // The key check: the output should NOT have the wrong sign pattern
        // Wrong pattern: Sub(c_sq, Neg(r_sq)) which equals c_sq + r_sq
        // Right pattern: Neg(Sub(c_sq, r_sq)) which equals -c_sq + r_sq

        // With FMA fusion, we expect: mul_add(d, d, ...)
        // And the third argument should involve a Neg wrapping the subtraction
        assert!(debug.contains("mul_add"), "Expected FMA fusion: {}", debug);

        // Check that Neg appears in the output (wrapping the inner expression)
        assert!(debug.contains("Neg") || debug.contains("neg"),
                "Expected Neg in third argument of mul_add: {}", debug);
    }

    #[test]
    fn test_discriminant_with_intrinsics() {
        // This matches the actual failing test more closely:
        // d_dot_c = X*cx + Y*cy + Z*cz
        // c_sq = cx*cx + cy*cy + cz*cz
        // r_sq = r*r
        // discriminant = d_dot_c*d_dot_c - (c_sq - r_sq)
        let input = quote! { |cx: f32, cy: f32, cz: f32, r: f32| {
            let d_dot_c = X * cx + Y * cy + Z * cz;
            let c_sq = cx * cx + cy * cy + cz * cz;
            let r_sq = r * r;
            d_dot_c * d_dot_c - (c_sq - r_sq)
        }};
        let debug = optimize_code_egraph(input, &CostModel::fully_optimized());
        eprintln!("Discriminant with intrinsics AST: {}", debug);

        // Check for FMA
        assert!(debug.contains("mul_add"), "Expected FMA fusion: {}", debug);

        // Check that Neg appears - the key correctness check
        assert!(debug.contains("Neg") || debug.contains("neg"),
                "Expected Neg in expression: {}", debug);
    }

    #[test]
    fn test_full_pipeline_discriminant() {
        use crate::codegen;

        // Full pipeline test matching actual kernel! macro
        let input = quote! { |cx: f32, cy: f32, cz: f32, r: f32| -> Jet3 {
            let d_dot_c = X * cx + Y * cy + Z * cz;
            let c_sq = cx * cx + cy * cy + cz * cz;
            let r_sq = r * r;
            d_dot_c * d_dot_c - (c_sq - r_sq)
        }};

        let kernel = parse(input).unwrap();
        let analyzed = analyze(kernel).unwrap();

        // This is what the kernel! macro does
        let optimized = optimize(analyzed);

        eprintln!("Optimized AST: {:?}", optimized.def.body);

        let output = codegen::emit(optimized);
        let output_str = output.to_string();

        eprintln!("Generated code:\n{}", output_str);

        // The key check: the output should have .neg() wrapping the inner subtraction
        // NOT: c_sq - r * r.neg() (which is c_sq + r²)
        // YES: (c_sq - r_sq).neg() (which is -c_sq + r²)

        // Check for the WRONG pattern (the bug)
        let has_wrong_pattern = output_str.contains("r . neg ( )") && !output_str.contains(") . neg ( )");
        assert!(!has_wrong_pattern, "Found wrong pattern (r.neg() without wrapping): {}", output_str);
    }
}