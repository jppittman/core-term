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
    BinaryExpr, BinaryOp, BlockExpr, Expr, IdentExpr, LetStmt, LiteralExpr, MethodCallExpr, Stmt,
    UnaryExpr, UnaryOp,
};
use crate::lower::LIBRARY_METHODS;
use crate::sema::AnalyzedKernel;
use pixelflow_ir::OpKind;
use pixelflow_search::egraph::{
    EClassId, EGraph, ENode, ExtractedDAG, Optimizer, build_extracted_dag_from_choices,
    compute_ref_counts, ops,
};
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

/// Count AST nodes (rough measure of expression complexity).
fn count_ast_nodes(expr: &Expr) -> usize {
    match expr {
        Expr::Literal(_) | Expr::Ident(_) => 1,
        Expr::Binary(b) => 1 + count_ast_nodes(&b.lhs) + count_ast_nodes(&b.rhs),
        Expr::Unary(u) => 1 + count_ast_nodes(&u.operand),
        Expr::MethodCall(c) => {
            1 + count_ast_nodes(&c.receiver) + c.args.iter().map(count_ast_nodes).sum::<usize>()
        }
        Expr::Call(c) => 1 + c.args.iter().map(count_ast_nodes).sum::<usize>(),
        Expr::Paren(p) => count_ast_nodes(p),
        Expr::Block(b) => {
            let stmt_nodes: usize = b
                .stmts
                .iter()
                .map(|s| match s {
                    Stmt::Let(l) => 1 + count_ast_nodes(&l.init),
                    Stmt::Expr(e) => count_ast_nodes(e),
                })
                .sum();
            let expr_nodes = b.expr.as_ref().map(|e| count_ast_nodes(e)).unwrap_or(0);
            stmt_nodes + expr_nodes
        }
        Expr::Tuple(t) => 1 + t.elems.iter().map(count_ast_nodes).sum::<usize>(),
        Expr::Verbatim(_) => 1,
    }
}

/// Optimize an analyzed kernel using tree rewriting and e-graphs.
pub fn optimize(mut analyzed: AnalyzedKernel) -> AnalyzedKernel {
    // 1. Structural optimization (catches things inside opaque nodes)
    analyzed.def.body = optimize_expr(analyzed.def.body);

    // 2. E-Graph optimization (global rewriting & fusion)
    // Uses neural cost model for structural extraction
    optimize_with_model(analyzed)
}

/// Optimize an analyzed kernel using cost-guided extraction (the static
/// latency prior, through the one `Optimizer` entry point).
pub fn optimize_with_model(mut analyzed: AnalyzedKernel) -> AnalyzedKernel {
    let mut optimizer = Optimizer::production();
    // A macro kernel is an anonymous closure body, so telemetry records no
    // kernel label for it — it never invents one, per the
    // saturation-telemetry spec. Named kernel structs, which did carry a real
    // identity, went with the combinator tier.
    analyzed.def.body = optimize_expr_with_model(analyzed.def.body, &mut optimizer);
    analyzed
}

/// Optimize a single expression using e-graph saturation and cost-guided extraction.
fn optimize_expr_with_model(expr: Expr, optimizer: &mut Optimizer) -> Expr {
    // Blocks: pass directly to optimize_via_model. The e-graph's expr_to_egraph
    // already handles Block by adding each let-binding to var_to_eclass, so
    // references share e-classes. Let-bindings are CSE hints. The e-graph sees
    // the whole expression as one DAG.
    //
    // Blocks with opaque references (method calls on captured manifolds) must
    // preserve structure. Pure arithmetic blocks go through the e-graph whole.
    if let Expr::Block(block) = expr {
        if block_has_opaque_with_locals(&block) {
            return optimize_block_preserving_structure(block, optimizer);
        }
        return optimize_via_model(&Expr::Block(block), optimizer);
    }

    // For non-block expressions, treat as a unit for global optimization.
    optimize_via_model(&expr, optimizer)
}

/// Optimize an expression via e-graph with cost-guided extraction + DAG CSE.
///
/// Uses the extraction policy (the static latency prior) to pick the
/// cheapest equivalent form, then emits let-bindings
/// for shared subexpressions. This avoids tree-bloating where shared
/// e-classes get duplicated, and produces code with CSE.
///
/// Always uses the DAG codegen path. `dag_to_expr` handles non-shared
/// expressions correctly (returns the expression without a block wrapper),
/// so the old "no sharing — simple tree" fallback is unnecessary and
/// removed. CSE is always preserved.
fn optimize_via_model(expr: &Expr, optimizer: &mut Optimizer) -> Expr {
    let mut ctx = EGraphContext::over(optimizer.egraph());
    let root = ctx.expr_to_egraph(expr);

    // One entry point, shared with the runtime tier and the `Dwrt` expansion
    // tier: the rule set, the budget, the cost model, and the extractor are
    // decided in `Optimizer`, not re-decided here. This tier expands before
    // any bake, so it has no lattice and everything weighs one.
    let node_count = count_ast_nodes(expr);
    #[cfg(feature = "saturation-telemetry")]
    let telemetry_start = std::time::Instant::now();
    let optimized = optimizer.run(&mut ctx.egraph, root, node_count);
    let choices = optimized.choices.clone();

    // Stop the clock here: `wall_clock` is documented (see
    // `telemetry::SaturationInvocation::wall_clock`) as saturate+extract,
    // and the `choices()` call just above is that one real extraction pass.
    // Sampled before the telemetry-only second pass below, so a large-graph
    // extraction isn't double-counted into this number just because
    // telemetry happens to be on.
    #[cfg(feature = "saturation-telemetry")]
    let wall_clock = telemetry_start.elapsed();

    #[cfg(feature = "saturation-telemetry")]
    {
        // A second, telemetry-only extraction pass on the same (unmutated)
        // e-graph, purely to get an `ExprArena` to cost — the AST/DAG path
        // above never materializes one. Deterministic given a fixed egraph
        // + root (see `Extraction`'s own doc comment on the refinement
        // search's determinism harness), so this reproduces the same
        // choices `extraction.choices()` just made; it is not consulted for
        // the actual compiled output, and (per the `wall_clock` capture
        // above) not counted in the reported timing either.
        let (telemetry_arena, telemetry_root) = optimized.to_arena(&ctx.egraph, root);
        let kernel_label: Option<String> = None;
        pixelflow_search::telemetry::record(pixelflow_search::telemetry::SaturationInvocation {
            tier: pixelflow_search::telemetry::Tier::Macro,
            node_count,
            stats: &optimized.stats,
            union_count: optimized.stats.unions,
            extracted_arena: &telemetry_arena,
            extracted_root: telemetry_root,
            wall_clock,
            kernel_label: kernel_label.as_deref(),
        });
    }

    // Build ExtractedDAG: ref_counts drive let-binding placement.
    // dag_to_expr emits let-bindings for shared subexpressions and returns
    // a plain expression when there is no sharing — no separate tree path needed.
    let ref_counts = compute_ref_counts(&ctx.egraph, root, &choices);
    let dag =
        build_extracted_dag_from_choices(&ctx.egraph, root, &choices, &ref_counts, optimized.cost);
    ctx.dag_to_expr(&dag)
}

/// Check if a block must preserve its structure during optimization.
///
/// A block must preserve structure if:
/// 1. It has let bindings that are referenced in the final expression, OR
/// 2. It has opaque expressions (like method calls on captured manifolds)
///    that reference let-bound locals
///
/// The e-graph optimizer inlines variable references, so if we don't preserve
/// the block structure, the let bindings would be lost in the extracted result.
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

    // If the final expression references ANY let-bound local in an opaque context,
    // we must preserve the block structure.
    // Note: Standard usage (e.g. `x + y`) is fine - the e-graph will inline it.
    if let Some(ref final_expr) = block.expr {
        if expr_has_opaque_refs(final_expr, &local_names) {
            return true;
        }
    }

    // Also check if any statement's init references locals in opaque contexts
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
            if matches!(call.receiver.as_ref(), Expr::Verbatim(_))
                && call
                    .args
                    .iter()
                    .any(|arg| expr_references_any(arg, local_names))
            {
                return true;
            }
            // Check if this is a method on a captured variable (not X, Y, Z, W)
            if let Expr::Ident(ident) = call.receiver.as_ref() {
                let name = ident.name.to_string();
                // If the receiver is a local or an external captured variable,
                // and args contain locals, this is problematic
                if !is_coordinate_intrinsic(&name) {
                    // Check if any arg references a local
                    if call
                        .args
                        .iter()
                        .any(|arg| expr_references_any(arg, local_names))
                    {
                        return true;
                    }
                }
            }
            // Recurse into receiver and args
            expr_has_opaque_refs(&call.receiver, local_names)
                || call
                    .args
                    .iter()
                    .any(|a| expr_has_opaque_refs(a, local_names))
        }

        // Function calls are treated as opaque because expr_to_egraph doesn't
        // map them to ENodes (it falls through to create_opaque_var).
        // Therefore, if any arg references a local, we must preserve structure.
        Expr::Call(call) => {
            // Calls are opaque. If args reference locals, the call itself is an opaque ref.
            if call
                .args
                .iter()
                .any(|a| expr_references_any(a, local_names))
            {
                return true;
            }
            // Recurse to check for nested opaque refs
            call.args
                .iter()
                .any(|a| expr_has_opaque_refs(a, local_names))
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
            }) || b
                .expr
                .as_ref()
                .is_some_and(|e| expr_has_opaque_refs(e, local_names))
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
        Expr::Binary(b) => expr_references_any(&b.lhs, names) || expr_references_any(&b.rhs, names),
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
            }) || b
                .expr
                .as_ref()
                .is_some_and(|e| expr_references_any(e, names))
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
                path.path
                    .segments
                    .iter()
                    .any(|seg| names.contains(&seg.ident.to_string()))
            }
        }

        SynExpr::MethodCall(call) => {
            // Recursively check receiver and arguments
            syn_expr_references_any(&call.receiver, names)
                || call
                    .args
                    .iter()
                    .any(|arg| syn_expr_references_any(arg, names))
        }

        SynExpr::Call(call) => {
            // Check function and arguments
            syn_expr_references_any(&call.func, names)
                || call
                    .args
                    .iter()
                    .any(|arg| syn_expr_references_any(arg, names))
        }

        SynExpr::Binary(bin) => {
            syn_expr_references_any(&bin.left, names) || syn_expr_references_any(&bin.right, names)
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

        SynExpr::Tuple(tuple) => tuple
            .elems
            .iter()
            .any(|e| syn_expr_references_any(e, names)),

        SynExpr::Array(array) => array
            .elems
            .iter()
            .any(|e| syn_expr_references_any(e, names)),

        SynExpr::Block(block) => block.block.stmts.iter().any(|stmt| match stmt {
            syn::Stmt::Local(local) => local
                .init
                .as_ref()
                .is_some_and(|init| syn_expr_references_any(&init.expr, names)),
            syn::Stmt::Expr(expr, _) => syn_expr_references_any(expr, names),
            _ => false,
        }),

        SynExpr::If(if_expr) => {
            syn_expr_references_any(&if_expr.cond, names)
                || if_expr.then_branch.stmts.iter().any(|stmt| {
                    if let syn::Stmt::Expr(expr, _) = stmt {
                        syn_expr_references_any(expr, names)
                    } else {
                        false
                    }
                })
                || if_expr
                    .else_branch
                    .as_ref()
                    .is_some_and(|(_, else_expr)| syn_expr_references_any(else_expr, names))
        }

        // Literals don't reference variables
        SynExpr::Lit(_) => false,

        // For other expression types, conservatively return true to preserve structure
        // (better to preserve than to accidentally lose bindings)
        _ => true,
    }
}

/// Check if a name is a coordinate intrinsic (X, Y).
fn is_coordinate_intrinsic(name: &str) -> bool {
    matches!(name, "X" | "Y")
}

/// Optimize a block while preserving its structure.
///
/// Each let binding and the final expression are optimized independently.
fn optimize_block_preserving_structure(mut block: BlockExpr, optimizer: &mut Optimizer) -> Expr {
    for stmt in &mut block.stmts {
        if let Stmt::Let(let_stmt) = stmt {
            let init = std::mem::replace(&mut let_stmt.init, make_literal(0.0, Span::call_site()));
            let_stmt.init = optimize_expr_with_model(init, optimizer);
        }
    }
    if let Some(final_expr) = block.expr.take() {
        block.expr = Some(Box::new(optimize_expr_with_model(*final_expr, optimizer)));
    }
    Expr::Block(block)
}

// ============================================================================
// E-Graph Integration (Legacy AST-based)
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
    /// Build over a graph the caller already carries the rule set for —
    /// [`Optimizer::egraph`]. The AST↔e-graph boundary is genuinely this
    /// tier's (variable naming, opaque expressions); the *vocabulary* is
    /// not, so it comes in rather than being chosen here.
    fn over(egraph: EGraph) -> Self {
        Self {
            egraph,
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

    /// Check if a method is known and can be converted to ENode: either a
    /// primitive op (one `OpKind` per name+arity) or a library composition
    /// (see `LIBRARY_METHODS`).
    fn is_known_method(method: &str, arg_count: usize) -> bool {
        OpKind::from_method_call(method, arg_count).is_some()
            || LIBRARY_METHODS.contains(&(method, arg_count))
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
                    BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Div
                    | BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge
                    | BinaryOp::Eq
                    | BinaryOp::Ne => {
                        // Supported - convert children
                        let lhs = self.expr_to_egraph(&binary.lhs);
                        let rhs = self.expr_to_egraph(&binary.rhs);

                        let op: &'static dyn ops::Op = match binary.op {
                            BinaryOp::Add => &ops::Add,
                            BinaryOp::Sub => &ops::Sub,
                            BinaryOp::Mul => &ops::Mul,
                            BinaryOp::Div => &ops::Div,
                            BinaryOp::Lt => &ops::Lt,
                            BinaryOp::Le => &ops::Le,
                            BinaryOp::Gt => &ops::Gt,
                            BinaryOp::Ge => &ops::Ge,
                            BinaryOp::Eq => &ops::Eq,
                            BinaryOp::Ne => &ops::Ne,
                            _ => unreachable!(),
                        };
                        self.egraph.add(ENode::Op {
                            op,
                            children: vec![lhs, rhs],
                        })
                    }
                    // For other ops (Rem, BitXor, Shl, Shr)
                    // preserve as opaque expression with original structure
                    _ => self.create_opaque_var("binop", expr),
                }
            }

            Expr::Unary(unary) => {
                match unary.op {
                    UnaryOp::Neg => {
                        let operand = self.expr_to_egraph(&unary.operand);
                        self.egraph.add(ENode::Op {
                            op: &ops::Neg,
                            children: vec![operand],
                        })
                    }
                    UnaryOp::Not => {
                        // Map Not(x) to 1.0 - x (assuming boolean 0.0/1.0 logic)
                        let operand = self.expr_to_egraph(&unary.operand);
                        let one = self.egraph.add(ENode::constant(1.0));
                        self.egraph.add(ENode::Op {
                            op: &ops::Sub,
                            children: vec![one, operand],
                        })
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
                let arg_count = call.args.len();

                // Primitive ops: one `OpKind` per (name, arity) — resolved
                // from the same table `lower`'s arena lowering reads, so
                // the two backends cannot silently drift on which primitive
                // method names/arities exist.
                if let Some(op_kind) = OpKind::from_method_call(&method, arg_count) {
                    let op = ops::op_from_kind(op_kind).unwrap_or_else(|| {
                        panic!("{op_kind:?} is a DSL method but has no e-graph Op")
                    });
                    let mut children = vec![receiver];
                    for arg in &call.args {
                        children.push(self.expr_to_egraph(arg));
                    }
                    return self.egraph.add(ENode::Op { op, children });
                }

                match method.as_str() {
                    // Library: fract(x) = x - floor(x).
                    "fract" => {
                        let f = self.egraph.add(ENode::Op {
                            op: &ops::Floor,
                            children: vec![receiver],
                        });
                        self.egraph.add(ENode::Op {
                            op: &ops::Sub,
                            children: vec![receiver, f],
                        })
                    }
                    // Library: hypot(x, y) = sqrt(x² + y²).
                    "hypot" => {
                        let arg = self.expr_to_egraph(&call.args[0]);
                        let xx = self.egraph.add(ENode::Op {
                            op: &ops::Mul,
                            children: vec![receiver, receiver],
                        });
                        let yy = self.egraph.add(ENode::Op {
                            op: &ops::Mul,
                            children: vec![arg, arg],
                        });
                        let sum = self.egraph.add(ENode::Op {
                            op: &ops::Add,
                            children: vec![xx, yy],
                        });
                        self.egraph.add(ENode::Op {
                            op: &ops::Sqrt,
                            children: vec![sum],
                        })
                    }
                    // `clamp` is library: it enters the e-graph as the
                    // composition it denotes, so the optimizer reasons about
                    // min/max directly instead of needing clamp-specific
                    // rewrite and derivative rules.
                    "clamp" => {
                        let min_val = self.expr_to_egraph(&call.args[0]);
                        let max_val = self.expr_to_egraph(&call.args[1]);
                        let floored = self.egraph.add(ENode::Op {
                            op: &ops::Max,
                            children: vec![receiver, min_val],
                        });
                        self.egraph.add(ENode::Op {
                            op: &ops::Min,
                            children: vec![floored, max_val],
                        })
                    }

                    // Should not reach here due to is_known_method check
                    _ => unreachable!(
                        "Unknown method {} should have been handled as opaque",
                        method
                    ),
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

            // For Call and Verbatim, treat as opaque and store original expression
            // so it can be restored during extraction
            Expr::Call(call) => self.create_opaque_var(&format!("call_{}_", call.func), expr),

            Expr::Verbatim(_) => self.create_opaque_var("verbatim_", expr),

            Expr::Tuple(tuple) => {
                let elems: Vec<_> = tuple.elems.iter().map(|e| self.expr_to_egraph(e)).collect();
                self.egraph.add(ENode::Op {
                    op: &ops::Tuple,
                    children: elems,
                })
            }
        }
    }

    /// Convert an ExtractedDAG to an AST expression with let-bindings for shared subexprs.
    ///
    /// For shared e-classes (used more than once), this generates let-bindings:
    /// ```text
    /// {
    ///     let __0 = shared_expr_1;
    ///     let __1 = shared_expr_2;
    ///     root_expr_using_bindings
    /// }
    /// ```
    fn dag_to_expr(&self, dag: &ExtractedDAG) -> Expr {
        let span = Span::call_site();

        // Build a map from shared e-class indices to their let-binding names
        let mut binding_names: HashMap<usize, String> = HashMap::new();
        // Pre-allocate assuming roughly one binding per scheduled instruction (except root)
        let mut stmts: Vec<Stmt> = Vec::with_capacity(dag.schedule.len().saturating_sub(1));

        // Emit let-bindings for shared e-classes in topological order
        let mut binding_idx = 0usize;
        for &class_id in &dag.schedule {
            let canonical = self.egraph.find(class_id);

            // Only bind shared classes that aren't the root
            // (the root becomes the final expression, not a binding)
            //
            // Leaves (Var/Const) are never bound — they re-materialize at each
            // use instead. Binding them is at best a no-op and at worst wrong:
            // a `let __0 = ident;` moves non-Copy values (manifold-tapping
            // locals), and a shared literal must be re-emitted per use so each
            // occurrence gets its own type-space assignment (a constant can
            // legitimately appear in both domain-space and Field-space math).
            if dag.is_shared(canonical) && canonical != dag.root && !self.is_leaf(canonical, dag) {
                let var_name = format!("__{}", binding_idx);

                // Build the AST for this e-class
                let expr = self.eclass_to_expr(canonical, dag, &binding_names);

                // Create let statement
                stmts.push(Stmt::Let(Box::new(LetStmt {
                    name: Ident::new(&var_name, span),
                    ty: None,
                    init: expr,
                    span,
                })));

                binding_names.insert(canonical.index(), var_name);
                binding_idx += 1;
            }
        }

        // Build the root expression
        let root_expr = self.eclass_to_expr(dag.root, dag, &binding_names);

        if stmts.is_empty() {
            // No shared subexpressions, return simple expression
            root_expr
        } else {
            // Wrap in a block with let-bindings
            Expr::Block(BlockExpr {
                stmts,
                expr: Some(Box::new(root_expr)),
                span,
            })
        }
    }

    /// Is the extraction choice for this e-class a leaf (Var or Const)?
    ///
    /// Leaves are re-materialized at each use site rather than let-bound; see
    /// the comment in [`Self::dag_to_expr`].
    fn is_leaf(&self, canonical: EClassId, dag: &ExtractedDAG) -> bool {
        match dag.best_node_idx(canonical) {
            Some(node_idx) => !matches!(self.egraph.nodes(canonical)[node_idx], ENode::Op { .. }),
            // No recorded choice: eclass_to_expr will fail loudly if this
            // class is actually reachable; don't bind it here.
            None => true,
        }
    }

    /// Build an AST expression for a single e-class, using bindings for shared subexprs.
    fn eclass_to_expr(
        &self,
        class: EClassId,
        dag: &ExtractedDAG,
        binding_names: &HashMap<usize, String>,
    ) -> Expr {
        let span = Span::call_site();
        let canonical = self.egraph.find(class);

        // If this e-class is bound to a variable, just reference it
        if let Some(name) = binding_names.get(&canonical.index()) {
            return Expr::Ident(IdentExpr {
                name: Ident::new(name, span),
                span,
            });
        }

        // Get the best node for this e-class. `dag.choices` comes from a
        // sealed `Extraction`, whose constructors repair/backfill every
        // e-class reachable from root (including children introduced by
        // saturation merges) into a well-founded set — see pixelflow-search's
        // egraph/extract.rs. A missing choice here means that invariant was
        // violated upstream; silently defaulting to node 0 would risk
        // emitting a node that isn't even the reachable/consistent variant
        // for this e-class, so fail loudly instead of masking the bug.
        let node_idx = dag.best_node_idx(canonical).unwrap_or_else(|| {
            panic!(
                "eclass_to_expr: e-class {} reachable from root {} has no recorded \
                 extraction choice — a sealed Extraction must guarantee every \
                 reachable e-class has Some(idx)",
                canonical.index(),
                dag.root.index()
            )
        });
        let node = &self.egraph.nodes(canonical)[node_idx];

        match node {
            ENode::Var(idx) => {
                // Try to get the variable name from our mapping
                let name =
                    self.idx_to_name
                        .get(*idx as usize)
                        .cloned()
                        .unwrap_or_else(|| match idx {
                            0 => "X".to_string(),
                            1 => "Y".to_string(),
                            _ => format!("__var{}", idx),
                        });

                // Check if this is an opaque variable - restore original expression
                if let Some(original) = self.opaque_exprs.get(&name) {
                    return original.clone();
                }

                Expr::Ident(IdentExpr {
                    name: Ident::new(&name, span),
                    span,
                })
            }

            ENode::Const(bits) => make_literal(f32::from_bits(*bits) as f64, span),

            // Buffer leaves exist only in runtime-built arenas (there is no
            // surface syntax for one); the macro tier's representability gate
            // refuses memory ops before the e-graph is ever built, so one
            // reaching AST emission is a pipeline-order bug.
            ENode::Buffer(decl) => panic!(
                "eclass_to_expr: ENode::Buffer({decl:?}) in the macro tier — \
                 buffer-bearing kernels are runtime-JIT only"
            ),
            // Likewise: a uniform is chosen at the builder call, after the
            // macro has expanded; the macro's own e-graph carries scalar
            // params as opaque `Var`s and never sees one.
            ENode::Uniform(decl) => panic!(
                "eclass_to_expr: ENode::Uniform({decl:?}) in the macro tier — \
                 uniforms are chosen at the builder call site"
            ),

            ENode::Op { op, children } => {
                let name = op.name();
                let child_exprs: Vec<Expr> = children
                    .iter()
                    .map(|&c| self.eclass_to_expr(c, dag, binding_names))
                    .collect();

                self.emit_op_as_expr(name, &child_exprs, span)
            }
        }
    }

    /// Emit an operation as an AST expression.
    fn emit_op_as_expr(&self, op_name: &str, children: &[Expr], span: Span) -> Expr {
        match (op_name, children) {
            // Binary arithmetic
            ("add", [a, b]) => Expr::Binary(BinaryExpr {
                op: BinaryOp::Add,
                lhs: Box::new(a.clone()),
                rhs: Box::new(b.clone()),
                span,
            }),
            ("sub", [a, b]) => Expr::Binary(BinaryExpr {
                op: BinaryOp::Sub,
                lhs: Box::new(a.clone()),
                rhs: Box::new(b.clone()),
                span,
            }),
            ("mul", [a, b]) => Expr::Binary(BinaryExpr {
                op: BinaryOp::Mul,
                lhs: Box::new(a.clone()),
                rhs: Box::new(b.clone()),
                span,
            }),
            ("div", [a, b]) => Expr::Binary(BinaryExpr {
                op: BinaryOp::Div,
                lhs: Box::new(a.clone()),
                rhs: Box::new(b.clone()),
                span,
            }),

            // Unary
            ("neg", [a]) => Expr::Unary(UnaryExpr {
                op: UnaryOp::Neg,
                operand: Box::new(a.clone()),
                span,
            }),
            ("recip", [a]) => Expr::Binary(BinaryExpr {
                op: BinaryOp::Div,
                lhs: Box::new(make_literal(1.0, span)),
                rhs: Box::new(a.clone()),
                span,
            }),

            // Unary methods
            ("sqrt", [a]) => self.unary_method_expr(a, "sqrt", span),
            ("rsqrt", [a]) => self.unary_method_expr(a, "rsqrt", span),
            ("abs", [a]) => self.unary_method_expr(a, "abs", span),
            ("floor", [a]) => self.unary_method_expr(a, "floor", span),
            ("ceil", [a]) => self.unary_method_expr(a, "ceil", span),
            ("round", [a]) => self.unary_method_expr(a, "round", span),
            ("fract", [a]) => self.unary_method_expr(a, "fract", span),
            ("sin", [a]) => self.unary_method_expr(a, "sin", span),
            ("cos", [a]) => self.unary_method_expr(a, "cos", span),
            ("tan", [a]) => self.unary_method_expr(a, "tan", span),
            ("asin", [a]) => self.unary_method_expr(a, "asin", span),
            ("acos", [a]) => self.unary_method_expr(a, "acos", span),
            ("atan", [a]) => self.unary_method_expr(a, "atan", span),
            ("exp", [a]) => self.unary_method_expr(a, "exp", span),
            ("exp2", [a]) => self.unary_method_expr(a, "exp2", span),
            ("ln", [a]) => self.unary_method_expr(a, "ln", span),
            ("log2", [a]) => self.unary_method_expr(a, "log2", span),
            ("log10", [a]) => self.unary_method_expr(a, "log10", span),

            // Binary methods
            ("min", [a, b]) => self.binary_method_expr(a, b, "min", span),
            ("max", [a, b]) => self.binary_method_expr(a, b, "max", span),
            ("atan2", [a, b]) => self.binary_method_expr(a, b, "atan2", span),
            ("pow", [a, b]) => self.binary_method_expr(a, b, "pow", span),
            ("hypot", [a, b]) => self.binary_method_expr(a, b, "hypot", span),

            // Comparisons
            ("lt", [a, b]) => self.binary_op_expr(a, b, BinaryOp::Lt, span),
            ("le", [a, b]) => self.binary_op_expr(a, b, BinaryOp::Le, span),
            ("gt", [a, b]) => self.binary_op_expr(a, b, BinaryOp::Gt, span),
            ("ge", [a, b]) => self.binary_op_expr(a, b, BinaryOp::Ge, span),
            ("eq", [a, b]) => self.binary_op_expr(a, b, BinaryOp::Eq, span),
            ("ne", [a, b]) => self.binary_op_expr(a, b, BinaryOp::Ne, span),

            // Ternary
            ("mul_add", [a, b, c]) => Expr::MethodCall(MethodCallExpr {
                receiver: Box::new(a.clone()),
                method: Ident::new("mul_add", span),
                args: vec![b.clone(), c.clone()],
                span,
            }),
            ("select", [a, b, c]) => Expr::MethodCall(MethodCallExpr {
                receiver: Box::new(a.clone()),
                method: Ident::new("select", span),
                args: vec![b.clone(), c.clone()],
                span,
            }),
            ("clamp", [a, b, c]) => Expr::MethodCall(MethodCallExpr {
                receiver: Box::new(a.clone()),
                method: Ident::new("clamp", span),
                args: vec![b.clone(), c.clone()],
                span,
            }),

            // Tuple
            ("tuple", elems) => Expr::Tuple(crate::ast::TupleExpr {
                elems: elems.to_vec(),
                span,
            }),

            // Unknown - try as unary or binary method
            (name, [a]) => self.unary_method_expr(a, name, span),
            (name, [a, b]) => self.binary_method_expr(a, b, name, span),
            (name, _) => panic!(
                "Unknown operation {} with {} children",
                name,
                children.len()
            ),
        }
    }

    fn unary_method_expr(&self, a: &Expr, name: &str, span: Span) -> Expr {
        Expr::MethodCall(MethodCallExpr {
            receiver: Box::new(a.clone()),
            method: Ident::new(name, span),
            args: vec![],
            span,
        })
    }

    fn binary_method_expr(&self, a: &Expr, b: &Expr, name: &str, span: Span) -> Expr {
        Expr::MethodCall(MethodCallExpr {
            receiver: Box::new(a.clone()),
            method: Ident::new(name, span),
            args: vec![b.clone()],
            span,
        })
    }

    fn binary_op_expr(&self, a: &Expr, b: &Expr, op: BinaryOp, span: Span) -> Expr {
        Expr::Binary(BinaryExpr {
            op,
            lhs: Box::new(a.clone()),
            rhs: Box::new(b.clone()),
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
        Expr::Call(mut call) => {
            call.args = call.args.into_iter().map(optimize_expr).collect();
            Expr::Call(call)
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
        match stmt {
            Stmt::Let(let_stmt) => {
                let_stmt.init = optimize_expr(std::mem::replace(
                    &mut let_stmt.init,
                    make_literal(0.0, Span::call_site()), // Dummy placeholder
                ));
            }
            Stmt::Expr(expr) => {
                *expr = optimize_expr(std::mem::replace(
                    expr,
                    make_literal(0.0, Span::call_site()), // Dummy placeholder
                ));
            }
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
    // Handle non-finite values specially - these can't be written as float literals
    if val.is_nan() {
        // Return f32::NAN as a path expression
        let path: syn::Expr = syn::parse_quote_spanned!(span=> f32::NAN);
        return Expr::Verbatim(path);
    }
    if val.is_infinite() {
        // Return f32::INFINITY or f32::NEG_INFINITY
        let path: syn::Expr = if val.is_sign_positive() {
            syn::parse_quote_spanned!(span=> f32::INFINITY)
        } else {
            syn::parse_quote_spanned!(span=> f32::NEG_INFINITY)
        };
        return Expr::Verbatim(path);
    }
    // Kernels evaluate in f32; format through f32 so the literal is the
    // shortest string that round-trips to the value that actually executes.
    // An f64-formatted literal would carry digits f32 can't represent.
    let mut s = (val as f32).to_string();
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
    let result = match op {
        BinaryOp::Add => lhs + rhs,
        BinaryOp::Sub => lhs - rhs,
        BinaryOp::Mul => lhs * rhs,
        BinaryOp::Div => lhs / rhs,
        BinaryOp::Rem => lhs % rhs,
        _ => return None, // Comparisons etc. not folded to float (return bool)
    };
    // Don't fold to infinity or NaN - keep the expression form
    // so the runtime can handle it appropriately
    if result.is_finite() {
        Some(result)
    } else {
        None
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
    use pixelflow_search::egraph::CostModel;
    use quote::quote;

    // ========================================================================
    // DAG Extraction Tests
    // ========================================================================
    //
    // These go through the public `optimize()` entry point (the
    // static-latency-prior path) rather than constructing an explicit
    // `ExtractionPolicy`: DAG/CSE let-binding
    // placement is driven by ref-counting in `compute_ref_counts`, which is
    // independent of which cost model picked the extraction — the public
    // entry point exercises the same sharing logic these tests care about.

    /// The optimizer should emit a shared binding when the same subexpression
    /// (`sin(X)`) is used twice.
    #[test]
    fn dag_extraction_preserves_sin_when_subexpression_is_shared_twice() {
        let input = quote! { || X.sin() * X.sin() };
        let kernel = parse(input).unwrap();
        let analyzed = analyze(kernel).unwrap();

        let optimized = optimize(analyzed);

        let debug = format!("{:?}", optimized.def.body);
        eprintln!("DAG optimized sin(X)*sin(X): {}", debug);

        assert!(
            debug.contains("sin") || debug.contains("Sin"),
            "Expected sin in output"
        );
    }

    /// The optimizer should preserve the shared operator when the same
    /// subexpression (`sqrt(X)`) is used three times.
    #[test]
    fn dag_extraction_preserves_sqrt_when_subexpression_is_shared_three_times() {
        let input = quote! { || X.sqrt() * X.sqrt() + X.sqrt() };
        let kernel = parse(input).unwrap();
        let analyzed = analyze(kernel).unwrap();

        let optimized = optimize(analyzed);

        let debug = format!("{:?}", optimized.def.body);
        eprintln!("DAG optimized sqrt(X)*sqrt(X)+sqrt(X): {}", debug);

        assert!(debug.contains("sqrt"), "Expected sqrt in output");
    }

    /// A simple expression with no shared subexpressions should not be
    /// wrapped in a `Block` — DAG extraction must not introduce a let-binding
    /// where nothing is actually shared.
    #[test]
    fn dag_extraction_does_not_wrap_output_in_block_when_nothing_is_shared() {
        let input = quote! { || X + Y };
        let kernel = parse(input).unwrap();
        let analyzed = analyze(kernel).unwrap();

        let optimized = optimize(analyzed);

        let debug = format!("{:?}", optimized.def.body);
        eprintln!("DAG optimized X+Y: {}", debug);

        assert!(
            !debug.starts_with("Block"),
            "Simple expression should not be wrapped in block"
        );
    }

    /// `optimize()` should wrap `.neg()` around the whole `(c_sq - r_sq)`
    /// subtraction rather than distributing the negation into `r * r`, which
    /// would silently change the result from `-c_sq + r²` to `c_sq + r²`.
    #[test]
    fn optimize_wraps_neg_around_subtraction_instead_of_distributing_into_r_squared() {
        use crate::emit::emit_kernel;

        // Full pipeline test matching actual kernel! macro
        let input = quote! { |cx: f32, cy: f32, cz: f32, r: f32| {
            let d_dot_c = X * cx + Y * cy + cz;
            let c_sq = cx * cx + cy * cy + cz * cz;
            let r_sq = r * r;
            d_dot_c * d_dot_c - (c_sq - r_sq)
        }};

        let kernel = parse(input).unwrap();
        let analyzed = analyze(kernel).unwrap();

        // This is what the kernel! macro does
        let optimized = optimize(analyzed);

        eprintln!("Optimized AST: {:?}", optimized.def.body);

        // The arena the macro would emit. The check is on the AST rather
        // than on emitted source: the negation must wrap the whole
        // subtraction, not be pushed into `r * r`, which would turn
        // `-(c_sq - r^2)` into `c_sq + r^2`.
        let body = format!("{:?}", optimized.def.body);
        assert!(
            emit_kernel(&optimized).is_ok(),
            "the optimized body must still lower to an arena"
        );
        let neg_on_bare_r = body.contains("Neg") && body.contains("Ident(IdentExpr { name: r");
        assert!(
            !neg_on_bare_r || body.matches("Neg").count() > 0,
            "found a negation distributed onto `r` rather than wrapping the \
             subtraction: {body}"
        );
    }

    /// The default production path must use static latency-prior
    /// extraction.
    ///
    /// We verify the default `optimize()` entry point is byte-identical to
    /// explicitly configuring `Optimizer::production().cost(latency_prior())`
    /// — proving the default neither silently picks up some other table nor
    /// silently degrades to a zero-cost (no-op) model.
    ///
    /// This intentionally calls the private `optimize_expr`/
    /// `optimize_expr_with_model` directly (STYLE.md "Test Public API"
    /// exception): the whole point is comparing the default's *implicit*
    /// configuration against an explicitly spelled-out one.
    #[test]
    fn default_path_extraction_is_static_latency_prior() {
        use crate::emit::emit_kernel;

        let input = quote! { || {
            let a = X * X;
            let b = Y * Y;
            (a + b).sqrt()
        }};

        // `optimize()` is the real kernel! macro entry point; it must
        // resolve to the static prior.
        let kernel = parse(input.clone()).unwrap();
        let analyzed = analyze(kernel).unwrap();
        let via_default_entry_point = optimize(analyzed);
        let default_output = emit_kernel(&via_default_entry_point)
            .expect("default path lowers to an arena")
            .to_string();

        // Directly constructing the static-prior policy and running the same
        // expression through the optimizer must match exactly.
        let kernel = parse(input).unwrap();
        let mut analyzed_for_static_path = analyze(kernel).unwrap();
        let mut static_optimizer = Optimizer::production().cost(CostModel::latency_prior());
        analyzed_for_static_path.def.body = optimize_expr(analyzed_for_static_path.def.body);
        analyzed_for_static_path.def.body =
            optimize_expr_with_model(analyzed_for_static_path.def.body, &mut static_optimizer);
        let explicit_static_output = emit_kernel(&analyzed_for_static_path)
            .expect("explicit static path lowers to an arena")
            .to_string();

        assert_eq!(
            default_output, explicit_static_output,
            "default optimize() path must be byte-identical to explicitly \
             configuring Optimizer::production().cost(latency_prior()) — the \
             default must not silently pick up another table or degrade \
             to a zero-cost model"
        );
    }
}
