//! Bridge between macro AST and pixelflow-ir.
//!
//! This module handles conversions between:
//! 1. Macro AST → arena IR
//! 2. arena IR → runtime construction code
//!
//! The IR becomes the canonical representation, with AST only used during parsing.

use crate::ast::{BinaryOp, Expr, UnaryOp};
use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId};
use proc_macro2::TokenStream;
use quote::quote;
use std::collections::HashMap;
use syn::Lit;

// ============================================================================
// AST → Arena IR Conversion
// ============================================================================

/// Build a `param_name → index` map over the params of a kernel.
///
/// Indices are dense in declaration order: each becomes a `Param(i)` arena
/// node, substituted by `substitute_params` with the builder closure's
/// arguments in the same order.
pub fn param_indices(analyzed: &crate::sema::AnalyzedKernel) -> HashMap<String, u8> {
    analyzed
        .def
        .params
        .iter()
        .enumerate()
        .map(|(i, p)| (p.name.to_string(), i as u8))
        .collect()
}

/// Convert macro AST to an arena-allocated IR.
///
/// Mirrors [`ast_to_ir`] exactly but pushes nodes into `arena` instead of
/// heap-allocating [`Arc`] wrappers. Children are recursed first so that
/// parent nodes always reference already-interned [`ExprId`]s.
///
/// `param_indices` maps parameter names to their declaration-order index (0-based).
/// Parameter identifiers are emitted as arena `Param(i)` nodes.
pub fn ast_to_arena(
    expr: &Expr,
    param_indices: &HashMap<String, u8>,
    arena: &mut ExprArena,
) -> Result<ExprId, String> {
    let mut locals: HashMap<String, ExprId> = HashMap::new();
    let ctx = Ctx { param_indices };
    ast_to_arena_inner(expr, &ctx, &mut locals, arena)
}

/// Name-resolution context for the AST → arena walk.
struct Ctx<'a> {
    param_indices: &'a HashMap<String, u8>,
}

/// Translate an AST node into the arena, resolving `let`-bound locals via
/// `locals`. The optimizer emits `let`-bindings (a [`Expr::Block`]) for shared
/// subexpressions; each binding maps to a single [`ExprId`], so the arena
/// faithfully preserves the discovered CSE as a DAG rather than duplicating
/// subtrees.
fn ast_to_arena_inner(
    expr: &Expr,
    ctx: &Ctx<'_>,
    locals: &mut HashMap<String, ExprId>,
    arena: &mut ExprArena,
) -> Result<ExprId, String> {
    match expr {
        Expr::Ident(ident) => {
            let name = ident.name.to_string();
            match name.as_str() {
                "X" => Ok(arena.push_var(0)),
                "Y" => Ok(arena.push_var(1)),
                "Z" => Ok(arena.push_var(2)),
                "W" => Ok(arena.push_var(3)),
                _ => {
                    if let Some(&id) = locals.get(&name) {
                        Ok(id)
                    } else if let Some(&idx) = ctx.param_indices.get(&name) {
                        Ok(arena.push_param(idx))
                    } else {
                        Err(format!("Unknown identifier: {}", name))
                    }
                }
            }
        }

        Expr::Literal(lit) => {
            if let Some(val) = extract_f64_from_lit(&lit.lit) {
                Ok(arena.push_const(val as f32))
            } else {
                Err("Non-numeric literal".to_string())
            }
        }

        Expr::Binary(binary) => {
            let lhs = ast_to_arena_inner(&binary.lhs, ctx, locals, arena)?;
            let rhs = ast_to_arena_inner(&binary.rhs, ctx, locals, arena)?;

            let op = match binary.op {
                BinaryOp::Add => OpKind::Add,
                BinaryOp::Sub => OpKind::Sub,
                BinaryOp::Mul => OpKind::Mul,
                BinaryOp::Div => OpKind::Div,
                BinaryOp::Lt => OpKind::Lt,
                BinaryOp::Le => OpKind::Le,
                BinaryOp::Gt => OpKind::Gt,
                BinaryOp::Ge => OpKind::Ge,
                BinaryOp::Eq => OpKind::Eq,
                BinaryOp::Ne => OpKind::Ne,
                // Mask combination: comparison results are canonical masks in
                // both tiers (all-ones SIMD lanes in the JIT, 1.0/0.0 in the
                // interpreter), so bitwise AND/OR is logical AND/OR exactly.
                BinaryOp::BitAnd => OpKind::BitAnd,
                BinaryOp::BitOr => OpKind::BitOr,
                _ => return Err(format!("Unsupported binary op: {:?}", binary.op)),
            };

            Ok(arena.push_binary(op, lhs, rhs))
        }

        Expr::Unary(unary) => {
            let operand = ast_to_arena_inner(&unary.operand, ctx, locals, arena)?;

            let op = match unary.op {
                UnaryOp::Neg => OpKind::Neg,
                UnaryOp::Not => return Err("Unsupported unary op: Not".to_string()),
            };

            Ok(arena.push_unary(op, operand))
        }

        Expr::MethodCall(call) => {
            let method = call.method.to_string();

            // `.at(x, y, z, w)` warped a manifold-typed macro param at a
            // call site. There are no manifold params: a kernel composes
            // `Kernel` values, and `Kernel::at` is the warp.
            if method == "at" {
                return Err(
                    ".at() inside a kernel body samples a manifold param, and there are none; \
                     compose Kernel values with Kernel::at instead"
                        .to_string(),
                );
            }

            let receiver = ast_to_arena_inner(&call.receiver, ctx, locals, arena)?;

            match (method.as_str(), call.args.len()) {
                // Arena expressions are values; `.clone()` (needed by the
                // combinator backend for non-Copy trees) is the identity here,
                // so one kernel body compiles under both backends.
                ("clone", 0) => Ok(receiver),

                // Unary methods - primitives
                ("sqrt", 0) => Ok(arena.push_unary(OpKind::Sqrt, receiver)),
                ("abs", 0) => Ok(arena.push_unary(OpKind::Abs, receiver)),
                ("neg", 0) => Ok(arena.push_unary(OpKind::Neg, receiver)),
                ("floor", 0) => Ok(arena.push_unary(OpKind::Floor, receiver)),
                ("ceil", 0) => Ok(arena.push_unary(OpKind::Ceil, receiver)),
                ("recip", 0) => Ok(arena.push_unary(OpKind::Recip, receiver)),
                ("rsqrt", 0) => Ok(arena.push_unary(OpKind::Rsqrt, receiver)),

                // Unary methods - transcendentals (lowered before JIT)
                ("sin", 0) => Ok(arena.push_unary(OpKind::Sin, receiver)),
                ("cos", 0) => Ok(arena.push_unary(OpKind::Cos, receiver)),
                ("tan", 0) => Ok(arena.push_unary(OpKind::Tan, receiver)),
                ("exp", 0) => Ok(arena.push_unary(OpKind::Exp, receiver)),
                ("exp2", 0) => Ok(arena.push_unary(OpKind::Exp2, receiver)),
                ("ln", 0) => Ok(arena.push_unary(OpKind::Ln, receiver)),
                ("log2", 0) => Ok(arena.push_unary(OpKind::Log2, receiver)),

                // Unary methods - inverse trigonometric
                ("atan", 0) => Ok(arena.push_unary(OpKind::Atan, receiver)),
                ("asin", 0) => Ok(arena.push_unary(OpKind::Asin, receiver)),
                ("acos", 0) => Ok(arena.push_unary(OpKind::Acos, receiver)),

                // Binary methods
                ("min", 1) => {
                    let arg = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Min, receiver, arg))
                }
                ("max", 1) => {
                    let arg = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Max, receiver, arg))
                }
                ("atan2", 1) => {
                    let arg = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Atan2, receiver, arg))
                }

                // Ternary methods
                ("mul_add", 2) => {
                    let b = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    let c = ast_to_arena_inner(&call.args[1], ctx, locals, arena)?;
                    Ok(arena.push_ternary(OpKind::MulAdd, receiver, b, c))
                }
                ("select", 2) => {
                    let if_true = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    let if_false = ast_to_arena_inner(&call.args[1], ctx, locals, arena)?;
                    Ok(arena.push_ternary(OpKind::Select, receiver, if_true, if_false))
                }
                // `clamp` is library, not a primitive: it denotes
                // `min(max(x, lo), hi)` and is built as that composition.
                ("clamp", 2) => {
                    let lo = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    let hi = ast_to_arena_inner(&call.args[1], ctx, locals, arena)?;
                    let floored = arena.push_binary(OpKind::Max, receiver, lo);
                    Ok(arena.push_binary(OpKind::Min, floored, hi))
                }

                // Comparison methods (emitted by e-graph extraction)
                ("lt", 1) => {
                    let a = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Lt, receiver, a))
                }
                ("le", 1) => {
                    let a = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Le, receiver, a))
                }
                ("gt", 1) => {
                    let a = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Gt, receiver, a))
                }
                ("ge", 1) => {
                    let a = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Ge, receiver, a))
                }
                ("eq", 1) => {
                    let a = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Eq, receiver, a))
                }
                ("ne", 1) => {
                    let a = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
                    Ok(arena.push_binary(OpKind::Ne, receiver, a))
                }

                _ => Err(format!("Unsupported method: {}", method)),
            }
        }

        // Derivative projections (V/DX/DY/DZ and the Hessian family) map to
        // `Dwrt` nodes: the runtime `lower_dwrt` pass (pixelflow-ir) rewrites
        // them into chain-rule arithmetic before codegen, replacing the
        // combinator backend's Jet2/Jet3 forward-mode evaluation. `V` is the
        // identity — every arena expression is already value-space.
        Expr::Call(call) => {
            let func = call.func.to_string();
            if call.args.len() != 1 {
                return Err(format!(
                    "Unsupported call: {}/{} (projections take one argument)",
                    func,
                    call.args.len()
                ));
            }
            let inner = ast_to_arena_inner(&call.args[0], ctx, locals, arena)?;
            match func.as_str() {
                "V" => Ok(inner),
                "DX" => Ok(push_dwrt(arena, inner, 0)),
                "DY" => Ok(push_dwrt(arena, inner, 1)),
                "DZ" => Ok(push_dwrt(arena, inner, 2)),
                "DXX" => {
                    let d = push_dwrt(arena, inner, 0);
                    Ok(push_dwrt(arena, d, 0))
                }
                "DXY" => {
                    let d = push_dwrt(arena, inner, 0);
                    Ok(push_dwrt(arena, d, 1))
                }
                "DYY" => {
                    let d = push_dwrt(arena, inner, 1);
                    Ok(push_dwrt(arena, d, 1))
                }
                _ => Err(format!("Unsupported call: {}", func)),
            }
        }

        // Parentheses are transparent - just recurse into the inner expression
        Expr::Paren(inner) => ast_to_arena_inner(inner, ctx, locals, arena),

        // Blocks carry the optimizer's CSE: each `let __n = <expr>;` binds a
        // shared subexpression to a single arena node, and the final expression
        // references those bindings by name.
        Expr::Block(block) => {
            for stmt in &block.stmts {
                match stmt {
                    crate::ast::Stmt::Let(let_stmt) => {
                        let id = ast_to_arena_inner(&let_stmt.init, ctx, locals, arena)?;
                        locals.insert(let_stmt.name.to_string(), id);
                    }
                    // A non-binding statement has no value to thread; evaluate
                    // it so any nested error surfaces, then discard the id.
                    crate::ast::Stmt::Expr(e) => {
                        let _ = ast_to_arena_inner(e, ctx, locals, arena)?;
                    }
                }
            }
            match &block.expr {
                Some(final_expr) => ast_to_arena_inner(final_expr, ctx, locals, arena),
                None => Err("Block has no final expression".to_string()),
            }
        }

        _ => Err("Unsupported expression type".to_string()),
    }
}

/// Generate runtime arena-construction code from macro AST.
///
/// Derivatives are eliminated here, at macro-expansion time, when possible:
/// see [`differentiate_in_optimizer`]. Kernels whose `Dwrt` nodes survive (an
/// op the e-graph cannot differentiate, or a saturation budget miss) emit the
/// `Dwrt`-carrying arena unchanged — the runtime `lower_dwrt` pass in
/// pixelflow-ir is the fallback tier and errors loudly only on genuinely
/// non-differentiable ops.
pub fn ast_to_runtime_arena(
    expr: &Expr,
    param_indices: &HashMap<String, u8>,
) -> Result<TokenStream, String> {
    let mut arena = ExprArena::new();
    let mut root = ast_to_arena(expr, param_indices, &mut arena)?;
    if let Some((optimized, optimized_root)) = differentiate_in_optimizer(&arena, root) {
        arena = optimized;
        root = optimized_root;
    }
    let nodes = arena.nodes_raw();
    let nary_children = arena.nary_children_raw();

    let node_tokens: Vec<TokenStream> = nodes
        .iter()
        .map(|node| match node {
            pixelflow_ir::arena::ExprNode::Var(i) => {
                quote! { ::pixelflow_core::__macro::ir::arena::ExprNode::Var(#i) }
            }
            // By bit pattern, not as a decimal literal: `quote`'s `f32`
            // impl goes through `Literal::f32_suffixed`, which asserts
            // `is_finite()` — and non-finite constants are ordinary here. A
            // true comparison mask is all-ones (`OpKind::mask`), which is
            // `BitAnd`'s monoid identity and therefore `all_over`'s seed, and
            // the folder now produces those. Bits also roundtrip exactly, with
            // no decimal-formatting question to get wrong.
            pixelflow_ir::arena::ExprNode::Const(v) => {
                let bits = v.to_bits();
                quote! { ::pixelflow_core::__macro::ir::arena::ExprNode::Const(f32::from_bits(#bits)) }
            }
            pixelflow_ir::arena::ExprNode::Param(i) => {
                quote! { ::pixelflow_core::__macro::ir::arena::ExprNode::Param(#i) }
            }
            // The `kernel!` macro has no buffer surface yet, so this is
            // unreachable in practice; fail loud rather than emit a node that
            // references a buffer table `from_raw` does not reconstruct.
            pixelflow_ir::arena::ExprNode::Buffer(b) => {
                panic!(
                    "kernel! produced ExprNode::Buffer({}) — lattice parameters are not wired \
                     into the compiler yet (KERNELS_AND_LATTICES.md M4)",
                    b.0
                )
            }
            pixelflow_ir::arena::ExprNode::Unary(op, child) => {
                let op_code = opkind_to_tokens(*op);
                let child = child.0;
                quote! { ::pixelflow_core::__macro::ir::arena::ExprNode::Unary(#op_code, ::pixelflow_core::__macro::ir::arena::ExprId(#child)) }
            }
            pixelflow_ir::arena::ExprNode::Binary(op, a, b) => {
                let op_code = opkind_to_tokens(*op);
                let a = a.0;
                let b = b.0;
                quote! { ::pixelflow_core::__macro::ir::arena::ExprNode::Binary(#op_code, ::pixelflow_core::__macro::ir::arena::ExprId(#a), ::pixelflow_core::__macro::ir::arena::ExprId(#b)) }
            }
            pixelflow_ir::arena::ExprNode::Ternary(op, a, b, c) => {
                let op_code = opkind_to_tokens(*op);
                let a = a.0;
                let b = b.0;
                let c = c.0;
                quote! { ::pixelflow_core::__macro::ir::arena::ExprNode::Ternary(#op_code, ::pixelflow_core::__macro::ir::arena::ExprId(#a), ::pixelflow_core::__macro::ir::arena::ExprId(#b), ::pixelflow_core::__macro::ir::arena::ExprId(#c)) }
            }
            pixelflow_ir::arena::ExprNode::Nary(op, start, len) => {
                let op_code = opkind_to_tokens(*op);
                quote! { ::pixelflow_core::__macro::ir::arena::ExprNode::Nary(#op_code, #start, #len) }
            }
        })
        .collect();

    let child_tokens: Vec<TokenStream> = nary_children
        .iter()
        .map(|id| {
            let id = id.0;
            quote! { ::pixelflow_core::__macro::ir::arena::ExprId(#id) }
        })
        .collect();

    let root = root.0;
    let tokens = quote! {{
        let __nodes = vec![#(#node_tokens),*];
        let __nary_children = vec![#(#child_tokens),*];
        let __arena = ::pixelflow_core::__macro::ir::arena::ExprArena::from_raw(__nodes, __nary_children);
        (__arena, ::pixelflow_core::__macro::ir::arena::ExprId(#root))
    }};
    Ok(tokens)
}

/// Push `Dwrt(expr, var)` — the variable index rides as a `Const` operand,
/// matching the encoding the e-graph `ChainRule` and `lower_dwrt` read.
fn push_dwrt(arena: &mut ExprArena, expr: ExprId, var: u8) -> ExprId {
    let v = arena.push_const(var as f32);
    arena.push_binary(OpKind::Dwrt, expr, v)
}

// ============================================================================
// Expansion-time differentiation (calculus in the optimizer)
// ============================================================================

/// Base `Var` index used to carry `Param(i)` leaves through the e-graph, which
/// has no Param representation: indices 0..4 are coordinates and 4..8 are
/// reduction indices, so params ride at 16+ and are mapped back after
/// extraction. To every rewrite rule they are opaque leaves — exactly the
/// semantics of an unbound scalar — and the chain rule gives them derivative
/// zero like any non-differentiation variable.
const PARAM_VAR_BASE: u8 = 16;

/// `Param(i) -> Var(16+i)`, the e-graph-side encoding [`differentiate_in_optimizer`]
/// round-trips through — factored out so the production path and its
/// measurement/regression tests build the identical encoded arena rather
/// than two copies of this mapping drifting apart.
fn encode_params_as_vars(arena: &ExprArena) -> ExprArena {
    use pixelflow_ir::arena::ExprNode;

    let encoded_nodes: Vec<ExprNode> = arena
        .nodes_raw()
        .iter()
        .map(|n| match n {
            ExprNode::Param(i) => {
                assert!(
                    usize::from(PARAM_VAR_BASE) + usize::from(*i) <= usize::from(u8::MAX),
                    "kernel has too many scalar params to encode for the e-graph"
                );
                ExprNode::Var(PARAM_VAR_BASE + i)
            }
            other => other.clone(),
        })
        .collect();
    ExprArena::from_raw(encoded_nodes, arena.nary_children_raw().to_vec())
}

/// Run the AOT-tier e-graph (full rule set: derivative + algebra + fusion)
/// over a `Dwrt`-carrying expansion arena, so derivatives are expanded *and
/// simplified* at macro-expansion time and the runtime never sees the
/// calculus.
///
/// Returns `None` when there is nothing to do (no `Dwrt`) or when the result
/// still contains a `Dwrt` (unsupported op / budget miss) — the caller then
/// emits the original arena and the runtime `lower_dwrt` pass takes over.
/// A budget miss is legitimate behavior, not a failure: the output's only
/// contract is that a `Some` is `Dwrt`-free and mathematically equivalent.
fn differentiate_in_optimizer(arena: &ExprArena, root: ExprId) -> Option<(ExprArena, ExprId)> {
    use pixelflow_search::egraph::Optimizer;

    if !contains_dwrt(arena) {
        return None;
    }

    let encoded = encode_params_as_vars(arena);

    // One saturation, full rule set: differentiation and algebra rewrite
    // TOGETHER — the point of the e-graph is that there is no pass ordering,
    // so the optimizer is free to simplify the differentiand before, during,
    // or after the chain rule fires (see the `derivative` module doc in
    // pixelflow-search). Same entry point as every other tier
    // (`pixelflow-compiler::optimize`'s macro tier and
    // `pixelflow-search::runtime::optimize_runtime_arena`'s runtime tier):
    // one `Optimizer` decides the rule set, the budget, the cost model and
    // the extractor, so a future policy change reaches this tier too instead
    // of being silently skipped. One entry point, no second copy.
    let mut optimizer = Optimizer::production();
    let mut eg = optimizer.egraph();
    // Declining is this tier's fallback, not an error: the runtime `lower_dwrt`
    // tier takes over. It replaces a hand-rolled pre-scan of every node that
    // existed only because `add_arena` panicked instead of declining.
    let root_class = pixelflow_search::egraph::insert(
        &encoded,
        root,
        &mut eg,
        pixelflow_search::egraph::Vocabulary::Templates,
    )
    .ok()?;
    let node_count = reachable_node_count(&encoded, root);
    let optimized = optimizer.run(&mut eg, root_class, node_count);

    extract_dwrt_free(&eg, root_class, &optimized)
}

/// Materialise `optimized` and undo [`encode_params_as_vars`]. `None` if a
/// `Dwrt` survived extraction — saturation stopped short of the chain rule's
/// fixed point (budget), or extraction picked one — and the runtime
/// `lower_dwrt` tier takes over.
fn extract_dwrt_free(
    eg: &pixelflow_search::egraph::EGraph,
    root_class: pixelflow_search::egraph::EClassId,
    optimized: &pixelflow_search::egraph::Optimized,
) -> Option<(ExprArena, ExprId)> {
    use pixelflow_ir::arena::ExprNode;

    let (out, out_root) = optimized.to_arena(eg, root_class);
    if contains_dwrt(&out) {
        return None;
    }

    // Var(16+i) -> Param(i).
    let decoded_nodes: Vec<ExprNode> = out
        .nodes_raw()
        .iter()
        .map(|n| match n {
            ExprNode::Var(i) if *i >= PARAM_VAR_BASE => ExprNode::Param(i - PARAM_VAR_BASE),
            other => other.clone(),
        })
        .collect();
    let decoded = ExprArena::from_raw(decoded_nodes, out.nary_children_raw().to_vec());
    Some((decoded, out_root))
}

/// Count of distinct nodes reachable from `root` — the size proxy
/// [`config_for_node_count`](pixelflow_search::egraph::config_for_node_count)
/// classifies into a saturation budget for arena-shaped input, matching the
/// proxy `pixelflow_search::runtime::optimize_runtime_arena` uses for the
/// runtime tier (an AST-node count serves the macro tier's own
/// `count_ast_nodes`, since there is no arena yet at that point).
fn reachable_node_count(arena: &ExprArena, root: ExprId) -> usize {
    let mut seen = vec![false; arena.nodes_raw().len()];
    let mut stack = vec![root];
    let mut n = 0usize;
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        n += 1;
        stack.extend(arena.children(id));
    }
    n
}

fn contains_dwrt(arena: &ExprArena) -> bool {
    arena.nodes_raw().iter().any(|n| {
        matches!(
            n,
            pixelflow_ir::arena::ExprNode::Binary(OpKind::Dwrt, _, _)
                | pixelflow_ir::arena::ExprNode::Unary(OpKind::Dwrt, _)
                | pixelflow_ir::arena::ExprNode::Ternary(OpKind::Dwrt, _, _, _)
        )
    })
}

/// Extract f64 from a syn::Lit.
fn extract_f64_from_lit(lit: &Lit) -> Option<f64> {
    match lit {
        Lit::Float(f) => f.base10_parse::<f64>().ok(),
        Lit::Int(i) => i.base10_parse::<i64>().ok().map(|v| v as f64),
        _ => None,
    }
}

/// Map OpKind to its token representation.
fn opkind_to_tokens(kind: OpKind) -> TokenStream {
    match kind {
        OpKind::Add => quote! { ::pixelflow_core::__macro::ir::OpKind::Add },
        OpKind::Sub => quote! { ::pixelflow_core::__macro::ir::OpKind::Sub },
        OpKind::Mul => quote! { ::pixelflow_core::__macro::ir::OpKind::Mul },
        OpKind::Div => quote! { ::pixelflow_core::__macro::ir::OpKind::Div },
        OpKind::Neg => quote! { ::pixelflow_core::__macro::ir::OpKind::Neg },
        OpKind::Sqrt => quote! { ::pixelflow_core::__macro::ir::OpKind::Sqrt },
        OpKind::Rsqrt => quote! { ::pixelflow_core::__macro::ir::OpKind::Rsqrt },
        OpKind::Recip => quote! { ::pixelflow_core::__macro::ir::OpKind::Recip },
        OpKind::Abs => quote! { ::pixelflow_core::__macro::ir::OpKind::Abs },
        OpKind::Min => quote! { ::pixelflow_core::__macro::ir::OpKind::Min },
        OpKind::Max => quote! { ::pixelflow_core::__macro::ir::OpKind::Max },
        OpKind::MulAdd => quote! { ::pixelflow_core::__macro::ir::OpKind::MulAdd },
        OpKind::Sin => quote! { ::pixelflow_core::__macro::ir::OpKind::Sin },
        OpKind::Cos => quote! { ::pixelflow_core::__macro::ir::OpKind::Cos },
        OpKind::Atan => quote! { ::pixelflow_core::__macro::ir::OpKind::Atan },
        OpKind::Asin => quote! { ::pixelflow_core::__macro::ir::OpKind::Asin },
        OpKind::Acos => quote! { ::pixelflow_core::__macro::ir::OpKind::Acos },
        OpKind::Atan2 => quote! { ::pixelflow_core::__macro::ir::OpKind::Atan2 },
        OpKind::Tan => quote! { ::pixelflow_core::__macro::ir::OpKind::Tan },
        OpKind::Exp => quote! { ::pixelflow_core::__macro::ir::OpKind::Exp },
        OpKind::Exp2 => quote! { ::pixelflow_core::__macro::ir::OpKind::Exp2 },
        OpKind::Ln => quote! { ::pixelflow_core::__macro::ir::OpKind::Ln },
        OpKind::Log2 => quote! { ::pixelflow_core::__macro::ir::OpKind::Log2 },
        OpKind::Log10 => quote! { ::pixelflow_core::__macro::ir::OpKind::Log10 },
        OpKind::Pow => quote! { ::pixelflow_core::__macro::ir::OpKind::Pow },
        OpKind::Floor => quote! { ::pixelflow_core::__macro::ir::OpKind::Floor },
        OpKind::Ceil => quote! { ::pixelflow_core::__macro::ir::OpKind::Ceil },
        OpKind::Round => quote! { ::pixelflow_core::__macro::ir::OpKind::Round },
        OpKind::Lt => quote! { ::pixelflow_core::__macro::ir::OpKind::Lt },
        OpKind::Le => quote! { ::pixelflow_core::__macro::ir::OpKind::Le },
        OpKind::Gt => quote! { ::pixelflow_core::__macro::ir::OpKind::Gt },
        OpKind::Ge => quote! { ::pixelflow_core::__macro::ir::OpKind::Ge },
        OpKind::Eq => quote! { ::pixelflow_core::__macro::ir::OpKind::Eq },
        OpKind::Ne => quote! { ::pixelflow_core::__macro::ir::OpKind::Ne },
        OpKind::Select => quote! { ::pixelflow_core::__macro::ir::OpKind::Select },
        // Mask combination (canonical masks in both tiers).
        OpKind::BitAnd => quote! { ::pixelflow_core::__macro::ir::OpKind::BitAnd },
        OpKind::BitOr => quote! { ::pixelflow_core::__macro::ir::OpKind::BitOr },
        // Lowered at runtime by pixelflow-ir's `lower_dwrt` before codegen.
        OpKind::Dwrt => quote! { ::pixelflow_core::__macro::ir::OpKind::Dwrt },
        _ => panic!("Unsupported OpKind for JIT: {:?}", kind),
    }
}

#[cfg(test)]
mod expansion_derivative_tests {
    use super::*;
    use pixelflow_ir::arena::ExprNode;
    use pixelflow_ir::binding::BindingTable;
    use pixelflow_ir::eval::eval_scalar;
    use pixelflow_ir::passes::lower_dwrt_owned;

    /// The optimizer's contract, checked differentially: whatever it returns
    /// for a `Dwrt`-carrying arena must agree numerically with the runtime
    /// `lower_dwrt` tier — two independent implementations of the same
    /// calculus checking each other. `None` (budget miss → fallback) is
    /// always legitimate and asserts nothing; a `Some` must be honest:
    /// `Dwrt`-free, params round-tripped, and mathematically equivalent.
    /// Whether the output is also *cheaper* is a cost-model concern for the
    /// bench harness (the `pixelflow-pipeline` bench bins), not a unit test.
    ///
    /// These tests build arenas by hand and call the private
    /// `differentiate_in_optimizer` directly (STYLE.md "Test Public API"
    /// exception): the claim under test is that this expansion-time pass
    /// agrees numerically with the independent runtime `lower_dwrt` tier,
    /// which is only checkable by comparing the two tiers' arena-level
    /// output directly — the public `ast_to_runtime_arena` entry point only
    /// exposes generated `TokenStream`, not the intermediate arena needed for
    /// this differential check.
    fn assert_matches_runtime_tier(a: &ExprArena, root: ExprId, params: &[f32], pts: &[[f32; 4]]) {
        let Some((out, out_root)) = differentiate_in_optimizer(a, root) else {
            return; // fallback tier's job; nothing claimed, nothing to check
        };
        RuntimeTierReference {
            arena: a,
            root,
            params,
            pts,
        }
        .assert_agrees(&out, out_root);
    }

    /// The original `Dwrt`-carrying arena plus the params and sample points
    /// the optimizer's output is checked at — the `Some` half of
    /// [`assert_matches_runtime_tier`], for callers that already hold that
    /// output (the deterministic-saturation tests below, which bypass the
    /// wall-clock budget).
    struct RuntimeTierReference<'a> {
        arena: &'a ExprArena,
        root: ExprId,
        params: &'a [f32],
        pts: &'a [[f32; 4]],
    }

    impl RuntimeTierReference<'_> {
        fn assert_agrees(&self, out: &ExprArena, out_root: ExprId) {
            let (a, root, params, pts) = (self.arena, self.root, self.params, self.pts);
            assert!(
                !super::contains_dwrt(out),
                "Some(..) must be Dwrt-free — that is the claim it makes"
            );
            assert!(
                !out.nodes_raw()
                    .iter()
                    .any(|n| matches!(n, ExprNode::Var(i) if *i >= PARAM_VAR_BASE)),
                "encoded param Var leaked through the round-trip undecoded"
            );

            // Reference: substitute params into the ORIGINAL arena and run the
            // runtime lowering tier on it.
            let mut reference = a.clone();
            let ref_root = reference.substitute_params(root, params);
            let (ref_arena, ref_root) =
                lower_dwrt_owned(&reference, ref_root).expect("runtime tier lowers the reference");

            let mut got = out.clone();
            let got_root = got.substitute_params(out_root, params);

            for p in pts {
                let want = eval_scalar(&ref_arena, ref_root, p, &BindingTable::empty());
                let g = eval_scalar(&got, got_root, p, &BindingTable::empty());
                let tol = 1e-3 * want.abs().max(1.0);
                assert!(
                    (g - want).abs() <= tol,
                    "at {p:?}: optimizer={g}, runtime tier={want} (tol {tol})"
                );
            }
        }
    }

    /// d/dx (p0 · √(x² + y²)) — a scalar param inside the differentiand
    /// exercises the Param ↔ Var(16+) round-trip alongside the calculus.
    #[test]
    fn param_derivative_matches_runtime_tier() {
        let mut a = ExprArena::new();
        let p0 = a.push_param(0);
        let x = a.push_var(0);
        let y = a.push_var(1);
        let x2 = a.push_binary(OpKind::Mul, x, x);
        let y2 = a.push_binary(OpKind::Mul, y, y);
        let sum = a.push_binary(OpKind::Add, x2, y2);
        let dist = a.push_unary(OpKind::Sqrt, sum);
        let e = a.push_binary(OpKind::Mul, p0, dist);
        let root = push_dwrt(&mut a, e, 0);

        assert_matches_runtime_tier(
            &a,
            root,
            &[2.0],
            &[
                [3.0, 4.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [-2.0, 5.0, 0.0, 0.0],
            ],
        );
    }

    /// The piecewise font ramp: min/max over a gradient-normalized ratio,
    /// with two shared `Dwrt` sites.
    #[test]
    fn piecewise_ramp_matches_runtime_tier() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let d = a.push_binary(OpKind::Sub, x, y);
        let dx = push_dwrt(&mut a, d, 0);
        let dy = push_dwrt(&mut a, d, 1);
        let dx2 = a.push_binary(OpKind::Mul, dx, dx);
        let dy2 = a.push_binary(OpKind::Mul, dy, dy);
        let s = a.push_binary(OpKind::Add, dx2, dy2);
        let grad = a.push_unary(OpKind::Sqrt, s);
        let ratio = a.push_binary(OpKind::Div, d, grad);
        let zero = a.push_const(0.0);
        let one = a.push_const(1.0);
        let mx = a.push_binary(OpKind::Max, ratio, zero);
        let root = a.push_binary(OpKind::Min, mx, one);

        assert_matches_runtime_tier(
            &a,
            root,
            &[],
            &[
                [2.0, 1.0, 0.0, 0.0],
                [0.5, 0.2, 0.0, 0.0],
                [-1.0, 1.0, 0.0, 0.0],
            ],
        );
    }

    /// No `Dwrt` -> nothing to do; the caller keeps the original arena (and
    /// the AST-level optimizer's existing output is not perturbed).
    #[test]
    fn differentiate_in_optimizer_returns_none_when_arena_has_no_dwrt() {
        let mut a = ExprArena::new();
        let x = a.push_var(0);
        let y = a.push_var(1);
        let root = a.push_binary(OpKind::Add, x, y);
        assert!(differentiate_in_optimizer(&a, root).is_none());
    }

    /// X - ((Y - y0) * dx_over_dy + x0), the crossing distance every
    /// `AnalyticalLine`/`AnalyticalQuad` coverage ramp in
    /// `pixelflow-graphics::fonts::ttf_curve_analytical` is built from, with
    /// `y0`/`dx_over_dy`/`x0` as scalar params (matching the real kernel's
    /// `Param` slots).
    fn winding_crossing_distance(a: &mut ExprArena) -> ExprId {
        let x = a.push_var(0);
        let y = a.push_var(1);
        let y0 = a.push_param(1);
        let dx_over_dy = a.push_param(2);
        let x0 = a.push_param(0);
        let y_sub_y0 = a.push_binary(OpKind::Sub, y, y0);
        let scaled = a.push_binary(OpKind::Mul, y_sub_y0, dx_over_dy);
        let shifted = a.push_binary(OpKind::Add, scaled, x0);
        a.push_binary(OpKind::Sub, x, shifted)
    }

    /// The gradient-normalized coverage ramp from `AnalyticalLine::kernel`
    /// (ttf_curve_analytical.rs:106-129), `d`'s crossing distance through
    /// `clamp(d / (‖∇d‖ + ε) + 0.5, 0, 1)` — everything downstream of `DX`/`DY`
    /// in the real kernel, minus the `in_y` boundary mask (see
    /// [`winding_kernel_dwrt_bails_to_runtime_tier`] for why that mask
    /// matters). `d` is shared three ways (`DX`, `DY`, the ratio numerator),
    /// so this is exactly the CSE case DAG extraction exists for.
    fn winding_ramp_core(a: &mut ExprArena) -> ExprId {
        let d = winding_crossing_distance(a);
        let min_grad = a.push_param(5);
        let dx = push_dwrt(a, d, 0);
        let dy = push_dwrt(a, d, 1);
        let dx2 = a.push_binary(OpKind::Mul, dx, dx);
        let dy2 = a.push_binary(OpKind::Mul, dy, dy);
        let sum2 = a.push_binary(OpKind::Add, dx2, dy2);
        let grad = a.push_unary(OpKind::Sqrt, sum2);
        let denom = a.push_binary(OpKind::Add, grad, min_grad);
        let ratio = a.push_binary(OpKind::Div, d, denom);
        let half = a.push_const(0.5);
        let plus_half = a.push_binary(OpKind::Add, ratio, half);
        let zero = a.push_const(0.0);
        let one = a.push_const(1.0);
        let clamped_lo = a.push_binary(OpKind::Max, plus_half, zero);
        a.push_binary(OpKind::Min, clamped_lo, one)
    }

    /// The full `AnalyticalLine::kernel` shape: [`winding_ramp_core`] gated
    /// by `in_y = (Y >= y_min) & (Y < y_max)`.
    ///
    /// **Finding**: `&` lexes to `OpKind::BitAnd`
    /// (`pixelflow-compiler/src/ast.rs:255,291`), and `BitAnd` has no
    /// rewrite-rule `Op` — `ops::op_from_kind` returns `None` for it by
    /// design, since bit-manip primitives are a *lowering* output, never an
    /// e-graph input (`pixelflow-search/src/egraph/ops.rs`). Every real
    /// `Dwrt`-bearing `kernel!` body in this codebase gates its ramp
    /// with exactly this kind of boundary mask (`AnalyticalLine`,
    /// `AnalyticalQuad`'s `in_t`/`valid_plus`/`valid_minus` — see the DX/DY
    /// grep in the PR description), so `differentiate_in_optimizer`'s
    /// `representable` check has always rejected the flagship glyph kernel
    /// and bailed to `None` — both before and after this unification. Its
    /// `Dwrt` resolution actually happens one tier down, in
    /// `pixelflow_search::runtime::optimize_runtime_arena` (via
    /// `Kernel::from_parts` → `Lattice::bake`), which already ran the ONE
    /// production policy before this PR. This unification changes behavior
    /// for `Dwrt` bodies that skip the representable gate — the two
    /// `*_matches_runtime_tier` tests above, and any future `kernel!`
    /// body with derivatives but no boolean mask — not for glyph rendering
    /// itself.
    fn winding_kernel_arena() -> (ExprArena, ExprId) {
        let mut a = ExprArena::new();
        let y = a.push_var(1);
        let y_min = a.push_param(3);
        let y_max = a.push_param(4);
        let in_y_lo = a.push_binary(OpKind::Ge, y, y_min);
        let in_y_hi = a.push_binary(OpKind::Lt, y, y_max);
        let in_y = a.push_binary(OpKind::BitAnd, in_y_lo, in_y_hi);
        let coverage = winding_ramp_core(&mut a);
        let dir = a.push_param(6);
        let scaled_cov = a.push_binary(OpKind::Mul, coverage, dir);
        let zero = a.push_const(0.0);
        let root = a.push_ternary(OpKind::Select, in_y, scaled_cov, zero);
        (a, root)
    }

    #[test]
    fn winding_kernel_dwrt_bails_to_runtime_tier() {
        let (a, root) = winding_kernel_arena();
        assert!(
            differentiate_in_optimizer(&a, root).is_none(),
            "the real glyph winding kernel's `in_y` BitAnd mask should keep tripping \
             the representable gate — see this fn's doc comment"
        );
    }

    /// Params for [`winding_ramp_core`] / [`winding_kernel_arena`]:
    /// `x0`, `y0`, `dx_over_dy`, `y_min`, `y_max`, `min_grad`, `dir`.
    const WINDING_PARAMS: [f32; 7] = [0.3, -0.2, 0.7, -1.0, 1.0, 1e-3, 1.0];
    const WINDING_POINTS: [[f32; 4]; 4] = [
        [0.0, 0.0, 0.0, 0.0],
        [0.4, 0.1, 0.0, 0.0],
        [-0.3, 0.5, 0.0, 0.0],
        [1.2, -0.7, 0.0, 0.0],
    ];

    /// Wall-clock ceiling for the tests below, standing in for the tier's own
    /// deadline.
    ///
    /// The deadline is the one part of the production budget these tests
    /// deliberately do not reproduce. `cargo test` builds this crate
    /// unoptimized and shares the machine with the rest of the workspace, so
    /// asserting anything under `rapid`'s 50ms (or `classical`'s 200ms) would
    /// pin the machine's load, not the policy — and a load-dependent
    /// assertion is a flake, which is worse than no assertion. Every other
    /// dimension (round cap, class cap, rule set, extraction policy) is the
    /// production one, so what these tests measure is whether *those* suffice.
    ///
    /// A deadline miss in production stays legitimate: it is the documented
    /// graceful fallback to the runtime `lower_dwrt` tier — see
    /// [`differentiate_in_optimizer`].
    /// Saturate `root` under the production configuration and extract.
    /// Mirrors [`differentiate_in_optimizer`] exactly — same param encoding,
    /// same `Optimizer::production()`.
    ///
    /// This helper used to substitute a 120-second ceiling for the tier's
    /// `hard_timeout`, because `cargo test` builds this crate unoptimized and
    /// shares the machine, so asserting anything under `rapid`'s 50 ms would
    /// have pinned the machine's load rather than the policy — a
    /// load-dependent assertion, i.e. a flake. `Budget` takes no clock, so
    /// there is nothing left to substitute: the production configuration *is*
    /// the untimed one, and these tests now exercise it unmodified.
    fn optimize_under_production_budget(
        a: &ExprArena,
        root: ExprId,
    ) -> (
        pixelflow_search::egraph::EGraph,
        pixelflow_search::egraph::EClassId,
        pixelflow_search::egraph::Optimized,
    ) {
        use pixelflow_search::egraph::Optimizer;

        let node_count = super::reachable_node_count(a, root);
        let encoded = super::encode_params_as_vars(a);

        let mut optimizer = Optimizer::production();
        let mut eg = optimizer.egraph();
        let root_class = pixelflow_search::egraph::insert(
            &encoded,
            root,
            &mut eg,
            pixelflow_search::egraph::Vocabulary::Templates,
        )
        .expect("insert into e-graph");
        let classes_before = eg.num_classes();
        let started = std::time::Instant::now();
        let optimized = optimizer.run(&mut eg, root_class, node_count);
        eprintln!(
            "[production budget] node_count={node_count} limits={:?} iterations={} \
             applications={} total_unions={} classes {classes_before}->{} stop={:?} elapsed={:?}",
            optimized.stats.limits,
            optimized.stats.iterations,
            optimized.stats.applications,
            optimized.stats.unions,
            eg.num_classes(),
            optimized.stats.stop,
            started.elapsed(),
        );
        (eg, root_class, optimized)
    }

    /// [`winding_ramp_core`] plus the assertion that it lands in the `rapid`
    /// tier, so the tests below cannot silently start measuring a different
    /// budget if the body or the tier thresholds change.
    fn winding_ramp_in_rapid_tier() -> (ExprArena, ExprId) {
        use pixelflow_search::egraph::{SaturationConfig, config_for_node_count};

        let mut a = ExprArena::new();
        let root = winding_ramp_core(&mut a);
        let node_count = super::reachable_node_count(&a, root);
        let tier = config_for_node_count(node_count);
        let rapid = SaturationConfig::rapid();
        assert!(
            (11..=50).contains(&node_count)
                && tier.max_classes == rapid.max_classes
                && tier.max_iterations == rapid.max_iterations,
            "winding_ramp_core should land in the rapid tier, got {node_count} nodes -> {tier:?}"
        );
        (a, root)
    }

    /// The `rapid` tier's iteration and class caps (50 / 2000) are enough for
    /// the real glyph winding ramp's `Dwrt` to reach the chain rule's fixed
    /// point — pinned with wall-clock taken out of the picture. Measured
    /// (2026-09-01): converges in 12 iterations at ~1400 classes, so 2000 is
    /// headroom, not a constraint; when `differentiate_in_optimizer` does
    /// bail on this body it is the tier's 50ms deadline on a loaded or
    /// unoptimized (proc-macro) build, and the runtime `lower_dwrt` tier is
    /// the documented fallback. Extraction is checked under the production
    /// policy (the static latency prior, spelled explicitly so the test
    /// does not depend on `env_extraction_policy`'s selection).
    #[test]
    fn rapid_caps_resolve_winding_ramp_dwrt_under_static_policy() {
        let (a, root) = winding_ramp_in_rapid_tier();
        let (eg, root_class, optimized) = optimize_under_production_budget(&a, root);

        let (out, out_root) = super::extract_dwrt_free(&eg, root_class, &optimized)
            .expect("Static: Dwrt must not survive a converged rapid-tier saturation");
        eprintln!("[Static] extracted(dag) node_count={}", out.len());
        RuntimeTierReference {
            arena: &a,
            root,
            params: &WINDING_PARAMS,
            pts: &WINDING_POINTS,
        }
        .assert_agrees(&out, out_root);
    }

    /// The same body through the production *policy seam* —
    /// [`env_extraction_policy`], the one place extraction policy is chosen —
    /// rather than a policy this test names itself.
    ///
    /// The derivative MUST be produced: a `None` here means the tier's caps
    /// or the selected policy failed to resolve the chain rule on a real
    /// glyph kernel, which is the regression this test exists to catch.
    /// Only the deadline is relaxed, to [`UNTIMED_CEILING`] — see its doc for
    /// why asserting under `rapid`'s 50ms would pin the machine instead.
    ///
    /// It also pins that the path runs to completion: the DAG extraction's
    /// children-cost sum used to overflow on a `Dwrt`-bearing graph
    /// (`extract_dag`, `usize::MAX / 4` sentinel), which surfaces here as a
    /// panic, never as a `None`.
    #[test]
    fn winding_ramp_core_takes_production_policy() {
        let (a, root) = winding_ramp_in_rapid_tier();
        let (eg, root_class, optimized) = optimize_under_production_budget(&a, root);

        let (out, out_root) = super::extract_dwrt_free(&eg, root_class, &optimized)
            .expect("production policy: Dwrt must not survive a converged rapid-tier saturation");
        eprintln!(
            "[production] converged; extracted(dag) node_count={}",
            out.len()
        );
        RuntimeTierReference {
            arena: &a,
            root,
            params: &WINDING_PARAMS,
            pts: &WINDING_POINTS,
        }
        .assert_agrees(&out, out_root);
    }

    /// Two independent ramp cores summed — an `AnalyticalQuad`-shaped
    /// stand-in (two `DX`/`DY` sites, 53 reachable nodes, so the
    /// `classical` tier rather than `rapid`). Same contract as
    /// [`winding_ramp_core_takes_production_policy`].
    fn quad_shaped_core(a: &mut ExprArena) -> ExprId {
        let c1 = winding_ramp_core(a);
        let c2 = winding_ramp_core(a);
        a.push_binary(OpKind::Add, c1, c2)
    }

    #[test]
    fn quad_shaped_core_takes_production_policy() {
        let mut a = ExprArena::new();
        let root = quad_shaped_core(&mut a);
        let node_count = super::reachable_node_count(&a, root);
        assert!(
            node_count > 50,
            "quad_shaped_core should land in the classical tier (51+ nodes), got {node_count}"
        );

        let (eg, root_class, optimized) = optimize_under_production_budget(&a, root);
        let (out, out_root) = super::extract_dwrt_free(&eg, root_class, &optimized).expect(
            "production policy: Dwrt must not survive a converged classical-tier saturation",
        );
        RuntimeTierReference {
            arena: &a,
            root,
            params: &WINDING_PARAMS,
            pts: &WINDING_POINTS,
        }
        .assert_agrees(&out, out_root);
    }
}

/// Production saturation telemetry, stage 2 of 2 (the `Dwrt` macro-time
/// e-graph; stage 1 is `pixelflow-search/src/runtime.rs`'s
/// `production_telemetry`, docs/results/2026-09-01-production-saturation-telemetry.md).
///
/// [`differentiate_in_optimizer`] (this file, above) is production's *other*
/// saturation site: it runs at `kernel!` macro-expansion time, on
/// every kernel whose body contains `Dwrt` (`DX`/`DY`), with a budget that is
/// NOT `config_for_node_count`-tiered — it is the hardcoded "standard
/// optimizer budget" `eg.saturate()` = 100 iterations / 10,000 classes /
/// 500 ms (`graph.rs:811-813`), called at `:722` above with the result
/// discarded exactly like `optimize_runtime_arena_uncached` discards its
/// `SaturationResult`. core-term's one reachable user of this path is the
/// glyph winding segment's coverage kernel
/// (`pixelflow-graphics/src/fonts/ttf_curve_analytical.rs:106-129`, a
/// `kernel!` whose `grad` term differentiates `d` through `DX`/`DY`).
///
/// This module drives the *exact* front-end pipeline the `kernel!`
/// proc-macro fn runs (`lib.rs:223-238`: `parser::parse` → `sema::analyze` →
/// `optimize::optimize` — a FIRST, algebra-only e-graph, AST-level, sized by
/// `config_for_node_count` on raw AST node count, distinct from the Dwrt
/// e-graph and measured here too for completeness) on the winding kernel's
/// literal closure source, reaching the same `AnalyzedKernel` that
/// `jit_backend::emit_kernel` would feed to
/// `ast_to_runtime_arena` (`:468-482`) — then replays
/// `differentiate_in_optimizer`'s own body verbatim (same param encoding,
/// same rule set, same extraction) with `saturate_with_limits` standing in
/// for `saturate()` so `SaturationStats` and provenance survive instead of
/// being dropped.
///
/// There is exactly one kernel to measure here, not a sweep: every glyph
/// segment shares this one closure body (only the 7 scalar params — the
/// endpoints and slope — differ per segment, substituted after arena
/// construction), so the arena `differentiate_in_optimizer` sees is
/// structurally identical for every call site. `param_indices` are threaded
/// through as arena `Param` leaves either way (see the encode step below),
/// so the measured arena does not depend on which segment's numbers would
/// eventually be substituted.
///
/// Nothing here changes production behavior: no signature, no visibility,
/// no code outside `#[cfg(test)]`.
#[cfg(test)]
mod production_telemetry {
    use super::*;
    use pixelflow_search::egraph::{
        CostModel, Optimizer, SaturationStats, SaturationStop, extract,
    };
    use std::time::{Duration, Instant};

    /// Verbatim closure body from
    /// `pixelflow-graphics/src/fonts/ttf_curve_analytical.rs:106-118` (the
    /// `kernel!` argument, up through the closing `}` of the body —
    /// the call-site arguments on `:119-127` are runtime values, irrelevant
    /// to the arena's shape). `crate::parser::parse` expects exactly the
    /// tokens between the macro's parens, per its own doc example
    /// (`lib.rs:217`: `kernel!(|cx: f32, r: f32| (X - cx) * r)`).
    const WINDING_KERNEL_SRC: &str = r#"
        |x0: f32,
         y0: f32,
         dx_over_dy: f32,
         dir: f32,
         y_min: f32,
         y_max: f32,
         min_grad: f32|
         -> Field {
            let in_y = (Y >= y_min) & (Y < y_max);
            let d = X - ((Y - y0) * dx_over_dy + x0);
            let grad = (DX(d.clone()) * DX(d.clone()) + DY(d.clone()) * DY(d.clone())).sqrt();
            let coverage = (V(d) / (grad + V(min_grad)) + V(0.5))
                .max(V(0.0))
                .min(V(1.0));
            in_y.select(coverage * V(dir), V(0.0))
        }
    "#;

    fn reachable_count(arena: &ExprArena, root: ExprId) -> usize {
        let len = arena.nodes_raw().len();
        let mut seen = vec![false; len];
        let mut stack = vec![root];
        let mut n = 0usize;
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            n += 1;
            stack.extend(arena.children(id));
        }
        n
    }

    /// Latency-prior DAG cost (leaves free, each reachable op counted once) —
    /// NOT `extract`'s own returned cost, which (like `extract_dag`'s
    /// `total_cost`) is a TREE cost and pays a shared subterm once per use.
    /// Since #1111 both are costed from the repaired choices, so neither
    /// carries `CYCLE_COST` inflation any more; the tree/DAG difference is
    /// what remains, and it is the whole gap on a sharing-heavy kernel. See
    /// `runtime.rs`'s `arena_cost` for the sibling of this function.
    fn arena_cost(arena: &ExprArena, root: ExprId, costs: &CostModel) -> usize {
        use pixelflow_ir::arena::ExprNode;
        let len = arena.nodes_raw().len();
        let mut seen = vec![false; len];
        let mut stack = vec![root];
        let mut total = 0usize;
        while let Some(id) = stack.pop() {
            if core::mem::replace(&mut seen[id.0 as usize], true) {
                continue;
            }
            let kind = match arena.node(id) {
                ExprNode::Var(_) | ExprNode::Const(_) | ExprNode::Buffer(_) => None,
                ExprNode::Unary(k, _)
                | ExprNode::Binary(k, _, _)
                | ExprNode::Ternary(k, _, _, _) => Some(*k),
                other @ (ExprNode::Param(_) | ExprNode::Nary(..)) => {
                    panic!(
                        "winding-kernel Dwrt arena contains {other:?}, unexpected pre-extraction"
                    )
                }
            };
            if let Some(k) = kind {
                total += costs.cost(k);
            }
            stack.extend(arena.children(id));
        }
        total
    }

    struct DwrtRun {
        stop: SaturationStop,
        iterations: usize,
        total_unions: usize,
        classes_after: usize,
        applications: usize,
        elapsed: Duration,
        cost: usize,
        dwrt_survived: bool,
    }

    /// Ops in `arena` that the e-graph cannot represent — the same test
    /// `differentiate_in_optimizer` runs at `:687-694` before ever building
    /// an `EGraph`, replicated (not called) so the caller can report *which*
    /// op blocked it rather than the production function's plain `None`.
    /// Empty means representable.
    fn unrepresentable_ops(arena: &ExprArena) -> Vec<OpKind> {
        use pixelflow_ir::arena::ExprNode;
        let mut bad: Vec<OpKind> = arena
            .nodes_raw()
            .iter()
            .filter_map(|n| match n {
                ExprNode::Unary(op, _)
                | ExprNode::Binary(op, _, _)
                | ExprNode::Ternary(op, _, _, _) => {
                    (pixelflow_search::egraph::ops::op_from_kind(*op).is_none()).then_some(*op)
                }
                ExprNode::Buffer(_) => Some(OpKind::Buffer), // ir_bridge.rs:690: Buffer => false unconditionally
                ExprNode::Var(_) | ExprNode::Const(_) | ExprNode::Param(_) => None,
                ExprNode::Nary(op, _, _) => {
                    (pixelflow_search::egraph::ops::op_from_kind(*op).is_none()).then_some(*op)
                }
            })
            .collect();
        bad.sort_by_key(|k| format!("{k:?}"));
        bad.dedup();
        bad
    }

    /// [`differentiate_in_optimizer`]'s own body (`:661-732`), replayed with
    /// budget as parameters and `saturate_with_limits` in place of
    /// `saturate()` so the discarded `SaturationStats` and the (unconditional,
    /// per `graph.rs:137,157`) provenance counters survive. Encode/decode and
    /// rule set are copied verbatim; nothing in the production function is
    /// called or modified. Returns `Err(blocking ops)` exactly where
    /// production's `representable` guard (`:687-694`) would return `None`
    /// WITHOUT ever constructing an `EGraph` — i.e. zero saturation, not a
    /// truncated one.
    fn run_dwrt_egraph(
        arena: &ExprArena,
        root: ExprId,
        max_iterations: usize,
        max_classes: usize,
        timeout: Duration,
    ) -> Result<DwrtRun, Vec<OpKind>> {
        use pixelflow_ir::arena::ExprNode;
        let bad = unrepresentable_ops(arena);
        if !bad.is_empty() {
            return Err(bad);
        }

        // ir_bridge.rs:697-712 — Param(i) -> Var(16+i) encoding.
        let encoded_nodes: Vec<ExprNode> = arena
            .nodes_raw()
            .iter()
            .map(|n| match n {
                ExprNode::Param(i) => {
                    assert!(
                        usize::from(PARAM_VAR_BASE) + usize::from(*i) <= usize::from(u8::MAX),
                        "too many scalar params to encode"
                    );
                    ExprNode::Var(PARAM_VAR_BASE + i)
                }
                other => other.clone(),
            })
            .collect();
        let encoded = ExprArena::from_raw(encoded_nodes, arena.nary_children_raw().to_vec());

        // ir_bridge.rs:734-735, with saturate_with_limits standing in for the
        // Optimizer's own budgeted run: this measurement varies the caps
        // directly, which `Optimizer::run` does not expose per call.
        // `standard_rules()` was replaced by `Optimizer::production()` in
        // pixelflow-search#1108; the rule set is the same one.
        let optimizer = Optimizer::production();
        let mut eg = optimizer.egraph();
        let root_class = pixelflow_search::egraph::insert(
            &encoded,
            root,
            &mut eg,
            pixelflow_search::egraph::Vocabulary::Templates,
        )
        .expect("insert into e-graph");
        let started = Instant::now();
        let stats: SaturationStats = eg.saturate_with_limits(max_iterations, max_classes, timeout);
        let elapsed = started.elapsed();

        // ir_bridge.rs:724 — extract's own returned cost carries the
        // cycle-penalty caveat above; recompute from the extracted arena.
        let costs = CostModel::latency_prior();
        let (out, out_root, _extract_reported_cost) = extract(&eg, root_class, &costs);
        let dwrt_survived = contains_dwrt(&out);
        let cost = if dwrt_survived {
            // A Dwrt-carrying "extraction" is not code arena_cost can price
            // (Dwrt has no CostModel entry) — the real fallback (:733)
            // recompiles the ORIGINAL arena unresolved, at the runtime tier;
            // report the pre-extraction node count's cost as N/A via 0 and
            // let `dwrt_survived` carry the signal instead of a fabricated number.
            0
        } else {
            arena_cost(&out, out_root, &costs)
        };

        Ok(DwrtRun {
            stop: stats.stop,
            iterations: stats.iterations,
            total_unions: stats.total_unions,
            classes_after: eg.num_classes(),
            // The budget's own counter (unconditional — never requires
            // `provenance-journal`, which pixelflow-compiler must never
            // depend on), not the provenance journal's length; they agree
            // whenever the journal is being kept at all.
            applications: eg.application_count() as usize,
            elapsed,
            cost,
            dwrt_survived,
        })
    }

    /// Drives `parser::parse` -> `sema::analyze` -> `optimize::optimize` ->
    /// `ast_to_arena`, i.e. everything `jit_backend::emit_kernel` does before
    /// calling `ast_to_runtime_arena` — reaching the identical
    /// pre-Dwrt-resolution arena, without going through
    /// `ast_to_runtime_arena` itself (which would call the production
    /// `differentiate_in_optimizer` and discard the stats we need).
    #[test]
    #[ignore = "measurement: cargo test -p pixelflow-compiler --release -- --ignored winding_kernel_dwrt_egraph_telemetry --nocapture"]
    fn winding_kernel_dwrt_egraph_telemetry() {
        assert!(
            std::env::var("PIXELFLOW_NNUE_WEIGHTS").is_err(),
            "PIXELFLOW_NNUE_WEIGHTS is set; optimize::optimize's e-graph #1 would use it — unset it"
        );

        // lib.rs:224-227
        let tokens: TokenStream = WINDING_KERNEL_SRC
            .parse()
            .expect("lex winding kernel source");
        let kernel_ast = crate::parser::parse(tokens).expect("parse winding kernel source");
        // lib.rs:228-231
        let analyzed = crate::sema::analyze(kernel_ast).expect("sema winding kernel");
        // lib.rs:236 — e-graph #1 (algebra, AST-level, config_for_node_count-tiered).
        let algebra_started = Instant::now();
        let analyzed = crate::optimize::optimize(analyzed);
        let algebra_elapsed = algebra_started.elapsed();

        let param_map = param_indices(&analyzed);
        assert_eq!(
            param_map.len(),
            7,
            "winding kernel has 7 scalar params (x0,y0,dx_over_dy,dir,y_min,y_max,min_grad)"
        );

        // The arena `differentiate_in_optimizer` receives.
        let mut arena = ExprArena::new();
        let root = ast_to_arena(&analyzed.def.body, &param_map, &mut arena)
            .expect("ast_to_arena on winding kernel body");
        assert!(
            contains_dwrt(&arena),
            "winding kernel body must contain Dwrt (DX/DY) after e-graph #1 — nothing to measure otherwise"
        );

        let node_count = reachable_count(&arena, root);
        println!(
            "winding kernel Dwrt arena: {node_count} reachable nodes (post-algebra-e-graph, pre-Dwrt-resolution)"
        );
        println!(
            "algebra e-graph (#1, AST-level) wall-clock: {:.1}ms",
            algebra_elapsed.as_secs_f64() * 1e3
        );

        // Production budget (ir_bridge.rs:718-722): NOT config_for_node_count
        // -tiered, the hardcoded EGraph::saturate() default.
        const PROD_MAX_ITERS: usize = 100;
        const PROD_MAX_CLASSES: usize = 10_000;
        const PROD_TIMEOUT_MS: u64 = 500;
        let mult: usize = std::env::var("PIXELFLOW_TELEMETRY_REF_MULT")
            .map(|s| s.parse().expect("REF_MULT must be an integer"))
            .unwrap_or(4);
        let ceiling = Duration::from_secs(
            std::env::var("PIXELFLOW_TELEMETRY_REF_CEILING_S")
                .map(|s| s.parse().expect("REF_CEILING_S must be an integer"))
                .unwrap_or(600),
        );

        // Guard first (matches differentiate_in_optimizer:687-694 exactly):
        // if the arena is not e-graph-representable, production returns
        // `None` WITHOUT constructing an `EGraph` at all — zero saturation,
        // not a truncated one. Check once; all three runs would bail
        // identically since the guard depends only on the (budget-independent)
        // arena contents.
        if let Err(bad_ops) = run_dwrt_egraph(
            &arena,
            root,
            PROD_MAX_ITERS,
            PROD_MAX_CLASSES,
            Duration::from_millis(PROD_TIMEOUT_MS),
        ) {
            println!(
                "RESULT: differentiate_in_optimizer bails at the representable guard (ir_bridge.rs:687-694) \
                 BEFORE constructing an EGraph — blocking op(s): {bad_ops:?}. Zero saturation happens for this \
                 kernel at macro-expansion time; Dwrt survives to ast_to_runtime_arena's fallback (:733), which \
                 emits the Dwrt-carrying arena unchanged, and the runtime `lower_dwrt` SYMBOLIC PASS (not an \
                 e-graph — runtime.rs:120) resolves it before the runtime tier's e-graph saturates the composed \
                 glyph arena. That runtime e-graph is the SAME one already measured per-glyph in \
                 pixelflow-search's production_telemetry (stage 1) — this kernel's Dwrt resolution has no \
                 budget of its own to report."
            );
            return;
        }
        let prod = run_dwrt_egraph(
            &arena,
            root,
            PROD_MAX_ITERS,
            PROD_MAX_CLASSES,
            Duration::from_millis(PROD_TIMEOUT_MS),
        )
        .expect("guard already checked representable");
        let refr = run_dwrt_egraph(
            &arena,
            root,
            PROD_MAX_ITERS * mult,
            PROD_MAX_CLASSES,
            ceiling,
        )
        .expect("guard already checked representable");
        assert!(
            refr.elapsed < ceiling,
            "reference run hit its {ceiling:?} safety ceiling — raise PIXELFLOW_TELEMETRY_REF_CEILING_S and re-run"
        );
        let lifted = run_dwrt_egraph(
            &arena,
            root,
            PROD_MAX_ITERS * mult,
            PROD_MAX_CLASSES * mult,
            ceiling,
        )
        .expect("guard already checked representable");
        assert!(
            lifted.elapsed < ceiling,
            "cap-lifted run hit its {ceiling:?} safety ceiling — raise PIXELFLOW_TELEMETRY_REF_CEILING_S and re-run"
        );

        assert!(
            !prod.dwrt_survived,
            "production run left Dwrt unresolved (budget miss) — the runtime lower_dwrt fallback would fire for every glyph; this changes the flat-answer conclusion, do not treat as a normal row"
        );
        assert!(
            !refr.dwrt_survived && !lifted.dwrt_survived,
            "reference/lifted run left Dwrt unresolved despite a larger budget — investigate before trusting cost numbers"
        );

        // Read off the loop's own decision (`SaturationStats::stop`),
        // never inferred from the reference runs; those exist only to price
        // the truncation.
        let stop = prod.stop;
        let loss_vs_ref = (prod.cost as f64 - refr.cost as f64) / refr.cost as f64 * 100.0;
        let loss_vs_lifted = (prod.cost as f64 - lifted.cost as f64) / lifted.cost as f64 * 100.0;

        println!(
            "prod:   stop={stop:?} iters={}/{PROD_MAX_ITERS} classes={} apps={} unions={} elapsed_ms={:.1} cost={}",
            prod.iterations,
            prod.classes_after,
            prod.applications,
            prod.total_unions,
            prod.elapsed.as_secs_f64() * 1e3,
            prod.cost
        );
        println!(
            "ref({mult}x iters, same class cap): iters={} classes={} apps={} elapsed_ms={:.1} cost={} loss_vs_ref={loss_vs_ref:.2}%",
            refr.iterations,
            refr.classes_after,
            refr.applications,
            refr.elapsed.as_secs_f64() * 1e3,
            refr.cost
        );
        println!(
            "lifted({mult}x iters, {mult}x classes): iters={} classes={} apps={} elapsed_ms={:.1} cost={} loss_vs_lifted={loss_vs_lifted:.2}%",
            lifted.iterations,
            lifted.classes_after,
            lifted.applications,
            lifted.elapsed.as_secs_f64() * 1e3,
            lifted.cost
        );
    }
}
