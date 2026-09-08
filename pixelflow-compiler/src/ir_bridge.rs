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
use quote::{format_ident, quote};
use std::collections::HashMap;
use syn::Lit;

// ============================================================================
// AST → Arena IR Conversion
// ============================================================================

/// DSL method calls that denote a fixed composition of primitive ops rather
/// than a single [`OpKind`] — `(name, arg_count)`, `arg_count` excluding the
/// receiver.
///
/// Each backend (arena lowering here, the e-graph in `optimize.rs`) builds
/// the composition in its own node representation, so the expansion itself
/// isn't shared — but this list is the one place that says which names and
/// arities exist, so `sema`'s validation and both backends' dispatch cannot
/// silently drift on which library methods a kernel body may call.
pub(crate) const LIBRARY_METHODS: &[(&str, usize)] = &[("fract", 0), ("hypot", 1), ("clamp", 2)];

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
    let mut lowering = Lowering {
        param_indices,
        locals: HashMap::new(),
        arena,
    };
    lowering.lower(expr)
}

/// State threaded through the AST → arena walk: parameter names (fixed for
/// the whole kernel), `let`-bound locals (grows as blocks are walked), and
/// the arena nodes are pushed into.
struct Lowering<'a> {
    param_indices: &'a HashMap<String, u8>,
    locals: HashMap<String, ExprId>,
    arena: &'a mut ExprArena,
}

impl Lowering<'_> {
    /// Translate an AST node into the arena, resolving `let`-bound locals via
    /// `self.locals`. The optimizer emits `let`-bindings (a [`Expr::Block`]) for
    /// shared subexpressions; each binding maps to a single [`ExprId`], so the
    /// arena faithfully preserves the discovered CSE as a DAG rather than
    /// duplicating subtrees.
    fn lower(&mut self, expr: &Expr) -> Result<ExprId, String> {
        match expr {
            Expr::Ident(ident) => {
                let name = ident.name.to_string();
                match name.as_str() {
                    "X" => Ok(self.arena.push_var(0)),
                    "Y" => Ok(self.arena.push_var(1)),
                    _ => {
                        if let Some(&id) = self.locals.get(&name) {
                            Ok(id)
                        } else if let Some(&idx) = self.param_indices.get(&name) {
                            Ok(self.arena.push_param(idx))
                        } else {
                            Err(format!("Unknown identifier: {}", name))
                        }
                    }
                }
            }

            Expr::Literal(lit) => {
                if let Some(val) = extract_f64_from_lit(&lit.lit) {
                    Ok(self.arena.push_const(val as f32))
                } else {
                    Err("Non-numeric literal".to_string())
                }
            }

            Expr::Binary(binary) => {
                let lhs = self.lower(&binary.lhs)?;
                let rhs = self.lower(&binary.rhs)?;

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

                Ok(self.arena.push_binary(op, lhs, rhs))
            }

            Expr::Unary(unary) => {
                let operand = self.lower(&unary.operand)?;

                let op = match unary.op {
                    UnaryOp::Neg => OpKind::Neg,
                    UnaryOp::Not => return Err("Unsupported unary op: Not".to_string()),
                };

                Ok(self.arena.push_unary(op, operand))
            }

            Expr::MethodCall(call) => {
                let method = call.method.to_string();

                // `.at(x, y)` warped a manifold-typed macro param at a
                // call site. There are no manifold params: a kernel composes
                // `Kernel` values, and `Kernel::at` is the warp.
                if method == "at" {
                    return Err(
                        ".at() inside a kernel body samples a manifold param, and there are none; \
                         compose Kernel values with Kernel::at instead"
                            .to_string(),
                    );
                }

                let receiver = self.lower(&call.receiver)?;
                let arg_count = call.args.len();

                // Arena expressions are values; `.clone()` (needed by the
                // combinator backend for non-Copy trees) is the identity here,
                // so one kernel body compiles under both backends.
                if method == "clone" && arg_count == 0 {
                    return Ok(receiver);
                }

                // Primitive ops: one `OpKind` per (name, arity), read from
                // the single table `OpKind::from_method_call` resolves
                // against — not re-listed here as a second copy that could
                // silently drift from it (see `LIBRARY_METHODS` below for
                // the one part of this dispatch that table doesn't cover).
                if let Some(op) = OpKind::from_method_call(&method, arg_count) {
                    let mut args = Vec::with_capacity(arg_count);
                    for arg in &call.args {
                        args.push(self.lower(arg)?);
                    }
                    return Ok(match *args.as_slice() {
                        [] => self.arena.push_unary(op, receiver),
                        [a] => self.arena.push_binary(op, receiver, a),
                        [a, b] => self.arena.push_ternary(op, receiver, a, b),
                        _ => unreachable!(
                            "OpKind::from_method_call only resolves ops of arity 1..=3"
                        ),
                    });
                }

                match (method.as_str(), arg_count) {
                    // `fract(x) = x - floor(x)`.
                    ("fract", 0) => {
                        let f = self.arena.push_unary(OpKind::Floor, receiver);
                        Ok(self.arena.push_binary(OpKind::Sub, receiver, f))
                    }
                    // `hypot(x, y) = sqrt(x² + y²)`.
                    ("hypot", 1) => {
                        let arg = self.lower(&call.args[0])?;
                        let xx = self.arena.push_binary(OpKind::Mul, receiver, receiver);
                        let yy = self.arena.push_binary(OpKind::Mul, arg, arg);
                        let sum = self.arena.push_binary(OpKind::Add, xx, yy);
                        Ok(self.arena.push_unary(OpKind::Sqrt, sum))
                    }
                    // `clamp` is library, not a primitive: it denotes
                    // `min(max(x, lo), hi)` and is built as that composition.
                    ("clamp", 2) => {
                        let lo = self.lower(&call.args[0])?;
                        let hi = self.lower(&call.args[1])?;
                        let floored = self.arena.push_binary(OpKind::Max, receiver, lo);
                        Ok(self.arena.push_binary(OpKind::Min, floored, hi))
                    }

                    _ => Err(format!("Unsupported method: {}", method)),
                }
            }

            // Derivative projections (V/DX/DY and the Hessian family) map to
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
                let inner = self.lower(&call.args[0])?;
                match func.as_str() {
                    "V" => Ok(inner),
                    "DX" => Ok(push_dwrt(self.arena, inner, 0)),
                    "DY" => Ok(push_dwrt(self.arena, inner, 1)),
                    "DZ" => Err(
                        "`DZ` is no longer a coordinate: a lattice has two axes, X and Y"
                            .to_string(),
                    ),
                    "DXX" => {
                        let d = push_dwrt(self.arena, inner, 0);
                        Ok(push_dwrt(self.arena, d, 0))
                    }
                    "DXY" => {
                        let d = push_dwrt(self.arena, inner, 0);
                        Ok(push_dwrt(self.arena, d, 1))
                    }
                    "DYY" => {
                        let d = push_dwrt(self.arena, inner, 1);
                        Ok(push_dwrt(self.arena, d, 1))
                    }
                    _ => Err(format!("Unsupported call: {}", func)),
                }
            }

            // Parentheses are transparent - just recurse into the inner expression
            Expr::Paren(inner) => self.lower(inner),

            // Blocks carry the optimizer's CSE: each `let __n = <expr>;` binds a
            // shared subexpression to a single arena node, and the final expression
            // references those bindings by name.
            Expr::Block(block) => {
                for stmt in &block.stmts {
                    match stmt {
                        crate::ast::Stmt::Let(let_stmt) => {
                            let id = self.lower(&let_stmt.init)?;
                            self.locals.insert(let_stmt.name.to_string(), id);
                        }
                        // A non-binding statement has no value to thread; evaluate
                        // it so any nested error surfaces, then discard the id.
                        crate::ast::Stmt::Expr(e) => {
                            let _ = self.lower(e)?;
                        }
                    }
                }
                match &block.expr {
                    Some(final_expr) => self.lower(final_expr),
                    None => Err("Block has no final expression".to_string()),
                }
            }

            _ => Err("Unsupported expression type".to_string()),
        }
    }
}

/// Generate runtime arena-construction code from macro AST.
///
/// `Dwrt` nodes are emitted as they were built and resolved at bake time, by
/// the one `LowerDwrt` pass in the runtime pipeline. Resolving them here
/// instead — which this function used to do — is not an optimization but a
/// miscompilation under composition: `Kernel::at` warps a kernel by
/// substituting into its `Var` leaves, so a surviving `Dwrt(f, x)` has the
/// warp reach its *operand* and differentiates the warped function, which is
/// the chain rule. A `Dwrt` already resolved to `f'` has no operand left to
/// warp, and the substitution silently lands in `f'` instead.
///
/// See docs/plans/2026-09-08-macro-tier-is-arena-native.md.
pub fn ast_to_runtime_arena(
    expr: &Expr,
    param_indices: &HashMap<String, u8>,
) -> Result<TokenStream, String> {
    let mut arena = ExprArena::new();
    let root = ast_to_arena(expr, param_indices, &mut arena)?;
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
            // Likewise unreachable: a uniform enters a kernel at the builder
            // call (`substitute_params`), never from the macro's own arena.
            pixelflow_ir::arena::ExprNode::Uniform(u) => {
                panic!(
                    "kernel! produced ExprNode::Uniform({}) — uniforms are chosen at the \
                     builder call site, not in the macro body",
                    u.0
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

/// Extract f64 from a syn::Lit.
fn extract_f64_from_lit(lit: &Lit) -> Option<f64> {
    match lit {
        Lit::Float(f) => f.base10_parse::<f64>().ok(),
        Lit::Int(i) => i.base10_parse::<i64>().ok().map(|v| v as f64),
        _ => None,
    }
}

/// The path naming `kind` in generated code.
///
/// One line per op used to live here — 40 of the 50, closing with a
/// `_ => panic!("Unsupported OpKind for JIT")` that refused ops the arena
/// holds and codegen emits perfectly well (`Reduce`, the integer-domain ops,
/// `Gather`). It was the fourth independently-maintained copy of the op
/// table, and the third one found drifting from it this week.
///
/// [`OpKind::variant_name`] is generated by `op_table!`, so the identifier
/// cannot drift from the enum and a newly added op needs no edit here.
fn opkind_to_tokens(kind: OpKind) -> TokenStream {
    let variant = format_ident!("{}", kind.variant_name());
    quote! { ::pixelflow_core::__macro::ir::OpKind::#variant }
}
