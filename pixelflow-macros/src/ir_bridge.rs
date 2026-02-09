#![allow(dead_code)]
//! Bridge between macro AST and pixelflow-ir.
//!
//! This module handles conversions between:
//! 1. Macro AST → IR (ast_to_ir)
//! 2. IR → E-graph (ir_to_egraph)
//! 3. E-graph → IR (egraph_to_ir)
//! 4. IR → Type-level code (ir_to_code)
//!
//! The IR becomes the canonical representation, with AST only used during parsing.

use crate::ast::{BinaryOp, Expr, UnaryOp};
use pixelflow_ir::{Expr as IR, OpKind};
use pixelflow_search::egraph::{EClassId, EGraph, ENode, ExprTree, Leaf, ops};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Lit;

// ============================================================================
// AST → IR Conversion
// ============================================================================

/// Convert macro AST to IR.
///
/// This flattens the high-level parsing structure (with source spans, etc.)
/// into the clean IR representation used for optimization.
pub fn ast_to_ir(expr: &Expr) -> Result<IR, String> {
    match expr {
        Expr::Ident(ident) => {
            // Map coordinate variables to their indices
            let name = ident.name.to_string();
            match name.as_str() {
                "X" => Ok(IR::Var(0)),
                "Y" => Ok(IR::Var(1)),
                "Z" => Ok(IR::Var(2)),
                "W" => Ok(IR::Var(3)),
                // Other identifiers are opaque - preserve as error for now
                _ => Err(format!("Unknown identifier: {}", name)),
            }
        }

        Expr::Literal(lit) => {
            if let Some(val) = extract_f64_from_lit(&lit.lit) {
                Ok(IR::Const(val as f32))
            } else {
                Err("Non-numeric literal".to_string())
            }
        }

        Expr::Binary(binary) => {
            let lhs = Box::new(ast_to_ir(&binary.lhs)?);
            let rhs = Box::new(ast_to_ir(&binary.rhs)?);

            let op = match binary.op {
                BinaryOp::Add => OpKind::Add,
                BinaryOp::Sub => OpKind::Sub,
                BinaryOp::Mul => OpKind::Mul,
                BinaryOp::Div => OpKind::Div,
                _ => return Err(format!("Unsupported binary op: {:?}", binary.op)),
            };

            Ok(IR::Binary(op, lhs, rhs))
        }

        Expr::Unary(unary) => {
            let operand = Box::new(ast_to_ir(&unary.operand)?);

            let op = match unary.op {
                UnaryOp::Neg => OpKind::Neg,
                UnaryOp::Not => return Err("Unsupported unary op: Not".to_string()),
            };

            Ok(IR::Unary(op, operand))
        }

        Expr::MethodCall(call) => {
            let method = call.method.to_string();
            let receiver = Box::new(ast_to_ir(&call.receiver)?);

            match (method.as_str(), call.args.len()) {
                // Unary methods
                ("sqrt", 0) => Ok(IR::Unary(OpKind::Sqrt, receiver)),
                ("abs", 0) => Ok(IR::Unary(OpKind::Abs, receiver)),
                ("neg", 0) => Ok(IR::Unary(OpKind::Neg, receiver)),

                // Binary methods
                ("min", 1) => {
                    let arg = Box::new(ast_to_ir(&call.args[0])?);
                    Ok(IR::Binary(OpKind::Min, receiver, arg))
                }
                ("max", 1) => {
                    let arg = Box::new(ast_to_ir(&call.args[0])?);
                    Ok(IR::Binary(OpKind::Max, receiver, arg))
                }

                // Ternary methods
                ("mul_add", 2) => {
                    let b = Box::new(ast_to_ir(&call.args[0])?);
                    let c = Box::new(ast_to_ir(&call.args[1])?);
                    Ok(IR::Ternary(OpKind::MulAdd, receiver, b, c))
                }

                _ => Err(format!("Unsupported method: {}", method)),
            }
        }

        _ => Err("Unsupported expression type".to_string()),
    }
}

/// Extract f64 from a syn::Lit.
fn extract_f64_from_lit(lit: &Lit) -> Option<f64> {
    match lit {
        Lit::Float(f) => f.base10_parse::<f64>().ok(),
        Lit::Int(i) => i.base10_parse::<i64>().ok().map(|v| v as f64),
        _ => None,
    }
}

// ============================================================================
// IR → E-graph Conversion (Flattening)
// ============================================================================

/// Context for flattening IR trees into E-graph.
pub struct IRToEGraphContext {
    pub egraph: EGraph,
}

impl IRToEGraphContext {
    pub fn new() -> Self {
        Self {
            egraph: EGraph::new(),
        }
    }

    /// Flatten an IR tree into the E-graph, returning the root e-class ID.
    pub fn ir_to_egraph(&mut self, ir: &IR) -> EClassId {
        match ir {
            IR::Var(idx) => self.egraph.add(ENode::Var(*idx)),

            IR::Const(val) => self.egraph.add(ENode::constant(*val)),

            IR::Unary(op, child) => {
                let child_id = self.ir_to_egraph(child);
                let op_ref = opkind_to_op(*op);
                self.egraph.add(ENode::Op {
                    op: op_ref,
                    children: vec![child_id],
                })
            }

            IR::Binary(op, lhs, rhs) => {
                let lhs_id = self.ir_to_egraph(lhs);
                let rhs_id = self.ir_to_egraph(rhs);
                let op_ref = opkind_to_op(*op);
                self.egraph.add(ENode::Op {
                    op: op_ref,
                    children: vec![lhs_id, rhs_id],
                })
            }

            IR::Ternary(op, a, b, c) => {
                let a_id = self.ir_to_egraph(a);
                let b_id = self.ir_to_egraph(b);
                let c_id = self.ir_to_egraph(c);
                let op_ref = opkind_to_op(*op);
                self.egraph.add(ENode::Op {
                    op: op_ref,
                    children: vec![a_id, b_id, c_id],
                })
            }

            IR::Nary(op, children) => {
                let child_ids: Vec<EClassId> = children
                    .iter()
                    .map(|child| self.ir_to_egraph(child))
                    .collect();
                let op_ref = opkind_to_op(*op);
                self.egraph.add(ENode::Op {
                    op: op_ref,
                    children: child_ids,
                })
            }
        }
    }
}

/// Map OpKind to a static Op trait object reference.
fn opkind_to_op(kind: OpKind) -> &'static dyn ops::Op {
    match kind {
        OpKind::Add => &ops::Add,
        OpKind::Sub => &ops::Sub,
        OpKind::Mul => &ops::Mul,
        OpKind::Div => &ops::Div,
        OpKind::Neg => &ops::Neg,
        OpKind::Sqrt => &ops::Sqrt,
        OpKind::Rsqrt => &ops::Rsqrt,
        OpKind::Recip => &ops::Recip,
        OpKind::Abs => &ops::Abs,
        OpKind::Min => &ops::Min,
        OpKind::Max => &ops::Max,
        OpKind::MulAdd => &ops::MulAdd,
        _ => panic!("Unsupported OpKind: {:?}", kind),
    }
}

// ============================================================================
// E-graph → IR Conversion (Extraction)
// ============================================================================

/// Convert an extracted ExprTree back to IR.
pub fn egraph_to_ir(tree: &ExprTree) -> IR {
    match tree {
        ExprTree::Leaf(Leaf::Var(idx)) => IR::Var(*idx),

        ExprTree::Leaf(Leaf::Const(val)) => IR::Const(*val),

        ExprTree::Op { op, children } => {
            let name = op.name();

            // Map op name back to OpKind
            let kind = match name {
                "add" => OpKind::Add,
                "sub" => OpKind::Sub,
                "mul" => OpKind::Mul,
                "div" => OpKind::Div,
                "neg" => OpKind::Neg,
                "sqrt" => OpKind::Sqrt,
                "rsqrt" => OpKind::Rsqrt,
                "recip" => OpKind::Recip,
                "abs" => OpKind::Abs,
                "min" => OpKind::Min,
                "max" => OpKind::Max,
                "mul_add" => OpKind::MulAdd,
                _ => panic!("Unknown op: {}", name),
            };

            // Convert children
            let child_irs: Vec<IR> = children.iter().map(egraph_to_ir).collect();

            match child_irs.len() {
                1 => IR::Unary(kind, Box::new(child_irs[0].clone())),
                2 => IR::Binary(
                    kind,
                    Box::new(child_irs[0].clone()),
                    Box::new(child_irs[1].clone()),
                ),
                3 => IR::Ternary(
                    kind,
                    Box::new(child_irs[0].clone()),
                    Box::new(child_irs[1].clone()),
                    Box::new(child_irs[2].clone()),
                ),
                _ => IR::Nary(kind, child_irs),
            }
        }
    }
}

// ============================================================================
// IR → Type-Level Code Generation
// ============================================================================

/// Generate type-level code from IR.
///
/// This emits the type-level AST that will be monomorphized by rustc.
pub fn ir_to_code(ir: &IR) -> TokenStream {
    match ir {
        IR::Var(idx) => {
            // Map variable indices to coordinate variables
            match idx {
                0 => quote! { X },
                1 => quote! { Y },
                2 => quote! { Z },
                3 => quote! { W },
                _ => {
                    let var_name = format_ident!("v{}", idx);
                    quote! { #var_name }
                }
            }
        }

        IR::Const(val) => {
            quote! { #val }
        }

        IR::Unary(op, child) => {
            let child_code = ir_to_code(child);
            match op {
                OpKind::Neg => quote! { Neg::new(#child_code) },
                OpKind::Sqrt => quote! { (#child_code).sqrt() },
                OpKind::Abs => quote! { (#child_code).abs() },
                OpKind::Rsqrt => quote! { (#child_code).rsqrt() },
                OpKind::Recip => quote! { (#child_code).recip() },
                _ => panic!("Unsupported unary op: {:?}", op),
            }
        }

        IR::Binary(op, lhs, rhs) => {
            let lhs_code = ir_to_code(lhs);
            let rhs_code = ir_to_code(rhs);
            match op {
                OpKind::Add => quote! { (#lhs_code) + (#rhs_code) },
                OpKind::Sub => quote! { (#lhs_code) - (#rhs_code) },
                OpKind::Mul => quote! { (#lhs_code) * (#rhs_code) },
                OpKind::Div => quote! { (#lhs_code) / (#rhs_code) },
                OpKind::Min => quote! { (#lhs_code).min(#rhs_code) },
                OpKind::Max => quote! { (#lhs_code).max(#rhs_code) },
                _ => panic!("Unsupported binary op: {:?}", op),
            }
        }

        IR::Ternary(op, a, b, c) => {
            let a_code = ir_to_code(a);
            let b_code = ir_to_code(b);
            let c_code = ir_to_code(c);
            match op {
                OpKind::MulAdd => quote! { (#a_code).mul_add(#b_code, #c_code) },
                _ => panic!("Unsupported ternary op: {:?}", op),
            }
        }

        IR::Nary(_, _children) => {
            panic!("N-ary ops not yet supported in codegen")
        }
    }
}
