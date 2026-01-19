//! # Code Generation
//!
//! Emits Rust code from the analyzed AST.
//!
//! ## Architecture: ZST Expression + Value Struct
//!
//! PixelFlow expressions are Copy when all components are ZST (zero-sized types).
//! The coordinate variables X, Y, Z, W are ZST, so expressions built purely from
//! them are also Copy. Non-ZST values (f32 parameters) break this.
//!
//! The solution is a two-layer architecture:
//!
//! 1. **ZST Expression**: Built using only coordinate variables (X, Y, Z, W)
//! 2. **Value Struct**: Stores non-ZST captured parameters
//! 3. **`.at()` binding**: Threads struct values into coordinate slots
//!
//! ## Coordinate Slot Allocation
//!
//! With 4 coordinate dimensions, we allocate:
//! - **X, Y**: Reserved for screen/pixel coordinates (passed through)
//! - **Z, W**: Available for parameter binding
//!
//! For 1-2 parameters, Z and W suffice. For more parameters, we use nested
//! `.at()` calls to create additional virtual coordinate layers.
//!
//! ## Example Transformation
//!
//! ```text
//! // User writes:
//! kernel!(|cx: f32, cy: f32| (X - cx) * (X - cx) + (Y - cy) * (Y - cy))
//!
//! // Becomes:
//! struct __Kernel { cx: f32, cy: f32 }
//!
//! impl Manifold for __Kernel {
//!     fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
//!         // ZST expression (Copy!)
//!         let __expr = (X - Z) * (X - Z) + (Y - W) * (Y - W);
//!         // Bind parameters via .at(), then evaluate
//!         __expr.at(X, Y, self.cx, self.cy).eval_raw(x, y, z, w)
//!     }
//! }
//! ```

use crate::ast::{
    BinaryExpr, BinaryOp, BlockExpr, Expr, LetStmt, MethodCallExpr, Stmt, UnaryExpr, UnaryOp,
};
use crate::sema::AnalyzedKernel;
use crate::symbol::SymbolKind;
use proc_macro2::TokenStream;
use quote::{quote, ToTokens};
use std::collections::HashMap;

/// Coordinate slots available for parameter binding.
/// X and Y are reserved for screen coordinates.
const PARAM_SLOTS: [&str; 2] = ["Z", "W"];

/// Emit Rust code for an analyzed kernel.
pub fn emit(analyzed: AnalyzedKernel) -> TokenStream {
    let mut emitter = CodeEmitter::new(&analyzed);
    emitter.emit_kernel()
}

/// The code emitter state.
struct CodeEmitter<'a> {
    analyzed: &'a AnalyzedKernel,
    /// Maps parameter names to their coordinate slot (Z or W)
    param_slots: HashMap<String, &'static str>,
}

impl<'a> CodeEmitter<'a> {
    fn new(analyzed: &'a AnalyzedKernel) -> Self {
        // Allocate coordinate slots for parameters
        let mut param_slots = HashMap::new();
        for (i, param) in analyzed.def.params.iter().enumerate() {
            if i < PARAM_SLOTS.len() {
                param_slots.insert(param.name.to_string(), PARAM_SLOTS[i]);
            }
            // TODO: Handle >2 params with nested .at() layers
        }

        CodeEmitter {
            analyzed,
            param_slots,
        }
    }

    /// Emit the complete kernel definition.
    fn emit_kernel(&mut self) -> TokenStream {
        let params = &self.analyzed.def.params;

        // Generate struct fields
        let struct_fields: Vec<TokenStream> = params
            .iter()
            .map(|p| {
                let name = &p.name;
                let ty = &p.ty;
                quote! { #name: #ty }
            })
            .collect();

        // Generate struct field names for construction
        let field_names: Vec<_> = params.iter().map(|p| &p.name).collect();

        // Generate closure parameters
        let closure_params: Vec<TokenStream> = params
            .iter()
            .map(|p| {
                let name = &p.name;
                let ty = &p.ty;
                quote! { #name: #ty }
            })
            .collect();

        // Transform and emit the body as a ZST expression
        let body = self.emit_expr(&self.analyzed.def.body);

        // Generate the .at() binding to inject parameters
        let at_binding = self.emit_at_binding();

        // Handle empty vs non-empty params
        let (struct_def, struct_init, closure) = if params.is_empty() {
            (
                quote! { struct __Kernel; },
                quote! { __Kernel },
                quote! { || },
            )
        } else {
            (
                quote! { struct __Kernel { #(#struct_fields),* } },
                quote! { __Kernel { #(#field_names),* } },
                quote! { |#(#closure_params),*| },
            )
        };

        quote! {
            {
                type __Field4 = (::pixelflow_core::Field, ::pixelflow_core::Field, ::pixelflow_core::Field, ::pixelflow_core::Field);

                #[derive(Clone)]
                #struct_def

                impl ::pixelflow_core::Manifold<__Field4> for __Kernel {
                    type Output = ::pixelflow_core::Field;

                    #[inline(always)]
                    fn eval(
                        &self,
                        __p: __Field4,
                    ) -> ::pixelflow_core::Field {
                        let (__x, __y, __z, __w) = __p;
                        // Import the coordinate variables and traits
                        use ::pixelflow_core::{X, Y, Z, W, ManifoldExt, ManifoldCompat};

                        // Build the ZST expression tree (this is Copy!)
                        let __expr = { #body };

                        // Bind parameters via .at(), then evaluate at screen coords
                        #at_binding
                    }
                }

                #closure #struct_init
            }
        }
    }

    /// Emit the .at() binding that injects parameter values into coordinate slots.
    fn emit_at_binding(&self) -> TokenStream {
        let params = &self.analyzed.def.params;

        if params.is_empty() {
            // No parameters - evaluate directly using ManifoldCompat
            quote! {
                ::pixelflow_core::ManifoldCompat::eval_raw(&__expr, __x, __y, __z, __w)
            }
        } else {
            // Build .at() arguments: X, Y pass through, Z/W get parameter values
            let z_arg = params
                .iter()
                .find(|p| self.param_slots.get(&p.name.to_string()) == Some(&"Z"))
                .map(|p| {
                    let name = &p.name;
                    quote! { self.#name }
                })
                .unwrap_or_else(|| quote! { Z });

            let w_arg = params
                .iter()
                .find(|p| self.param_slots.get(&p.name.to_string()) == Some(&"W"))
                .map(|p| {
                    let name = &p.name;
                    quote! { self.#name }
                })
                .unwrap_or_else(|| quote! { W });

            // Use .collapse() on the pinned At combinator
            quote! {
                __expr.at(__x, __y, #z_arg, #w_arg).collapse()
            }
        }
    }

    /// Emit code for an expression, transforming parameters to coordinate references.
    fn emit_expr(&self, expr: &Expr) -> TokenStream {
        match expr {
            Expr::Ident(ident_expr) => {
                let name = &ident_expr.name;
                let name_str = name.to_string();

                match self.analyzed.symbols.lookup(&name_str) {
                    Some(symbol) => match symbol.kind {
                        SymbolKind::Intrinsic => {
                            // Intrinsics (X, Y, Z, W) emitted as-is
                            quote! { #name }
                        }
                        SymbolKind::Parameter => {
                            // Parameters become their allocated coordinate slot
                            if let Some(&slot) = self.param_slots.get(&name_str) {
                                let slot_ident =
                                    syn::Ident::new(slot, proc_macro2::Span::call_site());
                                quote! { #slot_ident }
                            } else {
                                // Fallback for >2 params (TODO: proper nested .at())
                                quote! { #name }
                            }
                        }
                        SymbolKind::Local => {
                            // Locals emitted as-is
                            quote! { #name }
                        }
                    },
                    None => {
                        // Unknown - emit as-is
                        quote! { #name }
                    }
                }
            }

            Expr::Literal(lit_expr) => {
                let lit = &lit_expr.lit;
                quote! { #lit }
            }

            Expr::Binary(binary) => self.emit_binary(binary),

            Expr::Unary(unary) => self.emit_unary(unary),

            Expr::MethodCall(call) => self.emit_method_call(call),

            Expr::Block(block) => self.emit_block(block),

            Expr::Paren(inner) => {
                let inner_code = self.emit_expr(inner);
                quote! { (#inner_code) }
            }

            Expr::Verbatim(syn_expr) => {
                // Pass through verbatim
                syn_expr.to_token_stream()
            }
        }
    }

    /// Emit a binary expression.
    fn emit_binary(&self, binary: &BinaryExpr) -> TokenStream {
        let lhs = self.emit_expr(&binary.lhs);
        let rhs = self.emit_expr(&binary.rhs);

        match binary.op {
            BinaryOp::Add => quote! { #lhs + #rhs },
            BinaryOp::Sub => quote! { #lhs - #rhs },
            BinaryOp::Mul => quote! { #lhs * #rhs },
            BinaryOp::Div => quote! { #lhs / #rhs },
            BinaryOp::Rem => quote! { #lhs % #rhs },
            BinaryOp::Lt => quote! { #lhs.lt(#rhs) },
            BinaryOp::Le => quote! { #lhs.le(#rhs) },
            BinaryOp::Gt => quote! { #lhs.gt(#rhs) },
            BinaryOp::Ge => quote! { #lhs.ge(#rhs) },
            BinaryOp::Eq => quote! { #lhs.eq(#rhs) },
            BinaryOp::Ne => quote! { #lhs.ne(#rhs) },
        }
    }

    /// Emit a unary expression.
    fn emit_unary(&self, unary: &UnaryExpr) -> TokenStream {
        let operand = self.emit_expr(&unary.operand);

        match unary.op {
            UnaryOp::Neg => quote! { #operand.neg() },
            UnaryOp::Not => quote! { !#operand },
        }
    }

    /// Emit a method call.
    fn emit_method_call(&self, call: &MethodCallExpr) -> TokenStream {
        let receiver = self.emit_expr(&call.receiver);
        let method = &call.method;
        let args: Vec<TokenStream> = call.args.iter().map(|a| self.emit_expr(a)).collect();

        if args.is_empty() {
            quote! { #receiver.#method() }
        } else {
            quote! { #receiver.#method(#(#args),*) }
        }
    }

    /// Emit a block expression.
    fn emit_block(&self, block: &BlockExpr) -> TokenStream {
        let stmts: Vec<TokenStream> = block.stmts.iter().map(|s| self.emit_stmt(s)).collect();

        let final_expr = block.expr.as_ref().map(|e| self.emit_expr(e));

        match final_expr {
            Some(expr) => quote! {
                {
                    #(#stmts)*
                    #expr
                }
            },
            None => quote! {
                {
                    #(#stmts)*
                }
            },
        }
    }

    /// Emit a statement.
    fn emit_stmt(&self, stmt: &Stmt) -> TokenStream {
        match stmt {
            Stmt::Let(let_stmt) => self.emit_let(let_stmt),
            Stmt::Expr(expr) => {
                let code = self.emit_expr(expr);
                quote! { #code; }
            }
        }
    }

    /// Emit a let statement.
    fn emit_let(&self, let_stmt: &LetStmt) -> TokenStream {
        let name = &let_stmt.name;
        let init = self.emit_expr(&let_stmt.init);

        match &let_stmt.ty {
            Some(ty) => quote! { let #name: #ty = #init; },
            None => quote! { let #name = #init; },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use crate::sema::analyze;
    use quote::quote;

    fn compile(input: TokenStream) -> TokenStream {
        let kernel = parse(input).unwrap();
        let analyzed = analyze(kernel).unwrap();
        emit(analyzed)
    }

    #[test]
    fn emit_simple_kernel() {
        let input = quote! { |cx: f32| X - cx };
        let output = compile(input);
        let output_str = output.to_string();

        // Should contain struct definition
        assert!(output_str.contains("struct __Kernel"));
        // Should use Z coordinate for cx (parameter mapped to slot)
        assert!(output_str.contains("X - Z"));
        // Should have .at() binding
        assert!(output_str.contains(". at ("));
    }

    #[test]
    fn emit_two_params() {
        let input = quote! { |cx: f32, cy: f32| (X - cx) + (Y - cy) };
        let output = compile(input);
        let output_str = output.to_string();

        // cx → Z, cy → W
        assert!(output_str.contains("X - Z"));
        assert!(output_str.contains("Y - W"));
    }

    #[test]
    fn emit_empty_params() {
        let input = quote! { || X + Y };
        let output = compile(input);
        let output_str = output.to_string();

        // Should have unit struct
        assert!(output_str.contains("struct __Kernel ;"));
        // Should evaluate directly without .at()
        assert!(output_str.contains("eval_raw (& __expr"));
    }

    #[test]
    fn emit_method_calls() {
        let input = quote! { |r: f32| (X * X + Y * Y).sqrt() - r };
        let output = compile(input);
        let output_str = output.to_string();

        // r → Z
        assert!(output_str.contains(". sqrt ()"));
        assert!(output_str.contains("- Z"));
    }
}
