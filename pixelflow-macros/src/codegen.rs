//! # Code Generation
//!
//! Emits Rust code from the analyzed AST.
//!
//! ## Architecture: ZST Expression + Let/Var Binding
//!
//! PixelFlow expressions are Copy when all components are ZST (zero-sized types).
//! The coordinate variables X, Y, Z, W are ZST, and so are Var<N> references.
//! This means expressions using Var<N> remain Copy.
//!
//! The solution is a two-layer architecture:
//!
//! 1. **ZST Expression**: Built using coordinate variables (X, Y, Z, W) and Var<N>
//! 2. **Value Struct**: Stores non-ZST captured parameters (f32 values)
//! 3. **Let/Var binding**: Nested Let wrappers extend domain with parameter values
//!
//! ## Let/Var Binding (Peano-Encoded Stack)
//!
//! Parameters are bound using nested `Let::new()` calls that extend the domain:
//! - First param → deepest binding → `Var::<N{n-1}>`
//! - Last param → shallowest binding → `Var::<N0>` (head of stack)
//!
//! This allows **unlimited parameters** (no longer limited to 2).
//!
//! ## Example Transformation
//!
//! ```text
//! // User writes:
//! kernel!(|cx: f32, cy: f32, cz: f32| X - cx + Y - cy + Z - cz)
//!
//! // Becomes:
//! struct __Kernel { cx: f32, cy: f32, cz: f32 }
//!
//! impl Manifold<Field4> for __Kernel {
//!     fn eval(&self, __p: Field4) -> Field {
//!         // ZST expression using Var<N> (Copy!)
//!         let __expr = X - Var::<N2>::new() + Y - Var::<N1>::new() + Z - Var::<N0>::new();
//!         // Nested Let bindings extend domain with parameter values
//!         Let::new(self.cx,
//!           Let::new(self.cy,
//!             Let::new(self.cz,
//!               __expr))).eval(__p)
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

/// Peano type names for indices 0-7.
const PEANO_INDICES: [&str; 8] = ["N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7"];

/// Emit Rust code for an analyzed kernel.
pub fn emit(analyzed: AnalyzedKernel) -> TokenStream {
    let mut emitter = CodeEmitter::new(&analyzed);
    emitter.emit_kernel()
}

/// The code emitter state.
struct CodeEmitter<'a> {
    analyzed: &'a AnalyzedKernel,
    /// Maps parameter names to their Peano index for Var<N> access.
    /// First param → highest index (deepest in stack), last param → N0 (head).
    param_indices: HashMap<String, usize>,
    /// Whether we're generating for a Jet domain (Jet2, Jet3).
    /// If true, f32 literals must be wrapped as constant jets.
    use_jet_wrapper: bool,
}

impl<'a> CodeEmitter<'a> {
    fn new(analyzed: &'a AnalyzedKernel) -> Self {
        // Compute Peano indices for parameters.
        // With n parameters, first param is at index n-1, last param is at index 0.
        let n = analyzed.def.params.len();
        let mut param_indices = HashMap::new();
        for (i, param) in analyzed.def.params.iter().enumerate() {
            // First param (i=0) gets highest index (n-1), last param gets 0
            let peano_idx = n - 1 - i;
            param_indices.insert(param.name.to_string(), peano_idx);
        }

        // Determine if we need to wrap literals for Jet domains
        let use_jet_wrapper = match &analyzed.def.return_ty {
            Some(ty) => {
                let ty_str = quote! { #ty }.to_string();
                ty_str.contains("Jet3") || ty_str.contains("Jet2")
            }
            None => false,
        };

        CodeEmitter {
            analyzed,
            param_indices,
            use_jet_wrapper,
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

        // Generate the Peano type imports needed based on parameter count
        let peano_imports = self.emit_peano_imports();

        // Determine output type and infer domain type from it
        // - Jet3 output → Jet3_4 domain (for autodiff)
        // - Field output (or default) → Field4 domain
        let (output_type, domain_type, scalar_type) = match &self.analyzed.def.return_ty {
            Some(ty) => {
                // Check if return type is Jet3 (or Jet2, Jet4, etc.)
                let ty_str = quote! { #ty }.to_string();
                if ty_str.contains("Jet3") {
                    (
                        quote! { #ty },
                        quote! { (::pixelflow_core::jet::Jet3, ::pixelflow_core::jet::Jet3, ::pixelflow_core::jet::Jet3, ::pixelflow_core::jet::Jet3) },
                        quote! { ::pixelflow_core::jet::Jet3 },
                    )
                } else if ty_str.contains("Jet2") {
                    (
                        quote! { #ty },
                        quote! { (::pixelflow_core::jet::Jet2, ::pixelflow_core::jet::Jet2, ::pixelflow_core::jet::Jet2, ::pixelflow_core::jet::Jet2) },
                        quote! { ::pixelflow_core::jet::Jet2 },
                    )
                } else {
                    (
                        quote! { #ty },
                        quote! { (::pixelflow_core::Field, ::pixelflow_core::Field, ::pixelflow_core::Field, ::pixelflow_core::Field) },
                        quote! { ::pixelflow_core::Field },
                    )
                }
            }
            None => (
                quote! { ::pixelflow_core::Field },
                quote! { (::pixelflow_core::Field, ::pixelflow_core::Field, ::pixelflow_core::Field, ::pixelflow_core::Field) },
                quote! { ::pixelflow_core::Field },
            ),
        };

        // Generate the Let/Var binding to inject parameters (use stored use_jet_wrapper)
        let at_binding = self.emit_at_binding();

        quote! {
            {
                type __Domain = #domain_type;
                type __Scalar = #scalar_type;

                #[derive(Clone)]
                #struct_def

                impl ::pixelflow_core::Manifold<__Domain> for __Kernel {
                    type Output = #output_type;

                    #[inline(always)]
                    fn eval(
                        &self,
                        __p: __Domain,
                    ) -> #output_type {
                        // Import the coordinate variables, traits, and Peano types
                        use ::pixelflow_core::{X, Y, Z, W, ManifoldExt, ManifoldCompat, Manifold, Let, Var};
                        #peano_imports

                        // Build the ZST expression tree using Var<N> for parameters (this is Copy!)
                        let __expr = { #body };

                        // Wrap in nested Let bindings and evaluate
                        #at_binding
                    }
                }

                #closure #struct_init
            }
        }
    }

    /// Emit the Peano type imports needed based on parameter count.
    fn emit_peano_imports(&self) -> TokenStream {
        let n = self.analyzed.def.params.len();
        if n == 0 {
            return quote! {};
        }

        // Import N0..N{n-1}
        let imports: Vec<TokenStream> = (0..n)
            .map(|i| {
                let name = syn::Ident::new(PEANO_INDICES[i], proc_macro2::Span::call_site());
                quote! { #name }
            })
            .collect();

        quote! { use ::pixelflow_core::{ #(#imports),* }; }
    }

    /// Emit the Let/Var binding that injects parameter values into the domain.
    ///
    /// Generates nested Let::new() calls that extend the domain with each parameter,
    /// then evaluates the expression on the extended domain.
    ///
    /// For Jet domains (Jet2, Jet3), parameters are wrapped as constant jets:
    /// `Jet3::constant(Field::from(self.param))` instead of raw `self.param`.
    fn emit_at_binding(&self) -> TokenStream {
        let params = &self.analyzed.def.params;

        if params.is_empty() {
            // No parameters - evaluate expression directly on the input domain
            quote! {
                __expr.eval(__p)
            }
        } else {
            // Build nested Let bindings from outside to inside.
            // First parameter (index 0) is outermost, last parameter is innermost.
            // This creates: Let::new(p0, Let::new(p1, ..., Let::new(pN, __expr)))
            //
            // The body (__expr) uses Var<N{n-1}> for first param, Var<N0> for last param.
            // After evaluation, first param is deepest in stack, last param is head.

            let mut result = quote! { __expr };

            // Wrap from innermost to outermost (reverse order)
            for param in params.iter().rev() {
                let name = &param.name;
                // For Jet domains, wrap f32 params as constant jets
                let param_value = if self.use_jet_wrapper {
                    quote! { __Scalar::constant(::pixelflow_core::Field::from(self.#name)) }
                } else {
                    quote! { self.#name }
                };
                result = quote! {
                    Let::new(#param_value, #result)
                };
            }

            // Evaluate the whole Let-wrapped expression on the input domain
            quote! {
                #result.eval(__p)
            }
        }
    }

    /// Emit code for an expression, transforming parameters to Var<N> references.
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
                            // Parameters become Var::<N{index}>::new()
                            if let Some(&idx) = self.param_indices.get(&name_str) {
                                if idx < PEANO_INDICES.len() {
                                    let peano_name = syn::Ident::new(
                                        PEANO_INDICES[idx],
                                        proc_macro2::Span::call_site(),
                                    );
                                    quote! { Var::<#peano_name>::new() }
                                } else {
                                    // More than 8 parameters - emit compile error hint
                                    let err_msg = format!(
                                        "kernel! supports max 8 parameters, found index {}",
                                        idx
                                    );
                                    quote! { compile_error!(#err_msg) }
                                }
                            } else {
                                // Should not happen - parameter not in map
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
                if self.use_jet_wrapper {
                    // For Jet domains, wrap numeric literals as constant jets
                    // This ensures type unification: Jet3 + lit → Jet3 + Jet3
                    quote! { __Scalar::constant(::pixelflow_core::Field::from(#lit)) }
                } else {
                    quote! { #lit }
                }
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
        // Should use Var::<N0> for the single parameter
        assert!(
            output_str.contains("Var :: < N0 >"),
            "Expected Var::<N0>, got: {}",
            output_str
        );
        // Should have Let binding
        assert!(
            output_str.contains("Let :: new"),
            "Expected Let::new, got: {}",
            output_str
        );
    }

    #[test]
    fn emit_two_params() {
        let input = quote! { |cx: f32, cy: f32| (X - cx) + (Y - cy) };
        let output = compile(input);
        let output_str = output.to_string();

        // cx → Var::<N1> (first param, highest index)
        // cy → Var::<N0> (second param, head of stack)
        assert!(
            output_str.contains("Var :: < N1 >"),
            "Expected Var::<N1> for cx, got: {}",
            output_str
        );
        assert!(
            output_str.contains("Var :: < N0 >"),
            "Expected Var::<N0> for cy, got: {}",
            output_str
        );
        // Should have nested Let bindings
        assert!(
            output_str.contains("Let :: new (self . cx , Let :: new (self . cy"),
            "Expected nested Let bindings, got: {}",
            output_str
        );
    }

    #[test]
    fn emit_three_params() {
        let input = quote! { |a: f32, b: f32, c: f32| a + b + c };
        let output = compile(input);
        let output_str = output.to_string();

        // a → Var::<N2>, b → Var::<N1>, c → Var::<N0>
        assert!(
            output_str.contains("Var :: < N2 >"),
            "Expected Var::<N2> for a"
        );
        assert!(
            output_str.contains("Var :: < N1 >"),
            "Expected Var::<N1> for b"
        );
        assert!(
            output_str.contains("Var :: < N0 >"),
            "Expected Var::<N0> for c"
        );
        // Should import N0, N1, N2
        assert!(output_str.contains("N0"));
        assert!(output_str.contains("N1"));
        assert!(output_str.contains("N2"));
    }

    #[test]
    fn emit_empty_params() {
        let input = quote! { || X + Y };
        let output = compile(input);
        let output_str = output.to_string();

        // Should have unit struct
        assert!(output_str.contains("struct __Kernel ;"));
        // Should evaluate directly on __p
        assert!(
            output_str.contains("__expr . eval (__p)"),
            "Expected direct eval on __p, got: {}",
            output_str
        );
    }

    #[test]
    fn emit_method_calls() {
        let input = quote! { |r: f32| (X * X + Y * Y).sqrt() - r };
        let output = compile(input);
        let output_str = output.to_string();

        // r → Var::<N0>
        assert!(output_str.contains(". sqrt ()"));
        assert!(
            output_str.contains("Var :: < N0 >"),
            "Expected Var::<N0> for r"
        );
    }
}
