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

use crate::annotate::{
    annotate, AnnotatedExpr, AnnotatedStmt, AnnotationCtx, CollectedLiteral,
};
use crate::ast::{BinaryOp, ParamKind, UnaryOp};
use crate::sema::AnalyzedKernel;
use crate::symbol::SymbolKind;
use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};
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
    /// Maps manifold parameter names to their generic type index (M0, M1, ...).
    manifold_indices: HashMap<String, usize>,
    /// Whether we're generating for a Jet domain (Jet2, Jet3).
    /// If true, f32 literals must be wrapped as constant jets.
    use_jet_wrapper: bool,
    /// Collected literals from annotation pass (for Let bindings in Jet mode).
    collected_literals: Vec<CollectedLiteral>,
}

impl<'a> CodeEmitter<'a> {
    fn new(analyzed: &'a AnalyzedKernel) -> Self {
        // Compute Peano indices for ALL parameters (both scalar and manifold).
        // With n parameters, first param is at index n-1, last param is at index 0.
        let n = analyzed.def.params.len();
        let mut param_indices = HashMap::new();
        for (i, param) in analyzed.def.params.iter().enumerate() {
            // First param (i=0) gets highest index (n-1), last param gets 0
            let peano_idx = n - 1 - i;
            param_indices.insert(param.name.to_string(), peano_idx);
        }

        // Compute generic type indices for manifold parameters (M0, M1, ...).
        // These are numbered in declaration order.
        let mut manifold_indices = HashMap::new();
        let mut manifold_count = 0;
        for param in &analyzed.def.params {
            if matches!(param.kind, ParamKind::Manifold) {
                manifold_indices.insert(param.name.to_string(), manifold_count);
                manifold_count += 1;
            }
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
            manifold_indices,
            use_jet_wrapper,
            collected_literals: Vec::new(), // Populated during emit_kernel
        }
    }

    /// Emit the complete kernel definition.
    fn emit_kernel(&mut self) -> TokenStream {
        let params = &self.analyzed.def.params;

        // Count manifold parameters for generic type generation
        let manifold_count = self.manifold_indices.len();

        // Generate generic type parameter names (M0, M1, ...)
        let generic_names: Vec<syn::Ident> = (0..manifold_count)
            .map(|i| format_ident!("M{}", i))
            .collect();

        // Generate struct fields:
        // - Manifold params → generic field `inner: M0`
        // - Scalar params → concrete field `r: f32`
        let struct_fields: Vec<TokenStream> = params
            .iter()
            .map(|p| {
                let name = &p.name;
                match &p.kind {
                    ParamKind::Scalar(ty) => quote! { #name: #ty },
                    ParamKind::Manifold => {
                        let idx = self.manifold_indices[&name.to_string()];
                        let generic_name = &generic_names[idx];
                        quote! { #name: #generic_name }
                    }
                }
            })
            .collect();

        // Generate struct field names for construction
        let field_names: Vec<_> = params.iter().map(|p| &p.name).collect();

        // Generate closure parameters:
        // - Manifold params → just the name (type inferred from generic)
        // - Scalar params → `name: type`
        let closure_params: Vec<TokenStream> = params
            .iter()
            .map(|p| {
                let name = &p.name;
                match &p.kind {
                    ParamKind::Scalar(ty) => quote! { #name: #ty },
                    ParamKind::Manifold => quote! { #name },
                }
            })
            .collect();

        // Run annotation pass to collect literals and assign Var indices
        // Literals get indices 0..m-1, params get indices m..m+n-1
        let annotation_ctx = AnnotationCtx::new(params.len(), self.use_jet_wrapper);
        let (annotated_body, _, collected_literals) = annotate(&self.analyzed.def.body, annotation_ctx);
        let literal_count = collected_literals.len();
        self.collected_literals = collected_literals;

        // Adjust param indices to account for literals (literals are innermost)
        // Original: param0→n-1, param1→n-2, ..., param_{n-1}→0
        // Adjusted: param0→n+m-1, param1→n+m-2, ..., param_{n-1}→m
        for (_, idx) in self.param_indices.iter_mut() {
            *idx += literal_count;
        }

        // Transform and emit the body as a ZST expression
        let body = self.emit_annotated_expr(&annotated_body);

        // Generate the Peano type imports needed based on parameter + literal count
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
        let (manifold_eval_stmts, at_binding) = self.emit_unified_binding();

        // Handle empty vs non-empty params, with or without generics
        if params.is_empty() {
            // No parameters - simple case
            quote! {
                {
                    type __Domain = #domain_type;
                    type __Scalar = #scalar_type;

                    #[derive(Clone)]
                    struct __Kernel;

                    impl ::pixelflow_core::Manifold<__Domain> for __Kernel {
                        type Output = #output_type;

                        #[inline(always)]
                        fn eval(
                            &self,
                            __p: __Domain,
                        ) -> #output_type {
                            use ::pixelflow_core::{X, Y, Z, W, ManifoldExt, ManifoldCompat, Manifold, Let, Var};
                            #peano_imports

                            let __expr = { #body };
                            #at_binding
                        }
                    }

                    || __Kernel
                }
            }
        } else if manifold_count == 0 {
            // Only scalar parameters - no generics needed
            quote! {
                {
                    type __Domain = #domain_type;
                    type __Scalar = #scalar_type;

                    #[derive(Clone)]
                    struct __Kernel { #(#struct_fields),* }

                    impl ::pixelflow_core::Manifold<__Domain> for __Kernel {
                        type Output = #output_type;

                        #[inline(always)]
                        fn eval(
                            &self,
                            __p: __Domain,
                        ) -> #output_type {
                            use ::pixelflow_core::{X, Y, Z, W, ManifoldExt, ManifoldCompat, Manifold, Let, Var};
                            #peano_imports

                            let __expr = { #body };
                            #at_binding
                        }
                    }

                    |#(#closure_params),*| __Kernel { #(#field_names),* }
                }
            }
        } else {
            // Has manifold parameters - need generics and trait bounds
            // Generate trait bounds: M0: Manifold<__Domain, Output = __Scalar>
            let trait_bounds: Vec<TokenStream> = generic_names
                .iter()
                .map(|g| {
                    quote! { #g: ::pixelflow_core::Manifold<__Domain, Output = __Scalar> }
                })
                .collect();

            quote! {
                {
                    type __Domain = #domain_type;
                    type __Scalar = #scalar_type;

                    struct __Kernel<#(#generic_names),*> { #(#struct_fields),* }

                    impl<#(#generic_names),*> ::pixelflow_core::Manifold<__Domain> for __Kernel<#(#generic_names),*>
                    where
                        #(#trait_bounds),*
                    {
                        type Output = #output_type;

                        #[inline(always)]
                        fn eval(
                            &self,
                            __p: __Domain,
                        ) -> #output_type {
                            use ::pixelflow_core::{X, Y, Z, W, ManifoldExt, ManifoldCompat, Manifold, Let, Var};
                            #peano_imports

                            // Evaluate manifold parameters at current point (borrows via &self)
                            #manifold_eval_stmts

                            // Build the ZST expression tree using Var<N> for ALL parameters
                            let __expr = { #body };

                            // Wrap in nested Let bindings and evaluate
                            #at_binding
                        }
                    }

                    |#(#closure_params),*| __Kernel { #(#field_names),* }
                }
            }
        }
    }

    /// Emit the Peano type imports needed based on parameter + literal count.
    fn emit_peano_imports(&self) -> TokenStream {
        let total = self.analyzed.def.params.len() + self.collected_literals.len();
        if total == 0 {
            return quote! {};
        }

        // Import N0..N{total-1}
        let imports: Vec<TokenStream> = (0..total)
            .map(|i| {
                let name = syn::Ident::new(PEANO_INDICES[i], proc_macro2::Span::call_site());
                quote! { #name }
            })
            .collect();

        quote! { use ::pixelflow_core::{ #(#imports),* }; }
    }

    /// Emit unified Let/Var binding for params and literals.
    ///
    /// Returns two things:
    /// 1. Statements to evaluate manifold parameters at `__p`
    /// 2. Nested Let::new() calls that extend the domain
    ///
    /// Binding order (outermost to innermost):
    /// - Parameters (outermost) → highest Var indices
    /// - Literals (innermost) → lowest Var indices (0, 1, ...)
    ///
    /// This ensures the expression tree can use pure Var<N> references.
    fn emit_unified_binding(&self) -> (TokenStream, TokenStream) {
        let params = &self.analyzed.def.params;

        if params.is_empty() && self.collected_literals.is_empty() {
            // No bindings needed - evaluate expression directly
            return (quote! {}, quote! { __expr.eval(__p) });
        }

        // 1. Generate statements to evaluate manifold params
        let manifold_eval_stmts: Vec<TokenStream> = params
            .iter()
            .filter_map(|p| {
                if let ParamKind::Manifold = &p.kind {
                    let name = &p.name;
                    let idx = self.manifold_indices[&name.to_string()];
                    let var_name = format_ident!("__m{}", idx);
                    Some(quote! {
                        let #var_name = self.#name.eval(__p);
                    })
                } else {
                    None
                }
            })
            .collect();

        // 2. Build nested Let bindings from outside to inside.
        // Order: params first (outermost), then literals (innermost)
        let mut result = quote! { __expr };

        // First wrap literals (innermost - reversed so last literal is innermost)
        for collected in self.collected_literals.iter().rev() {
            let lit = &collected.lit;
            let lit_value = if self.use_jet_wrapper {
                quote! { __Scalar::constant(::pixelflow_core::Field::from(#lit)) }
            } else {
                quote! { #lit }
            };
            result = quote! {
                Let::new(#lit_value, #result)
            };
        }

        // Then wrap params (outermost - reversed so last param is innermost among params)
        for param in params.iter().rev() {
            let name = &param.name;

            let param_value = match &param.kind {
                ParamKind::Manifold => {
                    // Use the pre-evaluated manifold result
                    let idx = self.manifold_indices[&name.to_string()];
                    let var_name = format_ident!("__m{}", idx);
                    quote! { #var_name }
                }
                ParamKind::Scalar(_) => {
                    // For Jet domains, wrap f32 params as constant jets
                    if self.use_jet_wrapper {
                        quote! { __Scalar::constant(::pixelflow_core::Field::from(self.#name)) }
                    } else {
                        quote! { self.#name }
                    }
                }
            };

            result = quote! {
                Let::new(#param_value, #result)
            };
        }

        // Evaluate the whole Let-wrapped expression on the input domain
        let at_binding = quote! { #result.eval(__p) };

        (quote! { #(#manifold_eval_stmts)* }, at_binding)
    }

    /// Emit code for an annotated expression (pure, no mutation).
    ///
    /// Literals with var_index become Var<N> references.
    /// This is the clean functional version that works with the annotation pass.
    fn emit_annotated_expr(&self, expr: &AnnotatedExpr) -> TokenStream {
        match expr {
            AnnotatedExpr::Ident(ident_expr) => {
                let name = &ident_expr.name;
                let name_str = name.to_string();

                match self.analyzed.symbols.lookup(&name_str) {
                    Some(symbol) => match symbol.kind {
                        SymbolKind::Intrinsic => {
                            // Intrinsics (X, Y, Z, W) emitted as-is
                            quote! { #name }
                        }
                        SymbolKind::Parameter | SymbolKind::ManifoldParam => {
                            // Both scalar and manifold parameters become Var::<N{index}>::new()
                            if let Some(&idx) = self.param_indices.get(&name_str) {
                                if idx < PEANO_INDICES.len() {
                                    let peano_name = syn::Ident::new(
                                        PEANO_INDICES[idx],
                                        proc_macro2::Span::call_site(),
                                    );
                                    quote! { Var::<#peano_name>::new() }
                                } else {
                                    let err_msg = format!(
                                        "kernel! supports max 8 bindings, found index {}",
                                        idx
                                    );
                                    quote! { compile_error!(#err_msg) }
                                }
                            } else {
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

            AnnotatedExpr::Literal(lit) => {
                match lit.var_index {
                    Some(collection_idx) => {
                        // Literal has a Var binding (Jet mode).
                        // The collection_idx is the order in which literals were encountered.
                        // We need to invert this because the Let binding order puts:
                        // - Last literal (innermost Let) at N0
                        // - First literal at N(literal_count - 1)
                        let literal_count = self.collected_literals.len();
                        let var_idx = (literal_count - 1) - collection_idx;

                        if var_idx < PEANO_INDICES.len() {
                            let peano_name = syn::Ident::new(
                                PEANO_INDICES[var_idx],
                                proc_macro2::Span::call_site(),
                            );
                            quote! { Var::<#peano_name>::new() }
                        } else {
                            let err_msg = format!(
                                "kernel! supports max 8 bindings, found index {}",
                                var_idx
                            );
                            quote! { compile_error!(#err_msg) }
                        }
                    }
                    None => {
                        // No binding needed (non-Jet mode), emit literal directly
                        let l = &lit.lit;
                        quote! { #l }
                    }
                }
            }

            AnnotatedExpr::Binary(binary) => {
                let lhs = self.emit_annotated_expr(&binary.lhs);
                let rhs = self.emit_annotated_expr(&binary.rhs);

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

            AnnotatedExpr::Unary(unary) => {
                let operand = self.emit_annotated_expr(&unary.operand);
                match unary.op {
                    UnaryOp::Neg => quote! { #operand.neg() },
                    UnaryOp::Not => quote! { !#operand },
                }
            }

            AnnotatedExpr::MethodCall(call) => {
                let receiver = self.emit_annotated_expr(&call.receiver);
                let method = &call.method;
                let args: Vec<TokenStream> = call.args.iter()
                    .map(|a| self.emit_annotated_expr(a))
                    .collect();

                if args.is_empty() {
                    quote! { #receiver.#method() }
                } else {
                    quote! { #receiver.#method(#(#args),*) }
                }
            }

            AnnotatedExpr::Block(block) => {
                let stmts: Vec<TokenStream> = block.stmts.iter()
                    .map(|s| self.emit_annotated_stmt(s))
                    .collect();

                let final_expr = block.expr.as_ref().map(|e| self.emit_annotated_expr(e));

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

            AnnotatedExpr::Paren(inner) => {
                let inner_code = self.emit_annotated_expr(inner);
                quote! { (#inner_code) }
            }

            AnnotatedExpr::Verbatim(syn_expr) => {
                syn_expr.to_token_stream()
            }
        }
    }

    fn emit_annotated_stmt(&self, stmt: &AnnotatedStmt) -> TokenStream {
        match stmt {
            AnnotatedStmt::Let(let_stmt) => {
                let name = &let_stmt.name;
                let init = self.emit_annotated_expr(&let_stmt.init);

                match &let_stmt.ty {
                    Some(ty) => quote! { let #name: #ty = #init; },
                    None => quote! { let #name = #init; },
                }
            }
            AnnotatedStmt::Expr(expr) => {
                let code = self.emit_annotated_expr(expr);
                quote! { #code; }
            }
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

    #[test]
    fn emit_manifold_param() {
        let input = quote! { |inner: kernel, r: f32| inner - r };
        let output = compile(input);
        let output_str = output.to_string();

        // Should have generic struct: struct __Kernel<M0> { inner: M0, r: f32 }
        assert!(
            output_str.contains("struct __Kernel < M0 >"),
            "Expected generic struct, got: {}",
            output_str
        );

        // Should have trait bound for M0
        assert!(
            output_str.contains("M0 : :: pixelflow_core :: Manifold < __Domain"),
            "Expected trait bound for M0, got: {}",
            output_str
        );

        // inner → Var::<N1>, r → Var::<N0>
        assert!(
            output_str.contains("Var :: < N1 >"),
            "Expected Var::<N1> for inner"
        );
        assert!(
            output_str.contains("Var :: < N0 >"),
            "Expected Var::<N0> for r"
        );

        // Should evaluate manifold: let __m0 = self.inner.eval(__p);
        assert!(
            output_str.contains("let __m0 = self . inner . eval (__p)"),
            "Expected manifold eval, got: {}",
            output_str
        );
    }

    #[test]
    fn emit_multiple_manifold_params() {
        let input = quote! { |a: kernel, b: kernel| a + b };
        let output = compile(input);
        let output_str = output.to_string();

        // Should have two generic params
        assert!(
            output_str.contains("struct __Kernel < M0 , M1 >"),
            "Expected two generics, got: {}",
            output_str
        );

        // Both params become Var references
        assert!(
            output_str.contains("Var :: < N1 >"),
            "Expected Var::<N1> for a"
        );
        assert!(
            output_str.contains("Var :: < N0 >"),
            "Expected Var::<N0> for b"
        );

        // Should have two manifold evals
        assert!(
            output_str.contains("let __m0 = self . a . eval (__p)"),
            "Expected __m0 eval"
        );
        assert!(
            output_str.contains("let __m1 = self . b . eval (__p)"),
            "Expected __m1 eval"
        );
    }
}
