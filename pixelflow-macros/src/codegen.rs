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

// ============================================================================
// Shared Constants
// ============================================================================

/// Standard imports used in all generated eval functions.
///
/// This avoids duplicating the import list in every code path.
fn standard_imports() -> TokenStream {
    quote! {
        use ::pixelflow_core::{
            X, Y, Z, W,
            ManifoldExt, ManifoldCompat, Manifold,
            Let, Var, WithContext, CtxVar,
            A0, A1, A2, A3,
            GradientMag2D, GradientMag3D,
            Antialias2D, Antialias3D,
            Curvature2D, Normalized2D, Normalized3D,
            V, DX, DY, DZ, DXX, DXY, DYY,
            // Ops for explicit construction (bypasses trait inference)
            Min, Max, Sqrt, Abs, Floor,
            Sin, Cos,
            Exp, Exp2, Log2,
            Select, MulAdd, Rsqrt,
        };
    }
}

/// Sort indexed values by index and extract just the values.
///
/// This is a common pattern: collect (index, value) pairs, sort by index, keep values.
fn sort_by_index(indexed: impl IntoIterator<Item = (usize, TokenStream)>) -> Vec<TokenStream> {
    let mut pairs: Vec<_> = indexed.into_iter().collect();
    pairs.sort_by_key(|(idx, _)| *idx);
    pairs.into_iter().map(|(_, val)| val).collect()
}

/// Build a tuple expression from values, handling the single-element case.
///
/// Rust requires a trailing comma for single-element tuples: `(x,)` not `(x)`.
fn build_tuple(values: &[TokenStream]) -> TokenStream {
    match values.len() {
        0 => quote! { () },
        1 => {
            let val = &values[0];
            quote! { (#val,) }
        }
        _ => quote! { (#(#values),*) },
    }
}

/// Build an array expression wrapped in a single-element tuple.
///
/// Produces `([val0, val1, ...],)` - the format expected by WithContext
/// for the array-based context system.
fn build_array(values: &[TokenStream]) -> TokenStream {
    if values.is_empty() {
        quote! { () }
    } else {
        quote! { ([#(#values),*],) }
    }
}

// ============================================================================
// Struct Emitter (Builder Pattern)
// ============================================================================

/// Builder for generating struct definitions with Manifold impls.
struct StructEmitter {
    visibility: syn::Visibility,
    name: syn::Ident,
    generic_names: Vec<syn::Ident>,
    fields: Vec<TokenStream>,
    field_names: Vec<syn::Ident>,
    constructor_params: Vec<TokenStream>,
    derives: Derives,
    domain_config: DomainConfig,
    eval_body: Option<EvalBody>,
}

/// What traits to derive on the struct.
#[derive(Clone, Copy)]
enum Derives {
    /// Clone only (default for multi-field or manifold params).
    Clone,
    /// Clone + Copy (for unit struct or single scalar param).
    CloneCopy,
}

/// Domain type configuration.
enum DomainConfig {
    /// Fixed domain: `impl Manifold<__Domain> for Struct`
    Fixed {
        domain_type: TokenStream,
        scalar_type: TokenStream,
        output_type: TokenStream,
        trait_bounds: Vec<TokenStream>,
    },
    /// Generic domain: `impl<__P: Spatial> Manifold<__P> for Struct`
    Generic {
        output_type: TokenStream,
    },
}

/// The eval function body.
struct EvalBody {
    imports: TokenStream,
    peano_imports: TokenStream,
    pre_eval_stmts: TokenStream,
    expr: TokenStream,
    binding: TokenStream,
}

impl StructEmitter {
    fn new(visibility: syn::Visibility, name: syn::Ident) -> Self {
        Self {
            visibility,
            name,
            generic_names: Vec::new(),
            fields: Vec::new(),
            field_names: Vec::new(),
            constructor_params: Vec::new(),
            derives: Derives::Clone,
            domain_config: DomainConfig::Generic {
                output_type: quote! { ::pixelflow_core::Field },
            },
            eval_body: None,
        }
    }

    fn with_generics(mut self, names: Vec<syn::Ident>) -> Self {
        self.generic_names = names;
        self
    }

    fn with_derives(mut self, derives: Derives) -> Self {
        self.derives = derives;
        self
    }

    fn with_fields(
        mut self,
        fields: Vec<TokenStream>,
        field_names: Vec<syn::Ident>,
        constructor_params: Vec<TokenStream>,
    ) -> Self {
        self.fields = fields;
        self.field_names = field_names;
        self.constructor_params = constructor_params;
        self
    }

    fn with_fixed_domain(
        mut self,
        domain_type: TokenStream,
        scalar_type: TokenStream,
        output_type: TokenStream,
        trait_bounds: Vec<TokenStream>,
    ) -> Self {
        self.domain_config = DomainConfig::Fixed {
            domain_type,
            scalar_type,
            output_type,
            trait_bounds,
        };
        self
    }

    fn with_eval_body(
        mut self,
        imports: TokenStream,
        peano_imports: TokenStream,
        pre_eval_stmts: TokenStream,
        expr: TokenStream,
        binding: TokenStream,
    ) -> Self {
        self.eval_body = Some(EvalBody {
            imports,
            peano_imports,
            pre_eval_stmts,
            expr,
            binding,
        });
        self
    }

    fn build(self) -> TokenStream {
        let vis = &self.visibility;
        let name = &self.name;
        let generics = &self.generic_names;
        let fields = &self.fields;
        let field_names = &self.field_names;
        let ctor_params = &self.constructor_params;

        // Build struct definition
        let struct_def = if self.fields.is_empty() {
            // Unit struct
            quote! { #[derive(Clone, Copy)] #vis struct #name; }
        } else if generics.is_empty() {
            // Non-generic struct
            match self.derives {
                Derives::CloneCopy => quote! {
                    #[derive(Clone, Copy)]
                    #vis struct #name { #(#fields),* }
                },
                Derives::Clone => quote! {
                    #[derive(Clone)]
                    #vis struct #name { #(#fields),* }
                },
            }
        } else {
            // Generic struct - manual Clone/Copy impls
            quote! {
                #vis struct #name<#(#generics),*> { #(#fields),* }

                impl<#(#generics: Clone),*> Clone for #name<#(#generics),*> {
                    fn clone(&self) -> Self {
                        Self { #(#field_names: self.#field_names.clone()),* }
                    }
                }

                impl<#(#generics: Copy),*> Copy for #name<#(#generics),*> {}
            }
        };

        // Build constructor
        let constructor = if self.fields.is_empty() {
            quote! {
                impl #name {
                    pub fn new() -> Self { Self }
                }
                impl Default for #name {
                    fn default() -> Self { Self::new() }
                }
            }
        } else if generics.is_empty() {
            quote! {
                impl #name {
                    pub fn new(#(#ctor_params),*) -> Self {
                        Self { #(#field_names),* }
                    }
                }
            }
        } else {
            quote! {
                impl<#(#generics),*> #name<#(#generics),*> {
                    pub fn new(#(#ctor_params),*) -> Self {
                        Self { #(#field_names),* }
                    }
                }
            }
        };

        // Build Manifold impl
        let eval_body = self.eval_body.expect("eval_body required");
        let imports = &eval_body.imports;
        let peano_imports = &eval_body.peano_imports;
        let pre_eval = &eval_body.pre_eval_stmts;
        let expr = &eval_body.expr;
        let binding = &eval_body.binding;

        let manifold_impl = match &self.domain_config {
            DomainConfig::Fixed { domain_type, scalar_type, output_type, trait_bounds } => {
                if generics.is_empty() {
                    quote! {
                        type __Domain = #domain_type;
                        type __Scalar = #scalar_type;

                        impl ::pixelflow_core::Manifold<__Domain> for #name {
                            type Output = #output_type;

                            #[inline(always)]
                            fn eval(&self, __p: __Domain) -> #output_type {
                                #imports
                                #peano_imports
                                #pre_eval
                                let __expr = { #expr };
                                #binding
                            }
                        }
                    }
                } else {
                    quote! {
                        type __Domain = #domain_type;
                        type __Scalar = #scalar_type;

                        impl<#(#generics),*> ::pixelflow_core::Manifold<__Domain> for #name<#(#generics),*>
                        where
                            #(#trait_bounds),*
                        {
                            type Output = #output_type;

                            #[inline(always)]
                            fn eval(&self, __p: __Domain) -> #output_type {
                                #imports
                                #peano_imports
                                #pre_eval
                                let __expr = { #expr };
                                #binding
                            }
                        }
                    }
                }
            }

            DomainConfig::Generic { output_type } => {
                if generics.is_empty() {
                    quote! {
                        impl<__P> ::pixelflow_core::Manifold<__P> for #name
                        where
                            __P: Copy + Send + Sync + ::pixelflow_core::Spatial,
                        {
                            type Output = #output_type;

                            #[inline(always)]
                            fn eval(&self, __p: __P) -> #output_type {
                                #imports
                                #peano_imports
                                #pre_eval
                                let __expr = { #expr };
                                #binding
                            }
                        }
                    }
                } else {
                    quote! {
                        impl<#(#generics),*, __P> ::pixelflow_core::Manifold<__P> for #name<#(#generics),*>
                        where
                            __P: Copy + Send + Sync + ::pixelflow_core::Spatial,
                            #(#generics: ::pixelflow_core::Manifold<__P, Output = #output_type>),*,
                        {
                            type Output = #output_type;

                            #[inline(always)]
                            fn eval(&self, __p: __P) -> #output_type {
                                #imports
                                #peano_imports
                                #pre_eval
                                let __expr = { #expr };
                                #binding
                            }
                        }
                    }
                }
            }
        };

        quote! {
            #struct_def
            #constructor
            #manifold_impl
        }
    }
}

// ============================================================================
// Code Emitter
// ============================================================================

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
        let n = analyzed.def.params.len();

        // Compute Peano indices for ALL parameters (both scalar and manifold).
        // First param (i=0) gets highest index (n-1), last param gets 0.
        let param_indices: HashMap<String, usize> = analyzed
            .def
            .params
            .iter()
            .enumerate()
            .map(|(i, param)| (param.name.to_string(), n - 1 - i))
            .collect();

        // Compute generic type indices for manifold parameters (M0, M1, ...).
        // Numbered in declaration order.
        let manifold_indices: HashMap<String, usize> = analyzed
            .def
            .params
            .iter()
            .filter(|p| matches!(p.kind, ParamKind::Manifold))
            .enumerate()
            .map(|(idx, param)| (param.name.to_string(), idx))
            .collect();

        // Determine if we need to wrap literals (any non-Field domain type needs from_f32 wrapping)
        let use_jet_wrapper = analyzed
            .def
            .domain_ty
            .as_ref()
            .or(analyzed.def.return_ty.as_ref())
            .map(|ty| {
                let ty_str = quote! { #ty }.to_string();
                // Field doesn't need wrapping, everything else does
                !(ty_str == "Field" || ty_str.contains(":: Field") || ty_str.ends_with("::Field"))
            })
            .unwrap_or(false);

        CodeEmitter {
            analyzed,
            param_indices,
            manifold_indices,
            use_jet_wrapper,
            collected_literals: Vec::new(),
        }
    }

    /// Emit the complete kernel definition.
    fn emit_kernel(&mut self) -> TokenStream {
        // Dispatch based on whether this is a named or anonymous kernel
        if let Some(ref decl) = self.analyzed.def.struct_decl {
            self.emit_named_kernel(decl.visibility.clone(), decl.name.clone())
        } else {
            self.emit_closure_kernel()
        }
    }

    /// Emit an anonymous kernel as a closure returning WithContext.
    ///
    /// This allows natural environment capture via Rust's closure semantics.
    ///
    /// Output pattern:
    /// ```ignore
    /// move |cx: f32, cy: f32| {
    ///     use ::pixelflow_core::{X, Y, Z, W, WithContext, CtxVar, ...};
    ///     let __expr = { X - CtxVar::<N0>::new() };
    ///     WithContext::new((cx, cy), __expr)
    /// }
    /// ```
    fn emit_closure_kernel(&mut self) -> TokenStream {
        let params = &self.analyzed.def.params;
        let standard_imports = standard_imports();

        // Separate Scalar params (bound via Context) from Manifold params (captured by closure)
        // This ensures Manifold params are captured by the closure naturally, avoiding trait issues
        let scalar_params: Vec<&crate::ast::Param> = params
            .iter()
            .filter(|p| matches!(p.kind, ParamKind::Scalar(_)))
            .collect();

        // Recompute param indices ONLY for scalar params
        // We must clear the existing map and rebuild it for closure context
        self.param_indices.clear();
        let n_scalar = scalar_params.len();
        for (i, param) in scalar_params.iter().enumerate() {
            // Indexing strategy: First scalar param -> highest index
            self.param_indices.insert(param.name.to_string(), n_scalar - 1 - i);
        }

        // Run annotation pass to collect literals and assign Var indices
        let annotation_ctx = AnnotationCtx::new();
        let (annotated_body, _, collected_literals) =
            annotate(&self.analyzed.def.body, annotation_ctx);
        self.collected_literals = collected_literals;

        // Adjust param indices to account for literals in context
        let literal_count = self.collected_literals.len();
        for (_, idx) in self.param_indices.iter_mut() {
            *idx += literal_count;
        }

        // Transform and emit the body as a ZST expression
        // Manifold params (not in param_indices) will be emitted as Ident (captured)
        // Scalar params ARE in param_indices, so they will be emitted as CtxVar
        let body = self.emit_annotated_expr(&annotated_body);

        // Generate the Peano type imports needed
        let peano_imports = self.emit_peano_imports();

        // Generate closure parameters with types
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

        // Build the context array for WithContext
        // All values go into a single homogeneous array
        let n = scalar_params.len();
        if n == 0 {
            // No scalar parameters - but may have literals that need context binding
            if !self.collected_literals.is_empty() {
                // Literals need to go in context array for ZST preservation
                // For Jet mode: use Computational::from_f32
                // For Field mode: use Field::from
                let lit_values = sort_by_index(self.collected_literals.iter().map(|c| {
                    let lit = &c.lit;
                    let val = if self.use_jet_wrapper {
                        let scalar_type = match (&self.analyzed.def.domain_ty, &self.analyzed.def.return_ty) {
                            (Some(ty), _) => quote! { #ty },
                            (None, Some(ty)) => quote! { #ty },
                            (None, None) => quote! { ::pixelflow_core::Field },
                        };
                        quote! { <#scalar_type as ::pixelflow_core::Computational>::from_f32(#lit) }
                    } else {
                        quote! { ::pixelflow_core::Field::from(#lit) }
                    };
                    (c.index, val)
                }));
                // Build array expression: ([val0, val1, ...],)
                let array_expr = build_array(&lit_values);

                quote! {
                    move |#(#closure_params),*| {
                        #standard_imports
                        #peano_imports

                        let __expr = { #body };
                        WithContext::new(#array_expr, __expr)
                    }
                }
            } else {
                // No scalar parameters, no literals - simple case
                quote! {
                    move |#(#closure_params),*| {
                        #standard_imports
                        #peano_imports

                        let __expr = { #body };
                        WithContext::new((), __expr)
                    }
                }
            }
        } else {
            // Determine scalar type for non-Field modes
            let domain_for_scalar = self.analyzed.def.domain_ty.as_ref().or(self.analyzed.def.return_ty.as_ref());
            let scalar_type = match domain_for_scalar {
                Some(ty) => {
                    let ty_str = quote! { #ty }.to_string();
                    if ty_str == "Field" || ty_str.contains(":: Field") || ty_str.ends_with("::Field") {
                        None
                    } else {
                        Some(quote! { #ty })
                    }
                }
                None => None,
            };

            // Build context array with BOTH params AND literals
            let scalar_ty_ref = scalar_type.as_ref();

            // Build indexed param values (ONLY scalar params)
            let param_values = scalar_params.iter().map(|param| {
                let idx = self.param_indices[&param.name.to_string()];
                let param_name = &param.name;
                let value = match (&param.kind, scalar_ty_ref) {
                    (ParamKind::Scalar(_), Some(sty)) => {
                        quote! { <#sty as ::pixelflow_core::Computational>::from_f32(#param_name) }
                    }
                    (ParamKind::Scalar(_), None) => {
                        quote! { ::pixelflow_core::Field::from(#param_name) }
                    }
                    _ => unreachable!("Manifold param in scalar list"),
                };
                (idx, value)
            });

            // Build indexed literal values
            let literal_values: Box<dyn Iterator<Item = (usize, TokenStream)>> = if self.use_jet_wrapper {
                let sty = scalar_type.clone().unwrap_or(quote! { ::pixelflow_core::Field });
                Box::new(self.collected_literals.iter().map(move |c| {
                    let lit = &c.lit;
                    let val = quote! { <#sty as ::pixelflow_core::Computational>::from_f32(#lit) };
                    (c.index, val)
                }))
            } else {
                Box::new(self.collected_literals.iter().map(|c| {
                    let lit = &c.lit;
                    let val = quote! { ::pixelflow_core::Field::from(#lit) };
                    (c.index, val)
                }))
            };

            // Combine, sort by index, build array expression
            let array_values = sort_by_index(param_values.chain(literal_values));
            let array_expr = build_array(&array_values);

            quote! {
                move |#(#closure_params),*| {
                    #standard_imports
                    #peano_imports

                    let __expr = { #body };
                    WithContext::new(#array_expr, __expr)
                }
            }
        }
    }

    /// Emit a named kernel as a struct with Manifold impl.
    fn emit_named_kernel(&mut self, visibility: syn::Visibility, name: syn::Ident) -> TokenStream {
        // (Implementation remains unchanged, as it handles manifolds via pre-eval correctly)
        // But we need to use self.emit_annotated_expr which uses self.param_indices.
        // param_indices is computed correctly in new() for named kernels.

        let params = &self.analyzed.def.params;
        let standard_imports = standard_imports();

        // Count manifold parameters for generic type generation
        let manifold_count = self.manifold_indices.len();

        // Generate generic type parameter names (M0, M1, ...)
        let generic_names: Vec<syn::Ident> = (0..manifold_count)
            .map(|i| format_ident!("M{}", i))
            .collect();

        // Generate struct fields with pub visibility
        let struct_fields: Vec<TokenStream> = params
            .iter()
            .map(|p| {
                let field_name = &p.name;
                match &p.kind {
                    ParamKind::Scalar(ty) => quote! { pub #field_name: #ty },
                    ParamKind::Manifold => {
                        let idx = self.manifold_indices[&field_name.to_string()];
                        let generic_name = &generic_names[idx];
                        quote! { pub #field_name: #generic_name }
                    }
                }
            })
            .collect();

        // Generate struct field names for construction
        let field_names: Vec<_> = params.iter().map(|p| p.name.clone()).collect();

        // Generate constructor parameters
        let constructor_params: Vec<TokenStream> = params
            .iter()
            .map(|p| {
                let field_name = &p.name;
                match &p.kind {
                    ParamKind::Scalar(ty) => quote! { #field_name: #ty },
                    ParamKind::Manifold => {
                        let idx = self.manifold_indices[&field_name.to_string()];
                        let generic_name = &generic_names[idx];
                        quote! { #field_name: #generic_name }
                    }
                }
            })
            .collect();

        // Run annotation pass to collect literals and assign Var indices
        let annotation_ctx = AnnotationCtx::new();
        let (annotated_body, _, collected_literals) =
            annotate(&self.analyzed.def.body, annotation_ctx);
        self.collected_literals = collected_literals;

        // Always adjust param indices to account for literals in context
        // Literals occupy indices 0..literal_count, params start after
        // This ensures the expression tree remains ZST (all values via CtxVar)
        let literal_count = self.collected_literals.len();
        for (_, idx) in self.param_indices.iter_mut() {
            *idx += literal_count;
        }

        // Transform and emit the body as a ZST expression
        let body = self.emit_annotated_expr(&annotated_body);

        // Generate the Peano type imports needed
        let peano_imports = self.emit_peano_imports();

        // Determine output type and domain type
        let (output_type, domain_type, scalar_type) = match (&self.analyzed.def.domain_ty, &self.analyzed.def.return_ty) {
            (Some(domain), Some(output)) => (
                quote! { #output },
                quote! { (#domain, #domain, #domain, #domain) },
                quote! { #domain },
            ),
            (None, Some(ty)) => (
                quote! { #ty },
                quote! { (#ty, #ty, #ty, #ty) },
                quote! { #ty },
            ),
            (None, None) | (Some(_), None) => (
                quote! { ::pixelflow_core::Field },
                quote! { (::pixelflow_core::Field, ::pixelflow_core::Field, ::pixelflow_core::Field, ::pixelflow_core::Field) },
                quote! { ::pixelflow_core::Field },
            ),
        };

        // Generate the binding
        let (manifold_eval_stmts, at_binding) = self.emit_unified_binding();

        // Determine derives and domain config based on parameter configuration
        let has_fixed_domain = self.analyzed.def.domain_ty.is_some() || self.analyzed.def.return_ty.is_some();

        // Build trait bounds for generic structs with fixed domain
        let trait_bounds: Vec<TokenStream> = generic_names
            .iter()
            .map(|g| quote! { #g: ::pixelflow_core::Manifold<__Domain, Output = __Scalar> })
            .collect();

        let derives = if params.is_empty() {
            Derives::CloneCopy
        } else if manifold_count == 0 && params.len() == 1 {
            Derives::CloneCopy
        } else {
            Derives::Clone
        };

        // Build the emitter
        let mut emitter = StructEmitter::new(visibility, name)
            .with_generics(generic_names.clone())
            .with_derives(derives)
            .with_fields(struct_fields, field_names, constructor_params);

        if has_fixed_domain || manifold_count == 0 {
            emitter = emitter.with_fixed_domain(
                domain_type,
                scalar_type,
                output_type,
                trait_bounds,
            );
        }

        emitter = emitter.with_eval_body(
            standard_imports,
            peano_imports,
            manifold_eval_stmts,
            body,
            at_binding,
        );

        emitter.build()
    }

    /// Emit imports for array-based context system.
    fn emit_peano_imports(&self) -> TokenStream {
        quote! {}
    }

    /// Emit unified WithContext/CtxVar binding for params (and Let for literals).
    fn emit_unified_binding(&self) -> (TokenStream, TokenStream) {
        let params = &self.analyzed.def.params;

        if params.is_empty() && self.collected_literals.is_empty() {
            return (quote! {}, quote! { __expr.eval(__p) });
        }

        let manifold_count = self.manifold_indices.len();
        let has_scalar_params = params
            .iter()
            .any(|p| matches!(p.kind, ParamKind::Scalar(_)));
        let needs_pre_eval = manifold_count > 0 && (manifold_count > 1 || has_scalar_params);

        let has_fixed_domain = self.analyzed.def.domain_ty.is_some() || self.analyzed.def.return_ty.is_some();
        let scalar_type_token = if has_fixed_domain {
            quote! { __Scalar }
        } else {
            quote! { ::pixelflow_core::Field }
        };

        let mut pre_eval_stmts = Vec::new();
        if needs_pre_eval {
            for param in params.iter() {
                if matches!(param.kind, ParamKind::Manifold) {
                    let name = &param.name;
                    let eval_name = syn::Ident::new(
                        &format!("__eval_{}", name),
                        proc_macro2::Span::call_site(),
                    );
                    pre_eval_stmts.push(quote! {
                        let #eval_name: #scalar_type_token = self.#name.eval(__p);
                    });
                }
            }
        }

        let result = if params.is_empty() {
            quote! { __expr }
        } else {
            let n = params.len();
            let mut indexed_params: Vec<(usize, TokenStream)> = Vec::new();

            for param in params.iter() {
                let name = &param.name;
                let idx = self.param_indices[&name.to_string()];

                let param_value = match &param.kind {
                    ParamKind::Manifold => {
                        if needs_pre_eval {
                            let eval_name = syn::Ident::new(
                                &format!("__eval_{}", name),
                                proc_macro2::Span::call_site(),
                            );
                            quote! { #eval_name }
                        } else {
                            quote! { &self.#name }
                        }
                    }
                    ParamKind::Scalar(_) => {
                        if self.use_jet_wrapper {
                            quote! { <__Scalar as ::pixelflow_core::Computational>::from_f32(self.#name) }
                        } else if needs_pre_eval {
                            quote! { #scalar_type_token::from(self.#name) }
                        } else {
                            quote! { self.#name }
                        }
                    }
                };

                indexed_params.push((idx, param_value));
            }

            indexed_params.sort_by_key(|(idx, _)| *idx);
            let param_values: Vec<_> = indexed_params.into_iter().map(|(_, val)| val).collect();

            if param_values.len() == 1 {
                let val = &param_values[0];
                quote! {
                    WithContext::new((#val,), __expr)
                }
            } else {
                quote! {
                    WithContext::new((#(#param_values),*), __expr)
                }
            }
        };

        let result = if !self.collected_literals.is_empty() {
            let mut wrapped = result;
            for collected in self.collected_literals.iter().rev() {
                let lit = &collected.lit;
                let lit_value = if self.use_jet_wrapper {
                    quote! { <__Scalar as ::pixelflow_core::Computational>::from_f32(#lit) }
                } else {
                    quote! { ::pixelflow_core::Field::from(#lit) }
                };
                wrapped = quote! {
                    Let::new(#lit_value, #wrapped)
                };
            }
            wrapped
        } else {
            result
        };

        let at_binding = quote! { #result.eval(__p) };

        let stmts = if pre_eval_stmts.is_empty() {
            quote! {}
        } else {
            quote! { #(#pre_eval_stmts)* }
        };
        (stmts, at_binding)
    }

    fn emit_annotated_expr(&self, expr: &AnnotatedExpr) -> TokenStream {
        match expr {
            AnnotatedExpr::Ident(ident_expr) => {
                let name = &ident_expr.name;
                let name_str = name.to_string();

                match self.analyzed.symbols.lookup(&name_str) {
                    Some(symbol) => match symbol.kind {
                        SymbolKind::Intrinsic => {
                            quote! { #name }
                        }
                        SymbolKind::Parameter | SymbolKind::ManifoldParam => {
                            if let Some(&idx) = self.param_indices.get(&name_str) {
                                quote! { CtxVar::<A0, #idx>::new() }
                            } else {
                                quote! { #name }
                            }
                        }
                        SymbolKind::Local => {
                            quote! { #name }
                        }
                    },
                    None => {
                        quote! { #name }
                    }
                }
            }

            AnnotatedExpr::Literal(lit) => {
                if let Some(var_idx) = lit.var_index {
                    quote! { CtxVar::<A0, #var_idx>::new() }
                } else {
                    let l = &lit.lit;
                    if self.use_jet_wrapper {
                        quote! { <__Scalar as ::pixelflow_core::Computational>::from_f32(#l) }
                    } else {
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
                    BinaryOp::BitAnd => quote! { #lhs & #rhs },
                    BinaryOp::BitOr => quote! { #lhs | #rhs },
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
                let method_name = method.to_string();
                let args: Vec<TokenStream> = call
                    .args
                    .iter()
                    .map(|a| self.emit_annotated_expr(a))
                    .collect();

                // Explicitly map known methods to struct construction to avoid trait inference issues
                match method_name.as_str() {
                    "min" if args.len() == 1 => {
                        let arg = &args[0];
                        quote! { Min(#receiver, #arg) }
                    }
                    "max" if args.len() == 1 => {
                        let arg = &args[0];
                        quote! { Max(#receiver, #arg) }
                    }
                    "sqrt" if args.is_empty() => quote! { Sqrt(#receiver) },
                    "abs" if args.is_empty() => quote! { Abs(#receiver) },
                    "floor" if args.is_empty() => quote! { Floor(#receiver) },
                    "rsqrt" if args.is_empty() => quote! { Rsqrt(#receiver) },
                    "sin" if args.is_empty() => quote! { Sin(#receiver) },
                    "cos" if args.is_empty() => quote! { Cos(#receiver) },
                    "exp" if args.is_empty() => quote! { Exp(#receiver) },
                    "exp2" if args.is_empty() => quote! { Exp2(#receiver) },
                    "log2" if args.is_empty() => quote! { Log2(#receiver) },
                    "select" if args.len() == 2 => {
                        let if_true = &args[0];
                        let if_false = &args[1];
                        quote! { Select { cond: #receiver, if_true: #if_true, if_false: #if_false } }
                    }
                    "mul_add" if args.len() == 2 => {
                        let b = &args[0];
                        let c = &args[1];
                        quote! { MulAdd(#receiver, #b, #c) }
                    }
                    _ => {
                        // Fallback to method call
                        if args.is_empty() {
                            quote! { #receiver.#method() }
                        } else {
                            quote! { #receiver.#method(#(#args),*) }
                        }
                    }
                }
            }

            AnnotatedExpr::Call(call) => {
                // Free function call: V(m), DX(expr), etc.
                let func = &call.func;
                let args: Vec<TokenStream> = call
                    .args
                    .iter()
                    .map(|a| self.emit_annotated_expr(a))
                    .collect();

                if args.is_empty() {
                    quote! { #func() }
                } else {
                    quote! { #func(#(#args),*) }
                }
            }

            AnnotatedExpr::Block(block) => {
                let stmts: Vec<TokenStream> = block
                    .stmts
                    .iter()
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

            AnnotatedExpr::Verbatim(syn_expr) => syn_expr.to_token_stream(),
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
        assert!(output_str.contains("move | cx : f32 |"));
        assert!(output_str.contains("CtxVar :: < A0 , 0usize >"));
        assert!(output_str.contains("WithContext :: new"));
    }

    #[test]
    fn emit_manifold_param() {
        let input = quote! { |inner: kernel, r: f32| inner - r };
        let output = compile(input);
        let output_str = output.to_string();
        assert!(output_str.contains("move | inner , r : f32 |"));
        // inner should be captured (not CtxVar)
        assert!(output_str.contains("inner - CtxVar :: < A0 , 0usize >"));
        // r should be CtxVar
        assert!(output_str.contains("CtxVar :: < A0 , 0usize >"));
        assert!(output_str.contains("WithContext :: new"));
    }

    #[test]
    fn emit_method_mapping() {
        let input = quote! { |a: kernel, b: kernel| a.min(b) };
        let output = compile(input);
        let output_str = output.to_string();
        // Should emit Min(a, b) instead of a.min(b)
        assert!(output_str.contains("Min ( a , b )"));
    }
}
