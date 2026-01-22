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
// Binding Strategy
// ============================================================================

/// Strategy for binding parameter and literal values to the expression.
///
/// ## Categorical Perspective
///
/// Binding extends the domain of a manifold with captured values.
/// This is the "contramap" half of the profunctor: we're modifying
/// how coordinates are supplied, not what values are produced.
///
/// The three strategies represent different ways to encode this extension:
/// - **FlatTuple**: Single WithContext with a flat tuple (efficient for many params)
/// - **NestedLet**: Nested Let bindings (original approach, exponential trait solving)
/// - **Mixed**: Combine both for params + literals
#[derive(Debug)]
enum BindingStrategy {
    /// No bindings needed - evaluate expression directly.
    None,

    /// Flat tuple binding: `WithContext::new((v0, v1, ...), expr)`
    ///
    /// Most efficient for multiple bindings. Avoids trait solver explosion.
    FlatTuple {
        /// Values in tuple order (already sorted by index).
        values: Vec<TokenStream>,
    },

    /// Nested Let binding: `Let::new(v0, Let::new(v1, expr))`
    ///
    /// Legacy approach. Still used for edge cases.
    NestedLet {
        /// Values in binding order (outermost to innermost).
        values: Vec<TokenStream>,
    },

    /// Mixed strategy: FlatTuple for params, NestedLet for literals.
    ///
    /// Used in Jet mode where literals need Let wrapping but params use WithContext.
    Mixed {
        /// Params for WithContext tuple.
        param_tuple: Vec<TokenStream>,
        /// Literals for nested Let.
        literal_lets: Vec<TokenStream>,
    },
}

impl BindingStrategy {
    /// Emit the binding wrapper around an expression.
    ///
    /// Returns token stream that evaluates to the bound expression's result.
    fn emit(self, expr: TokenStream) -> TokenStream {
        match self {
            BindingStrategy::None => {
                quote! { #expr.eval(__p) }
            }

            BindingStrategy::FlatTuple { values } => {
                let tuple = build_tuple(&values);
                quote! { WithContext::new(#tuple, #expr).eval(__p) }
            }

            BindingStrategy::NestedLet { values } => {
                // Fold right: innermost binding is last value
                values.into_iter().rev().fold(
                    quote! { #expr.eval(__p) },
                    |inner, val| quote! { Let::new(#val, #inner).eval(__p) },
                )
            }

            BindingStrategy::Mixed { param_tuple, literal_lets } => {
                let tuple = build_tuple(&param_tuple);
                let with_ctx = quote! { WithContext::new(#tuple, #expr) };

                // Wrap with Let bindings for literals
                literal_lets.into_iter().rev().fold(
                    quote! { #with_ctx.eval(__p) },
                    |inner, val| quote! { Let::new(#val, #inner).eval(__p) },
                )
            }
        }
    }
}

// ============================================================================
// Struct Emitter (Builder Pattern)
// ============================================================================

/// Builder for generating struct definitions with Manifold impls.
///
/// ## Purpose
///
/// Consolidates the 8+ code paths in `emit_named_kernel` into a single
/// builder that handles all configuration combinations:
/// - With/without generic parameters (manifold params)
/// - With/without Copy derive (single scalar vs multiple)
/// - Fixed domain type vs generic domain P
///
/// ## Usage
///
/// ```ignore
/// StructEmitter::new(visibility, name)
///     .with_generics(generic_names)
///     .with_derives(Derives::CloneCopy)
///     .with_fields(fields)
///     .with_fixed_domain(domain_type, scalar_type, output_type)
///     .with_eval_body(body, imports)
///     .build()
/// ```
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

        // Run annotation pass to collect literals and assign Var indices
        let annotation_ctx = AnnotationCtx::new();
        let (annotated_body, _, collected_literals) = annotate(&self.analyzed.def.body, annotation_ctx);
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
        let n = params.len();
        if n == 0 {
            // No parameters - but may have literals that need context binding
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
                    || {
                        #standard_imports
                        #peano_imports

                        let __expr = { #body };
                        WithContext::new(#array_expr, __expr)
                    }
                }
            } else {
                // No parameters, no literals - simple case
                quote! {
                    || {
                        #standard_imports
                        #peano_imports

                        let __expr = { #body };
                        WithContext::new((), __expr)
                    }
                }
            }
        } else {
            // Determine scalar type for non-Field modes
            // Use domain_ty if specified, otherwise return_ty. Computational::from_f32 handles embedding.
            let domain_for_scalar = self.analyzed.def.domain_ty.as_ref().or(self.analyzed.def.return_ty.as_ref());
            let scalar_type = match domain_for_scalar {
                Some(ty) => {
                    let ty_str = quote! { #ty }.to_string();
                    // Field doesn't need wrapping, everything else does
                    if ty_str == "Field" || ty_str.contains(":: Field") || ty_str.ends_with("::Field") {
                        None
                    } else {
                        Some(quote! { #ty })
                    }
                }
                None => None,
            };

            // Build context array with BOTH params AND literals
            // All values are converted to the same type and go into a single array
            let scalar_ty_ref = scalar_type.as_ref();

            // Build indexed param values
            // All scalar params must be converted to the target type (Field or Jet)
            let param_values = params.iter().map(|param| {
                let idx = self.param_indices[&param.name.to_string()];
                let param_name = &param.name;
                let value = match (&param.kind, scalar_ty_ref) {
                    (ParamKind::Scalar(_), Some(sty)) => {
                        // Non-Field domain: use Computational::from_f32
                        quote! { <#sty as ::pixelflow_core::Computational>::from_f32(#param_name) }
                    }
                    (ParamKind::Scalar(_), None) => {
                        // Field domain: convert f32 to Field
                        quote! { ::pixelflow_core::Field::from(#param_name) }
                    }
                    _ => quote! { #param_name },
                };
                (idx, value)
            });

            // Build indexed literal values - always include for ZST preservation
            // For Jet mode: use Computational::from_f32 for type conversion
            // For Field mode: use literal directly (f32 converts to Field)
            let literal_values: Box<dyn Iterator<Item = (usize, TokenStream)>> = if self.use_jet_wrapper {
                let sty = scalar_type.clone().unwrap_or(quote! { ::pixelflow_core::Field });
                Box::new(self.collected_literals.iter().map(move |c| {
                    let lit = &c.lit;
                    let val = quote! { <#sty as ::pixelflow_core::Computational>::from_f32(#lit) };
                    (c.index, val)
                }))
            } else {
                // Field mode: use literal directly, Field::from will handle conversion
                Box::new(self.collected_literals.iter().map(|c| {
                    let lit = &c.lit;
                    let val = quote! { ::pixelflow_core::Field::from(#lit) };
                    (c.index, val)
                }))
            };

            // Combine, sort by index, build array expression
            let array_values = sort_by_index(param_values.chain(literal_values));
            let array_expr = build_array(&array_values);

            // No Let bindings needed - everything is in the flat context array
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
    ///
    /// This creates a user-named struct that can be used in struct fields.
    /// Uses the StructEmitter builder to consolidate all 8 code paths.
    fn emit_named_kernel(&mut self, visibility: syn::Visibility, name: syn::Ident) -> TokenStream {
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
        let (annotated_body, _, collected_literals) = annotate(&self.analyzed.def.body, annotation_ctx);
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
        //
        // Three cases:
        // 1. `Field -> Discrete`: domain_ty=Field, return_ty=Discrete
        //    → domain = Field4, output = Discrete, scalar = Field
        // 2. `-> Jet3`: domain_ty=None, return_ty=Jet3
        //    → domain = Jet3_4, output = Jet3, scalar = Jet3 (Coordinate type)
        // 3. (nothing): domain_ty=None, return_ty=None
        //    → domain = Field4, output = Field, scalar = Field
        //
        // The scalar_type is used for wrapping f32 params via Computational::from_f32.
        let (output_type, domain_type, scalar_type) = match (&self.analyzed.def.domain_ty, &self.analyzed.def.return_ty) {
            // Explicit domain and output: `Field -> Discrete`
            (Some(domain), Some(output)) => (
                quote! { #output },
                quote! { (#domain, #domain, #domain, #domain) },
                quote! { #domain },
            ),
            // Only output specified: `-> Jet3` (output = domain for Coordinate types)
            (None, Some(ty)) => (
                quote! { #ty },
                quote! { (#ty, #ty, #ty, #ty) },
                quote! { #ty },
            ),
            // Neither specified: default to Field
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

        // Determine derives:
        // - Unit struct or single scalar param → CloneCopy
        // - Single manifold param → Clone (Copy handled conditionally in StructEmitter)
        // - Multiple params → Clone only (multi-field structs shouldn't derive Copy)
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

        // Configure domain
        if has_fixed_domain || manifold_count == 0 {
            // Fixed domain: all scalar params, or explicit domain/return type
            emitter = emitter.with_fixed_domain(
                domain_type,
                scalar_type,
                output_type,
                trait_bounds,
            );
        }
        // else: generic domain (default in StructEmitter)

        // Configure eval body
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
    ///
    /// With the array-based approach, we use const generics instead of Peano numbers.
    /// The A0, A1, A2, A3 markers are already imported in standard_imports.
    fn emit_peano_imports(&self) -> TokenStream {
        // No additional imports needed - A0, A1, A2, A3 are in standard_imports
        // Const generic indices are written directly as literals
        quote! {}
    }

    /// Emit unified WithContext/CtxVar binding for params (and Let for literals).
    ///
    /// Returns two things:
    /// 1. Statements for pre-evaluating manifold params (when mixed with scalars)
    /// 2. WithContext::new(tuple, body) for params, wrapped in Let for literals
    ///
    /// NEW APPROACH: Uses flat WithContext tuple instead of nested Let for parameters.
    /// This avoids trait solver explosion for >4 parameters.
    ///
    /// Binding order:
    /// - Parameters → WithContext flat tuple (CtxVar<N0>, CtxVar<N1>, ...)
    /// - Literals (Jet mode only) → nested Let (wraps the WithContext)
    ///
    /// ## Mixed Param Handling
    ///
    /// When manifold params are mixed with scalar params, we must eagerly evaluate
    /// the manifold params to get concrete `__Scalar` values for type unification.
    fn emit_unified_binding(&self) -> (TokenStream, TokenStream) {
        let params = &self.analyzed.def.params;

        if params.is_empty() && self.collected_literals.is_empty() {
            // No bindings needed - evaluate expression directly
            return (quote! {}, quote! { __expr.eval(__p) });
        }

        // Determine if we need to pre-evaluate manifold params
        let manifold_count = self.manifold_indices.len();
        let has_scalar_params = params.iter().any(|p| matches!(p.kind, ParamKind::Scalar(_)));
        let needs_pre_eval = manifold_count > 0 && (manifold_count > 1 || has_scalar_params);

        // Determine the scalar type to use for pre-evaluation
        // Use __Scalar if we have a fixed domain (domain_ty or return_ty specified)
        let has_fixed_domain = self.analyzed.def.domain_ty.is_some() || self.analyzed.def.return_ty.is_some();
        let scalar_type_token = if has_fixed_domain {
            quote! { __Scalar }
        } else {
            quote! { ::pixelflow_core::Field }
        };

        // Pre-evaluate manifolds to get concrete scalar values
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

        // Build the parameter binding using WithContext for flat tuple approach
        let result = if params.is_empty() {
            // No params, just the expression
            quote! { __expr }
        } else {
            // Build tuple of param values ordered by index
            // Index 0 goes to tuple position 0, index 1 to position 1, etc.
            let n = params.len();
            let mut indexed_params: Vec<(usize, TokenStream)> = Vec::new();

            for param in params.iter() {
                let name = &param.name;
                let idx = self.param_indices[&name.to_string()];

                let param_value = match &param.kind {
                    ParamKind::Manifold => {
                        if needs_pre_eval {
                            // Use pre-evaluated value for type unification
                            let eval_name = syn::Ident::new(
                                &format!("__eval_{}", name),
                                proc_macro2::Span::call_site(),
                            );
                            quote! { #eval_name }
                        } else {
                            // Single manifold param without scalars - pass reference
                            quote! { &self.#name }
                        }
                    }
                    ParamKind::Scalar(_) => {
                        // For non-Field domains, wrap f32 params using from_f32
                        if self.use_jet_wrapper {
                            quote! { <__Scalar as ::pixelflow_core::Computational>::from_f32(self.#name) }
                        } else if needs_pre_eval {
                            // Mixed with manifolds: wrap scalar as Field
                            quote! { #scalar_type_token::from(self.#name) }
                        } else {
                            quote! { self.#name }
                        }
                    }
                };

                indexed_params.push((idx, param_value));
            }

            // Sort by index to get correct tuple order
            indexed_params.sort_by_key(|(idx, _)| *idx);
            let param_values: Vec<_> = indexed_params.into_iter().map(|(_, val)| val).collect();

            // Generate the WithContext tuple
            // CRITICAL: For 1-element tuples, we need a trailing comma: (value,)
            // Without it, Rust interprets (value) as just parentheses, not a tuple!
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

        // Wrap in Let bindings for literals - always, for ZST preservation
        let result = if !self.collected_literals.is_empty() {
            let mut wrapped = result;
            // Wrap literals (innermost - reversed so last literal is innermost)
            for collected in self.collected_literals.iter().rev() {
                let lit = &collected.lit;
                // For Jet mode: use from_f32; for Field mode: use Field::from
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

        // Evaluate the expression
        let at_binding = quote! { #result.eval(__p) };

        // Return pre-eval statements and the binding
        let stmts = if pre_eval_stmts.is_empty() {
            quote! {}
        } else {
            quote! { #(#pre_eval_stmts)* }
        };
        (stmts, at_binding)
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
                            // Both scalar and manifold parameters become CtxVar::<A0, INDEX>::new()
                            // using array-based indexing with const generics
                            if let Some(&idx) = self.param_indices.get(&name_str) {
                                // All values go into array A0 (homogeneous after conversion)
                                // The INDEX is the const generic position within the array
                                quote! { CtxVar::<A0, #idx>::new() }
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
                // Always emit literals as CtxVar references for ZST preservation
                // This ensures expression trees remain Copy (composed of ZST nodes)
                if let Some(var_idx) = lit.var_index {
                    // Literals go at indices in the context array
                    // Use array-based indexing: CtxVar::<A0, INDEX>::new()
                    quote! { CtxVar::<A0, #var_idx>::new() }
                } else {
                    // Fallback: no var_index assigned (shouldn't happen after annotation)
                    // For Jet mode, use from_f32; for Field mode, emit directly
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
                let args: Vec<TokenStream> = call.args.iter()
                    .map(|a| self.emit_annotated_expr(a))
                    .collect();

                if args.is_empty() {
                    quote! { #receiver.#method() }
                } else {
                    quote! { #receiver.#method(#(#args),*) }
                }
            }

            AnnotatedExpr::Call(call) => {
                // Free function call: V(m), DX(expr), etc.
                // Emit with transformed arguments (manifold params become Var<N>)
                let func = &call.func;
                let args: Vec<TokenStream> = call.args.iter()
                    .map(|a| self.emit_annotated_expr(a))
                    .collect();

                if args.is_empty() {
                    quote! { #func() }
                } else {
                    quote! { #func(#(#args),*) }
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
        // Anonymous kernel - emits closure returning WithContext
        let input = quote! { |cx: f32| X - cx };
        let output = compile(input);
        let output_str = output.to_string();

        // Should be a closure
        assert!(
            output_str.contains("move | cx : f32 |"),
            "Expected closure, got: {}",
            output_str
        );
        // Should use CtxVar::<A0, 0usize> for the single parameter
        assert!(
            output_str.contains("CtxVar :: < A0 , 0usize >"),
            "Expected CtxVar::<A0, 0usize>, got: {}",
            output_str
        );
        // Should use WithContext
        assert!(
            output_str.contains("WithContext :: new"),
            "Expected WithContext::new, got: {}",
            output_str
        );
    }

    #[test]
    fn emit_two_params() {
        // Anonymous kernel with two params
        let input = quote! { |cx: f32, cy: f32| (X - cx) + (Y - cy) };
        let output = compile(input);
        let output_str = output.to_string();

        // cx → CtxVar::<A0, 1usize> (first param, highest index in tuple, but index 1 in array?)
        // Wait, param_indices logic:
        // n=2. cx: index 1. cy: index 0.
        // Array values are sorted by index.
        // param_values = [(0, cy), (1, cx)].
        // Array = [cy, cx].
        // So cy is at A0[0], cx is at A0[1].
        
        assert!(
            output_str.contains("CtxVar :: < A0 , 1usize >"),
            "Expected CtxVar::<A0, 1usize> for cx, got: {}",
            output_str
        );
        assert!(
            output_str.contains("CtxVar :: < A0 , 0usize >"),
            "Expected CtxVar::<A0, 0usize> for cy, got: {}",
            output_str
        );
        // Should use WithContext with array
        assert!(
            output_str.contains("WithContext :: new ((["),
            "Expected WithContext with array, got: {}",
            output_str
        );
    }

    #[test]
    fn emit_three_params() {
        let input = quote! { |a: f32, b: f32, c: f32| a + b + c };
        let output = compile(input);
        let output_str = output.to_string();

        // a → index 2, b → index 1, c → index 0
        assert!(
            output_str.contains("CtxVar :: < A0 , 2usize >"),
            "Expected CtxVar::<A0, 2usize> for a"
        );
        assert!(
            output_str.contains("CtxVar :: < A0 , 1usize >"),
            "Expected CtxVar::<A0, 1usize> for b"
        );
        assert!(
            output_str.contains("CtxVar :: < A0 , 0usize >"),
            "Expected CtxVar::<A0, 0usize> for c"
        );
    }

    #[test]
    fn emit_empty_params() {
        // Anonymous kernel with no params
        let input = quote! { || X + Y };
        let output = compile(input);
        let output_str = output.to_string();

        // Should be a closure with empty params
        assert!(
            output_str.contains("||"),
            "Expected no-param closure, got: {}",
            output_str
        );
        // Should use WithContext with unit
        assert!(
            output_str.contains("WithContext :: new (() , __expr)"),
            "Expected WithContext with unit, got: {}",
            output_str
        );
    }

    #[test]
    fn emit_method_calls() {
        let input = quote! { |r: f32| (X * X + Y * Y).sqrt() - r };
        let output = compile(input);
        let output_str = output.to_string();

        // r → CtxVar::<A0, 0usize>
        assert!(output_str.contains(". sqrt ()"));
        assert!(
            output_str.contains("CtxVar :: < A0 , 0usize >"),
            "Expected CtxVar::<A0, 0usize> for r"
        );
    }

    #[test]
    fn emit_manifold_param() {
        // Anonymous kernel with manifold param - still emits closure
        let input = quote! { |inner: kernel, r: f32| inner - r };
        let output = compile(input);
        let output_str = output.to_string();

        // Should be a closure (anonymous kernels always emit closures)
        assert!(
            output_str.contains("move |"),
            "Expected closure, got: {}",
            output_str
        );

        // inner → index 1, r → index 0
        assert!(
            output_str.contains("CtxVar :: < A0 , 1usize >"),
            "Expected CtxVar::<A0, 1usize> for inner"
        );
        assert!(
            output_str.contains("CtxVar :: < A0 , 0usize >"),
            "Expected CtxVar::<A0, 0usize> for r"
        );

        // Should use WithContext
        assert!(
            output_str.contains("WithContext :: new"),
            "Expected WithContext::new, got: {}",
            output_str
        );
    }

    #[test]
    fn emit_multiple_manifold_params() {
        // Anonymous kernel with multiple manifold params
        let input = quote! { |a: kernel, b: kernel| a + b };
        let output = compile(input);
        let output_str = output.to_string();

        // Should be a closure
        assert!(
            output_str.contains("move |"),
            "Expected closure, got: {}",
            output_str
        );

        // a → index 1, b → index 0
        assert!(
            output_str.contains("CtxVar :: < A0 , 1usize >"),
            "Expected CtxVar::<A0, 1usize> for a"
        );
        assert!(
            output_str.contains("CtxVar :: < A0 , 0usize >"),
            "Expected CtxVar::<A0, 0usize> for b"
        );

        // Should use WithContext with tuple
        assert!(
            output_str.contains("WithContext :: new"),
            "Expected WithContext, got: {}",
            output_str
        );
    }

    #[test]
    fn emit_named_kernel() {
        // Named kernel - emits struct with given name
        let input = quote! { pub struct Circle = |cx: f32, cy: f32, r: f32| -> Field {
            let dx = X - cx;
            let dy = Y - cy;
            (dx * dx + dy * dy).sqrt() - r
        }};
        let output = compile(input);
        let output_str = output.to_string();

        // Should have struct named Circle
        assert!(
            output_str.contains("pub struct Circle"),
            "Expected pub struct Circle, got: {}",
            output_str
        );
        // Should have new constructor
        assert!(
            output_str.contains("pub fn new"),
            "Expected new constructor, got: {}",
            output_str
        );
        // Should implement Manifold
        assert!(
            output_str.contains("impl :: pixelflow_core :: Manifold"),
            "Expected Manifold impl, got: {}",
            output_str
        );
    }
}
