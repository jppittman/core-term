//! Core code emission logic for kernel compilation.

use std::collections::HashMap;

use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};

use crate::annotate::{annotate, AnnotatedExpr, AnnotatedStmt, AnnotationCtx};
use crate::ast::{BinaryOp, ParamKind, UnaryOp};
use crate::sema::AnalyzedKernel;
use crate::symbol::SymbolKind;

use super::struct_emitter::{Derives, StructEmitter};
use super::util::{build_array, sort_by_index, standard_imports};

/// The code emitter state.
pub struct CodeEmitter<'a> {
    analyzed: &'a AnalyzedKernel,
    /// Maps parameter names to their (ArrayID, Index) location.
    /// ArrayID: 0=A0 (Scalars), 1=A1 (M0), 2=A2 (M1), etc.
    param_indices: HashMap<String, (u8, usize)>,
    /// Maps manifold parameter names to their generic type index (M0, M1, ...).
    manifold_indices: HashMap<String, usize>,
    /// Whether we're generating for a Jet domain (Jet2, Jet3).
    /// If true, f32 literals must be wrapped as constant jets.
    use_jet_wrapper: bool,
    /// Collected literals from annotation pass (for Let bindings in Jet mode).
    collected_literals: Vec<crate::annotate::CollectedLiteral>,
}

impl<'a> CodeEmitter<'a> {
    pub fn new(analyzed: &'a AnalyzedKernel) -> Self {
        // Separate params into scalars and manifolds
        let mut scalar_params = Vec::new();
        let mut manifold_params = Vec::new();

        for param in &analyzed.def.params {
            match param.kind {
                ParamKind::Scalar(_) => scalar_params.push(param),
                ParamKind::Manifold => manifold_params.push(param),
            }
        }

        let mut param_indices = HashMap::new();
        let mut manifold_indices = HashMap::new();

        // Scalars go into A0 (Array 0)
        // Indices are assigned in reverse order of declaration (deepest first)
        // Literals will be added later, effectively extending this array
        let n_scalars = scalar_params.len();
        for (i, param) in scalar_params.iter().enumerate() {
            // Index: n-1-i (last param gets 0)
            param_indices.insert(param.name.to_string(), (0u8, n_scalars - 1 - i));
        }

        // Manifolds go into subsequent arrays (A1, A2, ...)
        // Each manifold gets its own array of size 1
        for (i, param) in manifold_params.iter().enumerate() {
            let array_id = (i + 1) as u8; // A1, A2, ...
            param_indices.insert(param.name.to_string(), (array_id, 0));
            manifold_indices.insert(param.name.to_string(), i);
        }

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
    pub fn emit_kernel(&mut self) -> TokenStream {
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
        let std_imports = standard_imports();

        // Run annotation pass to collect literals and assign Var indices
        let annotation_ctx = AnnotationCtx::new();
        let (annotated_body, _, collected_literals) = annotate(&self.analyzed.def.body, annotation_ctx);
        self.collected_literals = collected_literals;

        // Always adjust param indices to account for literals in context
        // Literals go into A0 (Scalars), so only adjust scalar params (ArrayID 0)
        let literal_count = self.collected_literals.len();
        for (_, (array_id, idx)) in self.param_indices.iter_mut() {
            if *array_id == 0 {
                *idx += literal_count;
            }
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
        let scalar_ty_ref = scalar_type.as_ref();
        let scalar_type_token = scalar_ty_ref.cloned().unwrap_or_else(|| quote! { ::pixelflow_core::Field });

        // Determine if we need special handling for manifold params
        let manifold_count = self.manifold_indices.len();
        let has_scalar_params = params.iter().any(|p| matches!(p.kind, ParamKind::Scalar(_)));
        let has_literals = !self.collected_literals.is_empty();

        // Single manifold + scalars/literals: use ManifoldBind (manifold handled separately)
        // Multiple manifolds: use Computed with pre-eval (fallback)
        // No manifolds: use plain WithContext
        let use_manifold_bind = manifold_count == 1 && (has_scalar_params || has_literals);
        let use_computed_fallback = manifold_count > 1;

        // Pre-evaluate manifolds to get concrete scalar values (for Computed fallback)
        let mut pre_eval_stmts = Vec::new();
        if use_computed_fallback {
            for param in params.iter() {
                if matches!(param.kind, ParamKind::Manifold) {
                    let name = &param.name;
                    let eval_name = format_ident!("__eval_{}", name);
                    pre_eval_stmts.push(quote! {
                        let #eval_name: #scalar_type_token = #name.eval(__p);
                    });
                }
            }
        }

        // Group values into arrays by ArrayID
        // A0: Scalars (literals + scalar params)
        // A1..AN: Manifold params (only for non-ManifoldBind cases)
        let mut arrays: Vec<Vec<(usize, TokenStream)>> = vec![Vec::new(); 16]; // Max 16 arrays

        // 1. Add Literals to A0
        for c in &self.collected_literals {
            let lit = &c.lit;
            let val = if self.use_jet_wrapper {
                quote! { <#scalar_type_token as ::pixelflow_core::Computational>::from_f32(#lit) }
            } else {
                quote! { ::pixelflow_core::Field::from(#lit) }
            };
            arrays[0].push((c.index, val));
        }

        // 2. Add Parameters to appropriate arrays
        for param in params.iter() {
            let name = &param.name;
            let (array_id, idx) = self.param_indices[&name.to_string()];

            let param_value = match &param.kind {
                ParamKind::Manifold => {
                    if use_manifold_bind {
                        // ManifoldBind handles this param - don't add to context
                        continue;
                    } else if use_computed_fallback {
                        // Computed fallback: use pre-evaluated value
                        let eval_name = format_ident!("__eval_{}", name);
                        quote! { #eval_name }
                    } else {
                        // Single manifold param with no scalars - can store directly
                        quote! { #name }
                    }
                }
                ParamKind::Scalar(_) => match scalar_ty_ref {
                    Some(sty) => quote! { <#sty as ::pixelflow_core::Computational>::from_f32(#name) },
                    None => quote! { ::pixelflow_core::Field::from(#name) },
                },
            };

            if (array_id as usize) < arrays.len() {
                arrays[array_id as usize].push((idx, param_value));
            }
        }

        // Build the context tuple
        // Empty context is (), single array is ([...],), multi-array is ([...], [...])
        let raw_arrays: Vec<TokenStream> = arrays.iter()
            .filter(|a| !a.is_empty())
            .map(|vals| {
                let sorted = sort_by_index(vals.clone());
                quote! { [#(#sorted),*] }
            })
            .collect();

        let context_tuple = if raw_arrays.is_empty() {
            quote! { () }
        } else if raw_arrays.len() == 1 {
            let a = &raw_arrays[0];
            quote! { (#a,) }
        } else {
            quote! { (#(#raw_arrays),*) }
        };

        // Choose code generation strategy based on param composition
        if use_manifold_bind {
            // Single manifold + scalars/literals: use ManifoldBind
            // ManifoldBind carries the manifold type in its signature, helping type inference
            let manifold_name = params.iter()
                .find(|p| matches!(p.kind, ParamKind::Manifold))
                .map(|p| &p.name)
                .expect("use_manifold_bind requires exactly one manifold param");

            quote! {
                move |#(#closure_params),*| {
                    #std_imports
                    #peano_imports

                    let __expr = { #body };
                    let __inner_body = WithContext::new(#context_tuple, __expr);
                    ManifoldBind::new(#manifold_name, __inner_body)
                }
            }
        } else if use_computed_fallback {
            // Multiple manifolds: use Computed with pre-eval
            // Type inference may still fail for some cases
            quote! {
                move |#(#closure_params),*| {
                    #std_imports
                    #peano_imports

                    let __expr = { #body };
                    // Pre-evaluate manifolds, then build context with evaluated values
                    Computed::new(move |__p| {
                        #(#pre_eval_stmts)*
                        WithContext::new(#context_tuple, __expr).eval(__p)
                    })
                }
            }
        } else {
            // No manifolds, or single manifold without scalars/literals
            quote! {
                move |#(#closure_params),*| {
                    #std_imports
                    #peano_imports

                    let __expr = { #body };
                    WithContext::new(#context_tuple, __expr)
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
        let std_imports = standard_imports();

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
        // Literals go into A0 (Scalars), so only adjust scalar params (ArrayID 0)
        let literal_count = self.collected_literals.len();
        for (_, (array_id, idx)) in self.param_indices.iter_mut() {
            if *array_id == 0 {
                *idx += literal_count;
            }
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
            .map(|g| quote! { #g: ::pixelflow_core::Manifold<#domain_type, Output = #scalar_type> })
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
            std_imports,
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

    /// Helper to determine the scalar type token (e.g., Field, Jet3).
    /// Returns None if the scalar type is Field (default).
    fn get_scalar_type(&self) -> Option<TokenStream> {
        let domain_for_scalar = self.analyzed.def.domain_ty.as_ref().or(self.analyzed.def.return_ty.as_ref());
        match domain_for_scalar {
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
        }
    }

    /// Emit unified WithContext/CtxVar binding for params (and Let for literals).
    fn emit_unified_binding(&self) -> (TokenStream, TokenStream) {
        let params = &self.analyzed.def.params;

        if params.is_empty() && self.collected_literals.is_empty() {
            return (quote! {}, quote! { __expr.eval(__p) });
        }

        // Determine if we need to pre-evaluate manifold params
        let manifold_count = self.manifold_indices.len();
        let has_scalar_params = params.iter().any(|p| matches!(p.kind, ParamKind::Scalar(_)));
        let needs_pre_eval = manifold_count > 0 && (manifold_count > 1 || has_scalar_params);

        // Get concrete scalar type token
        let scalar_type_opt = self.get_scalar_type();
        let scalar_type_token = scalar_type_opt.clone().unwrap_or(quote! { ::pixelflow_core::Field });

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

        // Group values into arrays by ArrayID
        // A0: Scalars (literals + scalar params)
        // A1..AN: Manifold params
        let mut arrays: Vec<Vec<(usize, TokenStream)>> = vec![Vec::new(); 16]; // Max 16 arrays

        // 1. Add Literals to A0
        for c in &self.collected_literals {
            let lit = &c.lit;
            let val = if self.use_jet_wrapper {
                quote! { <#scalar_type_token as ::pixelflow_core::Computational>::from_f32(#lit) }
            } else {
                quote! { ::pixelflow_core::Field::from(#lit) }
            };
            arrays[0].push((c.index, val));
        }

        // 2. Add Parameters to appropriate arrays
        for param in params.iter() {
            let name = &param.name;
            let (array_id, idx) = self.param_indices[&name.to_string()];
            
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
                        quote! { <#scalar_type_token as ::pixelflow_core::Computational>::from_f32(self.#name) }
                    } else if needs_pre_eval {
                        quote! { #scalar_type_token::from(self.#name) }
                    } else {
                        quote! { ::pixelflow_core::Field::from(self.#name) }
                    }
                }
            };
            
            if (array_id as usize) < arrays.len() {
                arrays[array_id as usize].push((idx, param_value));
            }
        }

        // Build the tuple of arrays
        let mut array_exprs = Vec::new();
        for array_values in arrays.iter() {
            if !array_values.is_empty() {
                // Sort by index and extract values
                let sorted_values = sort_by_index(array_values.clone());
                // Build array: ([val0, val1],)
                array_exprs.push(build_array(&sorted_values));
            }
        }

        // Generate the WithContext call
        // Note: We need a tuple of arrays. If there's only one array, we need (array,)
        
        let context_tuple = if array_exprs.is_empty() {
            quote! { () }
        } else {
            // Unwrap the single-element tuples from build_array to get raw arrays
            // build_array returns `([a,b],)` -> we want `[a,b]`
            // This relies on knowing build_array implementation details.
            // Let's modify the logic to construct arrays directly here.
            
            let raw_arrays: Vec<TokenStream> = arrays.iter()
                .filter(|a| !a.is_empty())
                .map(|vals| {
                    let sorted = sort_by_index(vals.clone());
                    quote! { [#(#sorted),*] }
                })
                .collect();
                
            if raw_arrays.len() == 1 {
                let a = &raw_arrays[0];
                quote! { (#a,) }
            } else {
                quote! { (#(#raw_arrays),*) }
            }
        };

        // Wrap in Let bindings for literals?
        // NO - literals are now in A0! We don't use nested Let bindings anymore.
        // We use the array context exclusively.
        
        let at_binding = quote! { 
            WithContext::new(#context_tuple, __expr).eval(__p)
        };

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
    pub fn emit_annotated_expr(&self, expr: &AnnotatedExpr) -> TokenStream {
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
                            // Parameters use CtxVar::<Ax, INDEX>::new()
                            if let Some(&(array_id, idx)) = self.param_indices.get(&name_str) {
                                let marker = match array_id {
                                    0 => quote! { A0 },
                                    1 => quote! { A1 },
                                    2 => quote! { A2 },
                                    3 => quote! { A3 },
                                    4 => quote! { A4 },
                                    5 => quote! { A5 },
                                    6 => quote! { A6 },
                                    7 => quote! { A7 },
                                    8 => quote! { A8 },
                                    9 => quote! { A9 },
                                    10 => quote! { A10 },
                                    11 => quote! { A11 },
                                    12 => quote! { A12 },
                                    13 => quote! { A13 },
                                    14 => quote! { A14 },
                                    15 => quote! { A15 },
                                    _ => panic!("Too many context arrays (max 16 supported)"),
                                };
                                quote! { CtxVar::<#marker, #idx>::new() }
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

                // Always wrap binary expressions in parentheses to preserve precedence.
                // This prevents issues like `(X + val).sqrt()` becoming `X + val.sqrt()`
                // when the binary expression is used as a method receiver.
                match binary.op {
                    BinaryOp::Add => quote! { (#lhs + #rhs) },
                    BinaryOp::Sub => quote! { (#lhs - #rhs) },
                    BinaryOp::Mul => quote! { (#lhs * #rhs) },
                    BinaryOp::Div => quote! { (#lhs / #rhs) },
                    BinaryOp::Rem => quote! { (#lhs % #rhs) },
                    BinaryOp::Lt => quote! { #lhs.lt(#rhs) },
                    BinaryOp::Le => quote! { #lhs.le(#rhs) },
                    BinaryOp::Gt => quote! { #lhs.gt(#rhs) },
                    BinaryOp::Ge => quote! { #lhs.ge(#rhs) },
                    BinaryOp::Eq => quote! { #lhs.eq(#rhs) },
                    BinaryOp::Ne => quote! { #lhs.ne(#rhs) },
                    BinaryOp::BitAnd => quote! { (#lhs & #rhs) },
                    BinaryOp::BitOr => quote! { (#lhs | #rhs) },
                }
            }

            AnnotatedExpr::Unary(unary) => {
                let operand = self.emit_annotated_expr(&unary.operand);
                match unary.op {
                    // Parentheses are required because method call binds tighter than binary operators.
                    // Without parens, `a - b.neg()` is parsed as `a - (b.neg())`, not `(a - b).neg()`.
                    UnaryOp::Neg => quote! { (#operand).neg() },
                    UnaryOp::Not => quote! { !(#operand) },
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

            AnnotatedExpr::Tuple(tuple) => {
                let elems: Vec<TokenStream> = tuple.elems.iter()
                    .map(|e| self.emit_annotated_expr(e))
                    .collect();
                quote! { (#(#elems),*) }
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
