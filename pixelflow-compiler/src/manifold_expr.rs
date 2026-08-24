//! # ManifoldExpr Derive Macro
//!
//! Generates implementations for the `ManifoldExpr` marker trait,
//! which gates access to `ManifoldExt` methods.
//!
//! ## Usage
//!
//! ```ignore
//! #[derive(ManifoldExpr)]
//! pub struct Sqrt<M>(pub M);
//! ```
//!
//! ## Generated Code
//!
//! For `Sqrt<M>`:
//! ```ignore
//! impl<M> ::pixelflow_core::ManifoldExpr for Sqrt<M> {}
//! ```
//!
//! ## Future: Chained Ops
//!
//! This macro could be extended to also generate operator impls
//! (Add, Sub, Mul, Div) currently handled by `impl_chained_ops!`.

use proc_macro2::TokenStream;
use quote::quote;
use syn::DeriveInput;

/// Generate the `ManifoldExpr` impl for a type.
pub fn derive_manifold_expr(input: DeriveInput) -> TokenStream {
    let name = &input.ident;
    let generics = &input.generics;

    // Extract generic parameters for the impl
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    quote! {
        impl #impl_generics ::pixelflow_core::ManifoldExpr for #name #ty_generics #where_clause {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_manifold_expr_generates_an_impl_naming_the_input_type_and_its_generics() {
        let input: DeriveInput = syn::parse_quote! {
            pub struct Sqrt<M>(pub M);
        };
        let generated = derive_manifold_expr(input).to_string();
        let expected =
            quote! { impl<M> ::pixelflow_core::ManifoldExpr for Sqrt<M> {} }.to_string();
        assert_eq!(generated, expected);
    }

    #[test]
    fn derive_manifold_expr_handles_a_type_with_no_generics() {
        let input: DeriveInput = syn::parse_quote! {
            pub struct Constant(pub f32);
        };
        let generated = derive_manifold_expr(input).to_string();
        let expected =
            quote! { impl ::pixelflow_core::ManifoldExpr for Constant {} }.to_string();
        assert_eq!(generated, expected);
    }
}
