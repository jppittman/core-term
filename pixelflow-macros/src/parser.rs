//! # Parser
//!
//! Parses the kernel DSL from token stream to AST.
//!
//! ## Grammar
//!
//! ```text
//! kernel     ::= '|' params '|' expr
//! params     ::= (param (',' param)*)?
//! param      ::= IDENT ':' type
//!
//! expr       ::= binary
//! binary     ::= unary (('+' | '-' | '*' | '/' | '%') unary)*
//! unary      ::= ('-' | '!')? postfix
//! postfix    ::= primary ('.' method_call)*
//! method_call::= IDENT '(' args? ')'
//! primary    ::= IDENT | LITERAL | '(' expr ')' | block
//! block      ::= '{' stmt* expr? '}'
//! stmt       ::= 'let' IDENT (':' type)? '=' expr ';'
//!              | expr ';'
//! ```
//!
//! ## Implementation Note
//!
//! We use syn to parse into its Expr types first, then convert to our AST.
//! This gives us Rust's expression parsing for free while maintaining our
//! own semantic layer.

use crate::ast::{
    BinaryExpr, BinaryOp, BlockExpr, CallExpr, Expr, IdentExpr, KernelDef, LetStmt, LiteralExpr,
    MethodCallExpr, Param, ParamKind, Stmt, UnaryExpr, UnaryOp,
};
use proc_macro2::{Span, TokenStream};
use syn::parse::{Parse, ParseStream};
use syn::{Pat, Token, Type};

/// Parse kernel input from token stream.
pub fn parse(input: TokenStream) -> syn::Result<KernelDef> {
    syn::parse2(input)
}

/// Parser state for the closure-like syntax.
struct KernelParser;

impl Parse for KernelDef {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Parse: |param: Type, ...| body
        input.parse::<Token![|]>()?;

        let mut params = Vec::new();

        // Handle empty params: || body
        if !input.peek(Token![|]) {
            // Parse parameter list manually
            loop {
                // Parse identifier
                let ident: syn::Ident = input.parse()?;
                // Parse colon
                input.parse::<Token![:]>()?;
                // Parse type
                let ty: Type = input.parse()?;

                // Detect `kernel` keyword as manifold parameter marker
                let kind = if is_kernel_keyword(&ty) {
                    ParamKind::Manifold
                } else {
                    ParamKind::Scalar(ty)
                };

                params.push(Param { name: ident, kind });

                // Check for comma or end of params
                if input.peek(Token![,]) {
                    input.parse::<Token![,]>()?;
                    // Allow trailing comma before |
                    if input.peek(Token![|]) {
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        input.parse::<Token![|]>()?;

        // Parse optional return type: -> Type
        let return_ty = if input.peek(Token![->]) {
            input.parse::<Token![->]>()?;
            Some(input.parse::<Type>()?)
        } else {
            None
        };

        // Parse the body expression
        let syn_expr: syn::Expr = input.parse()?;
        let body = convert_expr(syn_expr)?;

        Ok(KernelDef {
            params,
            return_ty,
            body,
        })
    }
}

/// Check if a type is the `kernel` keyword (manifold parameter marker).
fn is_kernel_keyword(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        type_path.path.is_ident("kernel")
    } else {
        false
    }
}

/// Convert syn::Expr to our AST Expr.
fn convert_expr(expr: syn::Expr) -> syn::Result<Expr> {
    match expr {
        syn::Expr::Path(expr_path) => {
            // Simple identifier: X, cx, etc.
            if expr_path.path.segments.len() == 1 && expr_path.qself.is_none() {
                let segment = &expr_path.path.segments[0];
                if segment.arguments.is_empty() {
                    return Ok(Expr::Ident(IdentExpr {
                        name: segment.ident.clone(),
                        span: segment.ident.span(),
                    }));
                }
            }
            // Complex path - pass through verbatim
            Ok(Expr::Verbatim(syn::Expr::Path(expr_path)))
        }

        syn::Expr::Lit(expr_lit) => Ok(Expr::Literal(LiteralExpr {
            span: expr_lit.lit.span(),
            lit: expr_lit.lit,
        })),

        syn::Expr::Binary(expr_binary) => {
            let op = BinaryOp::from_syn(&expr_binary.op).ok_or_else(|| {
                syn::Error::new_spanned(
                    expr_binary.op,
                    format!(
                        "unsupported binary operator `{}`\n\
                         note: kernel! supports: + - * / % < <= > >= == != & |",
                        quote::quote!(#expr_binary.op)
                    ),
                )
            })?;
            let lhs = convert_expr(*expr_binary.left)?;
            let rhs = convert_expr(*expr_binary.right)?;
            Ok(Expr::Binary(BinaryExpr {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span: Span::call_site(),
            }))
        }

        syn::Expr::Unary(expr_unary) => {
            let op = UnaryOp::from_syn(&expr_unary.op).ok_or_else(|| {
                syn::Error::new_spanned(
                    expr_unary.op,
                    "unsupported unary operator\n\
                     note: kernel! supports: - (negation), ! (logical not)",
                )
            })?;
            let operand = convert_expr(*expr_unary.expr)?;
            Ok(Expr::Unary(UnaryExpr {
                op,
                operand: Box::new(operand),
                span: Span::call_site(),
            }))
        }

        syn::Expr::MethodCall(expr_method) => {
            let receiver = convert_expr(*expr_method.receiver)?;
            let args = expr_method
                .args
                .into_iter()
                .map(convert_expr)
                .collect::<syn::Result<Vec<_>>>()?;
            Ok(Expr::MethodCall(MethodCallExpr {
                receiver: Box::new(receiver),
                method: expr_method.method,
                args,
                span: Span::call_site(),
            }))
        }

        syn::Expr::Call(expr_call) => {
            // Free function call: V(m), DX(expr), etc.
            // Extract the function name from the callee
            if let syn::Expr::Path(ref path) = *expr_call.func {
                if path.path.segments.len() == 1 && path.qself.is_none() {
                    let func = path.path.segments[0].ident.clone();
                    let args = expr_call
                        .args
                        .into_iter()
                        .map(convert_expr)
                        .collect::<syn::Result<Vec<_>>>()?;
                    return Ok(Expr::Call(CallExpr {
                        func,
                        args,
                        span: Span::call_site(),
                    }));
                }
            }
            // Complex call (qualified path, etc.) - pass through verbatim
            Ok(Expr::Verbatim(syn::Expr::Call(expr_call)))
        }

        syn::Expr::Paren(expr_paren) => {
            let inner = convert_expr(*expr_paren.expr)?;
            Ok(Expr::Paren(Box::new(inner)))
        }

        syn::Expr::Block(expr_block) => {
            let block = convert_block(expr_block.block)?;
            Ok(Expr::Block(block))
        }

        // Anything else - pass through verbatim for codegen to handle
        other => Ok(Expr::Verbatim(other)),
    }
}

/// Convert a syn::Block to our BlockExpr.
fn convert_block(block: syn::Block) -> syn::Result<BlockExpr> {
    let mut stmts = Vec::new();
    let mut final_expr = None;

    for (i, stmt) in block.stmts.iter().enumerate() {
        let is_last = i == block.stmts.len() - 1;

        match stmt {
            syn::Stmt::Local(local) => {
                // let binding
                let name = match &local.pat {
                    Pat::Ident(pat_ident) => pat_ident.ident.clone(),
                    Pat::Type(pat_type) => match &*pat_type.pat {
                        Pat::Ident(pat_ident) => pat_ident.ident.clone(),
                        _ => {
                            return Err(syn::Error::new_spanned(
                                &local.pat,
                                "expected identifier pattern in let",
                            ));
                        }
                    },
                    _ => {
                        return Err(syn::Error::new_spanned(
                            &local.pat,
                            "expected identifier pattern in let",
                        ));
                    }
                };

                let ty = match &local.pat {
                    Pat::Type(pat_type) => Some((*pat_type.ty).clone()),
                    _ => None,
                };

                let init = local.init.as_ref().ok_or_else(|| {
                    syn::Error::new_spanned(&local.pat, "let binding must have initializer")
                })?;

                let init_expr = convert_expr((*init.expr).clone())?;

                stmts.push(Stmt::Let(LetStmt {
                    name,
                    ty,
                    init: init_expr,
                    span: Span::call_site(),
                }));
            }

            syn::Stmt::Expr(expr, semi) => {
                let converted = convert_expr(expr.clone())?;
                if is_last && semi.is_none() {
                    // Final expression without semicolon - this is the block's value
                    final_expr = Some(Box::new(converted));
                } else {
                    stmts.push(Stmt::Expr(converted));
                }
            }

            syn::Stmt::Item(_) => {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "items (fn, struct, etc.) not allowed in kernel block\n\
                     help: define items outside the kernel! macro",
                ));
            }

            syn::Stmt::Macro(_) => {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "macro invocations not allowed in kernel block\n\
                     help: expand macros outside kernel! or use equivalent expressions",
                ));
            }
        }
    }

    Ok(BlockExpr {
        stmts,
        expr: final_expr,
        span: Span::call_site(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    #[test]
    fn parse_simple_kernel() {
        let input = quote! { |r: f32| X * X + Y * Y - r };
        let kernel = parse(input).unwrap();
        assert_eq!(kernel.params.len(), 1);
        assert_eq!(kernel.params[0].name.to_string(), "r");
    }

    #[test]
    fn parse_empty_params() {
        let input = quote! { || X * X + Y * Y };
        let kernel = parse(input).unwrap();
        assert_eq!(kernel.params.len(), 0);
    }

    #[test]
    fn parse_multiple_params() {
        let input = quote! { |cx: f32, cy: f32, r: f32| X - cx };
        let kernel = parse(input).unwrap();
        assert_eq!(kernel.params.len(), 3);
    }

    #[test]
    fn parse_method_call() {
        let input = quote! { |r: f32| (X * X + Y * Y).sqrt() - r };
        let kernel = parse(input).unwrap();
        // Should successfully parse the .sqrt() method call
        match kernel.body {
            Expr::Binary(_) => {} // Expected: sqrt() - r
            _ => panic!("expected binary expression"),
        }
    }

    #[test]
    fn parse_block_expr() {
        let input = quote! {
            |cx: f32, cy: f32| {
                let dx = X - cx;
                let dy = Y - cy;
                dx * dx + dy * dy
            }
        };
        let kernel = parse(input).unwrap();
        match kernel.body {
            Expr::Block(block) => {
                assert_eq!(block.stmts.len(), 2); // two let statements
                assert!(block.expr.is_some()); // final expression
            }
            _ => panic!("expected block expression"),
        }
    }

    #[test]
    fn parse_return_type() {
        let input = quote! { |cx: f32| -> Jet3 X - cx };
        let kernel = parse(input).unwrap();
        assert_eq!(kernel.params.len(), 1);
        assert!(kernel.return_ty.is_some());
        // Verify the return type is "Jet3"
        let ty = kernel.return_ty.unwrap();
        if let syn::Type::Path(type_path) = ty {
            assert_eq!(type_path.path.segments[0].ident.to_string(), "Jet3");
        } else {
            panic!("expected path type");
        }
    }

    #[test]
    fn parse_no_return_type() {
        let input = quote! { |cx: f32| X - cx };
        let kernel = parse(input).unwrap();
        assert!(kernel.return_ty.is_none());
    }

    #[test]
    fn parse_manifold_param() {
        // `kernel` keyword marks a manifold parameter
        let input = quote! { |inner: kernel, r: f32| inner - r };
        let kernel = parse(input).unwrap();
        assert_eq!(kernel.params.len(), 2);

        // First param should be Manifold
        assert!(
            matches!(kernel.params[0].kind, ParamKind::Manifold),
            "expected inner to be Manifold param"
        );
        assert_eq!(kernel.params[0].name.to_string(), "inner");

        // Second param should be Scalar(f32)
        assert!(
            matches!(kernel.params[1].kind, ParamKind::Scalar(_)),
            "expected r to be Scalar param"
        );
        assert_eq!(kernel.params[1].name.to_string(), "r");
    }

    #[test]
    fn parse_multiple_manifold_params() {
        let input = quote! { |a: kernel, b: kernel| a + b };
        let kernel = parse(input).unwrap();
        assert_eq!(kernel.params.len(), 2);
        assert!(matches!(kernel.params[0].kind, ParamKind::Manifold));
        assert!(matches!(kernel.params[1].kind, ParamKind::Manifold));
    }
}
