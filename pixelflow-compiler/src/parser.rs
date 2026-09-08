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
    MethodCallExpr, Param, Stmt, TupleExpr, UnaryExpr, UnaryOp,
};
use proc_macro2::{Span, TokenStream};
use syn::parse::{Parse, ParseStream};
use syn::{Pat, Token, Type};

/// Parse kernel input from token stream.
pub fn parse(input: TokenStream) -> syn::Result<KernelDef> {
    syn::parse2(input)
}

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

                params.push(Param {
                    name: ident,
                    ty: Box::new(ty),
                });

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

        // Parse the body expression
        let syn_expr: syn::Expr = input.parse()?;
        let body = convert_expr(syn_expr)?;

        Ok(KernelDef { params, body })
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
                let op_str = quote::quote!(#expr_binary.op).to_string();
                syn::Error::new_spanned(
                    expr_binary.op,
                    format!(
                        "unsupported binary operator `{}`\n\
                         \n\
                         note: the kernel! macro only supports these binary operators:\n\
                         note:   arithmetic: + - * / %\n\
                         note:   comparison: < <= > >= == !=\n\
                         note:   logical: & |\n\
                         \n\
                         help: if you need bitwise operations or other operators, extract them to a helper function",
                        op_str
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
                let op_str = quote::quote!(#expr_unary.op).to_string();
                syn::Error::new_spanned(
                    expr_unary.op,
                    format!(
                        "unsupported unary operator `{}`\n\
                         \n\
                         note: the kernel! macro supports these unary operators:\n\
                         note:   - (negation)   example: -X\n\
                         note:   ! (logical not) example: !condition\n\
                         \n\
                         help: for other unary operations, use method calls like .abs() or helper functions",
                        op_str
                    ),
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

        syn::Expr::Tuple(expr_tuple) => {
            let elems = expr_tuple
                .elems
                .into_iter()
                .map(convert_expr)
                .collect::<syn::Result<Vec<_>>>()?;
            Ok(Expr::Tuple(TupleExpr {
                elems,
                span: Span::call_site(),
            }))
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
    let mut stmts = Vec::with_capacity(block.stmts.len());
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
                                "complex pattern not supported in let binding\n\
                                 \n\
                                 note: kernel! only supports simple identifier patterns\n\
                                 \n\
                                 help: use a simple identifier like:\n\
                                 help:   let dx = X - cx;\n\
                                 help:   let result: f32 = calculation;",
                            ));
                        }
                    },
                    _ => {
                        return Err(syn::Error::new_spanned(
                            &local.pat,
                            "complex pattern not supported in let binding\n\
                             \n\
                             note: kernel! only supports simple identifier patterns\n\
                             \n\
                             help: destructuring, tuples, and other patterns are not allowed\n\
                             help: use a simple identifier like:\n\
                             help:   let value = expression;",
                        ));
                    }
                };

                let ty = match &local.pat {
                    Pat::Type(pat_type) => Some((*pat_type.ty).clone()),
                    _ => None,
                };

                let init = local.init.as_ref().ok_or_else(|| {
                    syn::Error::new_spanned(
                        &local.pat,
                        "let binding must have an initializer\n\
                         \n\
                         help: provide a value for this binding:\n\
                         help:   let dx = X - cx;",
                    )
                })?;

                let init_expr = convert_expr((*init.expr).clone())?;

                stmts.push(Stmt::Let(Box::new(LetStmt {
                    name,
                    ty,
                    init: init_expr,
                    span: Span::call_site(),
                })));
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

            syn::Stmt::Item(item) => {
                return Err(syn::Error::new_spanned(
                    item,
                    "item definitions are not allowed inside kernel! blocks\n\
                     \n\
                     note: kernel! blocks can only contain let bindings and expressions\n\
                     \n\
                     help: define functions, structs, and other items outside the kernel! macro:\n\
                     help:   fn helper(x: f32) -> f32 { x * 2.0 }\n\
                     help:   let my_kernel = kernel!(|| helper(X));",
                ));
            }

            syn::Stmt::Macro(mac) => {
                return Err(syn::Error::new_spanned(
                    mac,
                    "macro invocations are not allowed inside kernel! blocks\n\
                     \n\
                     note: kernel! needs to analyze the expression at compile time\n\
                     \n\
                     help: expand the macro outside the kernel! or use equivalent expressions:\n\
                     help:   let value = some_macro!();\n\
                     help:   let my_kernel = kernel!(|| value * X);",
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

    /// A parameter's declared type is a scalar type and nothing else: the
    /// `kernel` keyword that used to mark a manifold-typed slot is gone with
    /// the tier that spliced one, and kernels compose as `Kernel` values.
    #[test]
    fn parse_scalar_params_keep_their_declared_types() {
        let input = quote! { |cx: f32, n: i32| X * cx + n };
        let kernel = parse(input).unwrap();
        assert_eq!(kernel.params.len(), 2);
        assert_eq!(kernel.params[0].name.to_string(), "cx");
        assert_eq!(kernel.params[1].name.to_string(), "n");
        for (param, want) in kernel.params.iter().zip(["f32", "i32"]) {
            let syn::Type::Path(path) = &*param.ty else {
                panic!("expected a path type for {}", param.name);
            };
            assert_eq!(path.path.segments[0].ident.to_string(), want);
        }
    }

    #[test]
    fn parse_rejects_a_multi_segment_path_as_a_plain_identifier() {
        // A multi-segment path (qself-free but len() != 1) must fall through
        // to Verbatim, not be silently truncated to an Ident of its first
        // segment.
        let input = quote! { || std::f32::consts::PI };
        let kernel = parse(input).unwrap();
        assert!(
            matches!(kernel.body, Expr::Verbatim(_)),
            "expected Verbatim for a multi-segment path, got {:?}",
            kernel.body
        );
    }

    #[test]
    fn parse_let_with_type_annotation_extracts_both_name_and_type() {
        let input = quote! {
            || {
                let dx: f32 = X;
                dx
            }
        };
        let kernel = parse(input).unwrap();
        match kernel.body {
            Expr::Block(block) => {
                assert_eq!(block.stmts.len(), 1);
                match &block.stmts[0] {
                    Stmt::Let(let_stmt) => {
                        assert_eq!(let_stmt.name.to_string(), "dx");
                        let ty = let_stmt.ty.as_ref().expect("expected a type annotation");
                        if let Type::Path(type_path) = ty {
                            assert_eq!(type_path.path.segments[0].ident.to_string(), "f32");
                        } else {
                            panic!("expected path type");
                        }
                    }
                    _ => panic!("expected let statement"),
                }
            }
            _ => panic!("expected block expression"),
        }
    }

    #[test]
    fn a_terminal_statement_with_a_trailing_semicolon_is_not_the_blocks_final_expression() {
        let input = quote! { || { X; } };
        let kernel = parse(input).unwrap();
        match kernel.body {
            Expr::Block(block) => {
                assert_eq!(
                    block.stmts.len(),
                    1,
                    "the semicolon-terminated X is a statement"
                );
                assert!(
                    block.expr.is_none(),
                    "a block ending in `expr;` has no final expression"
                );
            }
            _ => panic!("expected block expression"),
        }
    }

    #[test]
    fn parse_block_body_with_a_scalar_param() {
        let input = quote! {
            |x: f32| {
                let a = X + x;
                a
            }
        };
        let kernel = parse(input).unwrap();
        assert_eq!(kernel.params.len(), 1);

        match kernel.body {
            Expr::Block(block) => {
                assert_eq!(block.stmts.len(), 1, "expected 1 let statement");
                assert!(block.expr.is_some(), "expected final expression");
            }
            other => panic!("expected block expression, got {other:?}"),
        }
    }
}
