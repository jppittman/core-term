//! # Semantic Analysis
//!
//! Analyzes the AST for semantic correctness using pure functional transformations.
//!
//! ## Functional Design
//!
//! Unlike the annotation pass which uses explicit state threading, semantic analysis
//! is purely validating - it doesn't transform the AST. We use pure functions that
//! work with immutable data and build up the symbol table functionally.
//!
//! ## Responsibilities
//!
//! 1. **Symbol Resolution**: Match identifiers to their definitions
//! 2. **Scope Management**: Track let bindings within blocks
//! 3. **Validation**: Ensure all referenced symbols are defined
//!
//! ## Symbol Resolution Rules
//!
//! When an identifier is encountered:
//! 1. Check if it's an intrinsic (X, Y, Z, W) → leave unchanged
//! 2. Check if it's a captured parameter → transform to `self.param`
//! 3. Check if it's a local variable → leave unchanged
//! 4. Otherwise → error (undefined symbol)
//!
//! ## Output
//!
//! The semantic phase produces an `AnalyzedKernel` which includes:
//! - The original AST (immutable)
//! - The populated symbol table
//! - Any resolved type information

use crate::ast::{BlockExpr, Expr, KernelDef, LetStmt, MethodCallExpr, Param, ParamKind, Stmt};
use crate::symbol::{SymbolKind, SymbolTable};
use syn::Ident;

/// The result of semantic analysis.
#[derive(Debug)]
pub struct AnalyzedKernel {
    /// The original kernel definition.
    pub def: KernelDef,
    /// The populated symbol table.
    pub symbols: SymbolTable,
}

/// Configuration for semantic analysis.
#[derive(Clone)]
struct SemaContext {
    /// Whether this is an anonymous kernel (allows captured variables).
    is_anonymous: bool,
}

/// Perform semantic analysis on a parsed kernel.
///
/// This is the entry point. It builds the symbol table and validates all symbols.
pub fn analyze(kernel: KernelDef) -> syn::Result<AnalyzedKernel> {
    // Anonymous kernels (no struct_decl) allow captured variables from environment
    let ctx = SemaContext {
        is_anonymous: kernel.struct_decl.is_none(),
    };

    // Build initial symbol table with parameters
    let symbols = register_parameters(&kernel.params)?;

    // Validate all symbols in the body
    validate_expr(&kernel.body, &symbols, &ctx)?;

    Ok(AnalyzedKernel {
        def: kernel,
        symbols,
    })
}

/// Register all parameters in the symbol table (pure function).
fn register_parameters(params: &[Param]) -> syn::Result<SymbolTable> {
    let mut symbols = SymbolTable::new();

    for param in params {
        register_parameter(&mut symbols, param)?;
    }

    Ok(symbols)
}

/// Register a single parameter in the symbol table.
///
/// This is a helper that mutates the SymbolTable for convenience,
/// but conceptually it's building up the table functionally.
fn register_parameter(symbols: &mut SymbolTable, param: &Param) -> syn::Result<()> {
    let name = param.name.to_string();

    // Check for shadowing intrinsics (error)
    if symbols.is_intrinsic(&name) {
        return Err(syn::Error::new(
            param.name.span(),
            format!(
                "parameter '{}' would shadow the intrinsic coordinate variable\n\
                 \n\
                 note: X, Y, Z, W are built-in coordinate variables provided by PixelFlow\n\
                 note: these are always available and represent the (x, y, z, w) evaluation point\n\
                 \n\
                 help: choose a different parameter name, for example:\n\
                 help:   |cx: f32| ...  // center x coordinate (not 'X')\n\
                 help:   |x_offset: f32| ...  // x offset (not 'X')",
                name
            ),
        ));
    }

    // Check for duplicate parameters
    if symbols.lookup(&name).is_some() {
        return Err(syn::Error::new(
            param.name.span(),
            format!(
                "duplicate parameter name '{}'\n\
                 \n\
                 note: each parameter in a kernel must have a unique name\n\
                 \n\
                 help: rename one of these parameters:\n\
                 help:   |cx: f32, cy: f32| ...  // good: different names\n\
                 help:   |cx: f32, cx: f32| ...  // bad: duplicate 'cx'",
                name
            ),
        ));
    }

    // Register based on parameter kind
    match &param.kind {
        ParamKind::Scalar(ty) => {
            symbols.register_parameter(param.name.clone(), ty.clone());
        }
        ParamKind::Manifold => {
            symbols.register_manifold_param(param.name.clone());
        }
    }
    Ok(())
}

/// Validate an expression for symbol resolution (pure function).
///
/// Takes immutable references and returns () or Error. No mutation.
fn validate_expr(expr: &Expr, symbols: &SymbolTable, ctx: &SemaContext) -> syn::Result<()> {
    match expr {
        Expr::Ident(ident_expr) => {
            resolve_ident(&ident_expr.name, symbols, ctx)?;
        }

        Expr::Literal(_) => {
            // Literals are always valid
        }

        Expr::Binary(binary) => {
            validate_expr(&binary.lhs, symbols, ctx)?;
            validate_expr(&binary.rhs, symbols, ctx)?;
        }

        Expr::Unary(unary) => {
            validate_expr(&unary.operand, symbols, ctx)?;
        }

        Expr::MethodCall(call) => {
            validate_method_call(call, symbols, ctx)?;
        }

        Expr::Call(call) => {
            // Validate all arguments (function name is external, not resolved here)
            for arg in &call.args {
                validate_expr(arg, symbols, ctx)?;
            }
        }

        Expr::Block(block) => {
            validate_block(block, symbols, ctx)?;
        }

        Expr::Paren(inner) => {
            validate_expr(inner, symbols, ctx)?;
        }

        Expr::Verbatim(_) => {
            // Verbatim expressions pass through without analysis
            // The Rust compiler will catch any errors
        }
    }
    Ok(())
}

/// Resolve an identifier reference (pure function).
fn resolve_ident(ident: &Ident, symbols: &SymbolTable, ctx: &SemaContext) -> syn::Result<SymbolKind> {
    let name = ident.to_string();

    match symbols.lookup(&name) {
        Some(symbol) => Ok(symbol.kind),
        None => {
            // For anonymous kernels, unknown symbols are captured from environment
            // The Rust closure will handle the capture - no error needed
            if ctx.is_anonymous {
                return Ok(SymbolKind::Local); // Treat as external/captured
            }

            // For named kernels, undefined symbols are errors
            let suggestion = find_similar_symbol(&name, symbols);
            let msg = match suggestion {
                Some(similar) => format!(
                    "cannot find value `{}` in this scope\n\
                     \n\
                     help: a value with a similar name exists: `{}`\n\
                     \n\
                     note: available intrinsic coordinates: X, Y, Z, W\n\
                     note: these represent the evaluation point (x, y, z, w)\n\
                     \n\
                     help: if you meant to use a variable from the environment,\n\
                     help: use an anonymous kernel instead:\n\
                     help:   kernel!(|| ...) instead of kernel!(struct Foo = || ...)",
                    name, similar
                ),
                None => format!(
                    "cannot find value `{}` in this scope\n\
                     \n\
                     note: not found in kernel parameters or intrinsic coordinates\n\
                     \n\
                     help: add `{}` as a parameter to the kernel:\n\
                     help:   |{}: f32| ...  // scalar parameter\n\
                     help:   |{}: kernel| ...  // manifold parameter\n\
                     \n\
                     note: or use one of the intrinsic coordinates:\n\
                     note:   X, Y, Z, W  // the (x, y, z, w) evaluation point",
                    name, name, name, name
                ),
            };
            Err(syn::Error::new(ident.span(), msg))
        }
    }
}

/// Find a similar symbol name for typo suggestions (pure function).
fn find_similar_symbol(name: &str, symbols: &SymbolTable) -> Option<String> {
    let name_lower = name.to_lowercase();

    // Check intrinsics first (common typos)
    let intrinsics = ["X", "Y", "Z", "W"];
    for intr in intrinsics {
        if intr.to_lowercase() == name_lower {
            return Some(intr.to_string());
        }
    }

    // Check parameters and locals
    for sym_name in symbols.all_names() {
        // Simple similarity: same length and differs by 1-2 chars
        if sym_name.len() == name.len() {
            let diff_count = sym_name
                .chars()
                .zip(name.chars())
                .filter(|(a, b)| a != b)
                .count();
            if diff_count <= 2 {
                return Some(sym_name);
            }
        }
        // Case-insensitive match
        if sym_name.to_lowercase() == name_lower {
            return Some(sym_name);
        }
    }

    None
}

/// Known methods from ManifoldExt and standard operations.
const KNOWN_METHODS: &[&str] = &[
    // ManifoldExt methods
    "abs", "sqrt", "floor", "ceil", "round", "fract",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "exp", "exp2", "ln", "log2", "log10", "pow",
    "min", "max", "clamp",
    "hypot", "rsqrt", "recip",
    // Comparison methods
    "lt", "le", "gt", "ge", "eq", "ne",
    // Selection
    "select",
    // Coordinate warp (contramap)
    "at",
    // Field/Jet specific
    "constant", "collapse",
    // Unary
    "neg",
    // Clone for reusing expressions
    "clone",
];

/// Validate a method call (pure function).
fn validate_method_call(call: &MethodCallExpr, symbols: &SymbolTable, ctx: &SemaContext) -> syn::Result<()> {
    // Validate the receiver
    validate_expr(&call.receiver, symbols, ctx)?;

    // Validate arguments
    for arg in &call.args {
        validate_expr(arg, symbols, ctx)?;
    }

    // Validate method name against known methods
    let method_name = call.method.to_string();
    if !KNOWN_METHODS.contains(&method_name.as_str()) {
        // Find similar method for suggestion
        let suggestion = find_similar_method(&method_name);

        let msg = match suggestion {
            Some(similar) => format!(
                "no method named `{}` found\n\
                 \n\
                 help: there is a method with a similar name: `{}`\n\
                 \n\
                 note: common math methods:\n\
                 note:   sqrt, abs, floor, ceil, round\n\
                 note:   sin, cos, tan, exp, ln, pow\n\
                 note:   min, max, clamp",
                method_name, similar
            ),
            None => format!(
                "no method named `{}` found\n\
                 \n\
                 note: available methods on manifold expressions:\n\
                 note:   math: sqrt, abs, floor, ceil, round, fract\n\
                 note:   trig: sin, cos, tan, asin, acos, atan, atan2\n\
                 note:   exp:  exp, exp2, ln, log2, log10, pow\n\
                 note:   ops:  min, max, clamp, hypot, rsqrt, recip\n\
                 note:   cmp:  lt, le, gt, ge, eq, ne\n\
                 note:   misc: select, at, constant, collapse, neg, clone\n\
                 \n\
                 help: see the ManifoldExt trait for complete documentation",
                method_name
            ),
        };

        return Err(syn::Error::new(call.method.span(), msg));
    }
    Ok(())
}

/// Find a similar method name for typo suggestions (pure function).
fn find_similar_method(method_name: &str) -> Option<&'static str> {
    KNOWN_METHODS
        .iter()
        .find(|&&m| {
            let m_lower = m.to_lowercase();
            let name_lower = method_name.to_lowercase();
            m_lower == name_lower
                || (m.len() == method_name.len()
                    && m.chars()
                        .zip(method_name.chars())
                        .filter(|(a, b)| a != b)
                        .count()
                        <= 2)
        })
        .copied()
}

/// Validate a block expression with scoped symbols (functional approach).
///
/// Blocks introduce new scopes. We handle this by creating a new symbol table
/// with the block's local bindings added.
fn validate_block(block: &BlockExpr, symbols: &SymbolTable, ctx: &SemaContext) -> syn::Result<()> {
    // Clone the symbol table to add block-local bindings
    let mut scoped_symbols = symbols.clone();
    scoped_symbols.push_scope();

    // Validate each statement, accumulating locals
    for stmt in &block.stmts {
        match stmt {
            Stmt::Let(let_stmt) => {
                // First, validate the initializer (uses current scope)
                validate_expr(&let_stmt.init, &scoped_symbols, ctx)?;

                // Then register the new binding in the scoped table
                let name = let_stmt.name.to_string();

                // Warning: shadowing intrinsics in let is allowed but unusual
                if scoped_symbols.is_intrinsic(&name) {
                    // Could emit a warning here in the future
                }

                scoped_symbols.register_local(let_stmt.name.clone(), let_stmt.ty.clone());
            }
            Stmt::Expr(expr) => {
                validate_expr(expr, &scoped_symbols, ctx)?;
            }
        }
    }

    // Validate the final expression with the full scoped symbols
    if let Some(expr) = &block.expr {
        validate_expr(expr, &scoped_symbols, ctx)?;
    }

    // Scope is automatically cleaned up when scoped_symbols is dropped
    // (Rust's ownership handles the "pop_scope" automatically)

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use quote::quote;

    #[test]
    fn analyze_simple_kernel() {
        let input = quote! { |r: f32| X * X + Y * Y - r };
        let kernel = parse(input).unwrap();
        let analyzed = analyze(kernel).unwrap();

        assert!(analyzed.symbols.is_parameter("r"));
        assert!(analyzed.symbols.is_intrinsic("X"));
        assert!(analyzed.symbols.is_intrinsic("Y"));
    }

    #[test]
    fn error_on_undefined_symbol() {
        // Named kernels reject undefined symbols (anonymous kernels allow captures)
        let input = quote! { struct Test = |r: f32| X * X + undefined_var };
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("cannot find value"));
    }

    #[test]
    fn anonymous_allows_captured_variables() {
        // Anonymous kernels allow captured variables from environment
        let input = quote! { |r: f32| X * X + captured_from_env };
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);
        assert!(result.is_ok(), "Anonymous kernels should allow captured variables");
    }

    #[test]
    fn error_on_shadowing_intrinsic() {
        let input = quote! { |X: f32| X * X }; // X shadows the intrinsic
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("would shadow"));
    }

    #[test]
    fn block_scoping() {
        let input = quote! {
            |cx: f32| {
                let dx = X - cx;
                dx * dx
            }
        };
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);
        assert!(result.is_ok());
    }

    #[test]
    fn typo_suggestion_for_intrinsic() {
        // Lowercase "x" should suggest uppercase "X" (named kernel rejects typos)
        let input = quote! { struct Test = || x * x };
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("cannot find value"));
        assert!(err.contains("similar name exists: `X`"));
    }

    #[test]
    fn typo_suggestion_for_parameter() {
        // "radiu" should suggest "radius" (named kernel rejects typos)
        let input = quote! { struct Test = |radius: f32| X - radiu };
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("cannot find value"));
        // Similar names with 1-2 char difference should be suggested
    }

    #[test]
    fn error_on_unknown_method() {
        let input = quote! { |r: f32| X.unknownmethod() };
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("no method named"));
    }

    #[test]
    fn typo_suggestion_for_method() {
        // "sqrtt" should suggest "sqrt"
        let input = quote! { || X.sqrtt() };
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("no method named"));
    }

    #[test]
    fn known_methods_accepted() {
        // All ManifoldExt methods should be accepted
        let input = quote! { || X.sqrt().abs().sin().cos().clone() };
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);
        assert!(result.is_ok());
    }

    #[test]
    fn error_on_duplicate_parameter() {
        let input = quote! { |r: f32, r: f32| X - r };
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("duplicate parameter"));
    }
}
