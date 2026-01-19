//! # Semantic Analysis
//!
//! Analyzes the AST for semantic correctness and annotates it with symbol information.
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
//! - The original AST (possibly annotated)
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

/// Perform semantic analysis on a parsed kernel.
pub fn analyze(kernel: KernelDef) -> syn::Result<AnalyzedKernel> {
    let mut analyzer = SemanticAnalyzer::new();

    // Register all parameters in the symbol table
    for param in &kernel.params {
        analyzer.register_parameter(param)?;
    }

    // Analyze the body expression
    analyzer.analyze_expr(&kernel.body)?;

    Ok(AnalyzedKernel {
        def: kernel,
        symbols: analyzer.symbols,
    })
}

/// The semantic analyzer state.
struct SemanticAnalyzer {
    symbols: SymbolTable,
}

impl SemanticAnalyzer {
    fn new() -> Self {
        SemanticAnalyzer {
            symbols: SymbolTable::new(),
        }
    }

    /// Register a parameter in the symbol table.
    fn register_parameter(&mut self, param: &Param) -> syn::Result<()> {
        let name = param.name.to_string();

        // Check for shadowing intrinsics (error)
        if self.symbols.is_intrinsic(&name) {
            return Err(syn::Error::new(
                param.name.span(),
                format!(
                    "parameter '{}' shadows intrinsic coordinate variable",
                    name
                ),
            ));
        }

        // Check for duplicate parameters
        if self.symbols.lookup(&name).is_some() {
            return Err(syn::Error::new(
                param.name.span(),
                format!("duplicate parameter '{}'", name),
            ));
        }

        // Register based on parameter kind
        match &param.kind {
            ParamKind::Scalar(ty) => {
                self.symbols
                    .register_parameter(param.name.clone(), ty.clone());
            }
            ParamKind::Manifold => {
                self.symbols.register_manifold_param(param.name.clone());
            }
        }
        Ok(())
    }

    /// Analyze an expression for symbol resolution.
    fn analyze_expr(&mut self, expr: &Expr) -> syn::Result<()> {
        match expr {
            Expr::Ident(ident_expr) => {
                self.resolve_ident(&ident_expr.name)?;
            }

            Expr::Literal(_) => {
                // Literals are always valid
            }

            Expr::Binary(binary) => {
                self.analyze_expr(&binary.lhs)?;
                self.analyze_expr(&binary.rhs)?;
            }

            Expr::Unary(unary) => {
                self.analyze_expr(&unary.operand)?;
            }

            Expr::MethodCall(call) => {
                self.analyze_method_call(call)?;
            }

            Expr::Block(block) => {
                self.analyze_block(block)?;
            }

            Expr::Paren(inner) => {
                self.analyze_expr(inner)?;
            }

            Expr::Verbatim(_) => {
                // Verbatim expressions pass through without analysis
                // The Rust compiler will catch any errors
            }
        }
        Ok(())
    }

    /// Resolve an identifier reference.
    fn resolve_ident(&self, ident: &Ident) -> syn::Result<SymbolKind> {
        let name = ident.to_string();

        match self.symbols.lookup(&name) {
            Some(symbol) => Ok(symbol.kind),
            None => Err(syn::Error::new(
                ident.span(),
                format!("undefined symbol '{}'", name),
            )),
        }
    }

    /// Analyze a method call.
    fn analyze_method_call(&mut self, call: &MethodCallExpr) -> syn::Result<()> {
        // Analyze the receiver
        self.analyze_expr(&call.receiver)?;

        // Analyze arguments
        for arg in &call.args {
            self.analyze_expr(arg)?;
        }

        // Validate known methods (optional - could be extended)
        // For now, we trust ManifoldExt to handle unknown methods
        Ok(())
    }

    /// Analyze a block expression.
    fn analyze_block(&mut self, block: &BlockExpr) -> syn::Result<()> {
        // Enter a new scope
        self.symbols.push_scope();

        // Analyze each statement
        for stmt in &block.stmts {
            match stmt {
                Stmt::Let(let_stmt) => {
                    self.analyze_let(let_stmt)?;
                }
                Stmt::Expr(expr) => {
                    self.analyze_expr(expr)?;
                }
            }
        }

        // Analyze the final expression
        if let Some(expr) = &block.expr {
            self.analyze_expr(expr)?;
        }

        // Exit the scope
        self.symbols.pop_scope();

        Ok(())
    }

    /// Analyze a let statement.
    fn analyze_let(&mut self, let_stmt: &LetStmt) -> syn::Result<()> {
        // First, analyze the initializer (uses current scope)
        self.analyze_expr(&let_stmt.init)?;

        // Then register the new binding
        let name = let_stmt.name.to_string();

        // Warning: shadowing intrinsics in let is allowed but unusual
        if self.symbols.is_intrinsic(&name) {
            // Could emit a warning here in the future
        }

        self.symbols
            .register_local(let_stmt.name.clone(), let_stmt.ty.clone());

        Ok(())
    }
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
        let input = quote! { |r: f32| X * X + undefined_var };
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("undefined symbol"));
    }

    #[test]
    fn error_on_shadowing_intrinsic() {
        let input = quote! { |X: f32| X * X }; // X shadows the intrinsic
        let kernel = parse(input).unwrap();
        let result = analyze(kernel);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("shadows intrinsic"));
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
}
