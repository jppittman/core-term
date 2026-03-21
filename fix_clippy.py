import re
import os

def fix_file(filepath, replacements):
    with open(filepath, 'r') as f:
        content = f.read()
    for search, replace in replacements:
        content = content.replace(search, replace)
    with open(filepath, 'w') as f:
        f.write(content)

fix_file('pixelflow-compiler/src/lib.rs', [
    ('#![forbid(unsafe_code)]\n', '#![forbid(unsafe_code)]\n#![allow(dead_code)]\n')
])

fix_file('pixelflow-compiler/src/annotate.rs', [
    ('Let(AnnotatedLet),', 'Let(Box<AnnotatedLet>),'),
    ('..ctx\n                }', '}'),
    ('Stmt::Let(l) => AnnotatedStmt::Let(AnnotatedLet {', 'Stmt::Let(l) => AnnotatedStmt::Let(Box::new(AnnotatedLet {'),
    ('}),\n            Stmt::Expr', '})),\n            Stmt::Expr'),
    ('AnnotatedStmt::Let(l) => {\n                let mut child_ctx', 'AnnotatedStmt::Let(l) => {\n                let mut child_ctx') # Will use regex for this if needed
])

fix_file('pixelflow-compiler/src/ast.rs', [
    ('Scalar(Type),', 'Scalar(Box<Type>),'),
    ('Let(LetStmt),', 'Let(Box<LetStmt>),')
])

fix_file('pixelflow-compiler/src/codegen/emitter.rs', [
    ('                if let Some(arg) = call.args.first() {\n                    // Check if the argument is a manifold param or a local bound to one\n                    if let AnnotatedExpr::Ident(ident_expr) = arg {\n', '                if let Some(AnnotatedExpr::Ident(ident_expr)) = call.args.first() {\n                    // Check if the argument is a manifold param or a local bound to one\n'),
    ('                    }\n                }', '                }'),
    ('            } else if manifold_count == 0 && params.len() == 1 {\n                Derives::CloneCopy', '')
])

fix_file('pixelflow-compiler/src/codegen/leveled.rs', [
    ('.find(|p| p.name.to_string() == name)', '.find(|p| p.name == name)'),
    ('(LeveledNodeKind::Unary { op: unary.op.clone(), operand }, operand_deps)', '(LeveledNodeKind::Unary { op: unary.op, operand }, operand_deps)'),
    ('(LeveledNodeKind::Binary { op: binary.op.clone(), left, right }, deps)', '(LeveledNodeKind::Binary { op: binary.op, left, right }, deps)')
])

fix_file('pixelflow-compiler/src/codegen/struct_emitter.rs', [
    ('    pub fn with_eval_body(', '    #[allow(clippy::too_many_arguments)]\n    pub fn with_eval_body(')
])

fix_file('pixelflow-compiler/src/fold.rs', [
    ('.filter_map(|stmt| match stmt {\n                    Stmt::Let(let_stmt) => {\n                        let init = fold_expr(folder, &let_stmt.init);\n                        Some(folder.fold_let(&let_stmt.name, init))\n                    }\n                    Stmt::Expr(e) => Some(fold_expr(folder, e)),\n                })', '.map(|stmt| match stmt {\n                    Stmt::Let(let_stmt) => {\n                        let init = fold_expr(folder, &let_stmt.init);\n                        folder.fold_let(&let_stmt.name, init)\n                    }\n                    Stmt::Expr(e) => fold_expr(folder, e),\n                })')
])

fix_file('pixelflow-compiler/src/ir_bridge.rs', [
    ('Err(format!("Non-numeric literal"))', 'Err("Non-numeric literal".to_string())'),
    ('Err(format!("Unsupported unary op: Not"))', 'Err("Unsupported unary op: Not".to_string())'),
    ('Err(format!("Unsupported expression type"))', 'Err("Unsupported expression type".to_string())'),
    ('.map(|c| egraph_to_ir(c))', '.map(egraph_to_ir)')
])

fix_file('pixelflow-compiler/src/optimize.rs', [
    ('            if matches!(call.receiver.as_ref(), Expr::Verbatim(_)) {\n                if call.args.iter().any(|arg| expr_references_any(arg, local_names)) {\n                    return true;\n                }\n            }', '            if matches!(call.receiver.as_ref(), Expr::Verbatim(_)) && call.args.iter().any(|arg| expr_references_any(arg, local_names)) {\n                return true;\n            }'),
    ('.map_or(false, |e| expr_has_opaque_refs(e, local_names))', '.is_some_and(|e| expr_has_opaque_refs(e, local_names))'),
    ('.map_or(false, |e| expr_references_any(e, names))', '.is_some_and(|e| expr_references_any(e, names))'),
    ('.map_or(false, |init| {', '.is_some_and(|init| {'),
    ('.map_or(false, |(_, else_expr)| {', '.is_some_and(|(_, else_expr)| {')
])

fix_file('pixelflow-compiler/src/parser.rs', [
    ('&expr_binary.op', 'expr_binary.op'),
    ('&expr_unary.op', 'expr_unary.op'),
    ('ParamKind::Scalar(ty.clone())', 'ParamKind::Scalar(Box::new(ty.clone()))'),
    ('Stmt::Let(let_stmt)', 'Stmt::Let(Box::new(let_stmt))')
])

fix_file('pixelflow-compiler/src/symbol.rs', [
    ('.map_or(false, |s| s.kind == SymbolKind::Intrinsic)', '.is_some_and(|s| s.kind == SymbolKind::Intrinsic)'),
    ('.map_or(false, |s| s.kind == SymbolKind::Parameter)', '.is_some_and(|s| s.kind == SymbolKind::Parameter)'),
    ('.map_or(false, |s| s.kind == SymbolKind::ManifoldParam)', '.is_some_and(|s| s.kind == SymbolKind::ManifoldParam)')
])

print("Fixes applied.")
