import re

def fix_file(filepath, replacements):
    with open(filepath, 'r') as f:
        content = f.read()
    for search, replace in replacements:
        content = content.replace(search, replace)
    with open(filepath, 'w') as f:
        f.write(content)

fix_file('pixelflow-compiler/src/annotate.rs', [
    ('use crate::ast::{BinaryOp, BlockExpr, CallExpr, Expr, IdentExpr, Stmt, UnaryOp};', 'use crate::ast::{BinaryOp, BlockExpr, Expr, IdentExpr, Stmt, UnaryOp};'),
])

fix_file('pixelflow-compiler/src/fold.rs', [
    ('use crate::ast::{BinaryExpr, BinaryOp, BlockExpr, CallExpr, Expr, IdentExpr, LiteralExpr, MethodCallExpr, Stmt, UnaryExpr, UnaryOp};', 'use crate::ast::{BinaryOp, Expr, Stmt, UnaryOp};'),
])

fix_file('pixelflow-compiler/src/ir_bridge.rs', [
    ('use crate::ast::{BinaryExpr, BinaryOp, Expr, LiteralExpr, UnaryOp};', 'use crate::ast::{BinaryOp, Expr, UnaryOp};'),
    ('use proc_macro2::{Span, TokenStream};', 'use proc_macro2::{TokenStream};'),
    ('use syn::{Ident, Lit};', 'use syn::{Lit};'),
])

print("Fixes applied.")
