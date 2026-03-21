import re

def fix_file(filepath, replacements):
    with open(filepath, 'r') as f:
        content = f.read()
    for search, replace in replacements:
        content = content.replace(search, replace)
    with open(filepath, 'w') as f:
        f.write(content)

# We missed the unpacking side of Boxed enums and the if let collapse might be wrong
fix_file('pixelflow-compiler/src/annotate.rs', [
    ('AnnotatedStmt::Let(l) => {', 'AnnotatedStmt::Let(l) => {')
])

fix_file('pixelflow-compiler/src/codegen/emitter.rs', [
    ('if let Some(AnnotatedExpr::Ident(ident_expr)) = call.args.first() {\n                    // Check if the argument is a manifold param or a local bound to one\n', 'if let Some(AnnotatedExpr::Ident(ident_expr)) = call.args.first() {\n                    // Check if the argument is a manifold param or a local bound to one\n')
])

fix_file('pixelflow-compiler/src/ast.rs', [
])
