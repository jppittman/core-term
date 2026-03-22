with open('pixelflow-compiler/src/annotate.rs', 'r') as f:
    content = f.read()
content = content.replace('Let(AnnotatedLet),', 'Let(Box<AnnotatedLet>),')
content = content.replace('..ctx', '')
with open('pixelflow-compiler/src/annotate.rs', 'w') as f:
    f.write(content)

with open('pixelflow-compiler/src/ast.rs', 'r') as f:
    content = f.read()
content = content.replace('Scalar(Type),', 'Scalar(Box<Type>),')
content = content.replace('Let(LetStmt),', 'Let(Box<LetStmt>),')
with open('pixelflow-compiler/src/ast.rs', 'w') as f:
    f.write(content)

with open('pixelflow-compiler/src/codegen/emitter.rs', 'r') as f:
    content = f.read()
content = content.replace('''                if let Some(arg) = call.args.first() {
                    // Check if the argument is a manifold param or a local bound to one
                    if let AnnotatedExpr::Ident(ident_expr) = arg {''',
'''                if let Some(AnnotatedExpr::Ident(ident_expr)) = call.args.first() {''')
with open('pixelflow-compiler/src/codegen/emitter.rs', 'w') as f:
    f.write(content)
