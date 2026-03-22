import re

with open('pixelflow-compiler/src/annotate.rs', 'r') as f:
    content = f.read()
content = content.replace('AnnotatedStmt::Let(AnnotatedLet {', 'AnnotatedStmt::Let(Box::new(AnnotatedLet {')
content = content.replace('                    span: let_stmt.span,\n                })', '                    span: let_stmt.span,\n                }))')
with open('pixelflow-compiler/src/annotate.rs', 'w') as f:
    f.write(content)

with open('pixelflow-compiler/src/optimize.rs', 'r') as f:
    content = f.read()
content = content.replace('stmts.push(Stmt::Let(LetStmt {', 'stmts.push(Stmt::Let(Box::new(LetStmt {')
content = content.replace('                    span,\n                }));', '                    span,\n                })));')
with open('pixelflow-compiler/src/optimize.rs', 'w') as f:
    f.write(content)

with open('pixelflow-compiler/src/parser.rs', 'r') as f:
    content = f.read()
content = content.replace('ParamKind::Scalar(ty)', 'ParamKind::Scalar(Box::new(ty))')
content = content.replace('stmts.push(Stmt::Let(LetStmt {', 'stmts.push(Stmt::Let(Box::new(LetStmt {')
content = content.replace('                    span: Span::call_site(),\n                }));', '                    span: Span::call_site(),\n                })));')
with open('pixelflow-compiler/src/parser.rs', 'w') as f:
    f.write(content)

with open('pixelflow-compiler/src/sema.rs', 'r') as f:
    content = f.read()
content = content.replace('.register_parameter(param.name.clone(), ty.clone());', '.register_parameter(param.name.clone(), *ty.clone());')
with open('pixelflow-compiler/src/sema.rs', 'w') as f:
    f.write(content)
