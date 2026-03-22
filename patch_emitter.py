with open('pixelflow-compiler/src/codegen/emitter.rs', 'r') as f:
    content = f.read()

import re

# Fix collapsible if let
content = re.sub(
    r'                if let Some\(arg\) = call\.args\.first\(\) \{\n                    // Check if the argument is a manifold param or a local bound to one\n                    if let AnnotatedExpr::Ident\(ident_expr\) = arg \{',
    r'''                if let Some(AnnotatedExpr::Ident(ident_expr)) = call.args.first() {''',
    content
)
# We need to remove the extra closing brace from `if let Some(arg)`
# This regex removes one layer of indentation and the closing brace.
content = re.sub(
    r'''                        if let Some\(manifold_name\) = locals_to_manifolds\.get\(&name_str\) \{\n                            result\.insert\(manifold_name\.clone\(\)\);\n                        \} else if let Some\(symbol\) = symbols\.lookup\(&name_str\) \{\n                            // Direct manifold param reference\n                            if matches!\(symbol\.kind, SymbolKind::ManifoldParam\) \{\n                                result\.insert\(name_str\);\n                            \}\n                        \}\n                    \}\n                \}''',
    r'''                        if let Some(manifold_name) = locals_to_manifolds.get(&name_str) {
                            result.insert(manifold_name.clone());
                        } else if let Some(symbol) = symbols.lookup(&name_str) {
                            // Direct manifold param reference
                            if matches!(symbol.kind, SymbolKind::ManifoldParam) {
                                result.insert(name_str);
                            }
                        }
                }''',
    content
)

with open('pixelflow-compiler/src/codegen/emitter.rs', 'w') as f:
    f.write(content)
