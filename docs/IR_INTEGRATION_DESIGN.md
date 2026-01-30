# IR Integration Design: Unifying the Compiler Pipeline

**Status:** Design Proposal
**Date:** 2026-01-30
**Context:** Finding #1 from COMPILER_ANALYSIS.md

## Problem Statement

The PixelFlow compiler currently maintains **three separate AST/IR representations**:

1. **Source AST** (`pixelflow-macros/src/ast.rs`) - 263 lines
2. **IR** (`pixelflow-ir/src/expr.rs`) - 63 lines, **UNUSED**
3. **ENode** (`pixelflow-search/src/egraph/node.rs`) - E-graph representation

This creates redundancy, conversion overhead, and maintenance burden.

## Current Pipeline

```rust
// pixelflow-macros/src/optimize.rs
Source Code
    ↓ Parse
AST (BinaryExpr, MethodCallExpr, etc.)
    ↓ ast_to_egraph() - CUSTOM CONVERSION
ENode (internal to E-graph)
    ↓ Saturate & extract
Optimized ENode
    ↓ tree_to_expr() - CUSTOM CONVERSION
AST (reconstructed)
    ↓ Codegen
Type-level code
```

## Proposed Pipeline

```rust
Source Code
    ↓ Parse
AST (temporary - for source tracking)
    ↓ ast_to_ir() - SINGLE CONVERSION
IR (pixelflow_ir::Expr)
    ↓ Add to EGraph<IR>
E-graph (operates on IR node IDs)
    ↓ Saturate & extract
Optimized IR
    ↓ ir_to_code() - SINGLE CONVERSION
Type-level code
```

**Key difference:** IR is the canonical representation. E-graph stores node IDs, not a separate representation.

## Design

### 1. Language Trait

Define a trait that IR must implement for E-graph compatibility:

```rust
// pixelflow-search/src/language.rs (NEW)
pub trait Language: Clone + Eq + Hash {
    /// Iterate over children (e-class IDs)
    fn children(&self) -> impl Iterator<Item = EClassId>;

    /// Create a new node with updated children
    fn with_children(&self, children: impl Iterator<Item = EClassId>) -> Self;

    /// Get operation cost for extraction
    fn cost(&self, costs: &CostModel) -> usize;

    /// Display for debugging
    fn display(&self) -> String;
}
```

### 2. Implement Language for IR

```rust
// pixelflow-ir/src/egraph_impl.rs (NEW)
use pixelflow_search::{Language, EClassId};

impl Language for Expr {
    fn children(&self) -> impl Iterator<Item = EClassId> {
        match self {
            Expr::Var(_) | Expr::Const(_) => {
                std::iter::empty()
            }
            Expr::Unary(_, child) => {
                std::iter::once(*child)
            }
            Expr::Binary(_, left, right) => {
                std::iter::once(*left)
                    .chain(std::iter::once(*right))
            }
            Expr::Ternary(_, a, b, c) => {
                std::iter::once(*a)
                    .chain(std::iter::once(*b))
                    .chain(std::iter::once(*c))
            }
            Expr::Nary(_, children) => {
                children.iter().copied()
            }
        }
    }

    fn with_children(&self, mut children: impl Iterator<Item = EClassId>) -> Self {
        match self {
            Expr::Var(v) => Expr::Var(*v),
            Expr::Const(c) => Expr::Const(*c),
            Expr::Unary(op, _) => {
                Expr::Unary(*op, Box::new(children.next().unwrap()))
            }
            Expr::Binary(op, _, _) => {
                let left = children.next().unwrap();
                let right = children.next().unwrap();
                Expr::Binary(*op, Box::new(left), Box::new(right))
            }
            Expr::Ternary(op, _, _, _) => {
                let a = children.next().unwrap();
                let b = children.next().unwrap();
                let c = children.next().unwrap();
                Expr::Ternary(*op, Box::new(a), Box::new(b), Box::new(c))
            }
            Expr::Nary(op, _) => {
                Expr::Nary(*op, children.collect())
            }
        }
    }

    fn cost(&self, costs: &CostModel) -> usize {
        match self {
            Expr::Var(_) | Expr::Const(_) => 0,
            Expr::Unary(op, _) | Expr::Binary(op, _, _) |
            Expr::Ternary(op, _, _, _) | Expr::Nary(op, _) => {
                costs.cost_by_name(op.name())
            }
        }
    }

    fn display(&self) -> String {
        match self {
            Expr::Var(idx) => format!("v{}", idx),
            Expr::Const(val) => format!("{}", val),
            Expr::Unary(op, _) => op.name().to_string(),
            Expr::Binary(op, _, _) => op.name().to_string(),
            Expr::Ternary(op, _, _, _) => op.name().to_string(),
            Expr::Nary(op, _) => op.name().to_string(),
        }
    }
}
```

### 3. Update E-graph to be Generic

```rust
// pixelflow-search/src/egraph/graph.rs
pub struct EGraph<L: Language> {
    classes: Vec<EClass<L>>,
    parent: Vec<EClassId>,
    memo: HashMap<L, EClassId>,  // Node → EClass
    // ...
}

impl<L: Language> EGraph<L> {
    pub fn add(&mut self, node: L) -> EClassId {
        if let Some(&id) = self.memo.get(&node) {
            return self.find(id);
        }

        let id = EClassId(self.classes.len());
        self.classes.push(EClass::new(node.clone()));
        self.memo.insert(node, id);
        id
    }

    // ... rest of implementation
}
```

### 4. Update Macro Pipeline

```rust
// pixelflow-macros/src/optimize.rs
use pixelflow_ir::Expr as IR;

pub fn optimize(analyzed: AnalyzedKernel) -> AnalyzedKernel {
    // 1. Convert AST → IR (once)
    let ir = ast_to_ir(&analyzed.def.body);

    // 2. Optimize via E-graph on IR
    let optimized_ir = optimize_ir(ir);

    // 3. Codegen from IR (not AST!)
    let code = ir_to_code(&optimized_ir);

    analyzed
}

fn optimize_ir(ir: IR) -> IR {
    let mut egraph = EGraph::<IR>::new();
    let root = egraph.add_recursive(ir);
    egraph.saturate();
    egraph.extract(root, &CostModel::load_or_default())
}

fn ast_to_ir(ast: &ast::Expr) -> IR {
    match ast {
        ast::Expr::Binary(b) => {
            let lhs = ast_to_ir(&b.lhs);
            let rhs = ast_to_ir(&b.rhs);
            let op = match b.op {
                ast::BinaryOp::Add => OpKind::Add,
                ast::BinaryOp::Mul => OpKind::Mul,
                // ...
            };
            IR::Binary(op, Box::new(lhs), Box::new(rhs))
        }
        // ... other conversions
    }
}

fn ir_to_code(ir: &IR) -> TokenStream {
    // Generate type-level AST from IR
    match ir {
        IR::Binary(OpKind::Add, left, right) => {
            let left_code = ir_to_code(left);
            let right_code = ir_to_code(right);
            quote! { Add::new(#left_code, #right_code) }
        }
        // ... other code generation
    }
}
```

## Implementation Phases

### Phase 1: Language Trait (Non-Breaking)
- Define `Language` trait in `pixelflow-search`
- Implement `Language` for current `ENode`
- Make `EGraph` generic over `Language`
- **Status:** Backward compatible

### Phase 2: IR Integration (Breaking)
- Implement `Language` for `pixelflow_ir::Expr`
- Add `ast_to_ir()` conversion in `pixelflow-macros`
- Add `ir_to_code()` codegen from IR
- Update `optimize()` to use IR pipeline
- **Status:** Breaking change

### Phase 3: Cleanup (Breaking)
- Delete `ENode` type (replaced by IR)
- Delete AST-based codegen (replaced by IR codegen)
- Update tests
- **Status:** Cleanup

## Benefits

1. **Single Source of Truth**: IR is the canonical representation
2. **Reduced Conversions**: 4 conversions → 2 conversions
3. **Reusable IR**: Can optimize expressions at runtime
4. **Simpler Maintenance**: One node type instead of three
5. **Cleaner Separation**:
   - AST = parsing with spans
   - IR = semantics
   - E-graph = optimization data structure

## Risks & Mitigations

### Risk 1: Breaking Changes
**Mitigation:** Implement in phases, keep backward compatibility in Phase 1

### Risk 2: Performance Regression
**Mitigation:**
- Benchmark before/after
- IR is simpler than current AST, likely faster
- E-graph operations unchanged

### Risk 3: Feature Loss
**Impact:** Current ENode has some features IR doesn't:
- Opaque nodes for method calls
- Special handling for tuples

**Mitigation:**
- Extend IR with needed features
- Add `OpKind::Opaque` for unoptimizable nodes
- Existing `Nary` handles tuples

### Risk 4: Circular Dependencies
**Current:**
```
pixelflow-core → pixelflow-macros → pixelflow-search → pixelflow-nnue
```

**After adding pixelflow-search → pixelflow-ir:**
```
pixelflow-core → pixelflow-macros → pixelflow-search → pixelflow-ir
                                   ↓
                            pixelflow-nnue
```

**Status:** No cycle - pixelflow-ir has no dependencies

## Testing Strategy

1. **Unit Tests**: Test `ast_to_ir()` and `ir_to_code()` conversions
2. **Integration Tests**: Existing macro tests should pass
3. **Benchmark Tests**: Compare optimization quality before/after
4. **Regression Tests**: Run full test suite

## Alternative Designs Considered

### Alternative 1: Keep ENode, Delete IR
**Rejected:** ENode is specific to E-graph, can't be used elsewhere

### Alternative 2: Keep Three Representations
**Rejected:** Current status quo has too much redundancy

### Alternative 3: Use egg's RecExpr
**Rejected:** Adds dependency, need custom features (opaque nodes)

## Decision

**Proceed with IR integration** using the phased approach:
- Phase 1: Make E-graph generic (non-breaking)
- Phase 2: Integrate IR (breaking, v2.0)
- Phase 3: Cleanup

This aligns with Finding #1 recommendation: "Fix in v2.0 (breaking)".

## Next Steps

1. Review this design with maintainers
2. Implement Phase 1 (Language trait)
3. Benchmark current pipeline
4. Implement Phase 2 (IR integration)
5. Validate with full test suite
6. Document migration guide

---

**Estimated Effort:** 2-3 days
**Breaking Change:** Yes (Phase 2+)
**Risk Level:** Medium
**Recommendation:** Implement in v2.0 milestone
