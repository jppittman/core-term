# PixelFlow Compiler Pipeline Analysis

**Date:** 2026-01-30
**Agent ID:** a2c6788
**Session:** claude/analyze-pixelflow-compiler-Q5H2f

## Executive Summary

This document analyzes the PixelFlow compiler pipeline (IR, search, macros, core) to identify missing integrations, optimization opportunities, and architectural inconsistencies. The analysis reveals **7 major findings**, ranked by impact and feasibility.

## Compiler Architecture Overview

PixelFlow uses a multi-stage compilation pipeline that transforms user code into optimized SIMD kernels:

```
User Code (kernel! macro)
    │
    ▼ [pixelflow-compiler] Lexer → Parser
Source AST (ast.rs)
    ├─ BinaryExpr, MethodCallExpr, BlockExpr
    └─ Parameter binding analysis
    │
    ▼ [pixelflow-compiler] Semantic Analysis
Analyzed AST + Symbol Table
    │
    ▼ [pixelflow-compiler] Optimization (optimize.rs)
    ├─ Pass 1: Structural (tree peephole)
    │   ├─ Constant folding: 1.0 + 2.0 → 3.0
    │   ├─ Identity removal: x + 0.0 → x
    │   └─ Zero propagation: x * 0.0 → 0.0
    │
    └─ Pass 2: Global (E-graph saturation)
        ├─ [pixelflow-search] EGraph equality saturation
        ├─ FMA fusion: a*b+c → mul_add(a,b,c)
        ├─ Rsqrt: 1/sqrt(y) → rsqrt(y)
        ├─ Algebraic identities (commutativity, associativity)
        └─ Cost-based extraction (minimal runtime cost)
    │
    ▼ [pixelflow-compiler] Code Generation (codegen/)
Type-Level AST (Rust code)
    ├─ Add<X, Y>, Mul<Z, W>, Sqrt<...>
    └─ WithContext for parameter binding
    │
    ▼ Rust Compiler (monomorphization)
    ├─ Inline all Manifold::eval calls
    ├─ Specialize for SIMD backend (AVX-512/SSE2/NEON)
    └─ Emit optimized machine code
    │
    ▼ Runtime Execution
SIMD Assembly (5ns/pixel @ 1080p)
```

### Three Representations

The pipeline uses **three distinct AST/IR representations**:

1. **Source AST** (`pixelflow-compiler/src/ast.rs`)
   - User-facing syntax tree
   - Preserves source structure (blocks, let bindings, method calls)
   - Example: `BinaryExpr { op: Add, lhs: X, rhs: Literal(1.0) }`

2. **IR AST** (`pixelflow-ir/src/expr.rs`) — **UNUSED!**
   - Intended as canonical optimization IR
   - Simpler enum: `Expr::Binary(OpKind, Box<Expr>, Box<Expr>)`
   - **Problem:** Not integrated into compilation pipeline

3. **Type-Level AST** (`pixelflow-core/src/ops/`)
   - Runtime representation encoded in Rust's type system
   - Example: `Add<Mul<X, CtxVar<A0, 0>>, Y>`
   - Manifold trait recursively evaluates to SIMD `Field` values

## Critical Findings

### 1. 🔴 CRITICAL: IR Crate Not Integrated into Pipeline

**Severity:** High
**Effort:** Medium (macro changes only, search stays generic)
**Impact:** Foundational - IR becomes canonical representation

#### Problem

The `pixelflow-ir` crate defines a clean `Expr` type:

```rust
// pixelflow-ir/src/expr.rs
pub enum Expr {
    Var(u8),
    Const(f32),
    Unary(OpKind, Box<Expr>),
    Binary(OpKind, Box<Expr>, Box<Expr>),
    Ternary(OpKind, Box<Expr>, Box<Expr>, Box<Expr>),
    Nary(OpKind, Vec<Expr>),
}
```

**BUT:** The macro compiler never uses it!

Current flow:
```
Macro AST → ENode (pixelflow-search) → Optimized AST → Code
```

IR is completely bypassed. The macro duplicates IR functionality with a 263-line AST.

#### Architecture Clarification

**pixelflow-search is a GENERIC framework:**
- E-graph + NNUE for optimization
- Works with `ENode` which can represent any language
- Should NOT depend on `pixelflow-ir`

**pixelflow-ir is the LANGUAGE:**
- Defines the specific IR for PixelFlow kernels
- Just data structures (Expr, OpKind)

**pixelflow-compiler is the FRONTEND:**
- Consumes BOTH search (framework) and IR (language)
- Should compile: AST → IR → E-graph → IR → Code

#### Why This Matters

1. **Code Duplication:** Macro AST duplicates what IR should do
2. **Lost Abstraction:** No canonical representation for kernel expressions
3. **Maintenance Burden:** Changes require updating both AST and conversion code
4. **No Reusability:** Can't build/optimize IR expressions outside macros
5. **Missed Opportunity:** IR exists but is completely unused

#### Recommended Fix

Add `pixelflow-ir` to macro pipeline:

```rust
// pixelflow-compiler/Cargo.toml
[dependencies]
pixelflow-ir = { path = "../pixelflow-ir" }
pixelflow-search = { path = "../pixelflow-search" }

// pixelflow-compiler/src/optimize.rs
use pixelflow_ir::Expr as IR;

pub fn optimize(analyzed: AnalyzedKernel) -> AnalyzedKernel {
    // 1. Convert macro AST → IR (once)
    let ir = ast_to_ir(&analyzed.def.body);

    // 2. Flatten IR tree → E-graph
    let mut egraph = EGraph::new();
    let root = ir_to_egraph(&ir, &mut egraph);

    // 3. Optimize via E-graph saturation
    egraph.saturate();

    // 4. Extract optimized IR tree
    let optimized_ir = egraph_to_ir(&egraph, root);

    // 5. Generate code from IR
    let code = ir_to_code(&optimized_ir);

    analyzed
}
```

**Benefits:**
- IR becomes the canonical representation
- E-graph stays generic (no dependency on specific IR)
- Can build/optimize IR expressions outside macros (future)
- Cleaner separation: parsing (AST) → semantics (IR) → optimization (E-graph) → codegen

---

### 2. 🟡 HIGH PRIORITY: No Cost Model Calibration from Benchmarks

**Severity:** Medium
**Effort:** Low (infrastructure exists, just needs integration)
**Impact:** High (immediate optimization quality improvement)

#### Problem

The cost model (`pixelflow-search/src/egraph/cost.rs`) has **hardcoded operation costs**:

```rust
impl Default for CostModel {
    fn default() -> Self {
        Self {
            add: 4,
            sub: 4,
            mul: 5,
            div: 15,
            sqrt: 15,
            mul_add: 10,  // ← GUESS! Should be 5 on AVX-512 with FMA
            // ...
        }
    }
}
```

These are **educated guesses**, not measured values!

#### Infrastructure Already Exists

1. **Benchmarks measure individual operations:**
   ```rust
   // pixelflow-core/benches/core_benches.rs
   fn bench_field_arithmetic(c: &mut Criterion) {
       group.bench_function("add", |b| { ... });    // ← Measures add latency
       group.bench_function("mul", |b| { ... });    // ← Measures mul latency
       group.bench_function("chained_mad", |b| {    // ← Measures FMA
           bencher.iter(|| (a * b + c).constant())
       });
   }
   ```

2. **Cost model can save/load from TOML:**
   ```rust
   impl CostModel {
       pub fn save_toml(&self, path: P) -> io::Result<()> { ... }
       pub fn load_toml(path: P) -> io::Result<Self> { ... }
       pub fn load_or_default() -> Self {
           // Checks $PIXELFLOW_COST_MODEL, ~/.config/pixelflow/, etc.
       }
   }
   ```

**Missing:** The integration layer to go from benchmark results → cost model TOML.

#### Current Consequences

1. **Suboptimal extraction:** E-graph might choose wrong form
   - Example: FMA fusion disabled because `mul_add` cost is too high (10 > 5+4)
   - Reality: FMA on AVX-512 is ~5 cycles (single instruction)

2. **Platform-blind:** Same costs used on AVX-512, SSE2, NEON, scalar
   - SSE2 doesn't have FMA → should cost 5+4=9
   - AVX-512 has FMA → should cost 5

3. **No empirical validation:** Can't A/B test cost models to improve quality

#### Recommended Fix

**Create benchmark harness that generates cost models:**

```rust
// pixelflow-search/src/bin/calibrate_costs.rs (NEW)
use criterion::Criterion;
use std::time::Duration;

pub fn calibrate_from_benchmarks() -> CostModel {
    let mut c = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(1000);

    // Measure each operation
    let add_ns = bench_operation(&mut c, "add", |a, b| a + b);
    let mul_ns = bench_operation(&mut c, "mul", |a, b| a * b);
    let mul_add_ns = bench_operation(&mut c, "mul_add", |a, b, c| {
        a.mul_add(b, c)
    });

    // Normalize to fastest operation
    let base = add_ns.min(mul_ns);

    CostModel {
        add: (add_ns / base) as usize,
        mul: (mul_ns / base) as usize,
        mul_add: (mul_add_ns / base) as usize,
        // ...
    }
}

fn main() {
    let model = calibrate_from_benchmarks();
    model.save_toml("~/.config/pixelflow/cost_model.toml")
        .expect("Failed to save cost model");
    println!("Calibrated cost model saved!");
}
```

**Usage:**
```bash
$ cargo run --bin calibrate_costs --release
Benchmarking add... 2.1ns
Benchmarking mul... 2.3ns
Benchmarking mul_add... 2.2ns (FMA detected!)
...
Calibrated cost model saved to ~/.config/pixelflow/cost_model.toml

$ cat ~/.config/pixelflow/cost_model.toml
# Learned cost model weights
# Generated from SIMD benchmark measurements on 2026-01-30
add = 4
mul = 5
mul_add = 5  # ← Now accurate!
```

**Impact:**
- Optimizer makes better decisions (FMA fusion, rsqrt, etc.)
- Platform-specific tuning (different models for AVX-512 vs SSE2)
- Empirical validation of optimization heuristics

---

### 3. 🟡 MEDIUM: Dead Code Elimination Not Implemented

**Severity:** Low
**Effort:** Low
**Impact:** Medium (cleaner generated code, faster compilation)

#### Problem

Unused let bindings are never eliminated:

```rust
kernel!(|x: f32| {
    let unused = X * X + Y * Y;  // ← Never referenced
    let used = Z * x;
    used
})
```

**Current behavior:** `unused` is computed and stored, even though it's never used.

**Expected:** DCE pass removes `unused` entirely.

#### Recommended Fix

Add DCE pass to `optimize.rs`:

```rust
fn eliminate_dead_code(block: &mut BlockExpr) {
    let used_vars = find_used_variables(&block.expr);
    block.stmts.retain(|stmt| {
        match stmt {
            Stmt::Let(let_stmt) => used_vars.contains(&let_stmt.name),
            _ => true,
        }
    });
}
```

---

### 4. 🟢 LOW: Type-Level AST ↔ IR Bridge Missing

**Severity:** Low
**Effort:** Medium
**Impact:** Low (architectural elegance)

#### Problem

Runtime types (pixelflow-core) and IR (pixelflow-ir) represent same structure differently:

**Runtime:**
```rust
// Type-level AST
type Circle = Sqrt<Add<Mul<X, X>, Mul<Y, Y>>>;
```

**IR:**
```rust
// Runtime IR
Expr::Unary(Sqrt, Box::new(
    Expr::Binary(Add,
        Box::new(Expr::Binary(Mul, Var(0), Var(0))),
        Box::new(Expr::Binary(Mul, Var(1), Var(1)))
    )
))
```

**Missing:** Bidirectional translation:
- `ir::Expr` → `impl Manifold` (generate runtime types from IR)
- `impl Manifold` → `ir::Expr` (reflection/introspection)

#### Recommended Fix

```rust
// pixelflow-core/src/ir_interop.rs (NEW)
pub trait ToIR {
    fn to_ir(&self) -> ir::Expr;
}

impl<L, R> ToIR for Add<L, R>
where
    L: ToIR,
    R: ToIR,
{
    fn to_ir(&self) -> ir::Expr {
        ir::Expr::Binary(
            ir::OpKind::Add,
            Box::new(self.0.to_ir()),
            Box::new(self.1.to_ir()),
        )
    }
}

// Generate Manifold impl from IR
pub fn generate_manifold(expr: &ir::Expr) -> TokenStream {
    // Used by macro codegen
}
```

---

### 5. 🟢 LOW: Rewrite Rules Missing Derivative Identities

**Severity:** Low
**Effort:** Medium
**Impact:** Low (niche optimization for autodiff)

#### Problem

E-graph doesn't optimize derivative computations:

```rust
kernel!(|x: f32| -> Jet2 {
    let y = x * x;
    DX(y)  // derivative of x² with respect to X
})
```

**Current:** Computes derivative symbolically, no algebraic simplification
**Opportunity:** Add rewrite rules for derivatives:
- `DX(X) → 1`
- `DX(const) → 0`
- `DX(f + g) → DX(f) + DX(g)`
- `DX(f * g) → DX(f)*g + f*DX(g)` (product rule)

#### Recommended Fix

```rust
// pixelflow-search/src/egraph/derivative_rules.rs (NEW)
pub struct DerivativeIdentity;

impl Rewrite for DerivativeIdentity {
    fn apply(&self, egraph: &EGraph, node: &ENode) -> Option<RewriteAction> {
        if let ENode::Op { op, children } = node {
            if op.name() == "dx" {
                let arg = children[0];
                // DX(X) → 1
                if is_coordinate_var(egraph, arg, CoordVar::X) {
                    return Some(RewriteAction::Create(ENode::constant(1.0)));
                }
                // DX(const) → 0
                if egraph.is_constant(arg) {
                    return Some(RewriteAction::Create(ENode::constant(0.0)));
                }
            }
        }
        None
    }
}
```

---

### 6. 🔵 RESEARCH: CSE Across Kernel Boundaries

**Severity:** Low
**Effort:** High
**Impact:** Medium (enables kernel fusion)

#### Problem

Multiple kernels that share subexpressions don't get merged:

```rust
let circle1 = kernel!(|cx: f32, cy: f32, r: f32| {
    let dx = X - cx;
    let dy = Y - cy;
    (dx*dx + dy*dy).sqrt() - r
});

let circle2 = kernel!(|cx: f32, cy: f32, r: f32| {
    let dx = X - cx;  // ← Duplicated!
    let dy = Y - cy;  // ← Duplicated!
    (dx*dx + dy*dy).sqrt() - r
});
```

**Opportunity:** Global CSE could identify common subexpressions across kernels and extract them.

#### Challenges

1. **Scope:** Requires whole-program analysis (beyond macro scope)
2. **Coordination:** Multiple `kernel!` invocations are independent
3. **Build system integration:** Needs build.rs or proc-macro cooperation

**Not recommended** for immediate implementation (research project).

---

### 7. 🔵 RESEARCH: Runtime E-graph Optimization API

**Severity:** Low
**Effort:** High
**Impact:** Low (advanced feature)

#### Problem

E-graph optimization only happens at compile-time in macro. Runtime expressions can't be optimized.

**Opportunity:** Expose e-graph API for runtime use:

```rust
// Hypothetical API
let expr = ir::Expr::Binary(
    ir::OpKind::Add,
    Box::new(ir::Expr::Var(0)),
    Box::new(ir::Expr::Const(0.0)),
);

let optimized = pixelflow_search::optimize(expr);  // ← x + 0 → x

// Compile to Manifold
let kernel: Box<dyn Manifold> = compile_to_manifold(optimized);
```

**Use Cases:**
- REPL: `pixelflow repl`
- JIT compilation: optimize expressions at runtime
- Dynamic shader compilation
- ML-driven kernel generation

**Challenges:**
- Trait objects (`Box<dyn Manifold>`) vs type-level AST
- Runtime monomorphization (requires JIT)
- Increased binary size

**Not recommended** for immediate implementation (research/future work).

---

## Recommendations Summary

| Finding | Priority | Effort | Impact | Recommendation |
|---------|----------|--------|--------|----------------|
| 1. IR not integrated | 🔴 Critical | High | High | Fix in v2.0 (breaking) |
| 2. Cost model calibration | 🟡 High | **Low** | **High** | **Implement immediately** |
| 3. Dead code elimination | 🟡 Medium | Low | Medium | Add to optimizer |
| 4. Type ↔ IR bridge | 🟢 Low | Medium | Low | Nice-to-have |
| 5. Derivative rules | 🟢 Low | Medium | Low | Specialized use case |
| 6. Cross-kernel CSE | 🔵 Research | High | Medium | Future work |
| 7. Runtime E-graph API | 🔵 Research | High | Low | Future work |

### Immediate Action Items

**Fix #2 (Cost Model Calibration)** has the best ROI:
- ✅ Low effort (infrastructure exists)
- ✅ High impact (better optimization decisions)
- ✅ Non-breaking change
- ✅ Measurable improvement

**Implementation Plan:**
1. Create `pixelflow-search/src/bin/calibrate_costs.rs`
2. Hook into existing `pixelflow-core/benches/core_benches.rs`
3. Generate `.config/pixelflow/cost_model.toml`
4. Update CI to run calibration on each platform (AVX-512, SSE2, NEON)
5. Document usage in README

---

## Consistency Checks

### ✅ Good Practices Found

1. **Two-pass optimization:** Structural + global is the right architecture
2. **E-graph saturation:** Discovers all equivalent forms correctly
3. **Cost-based extraction:** Allows platform-specific tuning
4. **Depth penalty:** Prevents compile-time blowup (type nesting limit)
5. **Opaque expression handling:** Preserves structure when optimization isn't safe

### ⚠️ Minor Inconsistencies

1. **Cost model defaults:** `mul_add: 10` is wrong for AVX-512 (should be 5)
2. **No platform detection:** Same costs used for AVX-512, SSE2, NEON, scalar
3. **Hardcoded operation list:** `cost_by_name()` has incomplete coverage
4. **No benchmark validation:** Can't prove optimizations actually improve performance

---

## Architecture Diagrams

### Current Pipeline (Simplified)

```
┌────────────────────────────────────────────────────────────┐
│ pixelflow-compiler                                           │
│                                                            │
│  Source AST ─→ EGraph ─→ Optimized AST ─→ Type-Level Code │
│       ↑          ↑                                         │
│       │          │                                         │
│    parser.rs  optimize.rs                                  │
└────────────────────────────────────────────────────────────┘
                     ↑
                     │ (uses)
                     │
┌────────────────────────────────────────────────────────────┐
│ pixelflow-search                                           │
│                                                            │
│  EGraph + Rewrite Rules + Cost Model                       │
│                      ↑                                     │
│                      │ hardcoded costs!                    │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ pixelflow-ir                                               │
│                                                            │
│  Expr (UNUSED!)                                            │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ pixelflow-core/benches                                     │
│                                                            │
│  Operation benchmarks (DISCONNECTED from cost model!)      │
└────────────────────────────────────────────────────────────┘
```

### Proposed Pipeline (With Fixes)

```
┌────────────────────────────────────────────────────────────┐
│ pixelflow-compiler                                           │
│                                                            │
│  Source AST ─→ IR ─→ EGraph ─→ Opt IR ─→ Type-Level Code  │
│       ↑        ↑                   ↑                       │
│       │        │                   │                       │
│    parser  ir_bridge.rs        codegen.rs                  │
└────────────────────────────────────────────────────────────┘
                   │                 │
                   ↓                 ↓
┌─────────────────────────────────────────────────────────────┐
│ pixelflow-ir (INTEGRATED!)                                  │
│                                                             │
│  Canonical IR Expr                                          │
└─────────────────────────────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ pixelflow-search                                            │
│                                                             │
│  EGraph + Rewrite Rules + Cost Model                        │
│                      ↑                                      │
│                      │ learned costs!                       │
│                      │                                      │
│          ┌───────────┴───────────┐                          │
│          ↓                       ↓                          │
│  ~/.config/pixelflow/    calibrate_costs binary            │
│    cost_model.toml       (NEW!)                            │
│          ↑                       ↑                          │
└──────────┼───────────────────────┼──────────────────────────┘
           │                       │
           │                       │
           │                       ↓
┌──────────┴───────────────────────────────────────────────────┐
│ pixelflow-core/benches                                       │
│                                                              │
│  Operation benchmarks → cost calibration (INTEGRATED!)       │
└──────────────────────────────────────────────────────────────┘
```

---

## Implementation Note: Fixes Implemented

This analysis document is accompanied by implementations of:

### ✅ Fix #2: Cost Model Calibration (COMPLETED)
- `pixelflow-core/src/bin/calibrate_costs.rs` — Benchmark harness
- Measures actual SIMD operation latencies
- Automatic TOML generation in `~/.config/pixelflow/cost_model.toml`
- 100x cost scaling for sub-nanosecond precision

### 🚧 Fix #1: IR Integration (IN PROGRESS)
**Goal:** Make `pixelflow-ir::Expr` the canonical representation in the macro compiler.

**Correct Architecture:**
```
┌─────────────────────────────────────┐
│ pixelflow-search (GENERIC)          │
│ - E-graph framework                 │
│ - NNUE learning                     │
│ - Works with any language via ENode │
└─────────────────────────────────────┘
         ↑
         │ (used by)
         │
┌─────────────────────────────────────┐
│ pixelflow-ir (LANGUAGE)             │
│ - Expr definition                   │
│ - OpKind enum                       │
│ - No optimization logic             │
└─────────────────────────────────────┘
         ↑
         │ (depends on both)
         │
┌─────────────────────────────────────┐
│ pixelflow-compiler (FRONTEND)        │
│ - Compiles to IR using E-graph      │
│ - AST → IR → EGraph → IR → Code    │
└─────────────────────────────────────┘
```

**Current Problem:**
- `pixelflow-compiler` has its own 263-line AST
- Never uses `pixelflow-ir::Expr`
- Directly converts AST → ENode, losing the IR abstraction

**Proposed Pipeline:**
```
Before:
  Parse → AST → ENode → ENode → AST → Codegen
                ↑_______________↑
                (optimization)

After:
  Parse → AST → IR → ENode → IR → Codegen
                     ↑______↑
                  (optimization)
```

**Implementation:**
1. Add `pixelflow-ir` dependency to `pixelflow-compiler`
2. Create `ast_to_ir()` converter
3. Create `ir_to_enode()` flattener (tree → graph)
4. Create `enode_to_ir()` extractor (graph → tree)
5. Create `ir_to_code()` generator
6. Update `optimize()` to use IR pipeline

**Status:** Refactoring in progress.

---

## Conclusion

The PixelFlow compiler pipeline is architecturally sound but has **two major gaps**:

1. **IR crate not integrated** — Foundational architectural issue
2. **Cost model not calibrated** — Easy fix with high impact ← **FIXED**

The immediate fix (cost calibration) improves optimization quality without breaking changes. The IR integration should be considered for v2.0 as a breaking architectural improvement.

---

**Analysis completed:** 2026-01-30
**Total findings:** 7 (1 critical, 2 high, 2 medium, 2 research)
**Implemented:** Fix #2 (Cost Model Calibration)
