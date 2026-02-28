## 2025-12-28 - Inherent Methods Shadowing Optimized Trait Implementations
**Learning:** I discovered that `Jet3` and `Jet2` had optimized `sqrt` implementations in their `Numeric` trait implementation, but their *inherent* `sqrt` methods (which shadow the trait methods when called directly) were still using the unoptimized slow path.
**Action:** Always check both trait implementations AND inherent methods when optimizing types in Rust, as inherent methods take precedence and might be legacy/unoptimized code.

## 2025-12-28 - Optimized Keybinding Lookups
**Learning:** `KeybindingsConfig` stored bindings as a `Vec`, causing O(n) lookup overhead on every keypress.
**Action:** Refactored `KeybindingsConfig` to maintain a `HashMap` for O(1) lookups while keeping the `Vec` for serialization/deserialization compatibility using `#[serde(from/into)]`. This ensures performance without breaking config file format.

## 2025-12-28 - AST Optimization and Parentheses
**Learning:** In the `pixelflow-macros` AST optimizer, I initially removed `Expr::Paren` wrappers assuming they were redundant during optimization recursion. This broke operator precedence (e.g., `(X - offset).abs()` became logic that failed tests).
**Action:** When implementing AST transformations, always preserve grouping/parentheses nodes unless you perform a specific precedence check proving they are redundant.

## 2025-12-28 - Rasterizer Inner Loop Hoisting
**Learning:** The inner loop of `execute_stripe` was re-evaluating `Field::sequential(start)` on every iteration, which involves multiple SIMD instructions (broadcast/load + add).
**Action:** Hoisted the initialization of `xs` out of the loop and updated it incrementally using a pre-computed `step` vector. This reduced the inner loop overhead significantly, yielding a ~34% improvement in rasterization throughput.

## 2025-12-28 - Solid Color Evaluation Bypass
**Learning:** In `pixelflow-graphics/src/render/color.rs`, the `Manifold` implementation for `NamedColor` and `Color` used `ColorCube` and `eval_raw` with floating-point math, leading to redundant calculations and poor throughput. We can bypass all AST-building and runtime floating-point overhead by using `pixelflow_core::Discrete::from(u32::from(*self))` which splats the pre-packed O(1) `PALETTE` `u32` onto the SIMD registers.
**Action:** When evaluating discrete types, look for opportunities to compute constant values once per process or block rather than letting the AST recursively perform operations that resolve to scalars.

## 2025-12-28 - SIMD Backend U32 Alias
**Learning:** In `pixelflow-core/src/lib.rs`, when trying to implement SIMD splatting for `u32` types like `Discrete`, the correct alias for the target backend `u32` type is `NativeU32Simd` (e.g., `<backend::x86::Avx512 as Backend>::U32`), not `NativeSimd_u32` as previously guessed.
**Action:** Always review the `#cfg` declarations at the top of the file mapping the target architecture to its specific `NativeSimd` and `NativeU32Simd` type before invoking backend-specific trait methods like `splat`.
