## 2024-02-24 - CI Fixes: Clippy, Compilation, and Feature Flags
**Learning:**
1. `pixelflow-graphics`: Closures cannot be passed to `.map()` on manifolds; use composed expressions like `X - 0.5`.
2. `pixelflow-nnue`: Tests depending on `Expr::eval` fail unless `pixelflow-ir/std` feature is explicitly enabled in `Cargo.toml`.
3. `pixelflow-ir`: Clippy `excessive_precision` on polynomial coefficients requires truncation (e.g., `1.41421356` -> `1.414_213_5`), and `approx_constant` requires using `core::f32::consts`.
**Action:** Always check feature flags when dependencies' test helpers are missing. Use explicit constants for math values.
