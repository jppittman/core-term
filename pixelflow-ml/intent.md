# pixelflow-ml

**Is.** The bases whose inner product has a closed form: the Hilbert space on
kernels beside sums over discrete kernels (DENOTATION §3, "a matrix is a
discrete kernel"). A spherical-harmonic feature map is a `Kernel` transformer,
so it composes into a larger kernel and reaches numbers the one way anything
does, compiled at a lattice's shape and collapsed. An experiment, not a layer
of the renderer.

## Owns
- The spherical-harmonic data: `SH_NORM`, `ShCoeffs<N>` with its
  `[l*l + l + m]` index, `Sh2`. Scalar data, here because this crate is their
  only consumer.
- The feature maps (`FeatureMap`, `EluFeature`, `ShFeatureMap::<9>::project`),
  the attention state (`HarmonicAttention`, `LinearAttention`) and the claim
  that linear attention is harmonic global illumination.

## Never
- No matmul, GEMM or tiled loop. A linear operator is a kernel of kernels;
  whether a contraction becomes a GEMM is the backend's factorization of its
  domain (the schedule note §3).
- No integral: `ShCoeffs::dot` is a sum over the basis's closed form.
- No lane, width, vector type or `Field`; the only non-test `pixelflow_core` import is
  `Kernel`. No dependency but `pixelflow-core` (optional) and `libm`; no
  graphics, runtime, terminal or compiler-optimization code.
- No heap allocation after construction (`accumulate`, `query`, `reset`).

## Contract
`src/lib.rs` declares `graphics` behind the default feature `graphics`;
without it the crate is empty. `src/graphics.rs` is all of it: `SH_NORM`,
`ShCoeffs`, `Sh2`, `FeatureMap`, `EluFeature`, `RandomFourierFeature`,
`HarmonicAttention`, `ShFeatureMap`, `LinearAttention`,
`HarmonicAttentionIsGlobalIllumination`. A new feature map is another
`impl FeatureMap`.

## Enforced by
- Types: `FeatureMap::apply` takes and returns `&Kernel`, so a feature map
  holds no lane; `Cargo.toml` lists `pixelflow-core` and `libm` only.
- Tests in `src/graphics.rs`: `elu_feature_positive` and
  `sh_feature_map_projects_direction` compare compiled kernels to scalar truth.
- `scripts/check-feature-matrix.sh` (CI job `feature-matrix`): the crate
  checks with `graphics` off and on.
- Nothing enforces the no-matmul, no-integral or no-allocation lines.

## Behind
- Attention is scalar: `HarmonicAttention::accumulate` and `query` are `f32`
  loops over `Vec`s, not kernels; the schedule note §3 (`Proposed`) has a
  feature as a discrete kernel and an operator as a kernel of kernels.
- `LinearAttention` has no `accumulate` or `query`; `RandomFourierFeature`
  does not implement `FeatureMap`.
- Zero dependents; a workspace member only because D4 of
  `docs/archive/plans/2026-09-01-dead-code-with-ideas.md` is open (2026-09-01).
- `libm` and `criterion` are declared and unused; the `Cargo.toml`
  description still says "compiler optimization"; no `#![no_std]`.

## Decided in
`docs/DENOTATION.md` §3, `docs/codegen/2026-10-15-a-schedule-is-a-factorization.md`
§3, `docs/archive/plans/2026-09-06-kernel-with-a-lattice.md` (SH data moved here),
`docs/archive/plans/2026-09-01-dead-code-with-ideas.md` (D4, R4).
