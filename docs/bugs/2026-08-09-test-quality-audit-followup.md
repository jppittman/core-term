# Test quality control follow-up — 2026-08-09

Scope: scheduled continuation, picking up recommendation #3 from
`docs/bugs/2026-08-08-test-quality-audit-followup.md` — the `pixelflow-core`
half of it (`pixelflow-core/src/backend/mod.rs`, 327 lines), which had "never
been mutation-tested under its new crate location" since the `pixelflow-ir` →
`pixelflow-core` move on 2026-08-02.

No new commits landed on this branch's base since the last audit (`58885c5`),
so there was no fresh delta to static-audit against STYLE.md's "Test Public
API" rule or the "it should ..." naming convention; this pass is scoped
entirely to the mutation-testing backlog item instead.

## Mutation testing: `pixelflow-core/src/backend/mod.rs`

`cargo-mutants` v27.1.0 (freshly installed — not present in this environment,
consistent with every prior pass), scoped to this one file
(`cargo mutants -p pixelflow-core -f pixelflow-core/src/backend/mod.rs`).

This file is almost entirely `SimdOps`'s **provided** (default-body) trait
methods — the required per-ISA primitives have no body here to mutate. Before
this pass, none of the provided trig/log/power/rounding methods (`exp`,
`tan`, `atan`, `asin`, `acos`, `ln`, `log10`, `pow`, `hypot`, `mul_rsqrt`,
`ceil`, `round`, `fract`, `clamp`) had ever been exercised by a test that
asserted on their numeric output: **45 of 46 mutants survived** (1 unviable).
The doc comment at the top of the file says this trait "backs the combinator
evaluation path that `kernel!` still produces by default" — i.e. this is not
dead scaffolding, it is the interpreter `.eval()` goes through whenever a
kernel isn't JIT-compiled, so the gap was real exposure, not a false
positive.

Added 14 tests to `pixelflow-core/tests/unit_tests.rs`
(`transcendental_ops_tests` module), each going through the public
`Manifold`/`ManifoldExt` combinator API exactly like the file's existing
`unary_operators_should_compute_correctly` test — `X.tan().eval(coords)` and
so on, asserted with `assert_field_approx_eq` — never touching
`SimdOps`/`Field` internals directly:

`exp_of_one_equals_eulers_number`, `tan_of_pi_over_four_equals_one`,
`atan_of_one_equals_pi_over_four`, `asin_of_one_half_equals_pi_over_six`,
`acos_of_one_half_equals_pi_over_three`, `ln_of_two_matches_std_ln`,
`log10_of_ten_equals_one`, `pow_raises_base_to_a_fractional_exponent`,
`hypot_of_three_and_four_equals_five`,
`mul_rsqrt_divides_by_the_square_root_of_the_second_argument`,
`ceil_rounds_a_fractional_value_up_to_the_next_integer`,
`round_rounds_a_fractional_value_to_the_nearest_integer`,
`fract_returns_the_fractional_part_of_a_value`,
`clamp_restricts_a_value_to_the_given_range`.

Input values were chosen specifically to distinguish the arithmetic-operator
mutations `cargo-mutants` generates (e.g. `pow`'s test uses base `4.0`,
exponent `0.5` rather than a base with `log2(base) == 1`, since `* ` vs `/`
of the exponent by `log2(base)` produce the same answer when
`log2(base) == 1` and the mutant would silently survive).

Re-running mutants after the fix: **36 of 46 caught** (up from 0), 1
unviable, **9 still missed** — all in `ln`, `pow`, and `mul_rsqrt`.

### The 9 remaining misses are dead code, not a test gap

Traced each of the three functions' actual callers and found none: `Field`
and `NumericOps` (`pixelflow-core/src/lib.rs`) and `MulRsqrt::eval`
(`pixelflow-core/src/ops/binary.rs`) each independently **reimplement the
same formula one layer up** instead of calling through to
`SimdOps::{ln,pow,mul_rsqrt}`:

- `Field::ln` (lib.rs:715): `self.0.log2() * NativeSimd::splat(LN_2)` —
  not `self.0.ln()`.
- `NumericOps::pow` for `Field` (lib.rs:1619-1622):
  `self.0.log2() * exp.0` then `.exp2()` — not `self.0.pow(exp.0)`.
- `MulRsqrt::eval` (ops/binary.rs:250-254):
  `self.0.eval(p).raw_mul(self.1.eval(p).rsqrt())` — not
  `self.0.eval(p).mul_rsqrt(...)`.

A workspace-wide `grep -rn "SimdOps"` confirms `backend/mod.rs` is the only
place these three trait methods are referenced at all — no call site
anywhere in the 12-crate workspace reaches them. Added tests for
`Field::ln`/`.pow()`/`.mul_rsqrt()` still land real, previously-absent
coverage (none of those three had a direct test before this pass either),
just not of `SimdOps`'s copies specifically, since nothing ever runs that
code. This is the same shape of finding as the 2026-08-08 audit's
`CostFunction::cost_by_kind` default body: pre-existing, genuinely
unreachable duplicate code, left as-is — deleting it is a production-code
call outside a test-quality pass's remit, and out of scope here.

`cargo test -p pixelflow-core`: 114/114 (unit_tests.rs) + all other suites
green, 0 failures. `cargo clippy -p pixelflow-core --tests`: clean.
`cargo fmt -p pixelflow-core -- --check`: clean.

## Verified

- `cargo test -p pixelflow-core`: all suites pass, 0 failures.
- `cargo test --workspace --lib`: all crates pass, 0 failures.
- `cargo clippy -p pixelflow-core --tests`: clean.
- `cargo fmt -p pixelflow-core -- --check`: clean.
- `cargo mutants -p pixelflow-core -f pixelflow-core/src/backend/mod.rs`:
  36 caught / 1 unviable / 9 missed (was 0/1/45) — remaining 9 are the
  documented dead-code case above.

## Recommended next steps (not done here)

1. **`Field::ln`, `NumericOps::pow`, and `MulRsqrt::eval` duplicate
   `SimdOps::{ln,pow,mul_rsqrt}`'s formulas instead of calling them.** Not a
   test-quality issue by itself (both copies are correct and now both are
   exercised), but it is exactly the kind of drift risk `SimdOps`'s own doc
   comment warns about implicitly: two independent implementations of the
   same math that could silently diverge if one is edited and the other
   isn't. Worth a design call on whether `backend/mod.rs`'s versions should
   be deleted (its doc-comment claim that it "backs the combinator
   evaluation path" is only true for the methods that don't have a
   duplicate) or whether the higher layer should call through instead.
2. `pixelflow-codegen/src/emit/*` (the other half of 2026-08-08's
   recommendation #3, ~7,200 lines across `mod.rs`/`aarch64.rs`/`avx2.rs`/
   `avx512.rs`/`x86_64.rs`/`regalloc.rs`/`executable.rs`) still has never
   been mutation-tested under its new crate location — not attempted here,
   `mod.rs` alone is 5,745 lines and needs a narrower scoped follow-up (by
   function or sub-module) rather than a whole-file sweep in one pass.
3. `pixelflow-search`'s `cost.rs` and `pixelflow-ir/src/kind.rs`'s
   pre-existing backlog (both flagged 2026-08-08) are still open — same
   reasoning as that doc: `pixelflow-search --lib` is too slow (110s+
   baseline) for a full sweep in one scheduled pass.
