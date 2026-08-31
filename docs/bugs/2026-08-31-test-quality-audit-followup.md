# Test quality control follow-up — 2026-08-31

Scope: scheduled continuation of the test-quality-audit series
(`docs/bugs/2026-08-26-test-quality-audit-followup.md` and earlier).
`main` had not moved on that pass's open backlog since `9fce6cf`. Backlog
item 3 (the actor-scheduler `backoff_unit_tests` finding) was recommended
for removal last pass as already resolved twice over — this pass drops it
and instead picks up backlog item 2: `pixelflow-codegen/src/emit/*`,
flagged as never mutation-tested since the 2026-08-08 audit first noted it.
Of that directory, `x86_64.rs` (the x86-64 SSE/VEX raw instruction
encoder, 864 lines) stood out for having exactly one `#[test]` — a
selection-completeness check — against a file whose whole job is byte-level
machine code emission.

`cargo-mutants` v27.1.0 (freshly installed — not present in this
environment, consistent with every prior pass).

## `pixelflow-codegen/src/emit/x86_64.rs`

### Dead code found and removed

`emit_xorps` and `emit_andps` (legacy 2-operand `XORPS`/`ANDPS`) were
`pub fn`s with zero callers anywhere in the repository — `grep`-confirmed
across every crate, not just this one. `OpKind::Neg`/`Abs` (the two ops
that plausibly wanted them) go through the VEX 3-operand
`emit_vxorps`/`emit_vandps` instead. Per CLAUDE.md's "subtract before you
add," the right fix for untested dead code is deletion, not a test that
props it up. Removed both (2 fewer mutants in every count below).

### First sweep

`cargo mutants -p pixelflow-codegen --file pixelflow-codegen/src/emit/x86_64.rs -j 4 -- --lib`:
**335 mutants, 118 missed, 216 caught, 1 unviable.** The file's only
existing test checks that every `REQUIRED_BINARY_OPS` op selects an
encoding; nothing pinned the actual byte sequences any emitter produces.
The 118 misses spanned nearly every function in the file — `emit_ternary`
and `emit_epilogue` had their whole bodies replaceable with `()` and
nothing noticed, because the driver-level tests in `emit/mod.rs`
(`x86_unary_builtins_match_scalar`, `x86_binary_ternary_builtins_match_scalar`,
etc.) exercise these emitters only through *execution* — compile a kernel,
run it, compare the resulting `f32` against a scalar oracle — using
register/offset choices that happened not to make most bit-manipulation
mutants observable (e.g. always compiling with registers < 8, so the
various `reg.0 >= 8` REX branches never diverge; always landing in one
offset range, so a `<` vs `<=` boundary never got exercised at the edge).

### Fixed: ~35 new tests, byte-exact

Added direct unit tests to `x86_64.rs`'s own `mod tests` for every
emitter with a missed mutant — `emit_movaps_load`/`_store` (offset==0,
just-under-128, and exactly-128 boundary cases, plus a high-register case
for the REX branch), `emit_f32_const` (zero fast path delegating to
`emit_vxorps`; nonzero path's embedded-constant bytes and RIP-relative
displacement arithmetic, both with and without REX), the general `emit_vex`
encoder (exercising `w=1`, high and low `reg`/`rm`), `emit_load_ptr_from_ctx`,
`emit_vpextrd_to_gpr`, `emit_vmovss_load_scaled`, `emit_movups_store_base`
(all four `src`/`base_gpr` REX-bit combinations, which also pins the
`||` in its guard condition), `emit_cmp_tail`, `emit_ternary`'s `MulAdd`
arm (both the dst-differs-from-a and dst-equals-a paths, pinning the `!=`
setup-move guard), `emit_epilogue` (same `!=` pattern for the result
register), and `emit_movmskps_eax`/`emit_cmp_eax_imm8`.

Testing private functions (`emit_vex`, `emit_f32_const`,
`emit_vpextrd_to_gpr`, ...) directly from `mod tests` follows the
2026-08-26 audit's precedent for `spatial_bsp.rs`: `mod tests` is a child
of the module, so it's testing the module's real contract, not reaching
over an API boundary — no `pub(crate)` widening needed anywhere in this
pass.

Since this is raw byte-sequence encoding, "test the public API" per
STYLE.md means asserting exact output bytes for known inputs — the same
approach the file's own precedent (`selects_every_required_binary_op`)
and this crate's `X86BinaryInsn` design (documented in the file's own
comments as splitting "which op" from "which bytes" specifically so each
question is independently checkable) already establish.

Caught one arithmetic mistake of my own in the process: the first draft of
the disp32-boundary tests for `emit_movaps_load`/`_store` used `Reg(0)`,
under which the ModRM reg-field shift (`(dst.0 & 7) << 3`) is `0` whichever
direction it shifts — so a `<<` → `>>` mutant at that exact branch stayed
missed on the first re-sweep. Re-running with a nonzero register (`Reg(3)`/
`Reg(5)`) fixed it; `cargo test` catching my own hand-computed expected
bytes wrong on the first attempt (a hardcoded `0x63` where the real
encoding produces `0xE1`) is what caught that before it shipped.

### Final sweep: 333 mutants, 50 missed, 282 caught, 1 unviable — all 50 documented equivalents

Every one of the 50 remaining misses is `replace | with ^` inside a
ModRM/VEX/REX byte-construction expression of the shape
`BASE | (field << shift)` (or a chain of such terms). This is not a
per-site coincidence: x86 ModRM/REX/VEX bytes are *defined* as fixed-width
bit fields packed side by side (REX bits at 0/2/6, ModRM `reg` at bits
3–5, ModRM `rm` at bits 0–2, VEX `R`/`X`/`B` at bits 5–7, ...) specifically
so they never collide — that disjointness is what makes the encoding
decodable at all. When two operands of a `|` can never share a set bit,
OR and XOR compute the identical byte for *every* possible input; no test,
however cleverly chosen, can tell them apart. I hand-verified the bit
layout at each of the 50 sites (`emit_vex_128_0f`, `emit_sse_rr`,
`emit_movaps_load`/`_store`'s three addressing-mode branches,
`emit_f32_const`'s RIP-relative ModRM byte, `emit_vex`, `emit_load_ptr_from_ctx`,
`emit_vpextrd_to_gpr`, `emit_vmovss_load_scaled`, the four rsp-relative
spill emitters, `emit_movups_store_base`, `emit_cmp_tail`,
`emit_movmskps_eax`) and confirmed disjointness in every case — the same
"OR of disjoint sets is unconditionally equal to XOR of disjoint sets"
argument applies uniformly, so I'm treating this as one documented
equivalence class rather than 50 separate findings (in the same spirit as
the `depth_cost` `>`/`>=` and `Round` `/2.0`/`*2.0` equivalent mutants
documented in the 2026-08-22 and 2026-08-26 passes). A comment at the top
of the new test block records this so a future pass doesn't re-chase them.

## Verified

- `cargo test -p pixelflow-codegen --lib`: 106 passed, 0 failed (was 78
  before this pass).
- `cargo test -p pixelflow-codegen` (all targets incl. doctests): passed,
  0 failed.
- `cargo build --workspace`: clean (confirms `emit_xorps`/`emit_andps`
  removal doesn't break any other crate).
- `cargo clippy -p pixelflow-codegen --lib --tests`: clean.
- `cargo fmt -p pixelflow-codegen -- --check`: clean.
- `cargo mutants -p pixelflow-codegen --file pixelflow-codegen/src/emit/x86_64.rs -- --lib`:
  333 mutants, 282 caught, 50 documented equivalent, 0 real gaps.

## Recommended next steps (not done here)

1. `pixelflow-codegen/src/emit/` has six more files never mutation-tested
   at this granularity: `executable.rs`, `regalloc.rs`, `avx2.rs`,
   `avx512.rs` (each already has some unit tests — 6, 6, 7, 8
   respectively — but unknown mutation coverage), `aarch64.rs` (74K, 30
   tests, the NEON counterpart to this pass's x86_64.rs — a natural next
   target given the same "byte-exact vs. execution-only" gap likely
   exists there too), and `mod.rs` (228K, 48 tests — likely needs
   splitting into several scoped mutants runs rather than one pass, per
   the `cost.rs`/`spatial_bsp.rs` precedent of scoping to a test-name
   filter to dodge slow baselines).
2. `pixelflow-core/src/backend/x86.rs`'s AVX2/AVX-512 (`F32x8`/`F32x16`/
   `U32x8`/`U32x16`/`Mask8`/`Mask16`) impls and `arm.rs`'s NEON impls —
   flagged since 2026-08-26 (and, before that, since whenever the SSE2
   baseline closed) as never unit-tested under a build that actually
   activates those ISA levels. Needs `xtask isa-matrix`-style
   target-feature setup before `cargo mutants` can even run against them.
