//! exp/log kernels through the JIT, checked numerically at this build's width.
//!
//! These lower (`expand_transcendentals`) to the integer-domain primitives
//! TruncToInt/IntToFloat/IAdd/Shl/Shr — ops a backend can silently lack while
//! every other test stays green, because `Field`'s combinator tier computes
//! its own log2/exp via native intrinsics and never consults the JIT. That is
//! exactly how Avx512Backend shipped refusing `ShiftImm`: `test_log2` passed
//! (combinator tier), the full workspace passed under `+avx512f`, and a
//! `Kernel::log2` still failed to compile. The completeness sweep catches the
//! emission gap; this catches the semantics — the bytes must also be right.

use pixelflow_ir::{Kernel, jit_cache};

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn call_x(jit: &pixelflow_ir::JitManifold, x: f32) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let v = _mm512_set1_ps(x);
        let z = _mm512_setzero_ps();
        _mm512_cvtss_f32(jit.call(v, z, z, z))
    }
}
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
fn call_x(jit: &pixelflow_ir::JitManifold, x: f32) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let v = _mm256_set1_ps(x);
        let z = _mm256_setzero_ps();
        _mm256_cvtss_f32(jit.call(v, z, z, z))
    }
}
#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx2"),
    not(target_feature = "avx512f")
))]
fn call_x(jit: &pixelflow_ir::JitManifold, x: f32) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let v = _mm_set1_ps(x);
        let z = _mm_setzero_ps();
        _mm_cvtss_f32(jit.call(v, z, z, z))
    }
}
#[cfg(target_arch = "aarch64")]
fn call_x(jit: &pixelflow_ir::JitManifold, x: f32) -> f32 {
    use core::arch::aarch64::*;
    unsafe {
        let v = vdupq_n_f32(x);
        let z = vdupq_n_f32(0.0);
        vgetq_lane_f32::<0>(jit.call(v, z, z, z))
    }
}

/// Two-operand form of [`call_x`]: X = `x`, Y = `y`, lane 0 of the result.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn call_xy(jit: &pixelflow_ir::JitManifold, x: f32, y: f32) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let z = _mm512_setzero_ps();
        _mm512_cvtss_f32(jit.call(_mm512_set1_ps(x), _mm512_set1_ps(y), z, z))
    }
}
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
fn call_xy(jit: &pixelflow_ir::JitManifold, x: f32, y: f32) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let z = _mm256_setzero_ps();
        _mm256_cvtss_f32(jit.call(_mm256_set1_ps(x), _mm256_set1_ps(y), z, z))
    }
}
#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx2"),
    not(target_feature = "avx512f")
))]
fn call_xy(jit: &pixelflow_ir::JitManifold, x: f32, y: f32) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let z = _mm_setzero_ps();
        _mm_cvtss_f32(jit.call(_mm_set1_ps(x), _mm_set1_ps(y), z, z))
    }
}
#[cfg(target_arch = "aarch64")]
fn call_xy(jit: &pixelflow_ir::JitManifold, x: f32, y: f32) -> f32 {
    use core::arch::aarch64::*;
    unsafe {
        let z = vdupq_n_f32(0.0);
        vgetq_lane_f32::<0>(jit.call(vdupq_n_f32(x), vdupq_n_f32(y), z, z))
    }
}

fn check(name: &str, k: &Kernel, inputs: &[f32], reference: impl Fn(f32) -> f32) {
    let (arena, root) = k.parts();
    let jit = jit_cache::compile_cached(arena, root)
        .unwrap_or_else(|e| panic!("{name}: kernel failed to compile on this backend: {e}"));
    for &x in inputs {
        let got = call_x(&jit, x);
        let want = reference(x);
        let rel = ((got - want) / want.abs().max(1e-6)).abs();
        assert!(
            rel <= 1e-3 || (got - want).abs() <= 1e-4,
            "{name}({x}): got {got}, want {want} (rel {rel:.2e})"
        );
    }
}

#[test]
fn exp_log_kernels_compile_and_agree_with_std() {
    // Log domains can range wide; exp inputs stay where e^x is a finite f32.
    // The expansion clamps its exponent to ±126 by design (expand_exp2,
    // lowering.rs) — the JIT saturates where std overflows to inf, so inputs
    // past ~88.7 (ln(f32::MAX)) compare a clamp against an inf, which is a
    // contract difference, not an encoding bug.
    let logs = [0.5f32, 1.0, 2.0, 8.0, 100.0, 4096.0];
    let exps = [-10.0f32, -1.0, 0.0, 0.5, 1.0, 8.0, 80.0];
    check("log2", &Kernel::x().log2(), &logs, f32::log2);
    check("ln", &Kernel::x().ln(), &logs, f32::ln);
    check("exp", &Kernel::x().exp(), &exps, f32::exp);
    check("exp2", &Kernel::x().exp2(), &exps, f32::exp2);
}

// ── Floating-point edge cases: the documented contract, not IEEE ─────────────
//
// The language's FP contract (CLAUDE.md, "Floating point at the edges") is
// *hardware semantics*, deliberately not scalar Rust's: NaN comparisons,
// Min/Max NaN handling and Round's tie-breaking follow whatever the SIMD
// instruction does, because matching `f32::min`/`f32::round` would cost extra
// instructions in the hot path for cases the language does not promise.
//
// What the contract DOES require is that the two tiers agree: the JIT and
// `OpKind::eval_*` (the differential oracle, and the e-graph's constant
// folder) must produce the same answer, or a folded expression differs from
// the same expression computed at runtime. These tests pin that agreement at
// whatever width the build selects — they are the reason the oracle was
// changed to match the hardware rather than the encoders changed to match
// scalar Rust.

/// Every lane of a binary kernel, JIT vs oracle, on edge-case inputs.
fn assert_tiers_agree_binary(name: &str, k: &Kernel, op: pixelflow_ir::OpKind) {
    let (arena, root) = k.parts();
    let jit = jit_cache::compile_cached(arena, root).unwrap_or_else(|e| panic!("{name}: {e}"));
    let nan = f32::NAN;
    for (x, y) in [
        (1.0f32, nan),
        (nan, 1.0f32),
        (nan, nan),
        (2.5, 2.5),
        (-0.0, 0.0),
        (1.0, 2.0),
        (2.0, 1.0),
        (f32::INFINITY, 1.0),
    ] {
        // Where the two hardwares disagree, the language promises nothing, so
        // there is no answer to assert — only that the folder declines to pick
        // one (checked by `folder_declines_platform_specific_nan_cases`).
        // x86 `minps` yields the second operand while ARM `FMIN` propagates the
        // NaN; x86's Gt/Ge are unordered while ARM's are ordered. CI caught
        // this on macos-latest: `min(NaN, 1)` gave NaN on aarch64 against an
        // oracle written from x86's behavior.
        if op.fold_is_platform_specific(&[x, y]) {
            continue;
        }
        let got = call_xy(&jit, x, y);
        let want = op.eval_binary(x, y).expect("oracle covers this op");
        // Bit-exact, not `==`: `-0.0 == 0.0` would hide a signed-zero
        // disagreement, which is observable downstream as `1.0/x` = ∓inf.
        assert!(
            got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
            "{name}({x}, {y}): JIT gave {got}, oracle gave {want} — tiers must agree"
        );
    }
}

#[test]
fn min_max_nan_handling_agrees_between_tiers() {
    use pixelflow_ir::OpKind;
    assert_tiers_agree_binary("min", &Kernel::x().min(&Kernel::y()), OpKind::Min);
    assert_tiers_agree_binary("max", &Kernel::x().max(&Kernel::y()), OpKind::Max);
}

/// Gt/Ge are the unordered predicates (true for NaN); Lt/Le are ordered. The
/// asymmetry is the hardware's, and the oracle records it.
///
/// Compared bit-for-bit against the raw comparison — no `select` wrapper and no
/// NaN escape hatch — because the *bit pattern* is the contract, not merely
/// which branch it would pick. An all-ones lane is a NaN, so a NaN-tolerant
/// comparison would accept `f32::NAN` where the hardware writes `0xFFFFFFFF`
/// and both would still "agree".
#[test]
fn nan_comparisons_agree_between_tiers() {
    use pixelflow_ir::OpKind;
    let nan = f32::NAN;
    for (name, op, k) in [
        ("gt", OpKind::Gt, Kernel::x().gt(&Kernel::y())),
        ("ge", OpKind::Ge, Kernel::x().ge(&Kernel::y())),
        ("lt", OpKind::Lt, Kernel::x().lt(&Kernel::y())),
        ("le", OpKind::Le, Kernel::x().le(&Kernel::y())),
    ] {
        let (arena, root) = k.parts();
        let jit = jit_cache::compile_cached(arena, root).unwrap_or_else(|e| panic!("{name}: {e}"));
        for (x, y) in [
            (1.0f32, nan),
            (nan, 1.0f32),
            (nan, nan),
            (1.0, 2.0),
            (2.0, 1.0),
            (2.5, 2.5),
            (-0.0, 0.0),
        ] {
            if op.fold_is_platform_specific(&[x, y]) {
                continue;
            }
            let got = call_xy(&jit, x, y);
            let want = op.eval_binary(x, y).expect("oracle covers this op");
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{name}({x}, {y}): JIT wrote {:#010x}, oracle {:#010x} — a mask \
                 lane is all-ones or all-zero in both tiers or it is not a mask",
                got.to_bits(),
                want.to_bits()
            );
        }
    }
}

/// A folded comparison must be substitutable for an executed one.
///
/// `Select` and `BitAnd` are bitwise on every backend, so a mask lane that is
/// merely *truthy* corrupts them: with `1.0` (`0x3f800000`) standing in for
/// true, `runtime_mask & 1.0` is `0x3f800000`, and blending `7.0` against `9.0`
/// through that pattern yields `4.5` — a value neither branch held. This is the
/// end-to-end shape: one operand of the `&` is a constant the folder produced,
/// the other is computed at runtime.
#[test]
fn a_folded_mask_blends_like_a_computed_one() {
    use pixelflow_ir::OpKind;

    // The folder's answer for `0.5 != nextafter(0.5)` — true, and (since the
    // epsilon fix) reachable for ordinary finite values.
    let near = f32::from_bits(0.5f32.to_bits() + 1);
    let folded = OpKind::Ne
        .eval_binary(0.5, near)
        .expect("oracle covers Ne");

    let k = Kernel::x()
        .gt(&Kernel::constant(0.0))
        .and(&Kernel::constant(folded))
        .select(&Kernel::constant(7.0), &Kernel::constant(9.0));
    let (arena, root) = k.parts();
    let jit = jit_cache::compile_cached(arena, root).expect("mask kernel compiles");

    assert_eq!(call_x(&jit, 1.0), 7.0, "mask true must select if_true exactly");
    assert_eq!(call_x(&jit, -1.0), 9.0, "mask false must select if_false exactly");
}

#[test]
fn round_agrees_between_tiers_away_from_ties() {
    use pixelflow_ir::OpKind;
    let rounded = Kernel::x().round();
    let (arena, root) = rounded.parts();
    let jit = jit_cache::compile_cached(arena, root).expect("round compiles");
    // Non-ties only. At a tie the three tiers disagree by design (x86
    // nearest-even, aarch64 FRINTA ties-away, combinator `(x+0.5).floor()`), so
    // there is no answer to assert — see `tie_result_is_platform_specific`.
    assert!(OpKind::Round.fold_is_platform_specific(&[2.5]));
    for x in [2.4f32, 2.6, -2.4, -2.6, 0.2, 7.0, -7.0] {
        let got = call_x(&jit, x);
        let want = OpKind::Round.eval_unary(x).expect("oracle covers Round");
        assert_eq!(got, want, "round({x}): JIT {got} vs oracle {want}");
    }
}

/// Which ops carry the platform-specific marker. The marker's *effect* on
/// constant folding is covered where the folder lives
/// (`pixelflow-search::math::algebra`, `platform_specific_fold_tests`) — this
/// only pins the classification, which is what the emitters and the contract
/// table in CLAUDE.md are written against.
#[test]
fn platform_specific_ops_are_classified() {
    use pixelflow_ir::OpKind;
    for op in [OpKind::Min, OpKind::Max, OpKind::Gt, OpKind::Ge] {
        assert!(
            op.fold_is_platform_specific(&[f32::NAN, 1.0]),
            "{op:?} diverges between x86 and aarch64 on NaN"
        );
    }
    // The ones both hardwares agree on stay foldable.
    for op in [OpKind::Lt, OpKind::Le, OpKind::Eq, OpKind::Ne, OpKind::Add] {
        assert!(
            !op.fold_is_platform_specific(&[f32::NAN, 1.0]),
            "{op:?} agrees across targets and should remain foldable"
        );
    }

    // Estimate instructions: no argument is safe, because an estimate is only
    // guaranteed close and the three targets are not close to each other.
    for op in [OpKind::Recip, OpKind::Rsqrt] {
        for x in [2.0f32, 3.0, 0.5, 1.0] {
            assert!(
                op.fold_is_platform_specific(&[x]),
                "{op:?}({x}) is an estimate and must never fold"
            );
        }
    }

    // MulAdd is value-aware: one rounding or two, and most inputs cannot tell.
    let fused_differs = [1.000_000_1f32, 4097.0, 4097.0];
    assert!(
        OpKind::MulAdd.fold_is_platform_specific(&fused_differs),
        "an input where FMA and mul-then-add disagree must not fold"
    );
    assert!(
        !OpKind::MulAdd.fold_is_platform_specific(&[2.0, 3.0, 4.0]),
        "exact inputs are the same either way and stay foldable"
    );
}

/// `Eq`/`Ne` are exact in every emitter, so the oracle must be too. The old
/// `(x-y).abs() < f32::EPSILON` form folded 0.5 and its next representable
/// neighbour to "equal" while the hardware said "not equal" — a fold-vs-runtime
/// miscompile on ordinary finite values, with no NaN or platform difference
/// involved.
#[test]
fn eq_ne_are_exact_not_epsilon() {
    use pixelflow_ir::OpKind;

    let near = f32::from_bits(0.5f32.to_bits() + 1);
    assert_ne!(0.5f32, near, "the two values really are distinct f32s");
    assert!((0.5f32 - near).abs() < f32::EPSILON, "and closer than EPSILON");

    let t = OpKind::mask(true);
    let f = OpKind::mask(false);
    assert_eq!(OpKind::Eq.eval_binary(0.5, near).map(f32::to_bits), Some(f.to_bits()));
    assert_eq!(OpKind::Ne.eval_binary(0.5, near).map(f32::to_bits), Some(t.to_bits()));
    assert_eq!(OpKind::Eq.eval_binary(0.5, 0.5).map(f32::to_bits), Some(t.to_bits()));

    // NaN: never equal, always unequal — agreed by every emitter.
    let nan = f32::NAN;
    assert_eq!(OpKind::Eq.eval_binary(nan, nan).map(f32::to_bits), Some(f.to_bits()));
    assert_eq!(OpKind::Ne.eval_binary(nan, nan).map(f32::to_bits), Some(t.to_bits()));

    // `Kernel` exposes no eq/ne constructor, so there is no JIT side to compare
    // against here — the folder IS the consumer that was wrong, and these
    // assertions are what pins it.
}
