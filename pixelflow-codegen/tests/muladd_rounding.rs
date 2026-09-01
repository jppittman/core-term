//! `MulAdd`'s rounding form, asserted through compiled code.
//!
//! CLAUDE.md's platform-divergence table calls `MulAdd` *value-aware*: one
//! rounding where the hardware has an FMA, two where it does not, and the two
//! disagree on inputs like `mul_add(1.0000001, 4097.0, 4097.0)`. That
//! divergence is the entire reason the emitter carries two shapes for one op —
//! `ResolvedOp::FusedMulAdd` and `ResolvedOp::DecomposedMulAdd` — and which
//! one a node gets is decided by register pressure alone.
//!
//! Every other JIT-vs-interpreter test in this crate compares within a
//! tolerance (`spill_pressure`'s 4 ULP, `oracle_reference`'s per-op
//! `Tolerance`), and one-rounding vs. two is a last-bit difference: it fits
//! inside all of them. So a backend that silently emitted the wrong shape —
//! decomposing where it has an FMA, or fusing a decomposition — would keep
//! every one of those tests green. These assert the *bits*, on inputs chosen
//! so the two forms cannot agree.
//!
//! x86-64 only, because it executes: the encodings themselves are pinned for
//! all four backends from any host by `emit::tests::muladd_encoding`.
#![cfg(target_arch = "x86_64")]

use pixelflow_codegen::JitManifold;
use pixelflow_codegen::emit::executable::Point4;
use pixelflow_codegen::emit::{EmitCtx, compile};
use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{ExprArena, ExprId};

// ── The two rounding forms, as scalar references ─────────────────────────────

/// `a*b + c` with **one** rounding — what an FMA instruction computes.
fn fused(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

/// `a*b + c` with **two** roundings — a multiply, rounded, then an add.
///
/// `black_box` is load-bearing: under `+fma` LLVM contracts a plain `a*b + c`
/// into a single `fma` instruction (which is exactly why `eval_scalar`'s
/// oracle agrees with the fused form on those builds), and this function's
/// whole job is to be the answer that contraction destroys.
fn decomposed(a: f32, b: f32, c: f32) -> f32 {
    core::hint::black_box(a * b) + c
}

/// An input where the two forms differ, so an assertion against one of them
/// genuinely rejects the other. `1.0000001 * 4097.0` needs more mantissa bits
/// than an `f32` has, and rounding it before the add loses the bit that the
/// add would otherwise have kept.
const A: f32 = 1.000_000_1;
const B: f32 = 4097.0;
const C: f32 = 4097.0;

/// `A`/`B` halved: `X + X` and `Y + Y` reconstruct them exactly (doubling only
/// touches the exponent), which is how the spilled scenario below gets a
/// divergent product out of operands that are computed values rather than
/// coordinates — a coordinate is precolored into an input register and never
/// spills.
const HALF_A: f32 = 0.500_000_06;
const HALF_B: f32 = 2048.5;

// ── JIT invocation at this build's width ─────────────────────────────────────

fn assert_bits(tag: &str, got: f32, want: f32) {
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "{tag}: JIT {got} ({:#010x}) is not {want} ({:#010x})",
        got.to_bits(),
        want.to_bits()
    );
}

/// The two forms must actually disagree on `A`/`B`/`C`, or every assertion
/// below is vacuous.
#[test]
fn the_reference_forms_disagree_on_these_inputs() {
    assert_ne!(
        fused(A, B, C).to_bits(),
        decomposed(A, B, C).to_bits(),
        "the chosen inputs no longer separate one rounding from two, so the \
         rest of this file proves nothing"
    );
    assert!(
        OpKind::MulAdd.fold_is_platform_specific(&[A, B, C]),
        "the IR must agree these inputs are platform-specific — this is the \
         same divergence `ConstantFold` declines to fold"
    );
    // The halved constants must double back exactly.
    assert_eq!((HALF_A + HALF_A).to_bits(), A.to_bits());
    assert_eq!((HALF_B + HALF_B).to_bits(), B.to_bits());
}

/// An unspilled `MulAdd(X, Y, Z)` reaches the backend as `FusedMulAdd`, and
/// what that compiles to is exactly what the target's hardware offers: one
/// rounding wherever there is an FMA, two on the SSE2 baseline, whose
/// `FusedMulAdd` arm is a `movaps`/`mulps`/`addps` stand-in because that is
/// all SSE2 has.
#[test]
fn an_unspilled_muladd_rounds_the_way_this_target_does() {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let z = a.push_var(2);
    let root = a.push_ternary(OpKind::MulAdd, x, y, z);

    let result = compile(&a, root).expect("compile MulAdd(X, Y, Z)");
    assert_eq!(result.spill_count, 0, "this scenario must not spill");
    let got = JitManifold::new(result.code).eval_at(Point4::new(A, B, C, 0.0));

    #[cfg(target_feature = "fma")]
    assert_bits("fused MulAdd on an FMA target", got, fused(A, B, C));
    #[cfg(not(target_feature = "fma"))]
    assert_bits(
        "fused MulAdd on the SSE2 baseline",
        got,
        decomposed(A, B, C),
    );
}

/// Under enough register pressure that `a` and `b` cannot both stay in
/// registers, the same node reaches the backend as `DecomposedMulAdd` — a
/// multiply and an add, two roundings, on *every* target including the ones
/// with an FMA.
///
/// This is the arm AVX-512 had no test for at all: `spill_pressure.rs`'s
/// scenarios are sized for the six-register SSE2 pool and stop spilling
/// against AVX-512's nineteen, so that whole file is `cfg`'d off there.
/// Shrinking the pool explicitly — `EmitCtx::with_max_regs`, which its own
/// doc calls "how a caller forces spilling deliberately" — reaches it at
/// every width instead of at whichever one the scenario happened to suit.
///
/// The multiplicands have to be *computed* values: a coordinate is precolored
/// into an input register and never spills, so `MulAdd(X, Y, Z)` cannot
/// decompose no matter how small the pool is. Doubling is the cheapest
/// computation that is also exact, which is why the inputs are `A`/`B` halved.
#[test]
fn a_spilled_muladd_rounds_twice_on_every_target() {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let z = a.push_var(2);
    // The multiplicands, defined first and consumed last, so they hold the
    // longest live ranges in the schedule.
    let ma = a.push_binary(OpKind::Add, x, x);
    let mb = a.push_binary(OpKind::Add, y, y);

    // The wall. One spilled multiplicand is not enough — `resolve_operands`
    // only decomposes when *both* are out of registers — so something has to
    // occupy the pool across the `MulAdd`. Every one of these is a distinct
    // node worth exactly +0.0 for finite inputs, so it pins registers without
    // perturbing a bit of the sum it lands in.
    let wall: Vec<ExprId> = [(x, y), (y, z), (z, x), (x, z), (y, x), (z, y)]
        .iter()
        .map(|&(l, r)| {
            let d = a.push_binary(OpKind::Sub, l, r);
            a.push_binary(OpKind::Sub, d, d)
        })
        .collect();

    let fma = a.push_ternary(OpKind::MulAdd, ma, mb, z);
    let root = wall
        .iter()
        .fold(fma, |acc, &w| a.push_binary(OpKind::Add, acc, w));

    let result = EmitCtx::with_max_regs(1)
        .compile(&a, root)
        .expect("compile spilled MulAdd");
    assert!(
        result.spill_count > 0,
        "scenario failed to create register pressure"
    );
    let got = JitManifold::new(result.code).eval_at(Point4::new(HALF_A, HALF_B, C, 0.0));
    assert_bits("decomposed MulAdd", got, decomposed(A, B, C));
}
