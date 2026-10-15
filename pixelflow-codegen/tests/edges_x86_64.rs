//! The x86-64 column of CLAUDE.md "Floating point at the edges", run on the
//! hardware.
//!
//! That table states what each op does on each target because the instructions
//! do. The x86 column is the documented answer for this target, so it is
//! pinned here against the code the x86 emitters produce. The aarch64 column is
//! not written: this host cannot run it.
//!
//! The kernels are compiled through [`emit::compile`], the emitter with no
//! optimizer in front of it. The e-graph may commute `Min`/`Max` (CLAUDE.md,
//! "within a target, the optimizer may still produce a different answer"), so
//! the instruction's own answer is only observable on the unoptimized route.
//! Every case runs on whichever x86 tier the build selects; `PIXELFLOW_ISA`
//! picks the tier, and both tiers document the same row.

#![cfg(target_arch = "x86_64")]

use pixelflow_codegen::emit::{CompiledKernel, compile};
use pixelflow_ir::{Kernel, LatticeShape};

const NAN: f32 = f32::NAN;

/// The emitter's code for `k`, at one point.
fn compiled(k: &Kernel) -> CompiledKernel {
    let (arena, root) = k.parts();
    compile(arena, root, LatticeShape::POINT)
        .expect("the kernel compiles")
        .code
}

/// The kernel's value at `(x, y)`, read back through the origin block.
fn eval(code: &CompiledKernel, x: f32, y: f32) -> f32 {
    let mut out = [0.0f32; 1];
    let origin = [x, y];
    // SAFETY: these kernels declare no buffer and no uniform, so `ctx[0]`, the
    // uniform-block slot, is unread; `ctx[1]` is the origin block, and `out`
    // holds the one sample a single-point lattice writes.
    let ctx: [*const f32; 2] = [core::ptr::null(), origin.as_ptr()];
    unsafe {
        code.call(ctx.as_ptr(), out.as_mut_ptr(), 1);
    }
    out[0]
}

/// `min` and `max` with one NaN operand return the second operand, whichever
/// position the NaN is in: `vminps`/`vmaxps` hand back their second source when
/// either is NaN.
#[test]
fn min_and_max_with_one_nan_operand_return_the_second_operand() {
    for (name, k) in [
        ("min", Kernel::x().min(&Kernel::y())),
        ("max", Kernel::x().max(&Kernel::y())),
    ] {
        let code = compiled(&k);
        assert_eq!(
            eval(&code, NAN, 1.0).to_bits(),
            1.0f32.to_bits(),
            "{name}(NaN, 1.0) must return the second operand, 1.0"
        );
        assert_eq!(
            eval(&code, 1.0, NAN).to_bits(),
            NAN.to_bits(),
            "{name}(1.0, NaN) must return the second operand, the NaN"
        );
    }
}

/// `Gt` and `Ge` are the unordered predicates: a NaN operand makes the lane
/// true. The lane is a mask, so true is all-ones (`0xFFFF_FFFF`, a NaN as an
/// f32) and false is all-zero.
#[test]
fn gt_and_ge_with_a_nan_operand_are_true_all_ones_masks() {
    const ALL_ONES: u32 = 0xFFFF_FFFF;
    for (name, k) in [
        ("gt", Kernel::x().gt(&Kernel::y())),
        ("ge", Kernel::x().ge(&Kernel::y())),
    ] {
        let code = compiled(&k);
        for (x, y) in [(NAN, 1.0), (1.0, NAN), (NAN, NAN)] {
            assert_eq!(
                eval(&code, x, y).to_bits(),
                ALL_ONES,
                "{name}({x}, {y}): a NaN operand must be an all-ones true mask"
            );
        }
        assert_eq!(
            eval(&code, 2.0, 1.0).to_bits(),
            ALL_ONES,
            "{name}(2.0, 1.0) is true, an all-ones mask"
        );
        assert_eq!(
            eval(&code, 1.0, 2.0).to_bits(),
            0,
            "{name}(1.0, 2.0) is false, an all-zero mask"
        );
    }
}

/// `Round` is nearest-even on x86 (`vroundps` imm 0x00), so an exact tie goes
/// to the even integer. A negative input in `[-0.5, -0.0]` keeps its sign.
#[test]
fn round_breaks_ties_to_even_and_keeps_the_sign_of_negative_zero() {
    let code = compiled(&Kernel::x().round());
    assert_eq!(
        eval(&code, 2.5, 0.0).to_bits(),
        2.0f32.to_bits(),
        "round(2.5) is the even neighbour, 2"
    );
    assert_eq!(
        eval(&code, 3.5, 0.0).to_bits(),
        4.0f32.to_bits(),
        "round(3.5) is the even neighbour, 4"
    );
    assert_eq!(
        eval(&code, -0.5, 0.0).to_bits(),
        (-0.0f32).to_bits(),
        "round(-0.5) is -0.0, with the sign bit set"
    );
}

/// An out-of-range or NaN float-to-int conversion yields the x86 "integer
/// indefinite" pattern, `i32::MIN`, not a saturated value. The result is an
/// integer bit pattern, so it is read as `i32`.
#[test]
fn trunc_to_int_of_nan_or_2_pow_31_is_i32_min() {
    let code = compiled(&Kernel::x().trunc_to_int().into_kernel());
    assert_eq!(
        eval(&code, NAN, 0.0).to_bits() as i32,
        i32::MIN,
        "cvttps2dq of NaN is the integer indefinite, i32::MIN"
    );
    assert_eq!(
        eval(&code, 2_147_483_648.0, 0.0).to_bits() as i32,
        i32::MIN,
        "cvttps2dq of 2^31 is out of range, so the integer indefinite, i32::MIN"
    );
}
