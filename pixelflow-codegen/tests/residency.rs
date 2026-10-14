//! A value stays in its register until something needs the register.
//!
//! These fail on an allocator that stores every value to the frame when it is
//! defined and reloads it at every read. They are written against what a
//! kernel emits, not against the allocator: its traffic, its frame, and the
//! numbers it computes.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_codegen::emit::{CompileResult, CompiledKernel, compile};
use pixelflow_ir::{ExprArena, ExprId, LatticeShape, OpKind};

/// `build`'s kernel over the lattice coordinates `x` and `y`, compiled at one
/// point.
fn compiled(build: impl FnOnce(&mut ExprArena, ExprId, ExprId) -> ExprId) -> CompileResult {
    let mut a = ExprArena::new();
    let [x, y] = [0, 1].map(|v| a.push_var(v));
    let root = build(&mut a, x, y);
    compile(&a, root, LatticeShape::POINT).expect("the kernel compiles")
}

/// The one sample of a kernel compiled at one point, at `(x, y)`.
fn eval_point(kernel: &CompiledKernel, x: f32, y: f32) -> f32 {
    let mut out = [0.0f32; 1];
    let origin = [x, y];
    let no_block: [f32; 0] = [];
    // SAFETY: `ctx[0]` is the uniform block, which these kernels declare
    // nothing for, `ctx[1]` is the origin block, and `out` holds the one
    // sample a single-point lattice writes.
    let ctx: [*const f32; 2] = [no_block.as_ptr(), origin.as_ptr()];
    unsafe {
        kernel.call(ctx.as_ptr(), out.as_mut_ptr(), 1);
    }
    out[0]
}

/// `MulAdd` accumulates into its addend. When the addend is read again after
/// the `MulAdd`, the allocator copies it first, so the copy is what is
/// overwritten.
#[test]
fn an_operand_overwritten_in_place_survives_when_it_is_read_again() {
    let kernel = compiled(|a, x, y| {
        let fused = a.push_ternary(OpKind::MulAdd, x, y, x);
        a.push_binary(OpKind::Add, fused, x)
    });
    let (x, y) = (3.0f32, 5.0f32);
    assert_eq!(
        eval_point(&kernel.code, x, y).to_bits(),
        (x.mul_add(y, x) + x).to_bits()
    );
}

/// Constants past the vector file, summed twice over in one chain, each still
/// read at its value: the first pass's reads leave every constant wanted again,
/// so the allocator must give up registers and define constants again where
/// they are read. The count past the file is sized from the pool, so the
/// pressure is real on every tier; only the value is asserted.
#[test]
fn constants_are_rematerialized_rather_than_spilled() {
    /// Constants that fit every tier's vector file with room to spare.
    const FEW: usize = 4;
    /// Constants past the vector file.
    const EXCESS: usize = 24;
    let check = |count: usize| -> CompileResult {
        let values: Vec<f32> = (0..count).map(|i| 1.0 + 0.25 * i as f32).collect();
        let kernel = compiled(|a, _, _| {
            let leaves: Vec<_> = values.iter().map(|&c| a.push_const(c)).collect();
            let once = leaves[1..]
                .iter()
                .fold(leaves[0], |acc, &c| a.push_binary(OpKind::Add, acc, c));
            leaves
                .iter()
                .fold(once, |acc, &c| a.push_binary(OpKind::Add, acc, c))
        });
        let want = 2.0 * values.iter().sum::<f32>();
        let got = eval_point(&kernel.code, 0.0, 0.0);
        assert!(
            (got - want).abs() <= want.abs() * 1e-5,
            "{count} constants: {got} against {want}"
        );
        kernel
    };
    let few = check(FEW).traffic;
    check(usize::from(few.pool) + EXCESS);
}
