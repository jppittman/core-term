//! A skipped `Select` arm is still a path, and the location table is what
//! every path after the join agrees on.
//!
//! The short-circuit guard branches around an arm whose mask is uniform. The
//! allocator, meanwhile, may schedule a *reconciliation* at the arm's end
//! point — a value coming back into a register for the code that follows the
//! arm. That reload belongs to both paths. Emitting it before patching the
//! branch put it inside the skipped region, so the uniform-mask path arrived
//! with the register unloaded while the emitter's table claimed the value was
//! in it, and every later read of that value read whatever the register
//! happened to hold.
//!
//! Nothing catches this except running the guarded path: the placement is
//! self-consistent, the arms are correct, and only the *ordering* of two
//! emissions at one index is wrong. So this sweeps a family of kernels that
//! spill a value across a guarded arm and read it after, and runs each under
//! masks that take the arm, skip it, and mix — against the scalar oracle.

use pixelflow_codegen::emit::EmitCtx;
use pixelflow_codegen::{JitManifold, Point4};
use pixelflow_ir::{BindingTable, ExprArena, ExprId, LatticeShape, OpKind, eval_scalar};

const LANES: usize = pixelflow_codegen::JIT_VECTOR_BYTES / 4;

/// A kernel that spills `X·Y` under pressure, guards a `Select` arm, and reads
/// the spilled value `after_reads` times *following* the arm — which is where
/// the allocator puts the reload that the skipped path used to jump over.
///
/// `width` sets how many independent live values compete for the pool, so the
/// sweep covers schedules that spill in different places rather than one.
fn kernel(width: u32, after_reads: usize) -> (ExprArena, ExprId) {
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let z = a.push_var(2);

    // The mask is Z's sign, so a caller can make it uniform per batch.
    let zero = a.push_const(0.0);
    let cond = a.push_binary(OpKind::Gt, z, zero);
    // Computed first, read last: what eviction takes when the filler fills
    // the pool.
    let split = a.push_binary(OpKind::Mul, x, y);

    let terms: Vec<ExprId> = (1..=width)
        .map(|i| {
            let c = a.push_const(i as f32);
            a.push_binary(OpKind::Add, x, c)
        })
        .collect();
    let mid = terms[1..]
        .iter()
        .fold(terms[0], |acc, &t| a.push_binary(OpKind::Add, acc, t));

    // One arm only: the false side is `split`, computed outside the `Select`,
    // so the true arm's end IS the `Select`'s own index and lies outside every
    // arm. The `Select` reads `split`, which by then is spilled, and `split` is
    // read again after — so the allocator brings it back into a pool register
    // exactly at the join. That reload is the one the skipping path must not
    // jump over.
    let base = a.push_binary(OpKind::Mul, mid, y);
    let t1 = a.push_binary(OpKind::Mul, base, base);
    let t2 = a.push_binary(OpKind::Add, t1, base);
    let t3 = a.push_binary(OpKind::Mul, t2, t1);
    let sel = a.push_ternary(OpKind::Select, cond, t3, split);

    let mut acc = sel;
    for _ in 0..after_reads {
        acc = a.push_binary(OpKind::Add, acc, split);
        acc = a.push_binary(OpKind::Mul, acc, split);
    }
    (a, acc)
}

/// Z decides the mask: all-negative skips the true arm, all-positive skips the
/// false arm, and the mixed row takes neither branch.
fn lanes_for(kind: usize) -> [f32; LANES] {
    core::array::from_fn(|i| match kind {
        0 => -1.0,
        1 => 1.0,
        _ if i % 2 == 0 => 1.0,
        _ => -1.0,
    })
}

#[test]
fn a_reload_at_a_guarded_arms_end_happens_on_the_skipping_path_too() {
    let bindings = BindingTable::empty();
    let mut checked = 0usize;
    for pool in [6u8, 7, 8, 10] {
        for width in 4u32..24 {
            for after_reads in 1..=4 {
                let (arena, root) = kernel(width, after_reads);
                let compiled = EmitCtx::with_max_regs(pool)
                    .compile(&arena, root)
                    .expect("compiles");
                let jit = JitManifold::new(compiled.code, LatticeShape::POINT);
                for kind in 0..3 {
                    let zs = lanes_for(kind);
                    let xs: [f32; LANES] = core::array::from_fn(|i| 0.5 + i as f32);
                    let ys: [f32; LANES] = core::array::from_fn(|i| 1.5 - i as f32 * 0.25);
                    let ws = [0.0f32; LANES];
                    let got: [f32; LANES] = unsafe { jit.call(Point4::new(xs, ys, zs, ws)) };
                    for lane in 0..LANES {
                        let point = [xs[lane], ys[lane], zs[lane], ws[lane]];
                        let want = eval_scalar(&arena, root, &point, &bindings);
                        assert_eq!(
                            got[lane].to_bits(),
                            want.to_bits(),
                            "pool {pool}, width {width}, {after_reads} reads after the arm, \
                         mask kind {kind}, lane {lane}: JIT {} vs oracle {want}",
                            got[lane],
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 0, "the sweep verified nothing");
}
