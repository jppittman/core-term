//! Collapse-loop equivalence: the internal-loop kernel produced by
//! `compile_collapse` must agree with the per-batch kernel compiled from the
//! same arena — the body is byte-identical, only the framing differs, so any
//! divergence is a bug in the loop scaffold (coordinate slots, X induction,
//! output stores, branch patching around the body).
//!
//! The per-batch path is the primary oracle (bit-exact by construction); the
//! interpreter (`eval_scalar`) additionally anchors the arithmetic and gather
//! cases to the language's reference semantics.
//!
//! Runs at whichever width the build selects (SSE2/AVX2/AVX-512/NEON): the
//! CI ISA matrix covers the x86 widths, macOS covers aarch64.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use pixelflow_ir::JIT_VECTOR_BYTES;
use pixelflow_ir::OpKind;
use pixelflow_ir::arena::{BufferDecl, ExprArena, ExprId};
use pixelflow_ir::backend::emit::executable::CollapseKernelFn;
use pixelflow_ir::backend::emit::{CompileResult, compile_arena_dag, compile_collapse};
use pixelflow_ir::binding::BindingTable;
use pixelflow_ir::eval_scalar;

const LANES: usize = JIT_VECTOR_BYTES / 4;

/// One collapse call: fill `out` (whose length must be `groups * LANES`)
/// from lane-sequential X starting at `x0`, with loop-invariant y/z/w.
#[allow(clippy::too_many_arguments)] // ctx + out + the four coordinates: the ABI's shape
fn run_collapse(res: &CompileResult, ctx: &[*const f32], out: &mut [f32], x0: f32, y: f32, z: f32, w: f32) {
    assert_eq!(out.len() % LANES, 0);
    let groups = out.len() / LANES;
    let seq: Vec<f32> = (0..LANES).map(|i| x0 + i as f32).collect();

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe {
        use core::arch::x86_64::*;
        let f: CollapseKernelFn = res.code.as_fn();
        f(
            ctx.as_ptr(),
            out.as_mut_ptr(),
            groups,
            _mm512_loadu_ps(seq.as_ptr()),
            _mm512_set1_ps(y),
            _mm512_set1_ps(z),
            _mm512_set1_ps(w),
        );
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512f")
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let f: CollapseKernelFn = res.code.as_fn();
        f(
            ctx.as_ptr(),
            out.as_mut_ptr(),
            groups,
            _mm256_loadu_ps(seq.as_ptr()),
            _mm256_set1_ps(y),
            _mm256_set1_ps(z),
            _mm256_set1_ps(w),
        );
    }
    #[cfg(all(
        target_arch = "x86_64",
        not(target_feature = "avx512f"),
        not(target_feature = "avx2")
    ))]
    unsafe {
        use core::arch::x86_64::*;
        let f: CollapseKernelFn = res.code.as_fn();
        f(
            ctx.as_ptr(),
            out.as_mut_ptr(),
            groups,
            _mm_loadu_ps(seq.as_ptr()),
            _mm_set1_ps(y),
            _mm_set1_ps(z),
            _mm_set1_ps(w),
        );
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let f: CollapseKernelFn = res.code.as_fn();
        f(
            ctx.as_ptr(),
            out.as_mut_ptr(),
            groups,
            vld1q_f32(seq.as_ptr()),
            vdupq_n_f32(y),
            vdupq_n_f32(z),
            vdupq_n_f32(w),
        );
    }
}

/// The per-batch oracle: evaluate the same arena batch by batch through the
/// per-batch entry (`CtxKernelFn` shape — a buffer-free kernel never reads
/// the context, so passing it unconditionally is sound) over the same row.
#[allow(clippy::too_many_arguments)] // same ABI shape as run_collapse
fn run_per_batch(res: &CompileResult, ctx: &[*const f32], out: &mut [f32], x0: f32, y: f32, z: f32, w: f32) {
    assert_eq!(out.len() % LANES, 0);
    for (g, chunk) in out.chunks_exact_mut(LANES).enumerate() {
        let base = x0 + (g * LANES) as f32;
        let seq: Vec<f32> = (0..LANES).map(|i| base + i as f32).collect();

        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            use core::arch::x86_64::*;
            let f: pixelflow_ir::backend::emit::executable::CtxKernelFn = res.code.as_fn();
            let r = f(
                ctx.as_ptr(),
                _mm512_loadu_ps(seq.as_ptr()),
                _mm512_set1_ps(y),
                _mm512_set1_ps(z),
                _mm512_set1_ps(w),
            );
            _mm512_storeu_ps(chunk.as_mut_ptr(), r);
        }
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            not(target_feature = "avx512f")
        ))]
        unsafe {
            use core::arch::x86_64::*;
            let f: pixelflow_ir::backend::emit::executable::CtxKernelFn = res.code.as_fn();
            let r = f(
                ctx.as_ptr(),
                _mm256_loadu_ps(seq.as_ptr()),
                _mm256_set1_ps(y),
                _mm256_set1_ps(z),
                _mm256_set1_ps(w),
            );
            _mm256_storeu_ps(chunk.as_mut_ptr(), r);
        }
        #[cfg(all(
            target_arch = "x86_64",
            not(target_feature = "avx512f"),
            not(target_feature = "avx2")
        ))]
        unsafe {
            use core::arch::x86_64::*;
            let f: pixelflow_ir::backend::emit::executable::CtxKernelFn = res.code.as_fn();
            let r = f(
                ctx.as_ptr(),
                _mm_loadu_ps(seq.as_ptr()),
                _mm_set1_ps(y),
                _mm_set1_ps(z),
                _mm_set1_ps(w),
            );
            _mm_storeu_ps(chunk.as_mut_ptr(), r);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use core::arch::aarch64::*;
            let f: pixelflow_ir::backend::emit::executable::CtxKernelFn = res.code.as_fn();
            let r = f(
                ctx.as_ptr(),
                vld1q_f32(seq.as_ptr()),
                vdupq_n_f32(y),
                vdupq_n_f32(z),
                vdupq_n_f32(w),
            );
            vst1q_f32(chunk.as_mut_ptr(), r);
        }
    }
}

/// Compile both entries from the same arena and require bit-identical output
/// over several rows; returns the collapse result for extra assertions.
fn assert_collapse_matches_per_batch(
    arena: &ExprArena,
    root: ExprId,
    ctx: &[*const f32],
    groups: usize,
    label: &str,
) -> CompileResult {
    let collapse = compile_collapse(arena, root)
        .unwrap_or_else(|e| panic!("{label}: collapse compile failed: {e}"));
    let batch = compile_arena_dag(arena, root)
        .unwrap_or_else(|e| panic!("{label}: per-batch compile failed: {e}"));

    for &(x0, y, z, w) in &[
        (0.5f32, 0.5f32, 0.0f32, 0.0f32),
        (-7.25, 3.5, 0.1, 0.9),
        (100.0, -2.0, 1.0, -1.0),
    ] {
        let mut got = vec![0.0f32; groups * LANES];
        let mut want = vec![0.0f32; groups * LANES];
        run_collapse(&collapse, ctx, &mut got, x0, y, z, w);
        run_per_batch(&batch, ctx, &mut want, x0, y, z, w);
        for (i, (g, e)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                g == e || (g.is_nan() && e.is_nan()),
                "{label}: collapse[{i}] = {g} != per-batch {e} (row x0={x0}, y={y})"
            );
        }
    }
    collapse
}

#[test]
fn arithmetic_row_matches_per_batch_and_interpreter() {
    // x*y + (x - 1.5): induction X must advance exactly like the per-batch
    // loop's, and Y must survive every iteration in its slot.
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let c = a.push_const(1.5);
    let xy = a.push_binary(OpKind::Mul, x, y);
    let xc = a.push_binary(OpKind::Sub, x, c);
    let root = a.push_binary(OpKind::Add, xy, xc);

    let res = assert_collapse_matches_per_batch(&a, root, &[], 5, "arith");

    // And anchor to the reference semantics.
    let mut out = vec![0.0f32; 5 * LANES];
    run_collapse(&res, &[], &mut out, 2.0, 3.0, 0.0, 0.0);
    for (i, &got) in out.iter().enumerate() {
        let xi = 2.0 + i as f32;
        let want = eval_scalar(&a, root, &[xi, 3.0, 0.0, 0.0], &BindingTable::empty());
        assert_eq!(got, want, "arith vs interp at lane {i}");
    }
}

#[test]
fn select_short_circuit_branches_inside_loop() {
    // select(x < t, x*2, y): different batches take different guard paths
    // (all-true, mixed, all-false), so the short-circuit branches and the
    // loop's own branches must patch independently.
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let t = a.push_const(8.0);
    let cond = a.push_binary(OpKind::Lt, x, t);
    let two = a.push_const(2.0);
    let x2 = a.push_binary(OpKind::Mul, x, two);
    let root = a.push_ternary(OpKind::Select, cond, x2, y);

    assert_collapse_matches_per_batch(&a, root, &[], 6, "select");
}

#[test]
fn transcendentals_rematerialize_per_iteration() {
    // sin/exp expand to polynomial chains full of constants: on aarch64 they
    // load X17-relative from the pool anchored in the collapse prologue, on
    // x86 they embed/broadcast — every iteration, inside the loop.
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let s = a.push_const(0.05);
    let xs = a.push_binary(OpKind::Mul, x, s);
    let sin = a.push_unary(OpKind::Sin, xs);
    let ey = a.push_unary(OpKind::Exp, y);
    let root = a.push_binary(OpKind::Add, sin, ey);

    assert_collapse_matches_per_batch(&a, root, &[], 4, "transcendental");
}

#[test]
fn gather_reads_ctx_every_iteration() {
    // The context register must survive the whole loop: a buffer read per
    // batch whose indices depend on the induction X.
    let w = 64usize;
    let buf: Vec<f32> = (0..w * 2).map(|k| (k as f32) * 0.25 - 3.0).collect();

    let mut a = ExprArena::new();
    let b = a.declare_buffer(BufferDecl {
        width: w as u32,
        height: 2,
    });
    let x = a.push_var(0);
    let y = a.push_var(1);
    let g = a.push_gather(b, x, y);
    let x01 = a.push_const(0.125);
    let xf = a.push_binary(OpKind::Mul, x, x01);
    let root = a.push_binary(OpKind::Add, g, xf);

    let ctx = [buf.as_ptr()];
    let res = assert_collapse_matches_per_batch(&a, root, &ctx, 4, "gather");

    // Interpreter anchor over one row.
    let bindings = BindingTable::bind(&a, &[buf.as_slice()]).unwrap();
    let mut out = vec![0.0f32; 4 * LANES];
    run_collapse(&res, &ctx, &mut out, 0.0, 1.0, 0.0, 0.0);
    for (i, &got) in out.iter().enumerate() {
        let want = eval_scalar(&a, root, &[i as f32, 1.0, 0.0, 0.0], &bindings);
        assert_eq!(got, want, "gather vs interp at lane {i}");
    }
}

#[test]
fn matmul_reduce_one_call_fills_output() {
    // out(j) = Σ_i W(i,j) * input(i): the reduce unrolls to a gather/mul/add
    // chain, and ONE collapse call fills the whole output vector.
    let (in_dim, out_dim) = (5usize, 4 * LANES);
    let w: Vec<f32> = (0..(in_dim * out_dim)).map(|k| (k as f32).sin()).collect();
    let input: Vec<f32> = (0..in_dim).map(|k| (k as f32 - 2.0) * 0.5).collect();

    let mut a = ExprArena::new();
    let wb = a.declare_buffer(BufferDecl {
        width: in_dim as u32,
        height: out_dim as u32,
    });
    let ib = a.declare_buffer(BufferDecl {
        width: in_dim as u32,
        height: 1,
    });
    let i = a.push_var(4);
    let j = a.push_var(0);
    let zero = a.push_const(0.0);
    let wg = a.push_gather(wb, i, j);
    let ig = a.push_gather(ib, i, zero);
    let prod = a.push_binary(OpKind::Mul, wg, ig);
    let root = a.push_reduce(OpKind::Add, 4, in_dim as u32, prod);

    let ctx = [w.as_ptr(), input.as_ptr()];
    let res = compile_collapse(&a, root).expect("matmul collapse compile");

    let mut out = vec![0.0f32; out_dim];
    run_collapse(&res, &ctx, &mut out, 0.0, 0.0, 0.0, 0.0);

    let bindings = BindingTable::bind(&a, &[w.as_slice(), input.as_slice()]).unwrap();
    for (j, &got) in out.iter().enumerate() {
        let want = eval_scalar(&a, root, &[j as f32, 0.0, 0.0, 0.0], &bindings);
        assert_eq!(got, want, "matmul out[{j}]");
    }
}

#[test]
fn spill_frame_coexists_with_coordinate_slots() {
    // Force spilling (more simultaneously-live values than the allocator's
    // budget) so the body's spill frame and the scaffold's coordinate slots
    // must share the stack without aliasing — in both x86 frame modes.
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let mut products = Vec::new();
    for k in 0..12 {
        let c = a.push_const(1.0 + k as f32 * 0.37);
        let xk = a.push_binary(OpKind::Add, x, c);
        let yk = a.push_binary(OpKind::Mul, y, c);
        products.push(a.push_binary(OpKind::Mul, xk, yk));
    }
    // Balanced tree so many products stay live at once.
    while products.len() > 1 {
        let mut next = Vec::new();
        for pair in products.chunks(2) {
            next.push(match pair {
                [l, r] => a.push_binary(OpKind::Add, *l, *r),
                [l] => *l,
                _ => unreachable!(),
            });
        }
        products = next;
    }
    let root = products[0];

    let res = assert_collapse_matches_per_batch(&a, root, &[], 3, "spill");
    assert!(
        res.spill_count > 0,
        "pressure kernel did not spill (budget grew?) — the scenario proves nothing"
    );
}
