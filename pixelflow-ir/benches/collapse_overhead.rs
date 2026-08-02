//! Isolate Rust-to-JIT call overhead from expression cost.
//!
//! Both cases execute the same compiled arena over the same 2D lattice. The
//! baseline calls `KernelFn` once per SIMD group from a Rust loop; the collapse
//! case calls `CollapseKernelFn` once for the whole frame.

#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use pixelflow_ir::JIT_VECTOR_BYTES;
use pixelflow_ir::OpKind;
use pixelflow_ir::arena::ExprArena;
use pixelflow_ir::emit::{CompileResult, compile_arena_dag, compile_collapse};

const LANES: usize = JIT_VECTOR_BYTES / core::mem::size_of::<f32>();
const GROUPS: usize = 240;
const ROWS: usize = 64;

fn arena() -> (ExprArena, pixelflow_ir::arena::ExprId) {
    let mut arena = ExprArena::new();
    let x = arena.push_var(0);
    let y = arena.push_var(1);
    let z = arena.push_var(2);
    let scale = arena.push_const(0.013);
    let bias = arena.push_const(1.75);
    let xs = arena.push_binary(OpKind::Mul, x, scale);
    let ys = arena.push_binary(OpKind::Mul, y, scale);
    let xy = arena.push_binary(OpKind::Mul, xs, ys);
    let z2 = arena.push_binary(OpKind::Mul, z, z);
    let sum = arena.push_binary(OpKind::Add, xy, z2);
    let root = arena.push_binary(OpKind::Add, sum, bias);
    (arena, root)
}

fn bench_collapse_overhead(c: &mut Criterion) {
    let (arena, root) = arena();
    let batch = compile_arena_dag(&arena, root).expect("per-batch compile must succeed");
    let collapse = compile_collapse(&arena, root).expect("collapse compile must succeed");
    let mut out = vec![0.0f32; GROUPS * LANES * ROWS];
    let seq: Vec<f32> = (0..LANES).map(|lane| lane as f32 + 0.5).collect();

    // Warm executable pages and branch predictors before Criterion samples.
    per_batch_frame(&batch, &mut out, &seq, GROUPS, ROWS);
    collapse_frame(&collapse, &mut out, &seq, GROUPS, ROWS);

    let mut group = c.benchmark_group("jit_collapse_call_overhead");
    group.throughput(Throughput::Elements((GROUPS * LANES * ROWS) as u64));
    group.bench_function(BenchmarkId::new("rust_per_batch_loop", LANES), |b| {
        b.iter(|| per_batch_frame(black_box(&batch), black_box(&mut out), &seq, GROUPS, ROWS));
    });
    group.bench_function(BenchmarkId::new("one_2d_collapse_call", LANES), |b| {
        b.iter(|| {
            collapse_frame(
                black_box(&collapse),
                black_box(&mut out),
                &seq,
                GROUPS,
                ROWS,
            )
        });
    });
    group.finish();
}

#[cfg(target_arch = "x86_64")]
fn per_batch_frame(
    result: &CompileResult,
    out: &mut [f32],
    seq: &[f32],
    groups: usize,
    rows: usize,
) {
    use core::arch::x86_64::*;
    for row in 0..rows {
        let y = row as f32 + 0.5;
        let row_out = &mut out[row * groups * LANES..][..groups * LANES];
        #[cfg(target_feature = "avx512f")]
        unsafe {
            let f: pixelflow_ir::emit::executable::KernelFn = result.code.as_fn();
            let step = _mm512_set1_ps(LANES as f32);
            let mut x = _mm512_loadu_ps(seq.as_ptr());
            let yv = _mm512_set1_ps(y);
            let zero = _mm512_setzero_ps();
            for chunk in row_out.chunks_exact_mut(LANES) {
                _mm512_storeu_ps(chunk.as_mut_ptr(), f(x, yv, zero, zero));
                x = _mm512_add_ps(x, step);
            }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        unsafe {
            let f: pixelflow_ir::emit::executable::KernelFn = result.code.as_fn();
            let step = _mm256_set1_ps(LANES as f32);
            let mut x = _mm256_loadu_ps(seq.as_ptr());
            let yv = _mm256_set1_ps(y);
            let zero = _mm256_setzero_ps();
            for chunk in row_out.chunks_exact_mut(LANES) {
                _mm256_storeu_ps(chunk.as_mut_ptr(), f(x, yv, zero, zero));
                x = _mm256_add_ps(x, step);
            }
        }
        #[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
        unsafe {
            let f: pixelflow_ir::emit::executable::KernelFn = result.code.as_fn();
            let step = _mm_set1_ps(LANES as f32);
            let mut x = _mm_loadu_ps(seq.as_ptr());
            let yv = _mm_set1_ps(y);
            let zero = _mm_setzero_ps();
            for chunk in row_out.chunks_exact_mut(LANES) {
                _mm_storeu_ps(chunk.as_mut_ptr(), f(x, yv, zero, zero));
                x = _mm_add_ps(x, step);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn per_batch_frame(
    result: &CompileResult,
    out: &mut [f32],
    seq: &[f32],
    groups: usize,
    rows: usize,
) {
    use core::arch::aarch64::*;
    unsafe {
        let f: pixelflow_ir::emit::executable::KernelFn = result.code.as_fn();
        let step = vdupq_n_f32(LANES as f32);
        let zero = vdupq_n_f32(0.0);
        for row in 0..rows {
            let mut x = vld1q_f32(seq.as_ptr());
            let y = vdupq_n_f32(row as f32 + 0.5);
            let row_out = &mut out[row * groups * LANES..][..groups * LANES];
            for chunk in row_out.chunks_exact_mut(LANES) {
                vst1q_f32(chunk.as_mut_ptr(), f(x, y, zero, zero));
                x = vaddq_f32(x, step);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn collapse_frame(
    result: &CompileResult,
    out: &mut [f32],
    seq: &[f32],
    groups: usize,
    rows: usize,
) {
    use core::arch::x86_64::*;
    #[cfg(target_feature = "avx512f")]
    unsafe {
        let f: pixelflow_ir::emit::executable::CollapseKernelFn = result.code.as_fn();
        let zero = _mm512_setzero_ps();
        f(
            core::ptr::null(),
            out.as_mut_ptr(),
            groups,
            rows,
            0,
            _mm512_loadu_ps(seq.as_ptr()),
            _mm512_set1_ps(0.5),
            zero,
            zero,
        );
    }
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    unsafe {
        let f: pixelflow_ir::emit::executable::CollapseKernelFn = result.code.as_fn();
        let zero = _mm256_setzero_ps();
        f(
            core::ptr::null(),
            out.as_mut_ptr(),
            groups,
            rows,
            0,
            _mm256_loadu_ps(seq.as_ptr()),
            _mm256_set1_ps(0.5),
            zero,
            zero,
        );
    }
    #[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
    unsafe {
        let f: pixelflow_ir::emit::executable::CollapseKernelFn = result.code.as_fn();
        let zero = _mm_setzero_ps();
        f(
            core::ptr::null(),
            out.as_mut_ptr(),
            groups,
            rows,
            0,
            _mm_loadu_ps(seq.as_ptr()),
            _mm_set1_ps(0.5),
            zero,
            zero,
        );
    }
}

#[cfg(target_arch = "aarch64")]
fn collapse_frame(
    result: &CompileResult,
    out: &mut [f32],
    seq: &[f32],
    groups: usize,
    rows: usize,
) {
    use core::arch::aarch64::*;
    unsafe {
        let f: pixelflow_ir::emit::executable::CollapseKernelFn = result.code.as_fn();
        let zero = vdupq_n_f32(0.0);
        f(
            core::ptr::null(),
            out.as_mut_ptr(),
            groups,
            rows,
            0,
            vld1q_f32(seq.as_ptr()),
            vdupq_n_f32(0.5),
            zero,
            zero,
        );
    }
}

criterion_group!(benches, bench_collapse_overhead);
criterion_main!(benches);
