//! Isolate the per-batch call overhead: the same small arena compiled
//! per-batch (one `extern "C"` call per SIMD batch, Rust loop carries the
//! coordinates) versus collapse (one call per row, X as an induction value).
//!
//! A small body over a wide row is where the boundary cost shows; large
//! bodies (glyph coverage) amortize it into noise. Run with --release.

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn main() {
    use pixelflow_ir::OpKind;
    use pixelflow_ir::arena::ExprArena;
    use pixelflow_ir::backend::emit::{compile_arena_dag, compile_collapse};
    use std::time::Instant;

    const LANES: usize = pixelflow_ir::JIT_VECTOR_BYTES / 4;
    const WIDTH: usize = 1920;
    const ROWS: usize = 2000;

    // Small shader-shaped body: x*0.01 + y*0.5 - min(x, y) * 0.25.
    let mut a = ExprArena::new();
    let x = a.push_var(0);
    let y = a.push_var(1);
    let c1 = a.push_const(0.01);
    let c2 = a.push_const(0.5);
    let c3 = a.push_const(0.25);
    let t1 = a.push_binary(OpKind::Mul, x, c1);
    let t2 = a.push_binary(OpKind::Mul, y, c2);
    let mn = a.push_binary(OpKind::Min, x, y);
    let t3 = a.push_binary(OpKind::Mul, mn, c3);
    let s = a.push_binary(OpKind::Add, t1, t2);
    let root = a.push_binary(OpKind::Sub, s, t3);

    let batch = compile_arena_dag(&a, root).expect("per-batch compile");
    let collapse = compile_collapse(&a, root).expect("collapse compile");

    let groups = WIDTH / LANES;
    let mut out = vec![0.0f32; WIDTH];
    let seq: Vec<f32> = (0..LANES).map(|i| i as f32).collect();

    // Per-batch: the Rust loop the old Lattice::bake ran.
    let t0 = Instant::now();
    for row in 0..ROWS {
        let yv = row as f32;
        per_batch_row(&batch, &mut out, &seq, yv, LANES);
    }
    let per_batch_ns = t0.elapsed().as_nanos() as f64 / (ROWS * WIDTH) as f64;
    std::hint::black_box(&out);

    // Collapse: one call per row.
    let t0 = Instant::now();
    for row in 0..ROWS {
        let yv = row as f32;
        collapse_row(&collapse, &mut out, &seq, yv, groups);
    }
    let collapse_ns = t0.elapsed().as_nanos() as f64 / (ROWS * WIDTH) as f64;
    std::hint::black_box(&out);

    println!("lanes/batch:      {LANES}");
    println!("per-batch calls:  {per_batch_ns:.3} ns/pixel");
    println!("collapse row:     {collapse_ns:.3} ns/pixel");
    println!("speedup:          {:.2}x", per_batch_ns / collapse_ns);
}

#[cfg(target_arch = "x86_64")]
fn per_batch_row(
    res: &pixelflow_ir::backend::emit::CompileResult,
    out: &mut [f32],
    seq: &[f32],
    yv: f32,
    lanes: usize,
) {
    use core::arch::x86_64::*;
    #[cfg(target_feature = "avx512f")]
    unsafe {
        let f: pixelflow_ir::backend::emit::executable::KernelFn = res.code.as_fn();
        let step = _mm512_set1_ps(lanes as f32);
        let mut xv = _mm512_loadu_ps(seq.as_ptr());
        let y = _mm512_set1_ps(yv);
        let z = _mm512_setzero_ps();
        for chunk in out.chunks_exact_mut(lanes) {
            _mm512_storeu_ps(chunk.as_mut_ptr(), f(xv, y, z, z));
            xv = _mm512_add_ps(xv, step);
        }
    }
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    unsafe {
        let f: pixelflow_ir::backend::emit::executable::KernelFn = res.code.as_fn();
        let step = _mm256_set1_ps(lanes as f32);
        let mut xv = _mm256_loadu_ps(seq.as_ptr());
        let y = _mm256_set1_ps(yv);
        let z = _mm256_setzero_ps();
        for chunk in out.chunks_exact_mut(lanes) {
            _mm256_storeu_ps(chunk.as_mut_ptr(), f(xv, y, z, z));
            xv = _mm256_add_ps(xv, step);
        }
    }
    #[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
    unsafe {
        let f: pixelflow_ir::backend::emit::executable::KernelFn = res.code.as_fn();
        let step = _mm_set1_ps(lanes as f32);
        let mut xv = _mm_loadu_ps(seq.as_ptr());
        let y = _mm_set1_ps(yv);
        let z = _mm_setzero_ps();
        for chunk in out.chunks_exact_mut(lanes) {
            _mm_storeu_ps(chunk.as_mut_ptr(), f(xv, y, z, z));
            xv = _mm_add_ps(xv, step);
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn per_batch_row(
    res: &pixelflow_ir::backend::emit::CompileResult,
    out: &mut [f32],
    seq: &[f32],
    yv: f32,
    lanes: usize,
) {
    use core::arch::aarch64::*;
    unsafe {
        let f: pixelflow_ir::backend::emit::executable::KernelFn = res.code.as_fn();
        let step = vdupq_n_f32(lanes as f32);
        let mut xv = vld1q_f32(seq.as_ptr());
        let y = vdupq_n_f32(yv);
        let z = vdupq_n_f32(0.0);
        for chunk in out.chunks_exact_mut(lanes) {
            vst1q_f32(chunk.as_mut_ptr(), f(xv, y, z, z));
            xv = vaddq_f32(xv, step);
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn collapse_row(
    res: &pixelflow_ir::backend::emit::CompileResult,
    out: &mut [f32],
    seq: &[f32],
    yv: f32,
    groups: usize,
) {
    use core::arch::x86_64::*;
    #[cfg(target_feature = "avx512f")]
    unsafe {
        let f: pixelflow_ir::backend::emit::executable::CollapseKernelFn = res.code.as_fn();
        let z = _mm512_setzero_ps();
        f(
            core::ptr::null(),
            out.as_mut_ptr(),
            groups,
            _mm512_loadu_ps(seq.as_ptr()),
            _mm512_set1_ps(yv),
            z,
            z,
        );
    }
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    unsafe {
        let f: pixelflow_ir::backend::emit::executable::CollapseKernelFn = res.code.as_fn();
        let z = _mm256_setzero_ps();
        f(
            core::ptr::null(),
            out.as_mut_ptr(),
            groups,
            _mm256_loadu_ps(seq.as_ptr()),
            _mm256_set1_ps(yv),
            z,
            z,
        );
    }
    #[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
    unsafe {
        let f: pixelflow_ir::backend::emit::executable::CollapseKernelFn = res.code.as_fn();
        let z = _mm_setzero_ps();
        f(
            core::ptr::null(),
            out.as_mut_ptr(),
            groups,
            _mm_loadu_ps(seq.as_ptr()),
            _mm_set1_ps(yv),
            z,
            z,
        );
    }
}

#[cfg(target_arch = "aarch64")]
fn collapse_row(
    res: &pixelflow_ir::backend::emit::CompileResult,
    out: &mut [f32],
    seq: &[f32],
    yv: f32,
    groups: usize,
) {
    use core::arch::aarch64::*;
    unsafe {
        let f: pixelflow_ir::backend::emit::executable::CollapseKernelFn = res.code.as_fn();
        let z = vdupq_n_f32(0.0);
        f(
            core::ptr::null(),
            out.as_mut_ptr(),
            groups,
            vld1q_f32(seq.as_ptr()),
            vdupq_n_f32(yv),
            z,
            z,
        );
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn main() {
    eprintln!("collapse_overhead: JIT backends exist only on x86-64/aarch64");
}
