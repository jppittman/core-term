//! JitManifold: a JIT-compiled function held as executable memory.
//!
//! This type owns an [`ExecutableCode`] and exposes it through a platform-specific
//! `call` method. It does NOT implement `pixelflow_core::Manifold` directly — that
//! would create a dependency cycle. Instead, `kernel_jit!` emits a thin wrapper in
//! the user's crate that calls through to `JitManifold`.

use crate::backend::emit::executable::{CtxKernelFn, ExecutableCode, KernelFn};

/// A JIT-compiled kernel. Owns the executable code for one specific parameter
/// combination. No cache — caller decides lifetime.
pub struct JitManifold {
    code: ExecutableCode,
}

impl JitManifold {
    /// Wrap compiled executable code.
    #[must_use]
    pub fn new(code: ExecutableCode) -> Self {
        Self { code }
    }
}

#[cfg(target_arch = "aarch64")]
impl JitManifold {
    /// Evaluate the kernel at the given coordinates.
    ///
    /// # Safety
    ///
    /// The caller must ensure the SIMD types match the platform ABI that the
    /// emitter generated code for (ARM64 NEON: `float32x4_t`).
    #[must_use]
    #[inline(always)]
    pub unsafe fn call(
        &self,
        x: core::arch::aarch64::float32x4_t,
        y: core::arch::aarch64::float32x4_t,
        z: core::arch::aarch64::float32x4_t,
        w: core::arch::aarch64::float32x4_t,
    ) -> core::arch::aarch64::float32x4_t {
        // SAFETY: The ExecutableCode was produced by our JIT compiler which
        // emits code matching the KernelFn signature.
        let func: KernelFn = unsafe { self.code.as_fn() };
        func(x, y, z, w)
    }

    /// Evaluate a bound-memory kernel: `ctx` is the array of buffer base
    /// pointers, indexed by [`BufferId`](crate::arena::BufferId), passed in
    /// the first integer register (`x0`).
    ///
    /// # Safety
    ///
    /// - The kernel must have been compiled from an arena that declares
    ///   buffers (its code reads the context register); use [`Self::call`]
    ///   for buffer-free kernels, whose code never looks at `x0`.
    /// - `ctx` must hold one valid base pointer per declared buffer, each
    ///   pointing at `width * height` readable `f32`s matching its
    ///   [`BufferDecl`](crate::arena::BufferDecl), and must outlive the call.
    /// - The SIMD types must match the platform ABI the emitter generated
    ///   code for (ARM64 NEON: `float32x4_t`).
    #[allow(clippy::too_many_arguments)] // ctx + the four fixed coordinates
    #[must_use]
    #[inline(always)]
    pub unsafe fn call_bound(
        &self,
        ctx: *const *const f32,
        x: core::arch::aarch64::float32x4_t,
        y: core::arch::aarch64::float32x4_t,
        z: core::arch::aarch64::float32x4_t,
        w: core::arch::aarch64::float32x4_t,
    ) -> core::arch::aarch64::float32x4_t {
        // SAFETY: The ExecutableCode was produced by our JIT compiler which
        // emits code matching the CtxKernelFn signature for buffer-declaring
        // arenas.
        let func: CtxKernelFn = unsafe { self.code.as_fn() };
        func(ctx, x, y, z, w)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    not(target_feature = "avx2")
))]
impl JitManifold {
    /// Evaluate the kernel at the given coordinates (SSE2, 128-bit, 4 lanes).
    ///
    /// # Safety
    ///
    /// The caller must ensure the SIMD types match the platform ABI the emitter
    /// generated code for (x86-64 SSE2: `__m128`).
    #[inline(always)]
    #[must_use]
    pub unsafe fn call(
        &self,
        x: core::arch::x86_64::__m128,
        y: core::arch::x86_64::__m128,
        z: core::arch::x86_64::__m128,
        w: core::arch::x86_64::__m128,
    ) -> core::arch::x86_64::__m128 {
        // SAFETY: The ExecutableCode was produced by our JIT compiler which
        // emits code matching the KernelFn signature.
        let func: KernelFn = unsafe { self.code.as_fn() };
        func(x, y, z, w)
    }

    /// Evaluate a bound-memory kernel: `ctx` is the array of buffer base
    /// pointers, indexed by [`BufferId`](crate::arena::BufferId), passed in
    /// the first integer register (`rdi`).
    ///
    /// # Safety
    ///
    /// - The kernel must have been compiled from an arena that declares
    ///   buffers (its code reads the context register); use [`Self::call`]
    ///   for buffer-free kernels, whose code never looks at `rdi`.
    /// - `ctx` must hold one valid base pointer per declared buffer, each
    ///   pointing at `width * height` readable `f32`s matching its
    ///   [`BufferDecl`](crate::arena::BufferDecl), and must outlive the call.
    /// - The SIMD types must match the platform ABI the emitter generated
    ///   code for (x86-64 SSE2: `__m128`).
    #[allow(clippy::too_many_arguments)] // ctx + the four fixed coordinates
    #[inline(always)]
    #[must_use]
    pub unsafe fn call_bound(
        &self,
        ctx: *const *const f32,
        x: core::arch::x86_64::__m128,
        y: core::arch::x86_64::__m128,
        z: core::arch::x86_64::__m128,
        w: core::arch::x86_64::__m128,
    ) -> core::arch::x86_64::__m128 {
        // SAFETY: The ExecutableCode was produced by our JIT compiler which
        // emits code matching the CtxKernelFn signature for buffer-declaring
        // arenas.
        let func: CtxKernelFn = unsafe { self.code.as_fn() };
        func(ctx, x, y, z, w)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
impl JitManifold {
    /// Evaluate the kernel at the given coordinates (AVX2, 256-bit, 8 lanes).
    ///
    /// # Safety
    ///
    /// The caller must ensure the SIMD types match the platform ABI the emitter
    /// generated code for (x86-64 AVX2: `__m256`).
    #[inline(always)]
    #[must_use]
    pub unsafe fn call(
        &self,
        x: core::arch::x86_64::__m256,
        y: core::arch::x86_64::__m256,
        z: core::arch::x86_64::__m256,
        w: core::arch::x86_64::__m256,
    ) -> core::arch::x86_64::__m256 {
        // SAFETY: The ExecutableCode was produced by our JIT compiler which
        // emits code matching the KernelFn signature.
        let func: KernelFn = unsafe { self.code.as_fn() };
        func(x, y, z, w)
    }

    /// Evaluate a bound-memory kernel: `ctx` is the array of buffer base
    /// pointers, indexed by [`BufferId`](crate::arena::BufferId), passed in
    /// the first integer register (`rdi`).
    ///
    /// # Safety
    ///
    /// - The kernel must have been compiled from an arena that declares
    ///   buffers (its code reads the context register); use [`Self::call`]
    ///   for buffer-free kernels, whose code never looks at `rdi`.
    /// - `ctx` must hold one valid base pointer per declared buffer, each
    ///   pointing at `width * height` readable `f32`s matching its
    ///   [`BufferDecl`](crate::arena::BufferDecl), and must outlive the call.
    /// - The SIMD types must match the platform ABI the emitter generated
    ///   code for (x86-64 AVX2: `__m256`).
    #[allow(clippy::too_many_arguments)] // ctx + the four fixed coordinates
    #[inline(always)]
    #[must_use]
    pub unsafe fn call_bound(
        &self,
        ctx: *const *const f32,
        x: core::arch::x86_64::__m256,
        y: core::arch::x86_64::__m256,
        z: core::arch::x86_64::__m256,
        w: core::arch::x86_64::__m256,
    ) -> core::arch::x86_64::__m256 {
        // SAFETY: The ExecutableCode was produced by our JIT compiler which
        // emits code matching the CtxKernelFn signature for buffer-declaring
        // arenas.
        let func: CtxKernelFn = unsafe { self.code.as_fn() };
        func(ctx, x, y, z, w)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
impl JitManifold {
    /// Evaluate the kernel at the given coordinates (AVX-512, 512-bit, 16 lanes).
    ///
    /// # Safety
    ///
    /// The caller must ensure the SIMD types match the platform ABI the emitter
    /// generated code for (x86-64 AVX-512: `__m512`).
    #[inline(always)]
    #[must_use]
    pub unsafe fn call(
        &self,
        x: core::arch::x86_64::__m512,
        y: core::arch::x86_64::__m512,
        z: core::arch::x86_64::__m512,
        w: core::arch::x86_64::__m512,
    ) -> core::arch::x86_64::__m512 {
        // SAFETY: The ExecutableCode was produced by our JIT compiler which
        // emits code matching the KernelFn signature.
        let func: KernelFn = unsafe { self.code.as_fn() };
        func(x, y, z, w)
    }

    /// Evaluate a bound-memory kernel: `ctx` is the array of buffer base
    /// pointers, indexed by [`BufferId`](crate::arena::BufferId), passed in
    /// the first integer register (`rdi`).
    ///
    /// # Safety
    ///
    /// - The kernel must have been compiled from an arena that declares
    ///   buffers (its code reads the context register); use [`Self::call`]
    ///   for buffer-free kernels, whose code never looks at `rdi`.
    /// - `ctx` must hold one valid base pointer per declared buffer, each
    ///   pointing at `width * height` readable `f32`s matching its
    ///   [`BufferDecl`](crate::arena::BufferDecl), and must outlive the call.
    /// - The SIMD types must match the platform ABI the emitter generated
    ///   code for (x86-64 AVX-512: `__m512`).
    #[allow(clippy::too_many_arguments)] // ctx + the four fixed coordinates
    #[inline(always)]
    #[must_use]
    pub unsafe fn call_bound(
        &self,
        ctx: *const *const f32,
        x: core::arch::x86_64::__m512,
        y: core::arch::x86_64::__m512,
        z: core::arch::x86_64::__m512,
        w: core::arch::x86_64::__m512,
    ) -> core::arch::x86_64::__m512 {
        // SAFETY: The ExecutableCode was produced by our JIT compiler which
        // emits code matching the CtxKernelFn signature for buffer-declaring
        // arenas.
        let func: CtxKernelFn = unsafe { self.code.as_fn() };
        func(ctx, x, y, z, w)
    }
}

// SAFETY: ExecutableCode is read-only mapped memory with no interior mutability.
unsafe impl Send for JitManifold {}
unsafe impl Sync for JitManifold {}

// =============================================================================
// Scanline JIT kernel — eliminates the per-batch function-pointer overhead
// =============================================================================

use crate::backend::emit::executable::ScanlineKernelFn;

/// A JIT-compiled scanline kernel that processes multiple pixels in a single call.
///
/// Unlike [`JitManifold`] which calls through an `extern "C"` function pointer
/// per batch, `ScanlineJitManifold` contains its own loop in the emitted code.
/// Y/Z/W stay in NEON registers for the entire scanline (loop-invariant by
/// construction), eliminating the ~2.2x overhead of the Rust-JIT boundary.
pub struct ScanlineJitManifold {
    code: ExecutableCode,
}

impl ScanlineJitManifold {
    /// Wrap compiled scanline executable code.
    #[must_use]
    pub fn new(code: ExecutableCode) -> Self {
        Self { code }
    }
}

#[cfg(target_arch = "aarch64")]
impl ScanlineJitManifold {
    /// Evaluate the kernel across a scanline of pixels.
    ///
    /// The JIT'd code contains its own loop — Y/Z/W are loaded once and stay in
    /// NEON registers for all pixels. Only X varies across pixels (loaded per batch from `xs`).
    ///
    /// # Safety
    ///
    /// - `xs` and `output` must be properly aligned for `float32x4_t` (16 bytes).
    /// - `output.len()` must be >= `xs.len()`.
    /// - The code must have been compiled by [`compile_arena_dag_scanline`].
    ///
    /// # Panics
    ///
    /// Panics if `output.len() < xs.len()`.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub unsafe fn eval_scanline(
        &self,
        xs: &[core::arch::aarch64::float32x4_t],
        y: core::arch::aarch64::float32x4_t,
        z: core::arch::aarch64::float32x4_t,
        w: core::arch::aarch64::float32x4_t,
        output: &mut [core::arch::aarch64::float32x4_t],
    ) {
        assert!(
            output.len() >= xs.len(),
            "ScanlineJitManifold::eval_scanline: output buffer too small \
             (have {}, need {})",
            output.len(),
            xs.len()
        );
        if xs.is_empty() {
            return;
        }
        let func: ScanlineKernelFn = unsafe { self.code.as_fn() };
        func(xs.as_ptr(), y, z, w, output.as_mut_ptr(), xs.len());
    }
}

#[cfg(target_arch = "x86_64")]
impl ScanlineJitManifold {
    /// Evaluate the kernel across a scanline of pixels.
    ///
    /// The JIT'd code contains its own loop — Y/Z/W are loaded once and stay in
    /// SSE registers for all pixels. Only X varies across pixels (loaded per batch from `xs`).
    ///
    /// # Safety
    ///
    /// - `xs` and `output` must be properly aligned for `__m128` (16 bytes).
    /// - `output.len()` must be >= `xs.len()`.
    /// - The code must have been compiled by [`compile_arena_dag_scanline`].
    ///
    /// # Panics
    ///
    /// Panics if `output.len() < xs.len()`.
    ///
    /// [`compile_arena_dag_scanline`]: crate::backend::emit::compile_arena_dag_scanline
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn eval_scanline(
        &self,
        xs: &[core::arch::x86_64::__m128],
        y: core::arch::x86_64::__m128,
        z: core::arch::x86_64::__m128,
        w: core::arch::x86_64::__m128,
        output: &mut [core::arch::x86_64::__m128],
    ) {
        assert!(
            output.len() >= xs.len(),
            "ScanlineJitManifold::eval_scanline: output buffer too small \
             (have {}, need {})",
            output.len(),
            xs.len()
        );
        if xs.is_empty() {
            return;
        }
        let func: ScanlineKernelFn = unsafe { self.code.as_fn() };
        func(xs.as_ptr(), y, z, w, output.as_mut_ptr(), xs.len());
    }
}

// SAFETY: ExecutableCode is read-only mapped memory with no interior mutability.
unsafe impl Send for ScanlineJitManifold {}
unsafe impl Sync for ScanlineJitManifold {}
