//! JitManifold: a JIT-compiled function held as executable memory.
//!
//! This type owns an [`ExecutableCode`] and exposes it through a platform-specific
//! `call` method. It does NOT implement `pixelflow_core::Manifold` directly — that
//! would create a dependency cycle. Instead, `kernel_jit!` emits a thin wrapper in
//! the user's crate that calls through to `JitManifold`.

use crate::backend::emit::executable::{ExecutableCode, KernelFn};

/// A JIT-compiled kernel. Owns the executable code for one specific parameter
/// combination. No cache — caller decides lifetime.
pub struct JitManifold {
    code: ExecutableCode,
}

impl JitManifold {
    /// Wrap compiled executable code.
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
    #[inline(always)]
    pub unsafe fn call(
        &self,
        x: core::arch::aarch64::float32x4_t,
        y: core::arch::aarch64::float32x4_t,
        z: core::arch::aarch64::float32x4_t,
        w: core::arch::aarch64::float32x4_t,
    ) -> core::arch::aarch64::float32x4_t {
        let func: KernelFn = self.code.as_fn();
        func(x, y, z, w)
    }
}

#[cfg(target_arch = "x86_64")]
impl JitManifold {
    /// Evaluate the kernel at the given coordinates.
    ///
    /// # Safety
    ///
    /// The caller must ensure the SIMD types match the platform ABI that the
    /// emitter generated code for (x86-64 SSE2: `__m128`).
    #[inline(always)]
    pub unsafe fn call(
        &self,
        x: core::arch::x86_64::__m128,
        y: core::arch::x86_64::__m128,
        z: core::arch::x86_64::__m128,
        w: core::arch::x86_64::__m128,
    ) -> core::arch::x86_64::__m128 {
        let func: KernelFn = self.code.as_fn();
        func(x, y, z, w)
    }
}

// SAFETY: ExecutableCode is read-only mapped memory with no interior mutability.
unsafe impl Send for JitManifold {}
unsafe impl Sync for JitManifold {}
