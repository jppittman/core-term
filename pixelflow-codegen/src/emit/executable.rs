//! Executable memory management for JIT.
//!
//! This module handles the mmap/mprotect dance to create executable code at runtime.

use core::ptr;

/// A region of executable memory containing JIT-compiled code.
///
/// The memory is allocated as read-write, code is written to it,
/// then it's flipped to read-execute (W^X security).
pub struct ExecutableCode {
    ptr: *mut u8,
    len: usize,
    capacity: usize,
}

// SAFETY: The code is immutable after compilation and can be shared across threads.
unsafe impl Send for ExecutableCode {}
unsafe impl Sync for ExecutableCode {}

impl ExecutableCode {
    /// Compile a code buffer into executable memory.
    ///
    /// # Safety
    /// The caller must ensure the code buffer contains valid machine code
    /// for the current architecture.
    #[cfg(unix)]
    pub unsafe fn from_code(code: &[u8]) -> Result<Self, &'static str> {
        use libc::{MAP_ANON, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE, mmap, mprotect};

        if code.is_empty() {
            return Err("empty code buffer");
        }

        // Round up to page size (usually 4KB, but 16KB on Apple Silicon)
        let page_size = page_size();
        let capacity = (code.len() + page_size - 1) & !(page_size - 1);

        // SAFETY: All syscalls below are safe given valid arguments (which we ensure).
        unsafe {
            // 1. Allocate read-write memory
            let ptr = mmap(
                ptr::null_mut(),
                capacity,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            );

            if ptr == libc::MAP_FAILED {
                return Err("mmap failed");
            }

            let ptr = ptr as *mut u8;

            // 2. Copy code into the buffer
            ptr::copy_nonoverlapping(code.as_ptr(), ptr, code.len());

            // 3. Flip to read-execute (W^X)
            let result = mprotect(ptr as *mut libc::c_void, capacity, PROT_READ | PROT_EXEC);

            if result != 0 {
                libc::munmap(ptr as *mut libc::c_void, capacity);
                return Err("mprotect failed");
            }

            // Instruction cache coherence on Apple Silicon: the I-cache is not
            // automatically coherent with the D-cache, so code written via the
            // store above can otherwise be invisible (or stale) to the fetch
            // unit until explicitly invalidated.
            #[cfg(target_os = "macos")]
            {
                unsafe extern "C" {
                    fn sys_icache_invalidate(start: *mut core::ffi::c_void, size: usize);
                }
                sys_icache_invalidate(ptr as *mut core::ffi::c_void, code.len());
            }

            Ok(Self {
                ptr,
                len: code.len(),
                capacity,
            })
        }
    }

    /// Get a function pointer to the compiled code.
    ///
    /// # Safety
    /// The caller must ensure the code implements the correct calling convention
    /// and signature for type `F`.
    #[inline]
    #[must_use]
    pub unsafe fn as_fn<F>(&self) -> F {
        // SAFETY: Caller guarantees F matches the compiled code's signature.
        unsafe { core::mem::transmute_copy(&self.ptr) }
    }

    /// Get the code as a byte slice (for debugging).
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Length of the compiled code in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the code is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for ExecutableCode {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            let rc = libc::munmap(self.ptr as *mut libc::c_void, self.capacity);
            // A failing `munmap` means the mapping this type believed it owned
            // was not the mapping the kernel had — a leak at best. Nothing
            // useful can be done about it while unwinding, but it must not
            // pass unremarked.
            debug_assert_eq!(
                rc, 0,
                "munmap failed for {:?} ({} bytes)",
                self.ptr, self.capacity
            );
        }
    }
}

/// Get the system page size.
#[cfg(unix)]
fn page_size() -> usize {
    // Apple Silicon uses 16KB pages
    #[cfg(target_os = "macos")]
    {
        16384
    }
    #[cfg(not(target_os = "macos"))]
    unsafe {
        libc::sysconf(libc::_SC_PAGESIZE) as usize
    }
}

// =============================================================================
// Kernel type aliases for JIT-compiled functions
// =============================================================================

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::float32x4_t;

// __m128 is only the kernel width on the SSE2 build: the deleted scanline
// ABI was the last 128-bit-at-every-width user.
#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    not(target_feature = "avx2")
))]
use core::arch::x86_64::__m128;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
use core::arch::x86_64::__m256;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use core::arch::x86_64::__m512;

/// JIT-compiled kernel signature for ARM64.
/// Args: X in v0, Y in v1, Z in v2, W in v3; returns result in v0.
/// Each arg/result is a SIMD vector, so one call computes one pixel per lane
/// (4 pixels for a 128-bit vector), not a single pixel; the caller loops.
#[allow(improper_ctypes_definitions)]
#[cfg(target_arch = "aarch64")]
pub type KernelFn =
    extern "C" fn(float32x4_t, float32x4_t, float32x4_t, float32x4_t) -> float32x4_t;

/// JIT-compiled kernel that reads bound memory (ARM64).
///
/// Identical to [`KernelFn`] plus a leading context pointer: an array of buffer
/// base pointers, one per declared [`BufferId`](pixelflow_ir::arena::BufferId) in slot
/// order. AAPCS64 places this integer-class pointer in `x0`, disjoint from the
/// coordinate vectors in `v0..3`, so the emitted body is byte-for-byte the same
/// as a `KernelFn` — only kernels containing a `Gather` read `x0`. The caller
/// picks this type iff the arena declared buffers.
#[allow(improper_ctypes_definitions)]
#[cfg(target_arch = "aarch64")]
pub type CtxKernelFn = extern "C" fn(
    *const *const f32,
    float32x4_t,
    float32x4_t,
    float32x4_t,
    float32x4_t,
) -> float32x4_t;

/// JIT-compiled *collapse* kernel (ARM64). See the AVX-512
/// [`CollapseKernelFn`] doc for the contract. AAPCS64 argument registers:
/// `x0` = context, `x1` = `out`, `x2` = `groups`, `x3` = `rows`, `x4` =
/// `row_skip_bytes`, `v0..3` = x0/y0/z/w. Batch width is 4 lanes; the
/// per-iteration X step is 4.0 and Y advances by 1.0 per row.
#[allow(improper_ctypes_definitions)]
#[cfg(target_arch = "aarch64")]
pub type CollapseKernelFn = extern "C" fn(
    *const *const f32,
    *mut f32,
    usize,
    usize,
    usize,
    float32x4_t,
    float32x4_t,
    float32x4_t,
    float32x4_t,
);

/// JIT-compiled per-batch kernel signature for x86-64.
///
/// Args: X/Y/Z/W in the first four vector registers; returns the result in the
/// first. One call computes one pixel per lane; the caller loops.
///
/// The width tracks the build's selected SIMD: 512-bit `__m512` (16 lanes) with
/// AVX-512, 256-bit `__m256` (8 lanes) with AVX2 (and not AVX-512), else
/// 128-bit `__m128` (4 lanes, SSE2). This MUST match `pixelflow-core`'s
/// `Field`; the `kernel_jit!` wrapper const-asserts
/// `size_of::<Field>() == JIT_VECTOR_BYTES`. The looping variant is
/// [`CollapseKernelFn`], at the same width.
#[allow(improper_ctypes_definitions)]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub type KernelFn = extern "C" fn(__m512, __m512, __m512, __m512) -> __m512;

/// JIT-compiled kernel that reads bound memory (x86-64 AVX-512).
///
/// Identical to [`KernelFn`] plus a leading context pointer: an array of buffer
/// base pointers, one per declared [`BufferId`](pixelflow_ir::arena::BufferId) in slot
/// order. System V places this integer-class pointer in `rdi`, disjoint from the
/// coordinate vectors in `zmm0..3`, so the emitted body is byte-for-byte the
/// same as a `KernelFn` — only kernels containing a `Gather` read `rdi`. The
/// caller picks this type iff the arena declared buffers.
#[allow(improper_ctypes_definitions)]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub type CtxKernelFn = extern "C" fn(*const *const f32, __m512, __m512, __m512, __m512) -> __m512;

/// JIT-compiled *collapse* kernel (x86-64 AVX-512): the whole 2D loop nest is
/// inside the emitted code, so one call fills `rows * groups` output batches —
/// no per-row or per-batch Rust↔JIT boundary. X is an induction value reset at
/// each row: the kernel starts
/// from the caller's lane-sequential `x0` and adds the batch width (16.0)
/// per group; Y starts at `y0` and advances by 1.0 per row; Z/W are invariant.
///
/// SysV argument registers: `rdi` = context (array of buffer base pointers,
/// one per declared buffer — pass a dangling-free null-less array or anything
/// when the arena declares none; a buffer-free kernel never reads it),
/// `rsi` = `out` (output, written 64 bytes at a time; must hold at least
/// `groups * 16` f32s per row), `rdx` = `groups`, `rcx` = `rows`, `r8` =
/// bytes to skip after each row's final full group, `zmm0..3` = x0/y0/z/w.
#[allow(improper_ctypes_definitions)]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub type CollapseKernelFn =
    extern "C" fn(*const *const f32, *mut f32, usize, usize, usize, __m512, __m512, __m512, __m512);

/// JIT-compiled per-batch kernel signature for x86-64 AVX2 (256-bit, 8 lanes).
/// See [`KernelFn`] (the AVX-512 variant) for the ABI contract; this is the
/// same shape at the AVX2 width, selected when the build has `avx2` but not
/// `avx512f`.
#[allow(improper_ctypes_definitions)]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub type KernelFn = extern "C" fn(__m256, __m256, __m256, __m256) -> __m256;

/// JIT-compiled kernel that reads bound memory (x86-64 AVX2). See the
/// AVX-512 [`CtxKernelFn`] doc for the context-pointer contract; `rdi` is
/// still disjoint from the coordinate vectors (now `ymm0..3`).
#[allow(improper_ctypes_definitions)]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub type CtxKernelFn = extern "C" fn(*const *const f32, __m256, __m256, __m256, __m256) -> __m256;

/// JIT-compiled *collapse* kernel (x86-64 AVX2). See the AVX-512
/// [`CollapseKernelFn`] doc for the contract; batch width is 8 lanes and the
/// per-iteration X step is 8.0.
#[allow(improper_ctypes_definitions)]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub type CollapseKernelFn =
    extern "C" fn(*const *const f32, *mut f32, usize, usize, usize, __m256, __m256, __m256, __m256);

#[allow(improper_ctypes_definitions)]
#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    not(target_feature = "avx2")
))]
pub type KernelFn = extern "C" fn(__m128, __m128, __m128, __m128) -> __m128;

/// JIT-compiled kernel that reads bound memory (x86-64, 128-bit).
///
/// Identical to [`KernelFn`] plus a leading context pointer: an array of buffer
/// base pointers, one per declared [`BufferId`](pixelflow_ir::arena::BufferId) in slot
/// order. System V places this integer-class pointer in `rdi`, disjoint from the
/// coordinate vectors in `xmm0..3`, so the emitted body is byte-for-byte the
/// same as a `KernelFn` — only kernels containing a `Gather` read `rdi`. The
/// caller picks this type iff the arena declared buffers.
#[allow(improper_ctypes_definitions)]
#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    not(target_feature = "avx2")
))]
pub type CtxKernelFn = extern "C" fn(*const *const f32, __m128, __m128, __m128, __m128) -> __m128;

/// JIT-compiled *collapse* kernel (x86-64, 128-bit). See the AVX-512
/// [`CollapseKernelFn`] doc for the contract; batch width is 4 lanes and the
/// per-iteration X step is 4.0.
#[allow(improper_ctypes_definitions)]
#[cfg(all(
    target_arch = "x86_64",
    not(target_feature = "avx512f"),
    not(target_feature = "avx2")
))]
pub type CollapseKernelFn =
    extern "C" fn(*const *const f32, *mut f32, usize, usize, usize, __m128, __m128, __m128, __m128);

// =============================================================================
// Tests
// =============================================================================

// These tests hand-assemble SSE2 byte sequences and call them through the
// 128-bit `KernelFn`, so they are specific to the SSE2 (no AVX2, no AVX-512)
// ABI. Under `+avx512f` or `+avx2`, `KernelFn` is `__m512`/`__m256` and these
// `__m128` call sites don't type check; those paths are covered by the
// `avx512`/`avx2` tests in `mod.rs`.
#[cfg(all(test, not(target_feature = "avx512f"), not(target_feature = "avx2")))]
mod tests {
    // These tests hand-assemble instruction words as `base | Rd | (Rn << 5) | ...`;
    // the `| 0` / `(0 << 5)` terms document zero register fields on purpose.
    #![allow(clippy::identity_op)]

    use super::*;

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn jit_return_x() {
        // Simplest kernel: return X (already in v0)
        // Just RET - input X is already in v0, which is the return register!

        let mut code = Vec::new();

        // RET
        code.extend_from_slice(&0xD65F03C0u32.to_le_bytes());

        unsafe {
            let exec = ExecutableCode::from_code(&code).expect("failed to create executable");
            let func: KernelFn = exec.as_fn();

            use core::arch::aarch64::*;
            let x = vdupq_n_f32(42.0);
            let y = vdupq_n_f32(0.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let result = func(x, y, z, w);

            // Extract first lane
            let val = vgetq_lane_f32(result, 0);
            assert_eq!(val, 42.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn jit_add_xy() {
        // kernel: X + Y
        // v0 = X, v1 = Y, return v0 + v1

        let mut code = Vec::new();

        // FADD V0.4S, V0.4S, V1.4S
        // Encoding: 0x4E20D400 | Rd | (Rn << 5) | (Rm << 16)
        let fadd = 0x4E20D400u32 | 0 | (0 << 5) | (1 << 16);
        code.extend_from_slice(&fadd.to_le_bytes());

        // RET
        code.extend_from_slice(&0xD65F03C0u32.to_le_bytes());

        unsafe {
            let exec = ExecutableCode::from_code(&code).expect("failed to create executable");
            let func: KernelFn = exec.as_fn();

            use core::arch::aarch64::*;
            let x = vdupq_n_f32(10.0);
            let y = vdupq_n_f32(32.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let result = func(x, y, z, w);
            let val = vgetq_lane_f32(result, 0);
            assert_eq!(val, 42.0);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn jit_complex_expr() {
        // kernel: (X + Y) * Z
        // Uses register allocation:
        //   v0=X, v1=Y, v2=Z, v3=W
        //   v0 = v0 + v1  (X + Y)
        //   v0 = v0 * v2  ((X+Y) * Z)
        //   ret

        let mut code = Vec::new();

        // FADD V0.4S, V0.4S, V1.4S
        let fadd = 0x4E20D400u32 | 0 | (0 << 5) | (1 << 16);
        code.extend_from_slice(&fadd.to_le_bytes());

        // FMUL V0.4S, V0.4S, V2.4S
        let fmul = 0x6E20DC00u32 | 0 | (0 << 5) | (2 << 16);
        code.extend_from_slice(&fmul.to_le_bytes());

        // RET
        code.extend_from_slice(&0xD65F03C0u32.to_le_bytes());

        unsafe {
            let exec = ExecutableCode::from_code(&code).expect("failed to create executable");
            let func: KernelFn = exec.as_fn();

            use core::arch::aarch64::*;
            let x = vdupq_n_f32(2.0);
            let y = vdupq_n_f32(5.0);
            let z = vdupq_n_f32(6.0);
            let w = vdupq_n_f32(0.0);

            let result = func(x, y, z, w);
            let val = vgetq_lane_f32(result, 0);
            assert_eq!(val, 42.0); // (2 + 5) * 6 = 42
        }
    }

    // =========================================================================
    // Integration tests using the compile() API
    // =========================================================================

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn jit_const_05_raw() {
        // Test raw constant loading for 0.5
        // MOVZ W16, #0
        // MOVK W16, #0x3F00, LSL #16  (0x3F000000 = 0.5f)
        // DUP V0.4S, W16
        // RET

        let mut code = Vec::new();

        // MOVZ W16, #0  (lo16 = 0)
        code.extend_from_slice(&0x52800010u32.to_le_bytes());

        // MOVK W16, #0x3F00, LSL #16
        code.extend_from_slice(&(0x72A00010u32 | (0x3F00 << 5)).to_le_bytes());

        // DUP V0.4S, W16
        code.extend_from_slice(&(0x4E040C00u32 | (16 << 5) | 0).to_le_bytes());

        // RET
        code.extend_from_slice(&0xD65F03C0u32.to_le_bytes());

        unsafe {
            let exec = ExecutableCode::from_code(&code).expect("failed to create executable");
            let func: KernelFn = exec.as_fn();

            use core::arch::aarch64::*;
            let x = vdupq_n_f32(0.0);
            let y = vdupq_n_f32(0.0);
            let z = vdupq_n_f32(0.0);
            let w = vdupq_n_f32(0.0);

            let result = func(x, y, z, w);
            assert_eq!(vgetq_lane_f32(result, 0), 0.5);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn jit_return_x_x86() {
        // Simplest kernel: return X (already in xmm0)

        let mut code = Vec::new();

        // RET
        code.push(0xC3);

        unsafe {
            let exec = ExecutableCode::from_code(&code).expect("failed to create executable");
            let func: KernelFn = exec.as_fn();

            use core::arch::x86_64::*;
            let x = _mm_set1_ps(42.0);
            let y = _mm_setzero_ps();
            let z = _mm_setzero_ps();
            let w = _mm_setzero_ps();

            let result = func(x, y, z, w);
            let val = _mm_cvtss_f32(result);
            assert_eq!(val, 42.0);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn jit_add_xy_x86() {
        // kernel: X + Y

        let mut code = Vec::new();

        // ADDPS xmm0, xmm1
        code.extend_from_slice(&[0x0F, 0x58, 0xC1]);

        // RET
        code.push(0xC3);

        unsafe {
            let exec = ExecutableCode::from_code(&code).expect("failed to create executable");
            let func: KernelFn = exec.as_fn();

            use core::arch::x86_64::*;
            let x = _mm_set1_ps(10.0);
            let y = _mm_set1_ps(32.0);
            let z = _mm_setzero_ps();
            let w = _mm_setzero_ps();

            let result = func(x, y, z, w);
            let val = _mm_cvtss_f32(result);
            assert_eq!(val, 42.0);
        }
    }
}
