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
        if code.is_empty() {
            return Err("empty code buffer");
        }

        let page_size = page_size();
        let capacity = (code.len() + page_size - 1) & !(page_size - 1);

        let mut pages = CodePages::map(capacity)?;
        pages.write(code);
        // Any `?` above unmapped the page through `Drop`; from here the
        // mapping's ownership moves into the returned `ExecutableCode`.
        pages.finish(code.len())
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
///
/// Asked of the machine rather than assumed from the OS: it is 4 KiB on
/// x86-64 and 16 KiB on Apple Silicon, which is a property of the hardware
/// the process is running on, not of the platform it was built for. This used
/// to hardcode 16384 whenever `target_os = "macos"`, which over-allocated on
/// Intel Macs.
#[cfg(unix)]
fn page_size() -> usize {
    // SAFETY: `sysconf` with a valid name has no preconditions.
    let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    // A negative return means the name was not recognised, which cannot
    // happen for `_SC_PAGESIZE`; fall back to the smallest size anything here
    // runs on rather than wrapping into a huge allocation.
    if n > 0 { n as usize } else { 4096 }
}

// =============================================================================
// Page preparation
// =============================================================================

/// A mapped, writable code page that is not yet executable.
///
/// The W^X transition happens exactly once, in [`CodePages::finish`], which is
/// the only way to reach an [`ExecutableCode`] — so "flip the permissions and
/// make the writes fetchable" cannot be forgotten or done twice. Anything that
/// leaves this scope without calling it (an early return, a `?`, a panic)
/// reaches `Drop`, which unmaps the page, so a writable mapping cannot leak.
/// The hand-written `munmap` on the old `mprotect` failure path is gone with
/// it, along with the obligation on every future error path to remember one.
#[cfg(unix)]
struct CodePages {
    ptr: *mut u8,
    capacity: usize,
}

#[cfg(unix)]
impl CodePages {
    /// Map `capacity` bytes readable and writable.
    fn map(capacity: usize) -> Result<Self, &'static str> {
        use libc::{MAP_ANON, MAP_PRIVATE, PROT_READ, PROT_WRITE, mmap};

        // SAFETY: a null hint with a non-zero length and no backing fd is the
        // ordinary anonymous-mapping call.
        let ptr = unsafe {
            mmap(
                ptr::null_mut(),
                capacity,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err("mmap failed");
        }
        Ok(Self {
            ptr: ptr.cast::<u8>(),
            capacity,
        })
    }

    /// Copy `code` to the start of the page.
    fn write(&mut self, code: &[u8]) {
        debug_assert!(code.len() <= self.capacity);
        // SAFETY: the mapping is writable and at least `code.len()` long.
        unsafe { ptr::copy_nonoverlapping(code.as_ptr(), self.ptr, code.len()) };
    }

    /// Flip to read-execute, make the written bytes visible to instruction
    /// fetch, and hand the mapping to an [`ExecutableCode`].
    fn finish(self, len: usize) -> Result<ExecutableCode, &'static str> {
        use libc::{PROT_EXEC, PROT_READ, mprotect};

        // SAFETY: `self` owns this mapping and `capacity` is its length.
        let rc = unsafe {
            mprotect(
                self.ptr.cast::<libc::c_void>(),
                self.capacity,
                PROT_READ | PROT_EXEC,
            )
        };
        if rc != 0 {
            // `self` is still live, so `Drop` unmaps on the way out.
            return Err("mprotect failed");
        }

        sync_instruction_cache(self.ptr, len);

        // The mapping now belongs to the returned value; suppress our `Drop`.
        let me = core::mem::ManuallyDrop::new(self);
        Ok(ExecutableCode {
            ptr: me.ptr,
            len,
            capacity: me.capacity,
        })
    }
}

#[cfg(unix)]
impl Drop for CodePages {
    fn drop(&mut self) {
        // SAFETY: we own this mapping and have not released it.
        unsafe { libc::munmap(self.ptr.cast::<libc::c_void>(), self.capacity) };
    }
}

// =============================================================================
// Instruction-cache coherence
// =============================================================================

/// Make the bytes written to `[ptr, ptr + len)` visible to instruction fetch.
///
/// This is an **architectural** property, not an OS one, and that distinction
/// is the whole point of the function. aarch64's instruction cache is not
/// coherent with its data cache, so code just written through a data store can
/// be stale — or invisible — to the fetch unit until the range is cleaned and
/// invalidated. x86-64's instruction cache is coherent and needs nothing.
///
/// It used to be gated on `target_os = "macos"`, with a comment attributing
/// the requirement to Apple Silicon. Apple Silicon is aarch64, so the code was
/// right about the machine and wrong about the axis: **Linux aarch64 skipped
/// the maintenance its architecture requires.** Nothing caught it, because the
/// only place aarch64 code is *executed* in CI is macOS — the aarch64 job only
/// type-checks — which is the same blind spot that produced the glyph-ink
/// regression and cost this crate a red CI run in #1055.
///
/// The risk is not theoretical: [`jit_cache`](crate::jit_cache) recompiles, so
/// mappings get recycled, and a stale I-cache line at a reused address is
/// exactly the failure this prevents.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn sync_instruction_cache(_ptr: *mut u8, _len: usize) {}

/// aarch64 on Apple platforms: libc exposes the maintenance sequence, and it
/// is the entry point Apple supports.
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[inline]
fn sync_instruction_cache(ptr: *mut u8, len: usize) {
    unsafe extern "C" {
        fn sys_icache_invalidate(start: *mut core::ffi::c_void, size: usize);
    }
    // SAFETY: the range was just written by this process and is mapped.
    unsafe { sys_icache_invalidate(ptr.cast::<core::ffi::c_void>(), len) };
}

/// aarch64 elsewhere (Linux): issue the maintenance ourselves.
///
/// The sequence is the architecturally specified one for self-modifying code:
/// clean the data cache to the point of unification so the stores are visible
/// to fetch, then invalidate any instruction-cache lines already holding the
/// same addresses, with the barriers that order the two against the
/// subsequent branch into the code.
///
/// Written as inline assembly rather than a call to `__clear_cache` on
/// purpose: an `extern` would only fail at link time, and nothing in CI links
/// for this target — the aarch64 job type-checks. Assembly is settled during
/// codegen, which `cargo build -p pixelflow-codegen --lib --target
/// aarch64-unknown-linux-gnu` does reach (an rlib needs no linker), so this
/// form is one the available checks can actually verify.
#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
fn sync_instruction_cache(ptr: *mut u8, len: usize) {
    use core::arch::asm;

    if len == 0 {
        return;
    }
    let start = ptr as usize;
    let end = start + len;

    // CTR_EL0 reports the minimum line sizes as log2 of a count of 4-byte
    // words: D-cache in bits [19:16], I-cache in bits [3:0].
    let ctr: u64;
    // SAFETY: CTR_EL0 is readable from EL0 and the read has no side effects.
    unsafe {
        asm!("mrs {ctr}, ctr_el0", ctr = out(reg) ctr, options(nomem, nostack, preserves_flags))
    };
    let dline = 4usize << ((ctr >> 16) & 0xF);
    let iline = 4usize << (ctr & 0xF);

    // Clean D-cache lines covering the range to the point of unification.
    let mut addr = start & !(dline - 1);
    while addr < end {
        // SAFETY: `dc cvau` is permitted at EL0 and only affects cache state.
        unsafe { asm!("dc cvau, {addr}", addr = in(reg) addr, options(nostack, preserves_flags)) };
        addr += dline;
    }
    // SAFETY: a barrier; no memory operands.
    unsafe { asm!("dsb ish", options(nostack, preserves_flags)) };

    // Invalidate I-cache lines covering the same range.
    let mut addr = start & !(iline - 1);
    while addr < end {
        // SAFETY: `ic ivau` is permitted at EL0 and only affects cache state.
        unsafe { asm!("ic ivau, {addr}", addr = in(reg) addr, options(nostack, preserves_flags)) };
        addr += iline;
    }
    // SAFETY: barriers ordering the maintenance before any later fetch.
    unsafe { asm!("dsb ish", "isb", options(nostack, preserves_flags)) };
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
/// Page preparation, independent of ISA level and architecture.
///
/// Ungated on purpose: `CodePages`, `page_size` and `sync_instruction_cache`
/// are exercised by every build, not just the SSE2 one the module below is
/// limited to.
#[cfg(all(test, unix))]
mod page_tests {
    use super::*;

    /// A single `ret` for the host, so the buffer really is valid machine code
    /// and `from_code`'s safety contract holds. Nothing here executes it.
    fn host_ret() -> Vec<u8> {
        #[cfg(target_arch = "x86_64")]
        {
            alloc::vec![0xC3]
        }
        #[cfg(target_arch = "aarch64")]
        {
            0xD65F_03C0u32.to_le_bytes().to_vec()
        }
    }

    /// Asked of the machine, not assumed from the OS. A bad `sysconf` fallback
    /// would show up here as a non-power-of-two or an absurd size.
    #[test]
    fn page_size_is_a_sane_power_of_two() {
        let n = page_size();
        assert!(n >= 4096, "page size {n} below the smallest we run on");
        assert!(n <= 1 << 20, "page size {n} implausibly large");
        assert!(n.is_power_of_two(), "page size {n} is not a power of two");
    }

    /// The whole `CodePages` path: map writable, write, flip to executable,
    /// sync the instruction cache. Reads the bytes back rather than running
    /// them, so it is meaningful on every host.
    #[test]
    fn code_survives_the_w_xor_x_flip() {
        let code = host_ret();
        // SAFETY: `code` is a single valid `ret` for this architecture.
        let exec = unsafe { ExecutableCode::from_code(&code) }.expect("map + flip");
        assert_eq!(exec.len(), code.len());
        assert!(!exec.is_empty());
        assert_eq!(
            exec.as_bytes(),
            code.as_slice(),
            "bytes changed across the flip"
        );
    }

    /// The mapping is rounded up to a whole page, so a one-byte kernel and a
    /// page-sized one both work and neither reports padding as code.
    #[test]
    fn length_reported_is_the_code_not_the_mapping() {
        let mut code = host_ret();
        let ret_len = code.len();
        code.resize(page_size() + ret_len, 0);
        code.rotate_right(ret_len); // keep the `ret` first
        // SAFETY: entry point is a valid `ret`; the padding is never executed.
        let exec = unsafe { ExecutableCode::from_code(&code) }.expect("map + flip");
        assert_eq!(exec.len(), code.len(), "len must be the code, not the page");
    }

    #[test]
    fn an_empty_buffer_is_refused() {
        // SAFETY: empty slice; rejected before anything is mapped.
        match unsafe { ExecutableCode::from_code(&[]) } {
            Err(e) => assert_eq!(e, "empty code buffer"),
            Ok(_) => panic!("an empty buffer must not map"),
        }
    }

    /// `sync_instruction_cache` must accept an empty range without touching
    /// memory — the aarch64 path computes a loop bound from it.
    #[test]
    fn syncing_an_empty_range_is_a_no_op() {
        let mut byte = 0u8;
        sync_instruction_cache(&raw mut byte, 0);
    }
}

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
