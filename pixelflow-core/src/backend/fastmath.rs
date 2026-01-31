//! Fast-math floating-point control for denormal handling.
//!
//! Denormals (subnormal numbers) are extremely slow on most CPUs - up to 100x slower
//! than normal floating-point operations. For real-time graphics, we treat them as zero.
//!
//! This module provides a RAII guard to enable flush-to-zero (FTZ) and denormals-are-zero (DAZ)
//! modes for the duration of a scope.

/// RAII guard for fast-math floating-point mode.
///
/// Enables flush-to-zero (FTZ) and denormals-are-zero (DAZ) on construction,
/// restores the original floating-point control state on drop.
///
/// # Platform Behavior
///
/// - **x86_64**: Sets both FTZ (bit 15) and DAZ (bit 6) in MXCSR
/// - **aarch64**: Sets FZ (bit 24) in FPCR
/// - **Other**: No-op (denormals handled normally)
///
/// # Example
///
/// ```ignore
/// use pixelflow_core::backend::FastMathGuard;
///
/// {
///     let _guard = unsafe { FastMathGuard::new() };
///     // Fast-math mode active: denormals treated as zero
///     // Rendering code here runs 2-100x faster on denormal-heavy workloads
/// } // _guard dropped, original FP mode restored
/// ```
///
/// # Safety
///
/// This modifies global CPU state (MXCSR or FPCR). It's safe in the sense that
/// it restores the original state on drop, but you must ensure:
///
/// - No other threads expect the default denormal behavior during the guard's lifetime
/// - You're okay with denormals being treated as zero (acceptable for graphics, not for
///   numerical algorithms requiring full IEEE 754 compliance)
pub struct FastMathGuard {
    #[cfg(target_arch = "x86_64")]
    old_mxcsr: u32,
    #[cfg(target_arch = "aarch64")]
    old_fpcr: u64,
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    _phantom: (),
}

impl FastMathGuard {
    /// Enable fast-math mode (FTZ/DAZ).
    ///
    /// # Safety
    ///
    /// Modifies global CPU floating-point control state. See [`FastMathGuard`] docs.
    #[inline]
    #[must_use]
    pub unsafe fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: Caller guarantees it's safe to modify FP control state
            let old_mxcsr = unsafe { set_fp_fast_mode_x86() };
            Self { old_mxcsr }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: Caller guarantees it's safe to modify FP control state
            let old_fpcr = unsafe { set_fp_fast_mode_arm() };
            Self { old_fpcr }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self { _phantom: () }
        }
    }
}

impl Drop for FastMathGuard {
    #[inline]
    fn drop(&mut self) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            restore_mxcsr(self.old_mxcsr);
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            restore_fpcr(self.old_fpcr);
        }
    }
}

// ============================================================================
// x86_64 Implementation
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn set_fp_fast_mode_x86() -> u32 {
    // Bit 15: Flush to Zero (FTZ) - output denormals become zero
    // Bit 6:  Denormals Are Zero (DAZ) - input denormals treated as zero
    // 0x8040 = (1 << 15) | (1 << 6)
    let mut mxcsr: u32 = 0;
    // SAFETY: Reading/writing MXCSR is safe, caller ensures global FP state can be modified
    unsafe {
        core::arch::asm!(
            "stmxcsr [{tmp}]",           // Store current MXCSR
            "mov {old:e}, [{tmp}]",      // Save old value to return
            "or dword ptr [{tmp}], 0x8040", // Set FTZ and DAZ bits
            "ldmxcsr [{tmp}]",           // Load updated MXCSR
            tmp = in(reg) &mut mxcsr,
            old = out(reg) mxcsr,
            options(nostack),
        );
    }
    mxcsr
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn restore_mxcsr(old_mxcsr: u32) {
    // SAFETY: Writing MXCSR is safe, restoring previous state
    unsafe {
        core::arch::asm!(
            "ldmxcsr [{tmp}]",
            tmp = in(reg) &old_mxcsr,
            options(nostack, readonly),
        );
    }
}

// ============================================================================
// ARM AArch64 Implementation
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn set_fp_fast_mode_arm() -> u64 {
    // Bit 24: Flush-to-Zero (FZ) mode
    // ARM's FZ bit handles both input and output denormals
    let old_fpcr: u64;
    // SAFETY: Reading/writing FPCR is safe, caller ensures global FP state can be modified
    unsafe {
        core::arch::asm!(
            "mrs {old}, fpcr",           // Read Floating-point Control Register
            "orr {new}, {old}, #(1 << 24)", // Set bit 24 (FZ)
            "msr fpcr, {new}",           // Write back to FPCR
            old = out(reg) old_fpcr,
            new = out(reg) _,
            options(nomem, nostack),
        );
    }
    old_fpcr
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn restore_fpcr(old_fpcr: u64) {
    // SAFETY: Writing FPCR is safe, restoring previous state
    unsafe {
        core::arch::asm!(
            "msr fpcr, {old}",
            old = in(reg) old_fpcr,
            options(nomem, nostack),
        );
    }
}
