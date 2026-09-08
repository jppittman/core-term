//! Linux executable code page implementation.
//!
//! Handles `mmap`, `mprotect`, and Linux ARM64 architectural instruction cache maintenance.

use core::ptr;
use libc::{
    _SC_PAGESIZE, MAP_ANON, MAP_PRIVATE, PROT_EXEC, PROT_READ, PROT_WRITE, mmap, mprotect, munmap,
    sysconf,
};

use super::{CodePage, ExecutableCode};
use crate::error::CompileError;

/// A mapped, writable code page on Linux.
pub struct LinuxCodePage {
    ptr: *mut u8,
    capacity: usize,
}

#[cfg(target_arch = "aarch64")]
fn sync_instruction_cache(ptr: *mut u8, len: usize) {
    use core::arch::asm;

    if len == 0 {
        return;
    }
    let start = ptr as usize;
    let end = start + len;

    let ctr: u64;
    unsafe {
        asm!("mrs {ctr}, ctr_el0", ctr = out(reg) ctr, options(nomem, nostack, preserves_flags))
    };
    let dline = 4usize << ((ctr >> 16) & 0xF);
    let iline = 4usize << (ctr & 0xF);

    let mut addr = start & !(dline - 1);
    while addr < end {
        unsafe { asm!("dc cvau, {addr}", addr = in(reg) addr, options(nostack, preserves_flags)) };
        addr += dline;
    }
    unsafe { asm!("dsb ish", options(nostack, preserves_flags)) };

    let mut addr = start & !(iline - 1);
    while addr < end {
        unsafe { asm!("ic ivau, {addr}", addr = in(reg) addr, options(nostack, preserves_flags)) };
        addr += iline;
    }
    unsafe { asm!("dsb ish", "isb", options(nostack, preserves_flags)) };
}

#[cfg(not(target_arch = "aarch64"))]
fn sync_instruction_cache(_ptr: *mut u8, _len: usize) {}

impl CodePage for LinuxCodePage {
    fn page_size() -> usize {
        let n = unsafe { sysconf(_SC_PAGESIZE) };
        if n > 0 { n as usize } else { 4096 }
    }

    fn map(capacity: usize) -> Result<Self, CompileError> {
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
            return Err(CompileError::Mmap);
        }
        Ok(Self {
            ptr: ptr.cast::<u8>(),
            capacity,
        })
    }

    fn write(&mut self, code: &[u8]) {
        debug_assert!(code.len() <= self.capacity);
        unsafe { ptr::copy_nonoverlapping(code.as_ptr(), self.ptr, code.len()) };
    }

    fn finish(self, len: usize) -> Result<ExecutableCode, CompileError> {
        let rc = unsafe {
            mprotect(
                self.ptr.cast::<libc::c_void>(),
                self.capacity,
                PROT_READ | PROT_EXEC,
            )
        };
        if rc != 0 {
            return Err(CompileError::Mprotect);
        }

        sync_instruction_cache(self.ptr, len);

        let me = core::mem::ManuallyDrop::new(self);
        Ok(ExecutableCode {
            ptr: me.ptr,
            len,
            capacity: me.capacity,
        })
    }
}

impl Drop for LinuxCodePage {
    fn drop(&mut self) {
        unsafe { munmap(self.ptr.cast::<libc::c_void>(), self.capacity) };
    }
}

#[cfg(test)]
pub(crate) fn test_sync_empty() {
    let mut byte = 0u8;
    sync_instruction_cache(&raw mut byte, 0);
}
