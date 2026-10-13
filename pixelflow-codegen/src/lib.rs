//! Expression graphs to machine code.
//!
//! Everything here is downstream of the language: `pixelflow-ir` defines what
//! an expression *is*, `pixelflow-search` decides what it *should be*, and this
//! crate turns the result into instructions.
//!
//! That ordering is the point. The compile entries in [`jit_cache`] run the
//! optimizer themselves, so there is no way to ask for a compiled kernel and
//! accidentally get an unoptimized one.
//!
//! This is also where every OS dependency lives. Executable memory needs
//! `mmap`/`mprotect`; the language does not link `libc`.

extern crate alloc;

// The whole public surface: the cached, linked compile a lattice runs
// (`jit_cache`), the emitter a measurement harness drives directly (`emit`),
// the tier this process emits for (`isa`) and the vector width it gives, why
// a compile fails, and the digest the pipeline's identities and the byte
// probes share.
pub mod emit;
pub mod isa;
// x86-64 and aarch64 are the architectures with emitters.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod jit_cache;
pub use error::CompileError;

/// Byte width of the vector the JIT emits for on this host: 64 (AVX-512), 32
/// (AVX2) or 16 (NEON). The register width of the backend [`isa::detect`]
/// selects, and the one width in the workspace — `pixelflow-core` asks for it
/// here rather than carrying a vector type of its own to assert against.
#[must_use]
pub fn jit_vector_bytes() -> usize {
    emit::vector_bytes(isa::detect())
}

mod error;
mod pass;
mod pipeline;
mod program;

/// FNV-1a's 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a's 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a over bytes: the one definition `pixelflow-pipeline`'s content
/// identities are built from (`schema::fnv1a64_const` is this function), and
/// the digest the byte probes print, so two runs diffed line by
/// line say whether a change moved any code.
///
/// Here because this is the lowest crate the pipeline and every probe that
/// digests emitted code already depend on; the only property it needs is that
/// different bytes give different digests, and a dependency for that would be
/// silly. A `const fn` so a schema identity can be derived at compile time.
#[must_use]
pub const fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    // `while`, not an iterator: iterators are not callable in a `const fn`.
    let mut i = 0;
    while i < bytes.len() {
        hash = (hash ^ bytes[i] as u64).wrapping_mul(FNV_PRIME);
        i += 1;
    }
    hash
}
