//! JitManifold: a JIT-compiled function held as executable memory.
//!
//! This type owns an [`ExecutableCode`] and exposes it through
//! [`call_collapse`](JitManifold::call_collapse) — the entry a collapse driver
//! uses to fill a whole tile with one call — plus the single-batch forms
//! [`call`](JitManifold::call) and [`call_bound`](JitManifold::call_bound) that
//! it is built from.
//!
//! There are no row/grid/point evaluators here. Tabulating a kernel over a
//! domain is `Lattice::bake`'s job in pixelflow-core, which owns the loop nest;
//! a second loop nest here would be a per-batch API competing with it.

use crate::JIT_VECTOR_BYTES;
use crate::emit::executable::{ExecutableCode, Point4, TileSlice};
use pixelflow_ir::LatticeShape;

const LANES: usize = JIT_VECTOR_BYTES / core::mem::size_of::<f32>();

/// A JIT-compiled kernel. Owns the executable memory for one specific parameter
/// combination. No cache — caller decides lifetime.
pub struct JitManifold {
    code: ExecutableCode,
    shape: LatticeShape,
}

/// Whether `tile` lies within the lattice a kernel was compiled for: no more
/// rows than the Y extent, and no more batches per row than the X extent
/// fills (`ceil(x / LANES)` — the last batch may be a partial one).
fn fits(shape: LatticeShape, tile: &TileSlice) -> bool {
    let [x, y, _, _] = shape.extent();
    tile.rows <= y as usize && tile.groups <= (x as usize).div_ceil(LANES)
}

impl JitManifold {
    /// Wrap newly compiled executable code into a `JitManifold` for a
    /// lattice of `shape`.
    ///
    /// The shape is the promise the code was compiled under: every tile
    /// handed to [`call_collapse`](Self::call_collapse) must lie within it.
    /// Today every shape's code accepts every tile, so the promise is checked
    /// in debug builds only; the loop-aware stages will emit code that keeps
    /// it by construction.
    #[must_use]
    pub const fn new(code: ExecutableCode, shape: LatticeShape) -> Self {
        Self { code, shape }
    }

    /// The lattice this kernel was compiled for.
    #[must_use]
    pub const fn shape(&self) -> LatticeShape {
        self.shape
    }

    /// The emitted machine code, for offline inspection (disassembly,
    /// profiler correlation). The bytes are the artifact, not an ABI.
    #[must_use]
    pub fn code_bytes(&self) -> &[u8] {
        self.code.as_bytes()
    }

    /// Evaluate the kernel over a single SIMD vector batch coordinate point `origin`.
    ///
    /// # Safety
    ///
    /// `V` must have `size_of::<V>() == JIT_VECTOR_BYTES`.
    #[must_use]
    #[inline(always)]
    pub unsafe fn call<V: Copy>(&self, origin: Point4<V>) -> V {
        assert_eq!(core::mem::size_of::<V>(), JIT_VECTOR_BYTES);
        let mut out = core::mem::MaybeUninit::<V>::uninit();
        // SAFETY: caller guarantees size_of::<V>() matches the JIT vector width.
        unsafe {
            self.call_collapse(
                core::ptr::null(),
                TileSlice::single(out.as_mut_ptr().cast::<f32>()),
                origin,
            );
            out.assume_init()
        }
    }

    /// Evaluate a bound-memory kernel for a single SIMD vector batch:
    /// `ctx` is the array of buffer base pointers passed in the first integer
    /// register.
    ///
    /// # Safety
    ///
    /// - `ctx` must hold valid pointers for every buffer declared by the arena.
    /// - `V` must have `size_of::<V>() == JIT_VECTOR_BYTES`.
    #[must_use]
    #[inline(always)]
    pub unsafe fn call_bound<V: Copy>(&self, ctx: *const *const f32, origin: Point4<V>) -> V {
        assert_eq!(core::mem::size_of::<V>(), JIT_VECTOR_BYTES);
        let mut out = core::mem::MaybeUninit::<V>::uninit();
        // SAFETY: caller guarantees valid ctx and size_of::<V>() matches JIT vector width.
        unsafe {
            self.call_collapse(
                ctx,
                TileSlice::single(out.as_mut_ptr().cast::<f32>()),
                origin,
            );
            out.assume_init()
        }
    }

    /// Run a 2D collapse kernel: one call fills `tile.rows * tile.groups` vector batches,
    /// resetting X and advancing Y by 1.0 after each row.
    ///
    /// # Safety
    ///
    /// - `tile.out` must point to writable memory with space for `tile.rows` stripes of
    ///   `tile.groups * LANES` floats plus `tile.row_skip_bytes` padding between rows.
    /// - `ctx` must hold valid base pointers for every buffer declared by the arena
    ///   (pass null/empty if the kernel has no buffers).
    /// - `size_of::<V>() == JIT_VECTOR_BYTES`.
    #[inline(always)]
    pub unsafe fn call_collapse<V: Copy>(
        &self,
        ctx: *const *const f32,
        tile: TileSlice,
        origin: Point4<V>,
    ) {
        debug_assert!(
            fits(self.shape, &tile),
            "a tile of {} batches × {} rows lies outside the {:?} this kernel was compiled for",
            tile.groups,
            tile.rows,
            self.shape
        );
        // SAFETY: Delegated to ExecutableCode which invokes the emitted KernelFn.
        unsafe { self.code.call_collapse(ctx, tile, origin) }
    }
}

// SAFETY: ExecutableCode is read-only mapped memory with no interior mutability.
unsafe impl Send for JitManifold {}
unsafe impl Sync for JitManifold {}
