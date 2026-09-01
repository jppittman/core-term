//! JitManifold: a JIT-compiled function held as executable memory.
//!
//! This type owns an [`ExecutableCode`] and exposes it through safe evaluation
//! and low-level collapse execution methods.

use crate::JIT_VECTOR_BYTES;
use crate::emit::executable::{ExecutableCode, Extent2D, Point4, TileSlice};

const LANES: usize = JIT_VECTOR_BYTES / core::mem::size_of::<f32>();

/// A JIT-compiled kernel. Owns the executable memory for one specific parameter
/// combination. No cache — caller decides lifetime.
pub struct JitManifold {
    code: ExecutableCode,
}

impl JitManifold {
    /// Wrap newly compiled executable code into a `JitManifold`.
    #[must_use]
    pub const fn new(code: ExecutableCode) -> Self {
        Self { code }
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
        // SAFETY: Delegated to ExecutableCode which invokes the emitted CollapseKernelFn.
        unsafe { self.code.call_collapse(ctx, tile, origin) }
    }

    /// Safe evaluator for a row of pixels.
    ///
    /// Writes `out.len()` consecutive pixels starting at `origin`.
    pub fn eval_row(&self, out: &mut [f32], origin: Point4<f32>) {
        let full_groups = out.len() / LANES;
        let mut x0_vec = [0.0f32; LANES];
        for (i, lane) in x0_vec.iter_mut().enumerate() {
            *lane = origin.x + i as f32;
        }
        let coords = Point4::new(
            x0_vec,
            [origin.y; LANES],
            [origin.z; LANES],
            [origin.w; LANES],
        );

        if full_groups > 0 {
            // SAFETY: out has space for full_groups * LANES floats.
            unsafe {
                self.call_collapse(
                    core::ptr::null(),
                    TileSlice::contiguous(out.as_mut_ptr(), full_groups, 1),
                    coords,
                );
            }
        }

        let tail = out.len() % LANES;
        if tail > 0 {
            let offset = full_groups * LANES;
            for (i, lane) in x0_vec.iter_mut().enumerate() {
                *lane = origin.x + (offset + i) as f32;
            }
            let tail_coords = Point4::new(
                x0_vec,
                [origin.y; LANES],
                [origin.z; LANES],
                [origin.w; LANES],
            );
            let mut scratch = [0.0f32; LANES];
            // SAFETY: scratch buffer has space for 1 whole batch.
            unsafe {
                self.call_collapse(
                    core::ptr::null(),
                    TileSlice::single(scratch.as_mut_ptr()),
                    tail_coords,
                );
            }
            out[offset..].copy_from_slice(&scratch[..tail]);
        }
    }

    /// Safe evaluator for a 2D rectangular grid of pixels.
    ///
    /// Evaluates `extent.width * extent.height` pixels starting at `origin`.
    pub fn eval_grid(&self, out: &mut [f32], extent: Extent2D, origin: Point4<f32>) {
        assert!(
            out.len() >= extent.len(),
            "output buffer too small for grid"
        );
        let full_groups = extent.width / LANES;
        let tail = extent.width % LANES;

        let mut x0_vec = [0.0f32; LANES];
        for (i, lane) in x0_vec.iter_mut().enumerate() {
            *lane = origin.x + i as f32;
        }
        let coords = Point4::new(
            x0_vec,
            [origin.y; LANES],
            [origin.z; LANES],
            [origin.w; LANES],
        );

        if tail == 0 && full_groups > 0 {
            // Whole rows can be evaluated in a single 2D collapse call across all rows!
            // SAFETY: out has space for width * height floats without padding.
            unsafe {
                self.call_collapse(
                    core::ptr::null(),
                    TileSlice::contiguous(out.as_mut_ptr(), full_groups, extent.height),
                    coords,
                );
            }
        } else {
            // Rows with scalar tail: evaluate row by row
            for row in 0..extent.height {
                let row_y = origin.y + row as f32;
                let row_slice = &mut out[row * extent.width..(row + 1) * extent.width];
                self.eval_row(row_slice, Point4::new(origin.x, row_y, origin.z, origin.w));
            }
        }
    }
}

// SAFETY: ExecutableCode is read-only mapped memory with no interior mutability.
unsafe impl Send for JitManifold {}
unsafe impl Sync for JitManifold {}
