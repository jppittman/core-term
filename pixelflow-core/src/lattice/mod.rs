//! # Lattice: Representable Functor for Kernel Evaluation
//!
//! A Lattice is a finite box domain that tabulates a [`Kernel`] into a
//! discrete buffer. This is the `tabulate`/`index` pair from representable
//! functors:
//!
//! - **[`Lattice::bake`]** = `tabulate`: `(Rep -> a) -> F a` -- the kernel at every point
//! - **`DiscreteManifold::eval`** = `index`: `F a -> Rep -> a` -- read back by coordinate
//! - **Isomorphism**: `index(bake(k, domain), i) = k(coord(i))` (up to discretization)
//!
//! Nothing computes until a Lattice demands it. A single-point evaluation is
//! just `Lattice::point` -- the degenerate case with all coordinates fixed.
//!
//! There is one `Lattice` type, not one per shape. An axis with extent 1 is
//! fixed at its origin; an axis with extent > 1 is a loop dimension. The
//! constructors (`frame`, `scanline`, `point`, `index2`, `index`) are sugar
//! for common shapes -- the shape is data, not a type. Extents only need to
//! be static at JIT-compile time, which is when the kernel is specialized.
//!
//! A kernel with a lattice is the evaluation API: `bake` compiles ONE kernel
//! for the whole domain and the loop nest lives inside the emitted code, so
//! the compiler owns the hoisting and the register allocation across all of
//! it. Reductions are a binder in the kernel (`Kernel::over` and friends),
//! not a fold the lattice performs over a manifold.
//!
//! [`Kernel`]: pixelflow_ir::Kernel

use crate::numeric::Numeric;
use crate::{Field, Manifold, PARALLELISM};
use alloc::vec;
use alloc::vec::Vec;

// ============================================================================
// DiscreteManifold: The result of collapsing a lattice
// ============================================================================

/// The result of collapsing a Lattice. A buffer of values that IS a Manifold.
///
/// `index`: `F a -> Rep -> a` -- read by coordinate.
/// This closes the representable functor isomorphism:
/// `index(collapse(f)) = f` (up to discretization).
#[derive(Clone, Debug)]
pub struct DiscreteManifold {
    /// Raw value buffer, row-major (y * width + x).
    pub(crate) buffer: Vec<f32>,
    /// Width of the grid (X dimension).
    pub(crate) width: usize,
    /// Height of the grid (Y dimension).
    pub(crate) height: usize,
    /// Which memory this is, for merging composed arenas. Clones share it,
    /// which is sound because the buffer is write-once — there is no mutable
    /// accessor, so a clone can never diverge from its original.
    pub(crate) id: pixelflow_ir::arena::BufferIdentity,
}

impl DiscreteManifold {
    /// Create a DiscreteManifold from a pre-filled buffer.
    ///
    /// # Panics
    ///
    /// Panics if `buffer.len() != width * height`.
    #[must_use]
    pub fn new(buffer: Vec<f32>, width: usize, height: usize) -> Self {
        assert_eq!(
            buffer.len(),
            width * height,
            "DiscreteManifold buffer size {} does not match dimensions {}x{} = {}",
            buffer.len(),
            width,
            height,
            width * height,
        );
        Self {
            buffer,
            width,
            height,
            id: pixelflow_ir::arena::BufferIdentity::mint(),
        }
    }

    /// Width of the grid (X extent).
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Height of the grid (Y extent).
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Read-only access to the underlying buffer (row-major).
    #[must_use]
    pub fn buffer(&self) -> &[f32] {
        &self.buffer
    }

    /// Consume the DiscreteManifold and return the buffer.
    #[must_use]
    pub fn into_buffer(self) -> Vec<f32> {
        self.buffer
    }
}

// Mark as a ManifoldExpr so ManifoldExt methods (`.at()`, etc.) work on it.
impl crate::ext::ManifoldExpr for DiscreteManifold {}

/// `index`: read by coordinate. This IS the representable functor's index.
///
/// Given Field coordinates (x, y, z, w), converts x and y to integer indices
/// via nearest-neighbor (floor + clamp), looks up the buffer value, and
/// returns it as a Field.
impl Manifold<(Field, Field, Field, Field)> for DiscreteManifold {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, p: (Field, Field, Field, Field)) -> Field {
        let (x, y, _, _) = p;

        // If the buffer is empty, there's nothing to look up.
        // Fail loud: this is a programming error, not a recoverable condition.
        assert!(
            !self.buffer.is_empty(),
            "DiscreteManifold::eval called on empty buffer ({}x{})",
            self.width,
            self.height,
        );

        let zero = Field::from(0.0);
        let max_x = Field::from((self.width.saturating_sub(1)) as f32);
        let max_y = Field::from((self.height.saturating_sub(1)) as f32);

        // Nearest-neighbor: floor then clamp to valid range.
        let xi = x.floor().max(zero).min(max_x);
        let yi = y.floor().max(zero).min(max_y);

        // Linear index = floor(y) * width + floor(x)
        let w_field = Field::from(self.width as f32);
        let indices = yi.raw_mul(w_field).raw_add(xi);

        Field::gather(&self.buffer, indices)
    }
}

// ============================================================================
// Lattice: a finite box domain over the four coordinate axes
// ============================================================================

/// A finite box domain over the four coordinate axes (X, Y, Z, W).
///
/// `extent[i]` is the number of samples along axis `i`; an axis with extent 1
/// is fixed at `origin[i]`. `origin[i]` is the coordinate of index 0 on each
/// axis. Iteration is row-major with X innermost (SIMD lanes ride X).
///
/// The shape is data, not a type: a frame, a scanline, a point, and a tensor
/// index range are all the same `Lattice` with different extents. The JIT
/// specializes on the extents at kernel-compile time.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Lattice {
    /// Samples per axis: `[x, y, z, w]`. Extent 1 = fixed axis.
    pub extent: [u32; 4],
    /// Coordinate of index 0 on each axis.
    pub origin: [f32; 4],
}

impl Lattice {
    /// A 2D pixel frame: X varies per pixel, Y per scanline; Z is the fixed
    /// frame time, W is fixed at 0.
    #[must_use]
    pub fn frame(width: usize, height: usize, z: f32) -> Self {
        Self {
            extent: [width as u32, height as u32, 1, 1],
            origin: [0.0, 0.0, z, 0.0],
        }
    }

    /// A 1D scanline: only X varies; Y, Z, W are fixed.
    #[must_use]
    pub fn scanline(width: usize, y: f32, z: f32, w: f32) -> Self {
        Self {
            extent: [width as u32, 1, 1, 1],
            origin: [0.0, y, z, w],
        }
    }

    /// A single point: all coordinates fixed. The degenerate (0-loop) case.
    #[must_use]
    pub fn point(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self {
            extent: [1, 1, 1, 1],
            origin: [x, y, z, w],
        }
    }

    /// A 1D index range `[0, len)` over X. Feature indices, not pixels.
    #[must_use]
    pub fn index(len: usize) -> Self {
        Self {
            extent: [len as u32, 1, 1, 1],
            origin: [0.0; 4],
        }
    }

    /// A 2D index range `[0, width) x [0, height)` over X and Y.
    /// Weight-matrix indices: X = input dim, Y = output dim.
    #[must_use]
    pub fn index2(width: usize, height: usize) -> Self {
        Self {
            extent: [width as u32, height as u32, 1, 1],
            origin: [0.0; 4],
        }
    }

    // ───────────────────── domain queries ──────────────────────

    /// Number of points in this domain.
    #[must_use]
    pub fn len(&self) -> usize {
        self.extent.iter().map(|&e| e as usize).product()
    }

    /// Whether the domain is empty (any axis has extent 0).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bitmask of loop axes (extent > 1): bit 0 = X, 1 = Y, 2 = Z, 3 = W.
    /// Fixed axes are constants from the kernel's point of view.
    #[must_use]
    pub fn loop_mask(&self) -> u8 {
        let mut mask = 0u8;
        for (i, &e) in self.extent.iter().enumerate() {
            if e > 1 {
                mask |= 1 << i;
            }
        }
        mask
    }

    /// Map a linear index to concrete coordinate values (X fastest).
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    #[must_use]
    pub fn coord(&self, index: usize) -> (f32, f32, f32, f32) {
        assert!(
            index < self.len(),
            "Lattice::coord index {} out of bounds (len = {})",
            index,
            self.len(),
        );
        let [ex, ey, ez, _] = self.extent.map(|e| e as usize);
        let x = index % ex;
        let rest = index / ex;
        let y = rest % ey;
        let rest = rest / ey;
        let z = rest % ez;
        let w = rest / ez;
        (
            self.origin[0] + x as f32,
            self.origin[1] + y as f32,
            self.origin[2] + z as f32,
            self.origin[3] + w as f32,
        )
    }

    // ──────────────── collapse (the per-batch remnant) ───────────────

    /// Evaluate a manifold at every point in the domain by calling its
    /// `eval` once per SIMD batch. Returns a discrete manifold (buffer
    /// lookup) with `width = extent[0]` and `height` = the product of the
    /// remaining extents.
    ///
    /// **Not the evaluation API.** [`Lattice::bake`] is: it hands the
    /// compiler the kernel and the whole loop nest, where this makes one call
    /// per SIMD batch. Every caller that can bake, bakes.
    ///
    /// What keeps this alive is the glyph cache's `CachedGlyph`, whose tests
    /// tabulate it and which cannot be baked for two independent reasons:
    /// it has no [`Lower`](pixelflow_ir::Lower) or `Kernel` form (it is in
    /// the same "does not lower yet" class as `Color`, `spatial_bsp` and
    /// `scene3d`), and its coverage lives in a buffer while `bake` binds no
    /// memory and refuses a kernel that declares one. Writing a `Kernel`
    /// twin of its `eval` would move the tested definition off the one the
    /// rasterizer actually runs, so it waits for the stage that makes the
    /// combinators lower. No new caller.
    pub fn collapse<M>(&self, manifold: &M) -> DiscreteManifold
    where
        M: Manifold<(Field, Field, Field, Field), Output = Field>,
    {
        let [ex, ey, ez, ew] = self.extent.map(|e| e as usize);
        let mut buffer = vec![0.0f32; self.len()];
        let mut packed = [0.0f32; PARALLELISM];
        let step = Field::from(PARALLELISM as f32);

        let mut row = 0usize;
        for w in 0..ew {
            let w_field = Field::from(self.origin[3] + w as f32);
            for z in 0..ez {
                let z_field = Field::from(self.origin[2] + z as f32);
                for y in 0..ey {
                    let y_field = Field::from(self.origin[1] + y as f32);
                    let row_offset = row * ex;
                    let mut x = 0usize;
                    let mut x_field = Field::sequential(self.origin[0]);

                    // SIMD hot path: full batches of PARALLELISM points.
                    while x + PARALLELISM <= ex {
                        let result = manifold.eval((x_field, y_field, z_field, w_field));
                        result.store(&mut packed);
                        buffer[row_offset + x..row_offset + x + PARALLELISM]
                            .copy_from_slice(&packed);
                        x += PARALLELISM;
                        x_field = x_field.raw_add(step);
                    }

                    // SIMD tail: evaluate the last partial batch.
                    if x < ex {
                        let result = manifold.eval((x_field, y_field, z_field, w_field));
                        result.store(&mut packed);
                        let tail_len = ex - x;
                        buffer[row_offset + x..row_offset + ex]
                            .copy_from_slice(&packed[..tail_len]);
                    }
                    row += 1;
                }
            }
        }

        DiscreteManifold::new(buffer, ex, ey * ez * ew)
    }

    /// Bake a [`Kernel`](pixelflow_ir::Kernel) — the front-end value — over the
    /// domain: JIT-compile it once (through the global cache) and tabulate. The
    /// JIT-first path: no combinator manifold, no `Lower`, just the arena the
    /// `Kernel` already carries. Its `Dwrt` derivatives are resolved by the
    /// compiler during codegen. Falls back to nothing — a `Kernel` is always
    /// an arena, always compilable — except when this build's `Field` width is
    /// not the JIT's, where it panics rather than silently mis-tabulating.
    ///
    /// Runtime-composed kernels (`Kernel::over`/`.at()`/arithmetic) never run
    /// through the `kernel!`/`kernel_jit!` macros' e-graph saturation. They do
    /// not need to be optimized here either: the compile entry runs the
    /// optimizer itself, so there is no arena shape that reaches a backend
    /// unoptimized and no caller that has to remember to ask.
    ///
    /// The compiled form is the collapse kernel: the X/Y loop nest lives
    /// *inside* the emitted code, so each Z/W plane's full-width region is one
    /// call rather than one `extern "C"` call per row or SIMD batch. Scalar
    /// tails remain explicit one-batch calls into scratch storage.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[must_use]
    pub fn bake(&self, kernel: &pixelflow_ir::Kernel) -> DiscreteManifold {
        assert_eq!(
            core::mem::size_of::<Field>(),
            pixelflow_codegen::JIT_VECTOR_BYTES,
            "Lattice::bake: Field width does not match the JIT's emitted width"
        );
        let (arena, root) = kernel.parts();
        let jit = pixelflow_codegen::jit_cache::compile(
            arena,
            root,
            pixelflow_ir::LatticeShape::new(self.extent),
        )
        .expect("kernel failed to compile");

        let [ex, ey, ez, ew] = self.extent.map(|e| e as usize);
        let mut buffer = vec![0.0f32; self.len()];
        let full_groups = ex / PARALLELISM;
        let tail = ex % PARALLELISM;
        // Nothing here binds memory, so the context register must never be
        // read. A kernel composed with a sampler (`BilinearSampler::kernel`)
        // does declare buffers, and its gathers would load base pointers out
        // of this null — refuse it here rather than fault in emitted code.
        assert!(
            arena.buffers().is_empty(),
            "Lattice::bake: kernel declares {} buffer(s), but bake binds none. \
             Bake a buffer-free kernel, or evaluate the sampler through a path \
             that binds its memory.",
            arena.buffers().len()
        );
        let ctx = core::ptr::null();
        let x0 = Field::sequential(self.origin[0]);
        let x_tail = Field::sequential(self.origin[0] + (full_groups * PARALLELISM) as f32);

        let mut row = 0usize;
        for w in 0..ew {
            let w_field = Field::from(self.origin[3] + w as f32);
            for z in 0..ez {
                let z_field = Field::from(self.origin[2] + z as f32);
                let plane_offset = row * ex;
                if full_groups > 0 && ey > 0 {
                    let y0 = Field::from(self.origin[1]);
                    // SAFETY: bake checked size_of::<Field>() == the JIT's
                    // emitted width. The 2D collapse writes `full_groups`
                    // batches per row, then skips the scalar tail bytes before
                    // advancing Y; `ey` rows fit in this Z/W plane.
                    unsafe {
                        jit.call_collapse(
                            ctx,
                            pixelflow_codegen::TileSlice::new(
                                buffer[plane_offset..].as_mut_ptr(),
                                full_groups,
                                ey,
                                tail * core::mem::size_of::<f32>(),
                            ),
                            pixelflow_codegen::Point4::new(x0, y0, z_field, w_field),
                        );
                    }
                }
                for y in 0..ey {
                    let y_field = Field::from(self.origin[1] + y as f32);
                    let row_offset = row * ex;
                    if tail > 0 {
                        let mut scratch = [0.0f32; PARALLELISM];
                        // SAFETY: as above; scratch holds one whole batch.
                        unsafe {
                            jit.call_collapse(
                                ctx,
                                pixelflow_codegen::TileSlice::single(scratch.as_mut_ptr()),
                                pixelflow_codegen::Point4::new(x_tail, y_field, z_field, w_field),
                            );
                        }
                        buffer[row_offset + full_groups * PARALLELISM..row_offset + ex]
                            .copy_from_slice(&scratch[..tail]);
                    }
                    row += 1;
                }
            }
        }

        DiscreteManifold::new(buffer, ex, ey * ez * ew)
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod cell_grid;

#[cfg(test)]
mod tests;

// ============================================================================
// BilinearSampler: smooth read-back of a collapsed lattice
// ============================================================================

/// Bilinear read-back of a [`DiscreteManifold`]: the smooth companion to its
/// nearest-neighbor `eval`.
///
/// Where the discrete manifold snaps a continuous coordinate to the containing
/// lattice cell, the sampler reads the four surrounding integer-grid texels
/// and blends them with the fractional coordinate weights. The blend is a
/// bound-memory kernel — four `Gather`s plus the weight arithmetic in one
/// arena — JIT-compiled once at construction and bound to the buffer at each
/// call, so the read-back goes through the same backend as everything else
/// rather than through a combinator tree.
///
/// # Coordinate convention
///
/// Integer-grid space: a query at `(i + fx, j + fy)` with `fx, fy ∈ [0, 1)`
/// blends the texels at `(i, j)`, `(i+1, j)`, `(i, j+1)`, `(i+1, j+1)`.
/// Consequences:
///
/// - At exact integer coordinates the stored texel is returned untouched
///   (the fractional weights are zero).
/// - A buffer holding samples of a function affine in x and y is reproduced
///   exactly everywhere.
/// - Out-of-range taps clamp to the edge texel, exactly as
///   `DiscreteManifold::eval` does — `Gather`'s reference semantics.
/// - No half-pixel convention is baked in. Callers that store samples at
///   texel *centers* must shift coordinates by −0.5 before sampling.
///
/// Z and W pass through unchanged; interpolation is 2D over X/Y only.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[derive(Clone)]
pub struct BilinearSampler {
    tex: DiscreteManifold,
    jit: alloc::sync::Arc<pixelflow_codegen::JitManifold>,
}

/// The 4-tap bilinear blend over one declared buffer, as an arena fragment:
/// `Σ tap(x?, y?) · weight` with `x0 = floor(X)`, `fx = X − x0`, and the
/// mirrored pair in y. Gather clamps each tap to the buffer edge.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn bilinear_arena(
    id: pixelflow_ir::arena::BufferIdentity,
    width: u32,
    height: u32,
) -> (pixelflow_ir::ExprArena, pixelflow_ir::ExprId) {
    use pixelflow_ir::arena::BufferDecl;
    use pixelflow_ir::{ExprArena, OpKind};

    let mut a = ExprArena::new();
    let buf = a.declare_buffer(BufferDecl { id, width, height });
    let x = a.push_var(0);
    let y = a.push_var(1);
    let one = a.push_const(1.0);

    let x0 = a.push_unary(OpKind::Floor, x);
    let y0 = a.push_unary(OpKind::Floor, y);
    let x1 = a.push_binary(OpKind::Add, x0, one);
    let y1 = a.push_binary(OpKind::Add, y0, one);
    let fx = a.push_binary(OpKind::Sub, x, x0);
    let fy = a.push_binary(OpKind::Sub, y, y0);
    let gx = a.push_binary(OpKind::Sub, one, fx);
    let gy = a.push_binary(OpKind::Sub, one, fy);

    let c00 = a.push_gather(buf, x0, y0);
    let c10 = a.push_gather(buf, x1, y0);
    let c01 = a.push_gather(buf, x0, y1);
    let c11 = a.push_gather(buf, x1, y1);

    let w00 = a.push_binary(OpKind::Mul, gx, gy);
    let w10 = a.push_binary(OpKind::Mul, fx, gy);
    let w01 = a.push_binary(OpKind::Mul, gx, fy);
    let w11 = a.push_binary(OpKind::Mul, fx, fy);

    let t00 = a.push_binary(OpKind::Mul, c00, w00);
    let t10 = a.push_binary(OpKind::Mul, c10, w10);
    let t01 = a.push_binary(OpKind::Mul, c01, w01);
    let t11 = a.push_binary(OpKind::Mul, c11, w11);

    let s0 = a.push_binary(OpKind::Add, t00, t10);
    let s1 = a.push_binary(OpKind::Add, s0, t01);
    let root = a.push_binary(OpKind::Add, s1, t11);
    (a, root)
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl BilinearSampler {
    /// The buffer this sampler reads (row-major f32).
    #[must_use]
    pub fn texture(&self) -> &DiscreteManifold {
        &self.tex
    }

    /// The 4-tap blend over a buffer of these extents, as a composable
    /// fragment — read it at *computed* coordinates inside a larger kernel
    /// with `.at(&u, &v, &z, &w)` instead of only at the caller's own.
    ///
    /// Takes extents rather than a sampler because extents are all the IR
    /// carries; the data binds later. So a program can compose against a
    /// buffer shape it has not filled yet, which is what lets a grid compile
    /// from its geometry alone. `bilinear` compiles from this same builder,
    /// so the composed and called forms cannot drift apart.
    #[must_use]
    pub fn kernel_for(
        id: pixelflow_ir::arena::BufferIdentity,
        width: u32,
        height: u32,
    ) -> pixelflow_ir::Kernel {
        // Same guard as `DiscreteManifold::kernel_for`, for the same reason:
        // `BindingTable::bind` accepts an empty slice against an empty
        // declaration, and gather lowering's `saturating_sub(1)` then clamps
        // every tap to index 0 — so an empty extent reaches the JIT and
        // dereferences a zero-length buffer instead of failing here.
        assert!(
            width > 0 && height > 0,
            "BilinearSampler::kernel_for: empty buffer ({width}x{height})"
        );
        let (arena, root) = bilinear_arena(id, width, height);
        pixelflow_ir::Kernel::from_parts(arena, root)
    }

    /// This sampler's blend as a composable fragment. See [`Self::kernel_for`].
    ///
    /// # Panics
    ///
    /// Panics when an extent exceeds `u32`.
    #[must_use]
    pub fn kernel(&self) -> pixelflow_ir::Kernel {
        Self::kernel_for(
            self.tex.id,
            u32::try_from(self.tex.width).expect("buffer width exceeds u32"),
            u32::try_from(self.tex.height).expect("buffer height exceeds u32"),
        )
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl DiscreteManifold {
    /// A nearest-neighbour read of a buffer of these extents, as a composable
    /// fragment — `.at(&idx, &row, &z, &w)` reads at *computed* indices.
    ///
    /// One `Gather`, which already carries [`Self::eval`]'s semantics: floor,
    /// clamp to the declared extents, index row-major. Takes extents rather
    /// than `&self` for the same reason as [`BilinearSampler::kernel_for`] —
    /// extents are what the IR holds, the data binds later.
    ///
    /// # Panics
    ///
    /// Panics on a zero extent, which would make every gather clamp onto an
    /// empty buffer.
    #[must_use]
    pub fn kernel_for(
        id: pixelflow_ir::arena::BufferIdentity,
        width: u32,
        height: u32,
    ) -> pixelflow_ir::Kernel {
        assert!(
            width > 0 && height > 0,
            "DiscreteManifold::kernel_for: empty buffer ({width}x{height})"
        );
        let mut a = pixelflow_ir::ExprArena::new();
        let buf = a.declare_buffer(pixelflow_ir::arena::BufferDecl { id, width, height });
        let (x, y) = (a.push_var(0), a.push_var(1));
        let root = a.push_gather(buf, x, y);
        pixelflow_ir::Kernel::from_parts(a, root)
    }

    /// This buffer's nearest-neighbour read as a composable fragment. See
    /// [`Self::kernel_for`].
    ///
    /// # Panics
    ///
    /// Panics on an empty buffer, or when an extent exceeds `u32`.
    #[must_use]
    pub fn kernel(&self) -> pixelflow_ir::Kernel {
        Self::kernel_for(
            self.id,
            u32::try_from(self.width).expect("buffer width exceeds u32"),
            u32::try_from(self.height).expect("buffer height exceeds u32"),
        )
    }

    /// Wrap this buffer in a [`BilinearSampler`], JIT-compiling the 4-tap
    /// blend kernel bound to it.
    ///
    /// # Panics
    ///
    /// Panics on an empty buffer, when an extent exceeds `u32`, or when this
    /// build's `Field` width does not match the JIT's emitted width.
    #[must_use]
    pub fn bilinear(self) -> BilinearSampler {
        assert!(
            !self.buffer.is_empty(),
            "DiscreteManifold::bilinear on an empty buffer ({}x{})",
            self.width,
            self.height,
        );
        assert_eq!(
            core::mem::size_of::<Field>(),
            pixelflow_codegen::JIT_VECTOR_BYTES,
            "DiscreteManifold::bilinear: Field width does not match the JIT's emitted width"
        );
        let width = u32::try_from(self.width).expect("buffer width exceeds u32");
        let height = u32::try_from(self.height).expect("buffer height exceeds u32");
        let (arena, root) = bilinear_arena(self.id, width, height);
        // Bound-memory arenas are uncacheable (the code bakes buffer slot
        // metadata); compile recognizes that and compiles fresh.
        // Sampled one batch at a time through `Manifold::eval`: a point shape.
        let jit =
            pixelflow_codegen::jit_cache::compile(&arena, root, pixelflow_ir::LatticeShape::POINT)
                .expect("bilinear sampler failed to compile");
        BilinearSampler { tex: self, jit }
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl crate::ext::ManifoldExpr for BilinearSampler {}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl Manifold<(Field, Field, Field, Field)> for BilinearSampler {
    type Output = Field;

    #[inline(always)]
    fn eval(&self, (x, y, z, w): (Field, Field, Field, Field)) -> Field {
        let ctx = [self.tex.buffer.as_ptr()];
        // SAFETY: `bilinear` checked size_of::<Field>() == JIT_VECTOR_BYTES;
        // the kernel was compiled from an arena declaring exactly one buffer
        // whose decl matches `tex`'s extents, and `ctx` binds that buffer's
        // live base pointer for the duration of the call.
        unsafe {
            self.jit
                .call_bound(ctx.as_ptr(), pixelflow_codegen::Point4::new(x, y, z, w))
        }
    }
}
