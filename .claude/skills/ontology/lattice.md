# The lattice and evaluation

### Lattice

- **Is:** an extent and nothing else. "Its points are the indices
  `[0, wₓ) × [0, w_y)` and nothing else. It is the representable functor's
  index, and the law [`index(collapse(f)) = f`] holds without a side
  condition." It has two axes. Its extent is part of the program: "L_s is
  the lattice. It is part of the program, not an argument." JP: "the lattice
  is part of the schedule." Its rows, batches and lanes are the schedule's
  traversal, not folds the kernel is wrapped in: each backend owns them
  (`docs/plans/2026-10-15-kernels-evaluate-kernels.md` §3).
- **Is not:**
  - A coordinate frame. JP, 2026-09-06: "forced all of the translation to be
    contramaps plus uniforms". With an origin, `index ∘ collapse = f ∘
    shift` and the law breaks.
  - Four-dimensional: "an axis that never varies is not an axis".
  - A runtime argument.
  - A home for uniforms.
  - A union of ranges.
  - Visible to the e-graph.
  - Superseded: "dissolves into the typed discrete field" (07-20, never
    built); "extents and origin" (kernel-with-a-lattice, retracted the same
    day by L2); `LoopShape { loop_mask, exact_lanes }` (09-01, "the wrong
    denotation"); extents kept out of the compile key (loop-aware, reversed
    09-02: "Resizing is recompilation").
- **Follows:** coordinates enter only through the kernel's own contramaps. A
  sub-lattice is a `usize` index range. The extent is in the compile key, so
  a resize recompiles. `Lattice::point` became a contramap over a 1×1 extent
  (`eval_at`). The law is pinned by a test that fails if a coordinate is put
  back on the index. The lattice's traversal is the schedule's and runs after
  extraction, so no rule reaches it (L4, open).
- **Lives:** `pixelflow_core::Lattice { extent: [u32; AXES] }`, `AXES =
  COORD_AXES = 2` (`pixelflow-core/src/lattice/mod.rs`);
  `pixelflow_ir::LatticeShape` (`pixelflow-ir/src/variance.rs`). Decided in
  lattice-is-the-index "The shape", §1, §7 (L1 #1194, L2 #1196),
  collapse-is-a-fold §2, and one-pipeline §4. Today
  `.claude/agents/pixelflow-core.md` still lists `origin: [f32; 2]`.
  `Lattice::eval_at` compiles at `[1, 1]` and places the point through the
  band's origin (`PlaneRegion::at_point`), not through a contramap.

### Shape (extent). Homonym

- **Is:** in a cache key, the static extent a kernel is compiled for
  (`LatticeShape`, "Lattice with origin erased and nothing else erased"):
  "The extent is baked." "Performance does not argue for baking; the
  language does." Other senses: `Shape<'a, R>`, the pattern functor the
  e-graph speaks (ir-as-a-trait); production shapes A (one scene kernel per
  frame) and B (many small glyph kernels at startup), which any pipeline
  result must hold on; and informal "kernel shape".
- **Is not:** a runtime bound ("a band of a different height is a different
  kernel"). Not removable from the key (a first-draft error, corrected in
  collapse-is-a-fold §2.5).
- **Follows:** saturation does not depend on the shape, but extraction
  does. The cache saturates per structure and extracts per shape, so a new
  band height costs one extraction and one emit, not a saturation. A frame
  stripes into at most two heights. An extent that is not a lane multiple
  adds a remainder arm.
- **Lives:** `LatticeShape` (`pixelflow-ir/src/variance.rs`), `Shape<'a, R>`
  (`pixelflow-ir/src/term.rs`), `pixelflow-codegen/src/jit_cache.rs`, the
  structure-keyed saturation cache in `pixelflow-search/src/runtime.rs`,
  `Manifold::compile(extent)`. Decided in collapse-is-a-fold §2.5 and
  one-pipeline §4.

### Manifold (compile, bake)

- **Is:** a kernel compiled at a lattice's shape, "the thing you can sample
  over a domain". It knows its extents and its code
  bytes. It has no `eval` and is not batch-shaped, and it keeps a table of
  the shapes it was compiled at. `bake = collapse(compile(k, extent))`: there is
  no bind and no `BoundManifold` in the language (kernels-evaluate-kernels §2).
- **Is not:** the per-batch trait `eval(point) -> value`: "That had the word
  bound to the wrong level." That trait is what forced a Rust loop around
  the JIT and the per-batch ABI cost ("every vector register caller-saved,
  every invariant recomputed — and that is the whole 2×"). Not
  `CompiledKernel`. Not "the composition substrate" (07-20). Not a colour
  object.
- **Follows:** "Per-batch evaluation is not an API." A colour scene,
  a discrete kernel and a bake are one object "sampled three ways". One
  name must not mean two things in two crates, so graphics' re-export of the
  trait is gone.
- **Lives:** `pixelflow_core::Manifold` (`BoundManifold` is retired with bind, kernels-evaluate-kernels §2)
  (`pixelflow-core/src/lattice/manifold.rs`). Decided in kernel-with-a-lattice
  "The shape", S1, S4b-1, S4b-2. `eval_at` on the retired bound manifold is a 1×1
  `collapse_rows`, not a per-batch entry.

### CompiledKernel

- **Is:** "one kernel's emitted bytes at one shape", in one
  executable-memory region that is freed on drop.
- **Is not:** a Manifold, which is a kernel at a shape and binds nothing.
  Never re-exported from core. Not a reusable code buffer (`CodeBuffer`,
  `CompileWorkspace` and `MAP_JIT` are deleted).
- **Follows:** core holds it only in private fields. The width it was
  compiled for belongs to the JIT.
- **Lives:** `pixelflow-codegen/src/compiled_kernel.rs`,
  `pixelflow-codegen/src/emit/executable.rs`. Renamed from `JitManifold` in
  S4b-1.

### Collapse

- **Is:** the one verb that turns a compiled kernel into numbers. It
  tabulates a compiled manifold over a lattice or a band of it. Its
  denotation is the table, `index(collapse(f)) = f` (kernels-evaluate-kernels
  §2). Its three SEQ folds around one store are the CPU module's traversal of
  that denotation (§3):
  `collapse(f) = fold_j fold_k fold_l Write(out, j, k, l, f(x0 + k + l, y0 + j))`.
  `collapse(extent)` followed by `pack(L)` is that CPU pass, not a
  target-agnostic one.
  The ABI is `fn(ctx, out, pitch)`, called once per stripe. The origin is
  two uniforms.
- **Is not:**
  - Per-batch evaluation.
  - A scaffold the emitter builds (`emit_collapse_loop`, `Level`,
    `CollapseBody`, `HoistCtx`, `Point4`, `TileSlice` are deleted).
  - A reduction API: `collapse_with` and `ReduceOp` were deleted because "a
    reduction is a binder inside the kernel".
  - Movable before saturation: the saturation key has no shape, `pack` needs
    L, and collapse refuses `Dwrt` (L4).
  - Superseded: one JIT call per row with `(x0, y, z, w)` arguments; "over
    eliminates a dimension; the render loop is the second shape" (07-28),
    retracted because the render loop is a fold over SEQ.
- **Follows:** the CPU module owns the loop nest, the hoisting, the choice of
  broadcast versus gather, and the tail (kernels-evaluate-kernels §2). The whole 2.3× win over LLVM was hoisting.
  No vector crosses the ABI. The remainder is the same fold, shorter, with a
  masked or lane-wise store. A band of a new height is a new shape.
- **Lives:** `Lattice::collapse` (`pixelflow-core/src/lattice/mod.rs`),
  `collapse_rows` on the retired `BoundManifold` (`pixelflow-core/src/lattice/manifold.rs`),
  `pixelflow_ir::passes::lattice::{collapse, pack}`
  (`pixelflow-ir/src/passes/lattice.rs`). Decided in collapse-is-a-fold
  §2.1–§2.5 and step 5 (JP: "I'd literally insert folds to iterate over the
  lattice"); the traversal is superseded by kernels-evaluate-kernels §3.

### Write and Seq (effects)

- **Is:** the store that realizes collapse. The language has no write:
  collapse yields a whole value, so no intermediate state is observable (§2). `Write { row, col, lane, value }` stores `value`'s
  first `len(lane)` lanes at `out + 4·(row·pitch + col + lane)`. It names
  binders, not an address expression. `OpKind::Seq` means "evaluate the
  left, then the right".
- **Is not:** constructible by an author (`push_write` is `pub(crate)`, and
  `insert` and `kernel!` refuse both nodes). Not an address expression,
  because contiguity holds by construction. Not a value. Not a node with an
  `out` field: "a field that can hold one value is a comment".
- **Follows:** a `Write` def *is* the lane fold, executed by lanes.
  `alu(Seq)` emits no bytes. A SEQ fold's accumulator slot is one dead
  vector, "what 'no special case' costs". Because binders are fields, no
  substitution rewrites them (L3, open).
- **Lives:** `ExprNode::Write`, `push_write` (`pixelflow-ir/src/arena.rs`),
  `ScheduledOp::{Write, Seq}` (`pixelflow-codegen/src/program/mod.rs`),
  `IsaBackend::store` (`pixelflow-codegen/src/emit/mod.rs`);
  collapse-is-a-fold §2.4, step 3.

### Lane, batch, width (SIMD). Homonym of the actor lane

- **Is:** an implementation detail of codegen. "pixelflow-core holds no
  vector type and no width. The one width in the workspace is the JIT's."
  The width is the chosen backend's register file
  (`B::FILE.vector_bytes()`), read through `jit_vector_bytes()`. A batch is
  one trip of the column fold, which is one vector. "Lanes are a scope: the
  fold a Write names as its lane binder is executed by lanes." That fold has
  no counter and no back edge, and its binder materialises as
  `[0, 1, …, len−1]`.
- **Is not:** in the language's vocabulary. Not a type in core (`Field`,
  `PARALLELISM`, `NativeSimd`, `JIT_VECTOR_BYTES` are deleted). Not something
  that crosses the ABI. Not decided at build time. Not an actor priority
  lane.
- **Follows:** a lane-uniform value is a broadcast, and a value varying by
  lane alone is invariant over the call, so it is a body root (step 5½).
  `pack(L)` is the CPU module's pass that takes L (§3).
- **Lives:** `ScheduledOp::Lanes` (`pixelflow-codegen/src/program/mod.rs`),
  `jit_vector_bytes()` (`pixelflow-codegen/src/isa/mod.rs`); CLAUDE.md "SIMD
  is an implementation detail"; collapse-is-a-fold §1a, §2.2;
  `desloppify/rules/simd-is-codegens.json`.

### pack (strip-mine)

- **Is:** the lattice's strip-mine by lane count L. It splits `fold_i` into
  `fold_k fold_l` with `i := k + l` and sequences a remainder fold. The two
  inner folds' ranges move, and nothing else does. It is a pass of the CPU
  module's chain (§3), reading the tier only through L.
- **Is not:** width-aware inside the IR: "`pixelflow-ir` never names a
  width". Not a rewrite rule ("A rule cannot run after extraction"). Not an
  unroller. Not a scalar remainder loop.
- **Follows:** `HalveFold` is this transform with L = 2 and the inner fold
  unrolled. Unifying them waits for a lattice fold to enter the e-graph (L2,
  Q6).
- **Lives:** `passes::lattice::pack` (`pixelflow-ir/src/passes/lattice.rs`),
  `Fold::strided` (`pixelflow-ir/src/fold.rs`); collapse-is-a-fold §2.3;
  one-pipeline §1.5.

### Execution rule (stripe)

- **Is:** how a fold is run, as distinct from what it denotes: by lanes, by
  threads, or interleaved by halving. A **stripe** is the unit one collapse
  call fills, pulled from a pool sized to the cores by work-stealing, and
  written straight into the frame.
- **Is not:** a loop of the language. A stripe is the schedule's loop, which
  each backend owns (§3); the lattice's traversal is not written as folds in
  pixelflow-core. Not a staging copy.
- **Follows:** "how the stripes are shared out does not change pixels". The
  stripes are each backend's schedule: the CPU module runs them on a thread
  pool, and a GPU backend maps one pixel to one thread. The L5 target, stripes
  as the row fold run by threads like a lane fold, is withdrawn (§3).
- **Lives:** kernel-with-a-lattice S2; one-pipeline §1.5, §1.6. Today the
  stripes are a Rust work-stealing loop in
  `pixelflow-graphics/src/render/scene.rs` around `collapse_rows`.

### DiscreteManifold (tabulation)

- **Is:** "the buffer that IS a manifold" by the law
  `index(collapse(f)) = f`. It is collapse's result, and is read back as a
  kernel. A tabulation's identity is `(key(f), shape)`.
- **Is not:** storage that merely backs a manifold. Not an evaluator (its
  `eval` was deleted in S4b-2). Not identified by a hash of its bytes.
- **Follows:** reading one is applying a kernel. Under "no tables",
  `CachedGlyph`, `CachedText`, the atlas and `BilinearSampler` become the
  font's programs and are deleted (D10).
- **Lives:** `DiscreteManifold`, `BilinearSampler`
  (`pixelflow-core/src/lattice/mod.rs`); kernel-with-a-lattice S4b-1;
  composition-is-linking §2. Today `CachedGlyph`, `CachedText`
  (`pixelflow-graphics/src/fonts/cache.rs`), `GlyphAtlas` (`fonts/atlas.rs`)
  and `BilinearSampler` are all live, and CLAUDE.md's "a glyph bakes once
  and reads back as a gather over its bound buffer" is stale.

### Origin

- **Is:** where a band's first sample is taken. In the emitted code it is
  "two uniforms the caller declares", read from an origin block of its own
  in the context, so a call copies nothing.
- **Is not:** a lattice field (removed in L2). Not a coordinate frame (that
  is a contramap). Not part of compilation. Not `Field::sequential(x0)`.
- **Follows:** the cache keys on shape alone. `(x0 + i)` is its own node, so
  the lane-uniform half of the address hoists. The origin block is read
  only by the body, so it is never a carry candidate.
- **Lives:** `emit::origin()` (defined in `pixelflow-codegen/src/pipeline.rs`,
  re-exported from `emit/mod.rs`; one process-global pair of minted
  identities), `PlaneRegion` (`pixelflow-core/src/lattice/manifold.rs`),
  `passes::lattice::Domain`; lattice-is-the-index L2; collapse-is-a-fold
  §2.1, step 5. Today `PlaneRegion` carries the pixel-centre convention:
  `PlaneRegion::rows` sets the origin to `(½, y0 + ½)`, so a coordinate
  frame rides on the origin, which contradicts "not a coordinate frame".

### Index range, band (and Union)

- **Is:** "A sub-lattice is an index range: rows `y₀ .. y₀ + n` of a plane.
  This is the one thing that legitimately lives on the domain side." One
  type carries two meanings, which must become two types (D3):
  - A **requested band**: a work partition chosen by the caller. It is
    public and never derived.
  - A **derived region**: what a mask implies. Retired with the domain split
    (§1).
- **Is not:** a coordinate. Not one concept just because both are
  rectangles. `Union`, a caller-declared disjoint sum of ranges, "a missing
  compiler capability that leaked into the public API", was deleted on
  2026-09-09: "it could never say anything `+` could not".
- **Follows:** "Overlap is what the select is for." `paint_complement`
  handles only a leading rectangle, an artifact of the two cases that wrote
  it.
- **Lives:** `pixelflow_core::IndexRange`, `IndexRange::paint_complement`
  (`pixelflow-core/src/lattice/union.rs`, a file still named for the deleted
  type), `PlaneRegion`, `CellGridShape::grid_range`
  (`pixelflow-core/src/lattice/cell_grid.rs`, a hand-derived region spelled
  as a band). Decided in lattice-is-the-index §3, §7 and one-conditional
  §5. CLAUDE.md still cites "`Union`'s explicit ranges" as live.

### Derived range and domain split (retired)

- **Is:** nothing in the language. A mask's implied rectangle existed to feed a
  domain split, which emits a second program. `If` has one shape, guards and
  one select (`docs/plans/2026-10-15-kernels-evaluate-kernels.md` §1), so the
  split is deleted.
- **Lives:** `pixelflow-ir/src/mask_support.rs` (`mask_support`,
  `MaskSupport`; D1, symbolic tier only) is the analysis still in code; the
  split was never built (D2) and is not to be built.

### Region. Homonym

- **Is:** three senses in force:
  1. An **ownership region**: a scope, or one arm of one `If` in it.
     Regions nest as `If`s do.
  2. `PlaneRegion`: a band of rows to collapse.
  3. `ScopeRegion`: the type name of the body scope, "which opens nowhere".
- **Is not:** a prologue in a chain of scopes (`Scope::Region(i)` and the
  frame/row/body "three regions" were deleted in collapse-is-a-fold step 5).
  Not R1's "fold region is a span" (retracted). Not 09-07's "region of equal
  demand", which became arm regions in code.
- **Follows:** whenever code says "region", ask which sense it means.
- **Lives:** `pixelflow-codegen/src/program/ownership.rs`,
  `pixelflow-codegen/src/program/mod.rs` (`ScopeRegion`),
  `pixelflow-ir/src/mask_support.rs`, `PlaneRegion`
  (`pixelflow-core/src/lattice/manifold.rs`).

### Scene (channel kernels, pack, colour)

- **Is:** "a compiled manifold with four channels": four channel kernels
  compiled together at the frame's shape, with the pack inside as integer IR
  ops that graphics composes. "No colour in the language." A choice between
  colours is one select on packed words.
- **Is not:** `Scene::Surface`, evaluated per batch (deleted S4a). Not
  `ColorCube`. Not four selects sharing one mask, which would make the
  reflected world un-guardable.
- **Follows:** shared geometry is emitted once. One guard can skip a whole
  reflected world. `Pixel::packed_shifts` is the single home of byte order.
- **Lives:** pixelflow-graphics `PackedManifold`, `PackedFrame`
  (`src/render/packed.rs`), `Scene::Packed`, `compile_packed_for`
  (`src/render/scene.rs`), `Pixel::packed_shifts` (`src/render/pixel.rs`);
  there is no `PackedProgram` type. Decided in kernel-with-a-lattice S2, S3b,
  S4a. Today `src/render/frame.rs`'s module doc still calls `Frame` "a
  Surface (read from)" written "via execute", and comments in
  `src/render/cell_grid.rs` and `core-term/src/terminal_app.rs` still name
  `ColorCube`.

### Pixel

- **Is:** centred. The pixel is the interval `[−½, ½)` about its sample, and
  the midpoint is the point sample every kernel computes.
- **Is not:** the corner cell `[x₀, x₀+1)` ("The corner cell is the typo").
- **Follows:** pixel-centre sampling is a contramap. A glyph's closed form
  bakes ±½ in, so a scaling `at` covers a pixel of the wrong size.
- **Lives:** `2026-09-23-an-integral-is-a-fold.md` §2 (that part survives
  its supersession); CLAUDE.md "Glyph coverage"; `SAMPLE_CENTER`,
  `PlaneRegion::rows` (`pixelflow-core/src/lattice/manifold.rs`). Today a
  pixel band's centres are built into the band's origin (`i + ½`, `j + ½`)
  by `PlaneRegion::rows`, not by a contramap on the kernel, which
  contradicts "pixel-centre sampling is a contramap".

### Pull-based rendering and the fixed observer

- **Is:** "Pixels are sampled, not pushed. Nothing computes until a lattice
  demands it." "Camera is at origin. Movement is achieved by warping
  coordinate space."
- **Is not:** immediate-mode drawing. Not a camera object. Not a lattice with
  an origin.
- **Follows:** "In a pull renderer the lattice does the asking." Movement is
  a contramap.
- **Lives:** CLAUDE.md "Philosophy"; lattice-is-the-index §3.
