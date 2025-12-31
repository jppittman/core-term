# PixelFlow Autodiff Rendering: The Gap Nobody Filled

## Thesis

Forward-mode automatic differentiation for real-time graphics on CPU+SIMD is a **genuinely unexplored** approach—not because it doesn't work, but because the people who understood the math (Conal Elliott, 2009) didn't have the toolchain, and the people with the toolchain (shader programmers) learned from IQ's finite-difference patterns.

PixelFlow is positioned in the gap.

## The Evidence

| Who | When | What | Why It Didn't Spread |
|-----|------|------|---------------------|
| Conal Elliott | 2009 | "Beautiful Differentiation" - explicitly cites surface normals for 3D rendering | Haskell, no SIMD, FP audience |
| Inigo Quilez | 2013 | Dual numbers shader on Shadertoy | GLSL can't do operator overloading elegantly; comments show confusion |
| Eric Jang | 2019 | JAX raytracer, shows autodiff normals 6x faster than finite diff | Python/JAX, reverse-mode ecosystem, ML audience |
| RXMesh paper | 2025 | Forward-mode AD on GPU beats PyTorch/JAX for mesh ops | Mesh optimization, not real-time rendering |

**The pattern**: Everyone who tries it finds it works. Nobody productionizes it because:

1. **GLSL/HLSL** - No operator overloading. Can't write `Dual<float>`.
2. **ML ecosystem** - Optimized for reverse-mode (many inputs → scalar loss). Graphics is the opposite (2 inputs → many outputs).
3. **Cultural inertia** - "6 evals for normals" is received wisdom. IQ's entire oeuvre is hand-derived gradients.
4. **Wrong audience** - Conal talks to FP people. IQ talks to shader wizards. Neither talks to systems programmers who want correctness AND performance.

## Why PixelFlow Is Different

```rust
// This compiles to fused SIMD. No tape. No graph. No runtime dispatch.
let normal = cross(hit_point.dx, hit_point.dy).normalize();
```

We have:
- **Rust** - Operator overloading that monomorphizes to optimal assembly
- **SIMD as algebra** - `Jet3<f32x8>` evaluates 8 rays with derivatives in parallel
- **Pull-based architecture** - Derivatives flow through composition automatically
- **No framework overhead** - Forward-mode is just carrying extra floats

The `Jet3` type already exists. The `ColorReflect` material already extracts normals from the tangent frame via cross product. **We're already doing it.**

## The Technical Claim

For ray-surface intersection with normals:

| Approach | Evaluations | Memory | Branches |
|----------|-------------|--------|----------|
| Finite differences (IQ standard) | 6 SDF evals | O(1) | 0 |
| Tetrahedron trick | 4 SDF evals | O(1) | 0 |
| Forward-mode AD (Jet3) | 1 eval, 4x wider | O(1) | 0 |

Forward-mode carries `(val, dx, dy, dz)` - 4 floats instead of 1. But you evaluate **once**. For any SDF more complex than a sphere, this wins.

For subdivision surfaces (Stam eigenanalysis), you evaluate a polynomial. Jet3 differentiates polynomials exactly. The normal emerges from the same computation that produces the position.

## What We're Building

### Phase 1: Validate on Catmull-Clark Subdivision (Current Target)

Stam 1998 gives us exact limit surface evaluation via eigendecomposition:

```
P(u,v) = Σᵢ λᵢⁿ⁻¹ · φᵢ(u,v) · cᵢ
```

Where `φᵢ(u,v)` are bicubic polynomials. Jet3 differentiates these exactly.

**Implementation**:
1. Parse OBJ cage mesh (done: 3DScanStore head, 12k quads)
2. Bake eigenstructures per valence at compile time
3. Evaluate limit surface with `Jet3<Field>` inputs
4. Normal = `cross(dP/du, dP/dv)` - falls out automatically
5. Newton iteration for ray intersection (Jet3 gives Jacobian for free)

### Phase 2: Benchmark Against Tessellation

The industry tessellates subdivision surfaces, then raytraces triangles. We raytrace the limit surface directly.

**Hypothesis**: For high-quality rendering (no LOD popping, infinite zoom), analytical wins. For game-engine style "close enough", tessellation wins.

We need numbers to know which is which.

### Phase 3: Generalize

If Phase 1 validates, the pattern extends to:
- **Bézier patches** (same eigenanalysis idea, simpler math)
- **NURBS** (CAD interop)
- **Neural SDFs** (same Jet3, MLP weights instead of control points)
- **Differentiable physics** (same autodiff, different domain)

## Risks & Honest Assessment

**What could go wrong:**

1. **Newton iteration cost** - Ray-subdivision intersection requires iterative refinement. If this dominates, tessellation wins.
2. **BVH complexity** - Bounding hierarchies over patches are messier than over triangles.
3. **Cache behavior** - Eigenstructure tables might thrash L1. Need to measure.
4. **SIMD utilization** - Divergent Newton iterations within a SIMD lane could hurt.

**What we're betting on:**

1. Monomorphization eliminates abstraction overhead
2. Forward-mode AD is genuinely cheaper than 4-6x evaluation
3. The algebraic elegance leads to optimization opportunities we can't see yet

**If we're wrong**: We learn why everyone tessellates. That's still valuable.

## Success Criteria

1. **Renders correctly** - Face mesh with smooth normals, no faceting
2. **Competitive performance** - Within 2x of tessellated equivalent at same visual quality
3. **Derivatives work** - Antialiasing, motion blur, curvature effects all emerge from Jet3

## Non-Goals (For Now)

- Production-quality asset pipeline
- Animation/rigging
- Multi-bounce GI
- Comparison with GPU approaches

## The Bet

Nobody does forward-mode AD for real-time CPU graphics because the communities don't talk to each other. We're in the gap. Either we find the pit everyone else saw and avoided, or we find the path nobody took.

Either way, we learn something.

---

*"The best time to plant a tree was 20 years ago. The second best time is now."*
*Conal planted the tree in 2009. We're harvesting.*

## Implementation Notes

### Eigen Table Structure

```rust
static EIGEN_TABLES: [EigenStructure; 48] = [
    EIGEN_V3,
    EIGEN_V4,  // This one is trivial - it's just B-spline
    EIGEN_V5,
    // ...
];

fn get_eigen(valence: usize) -> &'static EigenStructure {
    &EIGEN_TABLES[valence - 3]
}
```

The eigenstructures will be baked at compile time, one per valence configuration. Valence 4 (regular vertices) reduces to standard bicubic B-spline basis.
