# PixelFlow Graphics

**Composable 3D Rendering as Manifold Expressions**

Extends pixelflow-core with colors, fonts, and 3D scene composition. Everything stays compositional—no pre-computed geometry, no intermediate buffers.

```rust
use pixelflow_graphics::scene3d::{Surface, ScreenToDir, Reflect};
use pixelflow_core::{At, Select, Manifold, Jet3};

// Geometry: ray-sphere intersection (returns t)
let sphere_geom = Sphere { center: origin, radius: 100.0 };

// Material: fresnel reflection at hit point
let material = Reflect {
    inner: &sphere_geom,
};

// Background: sky gradient at ray direction (if miss)
let background = Sky { /* ... */ };

// Compose: blend material (at hit) with background (at ray) based on hit validity
let scene = Surface {
    geometry: sphere_geom,
    material,
    background,
};

// Render: evaluates fully as a manifold
let color = scene.eval_raw(rx, ry, rz, w);
```

## Architecture: The "Mullet" Approach

Three-layer pull-based architecture:

1. **Geometry** (expensive): Compute once per pixel via `Jet3` (with 3D derivatives for normals)
2. **Surface**: Warp coordinates from ray to hit point using the computed `t`
3. **Material**: Evaluate colors at hit point using derivatives for shading

Result: One geometric computation, colors flow as opaque `Discrete` (packed RGBA).

```
Manifold<Jet3, Output = Jet3>     (geometry: returns t with derivatives)
    ↓
Jet3 → At { inner: material, ... } (evaluates material at warped coords)
    ↓
Select with At { background, ... } (blends material/background as manifolds)
    ↓
Manifold<Jet3, Output = Discrete> (fully composed color)
```

## Philosophy: Pure Composition

**The idiomatic style**: Keep everything as manifolds. Compose at the type level.

✅ **Idiomatic**: All parts stay as manifolds
```rust
// Surface and ColorSurface use At + Select composition
let scene = At { inner: &material, x: hx, y: hy, z: hz, w };
let result = mask.select(scene, fallback);
```

❌ **Not idiomatic**: Materializing to intermediate values
```rust
// Don't do this - loses the compositional structure
let hit_point = t.eval_raw(...);  // Materialized!
let color_at_hit = material.eval_raw(hit_point);  // Lost context!
```

## Key Concepts

### Scene3D: Root of the Composition

Three variants:

- **Surface<G, M, B>**: Returns `Field` (single scalar per pixel)
- **ColorSurface<G, M, B>**: Returns `Discrete` (packed RGBA)

Both use the same pattern:
1. Evaluate geometry to get hit distance `t`
2. Validate `t` (positive, not too large, derivatives sensible)
3. Warp: compute hit point `P = ray * t`
4. Use `At` combinator to pin material/background to different coordinates
5. Use `Select` to blend based on hit validity

```rust
impl<G, M, B> Manifold<Jet3> for Surface<G, M, B> {
    fn eval_raw(&self, rx: Jet3, ry: Jet3, rz: Jet3, w: Jet3) -> Field {
        let t = self.geometry.eval_raw(rx, ry, rz, w);
        let mask = validate_t(t);  // Is this a valid hit?

        // Warp ray to hit point
        let safe_t = sanitize(t, mask);
        let hx = rx * safe_t;
        let hy = ry * safe_t;
        let hz = rz * safe_t;

        // Compose as manifolds - no materialization!
        let mat = At { inner: &self.material, x: hx, y: hy, z: hz, w };
        let bg = At { inner: &self.background, x: rx, y: ry, z: rz, w };

        // Select blends while staying compositional
        Select { cond: FieldMask(mask), if_true: mat, if_false: bg }
            .eval_raw(rx, ry, rz, w)
    }
}
```

### Reflect: The Crown Jewel

Reconstructs surface normals from the tangent frame implied by `Jet3` derivatives.

When you pin a manifold at a hit point using `At`, the derivatives tell you how the surface changes—that's the normal!

```rust
pub struct Reflect<M> {
    pub inner: M,  // Manifold<Jet3, Output = Field>
}

impl<M> Manifold<Jet3> for Reflect<M> {
    fn eval_raw(&self, x: Jet3, y: Jet3, z: Jet3, w: Jet3) -> Field {
        // x.dx, x.dy, x.dz form the tangent frame
        // Normal = normalize(tangent_u × tangent_v)
    }
}
```

## Compositionality Benefits

1. **Kernel fusion**: At + Select + geometry all inline into one SIMD kernel
2. **Genericity**: Works with `Field` (preview), `Jet3` (production), future types
3. **No intermediate buffers**: Rays don't materialize to hit points—they flow through the type system
4. **Automatic differentiation**: Derivatives propagate through all combinators

## Examples

### Simple Sphere

```rust
use pixelflow_graphics::scene3d::{Surface, Sphere};
use pixelflow_core::Manifold;

let sphere = Sphere {
    center: Field::from(0.0),
    radius: Field::from(100.0),
};

let material = ConstantColor { rgb: (1.0, 0.0, 0.0) };
let background = Sky { };

let scene = Surface {
    geometry: sphere,
    material,
    background,
};

// Evaluate at screen coordinates
let color = scene.eval_raw(x_jet, y_jet, z_jet, w_jet);
```

### Reflective Sphere

```rust
use pixelflow_graphics::scene3d::{Surface, Sphere, Reflect};
use pixelflow_core::Manifold;

let sphere = Sphere { /* ... */ };

// Material applies reflection (normal-based shading)
let material = Reflect {
    inner: LambertianShader {
        albedo: Field::from(0.8),
        normal_source: sphere,  // Computes normal from Jet3 derivatives
    },
};

let scene = Surface {
    geometry: sphere,
    material,
    background: Sky { },
};
```

## Coordinate Types

All manifolds are generic over the input coordinate type `I`:

- **Field**: Concrete SIMD values (for evaluation)
- **Jet2**: 2D automatic differentiation (gradients)
- **Jet3**: 3D automatic differentiation (normals for 3D surfaces)

The same expression works with all three—no rewrite needed.

## No Pre-Computation

The key rule: **Don't materialize intermediate results.**

❌ Bad:
```rust
let hit_point_x = hx.eval_raw(rx, ry, rz, w);  // Materialized!
```

✅ Good:
```rust
let mat_at_hit = At { inner: &material, x: hx, y: hy, z: hz, w };
// hx stays as a Jet3 expression, composes into the kernel
```

## Integration with pixelflow-core

This crate builds on top of `pixelflow-core`:

- Uses `Manifold`, `Select`, `At`, `Jet3`, etc. from core
- Adds scene-specific types: `Surface`, `ColorSurface`, `Reflect`
- All composition happens at the core manifold level

## License

MIT
