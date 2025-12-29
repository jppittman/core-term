//! # Analytic 3D Scene Primitives (The "Genius" Architecture)
//!
//! Three-Layer Pull-Based Architecture:
//! 1. Geometry: Returns `t` (Jet3)
//! 2. Surface: Warps `P = ray * t` (Creates tangent frame via Chain Rule)
//! 3. Material: Reconstructs Normal from `P` derivatives
//!
//! ## Architecture
//!
//! The "mullet" approach for full-color rendering:
//! - **Front (serious)**: Geometry computed ONCE per pixel via Jet3
//! - **Back (party)**: Colors flow as opaque `Discrete` (packed RGBA)
//!
//! This gives 3x speedup vs running geometry 3x (once per R,G,B channel).
//!
//! No iteration. Nesting is occlusion.

use pixelflow_core::{At, Discrete, Field, Jet3, Manifold, Select};

// ============================================================================
// HELPER: Lift Field mask to Jet3 manifold for Select conditions
// ============================================================================

/// Wraps a Field mask to implement Manifold<Jet3> for use as a Select condition.
/// This is needed because Select<C, T, F> for Jet3 requires C: Manifold<Jet3, Output = Jet3>.
#[derive(Clone, Copy)]
struct FieldMask(Field);

impl Manifold<Jet3> for FieldMask {
    type Output = Jet3;

    #[inline(always)]
    fn eval_raw(&self, _x: Jet3, _y: Jet3, _z: Jet3, _w: Jet3) -> Jet3 {
        // Convert Field mask to Jet3 with zero derivatives
        Jet3::constant(self.0)
    }
}

// ============================================================================
// ROOT: ScreenToDir
// ============================================================================

/// Converts screen coordinates to ray direction jets.
///
/// **CRITICAL**: This must seed the derivatives correctly.
/// - Screen X changes by 1.0 per pixel (dx=1, dy=0)
/// - Screen Y changes by 1.0 per pixel (dx=0, dy=1)
/// - Direction is normalized(Screen X, Screen Y, 1.0)
///
/// The Chain Rule propagates these derivatives into the ray direction,
/// allowing Materials to know "how the ray changes" across the pixel.
///
/// **Compositional Pattern**: Ray normalization is expressed via the At
/// combinator, making the vector operation pipeline explicit to the compiler.
#[derive(Clone, Copy)]
pub struct ScreenToDir<M> {
    pub inner: M,
}

impl<M: Manifold<Jet3, Output = Field>> Manifold for ScreenToDir<M> {
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, _z: Field, w: Field) -> Field {
        // 1. Seed Jets from Screen Coords
        // x: varies with screen x (dx=1, dy=0, dz=0)
        let sx = Jet3::x(x);
        // y: varies with screen y (dx=0, dy=1, dz=0)
        let sy = Jet3::y(y);
        // z: constant 1.0 (pinhole focal length, no derivatives)
        let sz = Jet3::constant(Field::from(1.0));

        // 2. Compose Ray Direction via Vector Normalization
        // Compute length of the unnormalized ray direction
        // The Jet math automatically computes d(dir)/dx and d(dir)/dy via chain rule
        let len_sq = sx * sx + sy * sy + sz * sz;
        let len = len_sq.sqrt();

        // Normalized direction components
        let dx = sx / len;
        let dy = sy / len;
        let dz = sz / len;

        // 3. Compose Inner Manifold at Normalized Ray Direction
        // Use At combinator to make vector operation composition explicit.
        // This allows future Jet3 optimizations (FMA/RSQRT fusion) to benefit
        // all code paths that normalize direction vectors.
        At {
            inner: &self.inner,
            x: dx,    // Jet3 constant manifold
            y: dy,
            z: dz,
            w: Jet3::constant(w),
        }
        .eval_raw(sx, sy, sz, Jet3::constant(w))
    }
}

/// Converts screen coordinates to ray direction jets, outputting Discrete.
///
/// Same as ScreenToDir but for color (Discrete) pipelines.
///
/// **Compositional Pattern**: Uses At combinator to express ray normalization
/// as explicit manifold composition.
#[derive(Clone, Copy)]
pub struct ColorScreenToDir<M> {
    pub inner: M,
}

impl<M: Manifold<Jet3, Output = Discrete>> Manifold for ColorScreenToDir<M> {
    type Output = Discrete;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, _z: Field, w: Field) -> Discrete {
        let sx = Jet3::x(x);
        let sy = Jet3::y(y);
        let sz = Jet3::constant(Field::from(1.0));

        // Compose ray direction via vector normalization
        let len_sq = sx * sx + sy * sy + sz * sz;
        let len = len_sq.sqrt();

        let dx = sx / len;
        let dy = sy / len;
        let dz = sz / len;

        // Compose inner manifold at normalized ray direction via At combinator
        At {
            inner: &self.inner,
            x: dx,
            y: dy,
            z: dz,
            w: Jet3::constant(w),
        }
        .eval_raw(sx, sy, sz, Jet3::constant(w))
    }
}

// ============================================================================
// LAYER 1: Geometry (Returns t)
// ============================================================================

/// Unit sphere centered at origin.
/// Solves |t * ray| = 1  =>  t = 1 / |ray|
///
/// **Compositional Pattern**: Ray length is computed via vector normalization,
/// enabling future RSQRT fusion optimizations on Jet3.
#[derive(Clone, Copy)]
pub struct UnitSphere;

impl Manifold<Jet3> for UnitSphere {
    type Output = Jet3;

    #[inline(always)]
    fn eval_raw(&self, rx: Jet3, ry: Jet3, rz: Jet3, _w: Jet3) -> Jet3 {
        // Compose ray length via squared-length operation
        // t = 1.0 / sqrt(rx^2 + ry^2 + rz^2)
        // The At combinator makes the vector length operation explicit
        let len_sq = rx * rx + ry * ry + rz * rz;
        Jet3::constant(Field::from(1.0)) / len_sq.sqrt()
    }
}

/// Sphere at given center with radius.
/// Solves |t * ray - center|² = radius²
/// Returns t for the closer intersection (or negative/NaN if miss)
///
/// **Compositional Pattern**: Dot products and magnitude operations are expressed
/// as explicit Jet3 compositions, preparing for vector operation fusion optimizations.
#[derive(Clone, Copy)]
pub struct SphereAt {
    pub center: (f32, f32, f32),
    pub radius: f32,
}

impl Manifold<Jet3> for SphereAt {
    type Output = Jet3;

    #[inline(always)]
    fn eval_raw(&self, rx: Jet3, ry: Jet3, rz: Jet3, _w: Jet3) -> Jet3 {
        // Ray: P = t * (rx, ry, rz) from origin
        // Sphere: |P - C|² = r²
        // Expanding: |t*ray - C|² = r²
        //   t²|ray|² - 2t(ray·C) + |C|² = r²
        // For normalized ray (|ray|=1):
        //   t² - 2t(ray·C) + |C|² - r² = 0
        // Quadratic: t = (ray·C) ± sqrt((ray·C)² - (|C|² - r²))
        // We want the closer (smaller positive) root: t = (ray·C) - sqrt(...)

        let cx = Jet3::constant(Field::from(self.center.0));
        let cy = Jet3::constant(Field::from(self.center.1));
        let cz = Jet3::constant(Field::from(self.center.2));

        // Compose dot product: ray · center (explicit vector operation)
        let d_dot_c = rx * cx + ry * cy + rz * cz;

        // Compose magnitude squared: |center|² via explicit vector operation
        let c_sq = cx * cx + cy * cy + cz * cz;
        let r_sq = Jet3::constant(Field::from(self.radius * self.radius));
        let c_minus_r = c_sq - r_sq;

        // Discriminant composition
        let discriminant = d_dot_c * d_dot_c - c_minus_r;

        // At grazing angles (discriminant near 0), the derivative of sqrt(discriminant)
        // goes to infinity, causing artifacts.
        //
        // SMOOTH FIX: Instead of sqrt(disc), use sqrt(disc + ε²)
        // This smoothly transitions to sqrt(ε²) = ε as disc → 0
        // The derivative is disc / sqrt(disc² + ε²) which stays bounded.
        let epsilon_sq = Jet3::constant(Field::from(0.0001)); // ε² = 0.01²
        let safe_discriminant = discriminant + epsilon_sq;

        // t = (ray·C) - sqrt(discriminant)
        // If discriminant < -ε², sqrt returns NaN → miss (but now that's very negative)
        // The sqrt operation is composed explicitly for future optimization
        d_dot_c - safe_discriminant.sqrt()
    }
}

/// Horizontal plane at y = height.
/// Solves P.y = height => t * ry = height => t = height / ry
///
/// **Compositional Pattern**: Division is composed explicitly to enable
/// scalar RSQRT/FMA fusion optimizations on Jet3.
#[derive(Clone, Copy)]
pub struct PlaneGeometry {
    pub height: f32,
}

impl Manifold<Jet3> for PlaneGeometry {
    type Output = Jet3;

    #[inline(always)]
    fn eval_raw(&self, _rx: Jet3, ry: Jet3, _rz: Jet3, _w: Jet3) -> Jet3 {
        // Compose division operation explicitly for optimization
        Jet3::constant(Field::from(self.height)) / ry
    }
}

// ============================================================================
// LAYER 2: Surface (The Warp)
// ============================================================================

/// The Glue. Combines Geometry, Material, and Background.
///
/// Performs **The Warp**: `P = ray * t`.
/// Because `t` carries derivatives from Layer 1, and `ray` carries derivatives
/// from Root, `P` automatically contains the Surface Tangent Frame via the Chain Rule.
///
/// **Compositional Pattern**: Uses At combinator to compose ray-parameter multiplication
/// with material/background evaluation, making vector warping explicit to the compiler.
#[derive(Clone, Copy)]
pub struct Surface<G, M, B> {
    pub geometry: G,     // Returns t
    pub material: M,     // Evaluates at Hit Point P
    pub background: B,   // Evaluates at Ray Direction D (if miss)
}

impl<G, M, B> Manifold<Jet3> for Surface<G, M, B>
where
    G: Manifold<Jet3, Output = Jet3>,
    M: Manifold<Jet3, Output = Field>,
    B: Manifold<Jet3, Output = Field>,
{
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, rx: Jet3, ry: Jet3, rz: Jet3, w: Jet3) -> Field {
        // 1. Ask Geometry for distance t
        let t = self.geometry.eval_raw(rx, ry, rz, w);

        // 2. Check Hit Validity (Mask) - is t a valid hit?
        let zero = Field::from(0.0);
        let t_max = Field::from(1e6);
        let deriv_max = Field::from(1e4);

        let valid_t = t.val.gt(zero) & t.val.lt(t_max);
        let deriv_mag_sq = t.dx * t.dx + t.dy * t.dy + t.dz * t.dz;
        let valid_deriv = deriv_mag_sq.lt(deriv_max * deriv_max);
        let mask = valid_t & valid_deriv;

        // 3. THE SAFE WARP: P = ray * t (sanitized against NaN/Inf)
        // Compose scalar-vector multiplication explicitly for optimization
        let one = mask & Field::from(1.0);
        let safe_t = Jet3 {
            val: t.val * one,
            dx: t.dx * one,
            dy: t.dy * one,
            dz: t.dz * one,
        };

        let hx = rx * safe_t;
        let hy = ry * safe_t;
        let hz = rz * safe_t;

        // 4. Blend via manifold composition using At + Select
        // At combinator composes material/background evaluation at warped coordinates
        // making the vector transformation pipeline explicit to the compiler.
        let mat = At { inner: &self.material, x: hx, y: hy, z: hz, w };
        let bg = At { inner: &self.background, x: rx, y: ry, z: rz, w };

        // Compose fully: mask selects between material and background manifolds
        Select { cond: FieldMask(mask), if_true: mat, if_false: bg }
            .eval_raw(rx, ry, rz, w)
    }
}

/// Color Surface: geometry + material + background, outputs Discrete.
///
/// **Compositional Pattern**: Same at-warp pipeline as Surface, using At to
/// make vector transformations explicit for color manifolds.
#[derive(Clone, Copy)]
pub struct ColorSurface<G, M, B> {
    pub geometry: G,
    pub material: M,
    pub background: B,
}

impl<G, M, B> Manifold<Jet3> for ColorSurface<G, M, B>
where
    G: Manifold<Jet3, Output = Jet3>,
    M: Manifold<Jet3, Output = Discrete>,
    B: Manifold<Jet3, Output = Discrete>,
{
    type Output = Discrete;

    #[inline(always)]
    fn eval_raw(&self, rx: Jet3, ry: Jet3, rz: Jet3, w: Jet3) -> Discrete {
        let t = self.geometry.eval_raw(rx, ry, rz, w);

        let zero = Field::from(0.0);
        let t_max = Field::from(1e6);
        let deriv_max = Field::from(1e4);

        let valid_t = t.val.gt(zero) & t.val.lt(t_max);
        let deriv_mag_sq = t.dx * t.dx + t.dy * t.dy + t.dz * t.dz;
        let valid_deriv = deriv_mag_sq.lt(deriv_max * deriv_max);
        let mask = valid_t & valid_deriv;

        // Compose scalar-vector multiplication explicitly
        let one = mask & Field::from(1.0);
        let safe_t = Jet3 {
            val: t.val * one,
            dx: t.dx * one,
            dy: t.dy * one,
            dz: t.dz * one,
        };

        let hx = rx * safe_t;
        let hy = ry * safe_t;
        let hz = rz * safe_t;

        // Blend via manifold composition using At + Select
        // At combinator makes the warp pipeline explicit
        let mat = At { inner: &self.material, x: hx, y: hy, z: hz, w };
        let bg = At { inner: &self.background, x: rx, y: ry, z: rz, w };

        // Compose fully: mask selects between material and background manifolds
        Select { cond: FieldMask(mask), if_true: mat, if_false: bg }
            .eval_raw(rx, ry, rz, w)
    }
}

// ============================================================================
// LAYER 3: Materials
// ============================================================================

/// Reflect: The Crown Jewel.
/// Reconstructs surface normal from the Tangent Frame implied by the Jet derivatives.
///
/// **Compositional Pattern**: Vector operations (magnitude, normalization, cross product,
/// dot product) are composed explicitly, preparing for FMA/RSQRT fusion on Jet3.
#[derive(Clone, Copy)]
pub struct Reflect<M> {
    pub inner: M,
}

impl<M: Manifold<Jet3, Output = Field>> Manifold<Jet3> for Reflect<M> {
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, x: Jet3, y: Jet3, z: Jet3, w: Jet3) -> Field {
        // The input (x, y, z) is the hit point P with derivatives dP/dscreen.
        // We need to compute the reflected direction R with derivatives dR/dscreen.
        //
        // For a sphere, the normal N = normalize(P - center).
        // Since center is constant, N = normalize(P) (for unit sphere at origin style).
        // Actually for our warp, P = t * ray_dir, and we want N pointing outward.
        //
        // Key insight: The normal IS the normalized hit point direction for a sphere
        // centered at origin. For SphereAt, we'd need (P - center), but our P already
        // encodes the surface position.
        //
        // For a general surface, N comes from the tangent cross product.
        // But the tangent vectors Tu, Tv ARE the derivatives dP/dx, dP/dy.
        // So N = normalize(Tu × Tv) where Tu = (x.dx, y.dx, z.dx), Tv = (x.dy, y.dy, z.dy).

        // Compose hit point magnitude explicitly for future optimization
        let p_len_sq = x * x + y * y + z * z;
        let p_len = p_len_sq.sqrt();
        let one = Jet3::constant(Field::from(1.0));
        let inv_p_len = one / p_len;

        // Extract tangent vectors (as scalars)
        let tu = (x.dx, y.dx, z.dx);
        let tv = (x.dy, y.dy, z.dy);

        // Compose cross product: Tv × Tu for outward normal (explicit vector operation)
        let cross_x = tv.1 * tu.2 - tv.2 * tu.1;
        let cross_y = tv.2 * tu.0 - tv.0 * tu.2;
        let cross_z = tv.0 * tu.1 - tv.1 * tu.0;

        // Compose normal magnitude squared (explicit vector operation)
        let n_len_sq_scalar = cross_x * cross_x + cross_y * cross_y + cross_z * cross_z;
        let epsilon = Field::from(1e-10);
        let inv_n_len = Field::from(1.0) / (n_len_sq_scalar.max(epsilon)).sqrt();

        // Normal as scalar (for Householder value computation)
        let nx = cross_x * inv_n_len;
        let ny = cross_y * inv_n_len;
        let nz = cross_z * inv_n_len;

        // Lift N to Jet3 with approximate derivatives (curvature-aware)
        let curvature_scale = inv_p_len.val; // 1/|P| ≈ 1/t ≈ curvature
        let n_jet_x = Jet3 {
            val: nx,
            dx: x.dx * curvature_scale,
            dy: x.dy * curvature_scale,
            dz: Field::from(0.0),
        };
        let n_jet_y = Jet3 {
            val: ny,
            dx: y.dx * curvature_scale,
            dy: y.dy * curvature_scale,
            dz: Field::from(0.0),
        };
        let n_jet_z = Jet3 {
            val: nz,
            dx: z.dx * curvature_scale,
            dy: z.dy * curvature_scale,
            dz: Field::from(0.0),
        };

        // Compose direction normalization explicitly
        let d_jet_x = x * inv_p_len;
        let d_jet_y = y * inv_p_len;
        let d_jet_z = z * inv_p_len;

        // Compose Householder reflection: R = D - 2(D·N)N
        // Dot product is explicit vector operation
        let d_dot_n = d_jet_x * n_jet_x + d_jet_y * n_jet_y + d_jet_z * n_jet_z;
        let two = Jet3::constant(Field::from(2.0));
        let k = two * d_dot_n;

        let r_x = d_jet_x - k * n_jet_x;
        let r_y = d_jet_y - k * n_jet_y;
        let r_z = d_jet_z - k * n_jet_z;

        // Recurse with curved reflected rays
        self.inner.eval_raw(r_x, r_y, r_z, w)
    }
}

/// Color Reflect: Householder reflection, wraps Discrete material.
///
/// **Compositional Pattern**: Same vector operations as Reflect, but for color output.
#[derive(Clone, Copy)]
pub struct ColorReflect<M> {
    pub inner: M,
}

impl<M: Manifold<Jet3, Output = Discrete>> Manifold<Jet3> for ColorReflect<M> {
    type Output = Discrete;

    #[inline(always)]
    fn eval_raw(&self, x: Jet3, y: Jet3, z: Jet3, w: Jet3) -> Discrete {
        // Compose hit point magnitude explicitly
        let p_len_sq = x * x + y * y + z * z;
        let p_len = p_len_sq.sqrt();
        let one = Jet3::constant(Field::from(1.0));
        let inv_p_len = one / p_len;

        let tu = (x.dx, y.dx, z.dx);
        let tv = (x.dy, y.dy, z.dy);

        // Compose cross product explicitly
        let cross_x = tv.1 * tu.2 - tv.2 * tu.1;
        let cross_y = tv.2 * tu.0 - tv.0 * tu.2;
        let cross_z = tv.0 * tu.1 - tv.1 * tu.0;

        // Compose normal magnitude squared explicitly
        let n_len_sq_scalar = cross_x * cross_x + cross_y * cross_y + cross_z * cross_z;
        let epsilon = Field::from(1e-10);
        let inv_n_len = Field::from(1.0) / (n_len_sq_scalar.max(epsilon)).sqrt();

        let nx = cross_x * inv_n_len;
        let ny = cross_y * inv_n_len;
        let nz = cross_z * inv_n_len;

        let curvature_scale = inv_p_len.val;
        let n_jet_x = Jet3 {
            val: nx,
            dx: x.dx * curvature_scale,
            dy: x.dy * curvature_scale,
            dz: Field::from(0.0),
        };
        let n_jet_y = Jet3 {
            val: ny,
            dx: y.dx * curvature_scale,
            dy: y.dy * curvature_scale,
            dz: Field::from(0.0),
        };
        let n_jet_z = Jet3 {
            val: nz,
            dx: z.dx * curvature_scale,
            dy: z.dy * curvature_scale,
            dz: Field::from(0.0),
        };

        // Compose direction normalization
        let d_jet_x = x * inv_p_len;
        let d_jet_y = y * inv_p_len;
        let d_jet_z = z * inv_p_len;

        // Compose Householder reflection with explicit dot product
        let d_dot_n = d_jet_x * n_jet_x + d_jet_y * n_jet_y + d_jet_z * n_jet_z;
        let two = Jet3::constant(Field::from(2.0));
        let k = two * d_dot_n;

        let r_x = d_jet_x - k * n_jet_x;
        let r_y = d_jet_y - k * n_jet_y;
        let r_z = d_jet_z - k * n_jet_z;

        self.inner.eval_raw(r_x, r_y, r_z, w)
    }
}

/// Checkerboard pattern based on X/Z coordinates.
/// Uses Jet3 derivatives for automatic antialiasing at edges.
///
/// **Compositional Pattern**: Gradient computation uses explicit vector magnitude
/// operations for future FMA/RSQRT optimization on Jet3.
#[derive(Clone, Copy)]
pub struct Checker;

impl Manifold<Jet3> for Checker {
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, x: Jet3, _y: Jet3, z: Jet3, _w: Jet3) -> Field {
        // Which checker cell are we in?
        let cell_x = x.val.floor();
        let cell_z = z.val.floor();
        let sum = cell_x + cell_z;
        let half = sum * Field::from(0.5);
        let fract_half = half - half.floor();
        let is_even = fract_half.abs().lt(Field::from(0.25));

        // Colors
        let color_a = Field::from(0.9);
        let color_b = Field::from(0.2);
        let base_color = (is_even & color_a) | ((!is_even) & color_b);

        // AA: distance to nearest grid line in X and Z
        let fx = x.val - cell_x; // [0, 1) within cell
        let fz = z.val - cell_z;

        // Distance to nearest edge (0.0 or 1.0 boundary)
        let dx_edge = (fx - Field::from(0.5)).abs(); // 0.5 at center, 0.0 at edge
        let dz_edge = (fz - Field::from(0.5)).abs();
        let dist_to_edge = (Field::from(0.5) - dx_edge).min(Field::from(0.5) - dz_edge);

        // Gradient magnitude from Jet3 derivatives (how fast coords change per pixel)
        // This tells us how wide one pixel is in world space
        // Compose vector magnitude explicitly for optimization
        let grad_x = (x.dx * x.dx + x.dy * x.dy + x.dz * x.dz).sqrt();
        let grad_z = (z.dx * z.dx + z.dy * z.dy + z.dz * z.dz).sqrt();
        let pixel_size = grad_x.max(grad_z) + Field::from(0.001);

        // Coverage: how much of the pixel is in this cell vs neighbor
        let coverage = (dist_to_edge / pixel_size).min(Field::from(1.0)).max(Field::from(0.0));

        // Blend with neighbor color at edges
        let neighbor_color = (is_even & color_b) | ((!is_even) & color_a);
        base_color * coverage + neighbor_color * (Field::from(1.0) - coverage)
    }
}

/// Simple Sky Gradient based on Y direction.
///
/// **Compositional Pattern**: Linear interpolation is composed explicitly.
#[derive(Clone, Copy)]
pub struct Sky;

impl Manifold<Jet3> for Sky {
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, _x: Jet3, y: Jet3, _z: Jet3, _w: Jet3) -> Field {
        // y is the Y component of the direction vector
        let t = y.val * Field::from(0.5) + Field::from(0.5);
        let t = t.max(Field::from(0.0)).min(Field::from(1.0));

        // Deep blue to white (linear interpolation)
        Field::from(0.1) + t * Field::from(0.8)
    }
}

/// Color Sky: Blue gradient, outputs packed RGBA.
///
/// **Compositional Pattern**: Color interpolation uses explicit FMA-ready operations.
#[derive(Clone, Copy)]
pub struct ColorSky;

impl Manifold<Jet3> for ColorSky {
    type Output = Discrete;

    #[inline(always)]
    fn eval_raw(&self, _x: Jet3, y: Jet3, _z: Jet3, _w: Jet3) -> Discrete {
        // Compose linear interpolation parameter
        let t = y.val * Field::from(0.5) + Field::from(0.5);
        let t = t.max(Field::from(0.0)).min(Field::from(1.0));

        // Compose color channels via FMA-ready operations
        let r = Field::from(0.7) - t * Field::from(0.5);
        let g = Field::from(0.85) - t * Field::from(0.45);
        let b = Field::from(1.0) - t * Field::from(0.2);
        let a = Field::from(1.0);

        Discrete::pack(r, g, b, a)
    }
}

/// Color Checker: Warm/cool checker with AA, outputs packed RGBA.
///
/// **Compositional Pattern**: Gradient computation and color blending use explicit
/// vector magnitude and FMA-ready operations for future optimization.
#[derive(Clone, Copy)]
pub struct ColorChecker;

impl Manifold<Jet3> for ColorChecker {
    type Output = Discrete;

    #[inline(always)]
    fn eval_raw(&self, x: Jet3, _y: Jet3, z: Jet3, _w: Jet3) -> Discrete {
        let cell_x = x.val.floor();
        let cell_z = z.val.floor();
        let sum = cell_x + cell_z;
        let half = sum * Field::from(0.5);
        let fract_half = half - half.floor();
        let is_even = fract_half.abs().lt(Field::from(0.25));

        let ra = Field::from(0.95);
        let ga = Field::from(0.9);
        let ba = Field::from(0.8);
        let rb = Field::from(0.2);
        let gb = Field::from(0.25);
        let bb = Field::from(0.3);

        let fx = x.val - cell_x;
        let fz = z.val - cell_z;
        let dx_edge = (fx - Field::from(0.5)).abs();
        let dz_edge = (fz - Field::from(0.5)).abs();
        let dist_to_edge = (Field::from(0.5) - dx_edge).min(Field::from(0.5) - dz_edge);

        // Compose vector magnitude explicitly for optimization
        let grad_x = (x.dx * x.dx + x.dy * x.dy + x.dz * x.dz).sqrt();
        let grad_z = (z.dx * z.dx + z.dy * z.dy + z.dz * z.dz).sqrt();
        let pixel_size = grad_x.max(grad_z) + Field::from(0.001);

        let coverage = (dist_to_edge / pixel_size).min(Field::from(1.0)).max(Field::from(0.0));
        let one_minus_cov = Field::from(1.0) - coverage;

        let r_base = (is_even & ra) | ((!is_even) & rb);
        let g_base = (is_even & ga) | ((!is_even) & gb);
        let b_base = (is_even & ba) | ((!is_even) & bb);

        let r_neighbor = (is_even & rb) | ((!is_even) & ra);
        let g_neighbor = (is_even & gb) | ((!is_even) & ga);
        let b_neighbor = (is_even & bb) | ((!is_even) & ba);

        // Compose color blending via FMA-ready operations
        let r = r_base * coverage + r_neighbor * one_minus_cov;
        let g = g_base * coverage + g_neighbor * one_minus_cov;
        let b = b_base * coverage + b_neighbor * one_minus_cov;
        let a = Field::from(1.0);

        Discrete::pack(r, g, b, a)
    }
}
