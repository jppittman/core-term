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

use pixelflow_core::jet::Jet3;
use pixelflow_core::*;

// ============================================================================
// LIFT: Field manifold → Jet3 manifold (explicit conversion)
// ============================================================================

/// Lifts a Field-based manifold to work with Jet3 inputs.
///
/// Uses `From<Jet3> for Field` to project jet coordinates to values,
/// discarding derivatives. Use this for constant-valued manifolds
/// (like Color) that don't need derivative information.
#[derive(Clone, Copy)]
pub struct Lift<M>(pub M);

impl<M: Manifold<Field> + Send + Sync> Manifold<Jet3> for Lift<M> {
    type Output = M::Output;

    #[inline(always)]
    fn eval_raw(&self, x: Jet3, y: Jet3, z: Jet3, w: Jet3) -> Self::Output {
        self.0.eval_raw(x.into(), y.into(), z.into(), w.into())
    }
}

// ============================================================================
// HELPER: Lift Field mask to Jet3 manifold for Select conditions
// ============================================================================

/// Wraps a Field mask to implement Manifold<Jet3> for use as a Select condition.
/// This is needed because Select<C, T, F> for Jet3 requires C: Manifold<Jet3, Output = Jet3>.
#[derive(Clone, Copy)]
struct FieldMask(Field);

impl Manifold<Jet3> for FieldMask {
    type Output = Jet3;

    #[inline]
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
#[derive(Clone, Copy)]
pub struct ScreenToDir<M> {
    pub inner: M,
}

impl<M: Manifold<Jet3, Output = Field>> Manifold for ScreenToDir<M> {
    type Output = Field;

    #[inline]
    fn eval_raw(&self, x: Field, y: Field, _z: Field, w: Field) -> Field {
        // 1. Seed Jets from Screen Coords
        // x: varies with screen x (dx=1, dy=0, dz=0)
        let sx = Jet3::x(x);
        // y: varies with screen y (dx=0, dy=1, dz=0)
        let sy = Jet3::y(y);
        // z: constant 1.0 (pinhole focal length, no derivatives)
        let sz = Jet3::constant(Field::from(1.0));

        // 2. Normalize to get Ray Direction
        // The Jet math automatically computes d(dir)/dx and d(dir)/dy
        let len_sq = sx * sx + sy * sy + sz * sz;
        let len = len_sq.sqrt();

        let dx = sx / len;
        let dy = sy / len;
        let dz = sz / len;

        // 3. Pass pure direction Jets to the scene
        self.inner.eval_raw(dx, dy, dz, Jet3::constant(w))
    }
}

/// Converts screen coordinates to ray direction jets, outputting Discrete.
///
/// Same as ScreenToDir but for color (Discrete) pipelines.
#[derive(Clone, Copy)]
pub struct ColorScreenToDir<M> {
    pub inner: M,
}

impl<M: Manifold<Jet3, Output = Discrete>> Manifold for ColorScreenToDir<M> {
    type Output = Discrete;

    #[inline]
    fn eval_raw(&self, x: Field, y: Field, _z: Field, w: Field) -> Discrete {
        let sx = Jet3::x(x);
        let sy = Jet3::y(y);
        let sz = Jet3::constant(Field::from(1.0));

        let len_sq = sx * sx + sy * sy + sz * sz;
        let len = len_sq.sqrt();

        let dx = sx / len;
        let dy = sy / len;
        let dz = sz / len;

        self.inner.eval_raw(dx, dy, dz, Jet3::constant(w))
    }
}

// ============================================================================
// LAYER 1: Geometry (Returns t)
// ============================================================================

/// Unit sphere centered at origin.
/// Solves |t * ray| = 1  =>  t = 1 / |ray|
#[derive(Clone, Copy)]
pub struct UnitSphere;

impl Manifold<Jet3> for UnitSphere {
    type Output = Jet3;

    #[inline]
    fn eval_raw(&self, rx: Jet3, ry: Jet3, rz: Jet3, _w: Jet3) -> Jet3 {
        // Ray is normalized, so |ray| = 1 usually, but let's be robust.
        // t = 1.0 / sqrt(rx^2 + ry^2 + rz^2)
        let len_sq = rx * rx + ry * ry + rz * rz;
        Jet3::constant(Field::from(1.0)) / len_sq.sqrt()
    }
}

/// Sphere at given center with radius.
/// Solves |t * ray - center|² = radius²
/// Returns t for the closer intersection (or negative/NaN if miss)
#[derive(Clone, Copy)]
pub struct SphereAt {
    pub center: (f32, f32, f32),
    pub radius: f32,
}

impl Manifold<Jet3> for SphereAt {
    type Output = Jet3;

    #[inline]
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

        // ray · center
        let d_dot_c = rx * cx + ry * cy + rz * cz;

        // |center|² - radius²
        let c_sq = cx * cx + cy * cy + cz * cz;
        let r_sq = Jet3::constant(Field::from(self.radius * self.radius));
        let c_minus_r = c_sq - r_sq;

        // discriminant = (ray·C)² - (|C|² - r²)
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
        d_dot_c - safe_discriminant.sqrt()
    }
}

/// Horizontal plane at y = height.
/// Solves P.y = height => t * ry = height => t = height / ry
#[derive(Clone, Copy)]
pub struct PlaneGeometry {
    pub height: f32,
}

impl Manifold<Jet3> for PlaneGeometry {
    type Output = Jet3;

    #[inline]
    fn eval_raw(&self, _rx: Jet3, ry: Jet3, _rz: Jet3, _w: Jet3) -> Jet3 {
        Jet3::constant(Field::from(self.height)) / ry
    }
}

/// Height field geometry: z = base_height + scale * f(x, y)
///
/// Single-step intersection: hit base plane, sample height, adjust t.
/// No iteration - just one evaluation of the height manifold.
#[derive(Clone, Copy)]
pub struct HeightFieldGeometry<H> {
    pub height_field: H,
    pub base_height: f32,
    pub scale: f32,
    pub uv_scale: f32, // Maps world (x, y) to (u, v) parameter space
}

impl<H: Manifold<Field, Output = Field>> Manifold<Jet3> for HeightFieldGeometry<H> {
    type Output = Jet3;

    #[inline]
    fn eval_raw(&self, rx: Jet3, ry: Jet3, rz: Jet3, _w: Jet3) -> Jet3 {
        // Step 1: Hit base plane at y = base_height
        let t_plane = Jet3::constant(Field::from(self.base_height)) / ry;

        // Step 2: Get (x, z) world coords at plane hit (note: scene uses y-up)
        let hit_x = rx * t_plane;
        let hit_z = rz * t_plane;

        // Step 3: Map to (u, v) and check bounds
        let uv_scale = Field::from(self.uv_scale);
        let u = (hit_x.val * uv_scale).constant();
        let v = (hit_z.val * uv_scale).constant();

        // Bounds check: (u, v) must be in [0, 1]
        let zero = Field::from(0.0);
        let one = Field::from(1.0);
        let in_bounds = u.ge(zero) & u.le(one) & v.ge(zero) & v.le(one);

        // Sample height field
        let h = self.height_field.eval_raw(u, v, zero, zero);

        // Step 4: Adjust t for height displacement
        let effective_height = (Field::from(self.base_height) + Field::from(self.scale) * h).constant();
        let t_hit = Jet3::constant(effective_height) / ry;

        // Return valid t if in bounds, else negative (miss)
        let miss = Field::from(-1.0);
        Jet3::new(
            in_bounds.select(t_hit.val, miss),
            in_bounds.select(t_hit.dx, miss),
            in_bounds.select(t_hit.dy, miss),
            in_bounds.select(t_hit.dz, miss),
        )
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
#[derive(Clone, Copy)]
pub struct Surface<G, M, B> {
    pub geometry: G,   // Returns t
    pub material: M,   // Evaluates at Hit Point P
    pub background: B, // Evaluates at Ray Direction D (if miss)
}

impl<G, M, B> Manifold<Jet3> for Surface<G, M, B>
where
    G: Manifold<Jet3, Output = Jet3>,
    M: Manifold<Jet3, Output = Field>,
    B: Manifold<Jet3, Output = Field>,
{
    type Output = Field;

    #[inline]
    fn eval_raw(&self, rx: Jet3, ry: Jet3, rz: Jet3, w: Jet3) -> Field {
        // 1. Ask Geometry for distance t
        let t = self.geometry.eval_raw(rx, ry, rz, w);

        // 2. Check Hit Validity (Mask) - is t a valid hit?
        let fzero = Field::from(0.0);
        let t_max = Field::from(1e6);
        let deriv_max = Field::from(1e4);

        // Collapse AST nodes to Field for mask operations
        let valid_t = t.val.gt(fzero) & t.val.lt(t_max);
        let deriv_mag_sq = (t.dx * t.dx + t.dy * t.dy + t.dz * t.dz).constant();
        let valid_deriv = deriv_mag_sq.lt((deriv_max * deriv_max).constant());
        let mask = valid_t & valid_deriv;

        // 3. THE SAFE WARP: P = ray * t (sanitized against NaN/Inf)
        let safe_t = Select {
            cond: FieldMask(mask),
            if_true: t,
            if_false: Jet3::constant(fzero),
        }
        .eval_raw(rx, ry, rz, w);

        let hx = rx * safe_t;
        let hy = ry * safe_t;
        let hz = rz * safe_t;

        // 4. Blend via manifold composition using At + Select
        // At takes Jet3 values as constant manifolds (Jet3 implements Manifold)
        let mat = At {
            inner: &self.material,
            x: hx,
            y: hy,
            z: hz,
            w,
        };
        let bg = At {
            inner: &self.background,
            x: rx,
            y: ry,
            z: rz,
            w,
        };

        // Compose fully: mask selects between material and background manifolds
        Select {
            cond: FieldMask(mask),
            if_true: mat,
            if_false: bg,
        }
        .eval_raw(rx, ry, rz, w)
    }
}

/// Color Surface: geometry + material + background, outputs Discrete.
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

    #[inline]
    fn eval_raw(&self, rx: Jet3, ry: Jet3, rz: Jet3, w: Jet3) -> Discrete {
        let t = self.geometry.eval_raw(rx, ry, rz, w);

        let fzero = Field::from(0.0);
        let t_max = Field::from(1e6);
        let deriv_max = Field::from(1e4);

        // Collapse AST nodes to Field for mask operations
        let valid_t = t.val.gt(fzero) & t.val.lt(t_max);
        let deriv_mag_sq = (t.dx * t.dx + t.dy * t.dy + t.dz * t.dz).constant();
        let valid_deriv = deriv_mag_sq.lt((deriv_max * deriv_max).constant());
        let mask = valid_t & valid_deriv;

        // 3. THE SAFE WARP: P = ray * t (sanitized against NaN/Inf)
        let safe_t = Select {
            cond: FieldMask(mask),
            if_true: t,
            if_false: Jet3::constant(fzero),
        }
        .eval_raw(rx, ry, rz, w);

        let hx = rx * safe_t;
        let hy = ry * safe_t;
        let hz = rz * safe_t;

        // 4. Blend via manifold composition using At + Select
        let mat = At {
            inner: &self.material,
            x: hx,
            y: hy,
            z: hz,
            w,
        };
        let bg = At {
            inner: &self.background,
            x: rx,
            y: ry,
            z: rz,
            w,
        };

        Select {
            cond: FieldMask(mask),
            if_true: mat,
            if_false: bg,
        }
        .eval_raw(rx, ry, rz, w)
    }
}

// ============================================================================
// SCENE COMPOSITION: Union via priority order
// ============================================================================

/// A Scene separates hit detection (mask) from appearance (color).
///
/// This enables Union composition: check S1 first, if miss check S2.
/// "First hit in scene graph wins" - not distance-based, but priority-based.
pub trait Scene {
    /// Mask manifold: evaluates to positive where ray hits this scene.
    type Mask: Manifold<Jet3, Output = Field>;
    /// Color manifold: evaluates to the color at the hit point.
    type Color: Manifold<Jet3, Output = Discrete>;

    fn mask(&self) -> Self::Mask;
    fn color(&self) -> Self::Color;
}

/// Union of two scenes: first hit wins.
///
/// Evaluates S1's mask first. If hit, use S1's color.
/// Otherwise, evaluate S2's mask. If hit, use S2's color.
/// Otherwise, use background.
#[derive(Clone, Copy)]
pub struct Union<S1, S2, B> {
    pub first: S1,
    pub second: S2,
    pub background: B,
}

impl<S1, S2, B> Manifold<Jet3> for Union<S1, S2, B>
where
    S1: Scene + Send + Sync,
    S2: Scene + Send + Sync,
    B: Manifold<Jet3, Output = Discrete>,
{
    type Output = Discrete;

    #[inline]
    fn eval_raw(&self, rx: Jet3, ry: Jet3, rz: Jet3, w: Jet3) -> Discrete {
        // First hit wins: nested Select
        // Select(S1.mask, S1.color, Select(S2.mask, S2.color, background))
        let m1 = self.first.mask();
        let c1 = self.first.color();
        let m2 = self.second.mask();
        let c2 = self.second.color();

        let mask1 = m1.eval_raw(rx, ry, rz, w);
        let mask2 = m2.eval_raw(rx, ry, rz, w);

        // Inner select: S2 vs background
        let color2 = c2.eval_raw(rx, ry, rz, w);
        let bg_color = self.background.eval_raw(rx, ry, rz, w);
        let inner = Discrete::select(mask2, color2, bg_color);

        // Outer select: S1 vs inner
        let color1 = c1.eval_raw(rx, ry, rz, w);
        Discrete::select(mask1, color1, inner)
    }
}

/// Simple scene wrapper: geometry + material with explicit mask exposure.
///
/// Unlike ColorSurface which hides the mask, SceneObject exposes it
/// for use in Union composition.
#[derive(Clone, Copy)]
pub struct SceneObject<G, M> {
    pub geometry: G,
    pub material: M,
}

/// Mask manifold for geometry hit detection.
#[derive(Clone, Copy)]
pub struct GeometryMask<G> {
    geometry: G,
}

impl<G: Manifold<Jet3, Output = Jet3>> Manifold<Jet3> for GeometryMask<G> {
    type Output = Field;

    #[inline]
    fn eval_raw(&self, rx: Jet3, ry: Jet3, rz: Jet3, w: Jet3) -> Field {
        let t = self.geometry.eval_raw(rx, ry, rz, w);

        let fzero = Field::from(0.0);
        let t_max = Field::from(1e6);
        let deriv_max = Field::from(1e4);

        let valid_t = t.val.gt(fzero) & t.val.lt(t_max);
        let deriv_mag_sq = (t.dx * t.dx + t.dy * t.dy + t.dz * t.dz).constant();
        let valid_deriv = deriv_mag_sq.lt((deriv_max * deriv_max).constant());
        valid_t & valid_deriv
    }
}

/// Color manifold for material evaluation at hit point.
#[derive(Clone, Copy)]
pub struct GeometryColor<G, M> {
    geometry: G,
    material: M,
}

impl<G, M> Manifold<Jet3> for GeometryColor<G, M>
where
    G: Manifold<Jet3, Output = Jet3>,
    M: Manifold<Jet3, Output = Discrete>,
{
    type Output = Discrete;

    #[inline]
    fn eval_raw(&self, rx: Jet3, ry: Jet3, rz: Jet3, w: Jet3) -> Discrete {
        let t = self.geometry.eval_raw(rx, ry, rz, w);

        // Compute hit point: P = ray * t
        let hx = rx * t;
        let hy = ry * t;
        let hz = rz * t;

        // Evaluate material at hit point
        self.material.eval_raw(hx, hy, hz, w)
    }
}

impl<G, M> Scene for SceneObject<G, M>
where
    G: Manifold<Jet3, Output = Jet3> + Clone + Copy,
    M: Manifold<Jet3, Output = Discrete> + Clone + Copy,
{
    type Mask = GeometryMask<G>;
    type Color = GeometryColor<G, M>;

    fn mask(&self) -> Self::Mask {
        GeometryMask {
            geometry: self.geometry,
        }
    }

    fn color(&self) -> Self::Color {
        GeometryColor {
            geometry: self.geometry,
            material: self.material,
        }
    }
}

// ============================================================================
// LAYER 3: Materials
// ============================================================================

/// Reflect: The Crown Jewel.
/// Reconstructs surface normal from the Tangent Frame implied by the Jet derivatives.
#[derive(Clone, Copy)]
pub struct Reflect<M> {
    pub inner: M,
}

impl<M: Manifold<Jet3, Output = Field>> Manifold<Jet3> for Reflect<M> {
    type Output = Field;

    #[inline]
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

        let p_len_sq = x * x + y * y + z * z;
        let p_len = p_len_sq.sqrt();
        let one = Jet3::constant(Field::from(1.0));
        let inv_p_len = one / p_len;

        // Extract tangent vectors (as scalars)
        let tu = (x.dx, y.dx, z.dx);
        let tv = (x.dy, y.dy, z.dy);

        // Cross product Tv × Tu for outward normal
        let cross_x = tv.1 * tu.2 - tv.2 * tu.1;
        let cross_y = tv.2 * tu.0 - tv.0 * tu.2;
        let cross_z = tv.0 * tu.1 - tv.1 * tu.0;

        // Collapse AST nodes to Field for scalar operations
        let fzero = Field::from(0.0);
        let n_len_sq_scalar =
            (cross_x * cross_x + cross_y * cross_y + cross_z * cross_z).constant();
        let inv_n_len = n_len_sq_scalar
            .max(Field::from(1e-10))
            .sqrt()
            .rsqrt()
            .constant();

        // Normal as scalar (for Householder value computation)
        let nx = (cross_x * inv_n_len).constant();
        let ny = (cross_y * inv_n_len).constant();
        let nz = (cross_z * inv_n_len).constant();

        // Lift N to Jet3 with approximate derivatives (curvature-aware)
        let curvature_scale = inv_p_len.val; // 1/|P| ≈ 1/t ≈ curvature
        let n_jet_x = Jet3 {
            val: nx,
            dx: (x.dx * curvature_scale).constant(),
            dy: (x.dy * curvature_scale).constant(),
            dz: fzero,
        };
        let n_jet_y = Jet3 {
            val: ny,
            dx: (y.dx * curvature_scale).constant(),
            dy: (y.dy * curvature_scale).constant(),
            dz: fzero,
        };
        let n_jet_z = Jet3 {
            val: nz,
            dx: (z.dx * curvature_scale).constant(),
            dy: (z.dy * curvature_scale).constant(),
            dz: fzero,
        };

        // D as Jets (normalized P)
        let d_jet_x = x * inv_p_len;
        let d_jet_y = y * inv_p_len;
        let d_jet_z = z * inv_p_len;

        // Householder Reflection: R = D - 2(D·N)N
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
#[derive(Clone, Copy)]
pub struct ColorReflect<M> {
    pub inner: M,
}

impl<M: Manifold<Jet3, Output = Discrete>> Manifold<Jet3> for ColorReflect<M> {
    type Output = Discrete;

    #[inline]
    fn eval_raw(&self, x: Jet3, y: Jet3, z: Jet3, w: Jet3) -> Discrete {
        let p_len_sq = x * x + y * y + z * z;
        let p_len = p_len_sq.sqrt();
        let one = Jet3::constant(Field::from(1.0));
        let inv_p_len = one / p_len;

        let tu = (x.dx, y.dx, z.dx);
        let tv = (x.dy, y.dy, z.dy);

        let cross_x = tv.1 * tu.2 - tv.2 * tu.1;
        let cross_y = tv.2 * tu.0 - tv.0 * tu.2;
        let cross_z = tv.0 * tu.1 - tv.1 * tu.0;

        // Collapse AST nodes to Field for scalar operations
        let fzero = Field::from(0.0);
        let n_len_sq_scalar =
            (cross_x * cross_x + cross_y * cross_y + cross_z * cross_z).constant();
        let inv_n_len = n_len_sq_scalar
            .max(Field::from(1e-10))
            .sqrt()
            .rsqrt()
            .constant();

        let nx = (cross_x * inv_n_len).constant();
        let ny = (cross_y * inv_n_len).constant();
        let nz = (cross_z * inv_n_len).constant();

        let curvature_scale = inv_p_len.val;
        let n_jet_x = Jet3 {
            val: nx,
            dx: (x.dx * curvature_scale).constant(),
            dy: (x.dy * curvature_scale).constant(),
            dz: fzero,
        };
        let n_jet_y = Jet3 {
            val: ny,
            dx: (y.dx * curvature_scale).constant(),
            dy: (y.dy * curvature_scale).constant(),
            dz: fzero,
        };
        let n_jet_z = Jet3 {
            val: nz,
            dx: (z.dx * curvature_scale).constant(),
            dy: (z.dy * curvature_scale).constant(),
            dz: fzero,
        };

        let d_jet_x = x * inv_p_len;
        let d_jet_y = y * inv_p_len;
        let d_jet_z = z * inv_p_len;

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
#[derive(Clone, Copy)]
pub struct Checker;

impl Manifold<Jet3> for Checker {
    type Output = Field;

    #[inline]
    fn eval_raw(&self, x: Jet3, _y: Jet3, z: Jet3, _w: Jet3) -> Field {
        // Which checker cell are we in?
        let cell_x = x.val.floor();
        let cell_z = z.val.floor();
        let sum = cell_x + cell_z;
        let half = sum * Field::from(0.5);
        let fract_half = half - half.floor();
        let is_even = fract_half.abs().lt(Field::from(0.25));

        // Colors - use select for branchless conditional
        let color_a = Field::from(0.9);
        let color_b = Field::from(0.2);
        let base_color = is_even.clone().select(color_a, color_b);

        // AA: distance to nearest grid line in X and Z
        let fx = x.val - cell_x; // [0, 1) within cell
        let fz = z.val - cell_z;

        // Distance to nearest edge (0.0 or 1.0 boundary)
        let dx_edge = (fx - Field::from(0.5)).abs(); // 0.5 at center, 0.0 at edge
        let dz_edge = (fz - Field::from(0.5)).abs();
        let dist_to_edge = (Field::from(0.5) - dx_edge).min(Field::from(0.5) - dz_edge);

        // Gradient magnitude from Jet3 derivatives (how fast coords change per pixel)
        // Build expression once with coordinate variables, evaluate at Jet derivative components
        let grad_mag = (X * X + Y * Y + Z * Z).sqrt();
        let grad_x = grad_mag.at(x.dx, x.dy, x.dz, Field::from(0.0)).eval();
        let grad_z = grad_mag.at(z.dx, z.dy, z.dz, Field::from(0.0)).eval();
        // This tells us how wide one pixel is in world space
        let pixel_size = grad_x.max(grad_z) + Field::from(0.001);

        // Coverage: how much of the pixel is in this cell vs neighbor
        let coverage = (dist_to_edge / pixel_size)
            .min(Field::from(1.0))
            .max(Field::from(0.0));

        // Blend with neighbor color at edges
        let neighbor_color = is_even.select(color_b, color_a);
        (base_color * coverage + neighbor_color * (Field::from(1.0) - coverage))
            .at(x.val, Field::from(0.0), z.val, Field::from(0.0))
            .eval()
    }
}

/// Simple Sky Gradient based on Y direction.
#[derive(Clone, Copy)]
pub struct Sky;

impl Manifold<Jet3> for Sky {
    type Output = Field;

    #[inline]
    fn eval_raw(&self, _x: Jet3, y: Jet3, _z: Jet3, _w: Jet3) -> Field {
        // y is the Y component of the direction vector
        let t = y.val * Field::from(0.5) + Field::from(0.5);
        let t = t.max(Field::from(0.0)).min(Field::from(1.0));

        // Deep blue to white
        let fzero = Field::from(0.0);
        (Field::from(0.1) + t * Field::from(0.8)).constant()
    }
}

/// Color Sky: Blue gradient, outputs packed RGBA.
#[derive(Clone, Copy)]
pub struct ColorSky;

impl Manifold<Jet3> for ColorSky {
    type Output = Discrete;

    #[inline]
    fn eval_raw(&self, _x: Jet3, y: Jet3, _z: Jet3, _w: Jet3) -> Discrete {
        let fzero = Field::from(0.0);
        let t = y.val * Field::from(0.5) + Field::from(0.5);
        let t = t.max(Field::from(0.0)).min(Field::from(1.0));

        let r = (Field::from(0.7) - t.clone() * Field::from(0.5)).constant();
        let g = (Field::from(0.85) - t.clone() * Field::from(0.45)).constant();
        let b = (Field::from(1.0) - t * Field::from(0.2)).constant();
        let a = Field::from(1.0);

        Discrete::pack(r, g, b, a)
    }
}

/// Color Checker: Warm/cool checker with AA, outputs packed RGBA.
#[derive(Clone, Copy)]
pub struct ColorChecker;

impl Manifold<Jet3> for ColorChecker {
    type Output = Discrete;

    #[inline]
    fn eval_raw(&self, x: Jet3, _y: Jet3, z: Jet3, _w: Jet3) -> Discrete {
        let fzero = Field::from(0.0);
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

        let grad_x = (x.dx * x.dx + x.dy * x.dy + x.dz * x.dz).sqrt();
        let grad_z = (z.dx * z.dx + z.dy * z.dy + z.dz * z.dz).sqrt();
        let pixel_size = grad_x.max(grad_z) + Field::from(0.001);

        let coverage = (dist_to_edge / pixel_size)
            .min(Field::from(1.0))
            .max(Field::from(0.0));
        let one_minus_cov = Field::from(1.0) - coverage.clone();

        // Use select for branchless color choice
        let r_base = is_even.clone().select(ra, rb);
        let g_base = is_even.clone().select(ga, gb);
        let b_base = is_even.clone().select(ba, bb);

        let r_neighbor = is_even.clone().select(rb, ra);
        let g_neighbor = is_even.clone().select(gb, ga);
        let b_neighbor = is_even.select(bb, ba);

        let r = (r_base * coverage.clone() + r_neighbor * one_minus_cov.clone()).constant();
        let g = (g_base * coverage.clone() + g_neighbor * one_minus_cov.clone()).constant();
        let b = (b_base * coverage + b_neighbor * one_minus_cov).constant();
        let a = Field::from(1.0);

        Discrete::pack(r, g, b, a)
    }
}
