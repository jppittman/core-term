//! # Analytic 3D Scene Primitives (The "Genius" Architecture)
//!
//! Three-Layer Pull-Based Architecture:
//! 1. Geometry: Returns `t` (Jet3)
//! 2. Surface: Warps `P = ray * t` (Creates tangent frame via Chain Rule)
//! 3. Material: Reconstructs Normal from `P` derivatives
//!
//! ## Idiomatic Unification
//!
//! This module uses the `Selectable` trait to unify Field and Discrete pipelines:
//! - Single `Surface<G, M, B, O>` generic over output type
//! - Single `ScreenToDir<M>` works with any Selectable output
//! - Single `Reflect<M>` works with any Selectable output
//! - Materials like `Checker` and `Sky` are polymorphic
//!
//! The "mullet" architecture still applies: geometry computed ONCE per pixel via Jet3,
//! colors flow as opaque `Discrete`. 3x speedup vs computing geometry per channel.
//!
//! No iteration. Nesting is occlusion.

use pixelflow_core::{Discrete, Field, Jet3, Manifold, Selectable};

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
/// Works with any output type supporting `Selectable`.
#[derive(Clone, Copy)]
pub struct ScreenToDir<M> {
    pub inner: M,
}

impl<M, O> Manifold for ScreenToDir<M>
where
    M: Manifold<Jet3, Output = O>,
    O: Selectable,
{
    type Output = O;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, _z: Field, w: Field) -> O {
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

// ============================================================================
// LAYER 1: Geometry (Returns t)
// ============================================================================

/// Unit sphere centered at origin.
/// Solves |t * ray| = 1  =>  t = 1 / |ray|
#[derive(Clone, Copy)]
pub struct UnitSphere;

impl Manifold<Jet3> for UnitSphere {
    type Output = Jet3;

    #[inline(always)]
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

    #[inline(always)]
    fn eval_raw(&self, _rx: Jet3, ry: Jet3, _rz: Jet3, _w: Jet3) -> Jet3 {
        Jet3::constant(Field::from(self.height)) / ry
    }
}

// ============================================================================
// LAYER 2: Surface (The Warp) - UNIFIED for all Selectable types
// ============================================================================

/// The Glue. Combines Geometry, Material, and Background.
///
/// Performs **The Warp**: `P = ray * t`.
/// Because `t` carries derivatives from Layer 1, and `ray` carries derivatives
/// from Root, `P` automatically contains the Surface Tangent Frame via the Chain Rule.
///
/// Generic over output type `O` via `Selectable` trait.
#[derive(Clone, Copy)]
pub struct Surface<G, M, B> {
    pub geometry: G,     // Returns t
    pub material: M,     // Evaluates at Hit Point P
    pub background: B,   // Evaluates at Ray Direction D (if miss)
}

impl<G, M, B, O> Manifold<Jet3> for Surface<G, M, B>
where
    G: Manifold<Jet3, Output = Jet3>,
    M: Manifold<Jet3, Output = O>,
    B: Manifold<Jet3, Output = O>,
    O: Selectable,
{
    type Output = O;

    #[inline(always)]
    fn eval_raw(&self, rx: Jet3, ry: Jet3, rz: Jet3, w: Jet3) -> O {
        // 1. Ask Geometry for distance t
        let t = self.geometry.eval_raw(rx, ry, rz, w);

        // 2. Check Hit Validity (Mask)
        // Must be positive and finite, and derivatives must be reasonable
        let zero = Field::from(0.0);
        let t_max = Field::from(1e6);
        let deriv_max = Field::from(1e4);

        // Basic validity: positive and not too large
        let valid_t = t.val.gt(zero) & t.val.lt(t_max);

        // Derivative sanity: reject if derivatives are extreme (grazing angle)
        let deriv_mag_sq = t.dx * t.dx + t.dy * t.dy + t.dz * t.dz;
        let valid_deriv = deriv_mag_sq.lt(deriv_max * deriv_max);

        let mask = valid_t & valid_deriv;

        // 3. THE SAFE WARP
        // If we missed, t might be NaN or Inf. Multiplying rx * NaN = NaN.
        // We must sanitize t before the warp to protect the Material arithmetic.
        // (We mask the color later, but we need the math to be valid now).
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

        // 4. Branch Execution with early-exit optimization
        // If all pixels hit, skip background evaluation (30-50% speedup in common case)
        if mask.all() {
            // All hit: only evaluate material
            return self.material.eval_raw(hx, hy, hz, w);
        }

        // If no pixels hit, skip material evaluation (rare case)
        if !mask.any() {
            return self.background.eval_raw(rx, ry, rz, w);
        }

        // Mixed hit/miss: evaluate both and blend via Selectable
        let fg = self.material.eval_raw(hx, hy, hz, w);
        let bg = self.background.eval_raw(rx, ry, rz, w);

        O::select_raw(mask, fg, bg)
    }
}

// ============================================================================
// LAYER 3: Materials - UNIFIED implementations
// ============================================================================

/// Reflect: The Crown Jewel.
/// Reconstructs surface normal from the Tangent Frame implied by the Jet derivatives.
/// Generic over output type via `Selectable`.
#[derive(Clone, Copy)]
pub struct Reflect<M> {
    pub inner: M,
}

impl<M, O> Manifold<Jet3> for Reflect<M>
where
    M: Manifold<Jet3, Output = O>,
    O: Selectable,
{
    type Output = O;

    #[inline(always)]
    fn eval_raw(&self, x: Jet3, y: Jet3, z: Jet3, w: Jet3) -> O {
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

/// Checkerboard pattern based on X/Z coordinates.
/// Uses Jet3 derivatives for automatic antialiasing at edges.
/// Outputs Field (grayscale).
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
/// Outputs Field (grayscale).
#[derive(Clone, Copy)]
pub struct Sky;

impl Manifold<Jet3> for Sky {
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, _x: Jet3, y: Jet3, _z: Jet3, _w: Jet3) -> Field {
        // y is the Y component of the direction vector
        let t = y.val * Field::from(0.5) + Field::from(0.5);
        let t = t.max(Field::from(0.0)).min(Field::from(1.0));

        // Deep blue to white
        Field::from(0.1) + t * Field::from(0.8)
    }
}

/// Color variant: Sky gradient with packed RGBA output.
/// For use with the "mullet" architecture where geometry is computed once.
#[derive(Clone, Copy)]
pub struct ColorSky;

impl Manifold<Jet3> for ColorSky {
    type Output = Discrete;

    #[inline(always)]
    fn eval_raw(&self, _x: Jet3, y: Jet3, _z: Jet3, _w: Jet3) -> Discrete {
        let t = y.val * Field::from(0.5) + Field::from(0.5);
        let t = t.max(Field::from(0.0)).min(Field::from(1.0));

        let r = Field::from(0.7) - t * Field::from(0.5);
        let g = Field::from(0.85) - t * Field::from(0.45);
        let b = Field::from(1.0) - t * Field::from(0.2);
        let a = Field::from(1.0);

        Discrete::pack(r, g, b, a)
    }
}

/// Color variant: Warm/cool checkerboard with packed RGBA output.
/// For use with the "mullet" architecture where geometry is computed once.
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

        let r = r_base * coverage + r_neighbor * one_minus_cov;
        let g = g_base * coverage + g_neighbor * one_minus_cov;
        let b = b_base * coverage + b_neighbor * one_minus_cov;
        let a = Field::from(1.0);

        Discrete::pack(r, g, b, a)
    }
}
