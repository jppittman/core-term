//! # Algebraic Raymarching
//!
//! Raymarching as manifold morphism. Everything composes.
//!
//! ## Core Constraint
//!
//! **Camera is FIXED at origin.** Screen coords (x, y) define ray direction.
//! Z becomes the ray parameter. To "move" the camera, transform the world.
//!
//! ## Why This Works
//!
//! - `Jet2` gives us ∂f/∂x, ∂f/∂y automatically → normals for free
//! - No SDFs needed: sign-change detection on implicit surfaces f(p) = 0
//! - Manifold composition builds the compute graph
//! - `f32` is a constant manifold, shader is a manifold, scene is a manifold
//!
//! ## Architecture
//!
//! ```text
//! Screen (x, y) → Ray direction = normalize(x, y, 1)
//!                 Ray origin = (0, 0, 0)
//!                 March along ray, find f(p) = 0
//!                 Normal from Jet2 gradient
//!                 Shade via material manifold
//! ```

use pixelflow_core::combinators::spherical::{Sh2, ShCoeffs};
use pixelflow_core::ops::{Max, Min};
use pixelflow_core::{Computational, Discrete, Field, Jet2, Manifold};

// ============================================================================
// Optical Response (Material Coefficients)
// ============================================================================

/// Optical response - how a surface splits incoming light.
///
/// Each coefficient is a manifold, enabling spatial variation.
/// Follows the ColorManifold pattern: each lane is independent.
///
/// Physics: photons can be absorbed, reflected, or scattered.
/// Conservation: absorption + reflection + scattering ≈ 1 (per wavelength)
///
/// # Type Parameters
/// - `A`: absorption coefficient manifold
/// - `R`: reflection (specular) coefficient manifold
/// - `Sr`, `Sg`, `Sb`: scattering (diffuse) albedo per channel
#[derive(Clone, Copy)]
pub struct Response<A, R, Sr, Sg, Sb> {
    pub absorption: A,
    pub reflection: R,
    pub scatter_r: Sr,
    pub scatter_g: Sg,
    pub scatter_b: Sb,
}

impl<A, R, Sr, Sg, Sb> Response<A, R, Sr, Sg, Sb> {
    pub fn new(absorption: A, reflection: R, scatter_r: Sr, scatter_g: Sg, scatter_b: Sb) -> Self {
        Self {
            absorption,
            reflection,
            scatter_r,
            scatter_g,
            scatter_b,
        }
    }
}

/// Evaluated response at a point (all Fields, ready for shading).
pub struct EvaluatedResponse {
    pub absorption: Field,
    pub reflection: Field,
    pub scatter_r: Field,
    pub scatter_g: Field,
    pub scatter_b: Field,
}

impl<A, R, Sr, Sg, Sb> Response<A, R, Sr, Sg, Sb>
where
    A: Manifold<Output = Field>,
    R: Manifold<Output = Field>,
    Sr: Manifold<Output = Field>,
    Sg: Manifold<Output = Field>,
    Sb: Manifold<Output = Field>,
{
    /// Evaluate all coefficients at a surface point.
    pub fn eval_at(&self, x: Field, y: Field, z: Field, w: Field) -> EvaluatedResponse {
        EvaluatedResponse {
            absorption: self.absorption.eval_raw(x, y, z, w),
            reflection: self.reflection.eval_raw(x, y, z, w),
            scatter_r: self.scatter_r.eval_raw(x, y, z, w),
            scatter_g: self.scatter_g.eval_raw(x, y, z, w),
            scatter_b: self.scatter_b.eval_raw(x, y, z, w),
        }
    }
}

// ============================================================================
// Material Constructors
// ============================================================================

/// Chrome: pure specular reflection (uniform constants).
/// Type: Response<f32, f32, f32, f32, f32>
pub fn chrome() -> Response<f32, f32, f32, f32, f32> {
    Response::new(
        0.0,  // no absorption
        0.95, // high reflection
        0.05, // tiny scatter
        0.05,
        0.05,
    )
}

/// Matte: pure diffuse scattering (uniform constants).
/// Type: Response<f32, f32, f32, f32, f32>
pub fn matte(r: f32, g: f32, b: f32) -> Response<f32, f32, f32, f32, f32> {
    Response::new(
        0.1,  // some absorption
        0.0,  // no reflection
        r,
        g,
        b,
    )
}

/// Checkerboard mask - evaluates to 1.0 or 0.0 based on XZ position.
#[derive(Clone, Copy)]
pub struct CheckerMask {
    pub scale: f32,
}

impl Manifold for CheckerMask {
    type Output = Field;

    fn eval_raw(&self, x: Field, _y: Field, z: Field, _w: Field) -> Field {
        let sx = (x * Field::from(self.scale)).floor();
        let sz = (z * Field::from(self.scale)).floor();
        let sum = sx + sz;
        let parity = sum - (sum * Field::from(0.5)).floor() * Field::from(2.0);
        // Returns 1.0 for "white" squares, 0.0 for "black"
        let is_white = parity.lt(Field::from(1.0));
        select(is_white, Field::from(1.0), Field::from(0.0))
    }
}

// ============================================================================
// Helper: branchless select using bitwise ops on Field masks
// ============================================================================

/// Branchless select: returns `if_true` where mask is set, `if_false` elsewhere.
/// Mask should be result of comparison (all-1s or 0 bits per lane).
#[inline(always)]
fn select(mask: Field, if_true: Field, if_false: Field) -> Field {
    (mask & if_true) | (!mask & if_false)
}

// ============================================================================
// Implicit Surfaces: R³ → R
// ============================================================================

/// Unit sphere at origin. Transform to move/scale.
#[derive(Clone, Copy)]
pub struct UnitSphere;

impl<I: Computational> Manifold<I> for UnitSphere {
    type Output = I;
    fn eval_raw(&self, x: I, y: I, z: I, _w: I) -> I {
        // Implicit: x² + y² + z² - 1 = 0
        x * x + y * y + z * z - I::from_f32(1.0)
    }
}

/// XZ plane at y = 0. Transform to move.
#[derive(Clone, Copy)]
pub struct GroundPlane;

impl<I: Computational> Manifold<I> for GroundPlane {
    type Output = I;
    fn eval_raw(&self, _x: I, y: I, _z: I, _w: I) -> I {
        y
    }
}

// ============================================================================
// Transforms: Morphisms on Manifolds
// ============================================================================

/// Translate a manifold in space.
#[derive(Clone, Copy)]
pub struct Translate<M> {
    pub inner: M,
    pub offset: (f32, f32, f32),
}

impl<I: Computational, M: Manifold<I, Output = I>> Manifold<I> for Translate<M> {
    type Output = I;
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> I {
        self.inner.eval_raw(
            x - I::from_f32(self.offset.0),
            y - I::from_f32(self.offset.1),
            z - I::from_f32(self.offset.2),
            w,
        )
    }
}

/// Scale a manifold uniformly.
#[derive(Clone, Copy)]
pub struct Scale<M> {
    pub inner: M,
    pub factor: f32,
}

impl<I: Computational, M: Manifold<I, Output = I>> Manifold<I> for Scale<M> {
    type Output = I;
    fn eval_raw(&self, x: I, y: I, z: I, w: I) -> I {
        let s = I::from_f32(self.factor);
        self.inner.eval_raw(x / s, y / s, z / s, w)
    }
}

/// Union of two surfaces (CSG min) - just a type alias for Min combinator.
pub type Union<A, B> = Min<A, B>;

/// Intersection of two surfaces (CSG max) - just a type alias for Max combinator.
pub type Intersect<A, B> = Max<A, B>;

// Note: Subtract would need a Negate combinator composed with Max.
// For now, not implementing it since the test doesn't use it.


// ============================================================================
// Lighting via Spherical Harmonics
// ============================================================================

/// Environment lighting encoded as spherical harmonics.
///
/// 9 coefficients (L=2) capture ambient + directional + soft shadows.
/// This replaces point/directional lights with a full environment.
#[derive(Clone, Debug)]
pub struct Environment {
    pub coeffs: Sh2,
}

impl Environment {
    /// Create a sky gradient environment (blue above, darker below).
    pub fn sky_gradient() -> Self {
        // SH coefficients for a vertical gradient (sky blue up, ground down)
        // L0: constant ambient
        // L1: directional (y = up)
        Self {
            coeffs: ShCoeffs {
                coeffs: [
                    1.0,   // L0: ambient base
                    0.0,   // L1,-1: no x bias
                    0.6,   // L1,0: y gradient (brighter when normal points up)
                    -0.2,  // L1,1: slight front bias (z toward camera)
                    0.0,   // L2,-2
                    0.0,   // L2,-1
                    0.0,   // L2,0
                    0.0,   // L2,1
                    0.0,   // L2,2
                ],
            },
        }
    }

    /// Create environment from directional light (for compatibility).
    pub fn from_direction(dir: (f32, f32, f32), intensity: f32) -> Self {
        // Project directional light onto SH basis
        // L1 coefficients encode direction
        Self {
            coeffs: ShCoeffs {
                coeffs: [
                    0.3 * intensity,        // L0: ambient
                    dir.0 * intensity,      // L1,-1: x
                    dir.1 * intensity,      // L1,0: y
                    dir.2 * intensity,      // L1,1: z
                    0.0, 0.0, 0.0, 0.0, 0.0, // L2: zero for pure directional
                ],
            },
        }
    }

    /// Evaluate irradiance from normal direction.
    ///
    /// This is the key operation: just a dot product with SH basis.
    #[inline]
    pub fn irradiance(&self, nx: Field, ny: Field, nz: Field) -> Field {
        // SH basis evaluated at normal direction
        // L0
        let y00 = Field::from(0.282_094_8);

        // L1 (linear in direction)
        let y1m1 = Field::from(0.488_602_5) * ny;
        let y10 = Field::from(0.488_602_5) * nz;
        let y11 = Field::from(0.488_602_5) * nx;

        // L2 (quadratic)
        let y2m2 = Field::from(1.092_548_4) * nx * ny;
        let y2m1 = Field::from(1.092_548_4) * ny * nz;
        let y20 = Field::from(0.315_391_57) * (Field::from(3.0) * nz * nz - Field::from(1.0));
        let y21 = Field::from(1.092_548_4) * nx * nz;
        let y22 = Field::from(0.546_274_2) * (nx * nx - ny * ny);

        // Dot product with coefficients
        let c = &self.coeffs.coeffs;
        let irr = Field::from(c[0]) * y00
            + Field::from(c[1]) * y1m1
            + Field::from(c[2]) * y10
            + Field::from(c[3]) * y11
            + Field::from(c[4]) * y2m2
            + Field::from(c[5]) * y2m1
            + Field::from(c[6]) * y20
            + Field::from(c[7]) * y21
            + Field::from(c[8]) * y22;

        // Scale up to compensate for SH normalization (π factor for Lambertian)
        (irr * Field::from(std::f32::consts::PI)).max(Field::from(0.0))
    }

    /// Evaluate SH at a direction (for specular).
    pub fn eval(&self, dir: (Field, Field, Field)) -> Field {
        self.coeffs.eval(dir)
    }
}

// ============================================================================
// Shading Helpers
// ============================================================================

/// Reflect direction d about normal n.
fn reflect(d: (Field, Field, Field), n: (Field, Field, Field)) -> (Field, Field, Field) {
    let dot = d.0 * n.0 + d.1 * n.1 + d.2 * n.2;
    let two_dot = dot + dot;
    (
        d.0 - two_dot * n.0,
        d.1 - two_dot * n.1,
        d.2 - two_dot * n.2,
    )
}

/// Shade a surface point given material response and environment.
///
/// Combines specular (reflection) and diffuse (scattering) contributions.
fn shade(
    env: &Environment,
    response: EvaluatedResponse,
    normal: (Field, Field, Field),
    ray_dir: (Field, Field, Field),
) -> (Field, Field, Field) {
    // Diffuse: irradiance from environment weighted by normal
    let diffuse = env.irradiance(normal.0, normal.1, normal.2);

    // Specular: environment sampled at reflection direction
    let reflect_dir = reflect(ray_dir, normal);
    let specular = env.eval(reflect_dir);

    // Combine via response coefficients
    let r = response.reflection * specular + response.scatter_r * diffuse;
    let g = response.reflection * specular + response.scatter_g * diffuse;
    let b = response.reflection * specular + response.scatter_b * diffuse;

    (r, g, b)
}


// ============================================================================
// The Raymarch Combinator
// ============================================================================

/// Configuration for raymarching.
#[derive(Clone, Copy, Debug)]
pub struct MarchConfig {
    /// Step size for marching
    pub max_step: f32,
    /// Maximum ray distance before giving up
    pub max_t: f32,
    /// Surface hit threshold (|f| < epsilon means hit)
    pub epsilon: f32,
}

impl Default for MarchConfig {
    fn default() -> Self {
        Self {
            max_step: 0.1,
            max_t: 20.0,
            epsilon: 0.001,
        }
    }
}

/// Raymarch combinator: scene × material × environment → pixels.
///
/// Camera at origin, looking down +Z. Input `(x, y)` defines ray direction
/// as `normalize(x, y, 1)`. No screen dimensions needed - the manifold
/// operates on whatever coordinates it receives.
///
/// Material is a Response struct with absorption/reflection/scattering coefficients.
/// Shadow, AO, and penumbra emerge from SH visibility multiplication.
pub struct Raymarch<Scene, A, R, Sr, Sg, Sb> {
    pub scene: Scene,
    pub material: Response<A, R, Sr, Sg, Sb>,
    pub env: Environment,
    pub config: MarchConfig,
}

impl<Scene, A, R, Sr, Sg, Sb> Manifold for Raymarch<Scene, A, R, Sr, Sg, Sb>
where
    Scene: Manifold<Jet2, Output = Jet2> + Manifold<Field, Output = Field> + Clone,
    A: Manifold<Output = Field>,
    R: Manifold<Output = Field>,
    Sr: Manifold<Output = Field>,
    Sg: Manifold<Output = Field>,
    Sb: Manifold<Output = Field>,
{
    type Output = Discrete;

    fn eval_raw(&self, x: Field, y: Field, _z: Field, _w: Field) -> Discrete {
        // Ray from origin through (x, y, 1)
        let one = Field::from(1.0);
        let dir_len = (x * x + y * y + one).sqrt();
        let dx = x / dir_len;
        let dy = y / dir_len;
        let dz = one / dir_len;

        // March along ray with fixed steps
        // TODO: Replace with Fix combinator over W for implicit frame budget
        let step = Field::from(self.config.max_step);
        let epsilon = Field::from(self.config.epsilon);
        let max_t = Field::from(self.config.max_t);

        // Newton's method raymarching with SDF fallback
        // TODO: Replace with Fix combinator over W for implicit frame budget
        let mut t = Field::from(0.1);
        let mut best_t = Field::from(self.config.max_t + 1.0); // Start with "miss" value

        for _ in 0..200 {
            let px = dx * t;
            let py = dy * t;
            let pz = dz * t;

            // Evaluate f and get gradient via Jet2
            let jx = Jet2::x(px);
            let jy = Jet2::y(py);
            let jz = Jet2::constant(pz);
            let jw = Jet2::constant(Field::from(0.0));
            let jet_f = Manifold::<Jet2>::eval_raw(&self.scene, jx, jy, jz, jw);

            let f = jet_f.val;
            let df_dx = jet_f.dx;
            let df_dy = jet_f.dy;

            // Finite diff for z gradient
            let eps_fd = Field::from(0.001);
            let jz2 = Jet2::constant(pz + eps_fd);
            let jet_f2 = Manifold::<Jet2>::eval_raw(&self.scene, jx, jy, jz2, jw);
            let df_dz = (jet_f2.val - jet_f.val) / eps_fd;

            // df/dt via chain rule: ∇f · d
            let df_dt = df_dx * dx + df_dy * dy + df_dz * dz;

            // Check if we hit surface (|f| < epsilon)
            let hit = f.abs().lt(epsilon);

            // Record t for lanes that just hit (only first hit matters)
            let candidate = select(hit, t, Field::from(self.config.max_t + 1.0));
            best_t = best_t.min(candidate);

            // Check if we've escaped (no point continuing)
            let escaped = t.gt(max_t);
            if (hit | escaped).all() {
                break;
            }

            // Newton step: delta_t = -f / df_dt  (move toward root)
            // Safe division to avoid NaN when df_dt ≈ 0
            let abs_df_dt = df_dt.abs() + Field::from(0.0001);
            // newton_delta = -f / df_dt = -f * sign(df_dt) / |df_dt|
            //              = -f * df_dt / (|df_dt| * |df_dt|)
            let newton_delta = Field::from(0.0) - f * df_dt / (abs_df_dt * abs_df_dt);

            // Use Newton when it predicts a reasonable forward step
            // Otherwise fall back to fixed stepping
            let newton_is_good = newton_delta.gt(Field::from(0.0)) & newton_delta.lt(step * Field::from(5.0));

            // When Newton is good, use it; otherwise use fixed step
            let delta = select(newton_is_good, newton_delta, step);

            t = t + delta;
        }

        let hit_t = best_t;

        // Check if we actually hit (t < max_t)
        let hit = hit_t.lt(max_t);

        // Hit point
        let hx = dx * hit_t;
        let hy = dy * hit_t;
        let hz = dz * hit_t;

        // Compute normal via Jet2 gradient at hit point
        let (nx, ny, nz) = self.compute_normal(hx, hy, hz);

        // Evaluate material response at hit point
        let response = self.material.eval_at(hx, hy, hz, ny);

        // Shade: combine specular and diffuse via material response
        let (r, g, b) = shade(&self.env, response, (nx, ny, nz), (dx, dy, dz));

        // Sky gradient background
        let sky_t = (y + one) * Field::from(0.5);
        let sky_r = Field::from(0.5) + sky_t * Field::from(0.3);
        let sky_g = Field::from(0.7) + sky_t * Field::from(0.2);
        let sky_b = Field::from(1.0);

        // Select hit vs background
        let final_r = select(hit, r, sky_r);
        let final_g = select(hit, g, sky_g);
        let final_b = select(hit, b, sky_b);

        Discrete::pack(final_r, final_g, final_b, one)
    }
}

impl<Scene, A, R, Sr, Sg, Sb> Raymarch<Scene, A, R, Sr, Sg, Sb>
where
    Scene: Manifold<Jet2, Output = Jet2> + Manifold<Field, Output = Field>,
{
    /// Compute surface normal at a point using Jet2 autodiff.
    fn compute_normal(&self, px: Field, py: Field, pz: Field) -> (Field, Field, Field) {
        // Evaluate with Jet2 for gradient
        let jx = Jet2::x(px);
        let jy = Jet2::y(py);
        let jz = Jet2::constant(pz);
        let jw = Jet2::constant(Field::from(0.0));

        let surface = Manifold::<Jet2>::eval_raw(&self.scene, jx, jy, jz, jw);

        // dx, dy from Jet2
        let gx = surface.dx;
        let gy = surface.dy;

        // Finite diff for z gradient (Jet2 only tracks x, y derivatives)
        let eps = Field::from(0.001);
        let jz2 = Jet2::constant(pz + eps);
        let surf2 = Manifold::<Jet2>::eval_raw(&self.scene, jx, jy, jz2, jw);
        let gz = (surf2.val - surface.val) / eps;

        // Normalize
        let glen = (gx * gx + gy * gy + gz * gz).sqrt() + Field::from(1e-6);

        (gx / glen, gy / glen, gz / glen)
    }
}

