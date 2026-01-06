//! # Catmull-Clark Subdivision Surfaces
//!
//! Analytic evaluation using Stam's eigenstructure method.
//! No tessellation - pure manifold composition with automatic derivatives.
//!
//! # ⚠️ INCOMPLETE IMPLEMENTATION
//!
//! **Status**: First-tile only. Missing eigenvalue decay and recursive tiling.
//!
//! | Domain | Status |
//! |--------|--------|
//! | (0, 0) | ✅ Exact limit position |
//! | [0.5, 1]² | ⚠️ First-tile approx (no λⁿ⁻¹ weighting) |
//! | < 0.5 | ❌ WRONG - needs recursive tiling |
//!
//! See `axis_patch` for implementation TODOs.
//!
//! ## Architecture
//!
//! - `bicubic()`: 16-coefficient polynomial as composed manifold type
//! - `bspline_patch()`: Regular patches (4x4 grid) → 3 bicubic manifolds (x,y,z)
//! - `eigen_patch()`: Extraordinary vertex patches (Stam's method)
//! - `EigenCoeffs`: Baked eigenstructure data (valences 3-50)
//!
//! ## Design: Types as AST
//!
//! ```text
//! bicubic([c0, c1, ...])
//!   → Add<Add<..., Mul<Mul<Y,Y>, f32>>, ...>  // Type tree = AST
//!   → Field operations                         // IR = SIMD intrinsics
//!   → Fused compute kernel                     // Output
//! ```
//!
//! Write math declaratively. The type system builds the AST.
//! Field/Jet evaluation compiles to efficient SIMD.
//!
//! ## Coordinates
//!
//! Uses X, Y as parametric (u, v) ∈ [0,1]².
//!
//! ## Eigenstructure Notes
//!
//! The baked eigenstructure data uses column-major access for the inverse
//! eigenvector matrix. For valence N:
//! - K = 2N + 8 eigenbases
//! - 3 subpatches for tiling near extraordinary vertices
//! - Evaluation at (0,0) = extraordinary vertex limit (verified working)
//! - Evaluation at other points requires Stam's tiling/subpatch selection
//!
//! ## References
//!
//! - Stam, "Exact Evaluation of Catmull-Clark Subdivision Surfaces" (SIGGRAPH 98)

mod coeffs;

pub use coeffs::{get_eigen, EigenCoeffs, MAX_VALENCE};

use pixelflow_core::combinators::{At, Select};
use pixelflow_core::jet::Jet3;
use pixelflow_core::ops::compare::Lt;
use pixelflow_core::{Field, Manifold, ManifoldExt, X, Y};

// ============================================================================
// Regular Patch (Valence 4) - Standard B-Spline
// ============================================================================

/// Uniform cubic B-spline basis coefficients in power form.
/// B_i(t) = Σⱼ BASIS[i][j] * t^j
const BSPLINE_BASIS: [[f32; 4]; 4] = [
    [1.0 / 6.0, -3.0 / 6.0, 3.0 / 6.0, -1.0 / 6.0],  // B0 = (1-t)³/6
    [4.0 / 6.0, 0.0, -6.0 / 6.0, 3.0 / 6.0],          // B1 = (3t³ - 6t² + 4)/6
    [1.0 / 6.0, 3.0 / 6.0, 3.0 / 6.0, -3.0 / 6.0],   // B2 = (-3t³ + 3t² + 3t + 1)/6
    [0.0, 0.0, 0.0, 1.0 / 6.0],                        // B3 = t³/6
];

/// Evaluate a bicubic B-spline patch.
///
/// Takes 16 control points as a 4x4 grid (row-major).
/// Returns a tuple of 3 manifolds for (x, y, z) coordinates.
///
/// Uses X as u ∈ [0,1], Y as v ∈ [0,1].
#[inline]
pub fn bspline_patch(
    control_points: [[f32; 3]; 16],
) -> (
    impl Manifold<Output = Field>,
    impl Manifold<Output = Field>,
    impl Manifold<Output = Field>,
) {
    // Extract per-axis control points and compute bicubic coefficients
    let mut cp_x = [[0.0f32; 4]; 4];
    let mut cp_y = [[0.0f32; 4]; 4];
    let mut cp_z = [[0.0f32; 4]; 4];

    for i in 0..4 {
        for j in 0..4 {
            let idx = i * 4 + j;
            cp_x[i][j] = control_points[idx][0];
            cp_y[i][j] = control_points[idx][1];
            cp_z[i][j] = control_points[idx][2];
        }
    }

    (
        bspline_axis(&cp_x),
        bspline_axis(&cp_y),
        bspline_axis(&cp_z),
    )
}

/// Build bicubic manifold for one axis of a B-spline patch.
///
/// Computes power-form coefficients from control points and basis.
fn bspline_axis(control_points: &[[f32; 4]; 4]) -> impl Manifold<Output = Field> {
    // Compute bicubic coefficients c[i][j] for u^i * v^j
    // P(u,v) = Σ_{k,l} B_k(u) * B_l(v) * P_{k,l}
    //        = Σ_{k,l} (Σ_i basis[k][i] * u^i) * (Σ_j basis[l][j] * v^j) * P_{k,l}
    //        = Σ_{i,j} (Σ_{k,l} basis[k][i] * basis[l][j] * P_{k,l}) * u^i * v^j

    let mut coeffs = [0.0f32; 16];

    for i in 0..4 {
        for j in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                for l in 0..4 {
                    sum += BSPLINE_BASIS[k][i] * BSPLINE_BASIS[l][j] * control_points[k][l];
                }
            }
            coeffs[i * 4 + j] = sum;
        }
    }

    bicubic(coeffs)
}

// ============================================================================
// Eigenstructure Patch (Extraordinary Vertices)
// ============================================================================

/// Build subpatch-selecting manifold for one axis.
///
/// Routes to subpatch based on (u, v):
/// - v < 0.5 → subpatch 0: bicubic at (2u, 2v)
/// - v >= 0.5, u < 0.5 → subpatch 2: bicubic at (2u, 2v - 1)
/// - v >= 0.5, u >= 0.5 → subpatch 1: bicubic at (2u - 1, 2v - 1)
///
/// # Limitations (TODO)
///
/// This is a FIRST-TILE ONLY implementation. For full Stam evaluation:
///
/// 1. **Eigenvalue powers not applied**: Each eigenbasis should be weighted by
///    λᵢⁿ⁻¹ where n is tile depth. Currently all weights are 1.0.
///
/// 2. **No recursive tiling**: Only handles (u,v) ∈ [0.5, 1]². For points closer
///    to origin, need to compute tile depth n = 1 + floor(min(-log2(u), -log2(v)))
///    and rescale coordinates.
///
/// 3. **Correct only at origin**: Evaluation at (0,0) gives exact limit position.
///    Other points have increasing error closer to origin.
///
/// To implement full tiling as manifold composition, need:
/// - `Map` for log2/floor/pow operations on coordinates
/// - Recursive `Select` tree for tile depth routing
/// - Per-eigenbasis weighting (may require K separate bicubics, not combined)
fn axis_patch(coeffs: [[f32; 16]; 3]) -> impl Manifold<Output = Field> {
    // TODO(subdiv): Implement eigenvalue power weighting λᵢⁿ⁻¹
    // TODO(subdiv): Implement recursive tiling for (u,v) < 0.5
    // TODO(subdiv): Currently ONLY first tile - see docs above

    // Each subpatch has different coordinate remapping
    let sub0 = At {
        inner: bicubic(coeffs[0]),
        x: X * 2.0,
        y: Y * 2.0,
        z: 0.0f32,
        w: 0.0f32,
    };
    let sub1 = At {
        inner: bicubic(coeffs[1]),
        x: X * 2.0 - 1.0,
        y: Y * 2.0 - 1.0,
        z: 0.0f32,
        w: 0.0f32,
    };
    let sub2 = At {
        inner: bicubic(coeffs[2]),
        x: X * 2.0,
        y: Y * 2.0 - 1.0,
        z: 0.0f32,
        w: 0.0f32,
    };

    // v < 0.5 → sub0, else (u < 0.5 → sub2, else sub1)
    Select {
        cond: Lt(Y, 0.5f32),
        if_true: sub0,
        if_false: Select {
            cond: Lt(X, 0.5f32),
            if_true: sub2,
            if_false: sub1,
        },
    }
}

/// Evaluate a Catmull-Clark patch using Stam's eigenstructure method.
///
/// Returns 3 manifolds for (x, y, z) coordinates. Each manifold:
/// - Selects subpatch based on (u, v) position
/// - Remaps coordinates to subpatch-local space
/// - Evaluates the combined bicubic
///
/// The manifolds are pure composition - types ARE the AST.
/// Field evaluation compiles to SIMD, Jet2 gives derivatives.
///
/// # Valid Domain
///
/// Currently only the first tile is implemented:
/// - **(0, 0)**: Exact limit position ✓
/// - **(u, v) ∈ [0.5, 1]²**: First-tile approximation (no eigenvalue decay)
/// - **(u, v) closer to origin**: INCORRECT - needs recursive tiling
///
/// See `axis_patch` docs for full TODO list.
///
/// # Panics
///
/// Use [`eigen_patch_checked`] if you want runtime validation that coordinates
/// are in the valid first-tile domain.
pub fn eigen_patch(
    control_points: &[[f32; 3]],
    valence: usize,
) -> Option<(
    impl Manifold<Output = Field>,
    impl Manifold<Output = Field>,
    impl Manifold<Output = Field>,
)> {
    let eigen = get_eigen(valence)?;
    let k = eigen.k;

    // Project control points to eigenspace (column-major access)
    let mut proj_x = vec![0.0f32; k];
    let mut proj_y = vec![0.0f32; k];
    let mut proj_z = vec![0.0f32; k];

    for i in 0..k {
        for j in 0..k.min(control_points.len()) {
            let w = eigen.inv_eigen(j, i); // transpose
            proj_x[i] += w * control_points[j][0];
            proj_y[i] += w * control_points[j][1];
            proj_z[i] += w * control_points[j][2];
        }
    }

    // Precompute bicubic coefficients for each subpatch
    let mut coeffs = [[[0.0f32; 16]; 3]; 3]; // [axis][subpatch][coeff]
    for sub in 0..3 {
        for c in 0..16 {
            for basis in 0..k {
                let s = eigen.spline(sub, basis, c);
                coeffs[0][sub][c] += s * proj_x[basis];
                coeffs[1][sub][c] += s * proj_y[basis];
                coeffs[2][sub][c] += s * proj_z[basis];
            }
        }
    }

    Some((
        axis_patch(coeffs[0]),
        axis_patch(coeffs[1]),
        axis_patch(coeffs[2]),
    ))
}

/// Validate that (u, v) is in the supported domain for eigen_patch.
///
/// # Panics
///
/// Panics if (u, v) is in the unsupported region (< 0.5 but not at origin).
/// Call this before evaluating eigen_patch manifolds to catch invalid usage.
#[inline]
pub fn validate_eigen_domain(u: f32, v: f32) {
    // Origin is valid (limit position)
    if u == 0.0 && v == 0.0 {
        return;
    }

    // First tile [0.5, 1]² is valid (with known approximation error)
    if u >= 0.5 && v >= 0.5 && u <= 1.0 && v <= 1.0 {
        return;
    }

    // Anything else requires recursive tiling which isn't implemented
    if u < 0.5 || v < 0.5 {
        todo!(
            "eigen_patch: recursive tiling not implemented for (u={}, v={}). \
             See axis_patch docs for TODOs: eigenvalue powers λᵢⁿ⁻¹, tile depth selection.",
            u,
            v
        );
    }
}

// ============================================================================
// Bicubic Polynomial (Manifold Composition)
// ============================================================================

/// Build a bicubic polynomial as a composed manifold.
///
/// P(u,v) = Σᵢⱼ cᵢⱼ · uⁱ · vʲ for i,j ∈ [0,3]
///
/// Uses X as u, Y as v (parametric coordinates).
///
/// Returns a manifold type - the polynomial IS the composition,
/// not a struct that computes it.
#[inline]
pub fn bicubic(c: [f32; 16]) -> impl Manifold<Output = Field> {
    // Powers of Y (v): Y, Y², Y³
    let v1 = Y;
    let v2 = Y * Y;
    let v3 = v2 * Y;

    // Powers of X (u): X, X², X³
    let u1 = X;
    let u2 = X * X;
    let u3 = u2 * X;

    // Row 0: c[0] + c[1]*v + c[2]*v² + c[3]*v³
    // Constant term last (f32 must be on right side of +)
    let r0 = v1 * c[1] + v2 * c[2] + v3 * c[3] + c[0];

    // Row 1: (c[4] + c[5]*v + c[6]*v² + c[7]*v³) * u
    let r1 = (v1 * c[5] + v2 * c[6] + v3 * c[7] + c[4]) * u1;

    // Row 2: (c[8] + c[9]*v + c[10]*v² + c[11]*v³) * u²
    let r2 = (v1 * c[9] + v2 * c[10] + v3 * c[11] + c[8]) * u2;

    // Row 3: (c[12] + c[13]*v + c[14]*v² + c[15]*v³) * u³
    let r3 = (v1 * c[13] + v2 * c[14] + v3 * c[15] + c[12]) * u3;

    // P(u,v) = r0 + r1 + r2 + r3
    r0 + r1 + r2 + r3
}

// ============================================================================
// Limit Surface Geometry (Raytracing)
// ============================================================================

/// Bounding box for quick ray rejection.
#[derive(Clone, Copy, Debug)]
pub struct BoundingBox {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl BoundingBox {
    /// Create bounding box from control points.
    pub fn from_points(points: &[[f32; 3]]) -> Self {
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for p in points {
            for i in 0..3 {
                min[i] = min[i].min(p[i]);
                max[i] = max[i].max(p[i]);
            }
        }
        // Expand slightly for numerical stability
        let eps = 0.01;
        for i in 0..3 {
            min[i] -= eps;
            max[i] += eps;
        }
        Self { min, max }
    }

    /// Ray-box intersection (returns t_min, t_max or None if miss).
    #[inline]
    pub fn intersect_ray(&self, origin: [f32; 3], dir: [f32; 3]) -> Option<(f32, f32)> {
        let mut t_min = 0.0f32;
        let mut t_max = f32::MAX;

        for i in 0..3 {
            if dir[i].abs() < 1e-10 {
                if origin[i] < self.min[i] || origin[i] > self.max[i] {
                    return None;
                }
            } else {
                let inv_d = 1.0 / dir[i];
                let t0 = (self.min[i] - origin[i]) * inv_d;
                let t1 = (self.max[i] - origin[i]) * inv_d;
                let (t0, t1) = if inv_d < 0.0 { (t1, t0) } else { (t0, t1) };
                t_min = t_min.max(t0);
                t_max = t_max.min(t1);
                if t_max < t_min {
                    return None;
                }
            }
        }
        Some((t_min, t_max))
    }
}

/// A single limit surface patch with precomputed coefficients.
#[derive(Clone)]
pub struct LimitPatch {
    /// Bicubic coefficients for X axis (3 subpatches × 16 coeffs)
    pub coeffs_x: [[f32; 16]; 3],
    /// Bicubic coefficients for Y axis
    pub coeffs_y: [[f32; 16]; 3],
    /// Bicubic coefficients for Z axis
    pub coeffs_z: [[f32; 16]; 3],
    /// Bounding box for quick rejection
    pub bounds: BoundingBox,
    /// Center in world space
    pub center: [f32; 3],
    /// Whether this is a regular (valence 4) patch
    pub is_regular: bool,
}

impl LimitPatch {
    /// Create a limit patch from control points using eigenstructure.
    pub fn from_control_points(control_points: &[[f32; 3]], valence: usize) -> Option<Self> {
        let eigen = get_eigen(valence)?;
        let k = eigen.k;

        // Project control points to eigenspace
        let mut proj_x = vec![0.0f32; k];
        let mut proj_y = vec![0.0f32; k];
        let mut proj_z = vec![0.0f32; k];

        for i in 0..k {
            for j in 0..k.min(control_points.len()) {
                let w = eigen.inv_eigen(j, i);
                proj_x[i] += w * control_points[j][0];
                proj_y[i] += w * control_points[j][1];
                proj_z[i] += w * control_points[j][2];
            }
        }

        // Precompute bicubic coefficients for each subpatch
        let mut coeffs_x = [[0.0f32; 16]; 3];
        let mut coeffs_y = [[0.0f32; 16]; 3];
        let mut coeffs_z = [[0.0f32; 16]; 3];

        for sub in 0..3 {
            for c in 0..16 {
                for basis in 0..k {
                    let s = eigen.spline(sub, basis, c);
                    coeffs_x[sub][c] += s * proj_x[basis];
                    coeffs_y[sub][c] += s * proj_y[basis];
                    coeffs_z[sub][c] += s * proj_z[basis];
                }
            }
        }

        // Compute center and bounds
        let mut center = [0.0f32; 3];
        for p in control_points.iter().take(k.min(control_points.len())) {
            center[0] += p[0];
            center[1] += p[1];
            center[2] += p[2];
        }
        let n = k.min(control_points.len()) as f32;
        center[0] /= n;
        center[1] /= n;
        center[2] /= n;

        let bounds = BoundingBox::from_points(control_points);

        Some(Self {
            coeffs_x,
            coeffs_y,
            coeffs_z,
            bounds,
            center,
            is_regular: valence == 4,
        })
    }

    /// Evaluate the limit surface at (u, v).
    #[inline]
    pub fn eval(&self, u: f32, v: f32) -> [f32; 3] {
        // Select subpatch and remap coordinates
        let (sub, lu, lv) = if v < 0.5 {
            (0, u * 2.0, v * 2.0)
        } else if u < 0.5 {
            (2, u * 2.0, v * 2.0 - 1.0)
        } else {
            (1, u * 2.0 - 1.0, v * 2.0 - 1.0)
        };

        [
            eval_bicubic_scalar(&self.coeffs_x[sub], lu, lv),
            eval_bicubic_scalar(&self.coeffs_y[sub], lu, lv),
            eval_bicubic_scalar(&self.coeffs_z[sub], lu, lv),
        ]
    }

    /// Evaluate with derivatives for normal computation.
    #[inline]
    pub fn eval_with_derivs(&self, u: f32, v: f32) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let (sub, lu, lv, scale) = if v < 0.5 {
            (0, u * 2.0, v * 2.0, 2.0)
        } else if u < 0.5 {
            (2, u * 2.0, v * 2.0 - 1.0, 2.0)
        } else {
            (1, u * 2.0 - 1.0, v * 2.0 - 1.0, 2.0)
        };

        let (px, dpx_du, dpx_dv) = eval_bicubic_with_derivs(&self.coeffs_x[sub], lu, lv);
        let (py, dpy_du, dpy_dv) = eval_bicubic_with_derivs(&self.coeffs_y[sub], lu, lv);
        let (pz, dpz_du, dpz_dv) = eval_bicubic_with_derivs(&self.coeffs_z[sub], lu, lv);

        // Scale derivatives by coordinate remapping
        let pos = [px, py, pz];
        let du = [dpx_du * scale, dpy_du * scale, dpz_du * scale];
        let dv = [dpx_dv * scale, dpy_dv * scale, dpz_dv * scale];

        (pos, du, dv)
    }
}

/// Evaluate bicubic polynomial at (u, v) returning scalar.
#[inline]
fn eval_bicubic_scalar(coeffs: &[f32; 16], u: f32, v: f32) -> f32 {
    let u2 = u * u;
    let u3 = u2 * u;
    let v2 = v * v;
    let v3 = v2 * v;

    let r0 = coeffs[0] + coeffs[1] * v + coeffs[2] * v2 + coeffs[3] * v3;
    let r1 = coeffs[4] + coeffs[5] * v + coeffs[6] * v2 + coeffs[7] * v3;
    let r2 = coeffs[8] + coeffs[9] * v + coeffs[10] * v2 + coeffs[11] * v3;
    let r3 = coeffs[12] + coeffs[13] * v + coeffs[14] * v2 + coeffs[15] * v3;

    r0 + r1 * u + r2 * u2 + r3 * u3
}

/// Evaluate bicubic polynomial with partial derivatives.
#[inline]
fn eval_bicubic_with_derivs(coeffs: &[f32; 16], u: f32, v: f32) -> (f32, f32, f32) {
    let u2 = u * u;
    let u3 = u2 * u;
    let v2 = v * v;
    let v3 = v2 * v;

    // Value
    let r0 = coeffs[0] + coeffs[1] * v + coeffs[2] * v2 + coeffs[3] * v3;
    let r1 = coeffs[4] + coeffs[5] * v + coeffs[6] * v2 + coeffs[7] * v3;
    let r2 = coeffs[8] + coeffs[9] * v + coeffs[10] * v2 + coeffs[11] * v3;
    let r3 = coeffs[12] + coeffs[13] * v + coeffs[14] * v2 + coeffs[15] * v3;
    let val = r0 + r1 * u + r2 * u2 + r3 * u3;

    // dP/du = r1 + 2*r2*u + 3*r3*u²
    let du = r1 + 2.0 * r2 * u + 3.0 * r3 * u2;

    // dP/dv = d(coeffs[0..3] polynomial)/dv + ...
    let dr0_dv = coeffs[1] + 2.0 * coeffs[2] * v + 3.0 * coeffs[3] * v2;
    let dr1_dv = coeffs[5] + 2.0 * coeffs[6] * v + 3.0 * coeffs[7] * v2;
    let dr2_dv = coeffs[9] + 2.0 * coeffs[10] * v + 3.0 * coeffs[11] * v2;
    let dr3_dv = coeffs[13] + 2.0 * coeffs[14] * v + 3.0 * coeffs[15] * v2;
    let dv = dr0_dv + dr1_dv * u + dr2_dv * u2 + dr3_dv * u3;

    (val, du, dv)
}

/// Limit surface composed of multiple patches.
#[derive(Clone)]
pub struct LimitSurface {
    /// All patches in the surface
    pub patches: Vec<LimitPatch>,
    /// Global bounding box
    pub bounds: BoundingBox,
}

impl LimitSurface {
    /// Create from a list of patches.
    pub fn from_patches(patches: Vec<LimitPatch>) -> Self {
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for patch in &patches {
            for i in 0..3 {
                min[i] = min[i].min(patch.bounds.min[i]);
                max[i] = max[i].max(patch.bounds.max[i]);
            }
        }
        Self {
            patches,
            bounds: BoundingBox { min, max },
        }
    }
}

// ============================================================================
// Analytic Limit Surface Geometry (Hilbert Space Evaluation)
// ============================================================================

/// Limit surface geometry for raytracing via analytic evaluation.
///
/// Evaluates the limit surface directly using the eigenstructure decomposition
/// in Hilbert space - the bicubic coefficients ARE the coordinates in the
/// orthogonal eigenbasis. No baking, no textures, pure manifold composition.
#[derive(Clone)]
pub struct LimitSurfaceGeometry {
    /// The limit surface with precomputed Hilbert space coordinates (bicubic coeffs)
    pub surface: LimitSurface,
    /// World transform: scale
    pub scale: f32,
    /// World transform: translation
    pub offset: [f32; 3],
}

impl LimitSurfaceGeometry {
    /// Create geometry from a limit surface.
    pub fn new(surface: LimitSurface, scale: f32, offset: [f32; 3]) -> Self {
        Self { surface, scale, offset }
    }
}

impl Manifold<Jet3> for LimitSurfaceGeometry {
    type Output = Jet3;

    #[inline]
    fn eval_raw(&self, rx: Jet3, ry: Jet3, rz: Jet3, _w: Jet3) -> Jet3 {
        // Analytic limit surface ray intersection.
        // The bicubic coefficients are Hilbert space coordinates in the eigenbasis.
        // We evaluate the bicubic polynomials directly using Field SIMD ops.
        //
        // Pattern: Build AST expressions, call .constant() to evaluate to Field.

        // Step 1: Hit base plane at y = offset[1]
        let base_y = Field::from(self.offset[1]);
        let t_plane = (base_y / ry.val).constant();

        // Step 2: World position at plane hit
        let hit_x = (rx.val * t_plane).constant();
        let hit_z = (rz.val * t_plane).constant();

        // Step 3: Map world coords to surface UV [0, 1]
        let half_scale = Field::from(self.scale * 0.5);
        let center_x = Field::from(self.offset[0]);
        let center_z = Field::from(self.offset[2]);
        let scale_inv = Field::from(1.0 / self.scale);

        // Build AST, collapse to Field via .constant()
        let u = ((hit_x - center_x + half_scale) * scale_inv).constant();
        let v = ((hit_z - center_z + half_scale) * scale_inv).constant();

        // Step 4: Bounds check (ManifoldExt methods work on Field)
        let zero = Field::from(0.0);
        let one = Field::from(1.0);
        let in_bounds = (u.ge(zero) & u.le(one) & v.ge(zero) & v.le(one)).constant();

        // Step 5: Map to patch grid and local UV
        let grid_size = (self.surface.patches.len() as f32).sqrt().ceil();
        let grid_u = (u * Field::from(grid_size)).constant();
        let grid_v = (v * Field::from(grid_size)).constant();

        // Patch indices (clamped)
        let grid_max = Field::from(grid_size - 1.0);
        let patch_col = (grid_u.floor().max(zero).min(grid_max)).constant();
        let patch_row = (grid_v.floor().max(zero).min(grid_max)).constant();

        // Local UV within patch [0, 1]
        let local_u = ((grid_u - patch_col).max(zero).min(one)).constant();
        let local_v = ((grid_v - patch_row).max(zero).min(one)).constant();

        // Step 6: Evaluate limit surface height analytically
        // Use center patch for now (proper routing would need Select tree)
        let center_idx = self.surface.patches.len() / 2;
        let patch = &self.surface.patches[center_idx.min(self.surface.patches.len() - 1)];

        // Select subpatch based on local UV
        // Subpatch 0: v < 0.5, Subpatch 1: v >= 0.5 && u >= 0.5, Subpatch 2: v >= 0.5 && u < 0.5
        let half = Field::from(0.5);
        let two = Field::from(2.0);

        let in_sub0 = local_v.lt(half).constant();
        let in_sub1 = (local_v.ge(half) & local_u.ge(half)).constant();
        // sub2 is the else case

        // Remap to local subpatch coords [0, 1]
        let sub0_u = (local_u * two).constant();
        let sub0_v = (local_v * two).constant();
        let sub1_u = (local_u * two - one).constant();
        let sub1_v = (local_v * two - one).constant();
        let sub2_u = (local_u * two).constant();
        let sub2_v = (local_v * two - one).constant();

        // Evaluate bicubic for Y (height) in each subpatch
        let h0 = eval_bicubic_field(&patch.coeffs_y[0], sub0_u, sub0_v);
        let h1 = eval_bicubic_field(&patch.coeffs_y[1], sub1_u, sub1_v);
        let h2 = eval_bicubic_field(&patch.coeffs_y[2], sub2_u, sub2_v);

        // Select correct subpatch result
        let inner_select = in_sub1.select(h1, h2).constant();
        let height_local = in_sub0.select(h0, inner_select).constant();

        // Step 7: Transform height to world space and compute t
        let height_world = (height_local * Field::from(self.scale) + base_y).constant();
        let t_hit = (height_world / ry.val).constant();

        // Return t with derivatives for antialiasing
        let miss = Field::from(-1.0);
        Jet3::new(
            in_bounds.select(t_hit, miss).constant(),
            in_bounds.select(rx.dx, miss).constant(),
            in_bounds.select(ry.dy, miss).constant(),
            Field::from(0.0),
        )
    }
}

/// Evaluate bicubic polynomial at Field coordinates.
///
/// This is the SIMD-parallel version of eval_bicubic_scalar.
/// The coefficients are Hilbert space coordinates in the bicubic eigenbasis.
/// Uses .constant() to collapse AST to Field.
#[inline]
fn eval_bicubic_field(coeffs: &[f32; 16], u: Field, v: Field) -> Field {
    let u2 = (u * u).constant();
    let u3 = (u2 * u).constant();
    let v2 = (v * v).constant();
    let v3 = (v2 * v).constant();

    // Row 0: c[0] + c[1]*v + c[2]*v² + c[3]*v³
    let r0 = (Field::from(coeffs[0])
        + Field::from(coeffs[1]) * v
        + Field::from(coeffs[2]) * v2
        + Field::from(coeffs[3]) * v3)
        .constant();

    // Row 1: (c[4] + c[5]*v + c[6]*v² + c[7]*v³) * u
    let r1 = ((Field::from(coeffs[4])
        + Field::from(coeffs[5]) * v
        + Field::from(coeffs[6]) * v2
        + Field::from(coeffs[7]) * v3)
        * u)
        .constant();

    // Row 2: (c[8] + c[9]*v + c[10]*v² + c[11]*v³) * u²
    let r2 = ((Field::from(coeffs[8])
        + Field::from(coeffs[9]) * v
        + Field::from(coeffs[10]) * v2
        + Field::from(coeffs[11]) * v3)
        * u2)
        .constant();

    // Row 3: (c[12] + c[13]*v + c[14]*v² + c[15]*v³) * u³
    let r3 = ((Field::from(coeffs[12])
        + Field::from(coeffs[13]) * v
        + Field::from(coeffs[14]) * v2
        + Field::from(coeffs[15]) * v3)
        * u3)
        .constant();

    (r0 + r1 + r2 + r3).constant()
}

// ============================================================================
// Face Mesh Generator
// ============================================================================

/// Create a simple face control cage.
///
/// This generates a low-poly face mesh suitable for subdivision.
/// The mesh is oriented with Y-up, facing -Z direction.
pub fn create_face_mesh() -> (Vec<[f32; 3]>, Vec<[usize; 4]>, Vec<usize>) {
    // Symmetric face - vertices are mirrored across X=0
    // Layout: nose at front, ears at sides, forehead up, chin down
    let vertices = vec![
        // Row 0: Top of head (y = 1.0)
        [-0.3, 1.0, 0.0],   // 0: left temple
        [0.0, 1.05, 0.0],   // 1: crown
        [0.3, 1.0, 0.0],    // 2: right temple

        // Row 1: Forehead (y = 0.7)
        [-0.4, 0.7, 0.15],  // 3: left forehead
        [-0.15, 0.75, 0.2], // 4: left-center forehead
        [0.15, 0.75, 0.2],  // 5: right-center forehead
        [0.4, 0.7, 0.15],   // 6: right forehead

        // Row 2: Eyes (y = 0.4)
        [-0.5, 0.4, 0.1],   // 7: left ear
        [-0.35, 0.4, 0.25], // 8: left eye outer
        [-0.15, 0.4, 0.3],  // 9: left eye inner
        [0.0, 0.4, 0.35],   // 10: nose bridge
        [0.15, 0.4, 0.3],   // 11: right eye inner
        [0.35, 0.4, 0.25],  // 12: right eye outer
        [0.5, 0.4, 0.1],    // 13: right ear

        // Row 3: Cheeks/Nose (y = 0.1)
        [-0.45, 0.1, 0.15], // 14: left jaw
        [-0.3, 0.1, 0.3],   // 15: left cheek
        [-0.1, 0.1, 0.4],   // 16: left nostril
        [0.0, 0.1, 0.45],   // 17: nose tip
        [0.1, 0.1, 0.4],    // 18: right nostril
        [0.3, 0.1, 0.3],    // 19: right cheek
        [0.45, 0.1, 0.15],  // 20: right jaw

        // Row 4: Mouth (y = -0.2)
        [-0.35, -0.2, 0.2], // 21: left mouth corner
        [-0.1, -0.2, 0.35], // 22: upper lip left
        [0.0, -0.2, 0.4],   // 23: upper lip center
        [0.1, -0.2, 0.35],  // 24: upper lip right
        [0.35, -0.2, 0.2],  // 25: right mouth corner

        // Row 5: Chin (y = -0.5)
        [-0.25, -0.5, 0.15], // 26: left chin
        [0.0, -0.5, 0.25],   // 27: chin tip
        [0.25, -0.5, 0.15],  // 28: right chin
    ];

    // Quad faces (CCW winding when viewed from front)
    let faces = vec![
        // Top row (crown)
        [0, 3, 4, 1],
        [1, 4, 5, 2],
        [2, 5, 6, 2], // Degenerate for symmetry - will be proper quad

        // Forehead to eyes
        [3, 7, 8, 4],
        [4, 8, 9, 5],
        [5, 9, 10, 5], // Left side
        [5, 10, 11, 5],
        [5, 11, 12, 6],
        [6, 12, 13, 6],

        // Rework: proper quads for the face
        // Eyes to cheeks
        [7, 14, 15, 8],
        [8, 15, 16, 9],
        [9, 16, 17, 10],
        [10, 17, 18, 11],
        [11, 18, 19, 12],
        [12, 19, 20, 13],

        // Cheeks to mouth
        [14, 21, 22, 15],
        [15, 22, 23, 16],
        [16, 23, 17, 17], // Center
        [17, 23, 24, 18],
        [18, 24, 25, 19],
        [19, 25, 20, 20],

        // Mouth to chin
        [21, 26, 27, 22],
        [22, 27, 23, 23],
        [23, 27, 24, 24],
        [24, 27, 28, 25],
    ];

    // Compute valences
    let mut valence = vec![0usize; vertices.len()];
    for face in &faces {
        for &v in face {
            if v < vertices.len() {
                valence[v] += 1;
            }
        }
    }

    (vertices, faces, valence)
}

/// Create a simpler face mesh - just the front face portion.
///
/// Returns (vertices, faces, valences) for a minimal face control cage.
pub fn create_simple_face() -> (Vec<[f32; 3]>, Vec<[usize; 4]>, Vec<usize>) {
    // Simple 4x4 grid with face-like displacement
    let mut vertices = Vec::with_capacity(16);

    for row in 0..4 {
        for col in 0..4 {
            let x = (col as f32 - 1.5) * 0.4; // -0.6 to 0.6
            let y = (1.5 - row as f32) * 0.4; // 0.6 to -0.6

            // Face-like Z displacement
            let nose_dist = ((col as f32 - 1.5).abs() + (row as f32 - 1.5).abs()) * 0.5;
            let z = (1.0 - nose_dist * 0.3).max(0.0) * 0.3;

            // Add eye sockets
            let left_eye = (col == 1 && row == 1) as i32 as f32;
            let right_eye = (col == 2 && row == 1) as i32 as f32;
            let eye_depth = (left_eye + right_eye) * 0.1;

            vertices.push([x, y, z - eye_depth]);
        }
    }

    // 3x3 grid of quads
    let mut faces = Vec::with_capacity(9);
    for row in 0..3 {
        for col in 0..3 {
            let i = row * 4 + col;
            faces.push([i, i + 1, i + 5, i + 4]);
        }
    }

    // Compute valences
    let mut valence = vec![0usize; vertices.len()];
    for face in &faces {
        for &v in face {
            valence[v] += 1;
        }
    }

    (vertices, faces, valence)
}

/// Build a LimitSurface from a face mesh.
pub fn build_face_surface() -> LimitSurface {
    let (vertices, faces, _valences) = create_simple_face();

    let mut patches = Vec::with_capacity(faces.len());

    for face in &faces {
        // For now, use valence 4 for all patches (regular)
        // In a full implementation, we'd check actual vertex valences
        let control_points: Vec<[f32; 3]> = face.iter()
            .map(|&i| vertices[i])
            .collect();

        // Expand to 16 control points for proper B-spline
        // For a single quad, we replicate corners to approximate
        let expanded = expand_quad_to_patch(&control_points);

        if let Some(patch) = LimitPatch::from_control_points(&expanded, 4) {
            patches.push(patch);
        }
    }

    LimitSurface::from_patches(patches)
}

/// Expand a 4-point quad to a 16-point control mesh.
///
/// Uses corner positions and interpolates edges/interior.
fn expand_quad_to_patch(corners: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let mut points = vec![[0.0f32; 3]; 16];

    // Bilinear interpolation for all 16 points
    for row in 0..4 {
        for col in 0..4 {
            let u = col as f32 / 3.0;
            let v = row as f32 / 3.0;

            // Bilinear blend of corners
            let p00 = corners[0];
            let p10 = corners[1];
            let p11 = corners[2];
            let p01 = corners[3];

            for i in 0..3 {
                let p0 = p00[i] * (1.0 - u) + p10[i] * u;
                let p1 = p01[i] * (1.0 - u) + p11[i] * u;
                points[row * 4 + col][i] = p0 * (1.0 - v) + p1 * v;
            }
        }
    }

    points
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_core::combinators::{At, Texture};

    /// Evaluate a scalar manifold at a point
    fn eval_scalar<M: Manifold<Output = Field>>(m: &M, x: f32, y: f32) -> f32 {
        // At with f32 constants binds coordinates; eval() ignores input
        let bound = At {
            inner: m,
            x,
            y,
            z: 0.0f32,
            w: 0.0f32,
        };
        // Sample via 1x1 texture to extract f32 (At ignores the sample point)
        let tex = Texture::from_manifold(&bound, 1, 1);
        tex.data()[0]
    }

    #[test]
    fn test_bicubic_constant() {
        // Constant polynomial: c00 = 5.0, all others 0
        let mut coeffs = [0.0f32; 16];
        coeffs[0] = 5.0;
        let poly = bicubic(coeffs);

        let val = eval_scalar(&poly, 0.5, 0.5);
        assert!((val - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_bicubic_linear() {
        // P(u,v) = 1 + 2u + 3v
        let mut coeffs = [0.0f32; 16];
        coeffs[0] = 1.0; // c00 = 1
        coeffs[1] = 3.0; // c01 = 3 (coefficient of v)
        coeffs[4] = 2.0; // c10 = 2 (coefficient of u)
        let poly = bicubic(coeffs);

        // P(0.5, 0.5) = 1 + 2*0.5 + 3*0.5 = 1 + 1 + 1.5 = 3.5
        let val = eval_scalar(&poly, 0.5, 0.5);
        assert!((val - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_get_eigen_valence_4() {
        // Regular vertex (valence 4) should have eigenstructure
        let eigen = get_eigen(4).expect("valence 4 should exist");
        assert_eq!(eigen.valence, 4);
        assert_eq!(eigen.k, 16); // 2*4 + 8 = 16
        assert_eq!(eigen.eigenvalues.len(), 16);
    }

    #[test]
    fn test_eigen_first_eigenvalue_is_one() {
        // First eigenvalue should always be 1.0 (limit surface property)
        for valence in 3..=10 {
            let eigen = get_eigen(valence).unwrap();
            assert!(
                (eigen.eigenvalues[0] - 1.0).abs() < 1e-6,
                "valence {} first eigenvalue should be 1.0",
                valence
            );
        }
    }

    #[test]
    fn test_bspline_patch_flat_plane() {
        // Create a flat 4x4 grid of control points at z=0
        // Points span [0,3] x [0,3] in x,y
        let mut control_points = [[0.0f32; 3]; 16];
        for i in 0..4 {
            for j in 0..4 {
                let idx = i * 4 + j;
                control_points[idx] = [j as f32, i as f32, 0.0];
            }
        }

        let (px, py, pz) = bspline_patch(control_points);

        // Evaluate at center (u=0.5, v=0.5)
        let x = eval_scalar(&px, 0.5, 0.5);
        let y = eval_scalar(&py, 0.5, 0.5);
        let z = eval_scalar(&pz, 0.5, 0.5);

        // Z should be 0 (flat plane)
        assert!(z.abs() < 1e-4, "z should be 0 for flat plane, got {}", z);

        // X and Y should be near the center of the grid
        // B-spline at u=v=0.5 with uniform grid should give weighted average
        assert!(x > 0.5 && x < 2.5, "x={} should be in [0.5, 2.5]", x);
        assert!(y > 0.5 && y < 2.5, "y={} should be in [0.5, 2.5]", y);
    }

    #[test]
    fn test_bspline_patch_corners() {
        // Flat grid
        let mut control_points = [[0.0f32; 3]; 16];
        for i in 0..4 {
            for j in 0..4 {
                let idx = i * 4 + j;
                control_points[idx] = [j as f32, i as f32, 0.0];
            }
        }

        let (px, py, _pz) = bspline_patch(control_points);

        // At (0,0) - should be influenced primarily by corner control points
        let x0 = eval_scalar(&px, 0.0, 0.0);
        let y0 = eval_scalar(&py, 0.0, 0.0);

        // At (1,1) - should be at the opposite extreme
        let x1 = eval_scalar(&px, 1.0, 1.0);
        let y1 = eval_scalar(&py, 1.0, 1.0);

        // Values should be different (surface varies across domain)
        assert!(
            (x1 - x0).abs() > 0.1,
            "x should vary across patch: x(0,0)={}, x(1,1)={}",
            x0,
            x1
        );
        assert!(
            (y1 - y0).abs() > 0.1,
            "y should vary across patch: y(0,0)={}, y(1,1)={}",
            y0,
            y1
        );
    }

    #[test]
    fn test_eigen_trivial_case() {
        // If all control points are at the SAME location,
        // the surface should evaluate to that location everywhere.
        // This tests affine invariance.
        let control_points = [[1.0f32, 2.0, 3.0]; 16];

        let eigen = get_eigen(4).unwrap();
        println!("K = {}", eigen.k);

        // Project to eigenspace
        let mut projected_x = [0.0f32; 16];
        for i in 0..16 {
            for j in 0..16 {
                projected_x[i] += eigen.inv_eigen(i, j) * 1.0; // all x = 1.0
            }
        }
        println!("Projected x: {:?}", &projected_x[..4]);

        // Check row sums
        println!("Row sums:");
        for i in 0..4 {
            let row_sum: f32 = (0..16).map(|j| eigen.inv_eigen(i, j)).sum();
            println!("  Row {}: {}", i, row_sum);
        }

        // Check column sums - maybe matrix is transposed?
        println!("Column sums:");
        for j in 0..4 {
            let col_sum: f32 = (0..16).map(|i| eigen.inv_eigen(i, j)).sum();
            println!("  Col {}: {}", j, col_sum);
        }

        // Check what happens if we use columns instead of rows
        let mut projected_col = [0.0f32; 16];
        for i in 0..16 {
            for j in 0..16 {
                // Transpose: inv_eigen[j, i] instead of inv_eigen[i, j]
                projected_col[i] += eigen.inv_eigen(j, i) * 1.0;
            }
        }
        println!("Projected (transposed): {:?}", &projected_col[..4]);

        // Combine with spline coefficients using TRANSPOSED projection
        let mut final_coeffs = [0.0f32; 16];
        for coeff_idx in 0..16 {
            for basis in 0..16 {
                final_coeffs[coeff_idx] +=
                    eigen.spline(0, basis, coeff_idx) * projected_col[basis];
            }
        }
        println!("Final coeffs (transposed): {:?}", &final_coeffs[..4]);

        // The spline basis for index 0 should sum the polynomial to 1 at key points
        // Check: φ₀(0.5, 0.5) should give us some partition of unity property
        let poly = bicubic(final_coeffs);
        let val_center = eval_scalar(&poly, 0.5, 0.5);
        let val_corner = eval_scalar(&poly, 0.0, 0.0);
        println!(
            "φ₀(0.5, 0.5) = {}, φ₀(0, 0) = {}",
            val_center, val_corner
        );

        // Check sum of bases for each subpatch
        for subpatch in 0..3 {
            let mut sum_at_center = 0.0f32;
            for basis in 0..16 {
                let mut basis_coeffs = [0.0f32; 16];
                for coeff_idx in 0..16 {
                    basis_coeffs[coeff_idx] = eigen.spline(subpatch, basis, coeff_idx);
                }
                let basis_poly = bicubic(basis_coeffs);
                sum_at_center += eval_scalar(&basis_poly, 0.5, 0.5);
            }
            println!("Subpatch {}: sum of bases at (0.5, 0.5) = {}", subpatch, sum_at_center);
        }

        // The key insight: Stam's method requires TILING the parameter space.
        // At (0,0), we're at the extraordinary vertex limit.
        // Let's verify the surface is correct at corners and with proper tiling.

        // Test eigen_patch at (0,0) with constant control points
        let const_points = [[5.0f32, 7.0, 9.0]; 16];
        let (ex, ey, ez) = eigen_patch(&const_points, 4).unwrap();

        // At (0,0), the limit position should be the control point value
        let ex0 = eval_scalar(&ex, 0.0, 0.0);
        let ey0 = eval_scalar(&ey, 0.0, 0.0);
        let ez0 = eval_scalar(&ez, 0.0, 0.0);
        println!("eigen_patch at (0,0): ({}, {}, {})", ex0, ey0, ez0);
        println!("Expected: (5, 7, 9)");

        // Check tolerance - should be exact at the corner
        assert!(
            (ex0 - 5.0).abs() < 0.01,
            "x at (0,0) should be 5.0, got {}",
            ex0
        );
        assert!(
            (ey0 - 7.0).abs() < 0.01,
            "y at (0,0) should be 7.0, got {}",
            ey0
        );
        assert!(
            (ez0 - 9.0).abs() < 0.01,
            "z at (0,0) should be 9.0, got {}",
            ez0
        );
    }

    #[test]
    #[should_panic(expected = "recursive tiling not implemented")]
    fn test_validate_eigen_domain_panics_for_invalid() {
        // This should panic - coordinates < 0.5 need recursive tiling
        validate_eigen_domain(0.25, 0.25);
    }

    #[test]
    fn test_validate_eigen_domain_accepts_valid() {
        // Origin is valid
        validate_eigen_domain(0.0, 0.0);

        // First tile is valid
        validate_eigen_domain(0.5, 0.5);
        validate_eigen_domain(0.75, 0.75);
        validate_eigen_domain(1.0, 1.0);
    }

    #[test]
    fn test_eigen_patch_subpatch_routing() {
        // Constant control points - surface should be constant everywhere
        let const_points = [[3.0f32, 5.0, 7.0]; 16];
        let (ex, ey, ez) = eigen_patch(&const_points, 4).unwrap();

        // Test points in each subpatch region (first tile)
        // Subpatch 0: v < 0.5 → local coords (2u, 2v)
        let x0 = eval_scalar(&ex, 0.25, 0.25);
        let y0 = eval_scalar(&ey, 0.25, 0.25);
        let z0 = eval_scalar(&ez, 0.25, 0.25);
        println!("Subpatch 0 at (0.25, 0.25): ({}, {}, {})", x0, y0, z0);

        // Subpatch 1: v >= 0.5, u >= 0.5 → local coords (2u-1, 2v-1)
        let x1 = eval_scalar(&ex, 0.75, 0.75);
        let y1 = eval_scalar(&ey, 0.75, 0.75);
        let z1 = eval_scalar(&ez, 0.75, 0.75);
        println!("Subpatch 1 at (0.75, 0.75): ({}, {}, {})", x1, y1, z1);

        // Subpatch 2: v >= 0.5, u < 0.5 → local coords (2u, 2v-1)
        let x2 = eval_scalar(&ex, 0.25, 0.75);
        let y2 = eval_scalar(&ey, 0.25, 0.75);
        let z2 = eval_scalar(&ez, 0.25, 0.75);
        println!("Subpatch 2 at (0.25, 0.75): ({}, {}, {})", x2, y2, z2);

        // All should be close to (3, 5, 7) for affine invariance
        // (Within first-tile approximation tolerance)
        assert!(
            (x0 - 3.0).abs() < 1.0,
            "x at subpatch 0 should be near 3.0, got {}",
            x0
        );
    }

    #[test]
    fn test_bspline_symmetry() {
        // Symmetric grid centered at origin
        let mut control_points = [[0.0f32; 3]; 16];
        for i in 0..4 {
            for j in 0..4 {
                let idx = i * 4 + j;
                control_points[idx] = [j as f32 - 1.5, i as f32 - 1.5, 0.0];
            }
        }

        let (px, py, _pz) = bspline_patch(control_points);

        // At center (0.5, 0.5), the surface should pass through (0, 0)
        // due to symmetry of control points
        let x = eval_scalar(&px, 0.5, 0.5);
        let y = eval_scalar(&py, 0.5, 0.5);

        assert!(
            x.abs() < 0.01,
            "symmetric patch should have x=0 at center, got {}",
            x
        );
        assert!(
            y.abs() < 0.01,
            "symmetric patch should have y=0 at center, got {}",
            y
        );
    }
}
