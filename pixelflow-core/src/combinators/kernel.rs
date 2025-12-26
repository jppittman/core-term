//! # Kernel Algebra
//!
//! Non-local operations (blur, shadows, AO, attention) share a common structure:
//!
//! ```text
//! Output(query) = ∫ K(query, source) · Value(source) d(source)
//! ```
//!
//! If the kernel K has **symmetry**, there exists a basis where K is sparse.
//! The kernel's eigenfunctions ARE the natural basis:
//!
//! | Symmetry | Basis |
//! |----------|-------|
//! | Rotation-invariant (3D) | Spherical Harmonics |
//! | Translation-invariant | Fourier |
//! | Scale-invariant | Wavelets |
//!
//! This module provides the algebraic infrastructure for working with
//! symmetry-compressed fields. Key types:
//!
//! - [`Symmetry`]: Type-level encoding of kernel symmetry
//! - [`Basis`]: Operations on a specific basis (projection, evaluation)
//! - [`Frame`]: Coordinate frame for gauge freedom
//! - [`Compressed`]: A field projected onto its natural basis
//!
//! ## Example: Spherical Harmonics for Irradiance
//!
//! ```ignore
//! // Environment light, baked once
//! let env = Compressed::<RotationInvariant3D>::bake(sky_manifold, Frame::identity());
//!
//! // At each shading point: multiply in basis space
//! let result = env.mul_coeffs(&visibility_sh);
//! let irradiance = result.sample(normal);
//! ```

use crate::manifold::Manifold;
use crate::numeric::Numeric;
use crate::Field;
use alloc::sync::Arc;
use core::marker::PhantomData;

// ============================================================================
// Clebsch-Gordan Coefficients for SH Multiplication
// ============================================================================

/// Clebsch-Gordan coefficient entry: (i, j, k, weight).
/// Represents: result[k] += coeffs_a[i] * coeffs_b[j] * weight
pub type CgEntry = (usize, usize, usize, f32);

/// Clebsch-Gordan coefficients for order 2 (9 coefficients) SH multiplication.
/// These are the "constants of the universe" for SO(3) representation theory.
///
/// Computed offline from triple integrals of SH basis functions:
/// C(l1,m1,l2,m2,l3,m3) = ∫ Y_l1^m1 · Y_l2^m2 · Y_l3^m3 dΩ
///
/// Only non-zero entries are stored.
pub static CG_ORDER_2: &[CgEntry] = &[
    // DC × DC → DC
    (0, 0, 0, 0.282_094_79),
    // DC × l=1 → l=1
    (0, 1, 1, 0.282_094_79),
    (0, 2, 2, 0.282_094_79),
    (0, 3, 3, 0.282_094_79),
    (1, 0, 1, 0.282_094_79),
    (2, 0, 2, 0.282_094_79),
    (3, 0, 3, 0.282_094_79),
    // DC × l=2 → l=2
    (0, 4, 4, 0.282_094_79),
    (0, 5, 5, 0.282_094_79),
    (0, 6, 6, 0.282_094_79),
    (0, 7, 7, 0.282_094_79),
    (0, 8, 8, 0.282_094_79),
    (4, 0, 4, 0.282_094_79),
    (5, 0, 5, 0.282_094_79),
    (6, 0, 6, 0.282_094_79),
    (7, 0, 7, 0.282_094_79),
    (8, 0, 8, 0.282_094_79),
    // l=1 × l=1 → DC (trace)
    (1, 1, 0, 0.282_094_79),
    (2, 2, 0, 0.282_094_79),
    (3, 3, 0, 0.282_094_79),
    // l=1 × l=1 → l=2
    (1, 2, 4, 0.218_509_68),
    (2, 1, 4, 0.218_509_68),
    (1, 3, 5, 0.218_509_68),
    (3, 1, 5, 0.218_509_68),
    (2, 3, 7, 0.218_509_68),
    (3, 2, 7, 0.218_509_68),
    (1, 1, 6, -0.126_156_63),
    (2, 2, 6, -0.126_156_63),
    (3, 3, 6, 0.252_313_26),
    (1, 1, 8, 0.218_509_68),
    (2, 2, 8, -0.218_509_68),
];

// ============================================================================
// Symmetry Trait
// ============================================================================

/// Type-level encoding of kernel symmetry.
///
/// Different symmetries have different natural bases:
/// - Rotation invariance → Spherical Harmonics
/// - Translation invariance → Fourier
/// - Scale invariance → Wavelets
///
/// The symmetry determines:
/// 1. Which basis functions to use
/// 2. How to multiply in basis space (product table)
pub trait Symmetry: Send + Sync {
    /// The basis type for this symmetry.
    type Basis: Basis;

    /// Product table for multiplying two fields in this basis.
    type ProductTable: AsRef<[CgEntry]> + Send + Sync;

    /// The precomputed product table (compile-time constant).
    const PRODUCT: Self::ProductTable;
}

// ============================================================================
// Basis Trait
// ============================================================================

/// Coefficient vector type that supports inner products.
pub trait Coefficients: Clone + Send + Sync + core::fmt::Debug {
    /// Number of coefficients.
    const NUM_COEFFS: usize;

    /// Zero coefficients.
    fn zero() -> Self;

    /// Inner product of two coefficient vectors.
    fn dot(&self, other: &Self) -> f32;

    /// Multiply two coefficient vectors using Clebsch-Gordan table.
    fn multiply(&self, other: &Self, table: &[CgEntry]) -> Self;

    /// Get mutable access to coefficient at index.
    fn get_mut(&mut self, idx: usize) -> &mut f32;

    /// Get coefficient at index.
    fn get(&self, idx: usize) -> f32;
}

/// Basis for a specific symmetry.
///
/// Defines how to:
/// 1. Project a manifold onto the basis (bake step)
/// 2. Evaluate basis functions at a direction
/// 3. Reconstruct a value from coefficients
pub trait Basis: Sized + Send + Sync {
    /// Coefficient storage type.
    type Coeffs: Coefficients;

    /// Evaluate all basis functions at a direction (x, y, z).
    /// Returns coefficients representing "point spread" in basis space.
    fn eval_at(x: Field, y: Field, z: Field) -> Self::Coeffs;

    /// Evaluate all basis functions at a direction (as f32 tuple).
    fn eval_at_scalar(x: f32, y: f32, z: f32) -> Self::Coeffs;
}

// ============================================================================
// Frame: Gauge Freedom
// ============================================================================

/// Coordinate frame for kernel evaluation.
///
/// The same kernel can be easier or harder depending on where you "stand":
/// - Light transport from a point: put frame at the light
/// - Irradiance at a surface: align frame with the normal
///
/// Frame is pure metadata—it doesn't change the underlying field,
/// just how we query it.
#[derive(Clone, Debug)]
pub struct Frame {
    /// Origin in world space.
    pub origin: (f32, f32, f32),
    /// Z-axis direction (for aligned frames).
    pub z_axis: (f32, f32, f32),
}

impl Default for Frame {
    fn default() -> Self {
        Self::identity()
    }
}

impl Frame {
    /// Identity frame at world origin with z-up.
    pub const fn identity() -> Self {
        Self {
            origin: (0.0, 0.0, 0.0),
            z_axis: (0.0, 0.0, 1.0),
        }
    }

    /// Frame centered at a point with z-up.
    pub const fn centered_at(origin: (f32, f32, f32)) -> Self {
        Self {
            origin,
            z_axis: (0.0, 0.0, 1.0),
        }
    }

    /// Frame aligned to a surface normal.
    ///
    /// The z-axis of the frame is aligned with the given normal.
    /// Useful for hemispherical integration (irradiance, AO).
    pub fn aligned_to(origin: (f32, f32, f32), normal: (f32, f32, f32)) -> Self {
        // Normalize the normal
        let len = libm::sqrtf(normal.0 * normal.0 + normal.1 * normal.1 + normal.2 * normal.2);
        let z_axis = if len > 1e-6 {
            (normal.0 / len, normal.1 / len, normal.2 / len)
        } else {
            (0.0, 0.0, 1.0)
        };

        Self { origin, z_axis }
    }

    /// Transform a world-space direction to frame-local space.
    #[inline(always)]
    pub fn to_local(&self, world: (f32, f32, f32)) -> (f32, f32, f32) {
        // Translate by origin
        let local = (
            world.0 - self.origin.0,
            world.1 - self.origin.1,
            world.2 - self.origin.2,
        );

        // If frame is aligned (z_axis != (0,0,1)), we'd rotate here.
        // For now, simplified: just translation.
        // Full implementation would build rotation matrix from z_axis.
        local
    }

    /// Transform a frame-local direction to world space.
    #[inline(always)]
    pub fn to_world(&self, local: (f32, f32, f32)) -> (f32, f32, f32) {
        (
            local.0 + self.origin.0,
            local.1 + self.origin.1,
            local.2 + self.origin.2,
        )
    }
}

// ============================================================================
// Rotation-Invariant Symmetry (Spherical Harmonics)
// ============================================================================

/// Rotation-invariant symmetry in 3D.
///
/// Natural basis: Spherical Harmonics (eigenfunctions of SO(3)).
/// Use for: irradiance, ambient occlusion, soft shadows, environment lighting.
#[derive(Clone, Copy, Debug)]
pub struct RotationInvariant3D<const ORDER: usize>;

/// Order 2 rotation-invariant symmetry (9 SH coefficients).
/// Good for: diffuse irradiance, soft shadows.
pub type RotInv2 = RotationInvariant3D<2>;

// ============================================================================
// Spherical Harmonics Basis Implementation
// ============================================================================

/// Spherical harmonics coefficients.
#[derive(Clone, Debug)]
pub struct ShBasis<const N: usize> {
    /// Coefficients indexed by flattened (l, m) index.
    pub coeffs: [f32; N],
}

impl<const N: usize> ShBasis<N> {
    /// Create zero coefficients.
    pub const fn zero() -> Self {
        Self { coeffs: [0.0; N] }
    }

    /// Create from array.
    pub const fn new(coeffs: [f32; N]) -> Self {
        Self { coeffs }
    }
}

impl<const N: usize> Coefficients for ShBasis<N> {
    const NUM_COEFFS: usize = N;

    fn zero() -> Self {
        Self::zero()
    }

    fn dot(&self, other: &Self) -> f32 {
        let mut sum = 0.0;
        for i in 0..N {
            sum += self.coeffs[i] * other.coeffs[i];
        }
        sum
    }

    fn multiply(&self, other: &Self, table: &[CgEntry]) -> Self {
        let mut result = Self::zero();
        for &(i, j, k, weight) in table {
            if i < N && j < N && k < N {
                result.coeffs[k] += self.coeffs[i] * other.coeffs[j] * weight;
            }
        }
        result
    }

    fn get_mut(&mut self, idx: usize) -> &mut f32 {
        &mut self.coeffs[idx]
    }

    fn get(&self, idx: usize) -> f32 {
        self.coeffs[idx]
    }
}

impl Basis for ShBasis<9> {
    type Coeffs = ShBasis<9>;

    #[inline(always)]
    fn eval_at(x: Field, y: Field, z: Field) -> Self::Coeffs {
        // Normalize direction
        let r = (x * x + y * y + z * z).sqrt();
        let inv_r = Field::from(1.0) / r;
        let nx = x * inv_r;
        let ny = y * inv_r;
        let nz = z * inv_r;

        // Extract first lane for scalar output
        let mut buf_x = [0.0f32; crate::PARALLELISM];
        let mut buf_y = [0.0f32; crate::PARALLELISM];
        let mut buf_z = [0.0f32; crate::PARALLELISM];
        nx.store(&mut buf_x);
        ny.store(&mut buf_y);
        nz.store(&mut buf_z);

        Self::eval_at_scalar(buf_x[0], buf_y[0], buf_z[0])
    }

    fn eval_at_scalar(x: f32, y: f32, z: f32) -> Self::Coeffs {
        use crate::SH_NORM;

        // Normalize direction
        let r = libm::sqrtf(x * x + y * y + z * z);
        let inv_r = if r > 1e-10 { 1.0 / r } else { 0.0 };
        let nx = x * inv_r;
        let ny = y * inv_r;
        let nz = z * inv_r;

        // Compute all 9 SH basis functions
        Self::new([
            // l=0
            SH_NORM[0][0],
            // l=1
            SH_NORM[1][1] * ny,
            SH_NORM[1][0] * nz,
            SH_NORM[1][1] * nx,
            // l=2
            SH_NORM[2][2] * nx * ny,
            SH_NORM[2][1] * ny * nz,
            SH_NORM[2][0] * (3.0 * nz * nz - 1.0),
            SH_NORM[2][1] * nx * nz,
            SH_NORM[2][2] * (nx * nx - ny * ny),
        ])
    }
}

impl Symmetry for RotationInvariant3D<2> {
    type Basis = ShBasis<9>;
    type ProductTable = &'static [CgEntry];
    const PRODUCT: Self::ProductTable = CG_ORDER_2;
}

// ============================================================================
// Compressed: Field in Basis Space
// ============================================================================

/// A field compressed into its symmetry-appropriate basis.
///
/// This is the key optimization: instead of evaluating the original
/// manifold (potentially expensive), we sample from precomputed coefficients.
///
/// Sampling is O(basis size), not O(field complexity).
///
/// ## Example
///
/// ```ignore
/// // Expensive: samples sky dome for every query
/// let expensive = sky_dome.eval_raw(x, y, z, w);
///
/// // Cheap: dot product with 9 SH coefficients
/// let compressed = Compressed::<RotInv2>::from_coeffs(sky_sh);
/// let cheap = compressed.sample(direction);
/// ```
#[derive(Clone, Debug)]
pub struct Compressed<S: Symmetry> {
    /// Coefficients in the natural basis.
    pub coeffs: <S::Basis as Basis>::Coeffs,
    /// Coordinate frame for queries.
    pub frame: Frame,
    /// Phantom data for symmetry type.
    _symmetry: PhantomData<S>,
}

impl<S: Symmetry> Compressed<S>
where
    S::Basis: Basis,
{
    /// Create compressed field from precomputed coefficients.
    pub fn from_coeffs(coeffs: <S::Basis as Basis>::Coeffs) -> Self {
        Self {
            coeffs,
            frame: Frame::identity(),
            _symmetry: PhantomData,
        }
    }

    /// Create compressed field with a specific frame.
    pub fn with_frame(coeffs: <S::Basis as Basis>::Coeffs, frame: Frame) -> Self {
        Self {
            coeffs,
            frame,
            _symmetry: PhantomData,
        }
    }

    /// Sample the compressed field at a direction.
    ///
    /// This is O(NUM_COEFFS) - a dot product.
    pub fn sample(&self, direction: (f32, f32, f32)) -> f32
    where
        <S::Basis as Basis>::Coeffs: Coefficients,
    {
        // Transform direction to local frame
        let local = self.frame.to_local(direction);

        // Evaluate basis at this direction
        let basis_at_dir = S::Basis::eval_at_scalar(local.0, local.1, local.2);

        // Inner product with coefficients
        self.coeffs.dot(&basis_at_dir)
    }

    /// Multiply two compressed fields coefficient-wise.
    ///
    /// Uses Clebsch-Gordan coefficients for SH multiplication.
    /// Result is also a compressed field in the same basis.
    pub fn mul_coeffs(&self, other: &Self) -> Self
    where
        <S::Basis as Basis>::Coeffs: Coefficients,
        S::ProductTable: AsRef<[CgEntry]>,
    {
        let result_coeffs = self.coeffs.multiply(&other.coeffs, S::PRODUCT.as_ref());
        Self {
            coeffs: result_coeffs,
            frame: self.frame.clone(),
            _symmetry: PhantomData,
        }
    }

    /// Get the DC (ambient) term.
    ///
    /// For SH, this is coefficient 0 - represents the average value.
    /// For visibility fields, this IS ambient occlusion.
    pub fn dc_term(&self) -> f32
    where
        <S::Basis as Basis>::Coeffs: Coefficients,
    {
        self.coeffs.get(0)
    }
}

// ============================================================================
// Compressed as Manifold
// ============================================================================

/// Wrapper to make Compressed usable as a Manifold.
///
/// Interprets (x, y, z) as a direction and samples the compressed field.
#[derive(Clone)]
pub struct CompressedManifold<S: Symmetry> {
    inner: Arc<Compressed<S>>,
}

impl<S: Symmetry> CompressedManifold<S>
where
    S::Basis: Basis,
    <S::Basis as Basis>::Coeffs: Coefficients,
{
    /// Create from compressed field.
    pub fn new(compressed: Compressed<S>) -> Self {
        Self {
            inner: Arc::new(compressed),
        }
    }
}

impl<I: Numeric> Manifold<I> for CompressedManifold<RotInv2> {
    type Output = I;

    #[inline(always)]
    fn eval_raw(&self, _x: I, _y: I, _z: I, _w: I) -> I {
        // For SIMD evaluation, we compute basis at each lane and dot product.
        // This is a simplified implementation - production would extract
        // per-lane values and compute SH at each direction.
        //
        // TODO: Full vectorized implementation would:
        // 1. Extract x,y,z to f32 arrays
        // 2. Compute SH basis at each direction
        // 3. Dot product with coefficients per lane

        // For now, sample at nominal +Z direction
        let val = self.inner.sample((0.0, 0.0, 1.0));
        I::from_f32(val)
    }
}

// ============================================================================
// Cosine Lobe (Analytic SH)
// ============================================================================

/// Analytic SH coefficients for a clamped cosine lobe.
///
/// The cosine lobe `max(0, n·ω)` has known SH projection:
/// - Only l=0, l=1 are non-zero for Lambertian BRDF
/// - Coefficients are: [π, 2π/3, 2π/3, 2π/3, 0, 0, 0, 0, 0]
///   when lobe is aligned with +z.
pub fn cosine_lobe_sh(normal: (f32, f32, f32)) -> Compressed<RotInv2> {
    // Normalize
    let len = libm::sqrtf(normal.0 * normal.0 + normal.1 * normal.1 + normal.2 * normal.2);
    let (nx, ny, nz) = if len > 1e-6 {
        (normal.0 / len, normal.1 / len, normal.2 / len)
    } else {
        (0.0, 0.0, 1.0)
    };

    // Analytic coefficients for cosine lobe
    // These are the ZH coefficients rotated to align with normal
    let a0 = core::f32::consts::PI; // l=0: π * Y_0^0
    let a1 = 2.0 * core::f32::consts::PI / 3.0; // l=1: 2π/3 * Y_1^m

    let coeffs = ShBasis::new([
        a0 * 0.282_094_79,                    // l=0
        a1 * 0.488_602_51 * ny,               // l=1, m=-1
        a1 * 0.488_602_51 * nz,               // l=1, m=0
        a1 * 0.488_602_51 * nx,               // l=1, m=1
        0.0, 0.0, 0.0, 0.0, 0.0,              // l=2 (zero for Lambertian)
    ]);

    Compressed::from_coeffs(coeffs)
}

// ============================================================================
// Irradiance Computation
// ============================================================================

/// Compute irradiance at a point given environment SH and normal.
///
/// This is the fundamental operation:
/// `E(n) = L_sh · T_sh(n)`
///
/// where L_sh is the environment lighting and T_sh is the
/// cosine-weighted transfer function.
pub fn irradiance(
    environment: &Compressed<RotInv2>,
    normal: (f32, f32, f32),
) -> f32 {
    let transfer = cosine_lobe_sh(normal);
    environment.mul_coeffs(&transfer).dc_term()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_identity() {
        let frame = Frame::identity();
        assert_eq!(frame.origin, (0.0, 0.0, 0.0));
        assert_eq!(frame.z_axis, (0.0, 0.0, 1.0));
    }

    #[test]
    fn test_frame_to_local() {
        let frame = Frame::centered_at((1.0, 2.0, 3.0));
        let local = frame.to_local((2.0, 4.0, 6.0));
        assert!((local.0 - 1.0).abs() < 1e-6);
        assert!((local.1 - 2.0).abs() < 1e-6);
        assert!((local.2 - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_sh_basis_dot() {
        let a = ShBasis::<9>::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b = ShBasis::<9>::new([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!((a.dot(&b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_sh_basis_eval() {
        // +Z direction should have strong l=0 and l=1,m=0 components
        let coeffs = ShBasis::<9>::eval_at_scalar(0.0, 0.0, 1.0);
        assert!(coeffs.coeffs[0] > 0.0); // DC term
        assert!(coeffs.coeffs[2] > 0.0); // Y_1^0 (z-aligned)
    }

    #[test]
    fn test_compressed_sample() {
        // Constant field (DC only)
        let coeffs = ShBasis::<9>::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let compressed = Compressed::<RotInv2>::from_coeffs(coeffs);

        // Should return approximately the DC term regardless of direction
        let val1 = compressed.sample((0.0, 0.0, 1.0));
        let val2 = compressed.sample((1.0, 0.0, 0.0));

        // DC contribution is constant * Y_0^0(dir) ≈ 0.282
        assert!(val1 > 0.0);
        assert!((val1 - val2).abs() < 0.1);
    }

    #[test]
    fn test_cg_multiplication() {
        // DC × DC → DC
        let a = ShBasis::<9>::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b = ShBasis::<9>::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let result = a.multiply(&b, CG_ORDER_2);

        // Should have non-zero DC term
        assert!(result.coeffs[0].abs() > 0.0);
    }

    #[test]
    fn test_cosine_lobe_sh() {
        let lobe = cosine_lobe_sh((0.0, 0.0, 1.0));

        // DC term should be non-zero (ambient)
        assert!(lobe.dc_term().abs() > 0.0);

        // Sample should be maximal in the lobe direction
        let up = lobe.sample((0.0, 0.0, 1.0));
        let side = lobe.sample((1.0, 0.0, 0.0));
        assert!(up > side);
    }

    #[test]
    fn test_irradiance() {
        // Uniform environment (DC only)
        let env = Compressed::<RotInv2>::from_coeffs(
            ShBasis::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        );

        // Irradiance at any normal should be roughly constant
        let e1 = irradiance(&env, (0.0, 0.0, 1.0));
        let e2 = irradiance(&env, (1.0, 0.0, 0.0));

        // Not exactly equal due to SH approximation, but close
        assert!((e1 - e2).abs() < 0.5);
    }
}
