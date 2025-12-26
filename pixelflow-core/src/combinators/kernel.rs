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
//! ## Example: Spherical Harmonics
//!
//! ```ignore
//! // Project a field onto SH basis
//! let compressed = project_to_sh(&manifold, 1000);
//!
//! // Multiply two SH fields (stays in basis)
//! let result = compressed.mul_coeffs(&other);
//!
//! // Sample at a direction
//! let value = result.sample((0.0, 0.0, 1.0));
//! ```

use crate::manifold::Manifold;
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
    /// Rotation matrix (row-major, 3x3) from world to local.
    /// Identity when z_axis = (0, 0, 1).
    rotation: [[f32; 3]; 3],
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
            rotation: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        }
    }

    /// Frame centered at a point with z-up.
    pub const fn centered_at(origin: (f32, f32, f32)) -> Self {
        Self {
            origin,
            rotation: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        }
    }

    /// Frame aligned to a surface normal.
    ///
    /// The z-axis of the frame is aligned with the given normal.
    /// Useful for hemispherical integration (irradiance, AO).
    ///
    /// Constructs an orthonormal basis where:
    /// - Z-axis = normalized normal
    /// - X and Y axes are perpendicular to Z and each other
    pub fn aligned_to(origin: (f32, f32, f32), normal: (f32, f32, f32)) -> Self {
        // Normalize the normal (this becomes local Z-axis)
        let len = libm::sqrtf(normal.0 * normal.0 + normal.1 * normal.1 + normal.2 * normal.2);
        let z = if len > 1e-6 {
            (normal.0 / len, normal.1 / len, normal.2 / len)
        } else {
            (0.0, 0.0, 1.0)
        };

        // Build orthonormal basis using Gram-Schmidt
        // Choose a vector not parallel to z
        let up = if libm::fabsf(z.1) < 0.999 {
            (0.0, 1.0, 0.0)
        } else {
            (1.0, 0.0, 0.0)
        };

        // x = normalize(up × z)
        let x_raw = (
            up.1 * z.2 - up.2 * z.1,
            up.2 * z.0 - up.0 * z.2,
            up.0 * z.1 - up.1 * z.0,
        );
        let x_len = libm::sqrtf(x_raw.0 * x_raw.0 + x_raw.1 * x_raw.1 + x_raw.2 * x_raw.2);
        let x = (x_raw.0 / x_len, x_raw.1 / x_len, x_raw.2 / x_len);

        // y = z × x (already normalized since z and x are orthonormal)
        let y = (
            z.1 * x.2 - z.2 * x.1,
            z.2 * x.0 - z.0 * x.2,
            z.0 * x.1 - z.1 * x.0,
        );

        // Rotation matrix: rows are the local axes expressed in world coords
        // To go from world to local, we use the transpose (since orthonormal)
        Self {
            origin,
            rotation: [
                [x.0, x.1, x.2],
                [y.0, y.1, y.2],
                [z.0, z.1, z.2],
            ],
        }
    }

    /// Get the z-axis (normal direction) of this frame.
    #[inline(always)]
    pub fn z_axis(&self) -> (f32, f32, f32) {
        (self.rotation[2][0], self.rotation[2][1], self.rotation[2][2])
    }

    /// Transform a world-space direction to frame-local space.
    #[inline(always)]
    pub fn to_local(&self, world: (f32, f32, f32)) -> (f32, f32, f32) {
        // Translate by origin
        let translated = (
            world.0 - self.origin.0,
            world.1 - self.origin.1,
            world.2 - self.origin.2,
        );

        // Apply rotation matrix (world to local)
        (
            self.rotation[0][0] * translated.0 + self.rotation[0][1] * translated.1 + self.rotation[0][2] * translated.2,
            self.rotation[1][0] * translated.0 + self.rotation[1][1] * translated.1 + self.rotation[1][2] * translated.2,
            self.rotation[2][0] * translated.0 + self.rotation[2][1] * translated.1 + self.rotation[2][2] * translated.2,
        )
    }

    /// Transform a frame-local direction to world space.
    #[inline(always)]
    pub fn to_world(&self, local: (f32, f32, f32)) -> (f32, f32, f32) {
        // Apply inverse rotation (transpose for orthonormal matrix)
        let rotated = (
            self.rotation[0][0] * local.0 + self.rotation[1][0] * local.1 + self.rotation[2][0] * local.2,
            self.rotation[0][1] * local.0 + self.rotation[1][1] * local.1 + self.rotation[2][1] * local.2,
            self.rotation[0][2] * local.0 + self.rotation[1][2] * local.1 + self.rotation[2][2] * local.2,
        );

        // Translate back
        (
            rotated.0 + self.origin.0,
            rotated.1 + self.origin.1,
            rotated.2 + self.origin.2,
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

    /// Scale all coefficients by a factor.
    pub fn scale(&self, factor: f32) -> Self
    where
        <S::Basis as Basis>::Coeffs: Coefficients,
    {
        let mut result = self.coeffs.clone();
        for i in 0..<<S::Basis as Basis>::Coeffs as Coefficients>::NUM_COEFFS {
            *result.get_mut(i) *= factor;
        }
        Self {
            coeffs: result,
            frame: self.frame.clone(),
            _symmetry: PhantomData,
        }
    }

    /// Add two compressed fields coefficient-wise.
    ///
    /// Both fields should be in the same frame for meaningful results.
    pub fn add(&self, other: &Self) -> Self
    where
        <S::Basis as Basis>::Coeffs: Coefficients,
    {
        let mut result = self.coeffs.clone();
        for i in 0..<<S::Basis as Basis>::Coeffs as Coefficients>::NUM_COEFFS {
            *result.get_mut(i) += other.coeffs.get(i);
        }
        Self {
            coeffs: result,
            frame: self.frame.clone(),
            _symmetry: PhantomData,
        }
    }

    /// Linear interpolation between two compressed fields.
    pub fn lerp(&self, other: &Self, t: f32) -> Self
    where
        <S::Basis as Basis>::Coeffs: Coefficients,
    {
        let mut result = <<S::Basis as Basis>::Coeffs as Coefficients>::zero();
        for i in 0..<<S::Basis as Basis>::Coeffs as Coefficients>::NUM_COEFFS {
            let a = self.coeffs.get(i);
            let b = other.coeffs.get(i);
            *result.get_mut(i) = a + t * (b - a);
        }
        Self {
            coeffs: result,
            frame: self.frame.clone(),
            _symmetry: PhantomData,
        }
    }

    /// Get the L2 norm (energy) of the coefficients.
    pub fn energy(&self) -> f32
    where
        <S::Basis as Basis>::Coeffs: Coefficients,
    {
        self.coeffs.dot(&self.coeffs)
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

impl Manifold<Field> for CompressedManifold<RotInv2> {
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, _w: Field) -> Field {
        use crate::SH_NORM;

        // Normalize directions (vectorized)
        let r = (x * x + y * y + z * z).sqrt();
        let inv_r = Field::from(1.0) / r;
        let nx = x * inv_r;
        let ny = y * inv_r;
        let nz = z * inv_r;

        // Compute SH basis functions (vectorized)
        // l=0
        let y00 = Field::from(SH_NORM[0][0]);

        // l=1
        let y1m1 = Field::from(SH_NORM[1][1]) * ny;
        let y10 = Field::from(SH_NORM[1][0]) * nz;
        let y11 = Field::from(SH_NORM[1][1]) * nx;

        // l=2
        let y2m2 = Field::from(SH_NORM[2][2]) * nx * ny;
        let y2m1 = Field::from(SH_NORM[2][1]) * ny * nz;
        let y20 = Field::from(SH_NORM[2][0]) * (Field::from(3.0) * nz * nz - Field::from(1.0));
        let y21 = Field::from(SH_NORM[2][1]) * nx * nz;
        let y22 = Field::from(SH_NORM[2][2]) * (nx * nx - ny * ny);

        // Dot product with coefficients (broadcast scalar coeffs, multiply vectorized basis)
        let c = &self.inner.coeffs.coeffs;
        Field::from(c[0]) * y00
            + Field::from(c[1]) * y1m1
            + Field::from(c[2]) * y10
            + Field::from(c[3]) * y11
            + Field::from(c[4]) * y2m2
            + Field::from(c[5]) * y2m1
            + Field::from(c[6]) * y20
            + Field::from(c[7]) * y21
            + Field::from(c[8]) * y22
    }
}

// ============================================================================
// SH Projection from Manifold
// ============================================================================

/// Project a direction manifold onto SH basis via Monte Carlo sampling.
///
/// Takes N samples over the sphere and computes SH coefficients.
/// The manifold should map directions (x,y,z) to scalar values.
pub fn project_to_sh<M>(manifold: &M, samples: usize) -> Compressed<RotInv2>
where
    M: Manifold<Field, Output = Field>,
{
    use crate::SH_NORM;

    let mut coeffs = [0.0f32; 9];

    // Use a simple uniform sampling pattern (Fibonacci sphere)
    let golden_ratio = (1.0 + libm::sqrtf(5.0)) / 2.0;

    for i in 0..samples {
        let i_f = i as f32;
        let n_f = samples as f32;

        // Fibonacci sphere point
        let theta = 2.0 * core::f32::consts::PI * i_f / golden_ratio;
        let phi = libm::acosf(1.0 - 2.0 * (i_f + 0.5) / n_f);

        let sin_phi = libm::sinf(phi);
        let cos_phi = libm::cosf(phi);
        let sin_theta = libm::sinf(theta);
        let cos_theta = libm::cosf(theta);

        let x = sin_phi * cos_theta;
        let y = sin_phi * sin_theta;
        let z = cos_phi;

        // Evaluate manifold at this direction
        let x_field = Field::from(x);
        let y_field = Field::from(y);
        let z_field = Field::from(z);
        let val = manifold.eval_raw(x_field, y_field, z_field, Field::from(0.0));

        // Extract first lane
        let mut buf = [0.0f32; crate::PARALLELISM];
        val.store(&mut buf);
        let value = buf[0];

        // Accumulate SH coefficients (basis × value)
        // l=0
        coeffs[0] += SH_NORM[0][0] * value;
        // l=1
        coeffs[1] += SH_NORM[1][1] * y * value;
        coeffs[2] += SH_NORM[1][0] * z * value;
        coeffs[3] += SH_NORM[1][1] * x * value;
        // l=2
        coeffs[4] += SH_NORM[2][2] * x * y * value;
        coeffs[5] += SH_NORM[2][1] * y * z * value;
        coeffs[6] += SH_NORM[2][0] * (3.0 * z * z - 1.0) * value;
        coeffs[7] += SH_NORM[2][1] * x * z * value;
        coeffs[8] += SH_NORM[2][2] * (x * x - y * y) * value;
    }

    // Normalize by solid angle per sample (4π / samples)
    let weight = 4.0 * core::f32::consts::PI / (samples as f32);
    for c in coeffs.iter_mut() {
        *c *= weight;
    }

    Compressed::from_coeffs(ShBasis::new(coeffs))
}


// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: compute vector length
    fn vec_len(v: (f32, f32, f32)) -> f32 {
        libm::sqrtf(v.0 * v.0 + v.1 * v.1 + v.2 * v.2)
    }

    // Helper: normalize vector
    fn normalize(v: (f32, f32, f32)) -> (f32, f32, f32) {
        let len = vec_len(v);
        (v.0 / len, v.1 / len, v.2 / len)
    }

    // Helper: dot product
    #[allow(dead_code)]
    fn dot3(a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
        a.0 * b.0 + a.1 * b.1 + a.2 * b.2
    }

    // ========================================================================
    // Frame Tests
    // ========================================================================

    #[test]
    fn test_frame_identity() {
        let frame = Frame::identity();
        assert_eq!(frame.origin, (0.0, 0.0, 0.0));
        let z = frame.z_axis();
        assert!((z.0).abs() < 1e-6);
        assert!((z.1).abs() < 1e-6);
        assert!((z.2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_frame_to_local_translation() {
        let frame = Frame::centered_at((1.0, 2.0, 3.0));
        let local = frame.to_local((2.0, 4.0, 6.0));
        assert!((local.0 - 1.0).abs() < 1e-6);
        assert!((local.1 - 2.0).abs() < 1e-6);
        assert!((local.2 - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_frame_roundtrip() {
        let frame = Frame::aligned_to((1.0, 2.0, 3.0), (1.0, 1.0, 1.0));
        let world = (5.0, 6.0, 7.0);
        let local = frame.to_local(world);
        let back = frame.to_world(local);

        assert!((back.0 - world.0).abs() < 1e-5);
        assert!((back.1 - world.1).abs() < 1e-5);
        assert!((back.2 - world.2).abs() < 1e-5);
    }

    #[test]
    fn test_frame_aligned_orthonormal() {
        let normal = normalize((1.0, 2.0, 3.0));
        let frame = Frame::aligned_to((0.0, 0.0, 0.0), normal);

        // Z-axis should be the normal
        let z = frame.z_axis();
        assert!((z.0 - normal.0).abs() < 1e-5);
        assert!((z.1 - normal.1).abs() < 1e-5);
        assert!((z.2 - normal.2).abs() < 1e-5);

        // Local (0,0,1) should map to normal in world space
        let local_z = frame.to_world((0.0, 0.0, 1.0));
        assert!((local_z.0 - normal.0).abs() < 1e-5);
        assert!((local_z.1 - normal.1).abs() < 1e-5);
        assert!((local_z.2 - normal.2).abs() < 1e-5);
    }

    #[test]
    fn test_frame_rotation_preserves_length() {
        let frame = Frame::aligned_to((0.0, 0.0, 0.0), (1.0, 1.0, 0.0));
        let v = (3.0, 4.0, 5.0);
        let local = frame.to_local(v);
        let world = frame.to_world(local);

        let len_orig = vec_len(v);
        let len_local = vec_len(local);
        let len_back = vec_len(world);

        assert!((len_orig - len_local).abs() < 1e-5);
        assert!((len_orig - len_back).abs() < 1e-5);
    }

    // ========================================================================
    // ShBasis Tests
    // ========================================================================

    #[test]
    fn test_sh_basis_dot() {
        let a = ShBasis::<9>::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b = ShBasis::<9>::new([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!((a.dot(&b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_sh_basis_dot_general() {
        let a = ShBasis::<9>::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let b = ShBasis::<9>::new([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        // 1+2+3+4+5+6+7+8+9 = 45
        assert!((a.dot(&b) - 45.0).abs() < 1e-6);
    }

    #[test]
    fn test_sh_basis_zero() {
        let z = ShBasis::<9>::zero();
        for c in z.coeffs {
            assert!(c.abs() < 1e-10);
        }
    }

    #[test]
    fn test_sh_basis_eval_z_direction() {
        // +Z direction should have strong l=0 and l=1,m=0 components
        let coeffs = ShBasis::<9>::eval_at_scalar(0.0, 0.0, 1.0);
        assert!(coeffs.coeffs[0] > 0.0); // DC term (Y_0^0)
        assert!(coeffs.coeffs[2] > 0.0); // Y_1^0 (z-aligned)
        // Y_1^{-1} and Y_1^{1} should be ~0 for +Z
        assert!(coeffs.coeffs[1].abs() < 1e-6);
        assert!(coeffs.coeffs[3].abs() < 1e-6);
    }

    #[test]
    fn test_sh_basis_eval_x_direction() {
        // +X direction should have strong Y_1^1 component
        let coeffs = ShBasis::<9>::eval_at_scalar(1.0, 0.0, 0.0);
        assert!(coeffs.coeffs[0] > 0.0); // DC term
        assert!(coeffs.coeffs[3] > 0.0); // Y_1^1 (x-aligned)
        // Y_1^0 should be ~0 for directions in xy-plane
        assert!(coeffs.coeffs[2].abs() < 1e-6);
    }

    #[test]
    fn test_sh_basis_eval_y_direction() {
        // +Y direction should have strong Y_1^{-1} component
        let coeffs = ShBasis::<9>::eval_at_scalar(0.0, 1.0, 0.0);
        assert!(coeffs.coeffs[0] > 0.0); // DC term
        assert!(coeffs.coeffs[1] > 0.0); // Y_1^{-1} (y-aligned)
    }

    #[test]
    fn test_sh_orthonormality_approximate() {
        // SH basis functions are orthonormal over the sphere
        // ∫ Y_l^m Y_l'^m' dΩ = δ_{ll'} δ_{mm'}
        //
        // We approximate this with discrete samples

        let samples = 1000;
        let golden = (1.0 + libm::sqrtf(5.0)) / 2.0;

        // Compute ∫ Y_0^0 * Y_0^0 dΩ ≈ 1 (self-dot of DC)
        let mut integral_00_00 = 0.0;
        // Compute ∫ Y_0^0 * Y_1^0 dΩ ≈ 0 (orthogonal)
        let mut integral_00_10 = 0.0;

        for i in 0..samples {
            let i_f = i as f32;
            let n_f = samples as f32;
            let theta = 2.0 * core::f32::consts::PI * i_f / golden;
            let phi = libm::acosf(1.0 - 2.0 * (i_f + 0.5) / n_f);

            let x = libm::sinf(phi) * libm::cosf(theta);
            let y = libm::sinf(phi) * libm::sinf(theta);
            let z = libm::cosf(phi);

            let sh = ShBasis::<9>::eval_at_scalar(x, y, z);
            integral_00_00 += sh.coeffs[0] * sh.coeffs[0];
            integral_00_10 += sh.coeffs[0] * sh.coeffs[2];
        }

        // Normalize by solid angle
        let weight = 4.0 * core::f32::consts::PI / (samples as f32);
        integral_00_00 *= weight;
        integral_00_10 *= weight;

        // Y_0^0 self-integral should be ~1
        assert!((integral_00_00 - 1.0).abs() < 0.1);
        // Orthogonal integral should be ~0
        assert!(integral_00_10.abs() < 0.1);
    }

    // ========================================================================
    // Clebsch-Gordan Tests
    // ========================================================================

    #[test]
    fn test_cg_multiplication_dc_dc() {
        // DC × DC → DC
        let a = ShBasis::<9>::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b = ShBasis::<9>::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let result = a.multiply(&b, CG_ORDER_2);

        // Should have non-zero DC term
        assert!(result.coeffs[0].abs() > 0.0);
        // Higher order terms should be ~0
        for i in 1..9 {
            assert!(result.coeffs[i].abs() < 1e-6);
        }
    }

    #[test]
    fn test_cg_multiplication_dc_l1() {
        // DC × l=1 → l=1 (DC acts like identity up to scale)
        let dc = ShBasis::<9>::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let l1 = ShBasis::<9>::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let result = dc.multiply(&l1, CG_ORDER_2);

        // Should have l=1 component
        assert!(result.coeffs[1].abs() > 0.0);
    }

    #[test]
    fn test_cg_multiplication_l1_l1_to_dc() {
        // l=1 × l=1 → includes DC (trace)
        let a = ShBasis::<9>::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b = ShBasis::<9>::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let result = a.multiply(&b, CG_ORDER_2);

        // Should have DC contribution (Y_1^{-1} × Y_1^{-1} → Y_0^0)
        assert!(result.coeffs[0].abs() > 0.0);
    }

    #[test]
    fn test_cg_multiplication_commutative() {
        let a = ShBasis::<9>::new([1.0, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]);
        let b = ShBasis::<9>::new([0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let ab = a.multiply(&b, CG_ORDER_2);
        let ba = b.multiply(&a, CG_ORDER_2);

        for i in 0..9 {
            assert!((ab.coeffs[i] - ba.coeffs[i]).abs() < 1e-6);
        }
    }

    // ========================================================================
    // Compressed Field Tests
    // ========================================================================

    #[test]
    fn test_compressed_sample_constant() {
        // Constant field (DC only)
        let coeffs = ShBasis::<9>::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let compressed = Compressed::<RotInv2>::from_coeffs(coeffs);

        // Sample in multiple directions - should be approximately constant
        let val1 = compressed.sample((0.0, 0.0, 1.0));
        let val2 = compressed.sample((1.0, 0.0, 0.0));
        let val3 = compressed.sample((0.0, 1.0, 0.0));
        let val4 = compressed.sample(normalize((1.0, 1.0, 1.0)));

        // DC contribution is constant * Y_0^0(dir) ≈ 0.282
        assert!(val1 > 0.0);
        assert!((val1 - val2).abs() < 0.05);
        assert!((val2 - val3).abs() < 0.05);
        assert!((val3 - val4).abs() < 0.05);
    }

    #[test]
    fn test_compressed_scale() {
        let coeffs = ShBasis::<9>::new([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let c = Compressed::<RotInv2>::from_coeffs(coeffs);
        let scaled = c.scale(2.0);

        assert!((scaled.coeffs.coeffs[0] - 2.0).abs() < 1e-6);
        assert!((scaled.coeffs.coeffs[1] - 4.0).abs() < 1e-6);
        assert!((scaled.coeffs.coeffs[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_compressed_add() {
        let a = Compressed::<RotInv2>::from_coeffs(
            ShBasis::new([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        );
        let b = Compressed::<RotInv2>::from_coeffs(
            ShBasis::new([3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        );
        let sum = a.add(&b);

        assert!((sum.coeffs.coeffs[0] - 4.0).abs() < 1e-6);
        assert!((sum.coeffs.coeffs[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_compressed_lerp() {
        let a = Compressed::<RotInv2>::from_coeffs(
            ShBasis::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        );
        let b = Compressed::<RotInv2>::from_coeffs(
            ShBasis::new([10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        );

        let mid = a.lerp(&b, 0.5);
        assert!((mid.coeffs.coeffs[0] - 5.0).abs() < 1e-6);
        assert!((mid.coeffs.coeffs[1] - 10.0).abs() < 1e-6);

        let quarter = a.lerp(&b, 0.25);
        assert!((quarter.coeffs.coeffs[0] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_compressed_energy() {
        let c = Compressed::<RotInv2>::from_coeffs(
            ShBasis::new([3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        );
        // Energy = 3² + 4² = 25
        assert!((c.energy() - 25.0).abs() < 1e-6);
    }

    // ========================================================================
    // Manifold Integration Tests
    // ========================================================================

    #[test]
    fn test_compressed_manifold_eval() {
        let coeffs = ShBasis::<9>::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let compressed = Compressed::<RotInv2>::from_coeffs(coeffs);
        let manifold = CompressedManifold::new(compressed);

        // Evaluate at +Z direction
        let result = manifold.eval_raw(
            Field::from(0.0),
            Field::from(0.0),
            Field::from(1.0),
            Field::from(0.0),
        );

        // Extract result
        let mut buf = [0.0f32; crate::PARALLELISM];
        result.store(&mut buf);

        // Should be non-zero
        assert!(buf[0].abs() > 0.0);
    }

    #[test]
    fn test_compressed_manifold_directional() {
        // Create SH for +Z directional
        let coeffs = ShBasis::<9>::eval_at_scalar(0.0, 0.0, 1.0);
        let compressed = Compressed::<RotInv2>::from_coeffs(coeffs);
        let manifold = CompressedManifold::new(compressed);

        // Evaluate at +Z vs +X
        let val_z = manifold.eval_raw(
            Field::from(0.0), Field::from(0.0), Field::from(1.0), Field::from(0.0)
        );
        let val_x = manifold.eval_raw(
            Field::from(1.0), Field::from(0.0), Field::from(0.0), Field::from(0.0)
        );

        let mut buf_z = [0.0f32; crate::PARALLELISM];
        let mut buf_x = [0.0f32; crate::PARALLELISM];
        val_z.store(&mut buf_z);
        val_x.store(&mut buf_x);

        // +Z direction should give higher value
        assert!(buf_z[0] > buf_x[0]);
    }

    // ========================================================================
    // Edge Cases and Numerical Stability
    // ========================================================================

    #[test]
    fn test_sh_eval_zero_direction() {
        // Zero direction should not panic
        let coeffs = ShBasis::<9>::eval_at_scalar(0.0, 0.0, 0.0);
        // With proper handling, should return zeros or handle gracefully
        for c in coeffs.coeffs {
            assert!(c.is_finite());
        }
    }

    #[test]
    fn test_sh_eval_unnormalized() {
        // Should work with unnormalized directions
        let coeffs1 = ShBasis::<9>::eval_at_scalar(0.0, 0.0, 1.0);
        let coeffs2 = ShBasis::<9>::eval_at_scalar(0.0, 0.0, 10.0);

        // Should be the same (direction only matters)
        for i in 0..9 {
            assert!((coeffs1.coeffs[i] - coeffs2.coeffs[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_frame_degenerate_normal() {
        // Nearly-zero normal should still produce valid frame
        let frame = Frame::aligned_to((0.0, 0.0, 0.0), (1e-10, 1e-10, 1e-10));
        let local = frame.to_local((1.0, 0.0, 0.0));
        let back = frame.to_world(local);

        // Should not be NaN
        assert!(back.0.is_finite());
        assert!(back.1.is_finite());
        assert!(back.2.is_finite());
    }

    #[test]
    fn test_compressed_zero_coefficients() {
        let zero = Compressed::<RotInv2>::from_coeffs(ShBasis::zero());

        // Sampling should return 0
        let val = zero.sample((0.0, 0.0, 1.0));
        assert!(val.abs() < 1e-10);

        // DC term should be 0
        assert!(zero.dc_term().abs() < 1e-10);

        // Energy should be 0
        assert!(zero.energy().abs() < 1e-10);
    }

    #[test]
    fn test_compressed_large_coefficients() {
        let large = Compressed::<RotInv2>::from_coeffs(
            ShBasis::new([1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6])
        );

        let val = large.sample((0.0, 0.0, 1.0));
        assert!(val.is_finite());
        assert!(val.abs() > 0.0);
    }
}
