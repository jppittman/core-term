//! # Spherical Harmonics Combinator
//!
//! Spherical harmonics (SH) are orthonormal basis functions on the unit sphere S².
//! They form the foundation for efficient global illumination in computer graphics.
//!
//! ## The Connection: Linear Attention IS Harmonic Global Illumination
//!
//! The profound insight connecting ML and graphics:
//!
//! **Spherical Harmonics**:
//! - Project lighting functions L(ω) onto orthonormal basis: L(ω) = Σ c_lm Y_lm(ω)
//! - Compress infinite-dimensional lighting into finite coefficients
//! - Irradiance computation becomes dot product of SH vectors
//!
//! **Linear Attention**:
//! - Decompose attention kernel: K(q,k) = φ(q)·φ(k)ᵀ
//! - Compress O(n²) attention into O(n) feature space
//! - Attention becomes matrix multiplication in projected space
//!
//! Both share the same mathematical structure:
//! - Project high-dimensional interactions onto orthogonal basis
//! - Compute interactions as inner products in compressed space
//! - Avoid quadratic complexity via basis decomposition
//!
//! This combinator enables the same trick: represent lighting/fields
//! as SH coefficients, compute irradiance as dot products.

use crate::manifold::Manifold;
use crate::numeric::Numeric;
use crate::Field;

// ============================================================================
// Constants
// ============================================================================

/// Normalization constants for real spherical harmonics up to band 3.
/// K_lm = sqrt((2l+1)/(4π) * (l-|m|)!/(l+|m|)!)
pub const SH_NORM: [[f32; 7]; 4] = [
    // l=0
    [0.282_094_8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    // l=1
    [0.488_602_5, 0.488_602_5, 0.0, 0.0, 0.0, 0.0, 0.0],
    // l=2
    [0.315_391_57, 1.092_548_4, 0.546_274_2, 0.0, 0.0, 0.0, 0.0],
    // l=3
    [0.373_176_33, 0.457_045_8, 1.445_305_8, 0.590_043_6, 0.0, 0.0, 0.0],
];

// ============================================================================
// SphericalHarmonic Basis Manifold
// ============================================================================

/// Spherical harmonic basis function Y_l^m.
///
/// Evaluates the real spherical harmonic of degree `l` and order `m`
/// at the direction given by (x, y, z) interpreted as a unit vector.
///
/// Input coordinates (x, y, z) are treated as a direction vector and
/// internally normalized to the unit sphere.
#[derive(Clone, Copy, Debug)]
pub struct SphericalHarmonic<const L: usize, const M: i32>;

impl<I: Numeric, const L: usize, const M: i32> Manifold<I> for SphericalHarmonic<L, M> {
    type Output = I;

    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, _w: I) -> I {
        // Normalize direction
        let r = (x * x + y * y + z * z).sqrt();
        let inv_r = I::from_f32(1.0) / r;
        let nx = x * inv_r;
        let ny = y * inv_r;
        let nz = z * inv_r;

        // Convert to spherical coordinates
        // θ = acos(z), φ = atan2(y, x)
        let cos_theta = nz;
        let sin_theta = (I::from_f32(1.0) - cos_theta * cos_theta).sqrt();
        let phi = ny.atan2(nx);

        // Evaluate Y_l^m
        eval_sh::<I, L, M>(cos_theta, sin_theta, phi)
    }
}

/// Evaluate real spherical harmonic Y_l^m.
#[inline(always)]
fn eval_sh<I: Numeric, const L: usize, const M: i32>(
    cos_theta: I,
    sin_theta: I,
    phi: I,
) -> I {
    let plm = legendre_p::<I, L, M>(cos_theta, sin_theta);
    let norm = I::from_f32(SH_NORM[L][M.unsigned_abs() as usize]);

    if M > 0 {
        // Y_l^m = K * P_l^m * cos(m*φ)
        let m_phi = I::from_f32(M as f32) * phi;
        norm * plm * m_phi.cos()
    } else if M < 0 {
        // Y_l^{-m} = K * P_l^m * sin(|m|*φ)
        let m_phi = I::from_f32((-M) as f32) * phi;
        norm * plm * m_phi.sin()
    } else {
        // Y_l^0 = K * P_l^0
        norm * plm
    }
}

/// Associated Legendre polynomial P_l^|m|(cos θ).
///
/// Uses recurrence relations optimized for low-order SH (l ≤ 3).
#[inline(always)]
fn legendre_p<I: Numeric, const L: usize, const M: i32>(cos_theta: I, sin_theta: I) -> I {
    let m = M.unsigned_abs() as usize;

    // Start with P_m^m = (-1)^m * (2m-1)!! * sin^m(θ)
    let mut pmm = I::from_f32(1.0);
    if m > 0 {
        let mut fact = I::from_f32(1.0);
        for i in 1..=m {
            fact = fact * I::from_f32((2 * i - 1) as f32);
        }
        // sin^m(θ)
        let mut sin_pow = I::from_f32(1.0);
        for _ in 0..m {
            sin_pow = sin_pow * sin_theta;
        }
        pmm = fact * sin_pow;
        if m % 2 == 1 {
            pmm = I::from_f32(0.0) - pmm; // (-1)^m
        }
    }

    if L == m {
        return pmm;
    }

    // P_{m+1}^m = x * (2m+1) * P_m^m
    let pmm1 = cos_theta * I::from_f32((2 * m + 1) as f32) * pmm;
    if L == m + 1 {
        return pmm1;
    }

    // Recurrence: P_l^m = ((2l-1)*x*P_{l-1}^m - (l+m-1)*P_{l-2}^m) / (l-m)
    let mut p_prev2 = pmm;
    let mut p_prev1 = pmm1;
    for l in (m + 2)..=L {
        let a = I::from_f32((2 * l - 1) as f32);
        let b = I::from_f32((l + m - 1) as f32);
        let c = I::from_f32((l - m) as f32);
        let p_curr = (a * cos_theta * p_prev1 - b * p_prev2) / c;
        p_prev2 = p_prev1;
        p_prev1 = p_curr;
    }
    p_prev1
}

// ============================================================================
// SH Coefficient Vector (for projection/reconstruction)
// ============================================================================

/// Spherical harmonic coefficient vector for band L.
///
/// Contains (L+1)² coefficients representing a function on S².
/// This is the compressed representation that enables O(n) irradiance.
///
/// We use a fixed-size array with NUM_COEFFS as a const generic parameter
/// to work around Rust's limitations on Self in const expressions.
#[derive(Clone, Debug)]
pub struct ShCoeffs<const NUM_COEFFS: usize> {
    /// Coefficients indexed as [l*l + l + m] for l ∈ [0, MAX_L], m ∈ [-l, l].
    pub coeffs: [f32; NUM_COEFFS],
}

impl<const NUM_COEFFS: usize> ShCoeffs<NUM_COEFFS> {
    /// Create zero coefficients.
    pub const fn zero() -> Self {
        Self {
            coeffs: [0.0; NUM_COEFFS],
        }
    }

    /// Get coefficient index for (l, m).
    #[inline(always)]
    pub const fn index(l: usize, m: i32) -> usize {
        (l * l) + l + (m as usize)
    }

    /// Dot product with another SH vector (computes integral over S²).
    ///
    /// This is the key operation: irradiance = light_sh · transfer_sh
    #[inline(always)]
    pub fn dot(&self, other: &Self) -> f32 {
        let mut sum = 0.0;
        for i in 0..NUM_COEFFS {
            sum += self.coeffs[i] * other.coeffs[i];
        }
        sum
    }
}

// ============================================================================
// SH Projection Manifold
// ============================================================================

/// Projects a direction manifold onto spherical harmonic basis.
///
/// Takes coordinates and outputs the SH basis vector for that direction.
/// This enables computing SH projections of any manifold.
#[derive(Clone, Copy, Debug)]
pub struct ShProject<const NUM_COEFFS: usize>;

// ============================================================================
// SH Reconstruction Manifold
// ============================================================================

/// Reconstructs a function value from SH coefficients.
///
/// Given SH coefficients and a direction, evaluates the reconstructed function.
/// This wraps the coefficients and evaluates Σ c_lm Y_lm(direction).
pub struct ShReconstruct<const NUM_COEFFS: usize, M> {
    /// The underlying direction manifold.
    pub direction: M,
    /// SH coefficients.
    pub coeffs: ShCoeffs<NUM_COEFFS>,
}

impl<const NUM_COEFFS: usize, M> ShReconstruct<NUM_COEFFS, M> {
    /// Create a new SH reconstruction.
    pub fn new(direction: M, coeffs: ShCoeffs<NUM_COEFFS>) -> Self {
        Self { direction, coeffs }
    }
}

// For now, implement only for 9 coefficients (band 2, diffuse lighting)
impl<M: Manifold<Field, Output = (Field, Field, Field)>> Manifold<Field> for ShReconstruct<9, M> {
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Field {
        let (dx, dy, dz) = self.direction.eval_raw(x, y, z, w);

        // Normalize direction
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        let inv_r = Field::from(1.0) / r;
        let nx = dx * inv_r;
        let ny = dy * inv_r;
        let nz = dz * inv_r;

        // Evaluate SH basis and accumulate
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

        // Dot product with coefficients
        Field::from(self.coeffs.coeffs[0]) * y00
            + Field::from(self.coeffs.coeffs[1]) * y1m1
            + Field::from(self.coeffs.coeffs[2]) * y10
            + Field::from(self.coeffs.coeffs[3]) * y11
            + Field::from(self.coeffs.coeffs[4]) * y2m2
            + Field::from(self.coeffs.coeffs[5]) * y2m1
            + Field::from(self.coeffs.coeffs[6]) * y20
            + Field::from(self.coeffs.coeffs[7]) * y21
            + Field::from(self.coeffs.coeffs[8]) * y22
    }
}

// ============================================================================
// Zonal Harmonics (Rotation-Invariant)
// ============================================================================

/// Zonal harmonic (m=0 spherical harmonic).
///
/// Zonal harmonics are rotationally symmetric around the z-axis.
/// They're particularly efficient for axis-aligned lighting.
#[derive(Clone, Copy, Debug)]
pub struct ZonalHarmonic<const L: usize>;

impl<I: Numeric, const L: usize> Manifold<I> for ZonalHarmonic<L> {
    type Output = I;

    #[inline(always)]
    fn eval_raw(&self, x: I, y: I, z: I, _w: I) -> I {
        // Normalize direction
        let r = (x * x + y * y + z * z).sqrt();
        let cos_theta = z / r;
        let sin_theta = (I::from_f32(1.0) - cos_theta * cos_theta).sqrt();

        // Y_l^0 = K_l^0 * P_l^0(cos θ)
        let plm = legendre_p::<I, L, 0>(cos_theta, sin_theta);
        I::from_f32(SH_NORM[L][0]) * plm
    }
}

// ============================================================================
// Convenience Type Aliases
// ============================================================================

/// First-order SH coefficients (4 coefficients, ambient + directional).
/// Band L=1: (1+1)² = 4 coefficients.
pub type Sh1 = ShCoeffs<4>;

/// Second-order SH coefficients (9 coefficients, diffuse lighting).
/// Band L=2: (2+1)² = 9 coefficients.
pub type Sh2 = ShCoeffs<9>;

/// Third-order SH coefficients (16 coefficients, glossy highlights).
/// Band L=3: (3+1)² = 16 coefficients.
pub type Sh3 = ShCoeffs<16>;

// ============================================================================
// SH Multiplication (Clebsch-Gordan)
// ============================================================================

/// Clebsch-Gordan coefficients for SH L2 × L2 → L2 product.
///
/// These are mathematical constants - the "multiplication table" for spherical harmonics.
/// Entry (i, j, k, w) means: a[i] × b[j] contributes weight w to result[k].
///
/// For band L=2, we have 9×9=81 possible products, but selection rules eliminate many.
/// Only products where |m1+m2| ≤ 2 contribute to band L≤2 output.
///
/// Reference: Green, "Spherical Harmonic Lighting: The Gritty Details"
pub static SH2_PRODUCT_TABLE: &[(usize, usize, usize, f32)] = &[
    // L0 × L0 → L0 (DC × DC = DC)
    // Y00 × Y00 → Y00 with weight = 1/(2√π) ≈ 0.282
    (0, 0, 0, 0.282_095),
    // L0 × L1 → L1 (DC scales L1)
    (0, 1, 1, 0.282_095),
    (1, 0, 1, 0.282_095),
    (0, 2, 2, 0.282_095),
    (2, 0, 2, 0.282_095),
    (0, 3, 3, 0.282_095),
    (3, 0, 3, 0.282_095),
    // L0 × L2 → L2 (DC scales L2)
    (0, 4, 4, 0.282_095),
    (4, 0, 4, 0.282_095),
    (0, 5, 5, 0.282_095),
    (5, 0, 5, 0.282_095),
    (0, 6, 6, 0.282_095),
    (6, 0, 6, 0.282_095),
    (0, 7, 7, 0.282_095),
    (7, 0, 7, 0.282_095),
    (0, 8, 8, 0.282_095),
    (8, 0, 8, 0.282_095),
    // L1 × L1 → L0, L2
    // Y1m × Y1m' contributions (selection rules: m+m' = output m)
    (1, 1, 0, 0.282_095),
    (2, 2, 0, 0.282_095),
    (3, 3, 0, 0.282_095),
    // L1 × L1 → L2 (various m combinations)
    (1, 2, 5, 0.126_157),
    (2, 1, 5, 0.126_157),
    (1, 3, 7, 0.126_157),
    (3, 1, 7, 0.126_157),
    (2, 3, 4, 0.126_157),
    (3, 2, 4, 0.126_157),
    (1, 1, 6, 0.218_509),
    (2, 2, 6, -0.218_509),
    (3, 3, 6, 0.218_509),
    (2, 2, 8, 0.126_157),
    (3, 3, 8, -0.126_157),
    // Higher-order products (L1×L2, L2×L1, L2×L2) truncate to band 2
    // These contribute to output but are band-limited
    // L1 × L2 → L1 (backprojection)
    (1, 6, 1, 0.218_509),
    (6, 1, 1, 0.218_509),
    (2, 6, 2, -0.218_509),
    (6, 2, 2, -0.218_509),
    (3, 6, 3, 0.218_509),
    (6, 3, 3, 0.218_509),
    // Additional cross terms
    (1, 5, 2, 0.126_157),
    (5, 1, 2, 0.126_157),
    (2, 5, 1, 0.126_157),
    (5, 2, 1, 0.126_157),
    (1, 7, 3, 0.126_157),
    (7, 1, 3, 0.126_157),
    (3, 7, 1, 0.126_157),
    (7, 3, 1, 0.126_157),
    (2, 4, 3, 0.126_157),
    (4, 2, 3, 0.126_157),
    (3, 4, 2, 0.126_157),
    (4, 3, 2, 0.126_157),
];

/// Multiply two SH2 coefficient vectors via Clebsch-Gordan product.
///
/// This computes the product of two spherical harmonic representations,
/// giving the SH coefficients of the pointwise product of the two functions.
///
/// # Mathematical Background
///
/// For two functions f, g on S² represented in SH:
///   f(ω) = Σ f_i Y_i(ω)
///   g(ω) = Σ g_j Y_j(ω)
///
/// Their product (f·g)(ω) has SH coefficients:
///   (f·g)_k = Σ_{i,j} C_{ijk} f_i g_j
///
/// where C_{ijk} are the Clebsch-Gordan coefficients (or Gaunt coefficients).
pub fn sh2_multiply(a: &Sh2, b: &Sh2) -> Sh2 {
    let mut result = [0.0f32; 9];
    for &(i, j, k, weight) in SH2_PRODUCT_TABLE {
        result[k] += a.coeffs[i] * b.coeffs[j] * weight;
    }
    ShCoeffs { coeffs: result }
}

/// Evaluate SH2 coefficients at a direction.
///
/// Reconstructs the function value from SH coefficients by computing
/// Σ c_i Y_i(direction).
impl Sh2 {
    /// Evaluate the SH representation at a direction.
    pub fn eval(&self, dir: (Field, Field, Field)) -> Field {
        let basis = sh2_basis_at(dir);
        let mut result = Field::from(0.0);
        for i in 0..9 {
            result = result + Field::from(self.coeffs[i]) * basis[i];
        }
        result
    }
}

/// Evaluate all 9 SH2 basis functions at a direction.
///
/// Returns [Y00, Y1-1, Y10, Y11, Y2-2, Y2-1, Y20, Y21, Y22] evaluated at dir.
pub fn sh2_basis_at(dir: (Field, Field, Field)) -> [Field; 9] {
    let (x, y, z) = dir;

    // Normalize direction
    let r2 = x * x + y * y + z * z;
    let inv_r = r2.rsqrt();
    let nx = x * inv_r;
    let ny = y * inv_r;
    let nz = z * inv_r;

    // SH basis functions (real, orthonormalized)
    // Using standard conventions from computer graphics
    [
        // L=0
        Field::from(SH_NORM[0][0]), // Y00 = 0.282...
        // L=1
        Field::from(SH_NORM[1][1]) * ny, // Y1-1
        Field::from(SH_NORM[1][0]) * nz, // Y10
        Field::from(SH_NORM[1][1]) * nx, // Y11
        // L=2
        Field::from(SH_NORM[2][2]) * nx * ny,                                        // Y2-2
        Field::from(SH_NORM[2][1]) * ny * nz,                                        // Y2-1
        Field::from(SH_NORM[2][0]) * (Field::from(3.0) * nz * nz - Field::from(1.0)), // Y20
        Field::from(SH_NORM[2][1]) * nx * nz,                                        // Y21
        Field::from(SH_NORM[2][2]) * (nx * nx - ny * ny),                             // Y22
    ]
}

// ============================================================================
// Field-Based SH (Runtime Computations)
// ============================================================================

/// SH2 coefficients in Field space (for per-pixel varying SH).
///
/// Unlike `Sh2` which stores f32 coefficients (for static environment lighting),
/// `Sh2Field` stores Field coefficients that vary per SIMD lane.
///
/// Used for visibility projection and other runtime SH computations.
#[derive(Clone)]
pub struct Sh2Field {
    /// The 9 SH coefficients as Field values.
    pub coeffs: [Field; 9],
}

impl Sh2Field {
    /// Create from constant coefficients.
    pub fn from_sh2(sh: &Sh2) -> Self {
        Self {
            coeffs: [
                Field::from(sh.coeffs[0]),
                Field::from(sh.coeffs[1]),
                Field::from(sh.coeffs[2]),
                Field::from(sh.coeffs[3]),
                Field::from(sh.coeffs[4]),
                Field::from(sh.coeffs[5]),
                Field::from(sh.coeffs[6]),
                Field::from(sh.coeffs[7]),
                Field::from(sh.coeffs[8]),
            ],
        }
    }

    /// Create zeroed SH2Field.
    pub fn zero() -> Self {
        Self {
            coeffs: [Field::from(0.0); 9],
        }
    }

    /// Evaluate at a direction.
    pub fn eval(&self, dir: (Field, Field, Field)) -> Field {
        let basis = sh2_basis_at(dir);
        let mut result = Field::from(0.0);
        for i in 0..9 {
            result = result + self.coeffs[i] * basis[i];
        }
        result
    }

    /// Extract L0 coefficient (the "DC term" / hemisphere average).
    pub fn l0(&self) -> Field {
        self.coeffs[0]
    }
}

/// Multiply Sh2 (static) × Sh2Field (runtime) → Sh2Field.
///
/// Used when multiplying static environment lighting by per-pixel visibility.
pub fn sh2_multiply_static_field(a: &Sh2, b: &Sh2Field) -> Sh2Field {
    let mut result = [Field::from(0.0); 9];
    for &(i, j, k, weight) in SH2_PRODUCT_TABLE {
        result[k] = result[k] + Field::from(a.coeffs[i]) * b.coeffs[j] * Field::from(weight);
    }
    Sh2Field { coeffs: result }
}

/// Multiply two Sh2Field (both runtime) → Sh2Field.
///
/// Used when both SH representations vary per-pixel.
pub fn sh2_multiply_field(a: &Sh2Field, b: &Sh2Field) -> Sh2Field {
    let mut result = [Field::from(0.0); 9];
    for &(i, j, k, weight) in SH2_PRODUCT_TABLE {
        result[k] = result[k] + a.coeffs[i] * b.coeffs[j] * Field::from(weight);
    }
    Sh2Field { coeffs: result }
}

/// Cosine lobe in SH2 basis (analytic, for Lambertian diffuse).
///
/// Given a surface normal, returns the SH coefficients of the cosine-weighted
/// hemisphere integral kernel. Dot product with environment SH gives irradiance.
pub fn cosine_lobe_sh2(n: (Field, Field, Field)) -> Sh2Field {
    // Zonal harmonics coefficients for clamped cosine lobe
    // From Ramamoorthi & Hanrahan "An Efficient Representation for Irradiance..."
    //
    // L0: π (integrated over hemisphere)
    // L1: 2π/3 × normal components
    // L2: π/4 × quadratic (small for Lambertian)

    let l0 = Field::from(0.886_227); // √π
    let l1_scale = Field::from(1.023_327); // 2√(π/3)

    Sh2Field {
        coeffs: [
            l0,
            l1_scale * n.1, // Y1-1 (y direction)
            l1_scale * n.2, // Y10  (z direction)
            l1_scale * n.0, // Y11  (x direction)
            Field::from(0.0), // L2 terms small for Lambertian
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        ],
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sh_orthonormality() {
        // Y_0^0 should be constant = 1/(2√π) ≈ 0.282
        let y00 = SphericalHarmonic::<0, 0>;
        let val = y00.eval_raw(
            Field::from(1.0),
            Field::from(0.0),
            Field::from(0.0),
            Field::from(0.0),
        );
        let mut buf = [0.0f32; crate::PARALLELISM];
        val.store(&mut buf);
        assert!((buf[0] - 0.282).abs() < 0.01);
    }

    #[test]
    fn test_sh_coeffs_dot() {
        let a = Sh2 {
            coeffs: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        let b = Sh2 {
            coeffs: [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        assert!((a.dot(&b) - 2.0).abs() < 0.001);
    }
}
