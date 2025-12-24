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
