//! Graphics integration for pixelflow-ml.
//!
//! This module provides harmonic attention and SH feature maps. A feature map
//! is a [`Kernel`] transformer -- the language's own value, an arena fragment --
//! so a feature map composes into a larger kernel and reaches numbers the one
//! way anything does: compiled at a lattice's shape, then collapsed.
//!
//! The SH coefficient vector and its normalization table live here rather than
//! in pixelflow-core. They are scalar data for this experiment, and the
//! `Field`-typed spherical-harmonic combinators they used to sit beside went
//! with the per-batch tier (docs/plans/2026-09-06-kernel-with-a-lattice.md,
//! S4b-2).
//!
//! ## Linear Attention IS Harmonic Global Illumination
//!
//! Both linear attention and spherical harmonic lighting solve the same problem:
//! compress infinite/quadratic interactions into finite/linear operations.

use alloc::vec::Vec;
use pixelflow_core::Kernel;

// ============================================================================
// Spherical harmonic coefficients (scalar data)
// ============================================================================

/// Normalization constants for real spherical harmonics up to band 3:
/// `K_lm = sqrt((2l+1)/(4pi) * (l-|m|)!/(l+|m|)!)`.
pub const SH_NORM: [[f32; 7]; 4] = [
    // l=0
    [0.282_094_8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    // l=1
    [0.488_602_5, 0.488_602_5, 0.0, 0.0, 0.0, 0.0, 0.0],
    // l=2
    [0.315_391_57, 1.092_548_4, 0.546_274_2, 0.0, 0.0, 0.0, 0.0],
    // l=3
    [
        0.373_176_33,
        0.457_045_8,
        1.445_305_8,
        0.590_043_6,
        0.0,
        0.0,
        0.0,
    ],
];

/// Spherical harmonic coefficient vector for the band with `NUM_COEFFS` =
/// (L+1) squared coefficients: the compressed representation that makes
/// irradiance O(n).
#[derive(Clone, Debug)]
pub struct ShCoeffs<const NUM_COEFFS: usize> {
    /// Coefficients indexed as `[l*l + l + m]` for l in [0, L], m in [-l, l].
    pub coeffs: [f32; NUM_COEFFS],
}

impl<const NUM_COEFFS: usize> ShCoeffs<NUM_COEFFS> {
    /// Create zero coefficients.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            coeffs: [0.0; NUM_COEFFS],
        }
    }

    /// Coefficient index for `(l, m)`.
    #[inline(always)]
    #[must_use]
    pub const fn index(l: usize, m: i32) -> usize {
        (l * l) + l + (m as usize)
    }

    /// Dot product with another SH vector -- the integral over the sphere.
    #[inline(always)]
    #[must_use]
    pub fn dot(&self, other: &Self) -> f32 {
        let mut sum = 0.0;
        for i in 0..NUM_COEFFS {
            sum += self.coeffs[i] * other.coeffs[i];
        }
        sum
    }
}

/// Second-order SH coefficients (9 coefficients, diffuse lighting).
pub type Sh2 = ShCoeffs<9>;

// ============================================================================
// Feature Maps: The Bridge Between Attention and SH
// ============================================================================

/// A feature map φ that transforms inputs for linear attention.
///
/// The key insight: φ should behave like spherical harmonic projection.
/// Different φ choices give different attention characteristics:
///
/// - `elu(x) + 1`: Positive features, smooth attention
/// - `exp(x)`: Approximates softmax (Performer)
/// - SH basis: Directly computes irradiance-like attention
pub trait FeatureMap: Send + Sync {
    /// Apply the feature map to a value.
    fn apply(&self, x: &Kernel) -> Kernel;

    /// Feature dimension (number of output features).
    fn dim(&self) -> usize;
}

/// ELU + 1 feature map (ensures positivity).
///
/// φ(x) = ELU(x) + 1 = max(x, 0) + min(exp(x) - 1, 0) + 1
///
/// This is the simplest positive feature map, used in many linear attention variants.
#[derive(Clone, Copy, Debug, Default)]
pub struct EluFeature;

impl FeatureMap for EluFeature {
    #[inline(always)]
    fn apply(&self, x: &Kernel) -> Kernel {
        // ELU(x) + 1 = max(0, x) + exp(min(0, x))
        let zero = Kernel::constant(0.0);
        x.max(&zero).add(&x.min(&zero).exp())
    }

    fn dim(&self) -> usize {
        1
    }
}

/// Random Fourier Features (approximates RBF kernel).
///
/// φ(x) = [cos(ω₁·x), sin(ω₁·x), ..., cos(ωₙ·x), sin(ωₙ·x)] / √n
///
/// This approximates the Gaussian RBF kernel k(x,y) = exp(-||x-y||²/2σ²).
/// The connection to SH: both are Fourier-like basis expansions!
#[derive(Clone, Debug)]
pub struct RandomFourierFeature {
    /// Random frequencies (sampled from N(0, 1/σ²)).
    pub frequencies: Vec<f32>,
    /// Number of features (should be even for sin/cos pairs).
    pub num_features: usize,
}

impl RandomFourierFeature {
    /// Create new random Fourier features.
    #[must_use]
    pub fn new(frequencies: Vec<f32>) -> Self {
        let num_features = frequencies.len() * 2;
        Self {
            frequencies,
            num_features,
        }
    }
}

// ============================================================================
// Harmonic Attention: The Synthesis
// ============================================================================

/// Harmonic Linear Attention using spherical harmonic-inspired features.
///
/// This is the key innovation: we use SH coefficients as the feature map,
/// making attention directly compute irradiance-like quantities.
///
/// ```text
/// Attention(q, K, V) = Σᵢ φ(q)·φ(kᵢ) · vᵢ
///                    = φ(q) · (Σᵢ φ(kᵢ) · vᵢ)
///                    = φ(q) · S
/// ```
///
/// Where S = Σᵢ φ(kᵢ)⊗vᵢ is the "spherical harmonic lighting" accumulated
/// from all key-value pairs.
#[derive(Clone, Debug)]
pub struct HarmonicAttention<const NUM_COEFFS: usize> {
    /// Accumulated SH coefficients (the "lighting environment").
    /// Shape: [NUM_COEFFS, value_dim]
    pub accumulated: Vec<ShCoeffs<NUM_COEFFS>>,
    /// Normalizing denominator (sum of feature norms).
    pub denominator: ShCoeffs<NUM_COEFFS>,
}

impl<const NUM_COEFFS: usize> HarmonicAttention<NUM_COEFFS> {
    /// Create a new harmonic attention layer.
    #[must_use]
    pub fn new(value_dim: usize) -> Self {
        Self {
            accumulated: (0..value_dim).map(|_| ShCoeffs::zero()).collect(),
            denominator: ShCoeffs::zero(),
        }
    }

    /// Accumulate a key-value pair into the attention state.
    ///
    /// This is like adding a light source to the SH environment.
    pub fn accumulate(&mut self, key_sh: &ShCoeffs<NUM_COEFFS>, value: &[f32]) {
        // S += φ(k) ⊗ v
        for (i, &v) in value.iter().enumerate() {
            if i < self.accumulated.len() {
                for j in 0..NUM_COEFFS {
                    self.accumulated[i].coeffs[j] += key_sh.coeffs[j] * v;
                }
            }
        }
        // denominator += φ(k)
        for j in 0..NUM_COEFFS {
            self.denominator.coeffs[j] += key_sh.coeffs[j];
        }
    }

    /// Query the attention with a direction (like computing irradiance).
    ///
    /// Returns the attended value for the given query direction.
    pub fn query(&self, query_sh: &ShCoeffs<NUM_COEFFS>, output: &mut [f32]) {
        // Compute normalization factor
        let norm = query_sh.dot(&self.denominator).max(1e-6);

        // Compute attention output: φ(q)·S / φ(q)·D
        for (i, out) in output.iter_mut().enumerate() {
            if i < self.accumulated.len() {
                *out = query_sh.dot(&self.accumulated[i]) / norm;
            }
        }
    }

    /// Reset the accumulated state.
    pub fn reset(&mut self) {
        for sh in self.accumulated.iter_mut() {
            *sh = ShCoeffs::zero();
        }
        self.denominator = ShCoeffs::zero();
    }
}

// ============================================================================
// SH Feature Projection
// ============================================================================

/// Projects coordinates into spherical harmonic feature space.
///
/// This manifold transforms (x, y, z) directions into SH coefficient vectors,
/// serving as the feature map φ for harmonic attention.
pub struct ShFeatureMap<const NUM_COEFFS: usize>;

impl ShFeatureMap<9> {
    /// The SH feature map at a direction, as nine kernels.
    ///
    /// The direction is normalized inside the kernel, so a caller may hand in
    /// the coordinate kernels (`Kernel::x()` and friends) or any computed
    /// direction. The nine share one normalization -- hash-consing folds the
    /// repeated subexpression before the backend ever sees it.
    #[must_use]
    pub fn project(x: &Kernel, y: &Kernel, z: &Kernel) -> [Kernel; 9] {
        let k = Kernel::constant;

        let inv_r = x.mul(x).add(&y.mul(y)).add(&z.mul(z)).rsqrt();
        let nx = x.mul(&inv_r);
        let ny = y.mul(&inv_r);
        let nz = z.mul(&inv_r);

        [
            // l=0
            k(SH_NORM[0][0]),
            // l=1
            k(SH_NORM[1][1]).mul(&ny),
            k(SH_NORM[1][0]).mul(&nz),
            k(SH_NORM[1][1]).mul(&nx),
            // l=2
            k(SH_NORM[2][2]).mul(&nx).mul(&ny),
            k(SH_NORM[2][1]).mul(&ny).mul(&nz),
            k(SH_NORM[2][0]).mul(&k(3.0).mul(&nz).mul(&nz).sub(&k(1.0))),
            k(SH_NORM[2][1]).mul(&nx).mul(&nz),
            k(SH_NORM[2][2]).mul(&nx.mul(&nx).sub(&ny.mul(&ny))),
        ]
    }
}

// ============================================================================
// Linear Attention Layer (General)
// ============================================================================

/// A general linear attention layer with configurable feature map.
///
/// ```text
/// Output = φ(Q) · (φ(K)^T · V) / (φ(Q) · φ(K)^T · 1)
/// ```
///
/// The denominator normalizes to prevent exploding values.
pub struct LinearAttention<F: FeatureMap> {
    /// The feature map to use.
    pub feature_map: F,
    /// Accumulated key-value outer products.
    pub kv_state: Vec<f32>,
    /// Accumulated key features for normalization.
    pub k_state: Vec<f32>,
    /// Feature dimension.
    pub feature_dim: usize,
    /// Value dimension.
    pub value_dim: usize,
}

impl<F: FeatureMap> LinearAttention<F> {
    /// Create a new linear attention layer.
    pub fn new(feature_map: F, feature_dim: usize, value_dim: usize) -> Self {
        Self {
            feature_map,
            kv_state: alloc::vec![0.0; feature_dim * value_dim],
            k_state: alloc::vec![0.0; feature_dim],
            feature_dim,
            value_dim,
        }
    }

    /// Reset the recurrent state.
    pub fn reset(&mut self) {
        self.kv_state.fill(0.0);
        self.k_state.fill(0.0);
    }
}

// ============================================================================
// The Unified Theory
// ============================================================================

/// The fundamental insight, encoded as a type.
///
/// This marker type documents the equivalence:
/// - Spherical Harmonic projection ≡ Feature map φ
/// - SH coefficients ≡ Attention key/query features
/// - Irradiance integral ≡ Attention computation
/// - Light transport ≡ Value propagation
///
/// Both are instances of kernel methods with basis decomposition.
pub struct HarmonicAttentionIsGlobalIllumination;

impl HarmonicAttentionIsGlobalIllumination {
    /// The mathematical correspondence:
    ///
    /// | Graphics (SH)              | ML (Linear Attention)      |
    /// |----------------------------|----------------------------|
    /// | Light direction ω          | Key vector k               |
    /// | Surface normal n           | Query vector q             |
    /// | Radiance L(ω)              | Value vector v             |
    /// | SH basis Y_lm(ω)           | Feature map φ(k)           |
    /// | SH coefficients c_lm       | Key features φ(K)          |
    /// | Transfer function T(n)     | Query features φ(Q)        |
    /// | Irradiance E(n) = L·T      | Attention output φ(Q)·S    |
    /// | Environment map            | KV cache                   |
    ///
    /// Both compress O(n²) or O(∞) interactions into O(n) via basis decomposition.
    pub const CORRESPONDENCE: &'static str = "
        Linear Attention:     Output = φ(Q) · (φ(K)^T · V)
        SH Global Illumination: E(n) = Y(n) · (Y(ω)^T · L(ω))

        The feature map φ IS the spherical harmonic basis.
        The KV cache IS the environment map.
        Attention IS irradiance computation.
    ";
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use pixelflow_core::Lattice;

    /// A kernel at one point: compiled at a one-sample lattice, then collapsed.
    fn at(k: &Kernel, x: f32, y: f32) -> f32 {
        Lattice::point(x, y).bake(k).into_buffer()[0]
    }

    fn close(got: f32, want: f32) -> bool {
        (got - want).abs() <= 1e-3 + 1e-3 * want.abs()
    }

    #[test]
    fn elu_feature_positive() {
        // ELU(x) + 1 is positive everywhere: max(0, x) + exp(min(0, x)).
        let phi = EluFeature.apply(&Kernel::x());
        for x in [-4.0_f32, -1.0, 0.0, 1.0, 4.0] {
            let (got, want) = (at(&phi, x, 0.0), x.max(0.0) + x.min(0.0).exp());
            assert!(close(got, want), "elu({x}): got {got}, want {want}");
            assert!(got > 0.0, "elu({x}) = {got} is not positive");
        }
    }

    #[test]
    fn elu_feature_dimension() {
        let f = EluFeature;
        assert_eq!(f.dim(), 1);
    }

    #[test]
    fn elu_feature_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EluFeature>();
    }

    #[test]
    fn random_fourier_feature_dimension_correct() {
        let rff = RandomFourierFeature::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(rff.num_features, 6);
    }

    #[test]
    fn harmonic_attention_accumulate() {
        let mut attn: HarmonicAttention<9> = HarmonicAttention::new(3);
        let key_sh = Sh2 {
            coeffs: [0.282, 0.0, 0.489, 0.0, 0.0, 0.0, 0.315, 0.0, 0.0],
        };
        let value = [1.0, 0.5, 0.0];
        attn.accumulate(&key_sh, &value);
        let mut output = [0.0f32; 3];
        attn.query(&key_sh, &mut output);
        assert!(output[0] > 0.5);
    }

    #[test]
    fn harmonic_attention_reset() {
        let mut attn: HarmonicAttention<9> = HarmonicAttention::new(3);
        let key_sh = Sh2 {
            coeffs: [1.0, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        attn.accumulate(&key_sh, &[1.0, 1.0, 1.0]);
        attn.reset();
        for sh in &attn.accumulated {
            for &coeff in &sh.coeffs {
                assert!(coeff.abs() < 1e-10);
            }
        }
    }

    #[test]
    fn sh_feature_map_projects_direction() {
        // Along +z only the m=0 members survive, at the band constants
        // themselves (the l=2 one carries 3*nz^2 - 1 = 2). The direction's
        // third component is a value, not a coordinate: a lattice has two
        // axes, and this projection is a function of a direction it is
        // handed rather than of where it is sampled.
        let basis = ShFeatureMap::<9>::project(&Kernel::x(), &Kernel::y(), &Kernel::constant(1.0));
        assert_eq!(basis.len(), 9);
        let want = [
            SH_NORM[0][0],
            0.0,
            SH_NORM[1][0],
            0.0,
            0.0,
            0.0,
            SH_NORM[2][0] * 2.0,
            0.0,
            0.0,
        ];
        for (i, (k, w)) in basis.iter().zip(want.iter()).enumerate() {
            let got = at(k, 0.0, 0.0);
            assert!(close(got, *w), "basis[{i}]: got {got}, want {w}");
        }
    }

    #[test]
    fn linear_attention_new() {
        let attn = LinearAttention::new(EluFeature, 4, 3);
        assert_eq!(attn.feature_dim, 4);
        assert_eq!(attn.value_dim, 3);
    }

    #[test]
    fn correspondence_doc() {
        let correspondence = HarmonicAttentionIsGlobalIllumination::CORRESPONDENCE;
        assert!(correspondence.contains("Linear Attention"));
    }
}
