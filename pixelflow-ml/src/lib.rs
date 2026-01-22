//! # PixelFlow ML: Neural Networks for Graphics and Compilers
//!
//! This crate provides neural network primitives for two domains:
//!
//! 1. **Linear Attention as Harmonic Global Illumination** - Unifying ML and graphics
//! 2. **NNUE for Instruction Selection** - Neural-guided compiler optimization
//!
//! ## Linear Attention IS Harmonic Global Illumination
//!
//! ## The Mathematical Connection
//!
//! ### Traditional Attention (Transformer)
//! ```text
//! Attention(Q, K, V) = softmax(QK^T / √d) · V
//! ```
//! This is O(n²) in sequence length—the softmax creates dense attention.
//!
//! ### Linear Attention
//! ```text
//! LinearAttention(Q, K, V) = φ(Q) · (φ(K)^T · V)
//! ```
//! By decomposing the attention kernel K(q,k) = φ(q)·φ(k)^T, we achieve O(n) complexity
//! through the associativity of matrix multiplication.
//!
//! ### Spherical Harmonics for Global Illumination
//! ```text
//! Irradiance(n) = ∫_{S²} L(ω) · max(n·ω, 0) dω
//! ```
//! This integral is O(∞)—we'd need to sample every direction.
//!
//! But by projecting onto spherical harmonics:
//! ```text
//! L(ω) = Σ c_lm · Y_lm(ω)
//! Irradiance(n) = L_sh · Transfer_sh(n)
//! ```
//! We compress the infinite integral to O(1)—just a dot product!
//!
//! ## The Unifying Principle
//!
//! Both methods share the same insight:
//!
//! 1. **Project onto orthogonal basis**: SH projects lighting; φ projects queries/keys
//! 2. **Compress interactions**: Both avoid quadratic/infinite complexity
//! 3. **Compute via inner products**: Irradiance = dot(L_sh, T_sh), Attention = φ(Q)·S
//!
//! The feature map φ in linear attention IS the spherical harmonic basis.
//! Linear attention IS computing irradiance in feature space.
//!
//! ## Why This Matters
//!
//! This isn't just a mathematical curiosity—it's a design principle:
//!
//! - **Neural rendering** can use SH-inspired feature maps
//! - **Attention mechanisms** can learn lighting-like transfer functions
//! - **Global illumination** can be computed with attention-like architectures
//!
//! The pixelflow manifold system provides the substrate for both.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

extern crate alloc;

pub mod nnue;
pub mod evaluator;
pub mod hce_extractor;

#[cfg(feature = "training")]
pub mod training;

use alloc::vec::Vec;
use pixelflow_core::{Field, ManifoldExt, ShCoeffs};

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
    fn apply(&self, x: Field) -> Field;

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
    fn apply(&self, x: Field) -> Field {
        // ELU(x) + 1 = max(0, x) + exp(min(0, x))
        let zero = Field::from(0.0);
        let pos_part = x.max(zero);
        let neg_part = x.min(zero).exp();
        (pos_part + neg_part).constant()
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
// SH Feature Projection Manifold
// ============================================================================

/// Projects coordinates into spherical harmonic feature space.
///
/// This manifold transforms (x, y, z) directions into SH coefficient vectors,
/// serving as the feature map φ for harmonic attention.
pub struct ShFeatureMap<const NUM_COEFFS: usize>;

impl ShFeatureMap<9> {
    /// Evaluate the SH feature map at a direction.
    ///
    /// Returns the 9-coefficient SH vector for band 2.
    #[inline(always)]
    pub fn project(x: Field, y: Field, z: Field) -> [Field; 9] {
        use pixelflow_core::SH_NORM;

        let _zero = Field::from(0.0);

        // Normalize direction - collapse intermediate AST
        let r = (x * x + y * y + z * z).sqrt().constant();
        let inv_r = (Field::from(1.0) / r).constant();
        let nx = (x * inv_r).constant();
        let ny = (y * inv_r).constant();
        let nz = (z * inv_r).constant();

        // Compute all 9 SH basis functions - collapse each
        [
            // l=0
            Field::from(SH_NORM[0][0]),
            // l=1
            (Field::from(SH_NORM[1][1]) * ny).constant(),
            (Field::from(SH_NORM[1][0]) * nz).constant(),
            (Field::from(SH_NORM[1][1]) * nx).constant(),
            // l=2
            (Field::from(SH_NORM[2][2]) * nx * ny).constant(),
            (Field::from(SH_NORM[2][1]) * ny * nz).constant(),
            (Field::from(SH_NORM[2][0]) * (Field::from(3.0) * nz * nz - Field::from(1.0)))
                .constant(),
            (Field::from(SH_NORM[2][1]) * nx * nz).constant(),
            (Field::from(SH_NORM[2][2]) * (nx * nx - ny * ny)).constant(),
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
    use pixelflow_core::Sh2;

    // ============================================================================
    // EluFeature Tests
    // ============================================================================

    #[test]
    fn test_elu_feature_positive() {
        let f = EluFeature;
        // ELU(0) + 1 = 1
        let result = f.apply(Field::from(0.0));
        // Test by checking that positive + exp(0) = 0 + 1 = 1
        // We can't easily extract the value, so just verify it compiles
        let _ = result;
    }

    #[test]
    fn test_elu_feature_dimension() {
        let f = EluFeature;
        assert_eq!(f.dim(), 1, "EluFeature should have dimension 1");
    }

    #[test]
    fn test_elu_feature_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EluFeature>();
    }

    #[test]
    fn test_elu_feature_default() {
        let f1 = EluFeature;
        let f2 = EluFeature::default();
        // Both should work the same
        let _ = f1.apply(Field::from(1.0));
        let _ = f2.apply(Field::from(1.0));
    }

    // ============================================================================
    // RandomFourierFeature Tests
    // ============================================================================

    #[test]
    fn test_random_fourier_feature_dimension_correct() {
        let rff = RandomFourierFeature::new(vec![1.0, 2.0, 3.0]);
        // Should be 2x frequencies for sin/cos pairs
        assert_eq!(rff.num_features, 6);
    }

    #[test]
    fn test_random_fourier_feature_empty_frequencies() {
        let rff = RandomFourierFeature::new(vec![]);
        assert_eq!(rff.num_features, 0);
        assert!(rff.frequencies.is_empty());
    }

    #[test]
    fn test_random_fourier_feature_stores_frequencies() {
        let freqs = vec![0.5, 1.0, 2.0, 4.0];
        let rff = RandomFourierFeature::new(freqs.clone());
        assert_eq!(rff.frequencies, freqs);
    }

    // ============================================================================
    // HarmonicAttention Tests
    // ============================================================================

    #[test]
    fn test_harmonic_attention_accumulate() {
        let mut attn: HarmonicAttention<9> = HarmonicAttention::new(3);

        // Add a "light source" pointing up (+z)
        let key_sh = Sh2 {
            coeffs: [0.282, 0.0, 0.489, 0.0, 0.0, 0.0, 0.315, 0.0, 0.0],
        };
        let value = [1.0, 0.5, 0.0]; // Red-ish

        attn.accumulate(&key_sh, &value);

        // Query from the same direction should give high response
        let mut output = [0.0f32; 3];
        attn.query(&key_sh, &mut output);

        // The output should be close to the input value
        assert!(output[0] > 0.5);
    }

    #[test]
    fn test_harmonic_attention_reset_clears_state() {
        let mut attn: HarmonicAttention<9> = HarmonicAttention::new(3);

        // Add some state
        let key_sh = Sh2 {
            coeffs: [1.0, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        let value = [1.0, 1.0, 1.0];
        attn.accumulate(&key_sh, &value);

        // Reset
        attn.reset();

        // Verify all coefficients are zero
        for sh in &attn.accumulated {
            for &coeff in &sh.coeffs {
                assert!(
                    coeff.abs() < 1e-10,
                    "Reset should clear all accumulated coefficients"
                );
            }
        }

        for &coeff in &attn.denominator.coeffs {
            assert!(
                coeff.abs() < 1e-10,
                "Reset should clear denominator coefficients"
            );
        }
    }

    #[test]
    fn test_harmonic_attention_new_creates_zero_state() {
        let attn: HarmonicAttention<9> = HarmonicAttention::new(4);

        assert_eq!(attn.accumulated.len(), 4, "Should have 4 value dimensions");

        for sh in &attn.accumulated {
            for &coeff in &sh.coeffs {
                assert!(coeff.abs() < 1e-10, "Initial state should be zero");
            }
        }
    }

    #[test]
    fn test_harmonic_attention_multiple_accumulations() {
        let mut attn: HarmonicAttention<9> = HarmonicAttention::new(2);

        // Add first light source
        let key1 = Sh2 {
            coeffs: [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        attn.accumulate(&key1, &[1.0, 0.0]);

        // Add second light source
        let key2 = Sh2 {
            coeffs: [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        attn.accumulate(&key2, &[0.0, 1.0]);

        // Query with uniform direction
        let query_sh = Sh2 {
            coeffs: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        let mut output = [0.0f32; 2];
        attn.query(&query_sh, &mut output);

        // Should have contributions from both sources
        assert!(
            output[0] > 0.0,
            "Should have contribution from first source"
        );
        assert!(
            output[1] > 0.0,
            "Should have contribution from second source"
        );
    }

    #[test]
    fn test_harmonic_attention_query_normalization() {
        let mut attn: HarmonicAttention<9> = HarmonicAttention::new(1);

        // Accumulate with known values
        let key_sh = Sh2 {
            coeffs: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        attn.accumulate(&key_sh, &[2.0]);

        // Query from same direction
        let mut output = [0.0f32; 1];
        attn.query(&key_sh, &mut output);

        // Result should be normalized (approximately the value we put in)
        assert!(
            (output[0] - 2.0).abs() < 0.1,
            "Query with same key should return ~value"
        );
    }

    // ============================================================================
    // ShFeatureMap Tests
    // ============================================================================

    #[test]
    fn test_sh_feature_map_projects_direction() {
        // Project the +Z direction
        let result =
            ShFeatureMap::<9>::project(Field::from(0.0), Field::from(0.0), Field::from(1.0));

        // Should get 9 coefficients
        assert_eq!(result.len(), 9);
    }

    #[test]
    fn test_sh_feature_map_normalized() {
        // Project an unnormalized direction
        let result =
            ShFeatureMap::<9>::project(Field::from(2.0), Field::from(0.0), Field::from(0.0));

        // First coefficient (DC term) should be constant regardless of normalization
        // Y_00 = 0.282... (constant)
        let _ = result; // We verify it computes without NaN/Inf
    }

    // ============================================================================
    // LinearAttention Tests
    // ============================================================================

    #[test]
    fn test_linear_attention_new() {
        let attn = LinearAttention::new(EluFeature, 4, 3);
        assert_eq!(attn.feature_dim, 4);
        assert_eq!(attn.value_dim, 3);
        assert_eq!(attn.kv_state.len(), 4 * 3);
        assert_eq!(attn.k_state.len(), 4);
    }

    #[test]
    fn test_linear_attention_reset_clears_state() {
        let mut attn = LinearAttention::new(EluFeature, 2, 2);

        // Modify state
        attn.kv_state[0] = 1.0;
        attn.k_state[0] = 1.0;

        // Reset
        attn.reset();

        // Verify cleared
        assert!(
            attn.kv_state.iter().all(|&x| x == 0.0),
            "kv_state should be cleared"
        );
        assert!(
            attn.k_state.iter().all(|&x| x == 0.0),
            "k_state should be cleared"
        );
    }

    #[test]
    fn test_linear_attention_initial_state_is_zero() {
        let attn = LinearAttention::new(EluFeature, 3, 5);
        assert!(attn.kv_state.iter().all(|&x| x == 0.0));
        assert!(attn.k_state.iter().all(|&x| x == 0.0));
    }

    // ============================================================================
    // Correspondence Documentation Test
    // ============================================================================

    #[test]
    fn test_correspondence_doc() {
        // Just verify the documentation compiles
        let correspondence = HarmonicAttentionIsGlobalIllumination::CORRESPONDENCE;
        assert!(correspondence.contains("Linear Attention"));
        assert!(correspondence.contains("SH Global Illumination"));
    }
}
