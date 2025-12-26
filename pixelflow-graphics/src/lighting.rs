//! # Lighting Module
//!
//! Spherical harmonics-based lighting for real-time rendering.
//!
//! This module provides lighting primitives built on the kernel algebra
//! from `pixelflow-core`. All operations work through manifolds - no scalar
//! escape hatches.
//!
//! ## Key Types
//!
//! - [`DirectionalLight`]: Creates SH coefficients for a directional light
//! - [`AmbientLight`]: Creates SH coefficients for uniform ambient lighting
//! - [`CosineLobe`]: Lambertian BRDF transfer function in SH
//! - [`IrradianceField`]: Baked 2D grid of SH coefficients (manifold interface)
//!
//! ## Example
//!
//! ```ignore
//! // Create environment lighting
//! let sun = DirectionalLight::new((0.5, 0.8, 0.3), 2.0);
//! let sky = AmbientLight::new(0.3);
//! let env = sun.coeffs().add(&sky.coeffs());
//!
//! // Bake at low resolution, sample at full resolution via manifold
//! let field = IrradianceField::bake(|_x, _y| env.clone(), 64, 64);
//! let manifold = field.into_manifold((0.0, 1.0, 0.0));  // fixed normal
//!
//! // Use in render pipeline - no scalars!
//! let lit = manifold.eval_raw(x, y, z, w);
//! ```

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::f32::consts::PI;
use pixelflow_core::combinators::{Basis, Compressed, RotInv2, ShBasis, Texture};
use pixelflow_core::{Field, Manifold, SH_NORM};

extern crate alloc;

// Helper for sqrt (libm not in scope here)
fn sqrtf(x: f32) -> f32 {
    // Use core intrinsic via unsafe - this is safe for finite positive values
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let result: f32;
        core::arch::asm!(
            "vsqrtss {out}, {out}, {x}",
            x = in(xmm_reg) x,
            out = lateout(xmm_reg) result,
            options(pure, nomem, nostack)
        );
        result
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        // Fallback using f32::sqrt which is always available
        x.sqrt()
    }
}

// ============================================================================
// Directional Light
// ============================================================================

/// A directional light source represented in SH.
///
/// A directional light is a delta function on the sphere, which projects
/// onto SH as the evaluation of each basis function at that direction.
pub struct DirectionalLight {
    coeffs: Compressed<RotInv2>,
}

impl DirectionalLight {
    /// Create a directional light.
    ///
    /// Direction is normalized internally. Intensity scales all coefficients.
    pub fn new(dir: (f32, f32, f32), intensity: f32) -> Self {
        let basis = ShBasis::<9>::eval_at_scalar(dir.0, dir.1, dir.2);
        let mut coeffs = basis.coeffs;
        for c in coeffs.iter_mut() {
            *c *= intensity;
        }
        Self {
            coeffs: Compressed::from_coeffs(ShBasis::new(coeffs)),
        }
    }

    /// Get the SH coefficients.
    pub fn coeffs(&self) -> &Compressed<RotInv2> {
        &self.coeffs
    }

    /// Consume and return the coefficients.
    pub fn into_coeffs(self) -> Compressed<RotInv2> {
        self.coeffs
    }
}

// ============================================================================
// Ambient Light
// ============================================================================

/// Uniform ambient lighting represented in SH.
///
/// Ambient light only has a DC (l=0) component.
pub struct AmbientLight {
    coeffs: Compressed<RotInv2>,
}

impl AmbientLight {
    /// Create ambient light with given intensity.
    pub fn new(intensity: f32) -> Self {
        // DC term for uniform sphere = intensity * √(4π)
        let dc = intensity * sqrtf(4.0 * PI);

        Self {
            coeffs: Compressed::from_coeffs(ShBasis::new([
                dc * SH_NORM[0][0],
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ])),
        }
    }

    /// Get the SH coefficients.
    pub fn coeffs(&self) -> &Compressed<RotInv2> {
        &self.coeffs
    }

    /// Consume and return the coefficients.
    pub fn into_coeffs(self) -> Compressed<RotInv2> {
        self.coeffs
    }
}

// ============================================================================
// Cosine Lobe (Lambertian BRDF)
// ============================================================================

/// Clamped cosine lobe in SH - the Lambertian BRDF transfer function.
///
/// The cosine lobe `max(0, n·ω)` has known SH projection:
/// - Only l=0 and l=1 are non-zero
/// - Coefficients: [π, 2π/3, 2π/3, 2π/3, 0, 0, 0, 0, 0] when aligned with +z
pub struct CosineLobe {
    coeffs: Compressed<RotInv2>,
}

impl CosineLobe {
    /// Create cosine lobe for a given surface normal.
    pub fn new(normal: (f32, f32, f32)) -> Self {
        // Normalize
        let len = sqrtf(normal.0 * normal.0 + normal.1 * normal.1 + normal.2 * normal.2);
        let (nx, ny, nz) = if len > 1e-6 {
            (normal.0 / len, normal.1 / len, normal.2 / len)
        } else {
            (0.0, 0.0, 1.0)
        };

        // Analytic coefficients for cosine lobe
        let a0 = PI; // l=0: π * Y_0^0
        let a1 = 2.0 * PI / 3.0; // l=1: 2π/3 * Y_1^m

        let coeffs = ShBasis::new([
            a0 * 0.282_094_79,                    // l=0
            a1 * 0.488_602_51 * ny,               // l=1, m=-1
            a1 * 0.488_602_51 * nz,               // l=1, m=0
            a1 * 0.488_602_51 * nx,               // l=1, m=1
            0.0, 0.0, 0.0, 0.0, 0.0,              // l=2 (zero for Lambertian)
        ]);

        Self {
            coeffs: Compressed::from_coeffs(coeffs),
        }
    }

    /// Get the SH coefficients.
    pub fn coeffs(&self) -> &Compressed<RotInv2> {
        &self.coeffs
    }

    /// Consume and return the coefficients.
    pub fn into_coeffs(self) -> Compressed<RotInv2> {
        self.coeffs
    }
}

// ============================================================================
// Irradiance Field
// ============================================================================

/// A 2D grid of SH coefficients for spatially-varying lighting.
///
/// Bakes expensive lighting computation at low resolution, then samples
/// via SIMD texture gather at full resolution.
///
/// **No scalar interface** - use `into_manifold()` or `coefficient_textures()`
/// for manifold-based sampling.
#[derive(Clone)]
pub struct IrradianceField {
    /// 9 textures, one per SH coefficient.
    coeffs: [Texture; 9],
    /// Grid width.
    width: usize,
    /// Grid height.
    height: usize,
}

impl IrradianceField {
    /// Bake an irradiance field by evaluating a function at each grid point.
    ///
    /// The function receives grid coordinates (cell centers) and returns SH coefficients.
    pub fn bake<F>(mut f: F, width: usize, height: usize) -> Self
    where
        F: FnMut(f32, f32) -> Compressed<RotInv2>,
    {
        let mut buffers: [Vec<f32>; 9] = core::array::from_fn(|_| {
            Vec::with_capacity(width * height)
        });

        for y in 0..height {
            for x in 0..width {
                let sh = f(x as f32 + 0.5, y as f32 + 0.5);
                for (i, buf) in buffers.iter_mut().enumerate() {
                    buf.push(sh.coeffs.coeffs[i]);
                }
            }
        }

        let coeffs = core::array::from_fn(|i| {
            Texture::new(core::mem::take(&mut buffers[i]), width, height)
        });

        Self { coeffs, width, height }
    }

    /// Get the grid dimensions.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Get read access to the coefficient textures.
    ///
    /// These can be sampled directly as manifolds for custom pipelines.
    pub fn coefficient_textures(&self) -> &[Texture; 9] {
        &self.coeffs
    }

    /// Convert to an irradiance manifold with a fixed normal.
    ///
    /// This is the standard way to use an IrradianceField - as a manifold
    /// that maps (x, y) coordinates to irradiance values.
    pub fn into_manifold(self, normal: (f32, f32, f32)) -> IrradianceManifold {
        IrradianceManifold::new(self, normal)
    }
}

// ============================================================================
// Irradiance Manifold
// ============================================================================

/// Manifold wrapper for IrradianceField.
///
/// Maps (x, y) grid coordinates to irradiance values for a fixed normal.
/// Uses only l=0 and l=1 coefficients (4 texture samples) for Lambertian.
#[derive(Clone)]
pub struct IrradianceManifold {
    field: Arc<IrradianceField>,
    /// Precomputed cosine lobe transfer coefficients for the fixed normal.
    transfer: [f32; 4],
}

impl IrradianceManifold {
    /// Create from irradiance field with a fixed normal.
    pub fn new(field: IrradianceField, normal: (f32, f32, f32)) -> Self {
        // Normalize
        let len = sqrtf(normal.0 * normal.0 + normal.1 * normal.1 + normal.2 * normal.2);
        let (nx, ny, nz) = if len > 1e-6 {
            (normal.0 / len, normal.1 / len, normal.2 / len)
        } else {
            (0.0, 0.0, 1.0)
        };

        // Precompute cosine lobe transfer (only l=0 and l=1 for Lambertian)
        let a0 = PI * 0.282_094_79;
        let a1 = 2.0 * PI / 3.0 * 0.488_602_51;

        Self {
            field: Arc::new(field),
            transfer: [a0, a1 * ny, a1 * nz, a1 * nx],
        }
    }

    /// Create with up-facing normal (0, 0, 1).
    pub fn facing_up(field: IrradianceField) -> Self {
        Self::new(field, (0.0, 0.0, 1.0))
    }
}

impl Manifold<Field> for IrradianceManifold {
    type Output = Field;

    #[inline(always)]
    fn eval_raw(&self, x: Field, y: Field, _z: Field, _w: Field) -> Field {
        let zero = Field::from(0.0);

        // Sample l=0 and l=1 coefficient textures (l=2 unused for Lambertian)
        let c0 = self.field.coeffs[0].eval_raw(x, y, zero, zero);
        let c1 = self.field.coeffs[1].eval_raw(x, y, zero, zero);
        let c2 = self.field.coeffs[2].eval_raw(x, y, zero, zero);
        let c3 = self.field.coeffs[3].eval_raw(x, y, zero, zero);

        // Dot product with precomputed transfer
        let [t0, t1, t2, t3] = self.transfer;
        c0 * Field::from(t0) + c1 * Field::from(t1) + c2 * Field::from(t2) + c3 * Field::from(t3)
    }
}

// ============================================================================
// Convenience Functions (return Compressed<RotInv2>)
// ============================================================================

/// Compute irradiance: environment SH dotted with cosine lobe.
///
/// This is the fundamental lighting operation:
/// `E(n) = L_sh · T_sh(n)`
#[inline]
pub fn irradiance(environment: &Compressed<RotInv2>, normal: (f32, f32, f32)) -> f32 {
    let transfer = CosineLobe::new(normal);
    environment.mul_coeffs(transfer.coeffs()).dc_term()
}

/// Shorthand for DirectionalLight::new(...).into_coeffs()
#[inline]
pub fn directional_light_sh(dir: (f32, f32, f32), intensity: f32) -> Compressed<RotInv2> {
    DirectionalLight::new(dir, intensity).into_coeffs()
}

/// Shorthand for AmbientLight::new(...).into_coeffs()
#[inline]
pub fn ambient_light_sh(intensity: f32) -> Compressed<RotInv2> {
    AmbientLight::new(intensity).into_coeffs()
}

/// Shorthand for CosineLobe::new(...).into_coeffs()
#[inline]
pub fn cosine_lobe_sh(normal: (f32, f32, f32)) -> Compressed<RotInv2> {
    CosineLobe::new(normal).into_coeffs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_directional_light() {
        let light = DirectionalLight::new((0.0, 0.0, 1.0), 1.0);
        let coeffs = light.coeffs();

        // Should have non-zero coefficients
        assert!(coeffs.dc_term().abs() > 0.0);
    }

    #[test]
    fn test_ambient_light() {
        let ambient = AmbientLight::new(1.0);
        let coeffs = ambient.coeffs();

        // Should be roughly equal in all directions
        let val_z = coeffs.sample((0.0, 0.0, 1.0));
        let val_x = coeffs.sample((1.0, 0.0, 0.0));
        assert!((val_z - val_x).abs() < 0.1);
    }

    #[test]
    fn test_cosine_lobe() {
        let lobe = CosineLobe::new((0.0, 0.0, 1.0));
        let coeffs = lobe.coeffs();

        // Should be maximal in lobe direction
        let up = coeffs.sample((0.0, 0.0, 1.0));
        let side = coeffs.sample((1.0, 0.0, 0.0));
        assert!(up > side);
    }

    #[test]
    fn test_irradiance_computation() {
        let env = directional_light_sh((0.0, 0.0, 1.0), 1.0);

        // Surface facing up should receive more light than facing down
        let e_up = irradiance(&env, (0.0, 0.0, 1.0));
        let e_down = irradiance(&env, (0.0, 0.0, -1.0));

        assert!(e_up > e_down);
    }

    #[test]
    fn test_irradiance_field_bake() {
        let env = ambient_light_sh(1.0);
        let field = IrradianceField::bake(|_x, _y| env.clone(), 4, 4);

        assert_eq!(field.dimensions(), (4, 4));
    }

    #[test]
    fn test_irradiance_manifold() {
        let env = ambient_light_sh(1.0);
        let field = IrradianceField::bake(|_x, _y| env.clone(), 4, 4);
        let manifold = field.into_manifold((0.0, 0.0, 1.0));

        // Evaluate and check positivity using Field comparison methods
        let result = manifold.eval_raw(
            Field::from(2.0),
            Field::from(2.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        // All lanes should be positive for uniform ambient lighting
        assert!(result.gt(Field::from(0.0)).all());
    }

    #[test]
    fn test_irradiance_field_spatial_variation() {
        // Bake with position-dependent lighting (intensity increases with x)
        let field = IrradianceField::bake(|x, _y| {
            ambient_light_sh(x / 4.0)
        }, 4, 4);

        let manifold = field.into_manifold((0.0, 0.0, 1.0));

        let left = manifold.eval_raw(
            Field::from(0.5), Field::from(2.0), Field::from(0.0), Field::from(0.0)
        );
        let right = manifold.eval_raw(
            Field::from(3.5), Field::from(2.0), Field::from(0.0), Field::from(0.0)
        );

        // Right side should be brighter than left (higher x = higher intensity)
        assert!(right.gt(left).all());
    }
}
