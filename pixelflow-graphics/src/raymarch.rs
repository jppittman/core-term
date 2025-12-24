//! # Jet2-Based Sphere Tracing
//!
//! Pull-based raymarching using automatic differentiation for free normals.
//!
//! ## Key Concepts
//!
//! - **No camera movement**: The world transforms, not the camera
//! - **Jet2 for normals**: Implicit surfaces with automatic gradient (∂f/∂x, ∂f/∂y)
//! - **Manifold composition**: Write math, get SIMD - the types are the AST
//!
//! ## Architecture
//!
//! ```text
//! // The pixelflow way: compose manifolds
//! let raymarch = Raymarch { scene: sphere_surface, config };
//! execute(&raymarch, frame.as_slice_mut(), shape);
//! //       ^^^^^^^^
//! //       This evaluates 4 pixels at once via Field SIMD operations!
//! ```

use pixelflow_core::{Discrete, Field, Jet2, Manifold, Numeric};

/// Configuration for raymarching.
#[derive(Clone, Copy, Debug)]
pub struct MarchConfig {
    pub max_steps: u32,
    pub step_size: f32,
    pub max_distance: f32,
    pub camera_z: f32,
}

impl Default for MarchConfig {
    fn default() -> Self {
        Self {
            max_steps: 100,
            step_size: 0.05,
            max_distance: 20.0,
            camera_z: -5.0,
        }
    }
}

/// Raymarch manifold: takes screen coords, returns colored pixels.
///
/// The scene manifold should return an implicit surface using Jet2:
/// - f(x,y,z) = 0 on the surface
/// - Jet2 provides free x,y gradients via automatic differentiation
///
/// ## Pull-based rendering
///
/// The camera is fixed at (0, 0, camera_z) looking down +Z.
/// To "move" the camera, transform the scene manifold instead.
pub struct Raymarch<Scene> {
    pub scene: Scene,
    pub config: MarchConfig,
}

impl<Scene> Manifold for Raymarch<Scene>
where
    Scene: Manifold<Jet2, Output = Jet2>,
{
    type Output = Discrete;

    fn eval_raw(&self, x: Field, y: Field, _z: Field, _w: Field) -> Discrete {
        // Fixed orthographic ray setup
        let origin_z = Field::from(self.config.camera_z);
        let step_size = Field::from(self.config.step_size);
        let max_dist = Field::from(self.config.max_distance);

        let mut t = Field::from(0.0);
        let mut hit = Field::from(0.0);
        let mut normal_x = Field::from(0.0);
        let mut normal_y = Field::from(0.0);
        let mut normal_z = Field::from(0.0);

        let mut last_val = Field::from(0.0);
        let mut first_step = true;

        // March along ray looking for sign change (surface crossing)
        for _step in 0..self.config.max_steps {
            let pz = origin_z + t;

            // Evaluate implicit surface with Jet2
            let jet_x = Jet2::x(x);
            let jet_y = Jet2::y(y);
            let jet_z = Jet2::constant(pz);
            let jet_w = Jet2::constant(Field::from(0.0));

            let surface = self.scene.eval_raw(jet_x, jet_y, jet_z, jet_w);
            let val = surface.val;

            let not_yet_hit = hit.lt(Field::from(0.5));

            // Check for sign change (surface crossing) - skip first step
            let hit_mask = if first_step {
                first_step = false;
                Field::from(0.0)  // No hit on first step
            } else {
                // Sign change detected when product is negative
                let product = last_val * val;
                let sign_changed = product.lt(Field::from(0.0));
                sign_changed & not_yet_hit
            };

            if hit_mask.any() {
                // Surface normal from Jet2 gradient
                let grad_x = surface.dx;
                let grad_y = surface.dy;

                // Finite difference for dz
                let delta_z = Field::from(0.01);
                let jet_z_plus = Jet2::constant(pz + delta_z);
                let surface_plus = self.scene.eval_raw(jet_x, jet_y, jet_z_plus, jet_w);
                let grad_z = (surface_plus.val - val) / delta_z;


                // Normalize gradient
                let grad_len = (grad_x * grad_x + grad_y * grad_y + grad_z * grad_z).sqrt();
                let inv_len = Field::from(1.0) / (grad_len + Field::from(1e-6));

                hit = Field::select(hit_mask, Field::from(1.0), hit);
                normal_x = Field::select(hit_mask, grad_x * inv_len, normal_x);
                normal_y = Field::select(hit_mask, grad_y * inv_len, normal_y);
                normal_z = Field::select(hit_mask, grad_z * inv_len, normal_z);
            }

            last_val = val;

            // March forward (only non-hit rays)
            // Use select to conditionally advance based on hit status
            t = Field::select(not_yet_hit, t + step_size, t);

            if t.ge(max_dist).all() {
                break;
            }
        }

        // Shade: normal to color
        let r = (normal_x + Field::from(1.0)) * Field::from(0.5);
        let g = (normal_y + Field::from(1.0)) * Field::from(0.5);
        let b = (normal_z + Field::from(1.0)) * Field::from(0.5);

        // Background color
        let bg_r = Field::from(20.0 / 255.0);
        let bg_g = Field::from(20.0 / 255.0);
        let bg_b = Field::from(40.0 / 255.0);

        // Blend
        let final_r = r * hit + bg_r * (Field::from(1.0) - hit);
        let final_g = g * hit + bg_g * (Field::from(1.0) - hit);
        let final_b = b * hit + bg_b * (Field::from(1.0) - hit);

        Discrete::pack(final_r, final_g, final_b, Field::from(1.0))
    }
}
