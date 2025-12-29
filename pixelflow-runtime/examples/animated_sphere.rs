//! Animated Chrome Sphere - windowed animation using the runtime.
//!
//! Demonstrates compositional animation:
//! - `Oscillate<M>` wraps any Jet3 geometry and makes it oscillate based on W (time)
//! - `TimeShift` advances W before evaluation (sets current time)
//! - Animation is injected by wrapping existing geometry nodes

use actor_scheduler::Message;
use pixelflow_core::ops::Sin;
use pixelflow_core::jet::Jet3;
use pixelflow_core::{Discrete, Field, Manifold, W};
use pixelflow_graphics::scene3d::{
    ColorChecker, ColorReflect, ColorScreenToDir, ColorSky, ColorSurface, PlaneGeometry,
};
use pixelflow_runtime::api::private::EngineData;
use pixelflow_runtime::api::public::AppData;
use pixelflow_runtime::{EngineConfig, EngineTroupe, WindowConfig};
use std::sync::Arc;
use std::time::Instant;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

// ============================================================================
// COMPOSITIONAL ANIMATION PRIMITIVES
// ============================================================================

/// Time shift - translates W by offset (sets current time).
#[derive(Clone)]
struct TimeShift<M> {
    inner: M,
    t: f32,
}

impl<M: Manifold<Output = Discrete> + Send + Sync> Manifold for TimeShift<M> {
    type Output = Discrete;

    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        self.inner.eval_raw(x, y, z, w + Field::from(self.t))
    }
}

/// OscillatingSphere - a sphere whose center oscillates based on W (time).
///
/// This is compositional: the motion is expressed as a manifold expression
/// using Sin(W * frequency) * amplitude. The algebra handles the rest.
#[derive(Clone, Copy)]
struct OscillatingSphere {
    /// Base center position
    base: (f32, f32, f32),
    /// Oscillation amplitude in X
    amplitude: f32,
    /// Oscillation frequency (radians per second)
    frequency: f32,
    /// Sphere radius
    radius: f32,
}

impl Manifold<Jet3> for OscillatingSphere {
    type Output = Jet3;

    #[inline(always)]
    fn eval_raw(&self, rx: Jet3, ry: Jet3, rz: Jet3, w: Jet3) -> Jet3 {
        // Build the oscillation offset using the ALGEBRA (compute graph)
        // offset = sin(w * frequency) * amplitude
        //
        // We use Sin(W * freq) as a manifold, then evaluate it with our inputs.
        // This is the compositional way - no direct .sin() calls.
        use pixelflow_core::ops::Mul;

        let oscillation = Sin(Mul(W, self.frequency));
        let offset: Jet3 = oscillation.eval_raw(rx, ry, rz, w);

        // Center = base + offset * amplitude_vector
        let cx = Jet3::constant(Field::from(self.base.0)) + offset * Jet3::constant(Field::from(self.amplitude));
        let cy = Jet3::constant(Field::from(self.base.1));
        let cz = Jet3::constant(Field::from(self.base.2));

        // Ray-sphere intersection: |t*D - C|² = r²
        // For normalized ray D: t² - 2t(D·C) + |C|² - r² = 0
        // Solution: t = (D·C) - sqrt((D·C)² - (|C|² - r²))

        let d_dot_c = rx * cx + ry * cy + rz * cz;
        let c_sq = cx * cx + cy * cy + cz * cz;
        let r_sq = Jet3::constant(Field::from(self.radius * self.radius));
        let discriminant = d_dot_c * d_dot_c - (c_sq - r_sq);

        // Smooth epsilon for grazing angles
        let epsilon_sq = Jet3::constant(Field::from(0.0001));
        let safe_discriminant = discriminant + epsilon_sq;

        d_dot_c - safe_discriminant.sqrt()
    }
}

/// Screen coordinate remapper.
#[derive(Clone)]
struct ScreenRemap<M> {
    inner: M,
    width: f32,
    height: f32,
}

impl<M: Manifold<Output = Discrete> + Send + Sync> Manifold for ScreenRemap<M> {
    type Output = Discrete;

    fn eval_raw(&self, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        let scale = 2.0 / self.height;
        let sx = (x - Field::from(self.width * 0.5)) * Field::from(scale);
        let sy = (Field::from(self.height * 0.5) - y) * Field::from(scale);
        self.inner.eval_raw(sx, sy, z, w)
    }
}

// ============================================================================
// SCENE CONSTRUCTION - Compositional Animation
// ============================================================================

/// Build scene with animated sphere.
///
/// The sphere's center is computed using Sin(W * frequency) - pure algebra.
/// TimeShift sets the current time (W), and the graph does the rest.
fn build_scene() -> impl Manifold<Output = Discrete> + Clone + Sync + Send {
    // Background: floor with checkerboard
    let world = ColorSurface {
        geometry: PlaneGeometry { height: -1.0 },
        material: ColorChecker,
        background: ColorSky,
    };

    // Animated sphere using the algebra: center.x = sin(w * freq) * amp
    let sphere = OscillatingSphere {
        base: (0.0, 0.0, 4.0),
        amplitude: 2.0,     // Oscillate ±2 units in X
        frequency: 1.0,     // 1 rad/s
        radius: 1.0,
    };

    let scene = ColorSurface {
        geometry: sphere,
        material: ColorReflect { inner: world },
        background: world,
    };

    ScreenRemap {
        inner: ColorScreenToDir { inner: scene },
        width: WIDTH as f32,
        height: HEIGHT as f32,
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("Animated Chrome Sphere");
    println!("======================");
    println!("Resolution: {}x{}", WIDTH, HEIGHT);
    println!();

    let config = EngineConfig {
        window: WindowConfig {
            title: "Animated Sphere".to_string(),
            width: WIDTH,
            height: HEIGHT,
        },
        ..Default::default()
    };

    let troupe = EngineTroupe::with_config(config)?;
    let engine_handle = troupe.engine_handle();
    let scene = build_scene();
    let start = Instant::now();

    // Spawn thread to send animated frames
    std::thread::spawn(move || {
        loop {
            let t = start.elapsed().as_secs_f32();

            // Wrap scene with current time (translate W)
            let timed_scene = TimeShift {
                inner: scene.clone(),
                t,
            };

            let arc: Arc<dyn Manifold<Output = Discrete> + Send + Sync> = Arc::new(timed_scene);

            if engine_handle
                .send(Message::Data(EngineData::FromApp(AppData::RenderSurface(arc))))
                .is_err()
            {
                break; // Engine shut down
            }

            std::thread::sleep(std::time::Duration::from_millis(16)); // ~60fps
        }
    });

    println!("Running... (close window to exit)");
    troupe.play().map_err(|e| anyhow::anyhow!("{}", e))?;

    println!("Done!");
    Ok(())
}
