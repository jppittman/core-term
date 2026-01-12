//! Animated Chrome Sphere - windowed animation using the runtime.
//!
//! Demonstrates the **pull-based rendering model**:
//! - Engine sends `RequestFrame` events when ready for a new frame
//! - App responds with a manifold computed at the requested timestamp
//! - No busy loops, no sleeps - vsync drives the cadence
//!
//! Also demonstrates compositional animation:
//! - `TimeShift<M>` wraps any manifold and translates W (time coordinate)
//! - Animation is injected by wrapping existing geometry nodes

use actor_scheduler::Message;
use pixelflow_core::combinators::At;
use pixelflow_core::jet::Jet3;
use pixelflow_core::ops::Sin;
use pixelflow_core::{Discrete, Field, Manifold, W};
use pixelflow_graphics::scene3d::{
    ColorChecker, ColorReflect, ColorScreenToDir, ColorSky, ColorSurface, PlaneGeometry,
};
use pixelflow_runtime::api::private::EngineData;
use pixelflow_runtime::api::public::{AppData, EngineEvent, EngineEventData};
use pixelflow_runtime::platform::ColorCube;
use pixelflow_runtime::{Application, EngineConfig, EngineTroupe, RuntimeError, WindowConfig};
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
        let w_shifted = w + Field::from(self.t);
        At {
            inner: &self.inner,
            x,
            y,
            z,
            w: w_shifted,
        }
        .eval()
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
        let cx = Jet3::constant(Field::from(self.base.0))
            + offset * Jet3::constant(Field::from(self.amplitude));
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
        At {
            inner: &self.inner,
            x: sx,
            y: sy,
            z,
            w,
        }
        .eval()
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
        material: ColorChecker::<ColorCube>::default(),
        background: ColorSky::<ColorCube>::default(),
    };

    // Animated sphere using the algebra: center.x = sin(w * freq) * amp
    let sphere = OscillatingSphere {
        base: (0.0, 0.0, 4.0),
        amplitude: 2.0, // Oscillate ±2 units in X
        frequency: 1.0, // 1 rad/s
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

// ============================================================================
// APPLICATION - Pull-based rendering
// ============================================================================

/// The animated sphere application.
///
/// Implements the pull-based rendering model:
/// - Receives `RequestFrame` events from the engine
/// - Responds with a manifold at the requested timestamp
struct AnimatedSphereApp<S> {
    /// The base scene (time-independent structure)
    scene: S,
    /// Animation start time
    start: Instant,
    /// Handle to send frames back to the engine
    engine_handle: pixelflow_runtime::api::private::EngineActorHandle,
}

impl<S> Application for AnimatedSphereApp<S>
where
    S: Manifold<Output = Discrete> + Clone + Send + Sync + 'static,
{
    fn send(&self, event: EngineEvent) -> Result<(), RuntimeError> {
        match event {
            // Engine is ready for a frame - this is the pull!
            EngineEvent::Data(EngineEventData::RequestFrame { timestamp, .. }) => {
                log::debug!("App received RequestFrame");

                // Compute elapsed time from animation start
                let t = timestamp.duration_since(self.start).as_secs_f32();

                // Build the scene at this moment in time
                let timed_scene = TimeShift {
                    inner: self.scene.clone(),
                    t,
                };

                let arc: Arc<dyn Manifold<Output = Discrete> + Send + Sync> = Arc::new(timed_scene);

                // Send the frame back to the engine
                log::debug!("App sending RenderSurface");
                self.engine_handle
                    .send(Message::Data(EngineData::FromApp(AppData::RenderSurface(
                        arc,
                    ))))
                    .map_err(|e| RuntimeError::EventSendError(e.to_string()))?;
            }
            EngineEvent::Control(ctrl) => {
                log::debug!("App received Control event: {:?}", ctrl);
            }
            _ => {
                log::debug!("App received other event");
            }
        }
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("Animated Chrome Sphere (Pull-based)");
    println!("====================================");
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
    let unregistered_handle = troupe.engine_handle();

    // Build the scene once - it's time-independent (W handles animation)
    let scene = build_scene();
    let start = Instant::now();

    // Get the raw engine handle for sending frames back
    // This must be obtained before registration (app needs it to respond to RequestFrame)
    let engine_handle_for_app = troupe.raw_engine_handle();

    // Create the pull-based app
    let app = AnimatedSphereApp {
        scene,
        start,
        engine_handle: engine_handle_for_app,
    };

    // Register app and create window
    use pixelflow_runtime::WindowDescriptor;
    let window = WindowDescriptor {
        width: WIDTH,
        height: HEIGHT,
        title: "Animated Sphere".into(),
        resizable: false,
    };
    let _engine_handle = unregistered_handle.register(Arc::new(app), window)?;

    println!("Running... (close window to exit)");
    troupe.play().map_err(|e| anyhow::anyhow!("{}", e))?;

    println!("Done!");
    Ok(())
}
