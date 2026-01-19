//! Animated Chrome Sphere - windowed animation using the runtime.
//!
//! Demonstrates the **pull-based rendering model**:
//! - Engine sends `RequestFrame` events when ready for a new frame
//! - App responds with a manifold computed at the requested timestamp
//! - No busy loops, no sleeps - vsync drives the cadence
//!
//! Animation approach:
//! - Each frame, a new scene is built with the sphere at the animated position
//! - The position is computed using sin(t * freq) * amplitude at the app level

use actor_scheduler::Message;
use pixelflow_core::combinators::At;
use pixelflow_core::jet::Jet3;
use pixelflow_core::{Discrete, Field, Manifold, ManifoldCompat};

type Field4 = (Field, Field, Field, Field);
type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);
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
// GEOMETRY PRIMITIVES
// ============================================================================

/// SphereAt - a sphere at a given center position.
///
/// Animation is handled at scene construction time by computing the offset
/// and baking it into the center position.
#[derive(Clone, Copy)]
struct SphereAt {
    /// Center position
    center: (f32, f32, f32),
    /// Sphere radius
    radius: f32,
}

impl Manifold<Jet3_4> for SphereAt {
    type Output = Jet3;

    #[inline(always)]
    fn eval(&self, p: Jet3_4) -> Jet3 {
        let (rx, ry, rz, _w) = p;
        let cx = Jet3::constant(Field::from(self.center.0));
        let cy = Jet3::constant(Field::from(self.center.1));
        let cz = Jet3::constant(Field::from(self.center.2));

        // Ray-sphere intersection: |t*D - C|² = r²
        // For normalized ray D: t² - 2t(D·C) + |C|² - r² = 0
        // Solution: t = (D·C) - sqrt((D·C)² - (|C|² - r²))

        let d_dot_c = rx * cx + ry * cy + rz * cz;
        let c_sq = cx * cx + cy * cy + cz * cz;
        let r_sq = Jet3::constant(Field::from(self.radius * self.radius));
        let discriminant: Jet3 = d_dot_c * d_dot_c - (c_sq - r_sq);

        // Smooth epsilon for grazing angles
        let epsilon_sq = Jet3::constant(Field::from(0.0001));
        let safe_discriminant: Jet3 = discriminant + epsilon_sq;

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

impl<M: ManifoldCompat<Field, Output = Discrete> + Send + Sync> Manifold<Field4> for ScreenRemap<M> {
    type Output = Discrete;

    fn eval(&self, p: Field4) -> Discrete {
        let (x, y, z, w) = p;
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
        .collapse()
    }
}

// ============================================================================
// SCENE CONSTRUCTION - Compositional Animation
// ============================================================================

/// Animation parameters for the oscillating sphere.
const BASE_CENTER: (f32, f32, f32) = (0.0, 0.0, 4.0);
const AMPLITUDE: f32 = 2.0;  // Oscillate ±2 units in X
const FREQUENCY: f32 = 1.0;  // 1 rad/s
const RADIUS: f32 = 1.0;

/// Build scene with sphere at the given animated position.
///
/// The animation offset is precomputed at the application level using
/// sin(t * frequency) * amplitude, then baked into the sphere's center.
fn build_scene_at_time(t: f32) -> impl Manifold<Output = Discrete> + Clone + Sync + Send {
    // Compute the animated X offset
    let x_offset = (t * FREQUENCY).sin() * AMPLITUDE;
    let center = (BASE_CENTER.0 + x_offset, BASE_CENTER.1, BASE_CENTER.2);

    // Background: floor with checkerboard
    let world = ColorSurface {
        geometry: PlaneGeometry { height: -1.0 },
        material: ColorChecker::<ColorCube>::default(),
        background: ColorSky::<ColorCube>::default(),
    };

    // Sphere at the computed animated position
    let sphere = SphereAt { center, radius: RADIUS };

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
struct AnimatedSphereApp {
    /// Animation start time
    start: Instant,
    /// Handle to send frames back to the engine
    engine_handle: pixelflow_runtime::api::private::EngineActorHandle,
}

impl Application for AnimatedSphereApp {
    fn send(&self, event: EngineEvent) -> Result<(), RuntimeError> {
        match event {
            // Engine is ready for a frame - this is the pull!
            EngineEvent::Data(EngineEventData::RequestFrame { timestamp, .. }) => {
                log::debug!("App received RequestFrame");

                // Compute elapsed time from animation start
                let t = timestamp.duration_since(self.start).as_secs_f32();

                // Build the scene at this moment in time with the sphere at the animated position
                let scene = build_scene_at_time(t);

                let arc: Arc<dyn Manifold<Output = Discrete> + Send + Sync> = Arc::new(scene);

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
    let start = Instant::now();

    // Get the raw engine handle for sending frames back
    // This must be obtained before registration (app needs it to respond to RequestFrame)
    let engine_handle_for_app = troupe.raw_engine_handle();

    // Create the pull-based app (scene is built per-frame with animation)
    let app = AnimatedSphereApp {
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
