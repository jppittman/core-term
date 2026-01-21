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
//!
//! Resize handling:
//! - App receives Resized events and updates stored dimensions
//! - Scene is rebuilt with new dimensions on next frame

use actor_scheduler::Message;
use pixelflow_core::jet::Jet3;
use pixelflow_core::{Discrete, Field, Manifold};
use pixelflow_macros::kernel;

type Jet3_4 = (Jet3, Jet3, Jet3, Jet3);
use pixelflow_graphics::scene3d::{
    ColorChecker, ColorReflect, ColorScreenToDir, ColorSky, ColorSurface, plane,
};
use pixelflow_runtime::api::private::EngineData;
use pixelflow_runtime::api::public::{AppData, EngineEvent, EngineEventControl, EngineEventData};
use pixelflow_runtime::platform::ColorCube;
use pixelflow_runtime::{Application, EngineConfig, EngineTroupe, RuntimeError, WindowConfig};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

// ============================================================================
// GEOMETRY PRIMITIVES (using kernel! macro)
// ============================================================================

/// Create a sphere geometry kernel.
///
/// The kernel! macro generates a struct that captures (cx, cy, cz, r, eps) and
/// implements Manifold<Jet3_4> with Output = Jet3 (inferred from `-> Jet3`).
///
/// This replaces the manual `SphereAt` struct with a declarative expression.
///
/// Note: We make epsilon a parameter because inline f32 literals in Jet3 kernels
/// create type conflicts during AST construction (f32 returns Field, not Jet3).
/// Parameters get wrapped as Jet3::constant() at the Let binding level, avoiding this.
fn sphere_at(cx: f32, cy: f32, cz: f32, r: f32) -> impl Manifold<Jet3_4, Output = Jet3> + Clone {
    // Smooth epsilon for grazing angles (as parameter to avoid type conflicts)
    const EPSILON_SQ: f32 = 0.0001;

    kernel!(|cx: f32, cy: f32, cz: f32, r: f32, eps: f32| -> Jet3 {
        // Ray-sphere intersection: |t*D - C|² = r²
        // For normalized ray D: t² - 2t(D·C) + |C|² - r² = 0
        // Solution: t = (D·C) - sqrt((D·C)² - (|C|² - r²))

        let d_dot_c = X * cx + Y * cy + Z * cz;
        let c_sq = cx * cx + cy * cy + cz * cz;
        let r_sq = r * r;
        let discriminant = d_dot_c * d_dot_c - (c_sq - r_sq);

        // Add epsilon for safe sqrt
        let safe_discriminant = discriminant + eps;

        d_dot_c - safe_discriminant.sqrt()
    })(cx, cy, cz, r, EPSILON_SQ)
}

/// Screen coordinate remapper using kernel! macro.
///
/// Maps pixel coordinates (0..width, 0..height) to normalized coordinates
/// centered at origin with height = 2.0 (aspect-correct).
fn screen_remap<M>(inner: M, width: f32, height: f32) -> impl Manifold<Output = Discrete> + Clone
where
    M: Manifold<Output = Discrete> + Clone + Send + Sync + 'static,
{
    // Precompute constants
    let half_width = width * 0.5;
    let half_height = height * 0.5;
    let scale = 2.0 / height;

    // The kernel maps screen coords to normalized coords, then samples inner
    kernel!(|half_w: f32, half_h: f32, scale: f32| -> Discrete {
        // Map pixel coords to normalized: center at origin, height = 2.0
        let sx = (X - half_w) * scale;
        let sy = (half_h - Y) * scale;
        inner.at(sx, sy, Z, W)
    })(half_width, half_height, scale)
}

// ============================================================================
// SCENE CONSTRUCTION - Compositional Animation
// ============================================================================

/// Animation parameters for the oscillating sphere.
const BASE_CENTER: (f32, f32, f32) = (0.0, 0.0, 4.0);
const AMPLITUDE: f32 = 2.0; // Oscillate ±2 units in X
const FREQUENCY: f32 = 1.0; // 1 rad/s
const RADIUS: f32 = 1.0;

/// Build scene with sphere at the given animated position.
///
/// The animation offset is precomputed at the application level using
/// sin(t * frequency) * amplitude, then baked into the sphere's center.
fn build_scene_at_time(
    t: f32,
    width: u32,
    height: u32,
) -> impl Manifold<Output = Discrete> + Clone + Sync + Send {
    // Compute the animated X offset
    let x_offset = (t * FREQUENCY).sin() * AMPLITUDE;
    let cx = BASE_CENTER.0 + x_offset;
    let cy = BASE_CENTER.1;
    let cz = BASE_CENTER.2;

    // Background: floor with checkerboard
    let world = ColorSurface {
        geometry: plane(-1.0),
        material: ColorChecker::<ColorCube>::default(),
        background: ColorSky::<ColorCube>::default(),
    };

    // Sphere at the computed animated position (using kernel! macro)
    let sphere = sphere_at(cx, cy, cz, RADIUS);

    let scene = ColorSurface {
        geometry: sphere,
        material: ColorReflect { inner: world },
        background: world,
    };

    screen_remap(
        ColorScreenToDir { inner: scene },
        width as f32,
        height as f32,
    )
}

// ============================================================================
// APPLICATION - Pull-based rendering
// ============================================================================

/// The animated sphere application.
///
/// Implements the pull-based rendering model:
/// - Receives `RequestFrame` events from the engine
/// - Responds with a manifold at the requested timestamp
/// - Handles resize events to update dimensions
struct AnimatedSphereApp {
    /// Animation start time
    start: Instant,
    /// Handle to send frames back to the engine
    engine_handle: pixelflow_runtime::api::private::EngineActorHandle,
    /// Current width (atomic for interior mutability)
    width: AtomicU32,
    /// Current height (atomic for interior mutability)
    height: AtomicU32,
}

impl Application for AnimatedSphereApp {
    fn send(&self, event: EngineEvent) -> Result<(), RuntimeError> {
        match event {
            // Engine is ready for a frame - this is the pull!
            EngineEvent::Data(EngineEventData::RequestFrame { timestamp, .. }) => {
                log::debug!("App received RequestFrame");

                // Compute elapsed time from animation start
                let t = timestamp.duration_since(self.start).as_secs_f32();

                // Get current dimensions
                let width = self.width.load(Ordering::Relaxed);
                let height = self.height.load(Ordering::Relaxed);

                // Build the scene at this moment in time with current dimensions
                let scene = build_scene_at_time(t, width, height);

                let arc: Arc<dyn Manifold<Output = Discrete> + Send + Sync> = Arc::new(scene);

                // Send the frame back to the engine
                log::debug!("App sending RenderSurface");
                self.engine_handle
                    .send(Message::Data(EngineData::FromApp(AppData::RenderSurface(
                        arc,
                    ))))
                    .map_err(|e| RuntimeError::EventSendError(e.to_string()))?;
            }
            // Handle resize events
            EngineEvent::Control(EngineEventControl::Resized {
                width_px, height_px, ..
            }) => {
                log::info!("App: Window resized to {}x{}", width_px, height_px);
                self.width.store(width_px, Ordering::Relaxed);
                self.height.store(height_px, Ordering::Relaxed);
            }
            EngineEvent::Control(EngineEventControl::WindowCreated {
                width_px, height_px, ..
            }) => {
                log::info!("App: Window created {}x{}", width_px, height_px);
                self.width.store(width_px, Ordering::Relaxed);
                self.height.store(height_px, Ordering::Relaxed);
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
        width: AtomicU32::new(WIDTH),
        height: AtomicU32::new(HEIGHT),
    };

    // Register app and create window
    use pixelflow_runtime::WindowDescriptor;
    let window = WindowDescriptor {
        width: WIDTH,
        height: HEIGHT,
        title: "Animated Sphere".into(),
        resizable: true,
    };
    let _engine_handle = unregistered_handle.register(Arc::new(app), window)?;

    println!("Running... (close window to exit)");
    troupe.play().map_err(|e| anyhow::anyhow!("{}", e))?;

    println!("Done!");
    Ok(())
}
