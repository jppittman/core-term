//! Psychedelic Shader - The PixelFlow Way
//!
//! Original GLSL (shadertoy style):
//! ```glsl
//! vec2 p=(FC.xy*2.-r)/r.y,l,v=p*(1.-(l+=abs(.7-dot(p,p))))/.2;
//! for(float i;i++<8.;o+=(sin(v.xyyx)+1.)*abs(v.x-v.y)*.2)
//!   v+=cos(v.yx*i+vec2(0,i)+t)/i+.7;
//! o=tanh(exp(p.y*vec4(1,-1,-2,0))*exp(-4.*l.x)/o);
//! ```
//!
//! The PixelFlow approach: DON'T translate the loop literally.
//! The GLSL loop is just summing interference at different frequencies.
//! That's algebra, not iteration. Express it as manifold composition.
//!
//! Time flows through W - use time_shift to set the current time.

use actor_scheduler::Message;
use pixelflow_core::{Discrete, Field, Manifold, ManifoldExt, W, X, Y};
use pixelflow_graphics::animation::{screen_remap, time_shift};
use pixelflow_macros::kernel;
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
// THE SHADER - Algebraic, not imperative
// ============================================================================

/// Build one color channel using a kernel.
/// y_mult: 1.0 for red, -1.0 for green, -2.0 for blue
/// Time comes from W coordinate (set via time_shift)
fn build_channel(y_mult: f32) -> impl Manifold<Output = Field> + Clone + Send + Sync {
    kernel!(|y_mult: f32| -> Field {
        // Radial field: distance from the magic ring at |p|² = 0.7
        let r_sq = X * X + Y * Y;
        let radial = (r_sq - 0.7).abs();

        // Swirl: animated interference pattern (W is time)
        let scale = (1.0 - radial) * 5.0;
        let vx = X * scale;
        let vy = Y * scale;

        // Time-based animation via W coordinate
        let phase = W * 0.5;
        let swirl = ((vx + phase).sin() + 1.0) * ((vx + phase) - (vy + phase * 0.7)).abs() * 0.2 + 0.001;

        // Vertical gradient with time modulation
        let y_factor = (Y * y_mult + (W * 0.3).sin() * 0.2).exp();

        // Radial falloff with pulsing
        let pulse = 1.0 + (W * 2.0).sin() * 0.1;
        let radial_factor = (radial * -4.0 * pulse).exp();

        // Combine with swirl
        let raw = y_factor * radial_factor / swirl;

        // Soft tanh approximation: x / (1 + |x|)
        let soft = raw / (raw.abs() + 1.0);
        (soft + 1.0) * 0.5
    })(y_mult)
}

/// Build the psychedelic scene (time-independent, uses W for animation).
fn build_psychedelic_scene() -> impl Manifold<Output = Discrete> + Clone + Sync + Send {
    // Build each channel independently
    let red = build_channel(1.0);
    let green = build_channel(-1.0);
    let blue = build_channel(-2.0);
    let alpha = Field::from(1.0);

    // Compose through ColorCube
    ColorCube::default().at(red, green, blue, alpha)
}

/// Apply screen remapping and time shift.
fn build_scene_at_time(
    t: f32,
    width: u32,
    height: u32,
) -> impl Manifold<Output = Discrete> + Clone + Sync + Send {
    let scene = build_psychedelic_scene();
    let remapped = screen_remap(scene, width as f32, height as f32);
    time_shift(remapped, t)
}

// ============================================================================
// APPLICATION
// ============================================================================

struct PsychedelicApp {
    start: Instant,
    engine_handle: pixelflow_runtime::api::private::EngineActorHandle,
    width: AtomicU32,
    height: AtomicU32,
}

impl Application for PsychedelicApp {
    fn send(&self, event: EngineEvent) -> Result<(), RuntimeError> {
        match event {
            EngineEvent::Data(EngineEventData::RequestFrame { timestamp, .. }) => {
                let t = timestamp.duration_since(self.start).as_secs_f32();
                let width = self.width.load(Ordering::Relaxed);
                let height = self.height.load(Ordering::Relaxed);

                let scene = build_scene_at_time(t, width, height);

                let arc: Arc<dyn Manifold<Output = Discrete> + Send + Sync> = Arc::new(scene);

                self.engine_handle
                    .send(Message::Data(EngineData::FromApp(AppData::RenderSurface(
                        arc,
                    ))))
                    .map_err(|e| RuntimeError::EventSendError(e.to_string()))?;
            }
            EngineEvent::Control(EngineEventControl::Resized {
                width_px, height_px, ..
            }) => {
                self.width.store(width_px, Ordering::Relaxed);
                self.height.store(height_px, Ordering::Relaxed);
            }
            EngineEvent::Control(EngineEventControl::WindowCreated {
                width_px, height_px, ..
            }) => {
                self.width.store(width_px, Ordering::Relaxed);
                self.height.store(height_px, Ordering::Relaxed);
            }
            _ => {}
        }
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("Psychedelic Shader (PixelFlow Native)");
    println!("=====================================");
    println!("Resolution: {}x{}", WIDTH, HEIGHT);
    println!();

    let config = EngineConfig {
        window: WindowConfig {
            title: "Psychedelic Shader".to_string(),
            width: WIDTH,
            height: HEIGHT,
        },
        ..Default::default()
    };

    let troupe = EngineTroupe::with_config(config)?;
    let unregistered_handle = troupe.engine_handle();
    let start = Instant::now();
    let engine_handle_for_app = troupe.raw_engine_handle();

    let app = PsychedelicApp {
        start,
        engine_handle: engine_handle_for_app,
        width: AtomicU32::new(WIDTH),
        height: AtomicU32::new(HEIGHT),
    };

    use pixelflow_runtime::WindowDescriptor;
    let window = WindowDescriptor {
        width: WIDTH,
        height: HEIGHT,
        title: "Psychedelic Shader".into(),
        resizable: true,
    };
    let _engine_handle = unregistered_handle.register(Arc::new(app), window)?;

    println!("Running... (close window to exit)");
    troupe.play().map_err(|e| anyhow::anyhow!("{}", e))?;

    println!("Done!");
    Ok(())
}
