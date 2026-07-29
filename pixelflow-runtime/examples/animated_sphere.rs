//! Animated Chrome Sphere - windowed animation using `kernel!` macro with dynamic struct parameters.
//!
//! Demonstrates zero-allocation, zero-recompile, zero-bake per-frame parameter updating
//! by passing dynamic runtime parameters (`t`, `width`, `height`) as struct fields.

use actor_scheduler::Message;
use pixelflow_compiler::kernel;
use pixelflow_core::{Discrete, Field, Manifold};
use pixelflow_graphics::Grayscale;
use pixelflow_runtime::api::private::EngineData;
use pixelflow_runtime::api::public::{AppData, EngineEvent, EngineEventControl, EngineEventData};
use pixelflow_runtime::{
    Application, EngineConfig, EngineTroupe, RuntimeError, WindowConfig, WindowDescriptor,
};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

kernel!(pub struct AnimatedSphereScene = |t: f32, width: f32, height: f32| Field -> Field {
    let scale = 2.0 / height;
    let sx = (X - width * 0.5) * scale;
    let sy = (height * 0.5 - Y) * scale;
    let sz = 1.0;

    let len = (sx * sx + sy * sy + sz * sz).sqrt();
    let dx = sx / len;
    let dy = sy / len;
    let dz = sz / len;

    let sky = (dy * 0.5 + 0.5).max(0.0).min(1.0) * 0.8 + 0.1;

    let x_offset = (t * 1.0).sin() * 2.0;
    let cx = 0.0 + x_offset;
    let cy = 0.0;
    let cz = 4.0;

    let d_dot_c = dx * cx + dy * cy + dz * cz;
    let c_sq = cx * cx + cy * cy + cz * cz;
    let discriminant = d_dot_c * d_dot_c - (c_sq - 1.0);
    let zero = 0.0;
    let disc_valid = discriminant.ge(zero);
    let safe_disc = disc_valid.select(discriminant, zero);
    let t_sphere = d_dot_c - safe_disc.sqrt();
    let hit_mask = disc_valid & t_sphere.gt(zero);

    let px = dx * t_sphere;
    let py = dy * t_sphere;
    let pz = dz * t_sphere;

    let nx = px - cx;
    let ny = py - cy;
    let nz = pz - cz;

    let ny_norm = ny / (nx * nx + ny * ny + nz * nz).sqrt();

    hit_mask.select(ny_norm, sky)
});

struct AnimatedSphereApp {
    start: Instant,
    engine_handle: std::sync::Mutex<pixelflow_runtime::api::private::EngineActorHandle>,
    width: AtomicU32,
    height: AtomicU32,
}

impl Application for AnimatedSphereApp {
    fn send(&self, event: EngineEvent) -> Result<(), RuntimeError> {
        match event {
            EngineEvent::Data(EngineEventData::RequestFrame { timestamp, .. }) => {
                let t = timestamp.duration_since(self.start).as_secs_f32();
                let width = self.width.load(Ordering::Relaxed);
                let height = self.height.load(Ordering::Relaxed);

                // Update struct fields (t, width, height) - zero allocation, zero JIT recompile, zero bake
                let scene = AnimatedSphereScene::new(t, width as f32, height as f32);
                let arc: Arc<dyn Manifold<Output = Discrete> + Send + Sync> =
                    Arc::new(Grayscale(scene));

                self.engine_handle
                    .lock()
                    .unwrap()
                    .send(Message::Data(EngineData::FromApp(AppData::RenderSurface(
                        arc,
                    ))))
                    .map_err(|e| RuntimeError::EventSendError(e.to_string()))?;
            }
            EngineEvent::Control(EngineEventControl::Resized {
                width_px,
                height_px,
                ..
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

    println!("Animated Chrome Sphere Demo (`kernel!` macro)");
    println!("===========================================");
    println!("Resolution: {}x{}", WIDTH, HEIGHT);

    let config = EngineConfig {
        window: WindowConfig {
            title: "Animated Chrome Sphere".to_string(),
            width: WIDTH,
            height: HEIGHT,
        },
        ..Default::default()
    };

    let mut troupe = EngineTroupe::with_config(config)?;
    let unregistered_handle = troupe.engine_handle();
    let engine_handle_for_app = troupe.raw_engine_handle();

    let app = AnimatedSphereApp {
        start: Instant::now(),
        engine_handle: std::sync::Mutex::new(engine_handle_for_app),
        width: AtomicU32::new(WIDTH),
        height: AtomicU32::new(HEIGHT),
    };

    let window = WindowDescriptor {
        width: WIDTH,
        height: HEIGHT,
        title: "Animated Chrome Sphere".into(),
        resizable: true,
    };
    let _engine_handle = unregistered_handle.register(Arc::new(app), window)?;

    troupe.play().map_err(|e| anyhow::anyhow!("{}", e))?;

    Ok(())
}
