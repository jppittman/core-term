//! Animated Chrome Sphere - windowed animation using JIT Kernel.

use actor_scheduler::Message;
use pixelflow_core::{Discrete, Kernel, Lattice, Manifold};
use pixelflow_graphics::scene3d::kernel_3d;
use pixelflow_graphics::Grayscale;
use pixelflow_runtime::api::private::EngineData;
use pixelflow_runtime::api::public::{AppData, EngineEvent, EngineEventControl, EngineEventData};
use pixelflow_runtime::{Application, EngineConfig, EngineTroupe, RuntimeError, WindowConfig, WindowDescriptor};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

const BASE_CENTER: (f32, f32, f32) = (0.0, 0.0, 4.0);
const AMPLITUDE: f32 = 2.0;
const FREQUENCY: f32 = 1.0;
const RADIUS: f32 = 1.0;

fn build_scene_kernel(t: f32, width: u32, height: u32) -> Kernel {
    let x_offset = (t * FREQUENCY).sin() * AMPLITUDE;
    let cx = BASE_CENTER.0 + x_offset;
    let cy = BASE_CENTER.1;
    let cz = BASE_CENTER.2;

    let scale = 2.0 / height as f32;
    let sx = Kernel::x()
        .sub(&Kernel::constant(width as f32 * 0.5))
        .mul(&Kernel::constant(scale));
    let sy = Kernel::constant(height as f32 * 0.5)
        .sub(&Kernel::y())
        .mul(&Kernel::constant(scale));

    let (dx, dy, dz) = kernel_3d::screen_to_ray(&sx, &sy, 1.0);
    let sky = kernel_3d::sky(&dy);

    let t_sphere = kernel_3d::sphere_at((cx, cy, cz), RADIUS, &dx, &dy, &dz);
    let px = dx.mul(&t_sphere);
    let py = dy.mul(&t_sphere);
    let pz = dz.mul(&t_sphere);

    let (_nx, ny, _nz) = kernel_3d::surface_normal(&px, &py, &pz);

    let hit_mask = t_sphere.gt(&Kernel::constant(0.0));
    hit_mask.select(&ny, &sky)
}

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

                let scene_kernel = build_scene_kernel(t, width, height);
                let lattice = Lattice::frame(width as usize, height as usize, 0.0);
                let baked = lattice.bake(&scene_kernel);
                let arc: Arc<dyn Manifold<Output = Discrete> + Send + Sync> =
                    Arc::new(Grayscale(baked));

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
    println!("Animated Chrome Sphere Demo (JIT Kernel)");
    println!("--------------------------------------");

    let config = EngineConfig {
        window: WindowConfig {
            title: "PixelFlow JIT Animated Chrome Sphere".to_string(),
            width: WIDTH,
            height: HEIGHT,
        },
        ..Default::default()
    };

    let mut troupe = EngineTroupe::with_config(config)?;
    let unregistered_handle = troupe.engine_handle();
    let start = Instant::now();
    let engine_handle_for_app = troupe.raw_engine_handle();

    let app = AnimatedSphereApp {
        start,
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
