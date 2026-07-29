//! Simple Image Viewer - displays the chrome sphere in a window using JIT Kernel.

use actor_scheduler::Message;
use pixelflow_core::{Discrete, Kernel, Lattice, Manifold};
use pixelflow_graphics::scene3d::kernel_3d;
use pixelflow_graphics::Grayscale;
use pixelflow_runtime::api::private::EngineData;
use pixelflow_runtime::api::public::{AppData, EngineEvent, EngineEventControl, EngineEventData};
use pixelflow_runtime::{Application, EngineConfig, EngineTroupe, WindowConfig, WindowDescriptor};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

const W: u32 = 1920;
const H: u32 = 1080;

fn build_scene() -> Kernel {
    let scale = 2.0 / H as f32;
    let sx = Kernel::x()
        .sub(&Kernel::constant(W as f32 * 0.5))
        .mul(&Kernel::constant(scale));
    let sy = Kernel::constant(H as f32 * 0.5)
        .sub(&Kernel::y())
        .mul(&Kernel::constant(scale));

    let (dx, dy, dz) = kernel_3d::screen_to_ray(&sx, &sy, 1.0);
    let sky = kernel_3d::sky(&dy);

    let (t_sphere, hit_mask) = kernel_3d::sphere_at((0.0, 0.0, 4.0), 1.0, &dx, &dy, &dz);
    let px = dx.mul(&t_sphere);
    let py = dy.mul(&t_sphere);
    let pz = dz.mul(&t_sphere);

    let (_nx, ny, _nz) = kernel_3d::surface_normal(&px, &py, &pz);

    hit_mask.select(&ny, &sky)
}

struct ImageViewerApp {
    engine_handle: std::sync::Mutex<pixelflow_runtime::api::private::EngineActorHandle>,
    width: AtomicU32,
    height: AtomicU32,
}

impl Application for ImageViewerApp {
    fn send(&self, event: EngineEvent) -> Result<(), pixelflow_runtime::RuntimeError> {
        match event {
            EngineEvent::Data(EngineEventData::RequestFrame { .. }) => {
                let width = self.width.load(Ordering::Relaxed);
                let height = self.height.load(Ordering::Relaxed);

                let scene_kernel = build_scene();
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
                    .map_err(|e| pixelflow_runtime::RuntimeError::EventSendError(e.to_string()))?;
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

    println!("Chrome Sphere JIT Image Viewer");
    println!("==============================");
    println!("Resolution: {}x{}", W, H);

    let config = EngineConfig {
        window: WindowConfig {
            title: "Chrome Sphere JIT Image Viewer".to_string(),
            width: W,
            height: H,
        },
        ..Default::default()
    };

    let mut troupe = EngineTroupe::with_config(config)?;
    let unregistered_handle = troupe.engine_handle();
    let engine_handle_for_app = troupe.raw_engine_handle();

    let app = ImageViewerApp {
        engine_handle: std::sync::Mutex::new(engine_handle_for_app),
        width: AtomicU32::new(W),
        height: AtomicU32::new(H),
    };

    let window = WindowDescriptor {
        width: W,
        height: H,
        title: "Image Viewer".into(),
        resizable: false,
    };
    let _engine_handle = unregistered_handle.register(Arc::new(app), window)?;

    println!("Sent initial frame to engine");
    troupe.play().map_err(|e| anyhow::anyhow!("{}", e))?;

    Ok(())
}
