//! Animated chrome sphere - windowed animation using the runtime.
//!
//! Demonstrates the **pull-based rendering model**:
//! - The engine sends `RequestFrame` when it is ready for a frame
//! - The app answers with a scene sampled at the requested timestamp
//! - No busy loops, no sleeps — vsync drives the cadence
//!
//! **Time is the W coordinate.** The sphere's centre is `sin(W)·amplitude`, a
//! kernel like any other, so the scene is compiled **once** and each frame is
//! the same compiled program collapsed on a later plane
//! (`PackedFrame::on_slice`). Baking the timestamp in as a constant — which
//! is what this example used to do — would mean a JIT compile per frame, and
//! this scene takes ~200 ms to compile.
//!
//! Resizing does recompile, because the camera's frame size is spent when the
//! rays are built; that is once per resize, not once per frame.

use actor_scheduler::Message;
use pixelflow_core::Kernel;
use pixelflow_graphics::render::packed::PackedProgram;
use pixelflow_graphics::render::scene::{compile_platform_packed, Scene};
use pixelflow_graphics::scene3d::{checker, sky, Plane, Ray, Rgba, Sphere};
use pixelflow_runtime::api::private::EngineData;
use pixelflow_runtime::api::public::{AppData, EngineEvent, EngineEventControl, EngineEventData};
use pixelflow_runtime::{Application, EngineConfig, EngineTroupe, RuntimeError, WindowConfig};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

/// The sphere's rest position, how far it swings, and how fast.
const BASE_CENTER: (f32, f32, f32) = (0.0, 0.0, 4.0);
const AMPLITUDE: f32 = 2.0;
const FREQUENCY: f32 = 1.0;
const RADIUS: f32 = 1.0;

fn k(v: f32) -> Kernel {
    Kernel::constant(v)
}

/// The checker floor under the sky, seen along `ray`.
fn world(ray: &Ray) -> Rgba {
    let floor = Plane::at_height(k(-1.0)).hit(ray);
    floor.select(
        &checker(&floor.point()[0], &floor.point()[2], &floor.footprint()),
        &sky(ray),
    )
}

/// The scene, with the sphere swinging in `W`. Compiled at the frame's shape;
/// every frame after that is a different `W`.
fn compile(width: u32, height: u32) -> PackedProgram {
    let ray = Ray::through_screen(width as f32, height as f32);
    let swing = Kernel::w().mul(&k(FREQUENCY)).sin().mul(&k(AMPLITUDE));
    let center = [
        k(BASE_CENTER.0).add(&swing),
        k(BASE_CENTER.1),
        k(BASE_CENTER.2),
    ];
    let sphere = Sphere::new(center, k(RADIUS)).hit(&ray);
    let mirrored = ray.reflected(sphere.normal());
    let color = sphere.select(&world(&mirrored), &world(&ray));
    compile_platform_packed(&color, [width, height])
}

/// The compiled scene and the frame size it was compiled for. A resize is the
/// only thing that invalidates it.
struct CompiledScene {
    width: u32,
    height: u32,
    program: PackedProgram,
}

impl CompiledScene {
    fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            program: compile(width, height),
        }
    }

    /// The scene at time `t`, recompiling only if the window changed size.
    fn at(&mut self, width: u32, height: u32, t: f32) -> Scene {
        if (width, height) != (self.width, self.height) {
            log::info!("recompiling the scene for {width}x{height}");
            *self = Self::new(width, height);
        }
        Scene::Packed(self.program.bind(&[]).on_slice(0.0, t))
    }
}

/// The animated sphere application: it answers `RequestFrame` with the scene
/// at that timestamp, and remembers the window's size.
struct AnimatedSphereApp {
    start: Instant,
    /// Handle to send frames back to the engine. `Mutex` satisfies `Sync` for
    /// `Arc<dyn Application + Send + Sync>`; only the engine actor thread
    /// calls `send`, so there is no contention.
    engine_handle: Mutex<pixelflow_runtime::api::private::EngineActorHandle>,
    /// The compiled scene, behind the same kind of uncontended lock.
    scene: Mutex<CompiledScene>,
    size: Mutex<(u32, u32)>,
}

impl Application for AnimatedSphereApp {
    fn send(&self, event: EngineEvent) -> Result<(), RuntimeError> {
        match event {
            // The engine is ready for a frame — this is the pull.
            EngineEvent::Data(EngineEventData::RequestFrame { timestamp, .. }) => {
                let t = timestamp.duration_since(self.start).as_secs_f32();
                let (width, height) = *self.size.lock().expect("size lock poisoned");
                let scene = self
                    .scene
                    .lock()
                    .expect("scene lock poisoned")
                    .at(width, height, t);
                self.engine_handle
                    .lock()
                    .expect("engine handle lock poisoned")
                    .send(Message::Data(EngineData::FromApp(AppData::RenderSurface(
                        scene,
                    ))))
                    .map_err(|e| RuntimeError::EventSendError(e.to_string()))?;
            }
            EngineEvent::Control(EngineEventControl::Resized {
                width_px,
                height_px,
                ..
            })
            | EngineEvent::Control(EngineEventControl::WindowCreated {
                width_px,
                height_px,
                ..
            }) => {
                log::info!("App: window is {width_px}x{height_px}");
                *self.size.lock().expect("size lock poisoned") = (width_px, height_px);
            }
            EngineEvent::Control(ctrl) => {
                log::debug!("App received Control event: {ctrl:?}");
            }
            _ => log::debug!("App received other event"),
        }
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("Animated Chrome Sphere (pull-based, time is W)");
    println!("=============================================");
    println!("Resolution: {WIDTH}x{HEIGHT}");
    println!();

    let config = EngineConfig {
        window: WindowConfig {
            title: "Animated Sphere".to_string(),
            width: WIDTH,
            height: HEIGHT,
        },
        ..Default::default()
    };

    let mut troupe = EngineTroupe::with_config(config)?;
    let unregistered_handle = troupe.engine_handle();
    let start = Instant::now();

    // The raw handle must be obtained before registration: the app needs it
    // to answer `RequestFrame`.
    let engine_handle_for_app = troupe.raw_engine_handle();

    let compiled = Instant::now();
    let scene = CompiledScene::new(WIDTH, HEIGHT);
    println!("Compiled the scene once in {:?}", compiled.elapsed());

    let app = AnimatedSphereApp {
        start,
        engine_handle: Mutex::new(engine_handle_for_app),
        scene: Mutex::new(scene),
        size: Mutex::new((WIDTH, HEIGHT)),
    };

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
