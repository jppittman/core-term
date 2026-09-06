//! Simple image viewer - displays the chrome sphere in a window.
//!
//! The minimal `pixelflow-runtime` usage:
//! 1. Create an `EngineTroupe` with a config
//! 2. Get the engine handle
//! 3. Compile a scene and send it
//! 4. Run the event loop

use pixelflow_core::Kernel;
use pixelflow_graphics::render::scene::{compile_platform_packed, Scene};
use pixelflow_graphics::scene3d::{checker, sky, Plane, Ray, Rgba, Sphere};
use pixelflow_runtime::{api::public::AppData, EngineConfig, EngineTroupe, WindowConfig};
use std::sync::Arc;

const W: u32 = 1920;
const H: u32 = 1080;

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

/// A chrome sphere at (0, 0, 4) reflecting that world, compiled at the
/// window's shape for the platform's pixel format.
fn scene() -> Scene {
    let ray = Ray::through_screen(W as f32, H as f32);
    let sphere = Sphere::new([k(0.0), k(0.0), k(4.0)], k(1.0)).hit(&ray);
    let mirrored = ray.reflected(sphere.normal());
    let channels = sphere
        .select(&world(&mirrored), &world(&ray))
        .into_channels();
    Scene::Packed(compile_platform_packed(&channels, [W, H]).bind(&[]))
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("Chrome Sphere");
    println!("=============");
    println!("Resolution: {W}x{H}");
    println!();

    let config = EngineConfig {
        window: WindowConfig {
            title: "Chrome Sphere".to_string(),
            width: W,
            height: H,
        },
        ..Default::default()
    };

    let mut troupe = EngineTroupe::with_config(config)?;
    let unregistered_handle = troupe.engine_handle();

    /// The window shows one still frame, so the app has nothing to answer.
    struct DummyApp;
    impl pixelflow_runtime::Application for DummyApp {
        fn send(
            &self,
            _event: pixelflow_runtime::EngineEvent,
        ) -> Result<(), pixelflow_runtime::RuntimeError> {
            Ok(())
        }
    }

    use pixelflow_runtime::WindowDescriptor;
    let window = WindowDescriptor {
        width: W,
        height: H,
        title: "Image Viewer".into(),
        resizable: false,
    };
    let engine_handle = unregistered_handle.register(Arc::new(DummyApp), window)?;

    use actor_scheduler::Message;
    use pixelflow_runtime::api::private::EngineData;

    engine_handle
        .send(Message::Data(EngineData::FromApp(AppData::RenderSurface(
            scene(),
        ))))
        .map_err(|e| anyhow::anyhow!("Failed to send initial frame: {}", e))?;

    println!("Sent initial frame to engine");
    println!("Running event loop... (close window to exit)");

    troupe.play().map_err(|e| anyhow::anyhow!("{}", e))?;

    println!("Exited cleanly.");
    Ok(())
}
