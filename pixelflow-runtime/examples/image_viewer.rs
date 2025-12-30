//! Simple Image Viewer - displays the chrome sphere in a window.
//!
//! This demonstrates the minimal pixelflow-runtime usage:
//! 1. Create EngineTroupe with config
//! 2. Get engine handle
//! 3. Send a manifold to render
//! 4. Run the event loop

use pixelflow_core::combinators::At;
use pixelflow_core::{Discrete, Field, Manifold};
use pixelflow_runtime::{api::public::AppData, EngineConfig, EngineTroupe, WindowConfig};
use std::sync::Arc;

const W: u32 = 1920;
const H: u32 = 1080;

// Import scene3d types
use pixelflow_graphics::patch::BezierPatch;
use pixelflow_graphics::render::Color;
use pixelflow_graphics::scene3d::{
    ColorChecker, ColorReflect, ColorScreenToDir, ColorSky, ColorSurface, HeightFieldGeometry,
    Lift, PlaneGeometry, SceneObject, SphereAt, Union,
};

/// Screen coordinate remapper for Discrete output.
#[derive(Clone, Copy)]
struct ScreenRemap<M> {
    inner: M,
    width: f32,
    height: f32,
}

impl<M: Manifold<Output = Discrete>> Manifold for ScreenRemap<M> {
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

/// Build the chrome sphere scene with bezier patch.
fn build_scene() -> impl Manifold<Output = Discrete> + Send + Sync + Clone {
    // Floor: checkerboard plane at y = -1
    let floor = SceneObject {
        geometry: PlaneGeometry { height: -1.0 },
        material: ColorChecker,
    };

    // Bezier patch: paraboloid bump on the ground
    let patch = BezierPatch::paraboloid(2.0, 0.5);
    let patch_geo = HeightFieldGeometry {
        height_field: patch,
        base_height: -0.8,
        scale: 0.3,
        uv_scale: 0.15, // Maps world coords to [0,1] parameter space
    };
    let patch_obj = SceneObject {
        geometry: patch_geo,
        material: Lift(Color::Rgb(180, 120, 60)), // Warm orange-brown
    };

    // Ground: patch in front of floor, with sky background
    // Union = first hit wins (patch occludes floor where it exists)
    let ground = Union {
        first: patch_obj,
        second: floor,
        background: ColorSky,
    };

    // Chrome sphere reflecting the ground
    let scene = ColorSurface {
        geometry: SphereAt {
            center: (0.0, 0.0, 4.0),
            radius: 1.0,
        },
        material: ColorReflect { inner: ground },
        background: ground,
    };

    ScreenRemap {
        inner: ColorScreenToDir { inner: scene },
        width: W as f32,
        height: H as f32,
    }
}

fn main() -> anyhow::Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("Chrome Sphere + Bezier Patch");
    println!("=============================");
    println!("Resolution: {}x{}", W, H);
    println!();

    // Configure the engine
    let config = EngineConfig {
        window: WindowConfig {
            title: "Chrome Sphere + Bezier Patch".to_string(),
            width: W,
            height: H,
        },
        ..Default::default()
    };

    // Phase 1: Create the troupe
    let troupe = EngineTroupe::with_config(config)?;

    // Phase 2: Get engine handle
    let engine_handle = troupe.engine_handle();

    // Build our scene manifold
    let scene = build_scene();
    let scene_arc: Arc<dyn Manifold<Output = Discrete> + Send + Sync> = Arc::new(scene);

    // Send initial frame
    use actor_scheduler::Message;
    use pixelflow_runtime::api::private::EngineData;

    engine_handle
        .send(Message::Data(EngineData::FromApp(AppData::RenderSurface(
            scene_arc.clone(),
        ))))
        .map_err(|e| anyhow::anyhow!("Failed to send initial frame: {}", e))?;

    println!("Sent initial frame to engine");
    println!("Running event loop... (close window to exit)");

    // Phase 3: Run the event loop (blocks)
    troupe.play().map_err(|e| anyhow::anyhow!("{}", e))?;

    println!("Exited cleanly.");
    Ok(())
}
