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
//! The PixelFlow approach: DON'T translate the loop literally. The GLSL loop
//! is just summing interference at different frequencies. That's algebra, not
//! iteration.
//!
//! **Time is the W coordinate**, as in `animated_sphere`: the three channels
//! differ only in one weight, they are compiled **once** at the window's
//! shape, and each frame is the same program collapsed on a later plane
//! (`PackedFrame::on_slice`). Baking the timestamp in as a constant would
//! mean a JIT compile per frame.

use actor_scheduler::Message;
use pixelflow_core::Kernel;
use pixelflow_graphics::render::packed::PackedProgram;
use pixelflow_graphics::render::scene::{compile_platform_packed, Scene};
use pixelflow_graphics::scene3d::Rgba;
use pixelflow_runtime::api::private::EngineData;
use pixelflow_runtime::api::public::{AppData, EngineEvent, EngineEventControl, EngineEventData};
use pixelflow_runtime::{Application, EngineConfig, EngineTroupe, RuntimeError, WindowConfig};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

// ============================================================================
// THE SHADER - four channel kernels over the screen and time
// ============================================================================

fn k(v: f32) -> Kernel {
    Kernel::constant(v)
}

/// One colour channel. The `y` weight is the only thing that separates the
/// three; everything else is shared, so the e-graph sees it once.
fn channel(width: f32, height: f32, y_weight: f32) -> Kernel {
    let scale = 2.0 / height;
    let x = Kernel::x().sub(&k(width * 0.5)).mul(&k(scale));
    let y = k(height * 0.5).sub(&Kernel::y()).mul(&k(scale));
    let time = Kernel::w();

    let r_sq = x.mul(&x).add(&y.mul(&y));
    let radial = r_sq.sub(&k(0.7)).abs();

    let swirl_scale = k(1.0).sub(&radial).mul(&k(5.0));
    let vx = x.mul(&swirl_scale);
    let vy = y.mul(&swirl_scale);

    let phase = time.mul(&k(0.5));
    let sin_w03 = time.mul(&k(0.3)).sin();
    let sin_w20 = time.mul(&k(2.0)).sin();

    let vxp = vx.add(&phase);
    let swirl = vxp
        .sin()
        .add(&k(1.0))
        .mul(&vxp.sub(&vy.add(&phase.mul(&k(0.7)))).abs())
        .mul(&k(0.2))
        .add(&k(0.001));

    let pulse = k(1.0).add(&sin_w20.mul(&k(0.1)));
    let radial_factor = radial.mul(&k(-4.0)).mul(&pulse).exp();

    let raw = y
        .mul(&k(y_weight))
        .add(&sin_w03.mul(&k(0.2)))
        .exp()
        .mul(&radial_factor)
        .div(&swirl);
    raw.div(&raw.abs().add(&k(1.0))).add(&k(1.0)).mul(&k(0.5))
}

/// The scene compiled at the window's shape. A resize is the only thing that
/// invalidates it.
fn compile(width: u32, height: u32) -> PackedProgram {
    let (w, h) = (width as f32, height as f32);
    let color = Rgba::from([
        channel(w, h, 1.0),
        channel(w, h, -1.0),
        channel(w, h, -2.0),
        k(1.0),
    ]);
    compile_platform_packed(&color, [width, height])
}

/// The compiled scene and the frame size it was compiled for.
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
            log::info!("recompiling the shader for {width}x{height}");
            *self = Self::new(width, height);
        }
        Scene::Packed(self.program.bind(&[]).on_slice(0.0, t))
    }
}

// ============================================================================
// APPLICATION
// ============================================================================

struct PsychedelicApp {
    start: Instant,
    // Mutex satisfies Sync for Arc<dyn Application + Send + Sync>.
    // No contention — only the engine actor thread calls send().
    engine_handle: Mutex<pixelflow_runtime::api::private::EngineActorHandle>,
    /// The compiled shader, behind the same kind of uncontended lock.
    scene: Mutex<CompiledScene>,
    size: Mutex<(u32, u32)>,
}

impl Application for PsychedelicApp {
    fn send(&self, event: EngineEvent) -> Result<(), RuntimeError> {
        match event {
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
            }) => {
                *self.size.lock().expect("size lock poisoned") = (width_px, height_px);
            }
            EngineEvent::Control(EngineEventControl::WindowCreated {
                width_px,
                height_px,
                ..
            }) => {
                *self.size.lock().expect("size lock poisoned") = (width_px, height_px);
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

    let mut troupe = EngineTroupe::with_config(config)?;
    let unregistered_handle = troupe.engine_handle();
    let start = Instant::now();
    let engine_handle_for_app = troupe.raw_engine_handle();

    let app = PsychedelicApp {
        start,
        engine_handle: Mutex::new(engine_handle_for_app),
        scene: Mutex::new(CompiledScene::new(WIDTH, HEIGHT)),
        size: Mutex::new((WIDTH, HEIGHT)),
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
