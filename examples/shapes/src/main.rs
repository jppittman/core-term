use anyhow::{Context, Result};
use clap::Parser;
use log::info;
use pixelflow_core::traits::{Surface, Volume};
use pixelflow_core::volumes::{FnVolume, Translate};
use pixelflow_core::batch::Batch;
use pixelflow_core::backend::{BatchArithmetic, FloatBatchOps};
use pixelflow_engine::{
    AppData, AppManagement, EngineActorHandle, EngineConfig, EngineEvent, EngineEventControl,
    EngineEventData, WindowConfig,
};
use pixelflow_render::{Pixel, PlatformPixel, Rgba};
use actor_scheduler::Actor;
use std::sync::Arc;

/// 3D raymarching renderer using Volume trait
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Window width in pixels
    #[arg(short, long, default_value = "800")]
    width: u32,

    /// Window height in pixels
    #[arg(long, default_value = "600")]
    height: u32,
}

/// Raymarch renderer - takes SDF and color volumes, produces 2D rendering
struct Raymarch<S, C> {
    sdf: S,    // Volume<f32> - signed distance field
    color: C,  // Volume<u32> - color at 3D point
    max_steps: u32,
    max_dist: f32,
    epsilon: f32,
}

impl<S, C> Raymarch<S, C>
where
    S: Volume<f32, f32>,
    C: Volume<u32, f32>,
{
    fn new(sdf: S, color: C) -> Self {
        Self {
            sdf,
            color,
            max_steps: 64,
            max_dist: 100.0,
            epsilon: 0.001,
        }
    }

    fn steps(mut self, n: u32) -> Self {
        self.max_steps = n;
        self
    }

    fn distance(mut self, d: f32) -> Self {
        self.max_dist = d;
        self
    }

    fn precision(mut self, e: f32) -> Self {
        self.epsilon = e;
        self
    }

    fn calc_normal(&self, px: Batch<f32>, py: Batch<f32>, pz: Batch<f32>) -> (Batch<f32>, Batch<f32>, Batch<f32>) {
        use pixelflow_core::backend::{BatchArithmetic, FloatBatchOps, SimdBatch};
        
        // Finite difference epsilon
        let eps = Batch::<f32>::splat(0.001);

        // x-derivative
        let dx1 = self.sdf.eval(px + eps, py, pz);
        let dx2 = self.sdf.eval(px - eps, py, pz);
        let nx = dx1 - dx2;

        // y-derivative
        let dy1 = self.sdf.eval(px, py + eps, pz);
        let dy2 = self.sdf.eval(px, py - eps, pz);
        let ny = dy1 - dy2;

        // z-derivative
        let dz1 = self.sdf.eval(px, py, pz + eps);
        let dz2 = self.sdf.eval(px, py, pz - eps);
        let nz = dz1 - dz2;

        // Normalize
        let len_sq = (nx * nx) + (ny * ny) + (nz * nz);
        // Avoid div by zero
        let len_inv = Batch::<f32>::splat(1.0) / (len_sq.sqrt() + Batch::<f32>::splat(1.0e-6));

        (nx * len_inv, ny * len_inv, nz * len_inv)
    }
}

/// Wrapper to convert u32 pixel coordinates to normalized f32 coordinates
/// and convert u32 colors to Pixel type
struct NormalizedRaymarch<S, C, P: Pixel> {
    raymarch: Raymarch<S, C>,
    width: f32,
    height: f32,
    _phantom: core::marker::PhantomData<P>,
}

impl<S, C, P> Surface<P, u32> for NormalizedRaymarch<S, C, P>
where
    S: Volume<f32, f32>,
    C: Volume<u32, f32>,
    P: Pixel + From<Rgba>,
{
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<P> {
        use pixelflow_core::backend::{BatchArithmetic, FloatBatchOps}; // Ensure these are imported as they are used

        // Convert u32 pixel coords to f32 and normalize to [-1, 1]
        let xf = x.to_f32();
        let yf = y.to_f32();

        let half_width = Batch::<f32>::splat(self.width / 2.0);
        let half_height = Batch::<f32>::splat(self.height / 2.0);

        // Normalize to [-1, 1]
        let nx = (xf - half_width) / half_width;
        let ny = (half_height - yf) / half_height;

        // Apply aspect ratio correction to make sphere round
        let aspect_ratio_val = self.width / self.height;
        let aspect_ratio = Batch::<f32>::splat(aspect_ratio_val);
        let nx_corrected = nx * aspect_ratio; 

        // Call raymarch with f32 coords, get u32 colors back
        let colors_u32 = self.raymarch.eval(nx_corrected, ny);

        colors_u32.bitcast::<P>()
    }
}

impl<S, C> Surface<u32, f32> for Raymarch<S, C>
where
    S: Volume<f32, f32>,
    C: Volume<u32, f32>,
{
    fn eval(&self, x: Batch<f32>, y: Batch<f32>) -> Batch<u32> {
        use pixelflow_core::backend::{BatchArithmetic, FloatBatchOps, SimdBatch};

        let zero = Batch::<f32>::splat(0.0);
        let one = Batch::<f32>::splat(1.0);
        let epsilon = Batch::<f32>::splat(self.epsilon);
        let max_dist = Batch::<f32>::splat(self.max_dist);

        // --- Helper: March ---
        let march = |ro_x: Batch<f32>, ro_y: Batch<f32>, ro_z: Batch<f32>, 
                     rd_x: Batch<f32>, rd_y: Batch<f32>, rd_z: Batch<f32>| -> Batch<f32> {
            let mut t = zero;
            for _ in 0..self.max_steps {
                let px = ro_x + rd_x * t;
                let py = ro_y + rd_y * t;
                let pz = ro_z + rd_z * t;
                let dist = self.sdf.eval(px, py, pz);
                
                // Optimization: Early bail if all rays hit or miss
                let done = dist.cmp_lt(epsilon) | t.cmp_gt(max_dist);
                if done.all() {
                    break;
                }

                t = t + dist;
            }
            t
        };

        // --- Helper: Lighting ---
        let get_light = |nx: Batch<f32>, ny: Batch<f32>, nz: Batch<f32>| -> Batch<f32> {
             // Light dir: (-0.577, 0.577, -0.577)
            let lx = Batch::<f32>::splat(-0.577);
            let ly = Batch::<f32>::splat(0.577);
            let lz = Batch::<f32>::splat(-0.577);
            
            let dot_nl = (nx * lx) + (ny * ly) + (nz * lz);
            let diffuse = dot_nl.max(zero);
            let ambient = Batch::<f32>::splat(0.2);
            (ambient + diffuse).min(one)
        };

        // 1. Primary Ray Setup
        let ro_x = zero;
        let ro_y = zero;
        let ro_z = Batch::<f32>::splat(-2.0);

                // Ray Direction: normalized vector (x, y, 1.0)

                let one = Batch::<f32>::splat(1.0);

                let len_sq = (x * x) + (y * y) + one;

                let len_inv = one / len_sq.sqrt();

        

                        let rd_x = x * len_inv;

        

                        let rd_y = y * len_inv;

        

                        let rd_z = len_inv;

        

                

        

                        // 2. Primary March

        

                        let t1 = march(ro_x, ro_y, ro_z, rd_x, rd_y, rd_z);

        let p1_x = ro_x + rd_x * t1;
        let p1_y = ro_y + rd_y * t1;
        let p1_z = ro_z + rd_z * t1;

        let d1 = self.sdf.eval(p1_x, p1_y, p1_z);
        let hit1 = d1.cmp_lt(epsilon) & t1.cmp_lt(max_dist);

        // 3. Primary Surface Properties
        let (n1_x, n1_y, n1_z) = self.calc_normal(p1_x, p1_y, p1_z);
        let light1 = get_light(n1_x, n1_y, n1_z);
        
        // Get Object Color (unpack u32 -> r,g,b)
        let col1_u32 = self.color.eval(p1_x, p1_y, p1_z);
        let c_inv = Batch::<f32>::splat(1.0/255.0);
        let col1_r = (col1_u32.bitcast::<u32>() >> 16 & Batch::<u32>::splat(0xFF)).to_f32() * c_inv;
        let col1_g = (col1_u32.bitcast::<u32>() >> 8 & Batch::<u32>::splat(0xFF)).to_f32() * c_inv;
        let col1_b = (col1_u32.bitcast::<u32>() & Batch::<u32>::splat(0xFF)).to_f32() * c_inv;

        // 4. Reflection Setup
        // R = D - 2(D.N)N
        let dot_dn = (rd_x * n1_x) + (rd_y * n1_y) + (rd_z * n1_z);
        let two = Batch::<f32>::splat(2.0);
        let r_x = rd_x - n1_x * dot_dn * two;
        let r_y = rd_y - n1_y * dot_dn * two;
        let r_z = rd_z - n1_z * dot_dn * two;

        // Offset origin to avoid self-intersection
        let ro2_x = p1_x + n1_x * Batch::<f32>::splat(0.01);
        let ro2_y = p1_y + n1_y * Batch::<f32>::splat(0.01);
        let ro2_z = p1_z + n1_z * Batch::<f32>::splat(0.01);

        // 5. Secondary March
        let t2 = march(ro2_x, ro2_y, ro2_z, r_x, r_y, r_z);
        
        let p2_x = ro2_x + r_x * t2;
        let p2_y = ro2_y + r_y * t2;
        let p2_z = ro2_z + r_z * t2;
        
        let d2 = self.sdf.eval(p2_x, p2_y, p2_z);
        let hit2 = d2.cmp_lt(epsilon) & t2.cmp_lt(max_dist);

        // 6. Secondary Color
        let (n2_x, n2_y, n2_z) = self.calc_normal(p2_x, p2_y, p2_z);
        let light2 = get_light(n2_x, n2_y, n2_z);
        
        let col2_u32 = self.color.eval(p2_x, p2_y, p2_z);
        let col2_r = (col2_u32.bitcast::<u32>() >> 16 & Batch::<u32>::splat(0xFF)).to_f32() * c_inv;
        let col2_g = (col2_u32.bitcast::<u32>() >> 8 & Batch::<u32>::splat(0xFF)).to_f32() * c_inv;
        let col2_b = (col2_u32.bitcast::<u32>() & Batch::<u32>::splat(0xFF)).to_f32() * c_inv;

        let hit2_r = col2_r * light2;
        let hit2_g = col2_g * light2;
        let hit2_b = col2_b * light2;

        // Reflection Miss: Return Black (Environment not visible here)
        let ref_r = hit2.bitcast::<f32>().select(hit2_r, zero);
        let ref_g = hit2.bitcast::<f32>().select(hit2_g, zero);
        let ref_b = hit2.bitcast::<f32>().select(hit2_b, zero);

        // 7. Combine Primary and Reflection
        // Final = Light1 * Col1 * (1-ref) + Ref * ref
        let reflectivity = Batch::<f32>::splat(0.4);
        
        let final_r = (col1_r * light1 * (one - reflectivity)) + (ref_r * reflectivity);
        let final_g = (col1_g * light1 * (one - reflectivity)) + (ref_g * reflectivity);
        let final_b = (col1_b * light1 * (one - reflectivity)) + (ref_b * reflectivity);

        // 8. Primary Miss -> Transparent
        let out_r = hit1.bitcast::<f32>().select(final_r, zero);
        let out_g = hit1.bitcast::<f32>().select(final_g, zero);
        let out_b = hit1.bitcast::<f32>().select(final_b, zero);
        let out_a = hit1.bitcast::<f32>().select(one, zero); // 1.0 if hit, 0.0 if miss

        // Pack
        let c255 = Batch::<f32>::splat(255.0);
        let r_u32 = (out_r.min(one) * c255).to_u32();
        let g_u32 = (out_g.min(one) * c255).to_u32();
        let b_u32 = (out_b.min(one) * c255).to_u32();
        let a_u32 = (out_a * c255).to_u32();

        // Shift using mul (safe)
        (a_u32 * Batch::<u32>::splat(16777216)) +
        (r_u32 * Batch::<u32>::splat(65536)) +
        (g_u32 * Batch::<u32>::splat(256)) +
        b_u32
    }
}


struct Sky {
    width: f32,
    height: f32,
}

impl<P: Pixel + From<Rgba>> Surface<P, u32> for Sky {
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<P> {
        use pixelflow_core::backend::{BatchArithmetic, FloatBatchOps, SimdBatch};

        // Convert to normalized coords [0, 1]
        let yf = x.to_f32(); // Optimization: Sky depends only on Y usually, but gradient is vertical.
                             // Wait, args are (x, y).
        let yf = y.to_f32();
        let h = Batch::<f32>::splat(self.height);
        
        // Normalized Y [-1, 1]
        // ny = (y - h/2) / (h/2)
        let ny = (yf - h * Batch::<f32>::splat(0.5)) / (h * Batch::<f32>::splat(0.5));

        // Gradient
        let one = Batch::<f32>::splat(1.0);
        let t = (ny + one) * Batch::<f32>::splat(0.5);

        let white_r = 1.0; let white_g = 1.0; let white_b = 1.0;
        let blue_r = 0.5; let blue_g = 0.7; let blue_b = 1.0;
        
        let r = Batch::<f32>::splat(white_r) * (one - t) + Batch::<f32>::splat(blue_r) * t;
        let g = Batch::<f32>::splat(white_g) * (one - t) + Batch::<f32>::splat(blue_g) * t;
        let b = Batch::<f32>::splat(white_b) * (one - t) + Batch::<f32>::splat(blue_b) * t;
        
        // Pack
        let c255 = Batch::<f32>::splat(255.0);
        let r_u32 = (r * c255).to_u32();
        let g_u32 = (g * c255).to_u32();
        let b_u32 = (b * c255).to_u32();
        let a_u32 = Batch::<u32>::splat(255);

        let col = (a_u32 * Batch::<u32>::splat(16777216)) +
                  (r_u32 * Batch::<u32>::splat(65536)) +
                  (g_u32 * Batch::<u32>::splat(256)) +
                  b_u32;
        
        col.bitcast::<P>()
    }
}

struct Composite<F, B, P> {
    fg: F,
    bg: B,
    _phantom: core::marker::PhantomData<P>,
}

impl<F, B, P> Surface<P, u32> for Composite<F, B, P>
where
    F: Surface<P, u32>,
    B: Surface<P, u32>,
    P: Pixel + From<Rgba>,
{
    fn eval(&self, x: Batch<u32>, y: Batch<u32>) -> Batch<P> {
        use pixelflow_core::backend::{BatchArithmetic, SimdBatch};
        
        let fg_col = self.fg.eval(x, y);
        let fg_u32 = fg_col.bitcast::<u32>();
        
        // Alpha check (alpha is top 8 bits)
        let alpha = fg_u32 >> 24;
        let has_alpha = alpha.cmp_gt(Batch::<u32>::splat(0));
        
        // If all lanes hit, return FG (optimization)
        if has_alpha.all() {
            return fg_col;
        }

        // If no lanes hit, return BG (optimization)
        if !has_alpha.any() {
            return self.bg.eval(x, y);
        }

        let bg_col = self.bg.eval(x, y);
        
        // Select: if alpha > 0, use FG, else BG
        // Note: Proper alpha blending would be better, but simple composition works for this opaque/transparent raymarch setup.
        let final_u32 = has_alpha.select(fg_u32, bg_col.bitcast::<u32>());
        
        final_u32.bitcast::<P>()
    }
}

/// Shapes rendering application
struct ShapesApp<P: Pixel> {
    engine_tx: EngineActorHandle<P>,
    scene: Arc<dyn Surface<P> + Send + Sync>,
}

impl<P: Pixel + Surface<P> + From<Rgba>> ShapesApp<P> {
    fn new(
        engine_tx: EngineActorHandle<P>,
        scene: Arc<dyn Surface<P> + Send + Sync>,
    ) -> Self {
        Self { engine_tx, scene }
    }

    fn render_frame(&mut self, frame_id: u64) {
        let scene_ref = Arc::clone(&self.scene);

        let _ = self.engine_tx.send(actor_scheduler::Message::Data(
            pixelflow_engine::EngineData::FromApp(AppData::RenderSurface {
                frame_id,
                surface: Box::new(scene_ref),
                app_submit_time: std::time::Instant::now(),
            })
        ));
    }
}

impl<P: Pixel + Surface<P> + From<Rgba>> Actor<EngineEvent, (), AppManagement> for ShapesApp<P> {
    fn handle_data(&mut self, event: EngineEvent) {
        match event {
            EngineEvent::Data(data) => match data {
                EngineEventData::RequestFrame { frame_id, .. } => {
                    self.render_frame(frame_id);
                }
            },
            EngineEvent::Control(ctrl) => match ctrl {
                EngineEventControl::Resize(w, h) => {
                    info!("ShapesApp: Window resized to {}x{}", w, h);
                }
                EngineEventControl::ScaleChanged(scale) => {
                    info!("ShapesApp: Scale changed to {}", scale);
                }
                EngineEventControl::CloseRequested => {
                    info!("ShapesApp: Close requested");
                }
            },
            EngineEvent::Management(_mgmt) => {
                // Handle keyboard/mouse events for interaction later
            }
        }
    }

    fn handle_control(&mut self, _: ()) {}
    fn handle_management(&mut self, _mgmt: AppManagement) {}
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Setup logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    info!("Starting shapes renderer...");

    // Engine configuration
    let engine_config = EngineConfig {
        window: WindowConfig {
            title: "Shapes - Reflective Sphere on Checkerboard".to_string(),
            width: args.width,
            height: args.height,
            initial_x: 100.0,
            initial_y: 100.0,
        },
        performance: Default::default(),
    };

    info!("Creating engine platform...");

    use pixelflow_engine::EnginePlatform;

    // 1. Create platform
    let platform = EnginePlatform::new(engine_config)
        .context("Failed to create platform")?;

    // 2. Init platform (spawns engine)
    let (platform, engine_handle) = platform.init()
        .context("Failed to initialize platform")?;

    // 3. Create sphere SDF
    // Sphere equation: distance = sqrt((x-cx)^2 + (y-cy)^2 + (z-cz)^2) - radius
    // Working in f32 normalized space
    let radius = 1.0f32;

    info!("Creating sphere SDF at origin with radius {}", radius);

    // Define sphere at origin (0,0,0)
    let sphere_sdf_origin: FnVolume<_, f32, f32> = FnVolume::new(move |x: Batch<f32>, y: Batch<f32>, z: Batch<f32>| {
        // Distance from origin
        let dist_sq = x * x + y * y + z * z;
        let dist = dist_sq.sqrt();

        // Signed distance: distance to center minus radius
        let r = Batch::<f32>::splat(radius);
        let dist_sphere = dist - r;

        // Plane: y = -1.0 -> dist = y - (-1.0) = y + 1.0
        let dist_plane = y + Batch::<f32>::splat(1.0);

        // Union
        dist_sphere.min(dist_plane)
    });

    // Translate to (0, 0, 3)
    let sphere_sdf = Translate::new(sphere_sdf_origin, 0.0, 0.0, 3.0);

    // 4. Create Color Volume (Checkerboard on floor, White sphere)
    let color_vol: FnVolume<_, u32, f32> = FnVolume::new(move |x: Batch<f32>, y: Batch<f32>, z: Batch<f32>| {
        use pixelflow_core::backend::{BatchArithmetic, FloatBatchOps, SimdBatch};
        
        // Check if floor (y < -0.99 approx)
        let floor_level = Batch::<f32>::splat(-0.99);
        let is_floor = y.cmp_lt(floor_level);

        // Checkerboard: xor(floor(x), floor(z))
        let scale = Batch::<f32>::splat(0.5); 
        let offset = Batch::<f32>::splat(1000.0);
        
        let sx = ((x + offset) * scale).to_u32();
        let sz = ((z + offset) * scale).to_u32();
        
        let one = Batch::<u32>::splat(1);
        let check = ((sx & one) ^ (sz & one)).cmp_eq(one);
        
        let white = Batch::<u32>::splat(0xFFFFFFFF);
        let grey = Batch::<u32>::splat(0xFF404040); // Darker grey for contrast

        let floor_col = check.select(white, grey);
        
        // If floor, return checker, else white sphere
        is_floor.bitcast::<u32>().select(floor_col, white)
    });

    // 5. Create raymarch scene
    let raymarch = Raymarch::new(sphere_sdf, color_vol);
    let normalized = NormalizedRaymarch::<_, _, PlatformPixel> {
        raymarch,
        width: args.width as f32,
        height: args.height as f32,
        _phantom: core::marker::PhantomData,
    };

    // 6. Create Sky
    let sky = Sky {
        width: args.width as f32,
        height: args.height as f32,
    };

    // 7. Composite Scene over Sky
    let composite = Composite {
        fg: normalized,
        bg: sky,
        _phantom: core::marker::PhantomData,
    };

    let scene: Arc<dyn Surface<PlatformPixel> + Send + Sync> = Arc::new(composite);

    // 8. Create app
    let app = ShapesApp::<PlatformPixel>::new(engine_handle.clone(), scene);

    // 9. Spawn app
    let app_handle = actor_scheduler::spawn_with_config(
        app,
        10,
        128,
        None,
    );

    info!("Starting main event loop...");

    // 10. Run platform (blocks until quit)
    platform.run(app_handle, &engine_handle)
        .context("Engine run failed")?;

    info!("Shapes renderer exited successfully.");
    Ok(())
}
