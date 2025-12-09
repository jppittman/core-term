use anyhow::{Context, Result};
use clap::Parser;
use log::info;
use pixelflow_core::traits::Surface;
use pixelflow_engine::{
    AppData, AppManagement, EngineActorHandle, EngineConfig, EngineEvent, EngineEventControl,
    EngineEventData, WindowConfig,
};
use pixelflow_render::{Frame, Pixel, PlatformPixel, Rgba};
use actor_scheduler::Actor;
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;

/// Simple image/video viewer built on pixelflow-engine
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to the image or video file
    path: String,

    /// Window width in pixels
    #[arg(short, long, default_value = "800")]
    width: u32,

    /// Window height in pixels
    #[arg(long, default_value = "600")]
    height: u32,
}

/// Media source (image or video)
enum MediaSource<P: Pixel> {
    Image(Arc<Frame<P>>),
    Video {
        frames: Arc<Vec<Arc<Frame<P>>>>,
        current_frame: usize,
    },
}

/// Image/video viewer application
struct ImageViewerApp<P: Pixel> {
    engine_tx: EngineActorHandle<P>,
    media: MediaSource<P>,
}

impl<P: Pixel + Surface<P> + From<Rgba>> ImageViewerApp<P> {
    fn new_image(
        engine_tx: EngineActorHandle<P>,
        image_frame: Arc<Frame<P>>,
    ) -> Self {
        Self {
            engine_tx,
            media: MediaSource::Image(image_frame),
        }
    }

    fn new_video(
        engine_tx: EngineActorHandle<P>,
        frames: Arc<Vec<Arc<Frame<P>>>>,
    ) -> Self {
        Self {
            engine_tx,
            media: MediaSource::Video {
                frames,
                current_frame: 0,
            },
        }
    }

    fn render_frame(&mut self, frame_id: u64) {
        match &mut self.media {
            MediaSource::Image(frame) => {
                // Clone the Arc (cheap - just increments ref count)
                let frame_ref = Arc::clone(frame);
                let _ = self.engine_tx.send(actor_scheduler::Message::Data(
                    pixelflow_engine::EngineData::FromApp(AppData::RenderSurface {
                        frame_id,
                        surface: Box::new(frame_ref),
                        app_submit_time: std::time::Instant::now(),
                    })
                ));
            }
            MediaSource::Video { frames, current_frame } => {
                let frame_idx = *current_frame % frames.len();
                let frame_ref = Arc::clone(&frames[frame_idx]);
                let _ = self.engine_tx.send(actor_scheduler::Message::Data(
                    pixelflow_engine::EngineData::FromApp(AppData::RenderSurface {
                        frame_id,
                        surface: Box::new(frame_ref),
                        app_submit_time: std::time::Instant::now(),
                    })
                ));
                *current_frame += 1;
            }
        }
    }
}

impl<P: Pixel + Surface<P> + From<Rgba>> Actor<EngineEvent, (), AppManagement> for ImageViewerApp<P> {
    fn handle_data(&mut self, event: EngineEvent) {
        match event {
            EngineEvent::Data(data) => match data {
                EngineEventData::RequestFrame { frame_id, .. } => {
                    self.render_frame(frame_id);
                }
            },
            EngineEvent::Control(ctrl) => match ctrl {
                EngineEventControl::Resize(w, h) => {
                    info!("ImageViewer: Window resized to {}x{}", w, h);
                }
                EngineEventControl::ScaleChanged(scale) => {
                    info!("ImageViewer: Scale changed to {}", scale);
                }
                EngineEventControl::CloseRequested => {
                    info!("ImageViewer: Close requested");
                }
            },
            EngineEvent::Management(_mgmt) => {
                // Handle keyboard/mouse events for pan/zoom later
            }
        }
    }

    fn handle_control(&mut self, _: ()) {}
    fn handle_management(&mut self, _mgmt: AppManagement) {}
}

/// Decode video file into frames using ffmpeg CLI
fn decode_video(path: &str) -> Result<Vec<Frame<Rgba>>> {
    use std::process::Command;
    use std::fs;

    info!("Decoding video: {}", path);

    // Create temporary directory for frames
    let temp_dir = tempfile::tempdir()?;
    let output_pattern = temp_dir.path().join("frame_%04d.png");

    info!("Extracting frames to: {:?}", temp_dir.path());

    // Run ffmpeg to extract frames
    let output = Command::new("ffmpeg")
        .arg("-i")
        .arg(path)
        .arg("-vf")
        .arg("fps=30")  // 30 fps
        .arg(output_pattern.to_str().unwrap())
        .output()
        .context("Failed to run ffmpeg - is it installed?")?;

    if !output.status.success() {
        anyhow::bail!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // Load all extracted frames
    let mut frames = Vec::new();
    let mut frame_num = 1;

    loop {
        let frame_path = temp_dir.path().join(format!("frame_{:04}.png", frame_num));

        if !frame_path.exists() {
            break;
        }

        let img = image::open(&frame_path)
            .with_context(|| format!("Failed to load frame: {:?}", frame_path))?;

        let rgba_img = img.to_rgba8();
        let (width, height) = rgba_img.dimensions();
        let image_data = rgba_img.into_raw();

        let frame = Frame::<Rgba>::from_bytes(image_data, width, height);
        frames.push(frame);

        if frames.len() % 30 == 0 {
            info!("Loaded {} frames", frames.len());
        }

        frame_num += 1;
    }

    info!("Video decoded: {} frames", frames.len());
    Ok(frames)
}

/// Check if file is a video based on extension
fn is_video_file(path: &str) -> bool {
    let path_lower = path.to_lowercase();
    path_lower.ends_with(".mp4")
        || path_lower.ends_with(".mov")
        || path_lower.ends_with(".avi")
        || path_lower.ends_with(".mkv")
        || path_lower.ends_with(".webm")
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Setup logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    info!("Starting image-viewer...");
    info!("Loading media: {}", args.path);

    let is_video = is_video_file(&args.path);

    // Engine configuration
    let engine_config = EngineConfig {
        window: WindowConfig {
            title: format!("{} - {}", if is_video { "Video Viewer" } else { "Image Viewer" }, args.path),
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

    // 3. Load media and create app
    let app = if is_video {
        // Decode video
        let frames_rgba = decode_video(&args.path)?;
        let frames_platform: Vec<Arc<Frame<PlatformPixel>>> = frames_rgba
            .into_iter()
            .map(|f| Arc::new(f.convert::<PlatformPixel>()))
            .collect();

        info!("Video frames converted to platform format");

        let frames = Arc::new(frames_platform);
        ImageViewerApp::<PlatformPixel>::new_video(engine_handle.clone(), frames)
    } else {
        // Load image
        let img = image::open(&args.path)
            .with_context(|| format!("Failed to load image: {}", args.path))?;

        let rgba_img = img.to_rgba8();
        let (width, height) = rgba_img.dimensions();
        let image_data = rgba_img.into_raw();

        info!("Image loaded: {}x{} ({} bytes)", width, height, image_data.len());

        let frame_rgba = Frame::<Rgba>::from_bytes(image_data, width, height);
        let frame_platform = Arc::new(frame_rgba.convert::<PlatformPixel>());

        ImageViewerApp::<PlatformPixel>::new_image(engine_handle.clone(), frame_platform)
    };

    // 4. Spawn app
    let app_handle = actor_scheduler::spawn_with_config(
        app,
        10,
        128,
        None,
    );

    info!("Starting main event loop...");

    // 5. Run platform (blocks until quit)
    platform.run(app_handle, &engine_handle)
        .context("Engine run failed")?;

    info!("image-viewer exited successfully.");
    Ok(())
}
